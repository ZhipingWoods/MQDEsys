"""Fake/real consistency (T-0702, stage S10).

Reads an exported GPTQ/AWQ checkpoint back with the reference unpacker (``_core.unpack_int``), rebuilds each
layer's integer codes, zeros and fp16 scales, and compares them with the fake-quantized model:

* codes / zeros must be identical (integer path: any difference is a packing bug);
* the dequantized real weight differs from the fake weight only by fp16 scale storage, reported as max_abs / rel.

``use_real_weights`` swaps the real dequantized weights into the fake model so the same evaluation (PPL,
logits probes) can be run on what the backend will actually compute with.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from mqde import core
from mqde.export.base import fake_linears
from mqde.export.int_checkpoint import layer_codes


def _layer_spec(cfg: dict[str, Any], name: str) -> tuple[int, int]:
    bits, gs = int(cfg["bits"]), int(cfg["group_size"])
    for pat, ov in (cfg.get("dynamic") or {}).items():
        if pat.startswith("+:") and re.search(pat[2:], name):
            bits, gs = int(ov.get("bits", bits)), int(ov.get("group_size", gs))
    return bits, gs


def load_real(export_dir: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    from safetensors.numpy import load_file

    d = Path(export_dir)
    cfg = json.loads((d / "config.json").read_text())["quantization_config"]
    sd = load_file(str(d / "model.safetensors"))
    layers: dict[str, dict[str, np.ndarray]] = {}
    for k in sd:
        if k.endswith(".qweight"):
            n = k[: -len(".qweight")]
            layers[n] = {f: sd[f"{n}.{f}"] for f in ("qweight", "qzeros", "scales") if f"{n}.{f}" in sd}
            if f"{n}.g_idx" in sd:
                layers[n]["g_idx"] = sd[f"{n}.g_idx"]
    return cfg, layers


def dequant_real(cfg: dict[str, Any], name: str, t: dict[str, np.ndarray], out: int, inn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(real weight float32 [out, in], codes, zeros) exactly as a kernel would dequantize them."""
    layout = cfg["quant_method"]
    bits, gs = _layer_spec(cfg, name)
    gs = inn if gs == -1 else gs
    codes, zeros = core.unpack_int(layout, t["qweight"], t["qzeros"], out, inn, gs, bits, v1=layout == "gptq")
    g = t["g_idx"].astype(np.int64) if "g_idx" in t else np.arange(inn) // gs
    scales = t["scales"].astype(np.float32).T  # [out, groups]
    w = scales[:, g] * (codes.astype(np.float32) - zeros[:, g].astype(np.float32))
    return w, codes, zeros


@torch.inference_mode()
def real_check(model: nn.Module, export_dir: str | Path) -> dict[str, Any]:
    cfg, real = load_real(export_dir)
    fq = fake_linears(model)
    rows = []
    for n, m in fq.items():
        if n not in real:
            rows.append({"name": n, "missing": True})
            continue
        w_real, codes, zeros = dequant_real(cfg, n, real[n], m.out_features, m.in_features)
        c_f, z_f, _, _ = layer_codes(m)
        w_fake = m.weight.detach().to(torch.float32).cpu().numpy()
        diff = np.abs(w_real - w_fake)
        rows.append({
            "name": n,
            "code_mismatch": int((codes != c_f).sum()),
            "zero_mismatch": int((zeros != z_f).sum()),
            "w_max_abs": float(diff.max()),
            "w_rel_l2": float(np.linalg.norm(w_real - w_fake) / max(np.linalg.norm(w_fake), 1e-30)),
        })
    ok = [r for r in rows if not r.get("missing")]
    return {
        "format": cfg["quant_method"],
        "layers": len(rows),
        "missing": [r["name"] for r in rows if r.get("missing")],
        "code_mismatch": sum(r["code_mismatch"] for r in ok),
        "zero_mismatch": sum(r["zero_mismatch"] for r in ok),
        "codes_identical": all(r["code_mismatch"] == 0 and r["zero_mismatch"] == 0 for r in ok) and len(ok) == len(rows),
        "w_max_abs": max((r["w_max_abs"] for r in ok), default=0.0),
        "w_rel_l2_mean": float(np.mean([r["w_rel_l2"] for r in ok])) if ok else 0.0,
        "per_layer": rows,
    }


@contextmanager
def use_real_weights(model: nn.Module, export_dir: str | Path) -> Iterator[None]:
    """Temporarily replace every fake weight with the checkpoint's dequantized weight (fp16 scales)."""
    cfg, real = load_real(export_dir)
    saved = {}
    try:
        for n, m in fake_linears(model).items():
            w, _, _ = dequant_real(cfg, n, real[n], m.out_features, m.in_features)
            saved[n] = m.weight
            m.weight = torch.from_numpy(w).to(m.weight.device, m.weight.dtype)
        yield
    finally:
        fq = fake_linears(model)
        for n, w in saved.items():
            fq[n].weight = w
