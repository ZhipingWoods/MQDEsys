"""compressed-tensors checkpoints (T-0701), the format vLLM loads natively via ``quant_method: compressed-tensors``.

Two layouts, chosen from the recipe:

* ``pack-quantized`` for integer weights (W4A16 / W8A16): ``weight_packed`` int32 [out, in*b/32] with values
  packed little-endian along the input dim, ``weight_scale`` [out, groups], ``weight_shape`` int64 [2], and for
  asymmetric specs ``weight_zero_point`` int32 packed along the output dim [out*b/32, groups].
  compressed-tensors stores a signed value v (and zero z) as v + 2^(b-1), which equals our unsigned code
  (``layer_codes``), so codes and zeros are packed as-is.
* ``float-quantized`` for FP8 E4M3 weights (W8A8-FP8): ``weight`` float8_e4m3fn, ``weight_scale`` float32
  [out, 1] (channel) or [1] (tensor), activations dynamic per-token / per-tensor, or static per-tensor with
  ``input_scale``.

Per-layer overrides become extra ``config_groups`` targeting ``re:^name$``; FP-kept layers go into ``ignore``.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from mqde.export.base import Exporter, ExportResult, fake_linears, register
from mqde.export.int_checkpoint import _COPY, layer_codes
from mqde.quant.modules import FakeQuantLinear
from mqde.quant.numerics import FP8_MAX


def pack_int32(v: np.ndarray, bits: int, dim: int) -> np.ndarray:
    """Pack unsigned values (< 2^bits) along ``dim`` into int32, element i of each word at bit i*bits."""
    pf = 32 // bits
    a = np.moveaxis(v.astype(np.uint32), dim, -1)
    if a.shape[-1] % pf:
        raise ValueError(f"dim {dim} size {a.shape[-1]} is not a multiple of {pf}")
    a = a.reshape(*a.shape[:-1], a.shape[-1] // pf, pf)
    shifts = (np.arange(pf, dtype=np.uint32) * bits).astype(np.uint32)
    words = np.bitwise_or.reduce(a << shifts, axis=-1).astype(np.uint32)
    return np.ascontiguousarray(np.moveaxis(words, -1, dim)).view(np.int32)


def unpack_int32(w: np.ndarray, bits: int, dim: int) -> np.ndarray:
    pf = 32 // bits
    a = np.moveaxis(w.view(np.uint32), dim, -1)
    shifts = (np.arange(pf, dtype=np.uint32) * bits).astype(np.uint32)
    vals = (a[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    vals = vals.reshape(*a.shape[:-1], a.shape[-1] * pf)
    return np.ascontiguousarray(np.moveaxis(vals, -1, dim)).astype(np.uint8)


def _strategy(gran: str) -> str:
    return {"per_group": "group", "per_channel": "channel", "per_tensor": "tensor"}[gran]


@register("compressed-tensors")
class CompressedTensorsExporter(Exporter):
    """compressed-tensors (pack-quantized int W4A16/W8A16, float-quantized FP8 W8A8)."""

    backends = ("vllm", "sglang")

    def __init__(self, dtype: torch.dtype = torch.float16) -> None:
        self.dtype = dtype

    def check_layer(self, name: str, m: FakeQuantLinear) -> str | None:
        s, a = m.w_spec, m.a_spec
        if m.in_scale is not None:
            return "activation smoothing scale (AWQ/SmoothQuant transform) must be folded into the previous op first"
        if m.w_scale is None:
            return "layer has no weight qparams (quantized before qparams were tracked)"
        if s.format == "int":
            if s.bits not in (4, 8):
                return f"{s.bits}-bit integer weights are not supported by compressed-tensors pack-quantized (4, 8)"
            if a.enabled:
                return "integer weights with quantized activations (W4A8/W8A8-int) are not exported yet"
            if m.w_zero is None:
                return "layer has no weight zero points"
            pf = 32 // s.bits
            if m.in_features % pf:
                return f"in_features {m.in_features} does not pack into 32-bit words at {s.bits} bit"
            if not s.symmetric and m.out_features % pf:
                return f"out_features {m.out_features} does not pack zero points into 32-bit words at {s.bits} bit"
            return None
        if s.format == "fp8_e4m3":
            if s.granularity == "per_group":
                return "per-group FP8 weight scales are not exported (use per-channel or per-tensor)"
            if a.enabled and a.format != "fp8_e4m3":
                return f"FP8 weights with {a.format} activations are not a compressed-tensors scheme"
            if a.enabled and m.act_scale is not None and a.granularity != "per_tensor":
                return "static activation scales are only exported per-tensor"
            return None
        return f"weight format {s.format} is not supported by compressed-tensors export"

    # ---------------------------------------------------------------- config
    @staticmethod
    def _scheme(m: FakeQuantLinear) -> dict[str, Any]:
        s, a = m.w_spec, m.a_spec
        if s.format == "int":
            w = {"num_bits": s.bits, "type": "int", "symmetric": s.symmetric, "strategy": _strategy(s.granularity),
                 "group_size": s.group_size if s.granularity == "per_group" else None, "dynamic": False,
                 "actorder": None, "observer": "minmax", "observer_kwargs": {}, "block_structure": None}
            return {"weights": w, "input_activations": None, "output_activations": None}
        w = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": _strategy(s.granularity),
             "group_size": None, "dynamic": False, "actorder": None, "observer": "minmax", "observer_kwargs": {},
             "block_structure": None}
        ia = None
        if a.enabled:
            static = m.act_scale is not None
            ia = {"num_bits": 8, "type": "float", "symmetric": True, "dynamic": not static,
                  "strategy": "tensor" if static or a.granularity == "per_tensor" else "token", "group_size": None,
                  "actorder": None, "observer": None if not static else "minmax", "observer_kwargs": {},
                  "block_structure": None}
        return {"weights": w, "input_activations": ia, "output_activations": None}

    def _config(self, fq: dict[str, FakeQuantLinear], ignore: list[str]) -> dict[str, Any]:
        schemes = {n: json.dumps(self._scheme(m), sort_keys=True) for n, m in fq.items()}
        base = Counter(schemes.values()).most_common(1)[0][0]
        groups: dict[str, dict[str, Any]] = {"group_0": {"targets": ["Linear"], **json.loads(base)}}
        others: dict[str, list[str]] = {}
        for n, sch in schemes.items():
            if sch != base:
                others.setdefault(sch, []).append(f"re:^{re.escape(n)}$")
        for i, (sch, targets) in enumerate(others.items(), start=1):
            groups[f"group_{i}"] = {"targets": sorted(targets), **json.loads(sch)}
        fmts = {"pack-quantized" if m.w_spec.format == "int" else "float-quantized" for m in fq.values()}
        return {"quant_method": "compressed-tensors", "format": fmts.pop() if len(fmts) == 1 else "mixed-precision",
                "config_groups": groups, "ignore": ignore, "quantization_status": "compressed",
                "kv_cache_scheme": None, "global_compression_ratio": None, "sparsity_config": {}}

    # ---------------------------------------------------------------- tensors
    def _layer_tensors(self, n: str, m: FakeQuantLinear) -> dict[str, torch.Tensor]:
        s = m.w_spec
        out: dict[str, torch.Tensor] = {}
        if s.format == "int":
            codes, zeros, scales, _ = layer_codes(m)
            out[f"{n}.weight_packed"] = torch.from_numpy(pack_int32(codes, s.bits, dim=1))
            out[f"{n}.weight_scale"] = torch.from_numpy(scales).to(self.dtype)
            out[f"{n}.weight_shape"] = torch.tensor([m.out_features, m.in_features], dtype=torch.int64)
            if not s.symmetric:
                out[f"{n}.weight_zero_point"] = torch.from_numpy(pack_int32(zeros, s.bits, dim=0))
        else:
            assert m.w_scale is not None
            rows = 1 if s.granularity == "per_tensor" else m.out_features
            scale = m.w_scale.detach().to(torch.float32).reshape(rows, 1)
            w = m.weight.detach().to(torch.float32)
            fm = FP8_MAX["fp8_e4m3"]
            out[f"{n}.weight"] = torch.clamp(w / scale, -fm, fm).to(torch.float8_e4m3fn).cpu()
            out[f"{n}.weight_scale"] = (scale.reshape(1) if rows == 1 else scale).cpu().contiguous()
            if m.a_spec.enabled and m.act_scale is not None:
                out[f"{n}.input_scale"] = m.act_scale.detach().to(torch.float32).reshape(1).cpu()
        if m.bias is not None:
            out[f"{n}.bias"] = m.bias.detach().to(self.dtype).cpu()
        return out

    # ---------------------------------------------------------------- export
    @torch.inference_mode()
    def export(self, model: nn.Module, out_dir: str | Path, source_dir: str | Path | None = None) -> ExportResult:
        from safetensors.torch import save_file

        problems = self.check(model)
        if problems:
            head = "; ".join(f"{n}: {r}" for n, r in problems[:5])
            raise ValueError(f"{self.name}: {len(problems)} layer(s) cannot be exported ({head})")
        fq = fake_linears(model)
        if not fq:
            raise ValueError(f"{self.name}: model has no quantized layers")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sd: dict[str, torch.Tensor] = {}
        for n, m in fq.items():
            sd.update(self._layer_tensors(n, m))
        prefixes = tuple(f"{n}." for n in fq)
        tied = bool(getattr(cast(Any, model).config, "tie_word_embeddings", False))
        for k, v in model.state_dict().items():
            if k.startswith(prefixes) or (tied and k == "lm_head.weight"):
                continue
            sd[k] = (v.to(self.dtype) if v.is_floating_point() else v).detach().cpu().contiguous()
        save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt", "producer": "mqde"})
        kept = sorted(n for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and not isinstance(mod, FakeQuantLinear))
        kept_q = [n for n in kept if n != "lm_head"]
        qcfg = self._config(fq, kept)
        cfg = dict(cast(Any, model.config).to_dict())
        cfg["torch_dtype"] = str(self.dtype).replace("torch.", "")
        cfg["quantization_config"] = qcfg
        cfg.pop("_attn_implementation_autoset", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        if source_dir:
            for f in _COPY:
                if (Path(source_dir) / f).is_file():
                    shutil.copy2(Path(source_dir) / f, out / f)
        files = sorted(p.name for p in out.iterdir())
        return ExportResult(format=self.name, out_dir=str(out), files=files, quantized=sorted(fq), kept_fp=kept_q,
                            quantization_config=qcfg, bytes=sum(p.stat().st_size for p in out.iterdir()))


# ---------------------------------------------------------------- read-back (S10 fake/real check)
def load_ct(export_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Per-layer arrays of a compressed-tensors checkpoint.

    pack-quantized layers: weight_packed, weight_scale, weight_shape and optionally weight_zero_point.
    float-quantized layers: ``fp8`` (float32 values of the stored FP8), ``fp8_bits`` (raw bytes), weight_scale."""
    from safetensors.torch import load_file

    sd = load_file(str(Path(export_dir) / "model.safetensors"))
    layers: dict[str, dict[str, np.ndarray]] = {}
    for k, v in sd.items():
        if k.endswith(".weight_packed"):
            n = k[: -len(".weight_packed")]
            layers[n] = {f: sd[f"{n}.{f}"].numpy() for f in ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point")
                         if f"{n}.{f}" in sd}
        elif k.endswith(".weight") and v.dtype == torch.float8_e4m3fn:
            n = k[: -len(".weight")]
            layers[n] = {"fp8": v.float().numpy(), "fp8_bits": v.view(torch.uint8).numpy(),
                         "weight_scale": sd[f"{n}.weight_scale"].float().numpy()}
    return layers


def layer_weights_cfg(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """The ``weights`` scheme that applies to ``name``: an explicit ``re:`` target wins over the base group."""
    groups = cfg["config_groups"]
    for gname, g in groups.items():
        if gname != "group_0" and any(t.startswith("re:") and re.search(t[3:], name) for t in g["targets"]):
            return cast(dict[str, Any], g["weights"])
    return cast(dict[str, Any], groups["group_0"]["weights"])


def dequant_ct(cfg: dict[str, Any], name: str, t: dict[str, np.ndarray], out: int, inn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(weight float32 [out, in], codes, zeros) as a kernel would dequantize the checkpoint.

    For FP8 layers ``codes`` are the raw FP8 bytes and ``zeros`` an empty [out, 1] placeholder."""
    if "fp8" in t:
        sc = t["weight_scale"].reshape(-1, 1)
        return t["fp8"] * sc, t["fp8_bits"], np.zeros((out, 1), np.uint8)
    w = layer_weights_cfg(cfg, name)
    bits = int(w["num_bits"])
    gs = int(w["group_size"]) if w.get("group_size") else inn
    codes = unpack_int32(t["weight_packed"], bits, 1)[:, :inn]
    groups = inn // gs
    if "weight_zero_point" in t:
        zeros = unpack_int32(t["weight_zero_point"], bits, 0)[:out, :groups]
    else:
        zeros = np.full((out, groups), 1 << (bits - 1), np.uint8)  # symmetric: codes carry the 2^(b-1) offset
    scales = t["weight_scale"].astype(np.float32)  # [out, groups]
    g = np.arange(inn) // gs
    return scales[:, g] * (codes.astype(np.float32) - zeros[:, g].astype(np.float32)), codes, zeros


def fake_ct_codes(m: FakeQuantLinear) -> tuple[np.ndarray, np.ndarray]:
    """What the exporter should have stored for ``m``: (codes, zeros) in the same convention as ``dequant_ct``."""
    if m.w_spec.format == "int":
        c, z, _, _ = layer_codes(m)
        if m.w_spec.symmetric:  # no zero points are stored; the reader assumes the 2^(b-1) offset
            z = np.full_like(z, 1 << (m.w_spec.bits - 1))
        return c, z
    assert m.w_scale is not None
    rows = 1 if m.w_spec.granularity == "per_tensor" else m.out_features
    sc = m.w_scale.detach().to(torch.float32).reshape(rows, 1)
    q = torch.clamp(m.weight.detach().to(torch.float32) / sc, -FP8_MAX["fp8_e4m3"], FP8_MAX["fp8_e4m3"])
    return q.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy(), np.zeros((m.out_features, 1), np.uint8)
