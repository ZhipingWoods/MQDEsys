"""Block-wise FP8 checkpoints (``quant_method: fp8``, ``weight_block_size: [bh, bw]``) read as plain floating-point weights.

Such a checkpoint stores each linear's weight as ``float8_e4m3fn`` plus ``<name>_scale_inv`` (one scale per bh x bw
block). Despite the name, the scale is **multiplied**: ``W = fp8 * scale_inv`` (checked on a real checkpoint, where the
product has the magnitude of a normal weight, mean |w| about 0.008, while dividing gives about 6e5).

Loading these through transformers needs its FP8 kernels (a GPU and a recent ``kernels`` package). This module does the
dequantization in plain PyTorch, so such a model can be used as the baseline of a quantization job on any device.

    sd = load_fp8_state_dict("ckpt-dir")          # {name: float tensor}, scale tensors consumed
    dequant_block(w_fp8, scale_inv, (128, 128))   # one weight
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

SCALE_SUFFIX = "_scale_inv"
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def dequant_block(w: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int] = (128, 128),
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``w`` [out, in] (FP8) times ``scale_inv`` [ceil(out/bh), ceil(in/bw)]; edge blocks may be partial."""
    if w.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(f"expected 2-D weight and scale, got {tuple(w.shape)} and {tuple(scale_inv.shape)}")
    bh, bw = block
    out, inn = w.shape
    want = ((out + bh - 1) // bh, (inn + bw - 1) // bw)
    if tuple(scale_inv.shape) != want:
        raise ValueError(
            f"scale shape {tuple(scale_inv.shape)} does not match weight {tuple(w.shape)} with {bh}x{bw} blocks (expected {want})"
        )
    full = scale_inv.to(torch.float32).repeat_interleave(bh, 0).repeat_interleave(bw, 1)[:out, :inn]
    return (w.to(torch.float32) * full).to(dtype)


def block_size(ckpt_dir: str | Path) -> tuple[int, int] | None:
    """weight_block_size from config.json (top level or text_config), or None if the checkpoint is not block-wise FP8."""
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
    for c in (cfg, cfg.get("text_config") or {}):
        q = c.get("quantization_config") or {}
        if q.get("quant_method") == "fp8" and q.get("weight_block_size"):
            bh, bw = q["weight_block_size"]
            return int(bh), int(bw)
    return None


def load_fp8_state_dict(ckpt_dir: str | Path, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Every tensor of the checkpoint; FP8 weights dequantized, their scale tensors removed. Needs a block-wise FP8 config."""
    from safetensors import safe_open

    d = Path(ckpt_dir)
    blk = block_size(d)
    if blk is None:
        raise ValueError(f"{d}: config.json has no block-wise fp8 quantization_config")
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{d}: no .safetensors files")
    raw: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():  # noqa: SIM118  (safe_open is not a Mapping)
                raw[k] = sf.get_tensor(k)
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.endswith(SCALE_SUFFIX) and k[: -len(SCALE_SUFFIX)] in raw:
            continue  # consumed with its weight
        if v.dtype in FP8_DTYPES:
            s = raw.get(k + SCALE_SUFFIX)
            if s is None:
                raise ValueError(f"{k}: FP8 weight without {k + SCALE_SUFFIX}")
            out[k] = dequant_block(v, s, blk, dtype)
        else:
            out[k] = v.to(dtype) if v.is_floating_point() else v
    return out
