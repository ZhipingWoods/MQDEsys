"""Torch (CPU/GPU) mirror of the C++ quantization core (cpp/src/quant/ops.cpp).

Contract: for min-max calibration the outputs are bit-identical to ``mqde._core.fake_quant``
(checked by python/tests/test_numerics_golden.py). The order of float32 operations below is part
of that contract; do not "simplify" it. MSE calibration may pick a different candidate on exact
error ties (different summation order), so it is compared with a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

FP8_MAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0}
FP8_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}


@dataclass(frozen=True)
class QuantSpec:
    format: str = "none"  # int | fp8_e4m3 | fp8_e5m2 | none
    bits: int = 16
    symmetric: bool = True
    granularity: str = "per_channel"  # per_tensor | per_channel (per_token) | per_group
    group_size: int = 0
    calib: str = "minmax"  # minmax | mse
    maxshrink: float = 0.8
    grid: int = 20

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> QuantSpec:
        if not d or d.get("format", "int") in ("none", "fp16", "bf16"):
            return cls()
        fmt = d.get("format", "int")
        fmt = "fp8_e4m3" if fmt == "fp8" else fmt
        gran = d.get("granularity", "per_channel")
        gran = "per_channel" if gran == "per_token" else gran
        return cls(
            format=fmt,
            bits=int(d.get("bits", 8)),
            symmetric=bool(d.get("symmetric", True)),
            granularity=gran,
            group_size=int(d.get("group_size", 0)),
            calib=d.get("calib", "minmax"),
            maxshrink=float(d.get("maxshrink", 0.8)),
            grid=int(d.get("grid", 20)),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"format": self.format}
        if self.enabled:
            d.update(bits=self.bits, symmetric=self.symmetric, granularity=self.granularity, calib=self.calib)
            if self.granularity == "per_group":
                d["group_size"] = self.group_size
            d.update(maxshrink=self.maxshrink, grid=self.grid)
        return d

    @property
    def enabled(self) -> bool:
        return self.format != "none"

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1

    def with_granularity(self, gran: str) -> QuantSpec:
        return QuantSpec(**{**self.__dict__, "granularity": gran})


def params_from_range(mn: torch.Tensor, mx: torch.Tensor, s: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Range -> (scale, zero), float32, same operation order as params_from_range() in ops.cpp."""
    if s.format == "int":
        qmin = torch.tensor(float(s.qmin), dtype=torch.float32, device=mn.device)
        qmax = torch.tensor(float(s.qmax), dtype=torch.float32, device=mn.device)
        if s.symmetric:
            amax = torch.maximum(mn.abs(), mx.abs())
            scale = amax / qmax
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            return scale, torch.zeros_like(scale)
        mn = torch.clamp(mn, max=0.0)
        mx = torch.clamp(mx, min=0.0)
        scale = (mx - mn) / (qmax - qmin)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.round(qmin - mn / scale).clamp(qmin, qmax)
        return scale, zero
    amax = torch.maximum(mn.abs(), mx.abs())
    scale = amax / torch.tensor(FP8_MAX[s.format], dtype=torch.float32, device=mn.device)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale, torch.zeros_like(scale)


def qdq(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, s: QuantSpec) -> torch.Tensor:
    """Quantize-dequantize with broadcastable scale/zero (float32)."""
    if s.format == "int":
        q = torch.clamp(torch.round(x / scale) + zero, float(s.qmin), float(s.qmax))
        return (q - zero) * scale
    fm = FP8_MAX[s.format]
    q = torch.clamp(x / scale, -fm, fm).to(FP8_DTYPE[s.format]).to(torch.float32)
    return q * scale


def _groups(x: torch.Tensor, s: QuantSpec) -> torch.Tensor:
    rows, cols = x.shape
    if s.granularity == "per_tensor":
        return x.reshape(1, -1)
    if s.granularity == "per_group":
        if cols % s.group_size != 0:
            raise ValueError(f"in_features ({cols}) is not divisible by group_size ({s.group_size})")
        return x.reshape(rows * (cols // s.group_size), s.group_size)
    return x


def compute_qparams(x: torch.Tensor, s: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group scale/zero with shape [n_groups, 1] for a 2-D float32 tensor."""
    g = _groups(x, s)
    mn = g.amin(dim=1, keepdim=True)
    mx = g.amax(dim=1, keepdim=True)
    scale, zero = params_from_range(mn, mx, s)
    if s.calib != "mse":
        return scale, zero
    best_err = torch.full_like(scale, float("inf"), dtype=torch.float64)
    one = np.float32(1.0)
    for i in range(s.grid + 1):
        # float32, same expression order as ops.cpp: 1 - (i * (1 - maxshrink)) / grid
        shrink = float(one - np.float32(i) * (one - np.float32(s.maxshrink)) / np.float32(s.grid))
        cs, cz = params_from_range(mn * shrink, mx * shrink, s)
        err = (qdq(g, cs, cz, s).double() - g.double()).pow(2).sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        scale = torch.where(better, cs, scale)
        zero = torch.where(better, cz, zero)
    return scale, zero


def fake_quant(x: torch.Tensor, s: QuantSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize-dequantize a 2-D tensor (computed in float32). Returns (dq, scale, zero)."""
    if not s.enabled:
        raise ValueError("quantization spec is disabled")
    xf = x.detach().to(torch.float32)
    if xf.dim() != 2:
        raise ValueError(f"expected a 2-D tensor, got shape {tuple(xf.shape)}")
    scale, zero = compute_qparams(xf, s)
    g = _groups(xf, s)
    out = qdq(g, scale, zero, s).reshape(xf.shape)
    return out, scale, zero


def fake_quant_act(x: torch.Tensor, s: QuantSpec) -> torch.Tensor:
    """Dynamic activation quantization over the last dim (per-token) or whole tensor (per-tensor)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).to(torch.float32)
    out, _, _ = fake_quant(x2, s)
    return out.reshape(shape).to(x.dtype)
