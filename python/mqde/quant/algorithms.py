"""Weight-quantization algorithms (T-0601 AWQ, T-0602 GPTQ, T-0603 SmoothQuant).

Every algorithm works on one linear layer at a time, given its FP weight ``W`` [out, in] and a
sample of its real input activations ``X`` [rows, in] captured on calibration data. It returns

* ``w``      the weight to hand to the recipe quantizer (possibly pre-compensated / pre-scaled), and
* ``in_scale`` an optional per-input-channel divisor ``s``: the layer then computes
  ``Q(X / s) @ Q(W * s)^T``. For a plain linear this is mathematically identical in FP, so the
  transform is exact before quantization ("transformed" eval position) and only redistributes
  quantization difficulty between activations and weights.

The final quantize-dequantize always goes through ``mqde.quant.numerics.fake_quant`` (bit-exact
with the C++ core), so every algorithm shares one numerical definition of the quantized format.
GPTQ is the exception that it needs the quantizer inside its column loop; it uses the same
``compute_qparams`` / ``qdq`` primitives with the group's parameters fixed up front, as GPTQModel does.

References: GPTQ (Frantar et al. 2022) / GPTQModel; AWQ (Lin et al. 2023) / AutoAWQ;
SmoothQuant (Xiao et al. 2022).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from mqde.quant.numerics import QuantSpec, compute_qparams, fake_quant, fake_quant_act, qdq


@dataclass
class AlgoResult:
    w: torch.Tensor  # float32 weight to quantize with the recipe spec
    in_scale: torch.Tensor | None = None  # [in] divisor applied to activations (None = identity)
    info: dict[str, Any] | None = None
    final: bool = False  # w is already quantize-dequantized (GPTQ); do not re-quantize
    scale: torch.Tensor | None = None  # when final: the weight qparams used, [n_groups, 1] (QuantSpec group order)
    zero: torch.Tensor | None = None


AlgoFn = Callable[[torch.Tensor, torch.Tensor | None, QuantSpec, QuantSpec, dict[str, Any]], AlgoResult]

ALGO_REGISTRY: dict[str, AlgoFn] = {}


def register(name: str) -> Callable[[AlgoFn], AlgoFn]:
    def deco(fn: AlgoFn) -> AlgoFn:
        ALGO_REGISTRY[name] = fn
        return fn

    return deco


def _data_dir() -> Path:
    from mqde.models.base import data_dir

    return data_dir()


def algorithm_catalog() -> list[dict[str, Any]]:
    with open(_data_dir() / "algorithms.yaml") as f:
        return yaml.safe_load(f)["algorithms"]


def algorithm_params(name: str) -> dict[str, Any]:
    for a in algorithm_catalog():
        if a["name"] == name:
            return dict(a.get("params", {}))
    return {}


def resolve(name: str) -> list[str]:
    """'awq+gptq' -> ['awq', 'gptq']; validates every part."""
    parts = [p.strip().lower() for p in name.split("+") if p.strip() and p.strip().lower() != "none"]
    if not parts:
        parts = ["rtn"]
    for p in parts:
        if p not in ALGO_REGISTRY:
            raise ValueError(f"unknown quantization algorithm '{p}' (known: {', '.join(sorted(ALGO_REGISTRY))})")
    transforms = [p for p in parts if p in ("awq", "smoothquant")]
    quantizers = [p for p in parts if p in ("rtn", "gptq")]
    if len(transforms) > 1 or len(quantizers) > 1:
        raise ValueError(f"algorithm '{name}': at most one transform (awq|smoothquant) and one quantizer (rtn|gptq)")
    return transforms + (quantizers or ["rtn"])


def run(name: str, w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec) -> AlgoResult:
    """Apply a (possibly combined) algorithm to one layer."""
    w = w.detach().to(torch.float32)
    in_scale: torch.Tensor | None = None
    info: dict[str, Any] = {}
    final = False
    scale = zero = None
    for step in resolve(name):
        needs_x = step != "rtn"
        if needs_x and (x is None or x.numel() == 0):
            info[step] = "skipped: no calibration activations"
            continue
        xs = x if in_scale is None or x is None else x / in_scale
        r = ALGO_REGISTRY[step](w, xs, ws, as_, algorithm_params(step))
        w = r.w
        if r.in_scale is not None:
            in_scale = r.in_scale if in_scale is None else in_scale * r.in_scale
        if r.info:
            info[step] = r.info
        if r.final:
            final, scale, zero = True, r.scale, r.zero
    return AlgoResult(w=w, in_scale=in_scale, info=info, final=final, scale=scale, zero=zero)


def materialize(ar: AlgoResult, ws: QuantSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(quantize-dequantized weight, scale, zero) of an algorithm result. The qparams are exactly the
    ones that produced the weight, so exporters can recover integer codes without re-deriving them."""
    if ar.final and ar.scale is not None:
        return ar.w, ar.scale, ar.zero
    if ar.final:  # a quantizer that did not report qparams: the weight is on its grid already
        raise ValueError("final algorithm result without qparams")
    return fake_quant(ar.w, ws)


def _subsample(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[0] <= rows:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, rows, device=x.device).long()
    return x[idx]


def _out_err(x: torch.Tensor, w: torch.Tensor, wq: torch.Tensor, xq: torch.Tensor | None = None) -> float:
    y = x @ w.T
    yq = (x if xq is None else xq) @ wq.T
    return float((y - yq).pow(2).mean())


# ---------------------------------------------------------------- RTN
@register("rtn")
def rtn(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec, p: dict[str, Any]) -> AlgoResult:
    return AlgoResult(w=w)


# ---------------------------------------------------------------- SmoothQuant
@register("smoothquant")
def smoothquant(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec, p: dict[str, Any]) -> AlgoResult:
    """s_j = max|X_j|^a / max|W_:,j|^(1-a); X/s and W*s. Exact in FP for a plain linear."""
    assert x is not None
    alpha = float(p.get("alpha", 0.5))
    x = _subsample(x.to(w.device, torch.float32), int(p.get("max_rows", 512)))
    ax = x.abs().amax(dim=0).clamp(min=1e-5)
    aw = w.abs().amax(dim=0).clamp(min=1e-5)
    s = (ax.pow(alpha) / aw.pow(1 - alpha)).clamp(min=1e-4, max=1e4)
    return AlgoResult(w=w * s, in_scale=s, info={"alpha": alpha, "s_max": float(s.max()), "s_min": float(s.min())})


# ---------------------------------------------------------------- AWQ
@register("awq")
def awq(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec, p: dict[str, Any]) -> AlgoResult:
    """Grid search s = mean|X_j|^a over a in [0,1) minimising the quantized layer's output MSE."""
    assert x is not None
    grid = int(p.get("grid", 20))
    x = _subsample(x.to(w.device, torch.float32), int(p.get("max_rows", 512)))
    ax = x.abs().mean(dim=0).clamp(min=1e-6)
    base, _, _ = fake_quant(w, ws)
    best_err, best_a, best_s = _out_err(x, w, base), 0.0, None
    for i in range(1, grid):
        a = i / grid
        s = ax.pow(a)
        s = s / (s.max() * s.min()).sqrt()  # keep the scale centred around 1
        wq, _, _ = fake_quant(w * s, ws)
        err = _out_err(x, w, wq / s)
        if err < best_err:
            best_err, best_a, best_s = err, a, s
    if best_s is None:
        return AlgoResult(w=w, info={"alpha": 0.0, "note": "identity was best"})
    return AlgoResult(w=w * best_s, in_scale=best_s, info={"alpha": best_a, "out_mse": best_err})


# ---------------------------------------------------------------- GPTQ
@register("gptq")
def gptq(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec, p: dict[str, Any]) -> AlgoResult:
    """Column-wise OBS update with lazy blocks (Frantar et al.), H = 2 X^T X / n + damp.

    Group parameters (scale/zero) are computed from the *current* weights when a group is entered,
    matching GPTQModel's static-groups=False behaviour; weights are quantized with the same
    ``qdq`` as the recipe. FP8 formats have no integer grid to project onto, so they fall back to RTN.
    """
    assert x is not None
    if ws.format != "int":
        return AlgoResult(w=w, info={"note": "fp8 weights: GPTQ not applicable, RTN used"})
    block = int(p.get("block_size", 128))
    damp = float(p.get("damp", 0.01))
    x = _subsample(x.to(w.device, torch.float32), int(p.get("max_rows", 2048)))
    if as_.enabled:  # the layer will see quantized activations at inference
        x = fake_quant_act(x, as_)
    cols = w.shape[1]
    W = w.clone()
    H = (2.0 / x.shape[0]) * (x.T @ x)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += torch.eye(cols, device=w.device) * damp * torch.mean(torch.diag(H))
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError:
        return AlgoResult(w=w, info={"note": "Hessian not positive definite, RTN used"})

    gs = ws.group_size if ws.granularity == "per_group" else cols
    col_spec = ws.with_granularity("per_channel")  # a [rows, g] slice with one scale per row
    scale = zero = None
    params: list[tuple[torch.Tensor, torch.Tensor]] = []
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if ws.granularity == "per_tensor":
                if scale is None:
                    scale, zero = compute_qparams(W, ws)
                    params.append((scale, zero))
            elif c % gs == 0:
                scale, zero = compute_qparams(W[:, c : c + gs].contiguous() if ws.granularity == "per_group" else W, col_spec)
                params.append((scale, zero))
            wcol = W1[:, i : i + 1]
            assert scale is not None and zero is not None  # set on the first column of every group
            q = qdq(wcol, scale, zero, ws)
            Q1[:, i : i + 1] = q
            e = (wcol - q) / Hinv1[i, i]
            W1[:, i:] -= e @ Hinv1[i : i + 1, i:]
            Err1[:, i : i + 1] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    # [rows, groups] -> [rows * groups, 1]: the row-major group order of QuantSpec / compute_qparams
    sc = torch.cat([p_[0] for p_ in params], dim=1).reshape(-1, 1)
    zr = torch.cat([p_[1] for p_ in params], dim=1).reshape(-1, 1)
    return AlgoResult(w=Q, final=True, scale=sc, zero=zr, info={"damp": damp, "block": block, "rows": int(x.shape[0])})
