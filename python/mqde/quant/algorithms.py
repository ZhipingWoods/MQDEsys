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
SmoothQuant (Xiao et al. 2022); HQQ (Badri & Shaji 2023) for the data-free 2-bit path.
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

# Quantizers that need no calibration activations. ``run`` uses this to decide whether a
# "no calibration data" warning is appropriate.
DATA_FREE = {"rtn", "hqq"}


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
    quantizers = [p for p in parts if p in ("rtn", "gptq", "hqq")]
    if len(transforms) > 1 or len(quantizers) > 1:
        raise ValueError(
            f"algorithm '{name}': at most one transform (awq|smoothquant) and one quantizer (rtn|gptq|hqq)")
    return transforms + (quantizers or ["rtn"])


def run(name: str, w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec) -> AlgoResult:
    """Apply a (possibly combined) algorithm to one layer."""
    w = w.detach().to(torch.float32)
    in_scale: torch.Tensor | None = None
    info: dict[str, Any] = {}
    final = False
    scale = zero = None
    for step in resolve(name):
        needs_x = step not in DATA_FREE
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
    scaled = w * best_s
    clip_grid = int(p.get("clip_grid", 0))
    if clip_grid > 0:
        scaled, clip_info = _awq_clip(x, w, scaled, best_s, ws, clip_grid)
    else:
        clip_info = {}
    return AlgoResult(w=scaled, in_scale=best_s, info={"alpha": best_a, "out_mse": best_err, **clip_info})


def _awq_clip(x: torch.Tensor, w_orig: torch.Tensor, w_scaled: torch.Tensor, s: torch.Tensor,
              ws: QuantSpec, grid: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """AWQ's optional weight clipping (llm-awq ``apply_clip``).

    For each output channel the search picks the clipping ratio in ``(0, 1]`` that minimises the
    layer's output error after quantization.  Clipping is applied to the already-scaled weight and
    the inverse scale is folded into ``in_scale`` by the caller, so it stays FP-exact before QDQ.
    """
    amax = w_scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    best = w_scaled.clone()
    best_ratio = torch.ones(w_scaled.shape[0], 1, device=w_scaled.device)
    best_err = torch.full((w_scaled.shape[0], 1), float("inf"), device=w_scaled.device, dtype=torch.float64)
    ratios = torch.linspace(1.0, 0.5, grid, device=w_scaled.device)
    y = x @ w_orig.T  # the FP reference output; clipping must be judged against this, not the scaled weight
    for r in ratios:
        cand = w_scaled.clamp(-r * amax, r * amax)
        wq, _, _ = fake_quant(cand, ws)
        # The layer will see (x / s) @ wq^T.  Compare that with the FP output x @ w^T; splitting the
        # division between the two operands (the previous version used (x / s) @ (wq / s)^T) divides
        # by s twice and makes the clip search minimise the wrong objective.
        err = ((y - (x / s) @ wq.T) ** 2).mean(dim=0, keepdim=True).T
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best = torch.where(better, cand, best)
        best_ratio = torch.where(better, torch.full_like(best_ratio, float(r)), best_ratio)
    return best, {"clip_grid": grid, "clip_ratio_mean": float(best_ratio.mean()),
                  "clip_ratio_min": float(best_ratio.min())}


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
    damp_step = float(p.get("damp_auto_increment", 0.0015))
    damp_tries = max(1, int(p.get("damp_tries", 6)))
    static_groups = bool(p.get("static_groups", False))
    x = _subsample(x.to(w.device, torch.float32), int(p.get("max_rows", 2048)))
    if as_.enabled:  # the layer will see quantized activations at inference
        x = fake_quant_act(x, as_)
    cols = w.shape[1]
    W = w.clone()
    H = (2.0 / x.shape[0]) * (x.T @ x)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    diag_mean = torch.mean(torch.diag(H))
    Hinv = None
    tries = 0
    while Hinv is None and tries < damp_tries:
        Hd = H + torch.eye(cols, device=w.device) * damp * diag_mean
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
        except RuntimeError:
            damp += damp_step
            tries += 1
    if Hinv is None:
        return AlgoResult(w=w, info={"note": f"Hessian not positive definite after {damp_tries} damping "
                                             "tries, RTN used", "damp": damp})

    gs = ws.group_size if ws.granularity == "per_group" else cols
    col_spec = ws.with_granularity("per_channel")  # a [rows, g] slice with one scale per row
    scale = zero = None
    params: list[tuple[torch.Tensor, torch.Tensor]] = []
    static_scale = static_zero = None
    if static_groups:
        static_scale, static_zero = compute_qparams(W, ws)
        n_groups = static_scale.shape[0]
        static_scale = static_scale.reshape(W.shape[0], n_groups // W.shape[0])
        static_zero = static_zero.reshape(W.shape[0], n_groups // W.shape[0])
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if static_groups:
                assert static_scale is not None and static_zero is not None
                gi = 0 if ws.granularity == "per_tensor" else c // gs
                scale, zero = static_scale[:, gi : gi + 1], static_zero[:, gi : gi + 1]
            elif ws.granularity == "per_tensor":
                if scale is None:
                    scale, zero = compute_qparams(W, ws)
                    params.append((scale, zero))
            elif c % gs == 0:
                scale, zero = compute_qparams(
                    W[:, c : c + gs].contiguous() if ws.granularity == "per_group" else W, col_spec)
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
    if static_groups:
        assert static_scale is not None and static_zero is not None
        sc = static_scale.reshape(-1, 1)
        zr = static_zero.reshape(-1, 1)
    else:
        # [rows, groups] -> [rows * groups, 1]: the row-major group order of QuantSpec / compute_qparams
        sc = torch.cat([p_[0] for p_ in params], dim=1).reshape(-1, 1)
        zr = torch.cat([p_[1] for p_ in params], dim=1).reshape(-1, 1)
    return AlgoResult(w=Q, final=True, scale=sc, zero=zr,
                      info={"damp": damp, "block": block, "rows": int(x.shape[0]),
                            "static_groups": static_groups, "damp_tries": tries})


# ---------------------------------------------------------------- HQQ
def _hqq_groups(w: torch.Tensor, ws: QuantSpec) -> tuple[torch.Tensor, tuple[int, ...]]:
    """View a weight as HQQ's [groups, group_size] matrix, preserving the input shape."""
    shape = tuple(w.shape)
    if ws.granularity == "per_tensor":
        return w.reshape(1, -1), shape
    if ws.granularity == "per_group":
        if w.shape[1] % ws.group_size:
            raise ValueError(f"in_features ({w.shape[1]}) is not divisible by group_size ({ws.group_size})")
        return w.reshape(-1, ws.group_size), shape
    return w.reshape(w.shape[0], -1), shape


def _shrink_lp(x: torch.Tensor, beta: float, lp: float) -> torch.Tensor:
    """Proximal operator of ``|x|^lp`` (HQQ's shrink_lp_op), used to pull the residual to zero."""
    ax = x.abs()
    if lp >= 1.0:
        return torch.sign(x) * (ax - 1.0 / beta).clamp(min=0.0)
    return torch.sign(x) * (ax - (1.0 / beta) * ax.clamp(min=1e-12).pow(lp - 1.0)).clamp(min=0.0)


@register("hqq")
def hqq(w: torch.Tensor, x: torch.Tensor | None, ws: QuantSpec, as_: QuantSpec, p: dict[str, Any]) -> AlgoResult:
    """Half-Quadratic Quantization (data-free, no calibration activations needed).

    HQQ alternates three exact steps per group: project the current weights onto the integer grid,
    apply the proximal operator of an Lp (p<1) penalty to the residual, and solve the two-variable
    least-squares problem for scale/zero.  The Lp residual is what makes 2-bit groups usable without
    calibration data; RTN is the special case with no residual and a single min/max scale.

    Like GPTQ this returns a weight already on the quantization grid plus the qparams that produced
    it, so the exporters can recover the integer codes without re-deriving them.
    """
    if ws.format != "int":
        return AlgoResult(w=w, info={"note": "fp8 weights: HQQ is for integer grids, RTN used"})
    iters = max(1, int(p.get("iters", 20)))
    lp = float(p.get("lp_norm", 0.7))
    beta = float(p.get("beta", 1.0))
    kappa = float(p.get("kappa", 1.01))
    eps = float(p.get("min_scale", 1e-8))

    W, shape = _hqq_groups(w, ws)
    W = W.to(torch.float32)
    n, k = W.shape
    qmin, qmax = float(ws.qmin), float(ws.qmax)
    # Start from the same min/max solution RTN would use, then refine.
    scale, zero = compute_qparams(w, ws)
    scale = scale.reshape(n, 1).to(torch.float32)
    zero = zero.reshape(n, 1).to(torch.float32)
    W_work = W.clone()
    for _ in range(iters):
        q = torch.clamp(torch.round(W_work / scale) + zero, qmin, qmax)
        W_r = (q - zero) * scale
        W_work = W_r + _shrink_lp(W_work - W_r, beta, lp)
        if ws.symmetric:
            denom = (q * q).sum(dim=1, keepdim=True).clamp(min=eps)
            scale = (W * q).sum(dim=1, keepdim=True) / denom
            zero = torch.zeros_like(zero)
        else:
            # W ~ a*q + b; zero = -b/a.  Solve the 2x2 normal equations per group.
            sq, sqq = q.sum(dim=1, keepdim=True), (q * q).sum(dim=1, keepdim=True)
            sw, swq = W.sum(dim=1, keepdim=True), (W * q).sum(dim=1, keepdim=True)
            denom = (k * sqq - sq * sq)
            a = (k * swq - sq * sw) / denom.clamp(min=eps)
            b = (sw - a * sq) / k
            good = denom.abs() > eps
            a = torch.where(good, a, scale)
            b = torch.where(good, b, zero * scale)
            scale = a.abs().clamp(min=eps)
            zero = (-b / scale).clamp(qmin, qmax)
        beta *= kappa
    # Checkpoints store integer zero-points, so finish on that grid and re-solve the scale for it.
    if not ws.symmetric:
        zero = torch.round(zero)
    q = torch.clamp(torch.round(W / scale) + zero, qmin, qmax)
    denom = ((q - zero) ** 2).sum(dim=1, keepdim=True).clamp(min=eps)
    scale = (W * (q - zero)).sum(dim=1, keepdim=True) / denom
    scale = scale.abs().clamp(min=eps)
    q = torch.clamp(torch.round(W / scale) + zero, qmin, qmax)
    dq = ((q - zero) * scale).reshape(shape)
    return AlgoResult(w=dq, final=True, scale=scale, zero=zero,
                      info={"iters": iters, "lp_norm": lp, "data_free": True})
