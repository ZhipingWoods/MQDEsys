"""Fake-quantization engine with per-layer error tracing (docs/ARCHITECTURE.md §6.7, §10).

Algorithm
---------
RTN (round-to-nearest with the recipe's calibration, e.g. MSE range search) is applied per linear
layer. Weight error is measured on the weight itself; output error is measured on *real*
calibration activations captured with forward hooks, so a layer is ranked by the damage it does
on actual VLA inputs, not on its weight statistics alone.

Stage attribution
-----------------
`apply()` can quantize an arbitrary subset of modalities, which lets the pipeline measure the
end-to-end action error of vision-only, language-only, ... quantization and attribute the total
error to the stage that introduces it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from mqde import core
from mqde.models.base import QuantModel, Unit
from mqde.quant import algorithms
from mqde.quant.modules import FakeQuantLinear, replace_module
from mqde.quant.numerics import QuantSpec, fake_quant, fake_quant_act

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class RecipeSpec:
    name: str
    weight: QuantSpec
    act: QuantSpec
    modalities: list[str]
    maturity: str = "production"
    default_algorithm: str = "rtn"

    @classmethod
    def load(cls, name: str) -> RecipeSpec:
        r = core.load_recipe(name)
        return cls(
            name=r["name"],
            weight=QuantSpec.from_dict(r["weight"]),
            act=QuantSpec.from_dict(r["act"]),
            modalities=list(r["modalities"]),
            maturity=r["maturity"],
            default_algorithm=str(r.get("default_algorithm", "rtn")),
        )


@dataclass
class LayerTrace:
    name: str
    component: str
    modality: str
    block: int
    shape: tuple[int, int]
    w_sqnr_db: float
    w_rel_l2: float
    w_cosine: float
    out_sqnr_db: float | None = None
    out_rel_l2: float | None = None
    out_cosine: float | None = None
    act_absmax: float | None = None
    act_kurtosis: float | None = None
    act_outlier_ratio: float | None = None
    w_kurtosis: float | None = None
    seconds: float = 0.0
    algorithm: str = "rtn"
    algo_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class ActivationCache:
    """Sub-sampled input activations per linear layer (rows = tokens)."""

    max_rows: int = 512
    data: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def add(self, name: str, x: torch.Tensor) -> None:
        x2 = x.detach().reshape(-1, x.shape[-1])
        have = sum(t.shape[0] for t in self.data.get(name, []))
        if have >= self.max_rows:
            return
        take = min(self.max_rows - have, x2.shape[0])
        idx = torch.linspace(0, x2.shape[0] - 1, take, device=x2.device).long()
        self.data.setdefault(name, []).append(x2[idx].to(torch.float32).cpu())

    def get(self, name: str) -> torch.Tensor | None:
        ts = self.data.get(name)
        return torch.cat(ts) if ts else None


def _finite(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    ref64, d = ref.double(), (ref.double() - approx.double())
    noise = float(d.pow(2).sum())
    return float("inf") if noise == 0 else 10.0 * float(np.log10(float(ref64.pow(2).sum()) / noise))


def compare(ref: torch.Tensor, approx: torch.Tensor) -> dict[str, float | None]:
    """Error metrics in float64 (same definitions as mqde::quant::measure)."""
    r, a = ref.double().reshape(-1), approx.double().reshape(-1)
    d = r - a
    sig, noise = float(r.pow(2).sum()), float(d.pow(2).sum())
    denom = float(r.norm() * a.norm())
    return {
        "mse": float(d.pow(2).mean()),
        "max_abs": float(d.abs().max()),
        "sqnr_db": _finite(float("inf") if noise == 0 else 10.0 * np.log10(sig / noise)),
        "cosine": float(torch.dot(r, a)) / denom if denom > 0 else 1.0,
        "rel_l2": _finite(np.sqrt(noise / sig) if sig > 0 else float("inf")),
    }


class FakeQuantEngine:
    def __init__(
        self, qmodel: QuantModel, recipe: RecipeSpec, progress: ProgressFn | None = None, algorithm: str = "rtn"
    ) -> None:
        self.qm = qmodel
        self.recipe = recipe
        self.algorithm = algorithm
        algorithms.resolve(algorithm)  # fail fast on unknown names
        self.progress = progress or (lambda e: None)
        self.originals: dict[str, nn.Linear] = {}
        self.traces: dict[str, LayerTrace] = {}

    # ---------------- selection ----------------
    def select_units(self, modalities: Iterable[str] | None = None) -> list[Unit]:
        mods = set(self.recipe.modalities if modalities is None else modalities)
        out = []
        for u in self.qm.units:
            if u.keep_fp or u.modality not in mods:
                continue
            if self.recipe.weight.granularity == "per_group" and u.shape[1] % self.recipe.weight.group_size != 0:
                continue  # left in original precision; reported by the planner
            out.append(u)
        return out

    # ---------------- calibration capture ----------------
    def capture(self, units: list[Unit], run_calibration: Callable[[], None], max_rows: int = 512) -> ActivationCache:
        cache = ActivationCache(max_rows=max_rows)
        handles = []
        for u in units:
            mod = self.qm.get_module(u.name)
            handles.append(mod.register_forward_hook(lambda m, inp, out, n=u.name: cache.add(n, inp[0])))
        try:
            run_calibration()
        finally:
            for h in handles:
                h.remove()
        return cache

    # ---------------- quantization ----------------
    @torch.inference_mode()
    def quantize_unit(self, u: Unit, acts: torch.Tensor | None) -> LayerTrace:
        t0 = time.perf_counter()
        lin = self.qm.get_module(u.name)
        assert isinstance(lin, nn.Linear), f"{u.name} is already replaced"
        w = lin.weight.detach().to(torch.float32)
        x_cal = acts.to(w.device) if acts is not None and acts.numel() > 0 else None
        ar = algorithms.run(self.algorithm, w, x_cal, self.recipe.weight, self.recipe.act)
        # Effective weight the layer applies to the *original* input: Q(W*s)/s for scaled algorithms.
        wq, w_scale, w_zero = algorithms.materialize(ar, self.recipe.weight)
        dq = wq / ar.in_scale if ar.in_scale is not None else wq
        wm = compare(w, dq)
        wstats = core.tensor_stats(w.cpu().numpy())
        tr = LayerTrace(
            name=u.name,
            component=u.component,
            modality=u.modality,
            block=u.block,
            shape=u.shape,
            w_sqnr_db=wm["sqnr_db"] if wm["sqnr_db"] is not None else float("inf"),
            w_rel_l2=wm["rel_l2"] or 0.0,
            w_cosine=wm["cosine"] or 1.0,
            w_kurtosis=wstats["kurtosis"],
        )
        tr.algorithm = self.algorithm
        tr.algo_info = ar.info or {}
        if x_cal is not None:
            x = x_cal
            y_ref = x @ w.T
            xs = x / ar.in_scale if ar.in_scale is not None else x
            xq = fake_quant_act(xs, self.recipe.act) if self.recipe.act.enabled else xs
            y_q = xq @ wq.T
            om = compare(y_ref, y_q)
            astats = core.tensor_stats(acts.numpy())
            tr.out_sqnr_db, tr.out_rel_l2, tr.out_cosine = om["sqnr_db"], om["rel_l2"], om["cosine"]
            tr.act_absmax, tr.act_kurtosis, tr.act_outlier_ratio = (
                astats["absmax"],
                astats["kurtosis"],
                astats["outlier_ratio"],
            )
        self.originals[u.name] = lin
        fq = FakeQuantLinear(lin, self.recipe.weight, self.recipe.act, u.name, w_override=wq, in_scale=ar.in_scale, final=True,
                             w_qparams=(w_scale, w_zero))
        fq.algo = self.algorithm
        replace_module(self.qm.model, u.name, fq)
        tr.seconds = time.perf_counter() - t0
        self.traces[u.name] = tr
        return tr

    def apply(self, units: list[Unit], cache: ActivationCache | None = None) -> list[LayerTrace]:
        out = []
        for i, u in enumerate(units):
            tr = self.quantize_unit(u, cache.get(u.name) if cache else None)
            out.append(tr)
            self.progress({"kind": "layer", "index": i, "total": len(units), "trace": tr.to_dict()})
        return out

    def restore(self) -> None:
        for name, lin in self.originals.items():
            replace_module(self.qm.model, name, lin)
        self.originals.clear()

    def quantized_bytes(self) -> int:
        bits = self.recipe.weight.bits if self.recipe.weight.format == "int" else 8
        return sum(t.shape[0] * t.shape[1] * bits // 8 for t in self.traces.values())
