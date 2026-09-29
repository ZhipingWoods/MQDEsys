"""Sensitivity analysis by per-group FP fallback (T-0902) and a mixed-precision suggestion (T-0904).

With the whole model quantized, each group of layers (a decoder block, or one layer type such as
``mlp.down_proj`` across all blocks) is put back to its original FP weights on its own and the metric is
re-measured. ``recovered`` = metric(all quantized) - metric(this group in FP): how much of the total
degradation that group is responsible for. The ranking drives ``suggest()``: raise the bits (or keep FP)
of the most sensitive groups until the predicted recovery reaches a target share, within a weight-memory
budget. Groups are independent measurements, so their recoveries only approximately add up.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from torch import nn

from mqde.quant.modules import FakeQuantLinear, replace_module

_BLOCK = re.compile(r"^(.*?\.(?:layers|blocks|h)\.(\d+))\.")


def group_key(name: str, by: str) -> str:
    """'block:<i>' for by=block, the layer role (e.g. 'mlp.down_proj') for by=type."""
    m = _BLOCK.match(name)
    if by == "block":
        return f"block:{m.group(2)}" if m else "other"
    return name[m.end() :] if m else name


def fallback_sensitivity(
    model: nn.Module,
    originals: dict[str, nn.Module],
    metric: Callable[[], float],
    by: str = "block",
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """originals: layer name -> the FP module it replaced (BlockwiseQuantizer.originals). Lower metric = better."""
    t0 = time.perf_counter()
    fq = {n: m for n, m in model.named_modules() if isinstance(m, FakeQuantLinear)}
    groups: dict[str, list[str]] = {}
    for n in fq:
        if n in originals:
            groups.setdefault(group_key(n, by), []).append(n)
    base = metric()
    rows = []
    for i, (g, names) in enumerate(sorted(groups.items(), key=lambda kv: _order(kv[0]))):
        try:
            for n in names:
                replace_module(model, n, originals[n])
            v = metric()
        finally:
            for n in names:
                replace_module(model, n, fq[n])
        w_bits = sum(fq[n].w_spec.bits * fq[n].in_features * fq[n].out_features for n in names)
        params = sum(fq[n].in_features * fq[n].out_features for n in names)
        rows.append({"group": g, "layers": len(names), "params": params, "metric": v, "recovered": base - v,
                     "avg_bits": w_bits / max(params, 1)})
        if progress:
            progress(i + 1, len(groups), g)
    ranked = sorted(rows, key=lambda r: -r["recovered"])
    return {"by": by, "metric_quantized": base, "groups": rows, "ranking": [r["group"] for r in ranked],
            "seconds": time.perf_counter() - t0}


def _order(g: str) -> tuple[int, str]:
    return (int(g.split(":")[1]), "") if g.startswith("block:") else (1 << 30, g)


def suggest(
    sens: dict[str, Any],
    metric_fp: float,
    target_share: float = 0.5,
    max_extra_bytes: float | None = None,
    bits: int = 8,
) -> dict[str, Any]:
    """Pick the most sensitive groups (greedy by recovered-per-extra-byte) until their summed recovery reaches
    target_share of the total degradation (metric_quantized - metric_fp) or the byte budget is exhausted."""
    total = sens["metric_quantized"] - metric_fp
    picked, rec, extra = [], 0.0, 0.0
    cands = [g for g in sens["groups"] if g["recovered"] > 0]
    cost = lambda g: g["params"] * max(bits - g["avg_bits"], 0) / 8  # noqa: E731
    for g in sorted(cands, key=lambda g: -g["recovered"] / max(cost(g), 1.0)):
        if total > 0 and rec >= target_share * total:
            break
        if max_extra_bytes is not None and extra + cost(g) > max_extra_bytes:
            continue
        picked.append(g)
        rec += g["recovered"]
        extra += cost(g)
    rules = []
    for g in picked:
        if sens["by"] == "block":
            rules.append({"match": g["group"], "bits": bits})
        else:
            rules.append({"match": re.escape(g["group"]) + "$", "bits": bits})
    return {
        "rules": rules,
        "groups": [g["group"] for g in picked],
        "predicted_recovered": rec,
        "predicted_metric": sens["metric_quantized"] - rec,
        "degradation": total,
        "extra_weight_bytes": extra,
        "bits": bits,
        "note": "sum of independent single-group recoveries; re-run the job to measure the actual effect",
    }
