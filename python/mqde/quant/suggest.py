"""Dynamic quantization (T-R10, after unsloth's dynamic quants): choose which layers to leave unquantized from the per-layer
quantization error alone, so no evaluation run is needed to propose them.

A layer is flagged when its output SQNR is a robust outlier *within its own component* (the vision tower is as a whole
noisier than the language model; that is the component, not a bad layer): below median - `z` * 1.4826 * MAD, and at least
`min_gap_db` below the median. At most `max_fraction` of all layers (rounded up) are flagged, lowest SQNR first. The rules are anchored
regexes in the `quant.mixed_precision` format the jobs already read.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any

MIN_PER_COMPONENT = 8  # fewer layers than this cannot tell an outlier from noise


def suggest_skips(layers: list[dict[str, Any]], bits: int, *, z: float = 3.5, min_gap_db: float = 3.0,
                  max_fraction: float = 0.05) -> dict[str, Any]:
    by_comp: dict[str, list[dict[str, Any]]] = {}
    for t in layers:
        if t.get("out_sqnr_db") is not None:
            by_comp.setdefault(t["component"], []).append(t)
    flagged: list[dict[str, Any]] = []
    for comp, ts in by_comp.items():
        if len(ts) < MIN_PER_COMPONENT:
            continue
        vals = [t["out_sqnr_db"] for t in ts]
        med = statistics.median(vals)
        mad = 1.4826 * statistics.median(abs(v - med) for v in vals)
        thr = min(med - z * mad, med - min_gap_db)
        flagged += [{"name": t["name"], "component": comp, "out_sqnr_db": t["out_sqnr_db"], "median_db": med, "shape": t["shape"]}
                    for t in ts if t["out_sqnr_db"] < thr]
    total = sum(len(ts) for ts in by_comp.values())
    flagged = sorted(flagged, key=lambda x: x["out_sqnr_db"])[: math.ceil(total * max_fraction)]
    return {
        "rules": [{"match": f"^{re.escape(x['name'])}$", "skip": True} for x in flagged],
        "layers": [{k: x[k] for k in ("name", "component", "out_sqnr_db", "median_db")} for x in flagged],
        "extra_weight_bytes": sum(x["shape"][0] * x["shape"][1] * (16 - bits) // 8 for x in flagged),
        "basis": f"output SQNR below its component's median by > {z} robust sigmas and >= {min_gap_db} dB",
    }
