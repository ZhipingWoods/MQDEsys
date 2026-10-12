#!/usr/bin/env python3
"""Checks that python/mqde/quant/algorithms.py's @register(...) names and data/algorithms.yaml's entries
agree (T-1302). The two are hand-maintained independently -- algorithms.py is what the Python engine
actually runs; data/algorithms.yaml is what GET /api/v1/algorithms serves and what the server validates
quant.algorithm against (cpp/src/planner/validate.cpp) -- and nothing enforced they stay in sync before
this script.

A yaml entry with kind: combo (e.g. "awq+gptq") is NOT expected to have a matching @register: combos are
resolved at runtime by splitting the name on "+" and validating each part (algorithms.py:resolve()), they
are never registry keys themselves. Every "quantizer"/"transform" kind entry must have one.

Parses algorithms.py as text (regex for @register("...")) rather than importing it, so this check needs
only pyyaml (already a base mqde dependency) and not torch, which importing the real module would pull in
transitively -- same style as scripts/check_deps.py's regex-based C++ #include scan.

Usage: scripts/check_algo_registry_sync.py [--root <repo>]   (exit 1 and a list of mismatches on failure)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REGISTER_RE = re.compile(r'@register\(\s*"([a-z_]+)"\s*\)')


def registered_names(algorithms_py: Path) -> set[str]:
    return set(REGISTER_RE.findall(algorithms_py.read_text()))


def yaml_entries(algorithms_yaml: Path) -> list[dict]:
    import yaml

    with open(algorithms_yaml) as f:
        return yaml.safe_load(f)["algorithms"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()

    algorithms_py = args.root / "python" / "mqde" / "quant" / "algorithms.py"
    algorithms_yaml = args.root / "data" / "algorithms.yaml"

    registered = registered_names(algorithms_py)
    entries = yaml_entries(algorithms_yaml)
    non_combo = {e["name"] for e in entries if e.get("kind") != "combo"}
    combo = {e["name"] for e in entries if e.get("kind") == "combo"}

    problems = []
    missing_from_yaml = registered - non_combo
    if missing_from_yaml:
        problems.append(
            f"registered in algorithms.py but missing from data/algorithms.yaml: {sorted(missing_from_yaml)}"
        )
    missing_from_registry = non_combo - registered
    if missing_from_registry:
        problems.append(
            f"in data/algorithms.yaml (non-combo) but never @register()ed in algorithms.py: "
            f"{sorted(missing_from_registry)}"
        )
    # A combo name must only reference registered parts (resolve()'s own job at runtime; checked here too
    # so a typo in data/algorithms.yaml is caught statically instead of only when someone submits a job).
    for name in sorted(combo):
        parts = [p for p in name.split("+") if p]
        unknown = [p for p in parts if p not in registered]
        if unknown:
            problems.append(f"combo '{name}' references unregistered part(s): {unknown}")

    if problems:
        print("check_algo_registry_sync: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"check_algo_registry_sync: OK ({len(registered)} registered, {len(entries)} yaml entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
