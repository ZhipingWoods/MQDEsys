#!/usr/bin/env python3
"""Module dependency direction check (T-0005, STRUCTURE.md 3.x).

Every `#include <mqde/<module>/...>` (or "mqde/...") in cpp/src, cpp/apps and cpp/bindings must point at the
including file's own module or at one it is allowed to depend on. The graph below is the transitive closure of
the CMake link graph; adding an edge here means changing the architecture, so it needs an L2 review.

    common  <- modelspec, quant
    modelspec, quant <- estimator
    estimator <- fit, hub
    fit, hub, estimator, probe <- server <- apps/bindings

Usage: scripts/check_deps.py [--root <repo>]   (exit 1 and a list of violations on failure)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED: dict[str, set[str]] = {
    "common": set(),
    "quant": {"common"},
    "modelspec": {"common"},
    "estimator": {"common", "modelspec", "quant"},
    "fit": {"common", "modelspec", "quant", "estimator"},
    "hub": {"common", "modelspec", "quant", "estimator"},
    "server": {"common", "modelspec", "quant", "estimator", "fit", "hub"},
}

# Entry points and tests may use every module.
EVERYTHING = {"common", "modelspec", "quant", "estimator", "fit", "hub", "server"}
ALLOWED.update({"apps": EVERYTHING, "bindings": EVERYTHING, "tests": EVERYTHING})

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]mqde/([a-z_]+)/')


def module_of(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    if parts[:2] == ("cpp", "src"):
        return parts[2]
    if parts[:2] in (("cpp", "apps"), ("cpp", "bindings"), ("cpp", "tests")):
        return parts[1]
    if parts[:3] == ("cpp", "include", "mqde"):  # public headers follow the same direction
        return parts[3]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    root = Path(a.root).resolve()
    bad: list[str] = []
    checked = 0
    for f in sorted((root / "cpp").rglob("*")):
        if f.suffix not in (".cpp", ".h", ".c", ".hpp") or not f.is_file():
            continue
        mod = module_of(f, root)
        if mod is None:
            continue
        allowed = ALLOWED.get(mod)
        if allowed is None:
            bad.append(f"{f.relative_to(root)}: unknown module '{mod}' (add it to scripts/check_deps.py)")
            continue
        checked += 1
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            m = INCLUDE.match(line)
            if m and (dep := m.group(1)) != mod and dep not in allowed:
                bad.append(f"{f.relative_to(root)}:{i}: '{mod}' must not depend on '{dep}' ({line.strip()})")
    for b in bad:
        print(f"error: {b}", file=sys.stderr)
    print(f"check_deps: {checked} files, {len(bad)} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
