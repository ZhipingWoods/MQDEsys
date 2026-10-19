#!/usr/bin/env python3
"""Checks that python/mqde/backend/registry.py's registered backend ids and
data/domains/*.yaml deploy_backends lists stay in sync (T-E).

Each backend id that any domain declares in its deploy_backends must be registered in
BACKEND_REGISTRY; any backend in the registry that no domain declares is a warning, not
an error (it may serve deploy-job flows that don't go through the domain wizard).

Parses backend/registry.py as text to extract register_backend(BackendEntry(id=...)) calls
rather than importing the module, so this script needs only PyYAML (a base mqde dependency)
and not torch -- same approach as check_algo_registry_sync.py.

Usage: scripts/check_backend_registry_sync.py [--root <repo>]   (exit 1 on mismatch)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENTRY_ID_RE = re.compile(r'BackendEntry\s*\(.*?id\s*=\s*"([a-z_]+)"', re.DOTALL)


def registered_ids(registry_py: Path) -> set[str]:
    return set(ENTRY_ID_RE.findall(registry_py.read_text()))


def domain_deploy_backends(domains_dir: Path) -> dict[str, list[str]]:
    """Returns {domain_id: [backend_id, ...]} for every domain that declares deploy_backends."""
    import yaml

    result: dict[str, list[str]] = {}
    for f in sorted(domains_dir.glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        if d and d.get("deploy_backends"):
            result[d["id"]] = d["deploy_backends"]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Check backend registry / domain deploy_backends sync")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.root)

    registry_py = root / "python" / "mqde" / "backend" / "registry.py"
    if not registry_py.exists():
        print(f"ERROR: {registry_py} not found", file=sys.stderr)
        return 1

    domains_dir = root / "data" / "domains"
    if not domains_dir.is_dir():
        print(f"ERROR: {domains_dir} not found", file=sys.stderr)
        return 1

    registered = registered_ids(registry_py)
    domains = domain_deploy_backends(domains_dir)

    errors: list[str] = []
    warnings: list[str] = []

    # Every backend declared by a domain must be registered
    all_domain_backends: set[str] = set()
    for domain_id, backends in domains.items():
        for b in backends:
            all_domain_backends.add(b)
            if b not in registered:
                errors.append(
                    f"  {domain_id}.yaml declares deploy_backend '{b}' but it is NOT in BACKEND_REGISTRY"
                )

    # Backends in the registry not claimed by any domain (informational)
    undeclared = registered - all_domain_backends
    if undeclared:
        warnings.append(
            f"  Backends registered but not declared by any domain: {sorted(undeclared)}"
            " (OK if used only in deploy-job flows)"
        )

    if errors:
        print("FAIL  backend registry / domain deploy_backends mismatch:")
        for e in errors:
            print(e)
        if warnings:
            print("NOTE (non-fatal):")
            for w in warnings:
                print(w)
        return 1

    print(f"OK  registry has {sorted(registered)} matching domain declarations {dict(sorted(domains.items()))}")
    if warnings:
        print("NOTE (non-fatal):")
        for w in warnings:
            print(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
