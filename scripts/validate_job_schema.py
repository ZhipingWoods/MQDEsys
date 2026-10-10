#!/usr/bin/env python3
"""Validates examples/jobs/*.yaml against schemas/job_spec.schema.json (T-1305, T-0007's documented method).

This checks structural correctness only (types, the handful of truly required fields); it is not a
substitute for mqde::planner::validate_jobspec, which checks things (recipe/algorithm names, calibration
requirements, export compatibility) that depend on data loaded at runtime and cannot be expressed here.

Usage: scripts/validate_job_schema.py [--root <repo>]   (exit 1 and a list of violations on failure)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()

    schema_path = args.root / "schemas" / "job_spec.schema.json"
    import json

    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema)

    jobs_dir = args.root / "examples" / "jobs"
    files = sorted(jobs_dir.glob("*.yaml"))
    if not files:
        print(f"validate_job_schema: no *.yaml found under {jobs_dir}")
        return 1

    failed = 0
    for f in files:
        doc = yaml.safe_load(f.read_text())
        errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)
        if errors:
            failed += 1
            print(f"FAIL {f.relative_to(args.root)}")
            for e in errors:
                loc = "/" + "/".join(str(p) for p in e.path)
                print(f"  {loc or '(root)'}: {e.message}")
        else:
            print(f"PASS {f.relative_to(args.root)}")

    print(f"\n{len(files) - failed}/{len(files)} example job(s) valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
