#!/usr/bin/env bash
# scripts/bench_reference.sh --suite m4 --server URL --model DIR [--out report.md] [--dry-run]
# or:     scripts/bench_reference.sh --report runs/<job>/report.json [--report ...] [--estimates est.json] [--out FILE]
exec python3 "$(dirname "$0")/bench_reference.py" "$@"
