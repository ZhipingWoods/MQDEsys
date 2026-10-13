#!/usr/bin/env bash
# T-0808: scripts/bench_reference.sh --suite m4 --server URL --model DIR [--out docs/reports/m4-reference.md] [--dry-run]
# or:     scripts/bench_reference.sh --report runs/<job>/report.json [--report ...] [--estimates est.json] [--out FILE]
exec python3 "$(dirname "$0")/bench_reference.py" "$@"
