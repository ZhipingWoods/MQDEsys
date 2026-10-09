"""``python -m mqde.report validate|html report.json``  (T-0806)

validate   check report.json against the schema; exit 1 and list every problem if invalid
html       write report.<lang>.html next to the report (both languages unless --lang is given)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .html import render_html
from .schema import validate_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mqde.report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("report")
    h = sub.add_parser("html")
    h.add_argument("report")
    h.add_argument("--lang", choices=["zh", "en"])
    a = ap.parse_args(argv)
    src = Path(a.report)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"{src}: cannot read as JSON: {e}", file=sys.stderr)
        return 2
    errs = validate_report(data)
    if a.cmd == "validate":
        for m in errs:
            print(m)
        print(f"{src}: {'INVALID, ' + str(len(errs)) + ' problem(s)' if errs else 'valid'}")
        return 1 if errs else 0
    if errs:
        print(f"{src}: not a valid report ({len(errs)} problem(s)); run `validate` for the list", file=sys.stderr)
        return 1
    for lang in [a.lang] if a.lang else ["en", "zh"]:
        out = src.with_name(f"report.{lang}.html")
        out.write_text(render_html(data, lang), encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
