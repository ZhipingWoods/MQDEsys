#!/usr/bin/env python3
"""Stand-in for the job executor, for UI tests: `mqde serve --python web/e2e/fake_worker.py`.

The server launches `<python> -m <module> <job.yaml> --job-id ID --device D`. This script ignores `-m <module>`, copies
the submitted job.yaml next to a minimal event stream, and exits 0 (or 1 when the job name contains "fail"), so a test
can submit through the real REST API without a GPU, a model or any quantization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", dest="module")
    ap.add_argument("job")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import re

    text = Path(a.job).read_text()
    out = Path(re.search(r"^output_dir:\s*(.+)$", text, re.M).group(1).strip().strip("'\"")) / a.job_id  # type: ignore[union-attr]
    out.mkdir(parents=True, exist_ok=True)
    seq = 0

    def emit(type_: str, **kw: object) -> None:
        nonlocal seq
        seq += 1
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps({"job_id": a.job_id, "seq": seq, "ts": time.time(), "type": type_, **kw}) + "\n")

    failed = "fail" in text.lower().split("name:", 1)[-1].split("\n", 1)[0]
    emit("job_state", state="RUNNING")
    emit("stage_started", stage="validate")
    emit("stage_finished", stage="validate", status="succeeded", seconds=0.01)
    (out / "report.json").write_text(json.dumps({"job_id": a.job_id, "kind": "llm", "fake_worker": True, "stages": []}))
    emit("job_state", state="FAILED" if failed else "SUCCEEDED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
