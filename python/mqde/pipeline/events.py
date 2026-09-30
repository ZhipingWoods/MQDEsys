"""Structured job events (api/events.schema.json shape): JSON Lines on disk + optional callback.

Every stage transition, progress tick, log line, metric and per-layer trace is an event with a
monotonically increasing `seq`, so the UI can replay a job or resume a stream after a disconnect.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class EventSink:
    def __init__(self, job_id: str, path: Path, echo: bool = True, callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.job_id = job_id
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)  # noqa: SIM115  (lives for the job; closed by close())
        self._seq = 0
        self._lock = threading.Lock()
        self.echo = echo
        self.callback = callback
        self.stage: str | None = None

    def emit(self, type_: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            ev = {"job_id": self.job_id, "seq": self._seq, "ts": time.time(), "type": type_, "stage": self.stage, **payload}
            self._f.write(json.dumps(ev, default=_json_default) + "\n")
        if self.echo and type_ in ("stage_started", "stage_finished", "log", "job_state", "metric"):
            msg = payload.get("msg") or payload.get("name") or payload.get("state") or ""
            extra = f" {payload['value']}" if "value" in payload else ""
            print(f"[{time.strftime('%H:%M:%S')}] {type_:<15} {self.stage or '-':<14} {msg}{extra}", file=sys.stderr, flush=True)
        if self.callback:
            self.callback(ev)
        return ev

    def log(self, msg: str, level: str = "info") -> None:
        self.emit("log", level=level, msg=msg)

    def start_stage(self, stage: str, **kw: Any) -> float:
        self.stage = stage
        self.emit("stage_started", msg=kw.pop("msg", ""), **kw)
        return time.perf_counter()

    def finish_stage(self, t0: float, status: str = "succeeded", **kw: Any) -> None:
        self.emit("stage_finished", status=status, seconds=time.perf_counter() - t0, msg=status, **kw)

    def close(self) -> None:
        self._f.close()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, float) and o != o:
        return None
    return str(o)
