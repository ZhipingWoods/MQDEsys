"""OOM / capacity boundary search (T-0804): the largest concurrency or context a backend serves, found by bisection.

Each probe starts a fresh backend configured for the point being tested (so a crash or an out-of-memory exit at one
point never poisons the next), sends ``concurrency`` simultaneous requests whose prompts fill the context, and
passes only if the server started, every request succeeded and the process is still alive. A start-up failure
carries the backend's own diagnosis (out of memory, KV cache too small, ...).

Bisection assumes monotonicity: if a point is served, every smaller one is too. ``find_max_concurrency`` fixes the
context length; ``find_max_context`` fixes the concurrency. Results include per-probe resource peaks
(``eval.resources``) and, for vLLM, the KV-cache capacity it reported, so the measured boundary can sit next to the
estimator's ``max_concurrency`` in a report.

    python -m mqde.eval.boundary vllm CKPT --gpu 3 --max-len 1024 --lo 1 --hi 256
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from mqde.backend.base import ServerBackend
from mqde.eval.loadtest import LoadSpec, run_load
from mqde.eval.resources import ResourceSampler

BackendFactory = Callable[[int, int], ServerBackend]  # (concurrency, max_model_len) -> not-yet-started backend


def bisect_max(lo: int, hi: int, probe: Callable[[int], dict[str, Any]], step: int = 1) -> dict[str, Any]:
    """Largest ``x = lo + k*step <= hi`` with ``probe(x)["ok"]``, assuming ok-ness is monotone decreasing.

    Returns {"max", "capped" (hi itself passed), "first_fail", "probes"}; ``max`` is None if even ``lo`` fails."""
    probes: list[dict[str, Any]] = []

    def run(x: int) -> bool:
        r = dict(probe(x))
        r["x"] = x
        probes.append(r)
        return bool(r["ok"])

    top = (hi - lo) // step  # index of the largest candidate
    if not run(lo):
        return {"max": None, "capped": False, "first_fail": lo, "probes": probes}
    if top == 0 or run(lo + top * step):
        return {"max": lo + top * step, "capped": True, "first_fail": None, "probes": probes}
    good, bad = 0, top  # indices: good passes, bad fails
    while bad - good > 1:
        mid = (good + bad) // 2
        if run(lo + mid * step):
            good = mid
        else:
            bad = mid
    return {"max": lo + good * step, "capped": False, "first_fail": lo + bad * step, "probes": probes}


def probe_serving(make_backend: BackendFactory, concurrency: int, max_len: int, *, n_predict: int = 8,
                  gpu: int | None = None, timeout: float = 600) -> dict[str, Any]:
    """Start a backend for (concurrency, max_len), fill the context with ``concurrency`` parallel requests."""
    backend = make_backend(concurrency, max_len)
    try:
        try:
            backend.start()
        except (RuntimeError, TimeoutError) as e:
            return {"ok": False, "stage": "startup", "reason": backend.diagnose() or str(e).splitlines()[0][:200]}
        assert backend.proc is not None
        with ResourceSampler(gpu=gpu, pid=backend.proc.pid) as smp:
            r = run_load(backend.port, LoadSpec(concurrency=concurrency, requests=concurrency, warmup=0,
                                                prompt_len=max_len - n_predict, n_predict=n_predict, timeout=timeout))
        alive = backend.proc is not None and backend.proc.poll() is None
        out: dict[str, Any] = {"ok": r["ok"] == concurrency and alive, "stage": "serve", "resources": smp.summary(),
                               "requests_ok": r["ok"], "errors": r["errors"]}
        if not alive:
            out["reason"] = backend.diagnose() or "server process died under load"
        elif r["ok"] != concurrency:
            out["reason"] = (r["errors"] or ["requests failed"])[0]
        engine = getattr(backend, "engine_info", None)
        if callable(engine):
            out["engine"] = engine()
        return out
    finally:
        backend.stop()


def find_max_concurrency(make_backend: BackendFactory, max_len: int, *, lo: int = 1, hi: int = 64, n_predict: int = 8,
                         gpu: int | None = None) -> dict[str, Any]:
    r = bisect_max(lo, hi, lambda c: probe_serving(make_backend, c, max_len, n_predict=n_predict, gpu=gpu))
    return {"axis": "concurrency", "max_len": max_len, **r}


def find_max_context(make_backend: BackendFactory, concurrency: int, *, lo: int = 256, hi: int = 8192, step: int = 256,
                     n_predict: int = 8, gpu: int | None = None) -> dict[str, Any]:
    r = bisect_max(lo, hi, lambda n: probe_serving(make_backend, concurrency, n, n_predict=n_predict, gpu=gpu), step=step)
    return {"axis": "context", "concurrency": concurrency, **r}


def compare_with_estimator(model: str, recipe: str, max_len: int, util: float, measured: dict[str, Any],
                           hardware: str = "nvidia/rtx4090", backend: str = "vllm") -> dict[str, Any]:
    """Estimator numbers for the same point beside what the backend measured (``measured`` = engine_info())."""
    from mqde import core

    e = core.estimate(model, recipe, hardware, backend, context_tokens=max_len, gpu_memory_utilization=util)
    out: dict[str, Any] = {"estimated_max_concurrency": e["max_concurrency"], "estimated_weights_bytes": e["breakdown"]["weights"],
                           "estimated_kv_bytes_per_token": e["kv_bytes_per_token"]}
    if "max_concurrency_at_len" in measured:
        out["measured_max_concurrency"] = measured["max_concurrency_at_len"]
        out["estimate_over_measured"] = e["max_concurrency"] / max(measured["max_concurrency_at_len"], 1e-9)
    if "weights_gib" in measured:
        out["weights_estimate_over_measured"] = e["breakdown"]["weights"] / (measured["weights_gib"] * 2**30)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["vllm"])
    ap.add_argument("model")
    ap.add_argument("--axis", choices=["concurrency", "context"], default="concurrency")
    ap.add_argument("--gpu", type=int, help="physical GPU index (CUDA_VISIBLE_DEVICES and nvidia-smi sampling)")
    ap.add_argument("--max-len", type=int, default=1024, help="context for the concurrency search")
    ap.add_argument("--concurrency", type=int, default=1, help="concurrency for the context search")
    ap.add_argument("--lo", type=int)
    ap.add_argument("--hi", type=int)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--util", type=float, default=0.3)
    ap.add_argument("--workdir", default="runs/boundary")
    a = ap.parse_args()

    from pathlib import Path

    from mqde.backend.vllm_server import VllmServer

    n = 0

    def make(c: int, length: int) -> ServerBackend:
        nonlocal n
        n += 1
        return VllmServer(a.model, Path(a.workdir) / f"probe-{n:02d}-c{c}-L{length}.log", gpu=a.gpu, max_model_len=length,
                          gpu_memory_utilization=a.util, max_num_seqs=max(c, 1), dtype="float16", enforce_eager=True)

    if a.axis == "concurrency":
        res = find_max_concurrency(make, a.max_len, lo=a.lo or 1, hi=a.hi or 64, gpu=a.gpu)
    else:
        res = find_max_context(make, a.concurrency, lo=a.lo or 256, hi=a.hi or 8192, step=a.step, gpu=a.gpu)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
