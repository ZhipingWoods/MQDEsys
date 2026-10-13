"""Serving load tester (T-0803 / S13): streaming TTFT / TPOT / ITL / throughput against a running server.

Two arrival modes:

* closed loop: ``concurrency`` workers send ``requests`` requests back to back;
* open loop: requests arrive as a Poisson process at ``rate`` req/s (seeded), at most ``max_inflight`` at once.
  ``schedule_lag_ms`` reports how late requests were started versus their schedule; a large p99 means the
  generator (or ``max_inflight``) was the bottleneck, not the server.

Inputs are token-id lists of exactly ``prompt_len`` tokens (no tokenizer needed). The first tokens differ per
request so prefix caching cannot serve one request from another. Generation ignores EOS and asks for exactly
``n_predict`` tokens.

Two wire protocols: ``openai`` (``POST /v1/completions``, SSE, vLLM / SGLang / llama-server) and ``llama``
(llama-server's native ``POST /completion``). Every request is timed with ``perf_counter`` on the client.

Definitions (per successful request, token arrival = SSE event carrying text):
    TTFT = first token arrival - request sent
    TPOT = (last arrival - first arrival) / (n_tokens - 1)
    ITL  = gaps between consecutive token arrivals (pooled over requests)
    E2E  = last arrival - request sent
Throughput = output tokens of successful requests / wall time of the measured phase.
"""

from __future__ import annotations

import http.client
import itertools
import json
import random
import socket
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

HOST = "127.0.0.1"


@dataclass
class LoadSpec:
    api: str = "openai"  # openai | llama
    model: str | None = None  # openai: served model name (auto-detected from /v1/models when None)
    mode: str = "closed"  # closed | open
    concurrency: int = 4  # closed loop workers
    requests: int = 16  # measured requests (both modes)
    rate: float = 0.0  # open loop arrival rate, req/s
    max_inflight: int = 256  # open loop cap on simultaneous requests
    prompt_len: int = 256
    n_predict: int = 64
    warmup: int = 2  # sequential requests before measuring, not counted
    token_id: int = 1000
    seed: int = 0
    timeout: float = 600.0
    host: str = HOST


@dataclass
class Sample:
    ok: bool
    error: str = ""
    ttft: float = 0.0
    e2e: float = 0.0
    tokens: int = 0
    tpot: float = 0.0
    itl: tuple[float, ...] = ()
    lag: float = 0.0  # actual start - scheduled start (open loop)


def make_prompt(i: int, length: int, token_id: int) -> list[int]:
    """``length`` token ids; the first 4 encode the request index so no two requests share a prefix."""
    head = [token_id + (i // 50**k) % 50 for k in range(4)]
    return (head + [token_id] * length)[:length]


def _request_body(spec: LoadSpec, ids: list[int]) -> tuple[str, dict[str, Any]]:
    if spec.api == "llama":
        return "/completion", {"prompt": ids, "n_predict": spec.n_predict, "stream": True, "ignore_eos": True,
                               "cache_prompt": False, "temperature": 0}
    body: dict[str, Any] = {"prompt": ids, "max_tokens": spec.n_predict, "min_tokens": spec.n_predict,
                            "temperature": 0, "stream": True, "ignore_eos": True,
                            "stream_options": {"include_usage": True}}
    if spec.model:
        body["model"] = spec.model
    return "/v1/completions", body


def _events(resp: http.client.HTTPResponse) -> Any:
    """Yield (arrival perf_counter, decoded JSON event) for each ``data:`` line; stops at [DONE]."""
    while True:
        line = resp.readline()
        if not line:
            return
        if not line.startswith(b"data:"):
            continue
        t = time.perf_counter()
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return
        yield t, json.loads(payload)


def one_request(spec: LoadSpec, port: int, i: int) -> Sample:
    path, body = _request_body(spec, make_prompt(i, spec.prompt_len, spec.token_id))
    c = http.client.HTTPConnection(spec.host, port, timeout=spec.timeout)
    arrivals: list[float] = []
    usage_tokens: int | None = None
    try:
        c.connect()
        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t0 = time.perf_counter()
        c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            return Sample(False, f"HTTP {r.status}: {r.read()[:200]!r}")
        for t, ev in _events(r):
            if spec.api == "llama":
                n = len(ev["tokens"]) if ev.get("tokens") else (0 if ev.get("stop") else 1)
                arrivals.extend([t] * n)
                if ev.get("stop"):
                    break
                continue
            if ev.get("usage"):
                usage_tokens = int(ev["usage"].get("completion_tokens", 0)) or usage_tokens
            if "error" in ev:
                return Sample(False, f"server error: {str(ev['error'])[:200]}")
            if any(ch.get("text") for ch in ev.get("choices") or []):
                arrivals.append(t)
    except Exception as e:
        return Sample(False, f"{type(e).__name__}: {e}")
    finally:
        c.close()
    if not arrivals:
        return Sample(False, "stream produced no tokens")
    n = usage_tokens if usage_tokens else len(arrivals)
    ttft = arrivals[0] - t0
    gaps = tuple(b - a for a, b in itertools.pairwise(arrivals))
    return Sample(True, ttft=ttft, e2e=arrivals[-1] - t0, tokens=n,
                  tpot=(arrivals[-1] - arrivals[0]) / (n - 1) if n > 1 else 0.0, itl=gaps)


def poisson_schedule(rate: float, n: int, seed: int = 0) -> list[float]:
    """Arrival offsets (seconds from the start) of ``n`` requests of a Poisson process with ``rate`` req/s."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(xs: list[float]) -> dict[str, float]:
    ms = [x * 1e3 for x in xs]
    return {"mean": statistics.fmean(ms), "p50": _pct(ms, 50), "p90": _pct(ms, 90), "p99": _pct(ms, 99), "max": max(ms)}


def _detect_model(spec: LoadSpec, port: int) -> str | None:
    if spec.api != "openai" or spec.model:
        return spec.model
    c = http.client.HTTPConnection(spec.host, port, timeout=30)
    try:
        c.request("GET", "/v1/models")
        r = c.getresponse()
        data = json.loads(r.read()).get("data", []) if r.status == 200 else []
        return str(data[0]["id"]) if data else None
    except (OSError, ValueError, http.client.HTTPException):
        return None  # unreachable / not JSON: the requests themselves will fail and be counted, not crash the run
    finally:
        c.close()


def run_load(port: int, spec: LoadSpec | None = None, *, on_sample: Callable[[Sample], None] | None = None,
             **kw: Any) -> dict[str, Any]:
    """Run warmup, then the measured phase; returns a JSON-serialisable summary."""
    spec = spec or LoadSpec(**kw)
    spec.model = _detect_model(spec, port)
    for w in range(spec.warmup):
        one_request(spec, port, 10_000_000 + w)
    samples: list[Sample] = []
    lock = threading.Lock()

    def record(s: Sample) -> None:
        with lock:
            samples.append(s)
        if on_sample:
            on_sample(s)

    cpu0 = time.process_time()
    t_start = time.perf_counter()
    if spec.mode == "open":
        if spec.rate <= 0:
            raise ValueError("open loop needs rate > 0")
        due = poisson_schedule(spec.rate, spec.requests, spec.seed)

        def fire(i: int, at: float) -> None:
            lag = time.perf_counter() - (t_start + at)
            s = one_request(spec, port, i)
            s.lag = max(0.0, lag)
            record(s)

        with ThreadPoolExecutor(max_workers=max(1, min(spec.max_inflight, spec.requests))) as pool:
            for i, at in enumerate(due):
                wait = t_start + at - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                pool.submit(fire, i, at)
    else:
        counter = iter(range(spec.requests))

        def worker() -> None:
            while True:
                with lock:
                    i = next(counter, None)
                if i is None:
                    return
                record(one_request(spec, port, i))

        threads = [threading.Thread(target=worker) for _ in range(max(1, spec.concurrency))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    wall = time.perf_counter() - t_start
    cpu = time.process_time() - cpu0
    good = [s for s in samples if s.ok]
    out: dict[str, Any] = {
        "spec": asdict(spec), "ok": len(good), "failed": len(samples) - len(good),
        "errors": sorted({s.error for s in samples if not s.ok})[:5], "wall_s": wall,
        "client_cpu_s": cpu, "client_cpu_frac": cpu / wall if wall > 0 else 0.0,
    }
    if good:
        out["ttft_ms"] = summarize([s.ttft for s in good])
        out["tpot_ms"] = summarize([s.tpot for s in good if s.tokens > 1] or [0.0])
        out["e2e_ms"] = summarize([s.e2e for s in good])
        itl = [g for s in good for g in s.itl]
        if itl:
            out["itl_ms"] = summarize(itl)
        out["output_tokens"] = sum(s.tokens for s in good)
        out["output_tokens_per_s"] = out["output_tokens"] / wall
        out["input_tokens_per_s"] = len(good) * spec.prompt_len / wall
        out["requests_per_s"] = len(good) / wall
        if spec.mode == "open":
            out["offered_rate"] = spec.rate
            out["schedule_lag_ms"] = summarize([s.lag for s in samples])
    return out


# ---- concurrency sweep (T-R12) ----
def sweep(port: int, levels: list[int], *, requests_per_level: int = 16, warmup: int = 1, **kw: Any) -> dict[str, Any]:
    """The same closed-loop load at several concurrency levels (ascending), each with its own TTFT / TPOT / ITL
    percentiles and throughput, plus the peak and where adding load stops paying (``saturation``). A level sends at least
    two requests per worker, so every worker runs more than once and the percentiles are not dominated by the ramp-up."""
    if not levels or any(c < 1 for c in levels) or len(set(levels)) != len(levels):
        raise ValueError(f"levels must be distinct positive integers, got {levels}")
    points: list[dict[str, Any]] = []
    for c in sorted(levels):
        r = run_load(port, LoadSpec(concurrency=c, requests=max(requests_per_level, 2 * c), warmup=warmup, **kw))
        points.append({"concurrency": c, **r})
    good = [p for p in points if p.get("output_tokens_per_s")]
    return {"points": points, "peak": max(good, key=lambda p: p["output_tokens_per_s"]) if good else None,
            "saturation": saturation(points)}


def saturation(points: list[dict[str, Any]], *, min_gain_pct: float = 10.0, max_ttft_p99_ms: float | None = None) -> dict[str, Any] | None:
    """The last concurrency worth running: the first level after which the next one gains < ``min_gain_pct`` throughput
    (reason ``throughput``) or whose TTFT p99 exceeds ``max_ttft_p99_ms`` (``ttft``). ``not-reached`` = still scaling at the
    top of the range. Levels with failed requests say nothing about capacity and are left out."""
    ok = sorted((p for p in points if not p.get("failed") and p.get("output_tokens_per_s")), key=lambda p: p["concurrency"])
    if not ok:
        return None
    if len(ok) == 1:
        return {"concurrency": ok[0]["concurrency"], "reason": "single-level", "next_gain_pct": None}
    for a, b in itertools.pairwise(ok):
        gain = 100.0 * (b["output_tokens_per_s"] / a["output_tokens_per_s"] - 1.0)
        if max_ttft_p99_ms is not None and (b.get("ttft_ms") or {}).get("p99", 0.0) > max_ttft_p99_ms:
            return {"concurrency": a["concurrency"], "reason": "ttft", "next_gain_pct": gain}
        if gain < min_gain_pct:
            return {"concurrency": a["concurrency"], "reason": "throughput", "next_gain_pct": gain}
    return {"concurrency": ok[-1]["concurrency"], "reason": "not-reached", "next_gain_pct": None}
