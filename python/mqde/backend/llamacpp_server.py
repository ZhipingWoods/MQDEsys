"""llama-server deployment adapter (T-0705 / S11 for GGUF).

``LlamaServer`` starts ``llama-server`` as a child process bound to 127.0.0.1, waits for ``/health``, keeps the
log in a file, recognises common failures (missing file, unknown architecture, allocation failure) and shuts the
process down. ``smoke`` sends one chat request; ``serve_bench`` measures streaming TTFT / TPOT / throughput at a
fixed concurrency with fixed-length token prompts (a light precursor of the S13 benchmarker, T-0803).

The server has no authentication, so it only ever listens on the loopback interface.

The binary comes from ``MQDE_LLAMA_SERVER_BIN`` or ``build/llamacpp-server/bin/llama-server`` (upstream
llama.cpp; the omni fork on the dev machine removed the llama-server target).
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
HOST = "127.0.0.1"

# log substring -> diagnosis shown in the report / UI
DIAGNOSES: list[tuple[str, str]] = [
    ("failed to open", "model file not found or unreadable"),
    ("unknown model architecture", "architecture not supported by this llama.cpp build"),
    ("failed to allocate", "out of memory while allocating the model or KV cache (lower ctx / parallel)"),
    ("unable to allocate", "out of memory while allocating the model or KV cache (lower ctx / parallel)"),
    ("out of memory", "out of memory while allocating the model or KV cache (lower ctx / parallel)"),
    ("invalid magic", "file is not a GGUF (or is truncated)"),
    ("tensor data is not within", "GGUF file is truncated or corrupt"),
    ("couldn't bind", "port already in use"),
    ("failed to load model", "model failed to load (see log)"),
]


def server_bin() -> Path:
    for c in [os.environ.get("MQDE_LLAMA_SERVER_BIN", ""), str(REPO / "build/llamacpp-server/bin/llama-server")]:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        "llama-server not found (build upstream llama.cpp into build/llamacpp-server or set MQDE_LLAMA_SERVER_BIN)"
    )


def diagnose(log: str) -> str | None:
    low = log.lower()
    for needle, why in DIAGNOSES:
        if needle in low:
            return why
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


def _request(port: int, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    c = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status}: {data[:300]!r}")
        return json.loads(data) if data else None
    finally:
        c.close()


class LlamaServer:
    """Context manager around one llama-server process."""

    def __init__(self, model: str | Path, log_path: str | Path, *, threads: int = 16, ctx: int = 4096,
                 parallel: int = 4, port: int | None = None, extra: list[str] | None = None,
                 binary: str | Path | None = None, startup_timeout: float = 300) -> None:
        self.model = Path(model)
        self.log_path = Path(log_path)
        self.port = port or free_port()
        self.startup_timeout = startup_timeout
        b = Path(binary) if binary else server_bin()
        self.args = [str(b), "-m", str(self.model), "--host", HOST, "--port", str(self.port), "-t", str(threads),
                     "-c", str(ctx), "-np", str(parallel), "--metrics", "--no-webui", *(extra or [])]
        self._env = {**os.environ, "LD_LIBRARY_PATH": f"{b.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        self.proc: subprocess.Popen[bytes] | None = None
        self.startup_s: float | None = None

    def log(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.is_file() else ""

    def start(self) -> LlamaServer:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with self.log_path.open("wb") as f:
            self.proc = subprocess.Popen(self.args, stdout=f, stderr=subprocess.STDOUT, env=self._env)
        deadline = t0 + self.startup_timeout
        while time.perf_counter() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                log = self.log()
                raise RuntimeError(f"llama-server exited with {rc}: {diagnose(log) or 'unknown error'}\n{log[-1500:]}")
            try:
                if (_request(self.port, "GET", "/health", timeout=2) or {}).get("status") == "ok":
                    self.startup_s = time.perf_counter() - t0
                    return self
            except (OSError, RuntimeError, ValueError):
                pass  # still loading (503) or not listening yet
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"llama-server not healthy after {self.startup_timeout:.0f}s: {diagnose(self.log()) or 'no diagnosis'}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def __enter__(self) -> LlamaServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def props(self) -> dict[str, Any]:
        p = _request(self.port, "GET", "/props")
        return {"total_slots": p.get("total_slots"), "build_info": p.get("build_info"),
                "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx")}


def smoke(port: int, prompt: str = "What is the capital of France? Answer in one word.", max_tokens: int = 16) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = _request(port, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0})
    text = r["choices"][0]["message"]["content"] or ""
    return {"prompt": prompt, "text": text, "ok": bool(text.strip()), "latency_s": time.perf_counter() - t0,
            "usage": r.get("usage")}


def _stream_one(port: int, prompt_ids: list[int], n_predict: int, timeout: float) -> dict[str, float]:
    """One streaming /completion request; returns ttft and per-token arrival times (seconds from send)."""
    body = {"prompt": prompt_ids, "n_predict": n_predict, "stream": True, "ignore_eos": True,
            "cache_prompt": False, "temperature": 0}
    c = http.client.HTTPConnection(HOST, port, timeout=timeout)
    t0 = time.perf_counter()
    try:
        c.request("POST", "/completion", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"/completion -> HTTP {r.status}: {r.read()[:300]!r}")
        arrivals: list[float] = []
        while True:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            now = time.perf_counter() - t0
            # each non-final event carries the token ids it streams; older builds omit them (count one)
            n_tok = len(ev["tokens"]) if ev.get("tokens") else (0 if ev.get("stop") else 1)
            arrivals.extend([now] * n_tok)
            if ev.get("stop"):
                break
    finally:
        c.close()
    if not arrivals:
        raise RuntimeError("stream produced no tokens")
    n = len(arrivals)
    return {"ttft": arrivals[0], "e2e": arrivals[-1], "tokens": n,
            "tpot": (arrivals[-1] - arrivals[0]) / (n - 1) if n > 1 else 0.0}


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))]


def serve_bench(port: int, *, concurrency: int = 4, requests: int = 16, prompt_len: int = 256, n_predict: int = 64,
                token_id: int = 1000, timeout: float = 600) -> dict[str, Any]:
    """Closed-loop load: ``concurrency`` workers send ``requests`` streaming requests in total.

    Prompts are ``prompt_len`` copies of one token id (plus the request index, so the prompt cache cannot reuse
    them); generation ignores EOS so every request produces exactly ``n_predict`` tokens."""
    results: list[dict[str, float]] = []
    errors: list[str] = []
    lock = threading.Lock()
    counter = iter(range(requests))

    def worker() -> None:
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            ids = [token_id + (i % 50)] + [token_id] * (prompt_len - 1)
            try:
                r = _stream_one(port, ids, n_predict, timeout)
                with lock:
                    results.append(r)
            except Exception as e:  # recorded, the run continues
                with lock:
                    errors.append(f"{type(e).__name__}: {e}")

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    out: dict[str, Any] = {"concurrency": concurrency, "requests": requests, "prompt_len": prompt_len,
                           "n_predict": n_predict, "ok": len(results), "errors": errors[:5], "wall_s": wall}
    if results:
        for k in ("ttft", "tpot", "e2e"):
            xs = [r[k] for r in results]
            out[f"{k}_ms"] = {"mean": statistics.fmean(xs) * 1e3, "p50": _pct(xs, 50) * 1e3,
                              "p90": _pct(xs, 90) * 1e3, "p99": _pct(xs, 99) * 1e3}
        out["output_tokens_per_s"] = sum(r["tokens"] for r in results) / wall
        out["requests_per_s"] = len(results) / wall
    return out
