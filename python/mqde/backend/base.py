"""Deployment backend base (T-0703 / S11): one child process serving an OpenAI-compatible HTTP API.

A backend is a context manager: ``start()`` launches the server on 127.0.0.1, polls readiness, keeps the
combined stdout/stderr in a log file, and turns an early exit or a timeout into an error that carries a
diagnosis recognised from the log (out of memory, unsupported quantization, missing file, port in use ...).
``stop()`` terminates the process group. The server has no authentication, so it only ever listens on the
loopback interface.

Subclasses set ``args`` / ``env``, implement ``ready()``, and may extend ``diagnoses``.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

HOST = "127.0.0.1"


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


def http_json(port: int, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600,
              host: str = HOST) -> Any:
    """One JSON request; returns the decoded body (None for an empty one). Raises RuntimeError on non-200."""
    c = http.client.HTTPConnection(host, port, timeout=timeout)
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


def match_diagnosis(log: str, table: list[tuple[str, str]]) -> str | None:
    low = log.lower()
    for needle, why in table:
        if needle.lower() in low:
            return why
    return None


class ServerBackend:
    """Process lifecycle shared by every serving backend."""

    name = "server"
    diagnoses: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, args: list[str], log_path: str | Path, *, port: int | None = None,
                 env: dict[str, str] | None = None, startup_timeout: float = 300) -> None:
        self.args = args
        self.log_path = Path(log_path)
        self.port = port or free_port()
        self.startup_timeout = startup_timeout
        self._env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen[bytes] | None = None
        self.startup_s: float | None = None

    # ---------------------------------------------------------------- to override
    def ready(self) -> bool:
        raise NotImplementedError

    # ---------------------------------------------------------------- lifecycle
    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def log(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.is_file() else ""

    def diagnose(self, log: str | None = None) -> str | None:
        return match_diagnosis(self.log() if log is None else log, self.diagnoses)

    def start(self) -> ServerBackend:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with self.log_path.open("wb") as f:
            # own session so stop() can take down workers the server spawned (vLLM's EngineCore)
            self.proc = subprocess.Popen(self.args, stdout=f, stderr=subprocess.STDOUT, env=self._env,
                                         start_new_session=True)
        deadline = t0 + self.startup_timeout
        while time.perf_counter() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                log = self.log()
                raise RuntimeError(f"{self.name} exited with {rc}: {self.diagnose(log) or 'unknown error'}\n{log[-1500:]}")
            try:
                if self.ready():
                    self.startup_s = time.perf_counter() - t0
                    return self
            except (OSError, RuntimeError, ValueError):
                pass  # still loading (503) or not listening yet
            time.sleep(0.25)
        why = self.diagnose() or "no diagnosis"
        self.stop()
        raise TimeoutError(f"{self.name} not healthy after {self.startup_timeout:.0f}s: {why}")

    def stop(self, grace: float = 20) -> None:
        p = self.proc
        self.proc = None
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()

    def __enter__(self) -> ServerBackend:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def openai_models(port: int) -> list[dict[str, Any]]:
    return list((http_json(port, "GET", "/v1/models", timeout=30) or {}).get("data", []))


def openai_smoke(port: int, prompt: str = "What is the capital of France? Answer in one word.", max_tokens: int = 16,
                 model: str | None = None) -> dict[str, Any]:
    """One non-streaming chat completion; ``ok`` means the reply has non-blank text."""
    if model is None:
        try:
            models = openai_models(port)
            model = models[0]["id"] if models else None
        except (OSError, RuntimeError, ValueError):
            model = None  # servers without /v1/models (or one model only) accept the request without a name
    body: dict[str, Any] = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    if model:
        body["model"] = model
    t0 = time.perf_counter()
    r = http_json(port, "POST", "/v1/chat/completions", body)
    text = r["choices"][0]["message"]["content"] or ""
    return {"prompt": prompt, "text": text, "ok": bool(text.strip()), "latency_s": time.perf_counter() - t0,
            "usage": r.get("usage")}
