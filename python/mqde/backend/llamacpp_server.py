"""llama-server deployment adapter (T-0705 / S11 for GGUF).

``LlamaServer`` starts ``llama-server`` on 127.0.0.1 and shares the process lifecycle, readiness polling and
failure diagnosis with the other backends (``backend.base``). ``smoke`` sends one chat request; ``serve_bench``
is a thin wrapper over ``eval.loadtest`` (llama-server's native streaming protocol).

The binary comes from ``MQDE_LLAMA_SERVER_BIN``, then the managed release ``scripts/setup_backend.py`` installs
(``$MQDE_ENVS_DIR/llamacpp/current``), then ``build/llamacpp-server/bin/llama-server`` (upstream llama.cpp; the omni
fork on the dev machine removed the llama-server target) -- ``backend.envs.resolve``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mqde.backend import envs
from mqde.backend.base import HOST, ServerBackend, free_port, http_json, match_diagnosis, openai_smoke
from mqde.eval.loadtest import LoadSpec, run_load

__all__ = ["DIAGNOSES", "HOST", "LlamaServer", "diagnose", "free_port", "serve_bench", "server_bin", "smoke"]

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
    r = envs.resolve("llamacpp")
    if r is None:
        raise FileNotFoundError("llama-server not found (run `scripts/setup_backend.py install llamacpp`, or set "
                                "MQDE_LLAMA_SERVER_BIN to your own build)")
    return r.path


def diagnose(log: str) -> str | None:
    return match_diagnosis(log, DIAGNOSES)


_request = http_json  # kept for callers that talk to the server directly (tests/integration)


class LlamaServer(ServerBackend):
    """Context manager around one llama-server process."""

    name = "llama-server"
    diagnoses = DIAGNOSES

    def __init__(self, model: str | Path, log_path: str | Path, *, threads: int = 16, ctx: int = 4096,
                 parallel: int = 4, port: int | None = None, extra: list[str] | None = None,
                 binary: str | Path | None = None, startup_timeout: float = 300,
                 gpu_layers: int | None = None) -> None:
        self.model = Path(model)
        b = Path(binary) if binary else server_bin()
        port = port or free_port()
        args = [str(b), "-m", str(self.model), "--host", HOST, "--port", str(port), "-t", str(threads),
                "-c", str(ctx), "-np", str(parallel), "--metrics", "--no-webui",
                *([] if gpu_layers is None else ["-ngl", str(gpu_layers)]), *(extra or [])]
        env = {"LD_LIBRARY_PATH": f"{b.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}",
               "DYLD_LIBRARY_PATH": f"{b.parent}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"}
        super().__init__(args, log_path, port=port, env=env, startup_timeout=startup_timeout)

    def ready(self) -> bool:
        return (http_json(self.port, "GET", "/health", timeout=2) or {}).get("status") == "ok"

    def props(self) -> dict[str, Any]:
        p = http_json(self.port, "GET", "/props")
        return {"total_slots": p.get("total_slots"), "build_info": p.get("build_info"),
                "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx")}


def smoke(port: int, prompt: str = "What is the capital of France? Answer in one word.", max_tokens: int = 16) -> dict[str, Any]:
    return openai_smoke(port, prompt, max_tokens)


def serve_bench(port: int, *, concurrency: int = 4, requests: int = 16, prompt_len: int = 256, n_predict: int = 64,
                token_id: int = 1000, timeout: float = 600) -> dict[str, Any]:
    """Closed-loop streaming load on llama-server's native protocol (see ``eval.loadtest`` for definitions).

    Returns the loadtest summary plus the flat fields earlier reports used (concurrency, requests, ...)."""
    spec = LoadSpec(api="llama", concurrency=concurrency, requests=requests, prompt_len=prompt_len,
                    n_predict=n_predict, token_id=token_id, timeout=timeout, warmup=0)
    out = run_load(port, spec)
    return {"concurrency": concurrency, "requests": requests, "prompt_len": prompt_len, "n_predict": n_predict, **out}
