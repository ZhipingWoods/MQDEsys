"""SGLang deployment adapter (T-0704 / S11): ``python -m sglang.launch_server`` as a child process on 127.0.0.1.

SGLang pins its own torch / transformers / flashinfer, so (like vLLM) it lives in its own environment and is
launched by interpreter path and spoken to over HTTP; nothing from SGLang is imported. The interpreter comes from
``MQDE_SGLANG_PYTHON``, then the managed environment ``scripts/setup_backend.py`` creates (``$MQDE_ENVS_DIR/sglang``),
then ``build/sglang/bin/python`` (``backend.envs.resolve``); a missing environment is reported, not guessed.

Failures seen on real runs are recognised from the log; ``engine_info`` reads what SGLang prints at startup
(weight memory, KV-cache capacity) so it can be compared with the estimator's numbers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from mqde.backend import envs
from mqde.backend.base import HOST, ServerBackend, free_port, http_json, match_diagnosis

# log substring -> diagnosis (first match wins, most specific first)
DIAGNOSES: list[tuple[str, str]] = [
    ("Not enough memory. Please try to increase --mem-fraction-static",
     "not enough GPU memory left for the KV cache (raise mem_fraction_static or free the GPU)"),
    ("CUDA out of memory", "GPU out of memory while loading or capturing CUDA graphs (lower mem_fraction_static / "
                           "context_length / max_running_requests, or disable CUDA graphs)"),
    ("OutOfMemoryError", "GPU out of memory while loading or capturing CUDA graphs (lower mem_fraction_static / "
                         "context_length / max_running_requests, or disable CUDA graphs)"),
    ("Capture cuda graph failed", "CUDA graph capture failed (set disable_cuda_graph to run without it)"),
    ("Unknown quantization method", "quantization method is not known to this SGLang version"),
    ("is not supported for quantization", "dtype is not supported for this quantization method (try dtype float16)"),
    ("No CUDA GPUs are available", "no visible CUDA device"),
    ("no cuda gpus are available", "no visible CUDA device"),
    ("Address already in use", "port already in use"),
    ("Can't load the configuration", "model path not found or not a Hugging Face checkpoint directory"),
    ("does not appear to have a file named", "checkpoint directory is missing files (config / weights / tokenizer)"),
    ("Cannot find any model weights", "no weight files found in the checkpoint directory"),
    ("No module named 'sglang'", "the interpreter has no sglang installed (pip install -e 3rdpart/sglang/python)"),
]


def sglang_python() -> Path:
    r = envs.resolve("sglang")
    if r is None:
        raise FileNotFoundError("sglang not found (run `scripts/setup_backend.py install sglang`, or set MQDE_SGLANG_PYTHON "
                                "to the python of your own SGLang environment)")
    return r.path


def diagnose(log: str) -> str | None:
    return match_diagnosis(log, DIAGNOSES)


def launch_args(python: str | Path, model: str | Path, port: int, *, max_model_len: int = 2048,
                mem_fraction_static: float = 0.5, max_running_requests: int | None = None, dtype: str = "auto",
                disable_cuda_graph: bool = False, served_name: str | None = None,
                extra: list[str] | None = None) -> list[str]:
    a = [str(python), "-m", "sglang.launch_server", "--model-path", str(model), "--host", HOST, "--port", str(port),
         "--context-length", str(max_model_len), "--mem-fraction-static", f"{mem_fraction_static:g}", "--dtype", dtype,
         "--skip-server-warmup"]  # warmup would send an extra generate request before /health; we smoke it ourselves
    if max_running_requests:
        a += ["--max-running-requests", str(max_running_requests)]
    if disable_cuda_graph:  # `--disable-cuda-graph` is deprecated upstream in favour of these two
        a += ["--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled"]
    if served_name:
        a += ["--served-model-name", served_name]
    return a + (extra or [])


class SglangServer(ServerBackend):
    """Context manager around one SGLang server process (startup takes tens of seconds to minutes)."""

    name = "sglang"
    diagnoses = DIAGNOSES

    def __init__(self, model: str | Path, log_path: str | Path, *, gpu: int | str | None = None, max_model_len: int = 2048,
                 mem_fraction_static: float = 0.5, max_running_requests: int | None = None, dtype: str = "auto",
                 disable_cuda_graph: bool = False, served_name: str | None = None, extra: list[str] | None = None,
                 port: int | None = None, python: str | Path | None = None, startup_timeout: float = 900) -> None:
        self.model = Path(model)
        port = port or free_port()
        args = launch_args(python or sglang_python(), model, port, max_model_len=max_model_len,
                           mem_fraction_static=mem_fraction_static, max_running_requests=max_running_requests,
                           dtype=dtype, disable_cuda_graph=disable_cuda_graph, served_name=served_name, extra=extra)
        # JIT kernel builds (flashinfer, tvm-ffi) call `ninja` / `nvcc`; ninja sits next to the env's python
        env = {"PYTHONUNBUFFERED": "1", "PATH": f"{Path(args[0]).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        super().__init__(args, log_path, port=port, env=env, startup_timeout=startup_timeout)

    def ready(self) -> bool:
        http_json(self.port, "GET", "/health", timeout=2)  # 200 once the scheduler answers; anything else raises
        return True

    def props(self) -> dict[str, Any]:
        d = ((http_json(self.port, "GET", "/v1/models", timeout=30) or {}).get("data") or [{}])[0]
        info = http_json(self.port, "GET", "/model_info", timeout=30) or {}
        return {"id": d.get("id"), "max_model_len": d.get("max_model_len") or info.get("max_model_len")}

    def engine_info(self) -> dict[str, Any]:
        """Facts SGLang prints at startup: weight memory, KV cache capacity."""
        log = self.log()
        out: dict[str, Any] = {}
        pats = {
            "weights_gib": (r"Load weight end\..*?mem usage=([\d.]+) GB", lambda m: float(m.group(1))),
            "weight_load_s": (r"Load weight end\. elapsed=([\d.]+) s", lambda m: float(m.group(1))),
            "kv_cache_tokens": (r"KV Cache is allocated\..*?#tokens: (\d+)", lambda m: int(m.group(1))),
            "avail_after_pool_gib": (r"Memory pool end\. avail mem=([\d.]+) GB", lambda m: float(m.group(1))),
            "cuda_graph_prefill_s": (r"Capture target prefill CUDA graph end\. elapsed=([\d.]+) s", lambda m: float(m.group(1))),
            "cuda_graph_decode_s": (r"Capture target decode CUDA graph end\. elapsed=([\d.]+) s", lambda m: float(m.group(1))),
        }
        for k, (rx, conv) in pats.items():
            m = re.search(rx, log)
            if m:
                out[k] = conv(m)
        # K and V are separate pools (MHA/GQA: "K size: a GB, V size: b GB"); MLA-style caches print one "KV size: c GB".
        # kv_cache_gib is the total either way.
        m = re.search(r"KV Cache is allocated\..*?K size: ([\d.]+) GB, V size: ([\d.]+) GB", log)
        if m:
            out["kv_cache_gib"] = float(m.group(1)) + float(m.group(2))
        elif m := re.search(r"KV Cache is allocated\..*?KV size: ([\d.]+) GB", log):
            out["kv_cache_gib"] = float(m.group(1))
        return out

def _register() -> None:
    from mqde.backend.registry import BackendEntry, register_backend
    register_backend(BackendEntry(
        id="sglang",
        server_class="SglangServer",
        spec_field="sglang",
        export_formats=["awq", "gptq", "compressed_tensors"],
        needs_cuda=True,
        bin_check=lambda: sglang_python() and None,
        module="mqde.backend.sglang_server",
    ))

_register()
