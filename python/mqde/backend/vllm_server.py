"""vLLM deployment adapter (T-0703 / S11): ``vllm serve`` as a child process on 127.0.0.1.

vLLM usually lives in its own environment (it pins torch / transformers), so it is launched as a subprocess by
binary path and spoken to over HTTP; nothing from vLLM is imported. The binary comes from ``MQDE_VLLM_BIN``, then
the managed environment ``scripts/setup_backend.py`` creates (``$MQDE_ENVS_DIR/vllm``), then ``vllm`` on PATH, then
``build/vllm/bin/vllm`` (``backend.envs.resolve``).

Failures that were seen on real runs are recognised from the log and reported with a fix hint; ``engine_info``
reads the KV-cache capacity vLLM prints at startup (used to compare against the estimator's max concurrency).
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
    ("Unsupported quantization config", "quantization config is not supported by this vLLM version "
                                        "(e.g. asymmetric GPTQ: try AWQ / compressed-tensors, or a newer vLLM)"),
    ("is less than desired GPU memory utilization", "not enough free GPU memory for the requested "
                                                   "gpu_memory_utilization (other processes use the GPU)"),
    ("No available memory for the cache blocks", "no GPU memory left for the KV cache "
                                                 "(raise gpu_memory_utilization or lower max_model_len)"),
    ("larger than the maximum number of tokens that can be stored in KV cache",
     "max_model_len exceeds the KV cache capacity (lower max_model_len or raise gpu_memory_utilization)"),
    ("CUDA out of memory", "GPU out of memory while loading or profiling (lower max_model_len / max_num_seqs "
                           "or gpu_memory_utilization)"),
    ("OutOfMemoryError", "GPU out of memory while loading or profiling (lower max_model_len / max_num_seqs "
                         "or gpu_memory_utilization)"),
    ("No CUDA GPUs are available", "no visible CUDA device"),
    ("Address already in use", "port already in use"),
    ("Invalid repository ID or local directory", "model path not found"),
    ("does not appear to have a file named", "checkpoint directory is missing files (config / weights / tokenizer)"),
    ("is not supported for quantization", "quantization method is not supported by this model architecture"),
    ("No such file or directory: 'ninja'", "vLLM compiles kernels at startup and could not find `ninja` "
                                           "(it lives next to the vllm binary: put that directory on PATH)"),
    ("Cannot find any model weights", "no weight files found in the checkpoint directory"),
]


def vllm_bin() -> Path:
    r = envs.resolve("vllm")
    if r is None:
        raise FileNotFoundError("vllm not found (run `scripts/setup_backend.py install vllm`, or set MQDE_VLLM_BIN to the "
                                "`vllm` binary of your own environment)")
    return r.path


def diagnose(log: str) -> str | None:
    return match_diagnosis(log, DIAGNOSES)


def launch_args(binary: str | Path, model: str | Path, port: int, *, max_model_len: int = 2048,
                gpu_memory_utilization: float = 0.5, max_num_seqs: int | None = None, dtype: str = "auto",
                enforce_eager: bool = False, served_name: str | None = None, extra: list[str] | None = None) -> list[str]:
    a = [str(binary), "serve", str(model), "--host", HOST, "--port", str(port), "--max-model-len", str(max_model_len),
         "--gpu-memory-utilization", f"{gpu_memory_utilization:g}", "--dtype", dtype]
    if max_num_seqs:
        a += ["--max-num-seqs", str(max_num_seqs)]
    if enforce_eager:
        a.append("--enforce-eager")
    if served_name:
        a += ["--served-model-name", served_name]
    return a + (extra or [])


class VllmServer(ServerBackend):
    """Context manager around one ``vllm serve`` process (startup takes tens of seconds to minutes)."""

    name = "vllm"
    diagnoses = DIAGNOSES

    def __init__(self, model: str | Path, log_path: str | Path, *, gpu: int | str | None = None, max_model_len: int = 2048,
                 gpu_memory_utilization: float = 0.5, max_num_seqs: int | None = None, dtype: str = "auto",
                 enforce_eager: bool = False, served_name: str | None = None, extra: list[str] | None = None,
                 port: int | None = None, binary: str | Path | None = None, startup_timeout: float = 900) -> None:
        self.model = Path(model)
        port = port or free_port()
        args = launch_args(binary or vllm_bin(), model, port, max_model_len=max_model_len,
                           gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs, dtype=dtype,
                           enforce_eager=enforce_eager, served_name=served_name, extra=extra)
        # vLLM's JIT kernel builds call `ninja`, which sits in the same bin/ as `vllm` in its own environment
        env = {"PYTHONUNBUFFERED": "1", "PATH": f"{Path(args[0]).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        super().__init__(args, log_path, port=port, env=env, startup_timeout=startup_timeout)

    def ready(self) -> bool:
        http_json(self.port, "GET", "/health", timeout=2)  # 200 with an empty body; anything else raises
        return True

    def props(self) -> dict[str, Any]:
        d = ((http_json(self.port, "GET", "/v1/models", timeout=30) or {}).get("data") or [{}])[0]
        return {"id": d.get("id"), "max_model_len": d.get("max_model_len")}

    def engine_info(self) -> dict[str, Any]:
        """Facts vLLM prints at startup: KV cache capacity, weight memory, quantization kernel."""
        log = self.log()
        out: dict[str, Any] = {}
        pats = {
            "kv_cache_tokens": (r"GPU KV cache size:\s*([\d,]+) tokens", lambda m: int(m.group(1).replace(",", ""))),
            "max_concurrency_at_len": (r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.,]+)x",
                                       lambda m: float(m.group(2).replace(",", ""))),
            "max_concurrency_len": (r"Maximum concurrency for ([\d,]+) tokens per request", lambda m: int(m.group(1).replace(",", ""))),
            "weights_gib": (r"Model loading took ([\d.]+) GiB", lambda m: float(m.group(1))),
            "kv_cache_gib": (r"Available KV cache memory:\s*([\d.]+) GiB", lambda m: float(m.group(1))),
            # "Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8" / "Using MarlinLinearKernel for AutoAWQ..."
            # (not "Using FlashInfer for top-p sampling", which is the sampler)
            "quant_kernel": (r"(?:Selected|Using) (\w+Kernel) for (\w+)", lambda m: m.group(1)),
            "quant_scheme": (r"(?:Selected|Using) (\w+Kernel) for (\w+)", lambda m: m.group(2)),
            "init_s": (r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s", lambda m: float(m.group(1))),
        }
        for k, (rx, conv) in pats.items():
            m = re.search(rx, log)
            if m:
                out[k] = conv(m)
        return out
