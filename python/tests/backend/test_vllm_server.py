"""vLLM adapter (T-0703): launch arguments, lifecycle against the mock, failure diagnosis, optional real run."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mock_backend

from mqde.backend import vllm_server as vs
from mqde.backend.base import openai_smoke
from mqde.eval.loadtest import LoadSpec, run_load


def test_launch_args() -> None:
    a = vs.launch_args("/x/vllm", "/m/ckpt", 8123, max_model_len=1024, gpu_memory_utilization=0.4, max_num_seqs=8,
                       dtype="float16", enforce_eager=True, served_name="q", extra=["--seed", "1"])
    assert a[:3] == ["/x/vllm", "serve", "/m/ckpt"]
    flags = dict(itertools.pairwise(a[3:]))
    assert flags["--host"] == "127.0.0.1" and flags["--port"] == "8123" and flags["--max-model-len"] == "1024"
    assert flags["--gpu-memory-utilization"] == "0.4" and flags["--max-num-seqs"] == "8" and flags["--dtype"] == "float16"
    assert "--enforce-eager" in a and flags["--served-model-name"] == "q" and a[-2:] == ["--seed", "1"]
    assert "--max-num-seqs" not in vs.launch_args("v", "m", 1)  # unset options are left to vLLM's defaults


def test_gpu_is_pinned_through_the_environment(tmp_path: Path) -> None:
    s = mock_backend.vllm(tmp_path, gpu=3)
    assert s._env["CUDA_VISIBLE_DEVICES"] == "3"


def test_lifecycle_info_smoke_and_load(tmp_path: Path) -> None:
    srv = mock_backend.vllm(tmp_path, max_model_len=512, served_name="qwen-test", ttft=0.05, tpot=0.01)
    with srv:
        assert srv.startup_s is not None and srv.startup_s >= 0.3
        assert srv.props() == {"id": "qwen-test", "max_model_len": 512}
        assert openai_smoke(srv.port)["ok"]
        r = run_load(srv.port, LoadSpec(requests=4, concurrency=2, prompt_len=32, n_predict=8, warmup=1))
        info = srv.engine_info()
    assert srv.proc is None
    assert r["ok"] == 4 and r["failed"] == 0
    assert info["kv_cache_tokens"] == 12345 and info["weights_gib"] == 0.43
    assert info["max_concurrency_len"] == 512 and info["max_concurrency_at_len"] == pytest.approx(12345 / 512, abs=0.01)


@pytest.mark.parametrize(("line", "kernel", "scheme"), [
    ("INFO [__init__.py:758] Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8",
     "CutlassFP8ScaledMMLinearKernel", "CompressedTensorsW8A8Fp8"),
    ("INFO [auto_awq.py:447] Using MarlinLinearKernel for AutoAWQMarlinLinearMethod",
     "MarlinLinearKernel", "AutoAWQMarlinLinearMethod"),
])
def test_engine_info_reads_the_quant_kernel_from_real_log_lines(tmp_path: Path, line: str, kernel: str, scheme: str) -> None:
    srv = mock_backend.vllm(tmp_path)
    srv.log_path.write_text("INFO [topk_topp_sampler.py:78] Using FlashInfer for top-p & top-k sampling.\n" + line + "\n"
                            "INFO [kv_cache_utils.py:2395] GPU KV cache size: 521,264 tokens\n"
                            "INFO [kv_cache_utils.py:2395] Maximum concurrency for 1,024 tokens per request: 509.05x\n"
                            "INFO [gpu_worker.py:640] Available KV cache memory: 5.97 GiB\n"
                            "INFO [model_runner.py:428] Model loading took 0.59 GiB memory and 2.1 seconds\n")
    info = srv.engine_info()
    assert info["quant_kernel"] == kernel and info["quant_scheme"] == scheme  # not the sampler line
    assert info["kv_cache_tokens"] == 521264 and info["max_concurrency_at_len"] == 509.05
    assert info["max_concurrency_len"] == 1024 and info["kv_cache_gib"] == 5.97 and info["weights_gib"] == 0.59


@pytest.mark.parametrize(("model", "why"), [
    ("oom-model", "out of memory"),
    ("badquant-model", "quantization config is not supported"),
])
def test_startup_failures_are_diagnosed(tmp_path: Path, model: str, why: str) -> None:
    srv = mock_backend.vllm(tmp_path, model_name=model, startup_timeout=20)
    with pytest.raises(RuntimeError, match=why):
        srv.start()


def test_missing_model_path(tmp_path: Path) -> None:
    srv = mock_backend.vllm(tmp_path, startup_timeout=20)
    srv.args[2] = str(tmp_path / "does-not-exist")
    with pytest.raises(RuntimeError, match="model path not found"):
        srv.start()


@pytest.mark.parametrize(("log", "why"), [
    ("ValidationError: Value error, Unsupported quantization config: bits=4, sym=False", "not supported by this vLLM"),
    ("ValueError: Free memory on device (17.98/23.52 GiB) on startup is less than desired GPU memory utilization (0.9, 21.2 GiB)",
     "not enough free GPU memory"),
    ("ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`", "KV cache"),
    ("ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache",
     "max_model_len exceeds"),
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "out of memory"),
    ("RuntimeError: No CUDA GPUs are available", "no visible CUDA"),
    ("OSError: [Errno 98] Address already in use", "port already in use"),
    ("INFO Using MarlinLinearKernel for AutoAWQMarlinLinearMethod", None),
])
def test_diagnose_real_log_lines(log: str, why: str | None) -> None:
    d = vs.diagnose(log)
    assert (d is None) if why is None else (why in (d or ""))


def test_binary_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MQDE_VLLM_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(vs, "REPO", tmp_path)
    with pytest.raises(FileNotFoundError, match="MQDE_VLLM_BIN"):
        vs.vllm_bin()
    b = mock_backend.wrapper(tmp_path, "vllm")
    monkeypatch.setenv("MQDE_VLLM_BIN", str(b))
    assert vs.vllm_bin() == b


@pytest.mark.skipif(not (os.environ.get("MQDE_VLLM_BIN") and os.environ.get("MQDE_TEST_VLLM_MODEL")),
                    reason="needs MQDE_VLLM_BIN and MQDE_TEST_VLLM_MODEL (a vLLM-loadable checkpoint) and a GPU")
def test_real_vllm(tmp_path: Path) -> None:
    gpu = os.environ.get("MQDE_TEST_VLLM_GPU", "3")
    with vs.VllmServer(os.environ["MQDE_TEST_VLLM_MODEL"], tmp_path / "vllm.log", gpu=gpu, max_model_len=512,
                       gpu_memory_utilization=0.3, enforce_eager=True) as srv:
        assert openai_smoke(srv.port, max_tokens=4)["ok"]
        r = run_load(srv.port, LoadSpec(requests=4, concurrency=2, prompt_len=32, n_predict=8, warmup=1))
        info = srv.engine_info()
    assert r["ok"] == 4 and r["tpot_ms"]["p50"] > 0
    assert info.get("kv_cache_tokens", 0) > 0
