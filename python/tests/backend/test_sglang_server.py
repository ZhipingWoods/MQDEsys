"""SGLang adapter (T-0704): launch arguments, lifecycle against the mock, failure diagnosis, optional real run."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mock_backend

from mqde.backend import sglang_server as ss
from mqde.backend.base import openai_smoke
from mqde.eval.loadtest import LoadSpec, run_load


def test_launch_args() -> None:
    a = ss.launch_args("/x/python", "/m/ckpt", 8123, max_model_len=1024, mem_fraction_static=0.4, max_running_requests=8,
                       dtype="float16", disable_cuda_graph=True, served_name="q", extra=["--seed", "1"])
    assert a[:3] == ["/x/python", "-m", "sglang.launch_server"]
    flags = dict(itertools.pairwise(a[3:]))
    assert flags["--model-path"] == "/m/ckpt" and flags["--host"] == "127.0.0.1" and flags["--port"] == "8123"
    assert flags["--context-length"] == "1024" and flags["--mem-fraction-static"] == "0.4" and flags["--dtype"] == "float16"
    assert flags["--max-running-requests"] == "8" and flags["--served-model-name"] == "q"
    assert flags["--cuda-graph-backend-decode"] == "disabled" and flags["--cuda-graph-backend-prefill"] == "disabled"
    assert "--skip-server-warmup" in a and a[-2:] == ["--seed", "1"]
    plain = ss.launch_args("p", "m", 1)
    assert "--max-running-requests" not in plain and "--cuda-graph-backend-decode" not in plain  # left to SGLang's defaults
    assert "--trust-remote-code" not in plain, "remote code execution is never on by default"


def test_gpu_is_pinned_through_the_environment(tmp_path: Path) -> None:
    assert mock_backend.sglang(tmp_path, gpu=3)._env["CUDA_VISIBLE_DEVICES"] == "3"


def test_the_interpreter_directory_is_first_on_path_so_sglang_finds_its_own_ninja(tmp_path: Path) -> None:
    s = mock_backend.sglang(tmp_path)
    assert s._env["PATH"].split(os.pathsep)[0] == str(tmp_path)
    assert os.environ["PATH"] in s._env["PATH"], "the rest of PATH is kept"


def test_lifecycle_info_smoke_and_load(tmp_path: Path) -> None:
    srv = mock_backend.sglang(tmp_path, max_model_len=512, served_name="qwen-test", ttft=0.05, tpot=0.01)
    with srv:
        assert srv.startup_s is not None and srv.startup_s >= 0.3
        assert srv.props() == {"id": "qwen-test", "max_model_len": 512}
        assert openai_smoke(srv.port)["ok"]
        r = run_load(srv.port, LoadSpec(requests=4, concurrency=2, prompt_len=32, n_predict=8, warmup=1))
        info = srv.engine_info()
    assert srv.proc is None
    assert r["ok"] == 4 and r["failed"] == 0
    assert info["kv_cache_tokens"] == 12345 and info["weights_gib"] == 0.93 and info["weight_load_s"] == 1.2
    assert info["kv_cache_gib"] == 0.07 and info["avail_after_pool_gib"] == 19.5 and info["cuda_graph_s"] == 3.4


def test_engine_info_reads_the_log_lines_sglang_really_prints(tmp_path: Path) -> None:
    """The format strings come from sglang/srt/model_executor/model_runner.py and mem_cache/memory_pool.py."""
    srv = mock_backend.sglang(tmp_path)
    srv.log_path.write_text(
        "[2026-10-12 15:00:01] Load weight end. elapsed=2.51 s, type=Qwen2ForCausalLM, avail mem=21.40 GB, mem usage=0.93 GB.\n"
        "[2026-10-12 15:00:01] KV Cache is allocated. #tokens: 521264, KV size: 5.97 GB\n"  # MLA-style single KV size
        "[2026-10-12 15:00:02] Memory pool end. avail mem=14.10 GB\n")
    info = srv.engine_info()
    assert info == {"weights_gib": 0.93, "weight_load_s": 2.51, "kv_cache_tokens": 521264, "kv_cache_gib": 5.97,
                    "avail_after_pool_gib": 14.1}


@pytest.mark.parametrize(("model", "why"), [
    ("oom-model", "out of memory"),
    ("badquant-model", "quantization method is not known"),
    ("lowmem-model", "mem_fraction_static"),
])
def test_startup_failures_are_diagnosed(tmp_path: Path, model: str, why: str) -> None:
    srv = mock_backend.sglang(tmp_path, model_name=model, startup_timeout=20)
    with pytest.raises(RuntimeError, match=why):
        srv.start()


def test_missing_model_path(tmp_path: Path) -> None:
    srv = mock_backend.sglang(tmp_path, startup_timeout=20)
    srv.args[srv.args.index("--model-path") + 1] = str(tmp_path / "does-not-exist")
    with pytest.raises(RuntimeError, match="model path not found"):
        srv.start()


@pytest.mark.parametrize(("log", "why"), [
    ("RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.", "mem_fraction_static"),
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "out of memory"),
    ("RuntimeError: Capture cuda graph failed: boom", "CUDA graph capture failed"),
    ("ValueError: Unknown quantization method: foo. Must be one of", "quantization method is not known"),
    ("RuntimeError: No CUDA GPUs are available", "no visible CUDA"),
    ("OSError: [Errno 98] Address already in use", "port already in use"),
    ("ModuleNotFoundError: No module named 'sglang'", "no sglang installed"),
    ("[2026-10-12] KV Cache is allocated. #tokens: 5, KV size: 0.01 GB", None),
])
def test_diagnose_real_log_lines(log: str, why: str | None) -> None:
    d = ss.diagnose(log)
    assert (d is None) if why is None else (why in (d or ""))


def test_interpreter_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MQDE_SGLANG_PYTHON", raising=False)
    monkeypatch.setattr(ss, "REPO", tmp_path)
    with pytest.raises(FileNotFoundError, match="MQDE_SGLANG_PYTHON"):
        ss.sglang_python()
    b = mock_backend.wrapper(tmp_path, "python")
    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(b))
    assert ss.sglang_python() == b


@pytest.mark.skipif(not (os.environ.get("MQDE_SGLANG_PYTHON") and os.environ.get("MQDE_TEST_SGLANG_MODEL")),
                    reason="needs MQDE_SGLANG_PYTHON and MQDE_TEST_SGLANG_MODEL (an HF checkpoint dir) and a GPU")
def test_real_sglang(tmp_path: Path) -> None:
    gpu = os.environ.get("MQDE_TEST_SGLANG_GPU", "5")
    with ss.SglangServer(os.environ["MQDE_TEST_SGLANG_MODEL"], tmp_path / "sglang.log", gpu=gpu, max_model_len=512,
                         mem_fraction_static=0.3, dtype="float16") as srv:
        assert openai_smoke(srv.port, max_tokens=4)["ok"]
        r = run_load(srv.port, LoadSpec(requests=4, concurrency=2, prompt_len=32, n_predict=8, warmup=1))
        info = srv.engine_info()
    assert r["ok"] == 4 and r["tpot_ms"]["p50"] > 0
    assert info.get("kv_cache_tokens", 0) > 0
