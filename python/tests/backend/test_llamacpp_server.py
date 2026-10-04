"""llama-server adapter (T-0705): process lifecycle, failure diagnosis and streaming timing.

The fake server (fake_llama_server.py) streams with TTFT = 100 ms and TPOT = 20 ms, so the measured values are
checked against known ground truth. A real-binary test runs when llama-server and a GGUF are available.
"""

from __future__ import annotations

import glob
import os
import stat
import sys
from pathlib import Path

import pytest

from mqde.backend import llamacpp_server as ls

FAKE = Path(__file__).with_name("fake_llama_server.py")


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    b = tmp_path / "llama-server"
    b.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    return b


def test_lifecycle_smoke_and_timing(fake_bin: Path, tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    srv = ls.LlamaServer(model, tmp_path / "server.log", parallel=4, ctx=4096, binary=fake_bin)
    assert "--host" in srv.args and srv.args[srv.args.index("--host") + 1] == "127.0.0.1"
    with srv:
        assert srv.startup_s is not None and srv.startup_s >= 0.3
        assert srv.props()["total_slots"] == 4
        sm = ls.smoke(srv.port)
        assert sm["ok"] and sm["text"] == "Paris"
        r = ls.serve_bench(srv.port, concurrency=4, requests=8, prompt_len=16, n_predict=10)
    assert srv.proc is None
    assert r["ok"] == 8 and not r["errors"]
    # ground truth: ttft 100 ms, tpot 20 ms, e2e = 100 + 9 * 20 = 280 ms
    assert r["ttft_ms"]["p50"] == pytest.approx(100, abs=15)
    assert r["tpot_ms"]["p50"] == pytest.approx(20, abs=3)
    assert r["e2e_ms"]["p50"] == pytest.approx(280, abs=30)
    assert r["output_tokens_per_s"] > 0 and r["requests_per_s"] > 0


def test_gpu_layers_flag(fake_bin: Path, tmp_path: Path) -> None:
    default = ls.LlamaServer(tmp_path / "m.gguf", tmp_path / "a.log", binary=fake_bin)
    assert "-ngl" not in default.args  # None keeps llama.cpp's own default
    metal = ls.LlamaServer(tmp_path / "m.gguf", tmp_path / "b.log", binary=fake_bin, gpu_layers=99)
    assert metal.args[metal.args.index("-ngl") + 1] == "99"
    assert str(fake_bin.parent) in metal._env["DYLD_LIBRARY_PATH"]


def test_load_failure_is_diagnosed(fake_bin: Path, tmp_path: Path) -> None:
    srv = ls.LlamaServer(tmp_path / "missing.gguf", tmp_path / "server.log", binary=fake_bin, startup_timeout=20)
    with pytest.raises(RuntimeError, match="model file not found"):
        srv.start()
    assert srv.proc is not None and srv.proc.poll() == 1


@pytest.mark.parametrize(("log", "why"), [
    ("ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 9e9", "out of memory"),
    ("llama_model_load: error loading model: unknown model architecture: 'foo'", "not supported"),
    ("gguf_init_from_file_impl: invalid magic characters", "not a GGUF"),
    ("all fine", None),
])
def test_diagnose(log: str, why: str | None) -> None:
    d = ls.diagnose(log)
    assert (d is None) if why is None else (why in (d or ""))


def _real() -> tuple[str, str] | None:
    try:
        b = str(ls.server_bin())
    except FileNotFoundError:
        return None
    m = os.environ.get("MQDE_TEST_GGUF") or next(iter(sorted(
        glob.glob("runs/*/export/gguf/model-q4_k.gguf") + glob.glob("runs/*/model-q4_k.gguf"))), None)
    return (b, m) if m and Path(m).is_file() else None


@pytest.mark.skipif(_real() is None, reason="llama-server or a GGUF (MQDE_TEST_GGUF) not available")
def test_real_llama_server(tmp_path: Path) -> None:
    b, m = _real()  # type: ignore[misc]
    ngl = int(os.environ["MQDE_TEST_NGL"]) if os.environ.get("MQDE_TEST_NGL") else None
    with ls.LlamaServer(m, tmp_path / "server.log", threads=4, ctx=2048, parallel=2, binary=b, gpu_layers=ngl) as srv:
        assert ls.smoke(srv.port, max_tokens=4)["ok"]
        r = ls.serve_bench(srv.port, concurrency=2, requests=2, prompt_len=32, n_predict=8)
    assert r["ok"] == 2 and r["tpot_ms"]["p50"] > 0
