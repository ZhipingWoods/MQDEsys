"""Helpers that start the mock serving backend (backend/fake_llama_server.py) through the real adapters."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

from mqde.backend.llamacpp_server import LlamaServer
from mqde.backend.sglang_server import SglangServer
from mqde.backend.vllm_server import VllmServer

FAKE = Path(__file__).parent / "backend" / "fake_llama_server.py"


def wrapper(tmp_path: Path, name: str) -> Path:
    """An executable that runs the mock under the name of the real binary (adapters launch it by path)."""
    b = tmp_path / name
    b.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} "$@"\n')
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    return b


def llama(tmp_path: Path, *, ttft: float = 0.10, tpot: float = 0.02, parallel: int = 4, ctx: int = 4096,
          **kw: object) -> LlamaServer:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    return LlamaServer(model, tmp_path / "llama.log", parallel=parallel, ctx=ctx, binary=wrapper(tmp_path, "llama-server"),
                       extra=["--ttft", str(ttft), "--tpot", str(tpot)], **kw)  # type: ignore[arg-type]


def vllm(tmp_path: Path, *, model_name: str = "tiny", ttft: float = 0.10, tpot: float = 0.02, **kw: object) -> VllmServer:
    model = tmp_path / model_name
    model.mkdir(exist_ok=True)
    return VllmServer(model, tmp_path / "vllm.log", binary=wrapper(tmp_path, "vllm"),
                      extra=["--ttft", str(ttft), "--tpot", str(tpot)], **kw)  # type: ignore[arg-type]


def sglang(tmp_path: Path, *, model_name: str = "tiny", ttft: float = 0.10, tpot: float = 0.02, **kw: object) -> SglangServer:
    model = tmp_path / model_name
    model.mkdir(exist_ok=True)
    return SglangServer(model, tmp_path / "sglang.log", python=wrapper(tmp_path, "python"),
                        extra=["--ttft", str(ttft), "--tpot", str(tpot)], **kw)  # type: ignore[arg-type]
