"""Boundary search and resource sampling (T-0804) against the mock backend and stand-in tools."""

from __future__ import annotations

import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mock_backend

from mqde.backend.base import ServerBackend
from mqde.eval import boundary
from mqde.eval.resources import ResourceSampler, gpu_snapshot, tree_rss_mib


# ---------------------------------------------------------------- bisection
@pytest.mark.parametrize(("lo", "hi", "limit", "step"), [(1, 64, 37, 1), (1, 64, 1, 1), (1, 64, 64, 1), (256, 8192, 3000, 256),
                                                         (5, 5, 5, 1), (1, 1000, 999, 1)])
def test_bisect_finds_the_largest_passing_point(lo: int, hi: int, limit: int, step: int) -> None:
    r = boundary.bisect_max(lo, hi, lambda x: {"ok": x <= limit}, step=step)
    best = max(x for x in range(lo, hi + 1, step) if x <= limit)
    assert r["max"] == best
    assert r["capped"] == (best == lo + (hi - lo) // step * step)
    assert len(r["probes"]) <= math.ceil(math.log2((hi - lo) // step + 1)) + 2  # logarithmic, not a scan
    assert all(p["x"] in range(lo, hi + 1, step) for p in r["probes"])
    if not r["capped"]:
        assert r["first_fail"] == r["max"] + step


def test_bisect_when_even_the_smallest_point_fails() -> None:
    r = boundary.bisect_max(4, 100, lambda x: {"ok": False, "reason": "no"})
    assert r["max"] is None and r["first_fail"] == 4 and len(r["probes"]) == 1


# ---------------------------------------------------------------- serving probes against the mock
def _factory(tmp_path: Path, **kw: object) -> boundary.BackendFactory:
    n = [0]

    def make(c: int, length: int) -> ServerBackend:
        n[0] += 1
        d = tmp_path / f"p{n[0]}"
        d.mkdir()
        return mock_backend.vllm(d, max_model_len=length, max_num_seqs=c, ttft=0.02, tpot=0.002, startup_timeout=30, **kw)

    return make


def test_max_concurrency_stops_at_the_out_of_memory_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_OOM_ABOVE_SEQS", "11")
    r = boundary.find_max_concurrency(_factory(tmp_path), max_len=64, lo=1, hi=40)
    assert r["max"] == 11 and not r["capped"] and r["first_fail"] == 12 and r["axis"] == "concurrency"
    fail = next(p for p in r["probes"] if not p["ok"])
    assert fail["stage"] == "startup" and "out of memory" in fail["reason"]
    good = next(p for p in r["probes"] if p["ok"])
    assert good["requests_ok"] == good["x"] and good["engine"]["kv_cache_tokens"] == 12345  # vLLM-style engine facts kept


def test_max_context_follows_the_kv_cache_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MAX_LEN_CAP", "2000")
    r = boundary.find_max_context(_factory(tmp_path), concurrency=2, lo=256, hi=4096, step=256)
    assert r["max"] == 1792 and r["first_fail"] == 2048 and r["axis"] == "context"
    fail = next(p for p in r["probes"] if p["x"] == 2048)
    assert "max_model_len exceeds" in fail["reason"]


def test_capped_when_everything_passes(tmp_path: Path) -> None:
    r = boundary.find_max_concurrency(_factory(tmp_path), max_len=64, lo=1, hi=4)
    assert r["max"] == 4 and r["capped"] and r["first_fail"] is None and len(r["probes"]) == 2  # only lo and hi are tried


def test_probe_fails_when_the_requests_fail(tmp_path: Path) -> None:
    class Mute(ServerBackend):
        name = "mute"

        def ready(self) -> bool:
            return True  # reports healthy but nothing listens on the port, so every request fails

    def make(c: int, length: int) -> ServerBackend:
        return Mute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "d.log", port=1, startup_timeout=10)

    r = boundary.probe_serving(make, 2, 64, timeout=3)
    assert not r["ok"] and r["stage"] == "serve" and r["requests_ok"] == 0 and r["reason"]


def test_probe_notices_a_server_process_that_died(tmp_path: Path) -> None:
    class Quits(ServerBackend):
        name = "quits"

        def ready(self) -> bool:
            return True

    def make(c: int, length: int) -> ServerBackend:
        code = "import time; time.sleep(1.0); print('torch.OutOfMemoryError: CUDA out of memory')"
        return Quits([sys.executable, "-c", code], tmp_path / "q.log", port=1, startup_timeout=10)

    Quits.diagnoses = [("out of memory", "out of memory under load")]  # type: ignore[misc]
    r = boundary.probe_serving(make, 2, 64, timeout=4)  # requests fail fast; by the time they are counted the process has exited
    assert not r["ok"] and r["stage"] == "serve"


# ---------------------------------------------------------------- resource sampling
def test_gpu_sampling_with_a_stand_in_nvidia_smi(tmp_path: Path) -> None:
    smi = tmp_path / "nvidia-smi"
    smi.write_text('#!/bin/sh\n'
                   'case "$*" in\n'
                   '  *-lms*) i=0; while true; do echo "$((2000 + i * 100)), $((100 + i)).5, 40"; i=$((i + 1)); sleep 0.03; done;;\n'
                   '  *) echo "1500, 90.0, 5";;\n'
                   'esac\n')
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    assert gpu_snapshot(0, str(smi)) == {"gpu_mem_mib": 1500.0, "gpu_power_w": 90.0, "gpu_util_pct": 5.0}
    with ResourceSampler(gpu=0, smi=str(smi), interval=0.03) as s:
        time.sleep(0.5)
    r = s.summary()
    assert r["samples"] >= 5 and r["gpu_mem_baseline_mib"] == 1500.0
    assert r["gpu_mem_peak_mib"] > 2000 and r["gpu_mem_peak_above_baseline_mib"] == r["gpu_mem_peak_mib"] - 1500
    assert r["gpu_power_peak_w"] >= r["gpu_power_mean_w"] > 100 and r["gpu_util_mean_pct"] == 40
    assert s._smi_proc is not None and s._smi_proc.poll() is not None  # the sampler cleaned up its nvidia-smi
    assert len(s.series(3)) <= 8


def test_missing_nvidia_smi_disables_the_gpu_part_only() -> None:
    assert gpu_snapshot(0, "definitely-not-a-binary") is None
    with ResourceSampler(gpu=0, pid=os.getpid(), smi="definitely-not-a-binary") as s:
        time.sleep(0.3)
    r = s.summary()
    assert "gpu_mem_peak_mib" not in r and r["host_rss_peak_mib"] > 10


def test_host_rss_covers_the_process_tree(tmp_path: Path) -> None:
    (tmp_path / "child.py").write_text("import time\nx = bytearray(b'\\x01') * (150 * 2**20)\ntime.sleep(30)\n")
    (tmp_path / "parent.py").write_text(
        f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, {str(tmp_path / 'child.py')!r}])\ntime.sleep(30)\n")
    p = subprocess.Popen([sys.executable, str(tmp_path / "parent.py")])
    try:
        deadline = time.time() + 20  # wait for the child to start and touch its memory, however loaded the machine is
        while tree_rss_mib(p.pid) <= 140 and time.time() < deadline:
            time.sleep(0.1)
        assert tree_rss_mib(p.pid) > 140  # the parent itself is ~10 MiB; the rest is the child's 150 MiB
        assert tree_rss_mib(2**22 + 12345) == 0.0  # no such process
    finally:
        p.kill()
        p.wait()


def test_default_workdir_is_not_a_job_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The server lists every directory under runs/ as a job: probe logs there turn up as fake jobs with state UNKNOWN."""
    seen: list[str] = []

    def fake_find(make: boundary.BackendFactory, max_len: int, **kw: object) -> dict:
        seen.append(str(make(1, max_len).log_path))
        return {"axis": "concurrency", "max_len": max_len, "max": 1, "capped": True, "first_fail": None, "probes": []}

    monkeypatch.setattr(boundary, "find_max_concurrency", fake_find)
    monkeypatch.setattr(sys, "argv", ["boundary", "vllm", str(tmp_path)])
    monkeypatch.setenv("MQDE_VLLM_BIN", str(mock_backend.wrapper(tmp_path, "vllm")))
    boundary.main()
    (log,) = seen
    assert Path(log).parts[0] == "boundary-runs" and "runs" not in Path(log).parts
