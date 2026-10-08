"""Resource sampler (T-0804): device memory / power / utilisation and host memory while a server is under load.

GPU numbers come from one long-running ``nvidia-smi --query-gpu ... -lms <interval>`` process (100 ms by default),
so sampling costs no per-sample process start. ``gpu`` is the *physical* index nvidia-smi uses, which is also what
``CUDA_VISIBLE_DEVICES`` takes. Device memory is whole-device ``memory.used``: on a shared GPU other processes are
included, so the summary also reports the baseline taken before sampling and the peak above it.

Host memory is the resident set of ``pid`` plus all its descendants (vLLM runs its engine in a child process),
read with ``ps`` every ``host_interval`` seconds. Nothing here needs a third-party package; a missing
``nvidia-smi`` just disables the GPU part.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any


def gpu_snapshot(gpu: int, smi: str = "nvidia-smi") -> dict[str, float] | None:
    """One reading of (memory_mib, power_w, util_pct) for physical GPU ``gpu``; None if unavailable."""
    if not shutil.which(smi) and "/" not in smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=memory.used,power.draw,utilization.gpu", "--format=csv,noheader,nounits",
                              "-i", str(gpu)], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_gpu_line(out.strip().splitlines()[0]) if out.strip() else None


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")  # "[N/A]" on boards without a power sensor


def _parse_gpu_line(line: str) -> dict[str, float] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    return {"gpu_mem_mib": _num(parts[0]), "gpu_power_w": _num(parts[1]), "gpu_util_pct": _num(parts[2])}


def tree_rss_mib(pid: int) -> float:
    """Resident memory (MiB) of ``pid`` and all its descendants, via ``ps``; 0 if the process is gone."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for ln in out.splitlines():
        f = ln.split()
        if len(f) == 3 and all(x.isdigit() for x in f):
            p, pp, r = int(f[0]), int(f[1]), int(f[2])
            rss[p] = r
            kids.setdefault(pp, []).append(p)
    todo, total = [pid], 0
    while todo:
        p = todo.pop()
        total += rss.get(p, 0)
        todo.extend(kids.get(p, []))
    return total / 1024.0


class ResourceSampler:
    def __init__(self, gpu: int | None = None, pid: int | None = None, interval: float = 0.1,
                 host_interval: float = 0.25, smi: str = "nvidia-smi") -> None:
        self.gpu, self.pid, self.interval, self.host_interval, self.smi = gpu, pid, interval, host_interval, smi
        self.samples: list[dict[str, float]] = []
        self.baseline_gpu_mib: float | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi_proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._t1 = 0.0

    def _add(self, s: dict[str, float]) -> None:
        s["t"] = time.perf_counter() - self._t0
        with self._lock:
            self.samples.append(s)

    def _gpu_loop(self) -> None:
        assert self._smi_proc is not None and self._smi_proc.stdout is not None
        for line in self._smi_proc.stdout:
            s = _parse_gpu_line(line)
            if s:
                self._add(s)

    def _host_loop(self) -> None:
        assert self.pid is not None
        while not self._stop.is_set():
            self._add({"host_rss_mib": tree_rss_mib(self.pid)})
            self._stop.wait(self.host_interval)

    def start(self) -> ResourceSampler:
        self._t0 = time.perf_counter()
        if self.gpu is not None:
            base = gpu_snapshot(self.gpu, self.smi)
            if base:
                self.baseline_gpu_mib = base["gpu_mem_mib"]
                self._smi_proc = subprocess.Popen(
                    [self.smi, "--query-gpu=memory.used,power.draw,utilization.gpu", "--format=csv,noheader,nounits",
                     "-i", str(self.gpu), "-lms", str(max(10, int(self.interval * 1000)))],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                self._threads.append(threading.Thread(target=self._gpu_loop, daemon=True))
        if self.pid is not None:
            self._threads.append(threading.Thread(target=self._host_loop, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._t1 = time.perf_counter()
        self._stop.set()
        if self._smi_proc is not None and self._smi_proc.poll() is None:
            self._smi_proc.terminate()
            try:
                self._smi_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._smi_proc.kill()
        for t in self._threads:
            t.join(timeout=5)

    def __enter__(self) -> ResourceSampler:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            ss = list(self.samples)
        out: dict[str, Any] = {"duration_s": (self._t1 or time.perf_counter()) - self._t0, "samples": len(ss)}

        def col(k: str) -> list[float]:
            return [s[k] for s in ss if k in s and s[k] == s[k]]  # drops NaN

        if (mem := col("gpu_mem_mib")):
            out["gpu_mem_peak_mib"] = max(mem)
            out["gpu_mem_baseline_mib"] = self.baseline_gpu_mib
            if self.baseline_gpu_mib is not None:
                out["gpu_mem_peak_above_baseline_mib"] = max(mem) - self.baseline_gpu_mib
        if (pw := col("gpu_power_w")):
            out["gpu_power_peak_w"], out["gpu_power_mean_w"] = max(pw), sum(pw) / len(pw)
        if (ut := col("gpu_util_pct")):
            out["gpu_util_mean_pct"] = sum(ut) / len(ut)
        if (rss := col("host_rss_mib")):
            out["host_rss_peak_mib"] = max(rss)
        return out

    def series(self, max_points: int = 300) -> list[dict[str, float]]:
        """Samples thinned to at most ``max_points`` evenly spaced ones (for charts)."""
        with self._lock:
            ss = list(self.samples)
        step = max(1, len(ss) // max_points)
        return ss[::step]
