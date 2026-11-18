"""Benchmark accuracy through the official ``evalscope`` command line.

Like the lm-eval adapter, this module never imports evalscope. It builds an argument list, runs the
executable the user installed (``MQDE_EVALSCOPE_BIN``, else ``evalscope`` on PATH), and reads the JSON
reports it writes. The server under test is the OpenAI-compatible endpoint the deployment stage already
started (vLLM / SGLang), so no extra model process is created here.

    run_evalscope(port, "m", ["gsm8k"], Path("out"), limit=50)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from mqde.backend.base import HOST

ENV_BIN = "MQDE_EVALSCOPE_BIN"


def evalscope_bin() -> Path:
    for c in (os.environ.get(ENV_BIN, ""), shutil.which("evalscope") or ""):
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        f"evalscope not found (install it into an environment of your own and set {ENV_BIN} to its `evalscope` binary)")


def build_command(binary: str | Path, port: int, model: str, datasets: list[str], out_dir: str | Path, *,
                  limit: int | float | None = None, generation_config: str | None = None,
                  eval_batch_size: int | None = None, seed: int = 42, extra: list[str] | None = None) -> list[str]:
    """The exact argument list (never a shell string) for `evalscope eval`."""
    if not datasets:
        raise ValueError("datasets is empty")
    cmd = [str(binary), "eval", "--model", model, "--api-url", f"http://{HOST}:{port}/v1",
           "--api-key", "EMPTY", "--eval-type", "openai_api", "--datasets", *datasets,
           "--work-dir", str(out_dir), "--no-timestamp", "--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if generation_config:
        cmd += ["--generation-config", generation_config]
    if eval_batch_size is not None:
        cmd += ["--eval-batch-size", str(eval_batch_size)]
    return cmd + list(extra or [])


def first_problem(log: str) -> str | None:
    for line in log.splitlines():
        if any(k in line for k in ("Error", "error:", "ModuleNotFoundError", "ImportError", "Connection refused",
                                   "No such file", "Traceback")):
            return line.strip()[:600]
    return None


def find_reports(out_dir: str | Path) -> list[Path]:
    """Every report JSON evalscope wrote below out_dir/reports (newest last)."""
    root = Path(out_dir) / "reports"
    found = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime) if root.is_dir() else []
    return [p for p in found if p.name != "collection_report.json"]


def parse_report(doc: dict[str, Any], fallback_name: str = "") -> dict[str, Any]:
    """Normalize one evalscope report (v1 `name`/`score` and v2 `identity`/`score` shapes)."""
    dataset = str(doc.get("dataset_name") or fallback_name or "unknown")
    metrics: dict[str, dict[str, Any]] = {}
    for m in doc.get("metrics") or []:
        if not isinstance(m, dict):
            continue
        identity = m.get("identity") if isinstance(m.get("identity"), dict) else {}
        key = str(m.get("name") or m.get("legacy_name") or identity.get("key") or identity.get("name") or "metric")
        score = m.get("score", m.get("value"))
        metrics[key] = {
            "value": float(score) if isinstance(score, (int, float)) else None,
            "num": m.get("num", m.get("sample_len")),
            "macro": m.get("macro_score"),
        }
    perf = doc.get("perf_metrics") or {}
    summary = perf.get("summary") if isinstance(perf, dict) else None
    latency = summary.get("latency") if isinstance(summary, dict) else None
    return {
        "dataset": dataset,
        "pretty_name": doc.get("dataset_pretty_name") or dataset,
        "model": doc.get("model_name"),
        "score": doc.get("score"),
        "metrics": metrics,
        "n_samples": summary.get("n_samples") if isinstance(summary, dict) else None,
        "latency_ms_mean": latency.get("mean") if isinstance(latency, dict) else None,
        "analysis": doc.get("analysis") if doc.get("analysis") != "N/A" else None,
    }


def parse_results(paths: list[Path]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or not (doc.get("metrics") or doc.get("perf_metrics")):
            continue
        item = parse_report(doc, path.stem)
        datasets[item["dataset"]] = item
    if not datasets:
        raise ValueError("evalscope wrote no report with metrics")
    return {"framework": "evalscope", "datasets": datasets}


def run_evalscope(port: int, model: str, datasets: list[str], out_dir: Path, *, limit: int | float | None = None,
                  generation_config: str | None = None, eval_batch_size: int | None = None, seed: int = 42,
                  extra: list[str] | None = None, timeout: float = 7200, binary: str | Path | None = None,
                  env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the official CLI and return normalized reports, command line, wall time and log tail.

    Raises FileNotFoundError when the executable is missing (callers mark the stage skipped), RuntimeError
    when the process fails or times out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(binary or evalscope_bin(), port, model, datasets, out_dir, limit=limit,
                        generation_config=generation_config, eval_batch_size=eval_batch_size, seed=seed, extra=extra)
    t0 = time.perf_counter()
    log = out_dir / "evalscope.log"
    with log.open("wb") as f:
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               env={**os.environ, **(env or {})}, check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"evalscope timed out after {timeout:.0f} s (log: {log})") from e
    wall = time.perf_counter() - t0
    tail = log.read_text(errors="replace")[-1500:]
    if r.returncode != 0:
        raise RuntimeError(f"evalscope exited with {r.returncode}: {first_problem(log.read_text(errors='replace')) or '...' + tail}")
    reports = find_reports(out_dir)
    out = parse_results(reports)
    out.update({"command": cmd, "seconds": wall, "reports": [str(p) for p in reports],
                "limit": limit, "seed": seed})
    return out
