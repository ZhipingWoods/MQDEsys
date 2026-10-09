"""Task accuracy through the official ``lm-eval`` command line (T-0801, ADR-0002 option A).

This module never imports lm-evaluation-harness. It builds a command line, runs the executable the user installed in
an environment of their choice (``MQDE_LM_EVAL_BIN``, else ``lm-eval`` on PATH), and reads the JSON file it writes. The
model under test is an OpenAI-compatible server that the deployment stage already started (vLLM, llama-server), reached
through lm-eval's ``local-completions`` model on 127.0.0.1. Because the executable is the official one, a run through
this adapter and a direct run with the same arguments and seed give the same numbers.

    run_lm_eval(port, served_name="m", tokenizer="/ckpt", tasks=["mmlu"], limit=50, out_dir=Path("out"))
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

ENV_BIN = "MQDE_LM_EVAL_BIN"


def lm_eval_bin() -> Path:
    for c in (os.environ.get(ENV_BIN, ""), shutil.which("lm-eval") or ""):
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        f"lm-eval not found (install lm-eval into an environment of your own and set {ENV_BIN} to its `lm-eval` binary)")


def build_command(binary: str | Path, port: int, served_name: str, tokenizer: str | Path, tasks: list[str], out_dir: str | Path, *,
                  limit: int | float | None = None, num_fewshot: int | None = None, concurrency: int = 8, seed: int = 1234,
                  max_gen_toks: int | None = None, extra: list[str] | None = None) -> list[str]:
    """The exact argument list (never a shell string). Same flags the official CLI documents."""
    if not tasks:
        raise ValueError("tasks is empty")
    model_args = (f"model={served_name},base_url=http://{HOST}:{port}/v1/completions,num_concurrent={concurrency},"
                  f"tokenizer_backend=huggingface,tokenizer={tokenizer},max_retries=3")
    cmd = [str(binary), "--model", "local-completions", "--model_args", model_args, "--tasks", ",".join(tasks),
           "--output_path", str(out_dir), "--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if max_gen_toks is not None:
        cmd += ["--gen_kwargs", f"max_gen_toks={max_gen_toks}"]
    return cmd + list(extra or [])


def first_problem(log: str) -> str | None:
    """The first line that says what went wrong (server rejection, missing module...), instead of the end of a long traceback."""
    for line in log.splitlines():
        if any(k in line for k in ("API request failed", "ModuleNotFoundError", "ImportError", "Connection refused", "No such file")):
            return line.strip()[:600]
    return None


def find_results(out_dir: str | Path) -> Path:
    """The newest ``results_*.json`` below out_dir (lm-eval writes <out_dir>/<model name>/results_<timestamp>.json)."""
    found = sorted(Path(out_dir).rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"lm-eval wrote no results_*.json under {out_dir}")
    return found[-1]


def parse_results(doc: dict[str, Any]) -> dict[str, Any]:
    """Per task: every metric as {"value", "stderr"}. lm-eval names them ``<metric>,<filter>`` with ``<metric>_stderr,<filter>``."""
    if not isinstance(doc.get("results"), dict):
        raise ValueError("not an lm-eval results file: no 'results' object")
    tasks: dict[str, dict[str, Any]] = {}
    for task, res in doc["results"].items():
        metrics: dict[str, dict[str, float | None]] = {}
        for key, val in res.items():
            # "name" / "alias" are strings; "sample_len" is the number of samples scored, not a metric
            if key in ("alias", "name", "sample_len") or "_stderr," in key or not isinstance(val, (int, float)):
                continue
            name, _, flt = key.partition(",")
            err = res.get(f"{name}_stderr,{flt}")
            stderr = float(err) if isinstance(err, (int, float)) else None
            metrics[key] = {"metric": name, "filter": flt, "value": float(val), "stderr": stderr}
        tasks[task] = {"alias": res.get("alias", task), "metrics": metrics}
    samples = doc.get("n-samples") or {}
    for task, n in samples.items():
        if task in tasks and isinstance(n, dict):
            tasks[task]["samples"] = {"original": n.get("original"), "effective": n.get("effective")}
    return {"framework": "lm-eval", "version": doc.get("lm_eval_version"), "git_hash": doc.get("git_hash"), "tasks": tasks,
            "n_shot": doc.get("n-shot"), "task_versions": doc.get("versions"), "task_hashes": doc.get("task_hashes")}


def run_lm_eval(port: int, served_name: str, tokenizer: str | Path, tasks: list[str], out_dir: Path, *, limit: int | float | None = None,
                num_fewshot: int | None = None, concurrency: int = 8, seed: int = 1234, max_gen_toks: int | None = None,
                extra: list[str] | None = None, timeout: float = 7200, binary: str | Path | None = None,
                env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the official CLI and return the parsed results with the command line, wall time and the log tail.

    Raises FileNotFoundError when the executable is missing (callers record the stage as skipped), RuntimeError when the
    process fails or times out (the message carries the end of its output).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(binary or lm_eval_bin(), port, served_name, tokenizer, tasks, out_dir, limit=limit, num_fewshot=num_fewshot,
                        concurrency=concurrency, seed=seed, max_gen_toks=max_gen_toks, extra=extra)
    t0 = time.perf_counter()
    log = out_dir / "lm_eval.log"
    with log.open("wb") as f:
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env={**os.environ, **(env or {})}, check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"lm-eval timed out after {timeout:.0f} s (log: {log})") from e
    wall = time.perf_counter() - t0
    tail = log.read_text(errors="replace")[-1500:]
    if r.returncode != 0:
        raise RuntimeError(f"lm-eval exited with {r.returncode}: {first_problem(log.read_text(errors='replace')) or '...' + tail}")
    path = find_results(out_dir)
    out = parse_results(json.loads(path.read_text(encoding="utf-8")))
    out.update({"command": cmd, "seconds": wall, "results_file": str(path), "limit": limit, "num_fewshot": num_fewshot, "seed": seed})
    return out
