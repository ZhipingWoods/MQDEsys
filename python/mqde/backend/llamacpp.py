"""llama.cpp backend (T-0705): run a GGUF with ``llama-perplexity`` / ``llama-bench`` on CPU (or Metal/CUDA
builds). Binaries come from MQDE_LLAMACPP_BIN or the repo's out-of-tree build ``build/llamacpp/bin``."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def bin_dir() -> Path:
    for c in [os.environ.get("MQDE_LLAMACPP_BIN", ""), str(REPO / "build/llamacpp/bin")]:
        if c and (Path(c) / "llama-perplexity").is_file():
            return Path(c)
    raise FileNotFoundError("llama.cpp binaries not found (build them into build/llamacpp or set MQDE_LLAMACPP_BIN)")


def _env() -> dict[str, str]:
    b = str(bin_dir())
    return {**os.environ, "LD_LIBRARY_PATH": f"{b}:{os.environ.get('LD_LIBRARY_PATH', '')}", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def perplexity(model: str | Path, text_file: str | Path, ctx: int = 512, chunks: int = 40, threads: int = 32, timeout: int = 3600) -> dict[str, Any]:
    """llama-perplexity: scores the second half of each ctx-token chunk (its standard protocol)."""
    cmd = [str(bin_dir() / "llama-perplexity"), "-m", str(model), "-f", str(text_file), "-c", str(ctx),
           "--chunks", str(chunks), "-t", str(threads), "-b", str(ctx)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(), timeout=timeout)
    log = r.stdout + r.stderr
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", log)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"llama-perplexity failed ({r.returncode}): {log[-1500:]}")
    speed = re.search(r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens", log)
    return {"ppl": float(m.group(1)), "ppl_err": float(m.group(2)), "ctx": ctx, "chunks": chunks, "threads": threads,
            "prompt_tokens_per_s": (int(speed.group(2)) / float(speed.group(1)) * 1e3) if speed else None}


def bench(model: str | Path, pp: int = 512, tg: int = 128, threads: int = 32, reps: int = 3, timeout: int = 1800) -> dict[str, Any]:
    cmd = [str(bin_dir() / "llama-bench"), "-m", str(model), "-p", str(pp), "-n", str(tg), "-t", str(threads),
           "-r", str(reps), "-o", "json"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(), timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({r.returncode}): {r.stderr[-1500:]}")
    rows = json.loads(r.stdout[r.stdout.index("[") :])
    out: dict[str, Any] = {"threads": threads}
    for row in rows:
        kind = "pp" if row.get("n_prompt", 0) > 0 else "tg"
        out[f"{kind}{row.get('n_prompt') or row.get('n_gen')}_tokens_per_s"] = row.get("avg_ts")
        out[f"{kind}_stddev"] = row.get("stddev_ts")
        out.setdefault("model_size", row.get("model_size"))
        out.setdefault("backend", row.get("backends") or row.get("backend"))
    return out
