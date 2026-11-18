"""Throughput benchmark for one model on one GPU (driven by `mqde bench`, the reference notes A.8).

Prints exactly one JSON object on the LAST stdout line; everything else goes to stderr.

Currently supports VLA checkpoints (MiniCPM-RobotManip family): one "step" is a full decision
step at concurrency 1 — vision + language forward, then the flow-matching action head — which is
the same unit `mqde fit` estimates as steps/s. LLM decode tok/s is not implemented yet.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def bench_vla(model: str, runs: int, warmup: int, dataset: str, seed: int) -> dict:
    from mqde.data.vla import load_samples
    from mqde.models.minicpm_vla import MiniCPMVLA

    t0 = time.perf_counter()
    vla = MiniCPMVLA(model, device="cuda:0")
    _log(f"loaded {model} in {time.perf_counter() - t0:.1f}s on {torch.cuda.get_device_name(0)}")
    samples = load_samples(dataset, max(runs, 1), seed=seed)
    inputs = [vla.preprocess(s) for s in samples]

    def step(i: int) -> float:
        s, inp = samples[i % len(samples)], inputs[i % len(inputs)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        vla.predict(s, seed=seed + i, inputs=inp)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3

    for i in range(warmup):
        step(i)
    lat = [step(i) for i in range(runs)]
    torch.cuda.reset_peak_memory_stats()
    step(0)
    peak = torch.cuda.max_memory_allocated()
    mean = statistics.fmean(lat)
    _log(f"{runs} steps: mean {mean:.1f} ms, min {min(lat):.1f}, max {max(lat):.1f}")
    return {
        "model": model,
        "quant": "bf16",
        "runtime": "hf_transformers",
        "unit": "steps/s",
        "num_runs": runs,
        "avg_tps": 1e3 / mean,
        "min_tps": 1e3 / max(lat),
        "max_tps": 1e3 / min(lat),
        "latency_ms": lat,
        "peak_mem_bytes": peak,
        "context_tokens": int(inputs[0]["input_ids"].shape[1]),
        "device_name": torch.cuda.get_device_name(0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dataset", default="synthetic", help="'synthetic' or a LeRobot v3 dataset dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.runs < 1:
        ap.error("--runs must be >= 1")
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA is not available"}))
        sys.exit(2)
    try:
        with open(f"{a.model}/config.json") as f:
            cfg = json.load(f)
    except OSError as e:
        print(json.dumps({"error": f"cannot read config.json: {e}"}))
        sys.exit(2)
    if "action_horizon" not in cfg:
        print(json.dumps({"error": "only VLA checkpoints (config.action_horizon) are supported by mqde bench so far"}))
        sys.exit(2)
    print(json.dumps(bench_vla(a.model, a.runs, a.warmup, a.dataset, a.seed)), flush=True)


if __name__ == "__main__":
    main()
