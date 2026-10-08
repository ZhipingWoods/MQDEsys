"""Perplexity of a *deployed* model (T-0802 / S12), from the serving backend's own prompt log-probabilities.

Same protocol as ``eval.ppl.perplexity`` (non-overlapping windows of ``seq_len`` tokens, every token but the first
of each window scored) so the number is directly comparable with the fake-quant and real-weight PPL of the same
job; the difference is what the serving stack (kernels, dtype, scheduling) adds on top of the checkpoint.

Needs a backend that returns ``prompt_logprobs`` on ``/v1/completions`` (vLLM does; llama-server does not).
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mqde.backend.base import HOST, http_json


def window_nll(port: int, window: list[int], model: str | None, timeout: float, host: str = HOST) -> float:
    """Summed negative log-likelihood of tokens 1.. of ``window`` under the served model."""
    body: dict[str, Any] = {"prompt": window, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    if model:
        body["model"] = model
    r = http_json(port, "POST", "/v1/completions", body, timeout=timeout, host=host)
    lps = r["choices"][0].get("prompt_logprobs")
    if not lps or len(lps) != len(window):
        raise RuntimeError("backend returned no prompt_logprobs (does it support them?)")
    nll = 0.0
    for tok, entry in zip(window[1:], lps[1:], strict=True):
        lp = (entry or {}).get(str(tok))
        if lp is None:
            raise RuntimeError(f"prompt_logprobs lacks the prompt token {tok}")
        nll -= float(lp["logprob"] if isinstance(lp, dict) else lp)
    return nll


def served_perplexity(port: int, ids: list[int], seq_len: int = 512, max_windows: int | None = None,
                      model: str | None = None, concurrency: int = 4, timeout: float = 600,
                      host: str = HOST) -> dict[str, float]:
    n = len(ids) // seq_len
    if max_windows:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError(f"need at least {seq_len} token ids, got {len(ids)}")
    if model is None:
        data = (http_json(port, "GET", "/v1/models", timeout=30, host=host) or {}).get("data") or []
        model = data[0]["id"] if data else None
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        nlls = list(pool.map(lambda i: window_nll(port, ids[i * seq_len:(i + 1) * seq_len], model, timeout, host), range(n)))
    count = n * (seq_len - 1)
    return {"ppl": math.exp(sum(nlls) / count), "tokens": count, "windows": n, "seq_len": seq_len,
            "seconds": time.perf_counter() - t0}
