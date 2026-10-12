"""Perplexity of a *deployed* model (T-0802 / S12), from the serving backend's own prompt log-probabilities.

Same protocol as ``eval.ppl.perplexity`` (non-overlapping windows of ``seq_len`` tokens, every token but the first
of each window scored) so the number is directly comparable with the fake-quant and real-weight PPL of the same
job; the difference is what the serving stack (kernels, dtype, scheduling) adds on top of the checkpoint.

Needs a backend that returns prompt log-probabilities: ``prompt_logprobs`` on ``/v1/completions`` (vLLM; the
default ``api="vllm"``) or SGLang's native ``/generate`` with ``return_logprob`` (``api="sglang"``, scored over
token ids so no tokenizer round trip can shift a window). llama-server has neither.
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


def window_nll_sglang(port: int, window: list[int], timeout: float, host: str = HOST) -> float:
    """Same quantity as ``window_nll`` from SGLang's native endpoint: ``input_token_logprobs`` is a list of
    ``[logprob | None, token_id, text | None]``, one per prompt token, the first one's logprob being None."""
    body = {"input_ids": window, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": 0}
    r = http_json(port, "POST", "/generate", body, timeout=timeout, host=host)
    lps = ((r or {}).get("meta_info") or {}).get("input_token_logprobs")
    if not lps or len(lps) != len(window):
        raise RuntimeError("backend returned no input_token_logprobs (is this an SGLang server?)")
    nll = 0.0
    for tok, entry in zip(window[1:], lps[1:], strict=True):
        if entry is None or entry[0] is None:
            raise RuntimeError(f"input_token_logprobs lacks a logprob for prompt token {tok}")
        if int(entry[1]) != tok:
            raise RuntimeError(f"input_token_logprobs is misaligned: expected token {tok}, got {entry[1]}")
        nll -= float(entry[0])
    return nll


def served_perplexity(port: int, ids: list[int], seq_len: int = 512, max_windows: int | None = None,
                      model: str | None = None, concurrency: int = 4, timeout: float = 600,
                      host: str = HOST, api: str = "vllm") -> dict[str, float]:
    if api not in ("vllm", "sglang"):
        raise ValueError(f"api must be 'vllm' or 'sglang', got {api!r}")
    n = len(ids) // seq_len
    if max_windows:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError(f"need at least {seq_len} token ids, got {len(ids)}")
    if api == "vllm" and model is None:
        data = (http_json(port, "GET", "/v1/models", timeout=30, host=host) or {}).get("data") or []
        model = data[0]["id"] if data else None
    t0 = time.perf_counter()

    def one(i: int) -> float:
        w = ids[i * seq_len:(i + 1) * seq_len]
        return window_nll_sglang(port, w, timeout, host) if api == "sglang" else window_nll(port, w, model, timeout, host)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        nlls = list(pool.map(one, range(n)))
    count = n * (seq_len - 1)
    return {"ppl": math.exp(sum(nlls) / count), "tokens": count, "windows": n, "seq_len": seq_len,
            "seconds": time.perf_counter() - t0}
