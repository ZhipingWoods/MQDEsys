"""Perplexity (T-0510): non-overlapping windows over the wikitext-2 test split, the standard
GPTQ / AWQ / lm-eval "wikitext word-perplexity-free" token PPL used to compare quantized models."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F


@torch.inference_mode()
def perplexity(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    ids: torch.Tensor,
    seq_len: int = 512,
    max_windows: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    n = ids.numel() // seq_len
    if max_windows:
        n = min(n, max_windows)
    nll, count = 0.0, 0
    t0 = time.perf_counter()
    for i in range(n):
        w = ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
        logits = logits_fn(w).float()
        loss = F.cross_entropy(logits[0, :-1], w[0, 1:].to(logits.device), reduction="sum")
        nll += float(loss)
        count += seq_len - 1
        if progress:
            progress(i + 1, n)
    return {"ppl": math.exp(nll / count), "tokens": count, "windows": n, "seq_len": seq_len, "seconds": time.perf_counter() - t0}
