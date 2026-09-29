"""Text calibration / evaluation data (wikitext-2 raw, read from the local HF datasets cache).

No network: the parquet files must already be in the HF cache (``datasets--wikitext``). Calibration
samples come from the *train* split and evaluation from *test*, so they never overlap.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch


def _wikitext_file(split: str) -> Path:
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = sorted(glob.glob(f"{root}/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/{split}-*.parquet"))
    if not hits:
        raise FileNotFoundError(f"wikitext-2-raw-v1 {split} not found under {root}/datasets--wikitext")
    return Path(hits[0])


def wikitext_text(split: str) -> str:
    import pyarrow.parquet as pq

    rows = pq.read_table(_wikitext_file(split)).column("text").to_pylist()
    return "\n\n".join(r for r in rows if r.strip())


def calib_batches(tokenizer, n_samples: int = 32, seq_len: int = 512, seed: int = 0) -> list[torch.Tensor]:
    """n_samples random windows of seq_len tokens from the train split (GPTQ/AWQ convention)."""
    ids = tokenizer(wikitext_text("train"), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq_len - 1, (n_samples,), generator=g)
    return [ids[s : s + seq_len].unsqueeze(0) for s in starts.tolist()]


def eval_ids(tokenizer, max_tokens: int | None = None) -> torch.Tensor:
    ids = tokenizer(wikitext_text("test"), return_tensors="pt").input_ids[0]
    return ids[:max_tokens] if max_tokens else ids
