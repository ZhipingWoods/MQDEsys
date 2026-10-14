"""Text calibration / evaluation data.

Evaluation is always wikitext-2 raw *test*, read from the local HF datasets cache (the PPL protocol has to be the same
across jobs for numbers to be comparable). Calibration is wikitext-2 *train* by default, or any local text the user names:
a file (.txt .md .jsonl .json .parquet) or a directory of them (read recursively, in path order).

No network: the parquet files must already be in the HF cache (``datasets--wikitext``).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

BUILTIN_CALIB = "wikitext-2-raw-v1:train"
TEXT_SUFFIXES = (".txt", ".md", ".jsonl", ".json", ".parquet")
# A corpus is read in path order until this many characters: tokenizing gigabytes to draw a few dozen windows helps nobody.
MAX_CALIB_CHARS = 50_000_000


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


@dataclass
class CalibText:
    text: str
    files: int
    chars: int
    truncated: bool


def _record_text(rec: Any) -> str | None:
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict) and isinstance(rec.get("text"), str):
        return rec["text"]
    return None


def _file_texts(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [path.read_text(encoding="utf-8", errors="replace")]
    if suffix == ".jsonl":
        out: list[str] = []
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                t = _record_text(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{n}: not valid JSON ({e.msg})") from e
            if t:
                out.append(t)
        return out
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name}: not valid JSON ({e.msg})") from e
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: expected a JSON list of strings or of objects with a 'text' field")
        return [t for t in (_record_text(r) for r in data) if t]
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if "text" not in table.column_names:
        raise ValueError(f"{path.name}: no 'text' column (it has: {', '.join(table.column_names)})")
    return [t for t in table.column("text").to_pylist() if isinstance(t, str) and t.strip()]


def load_calib_text(source: str = BUILTIN_CALIB) -> CalibText:
    """The calibration corpus as one string. `source` is BUILTIN_CALIB or a path to a file / directory."""
    if source == BUILTIN_CALIB:
        t = wikitext_text("train")
        return CalibText(t, 1, len(t), False)
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"calibration data '{source}' not found")
    if root.is_dir():
        files = sorted(
            (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES),
            key=lambda p: p.relative_to(root).as_posix(),
        )
    else:
        files = [root] if root.suffix.lower() in TEXT_SUFFIXES else []
    if not files:
        raise ValueError(f"no readable text in '{source}' (supported: {', '.join(TEXT_SUFFIXES)})")
    parts: list[str] = []
    chars, used, truncated = 0, 0, False
    for f in files:
        before = len(parts)
        for t in _file_texts(f):
            if not t.strip():
                continue
            sep = 2 if parts else 0  # the "\n\n" between pieces
            room = MAX_CALIB_CHARS - chars - sep
            if room <= 0:
                truncated = True
                break
            if len(t) > room:
                t, truncated = t[:room], True
            parts.append(t)
            chars += len(t) + sep
        used += len(parts) > before  # a file counts when any of it was read
        if truncated:
            break
    if not parts:
        raise ValueError(f"no readable text in '{source}' (supported: {', '.join(TEXT_SUFFIXES)})")
    return CalibText("\n\n".join(parts), used, chars, truncated)


def calib_batches(tokenizer: Any, n_samples: int = 32, seq_len: int = 512, seed: int = 0, source: str = BUILTIN_CALIB,
                  info: dict[str, Any] | None = None) -> list[torch.Tensor]:
    """n_samples random windows of seq_len tokens from the calibration text (GPTQ/AWQ convention).

    `info`, when given, is filled with what was read: dataset, files, chars, tokens, truncated."""
    corpus = load_calib_text(source)
    ids = tokenizer(corpus.text, return_tensors="pt").input_ids[0]
    if info is not None:
        info.update(dataset=source, files=corpus.files, chars=corpus.chars, tokens=int(ids.numel()), truncated=corpus.truncated)
    if ids.numel() < seq_len + 2:
        raise ValueError(
            f"calibration data '{source}' has {ids.numel()} tokens, fewer than one window of {seq_len} (needs at least {seq_len + 2})"
        )
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq_len - 1, (n_samples,), generator=g)
    return [ids[s : s + seq_len].unsqueeze(0) for s in starts.tolist()]


def eval_ids(tokenizer: Any, max_tokens: int | None = None) -> torch.Tensor:
    ids = tokenizer(wikitext_text("test"), return_tensors="pt").input_ids[0]
    return ids[:max_tokens] if max_tokens else ids
