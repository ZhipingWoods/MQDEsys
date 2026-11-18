"""Offline accuracy benchmarks that need no serving backend.

The functions here take small callables (``encode``, ``logits_fn``, ``generate_fn``) instead of a
concrete transformers model, so they are testable with a fake model and can also be reused by the
quantization report.  ``eval_job`` binds them to an ``HFLogits`` adapter.

Kinds:

* ``ppl``  -- perplexity over token windows (delegates to ``eval.ppl``).
* ``mcq``  -- multiple-choice accuracy scored by length-normalised answer log-probability.
* ``qa``   -- free-form generation scored with the existing ``custom`` metrics (exact / contains /
              regex / ROUGE-L).

Every result carries the number of samples, the limit actually used and the per-category breakdown,
which is what the report/Compare pages need to avoid mixing incompatible runs.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from mqde.eval import custom
from mqde.eval.ppl import perplexity
from mqde.eval.registry import BenchmarkMeta, get, suite

EncodeFn = Callable[[str], list[int]]
LogitsFn = Callable[[torch.Tensor], torch.Tensor]
GenerateFn = Callable[[Sequence[str], int], list[str]]


def _rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from e
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{n}: expected an object")
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    if not out:
        raise ValueError(f"{path}: no cases")
    return out


def _choice_index(row: dict[str, Any]) -> int:
    ans = row["answer"]
    choices = row["choices"]
    if isinstance(ans, int):
        if 0 <= ans < len(choices):
            return ans
        if 1 <= ans <= len(choices):  # tolerate 1-based answer keys
            return ans - 1
        raise ValueError(f"answer index {ans} outside {len(choices)} choices")
    a = str(ans).strip()
    for i, c in enumerate(choices):
        if str(c).strip() == a:
            return i
    if len(a) == 1 and a.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        i = ord(a.upper()) - ord("A")
        if i < len(choices):
            return i
    raise ValueError(f"answer '{ans}' is not one of the choices")


def _logprob(ids: list[int], prompt_len: int, logits_fn: LogitsFn) -> float:
    """Mean log-probability of ``ids[prompt_len:]`` under the model (teacher forcing)."""
    if len(ids) <= prompt_len:
        return float("-inf")
    with torch.inference_mode():
        logits = logits_fn(torch.tensor([ids], dtype=torch.long))[0].float()
    lp = torch.log_softmax(logits[prompt_len - 1 : -1], dim=-1)
    targets = torch.tensor(ids[prompt_len:], dtype=torch.long)
    return float(lp.gather(1, targets[:, None]).mean())


def score_mcq(row: dict[str, Any], encode: EncodeFn, logits_fn: LogitsFn,
              template: str = "{question}\nAnswer:") -> dict[str, Any]:
    """Pick the choice with the highest length-normalised log-probability."""
    prompt = template.format(question=row.get("question") or row.get("prompt") or "")
    prompt_ids = encode(prompt)
    scores = []
    for c in row["choices"]:
        ids = prompt_ids + encode(" " + str(c))
        scores.append(_logprob(ids, len(prompt_ids), logits_fn))
    pred = max(range(len(scores)), key=lambda i: scores[i])
    want = _choice_index(row)
    return {"id": row.get("id"), "category": row.get("subject") or row.get("category") or "all",
            "predicted": pred, "answer": want, "passed": pred == want, "scores": scores}


def run_mcq(meta: BenchmarkMeta, encode: EncodeFn, logits_fn: LogitsFn, *, limit: int | None = None,
            template: str = "{question}\nAnswer:") -> dict[str, Any]:
    if not meta.path:
        raise ValueError(f"benchmark '{meta.name}' has no dataset path")
    t0 = time.perf_counter()
    results = [score_mcq(r, encode, logits_fn, template) for r in _rows(meta.path, limit)]
    by: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by[r["category"]].append(bool(r["passed"]))
    n = len(results)
    return {"kind": "mcq", "metrics": {"accuracy": sum(r["passed"] for r in results) / n},
            "n": n, "limit": limit, "seconds": time.perf_counter() - t0,
            "categories": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by.items())},
            "results": results}


def run_qa(meta: BenchmarkMeta, generate: GenerateFn, *, limit: int | None = None,
           max_new_tokens: int = 64, concurrency: int = 4) -> dict[str, Any]:
    if not meta.path:
        raise ValueError(f"benchmark '{meta.name}' has no dataset path")
    t0 = time.perf_counter()
    cases = _rows(meta.path, limit)
    from mqde.eval.custom import load_cases

    # Reuse the same validation/defaults users get for a custom JSONL file.
    validated = load_cases(meta.path)
    if limit is not None:
        validated = validated[:limit]
    outputs: list[str] = []
    # The adapter may batch; fall back to one prompt at a time for simple fakes.
    try:
        outputs = list(generate([c["prompt"] for c in validated], max_new_tokens))
    except TypeError:
        outputs = [generate([c["prompt"]], max_new_tokens)[0] for c in validated]
    results = []
    for c, resp in zip(validated, outputs, strict=True):
        results.append({"id": c["id"], "metric": c["metric"], "passed": custom.score(c, resp), "response": resp})
    n = len(results)
    return {"kind": "qa", "metrics": {"accuracy": sum(r["passed"] for r in results) / n},
            "n": n, "limit": limit, "seconds": time.perf_counter() - t0,
            "results": results}


def run_ppl(ids: torch.Tensor, logits_fn: LogitsFn, *, seq_len: int = 512,
            max_windows: int | None = None) -> dict[str, Any]:
    r = perplexity(logits_fn, ids, seq_len, max_windows)
    return {"kind": "ppl", "metrics": {"ppl": r["ppl"]}, "n": r["windows"], "tokens": r["tokens"],
            "windows": r["windows"], "seq_len": seq_len, "seconds": r["seconds"],
            "window_nll": r["window_nll"]}


def run_benchmarks(names: str | Sequence[str] | None, *, encode: EncodeFn | None = None,
                   logits_fn: LogitsFn | None = None, generate: GenerateFn | None = None,
                   ppl_ids: torch.Tensor | None = None, limit: int | None = None, seq_len: int = 512,
                   max_windows: int | None = None, max_new_tokens: int = 64,
                   progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    """Run a suite / explicit benchmark list and return one report object."""
    names = suite(names) if isinstance(names, str) else list(names or [])
    out: dict[str, Any] = {"framework": "mqde-offline", "benchmarks": {}}
    for i, name in enumerate(names, start=1):
        meta = get(name)
        if progress:
            progress(name, i, len(names))
        if meta.runner == "native":
            try:
                from mqde.eval.native import run_native

                out["benchmarks"][meta.name] = run_native(
                    meta, encode=encode, logits_fn=logits_fn, generate=generate, limit=limit,
                    progress=lambda _n, done, total: progress(meta.name, done, total) if progress else None)
            except Exception as e:
                out["benchmarks"][meta.name] = {
                    "status": "failed", "runner": "native", "reason": f"{type(e).__name__}: {e}"}
        elif meta.kind == "ppl" and ppl_ids is not None and logits_fn is not None:
            out["benchmarks"][meta.name] = run_ppl(ppl_ids, logits_fn, seq_len=seq_len, max_windows=max_windows)
        elif meta.kind == "mcq" and encode is not None and logits_fn is not None:
            out["benchmarks"][meta.name] = run_mcq(meta, encode, logits_fn, limit=limit)
        elif meta.kind == "qa" and generate is not None:
            out["benchmarks"][meta.name] = run_qa(meta, generate, limit=limit, max_new_tokens=max_new_tokens)
        else:
            out["benchmarks"][meta.name] = {
                "status": "skipped",
                "reason": f"benchmark '{meta.name}' ({meta.kind}) needs a different model adapter",
            }
    return out
