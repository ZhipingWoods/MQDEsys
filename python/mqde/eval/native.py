"""Native evaluation datasets (GSM8K, MMLU, ARC, HellaSwag, PIQA, Winogrande).

This is the in-process counterpart of the external lm-eval / EvalScope adapters. Rows are loaded
with HuggingFace ``datasets`` (optional dependency), normalized to either an MCQ case or a
question/answer case, then scored with the same encode / logits / generate callables the offline
benchmarks use. No serving backend is required.

The dataset schemas differ in small ways (ARC uses ``answerKey``, HellaSwag uses ``ctx`` +
``endings``, PIQA uses ``sol1/sol2``), so normalization lives here instead of leaking into each
pipeline.
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from mqde.eval.offline import EncodeFn, GenerateFn, LogitsFn, score_mcq
from mqde.eval.registry import BenchmarkMeta


def _datasets() -> Any:
    try:
        import datasets
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("native benchmarks need the optional `datasets` package (pip install datasets)") from e
    return datasets


def load_native_rows(meta: BenchmarkMeta, limit: int | None = None, *, cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load one native dataset split. ``limit`` is applied before any model work."""
    ds = _datasets()
    kwargs: dict[str, Any] = {"split": meta.split or "test", "trust_remote_code": False}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    def load():
        return ds.load_dataset(meta.dataset, meta.subset, **kwargs) if meta.subset else ds.load_dataset(meta.dataset, **kwargs)

    try:
        loaded = load()
    except Exception:
        # Same fallback as the Hub API: direct huggingface.co can be blocked while the mirror is not.
        if os.environ.get("HF_ENDPOINT"):
            raise
        os.environ["HF_ENDPOINT"] = os.environ.get("MQDE_HF_FALLBACK_ENDPOINT", "https://hf-mirror.com")
        loaded = load()
    rows = []
    for row in loaded:
        rows.append(dict(row))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"native benchmark '{meta.name}' returned no rows from {meta.dataset} [{meta.subset or '-'}:{meta.split}]")
    return rows


def _answer_index(value: Any, choices: Sequence[str]) -> int:
    if isinstance(value, (int,)) and not isinstance(value, bool):
        if 0 <= value < len(choices):
            return value
        if 1 <= value <= len(choices):
            return value - 1
        raise ValueError(f"answer index {value} outside {len(choices)} choices")
    s = str(value).strip()
    if s.lstrip("+-").isdigit():
        i = int(s)
        if 0 <= i < len(choices):
            return i
        if 1 <= i <= len(choices):
            return i - 1
        raise ValueError(f"answer index {s} outside {len(choices)} choices")
    for i, c in enumerate(choices):
        if s == str(c).strip():
            return i
    if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        i = ord(s.upper()) - ord("A")
        if i < len(choices):
            return i
    raise ValueError(f"answer {value!r} is not one of {list(choices)!r}")


def _choice_texts(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    if isinstance(choices, dict):
        choices = choices.get("text") or choices.get("label") or []
    if isinstance(choices, list):
        out = []
        for c in choices:
            if isinstance(c, dict):
                out.append(str(c.get("text") or c.get("label") or c))
            else:
                out.append(str(c))
        return out
    endings = row.get("endings")
    if isinstance(endings, list):
        ctx = str(row.get("ctx") or row.get("context") or "")
        return [f"{ctx} {e}".strip() for e in endings]
    if "sol1" in row and "sol2" in row:
        return [str(row["sol1"]), str(row["sol2"])]
    if "option1" in row and "option2" in row:
        sentence = str(row.get("sentence") or "")
        return [sentence.replace("_", str(row["option1"])), sentence.replace("_", str(row["option2"]))]
    raise ValueError(f"cannot derive choices from row keys {sorted(row)}")


def _answer_value(row: dict[str, Any]) -> Any:
    for key in ("answer", "answerKey", "label", "target"):
        if key in row:
            return row[key]
    raise ValueError(f"cannot find answer in row keys {sorted(row)}")


def normalize_mcq(meta: BenchmarkMeta, row: dict[str, Any]) -> dict[str, Any]:
    choices = _choice_texts(row)
    answer = _answer_index(_answer_value(row), choices)
    question = str(row.get("question") or row.get("goal") or row.get("ctx") or row.get("sentence") or row.get("prompt") or "").strip()
    if not question:
        raise ValueError(f"native benchmark '{meta.name}': empty question in row keys {sorted(row)}")
    return {"id": row.get("id"), "category": row.get("subject") or row.get("category") or "all",
            "question": question, "choices": choices, "answer": answer}


_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _canonical_number(text: str) -> str | None:
    nums = _NUMBER.findall(text.replace("$", ""))
    if not nums:
        return None
    value = nums[-1].replace(",", "")
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else format(f, ".12g")
    except ValueError:
        return None


def _reference_number(text: str) -> str | None:
    # GSM8K stores "explanation\n#### 42"; later rows sometimes use only the number.
    return _canonical_number(text)


def run_native_mcq(meta: BenchmarkMeta, encode: EncodeFn, logits_fn: LogitsFn, *, limit: int | None = None,
                   progress: Any = None) -> dict[str, Any]:
    rows = load_native_rows(meta, limit)
    cases = [normalize_mcq(meta, r) for r in rows]
    t0 = time.perf_counter()
    results = []
    for i, row in enumerate(cases, start=1):
        if progress:
            progress(meta.name, i, len(cases))
        results.append(score_mcq(row, encode, logits_fn, meta.template or "Question: {question}\nAnswer:"))
    by: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by[r["category"]].append(bool(r["passed"]))
    n = len(results)
    return {"kind": "mcq", "runner": "native", "metrics": {"accuracy": sum(r["passed"] for r in results) / n},
            "n": n, "limit": limit, "seconds": time.perf_counter() - t0,
            "categories": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by.items())},
            "results": results}


def run_native_qa(meta: BenchmarkMeta, generate: GenerateFn, *, limit: int | None = None,
                  progress: Any = None) -> dict[str, Any]:
    rows = load_native_rows(meta, limit)
    prompts, refs, ids = [], [], []
    for row in rows:
        q = str(row.get("question") or row.get("prompt") or row.get("problem") or "").strip()
        a = str(_answer_value(row))
        if not q:
            raise ValueError(f"native benchmark '{meta.name}': empty question")
        prompts.append((meta.template or "{question}").format(question=q))
        refs.append(_reference_number(a) if meta.name == "gsm8k" else a.strip())
        ids.append(row.get("id"))
    t0 = time.perf_counter()
    try:
        outputs = list(generate(prompts, meta.max_new_tokens))
    except TypeError:
        outputs = [generate([p], meta.max_new_tokens)[0] for p in prompts]
    results = []
    for i, (out, ref) in enumerate(zip(outputs, refs, strict=True), start=1):
        if progress:
            progress(meta.name, i, len(refs))
        pred = _canonical_number(out) if meta.name == "gsm8k" else out.strip()
        ok = pred == ref if meta.name == "gsm8k" else pred.lower() == ref.lower()
        results.append({"id": ids[i - 1], "predicted": pred, "reference": ref, "passed": bool(ok), "response": out})
    n = len(results)
    return {"kind": "qa", "runner": "native", "metrics": {"accuracy": sum(r["passed"] for r in results) / n},
            "n": n, "limit": limit, "seconds": time.perf_counter() - t0, "results": results}


def run_native(meta: BenchmarkMeta, *, encode: EncodeFn | None = None, logits_fn: LogitsFn | None = None,
               generate: GenerateFn | None = None, limit: int | None = None, progress: Any = None) -> dict[str, Any]:
    if meta.kind == "mcq":
        if encode is None or logits_fn is None:
            raise ValueError(f"native MCQ '{meta.name}' needs encode + logits")
        return run_native_mcq(meta, encode, logits_fn, limit=limit, progress=progress)
    if meta.kind == "qa":
        if generate is None:
            raise ValueError(f"native QA '{meta.name}' needs generate")
        return run_native_qa(meta, generate, limit=limit, progress=progress)
    raise ValueError(f"native benchmark '{meta.name}' has unsupported kind '{meta.kind}'")
