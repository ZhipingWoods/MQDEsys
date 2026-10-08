"""Task-level checks from a JSONL file (T-0801, the part that needs no third-party framework).

Each line is one case::

    {"id": "capital-fr", "prompt": "What is the capital of France? Answer in one word.",
     "answer": "Paris", "metric": "contains"}

``metric`` is one of ``exact_match`` (after trimming whitespace and a trailing period, case-insensitive),
``contains`` (case-insensitive substring), ``regex`` (``re.search`` of ``answer`` as a pattern), ``rouge_l`` (LCS
based F1 over whitespace tokens; passes at ``threshold``, default 0.5). ``answer`` may be a list: the case passes if
any item matches. Optional per-case ``max_tokens``.

``evaluate`` sends every case to an OpenAI-compatible chat endpoint (vLLM, llama-server) at temperature 0 and returns
the pass rate with every response, so a failure can be read, not just counted. This is a smoke check of the deployed
model, not a replacement for MMLU / GSM8K (see docs/adr/0002-accuracy-evaluation-frameworks.md).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mqde.backend.base import HOST, http_json

METRICS = ("exact_match", "contains", "regex", "rouge_l")


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from e
        for k in ("prompt", "answer"):
            if k not in c:
                raise ValueError(f"{path}:{n}: missing '{k}'")
        c.setdefault("metric", "contains")
        if c["metric"] not in METRICS:
            raise ValueError(f"{path}:{n}: unknown metric '{c['metric']}' (known: {', '.join(METRICS)})")
        if c["metric"] == "regex":
            for a in _answers(c):
                try:
                    re.compile(a)
                except re.error as e:
                    raise ValueError(f"{path}:{n}: bad regex '{a}' ({e})") from e
        c.setdefault("id", f"case-{n}")
        cases.append(c)
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def _answers(c: dict[str, Any]) -> list[str]:
    a = c["answer"]
    return [str(x) for x in a] if isinstance(a, list) else [str(a)]


def _norm(s: str) -> str:
    return s.strip().rstrip(".").strip().lower()


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 over whitespace tokens (longest common subsequence)."""
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(r, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def score(case: dict[str, Any], response: str) -> bool:
    m = case["metric"]
    for a in _answers(case):
        if m == "exact_match" and _norm(response) == _norm(a):
            return True
        if m == "contains" and a.lower() in response.lower():
            return True
        if m == "regex" and re.search(a, response):
            return True
        if m == "rouge_l" and rouge_l(response, a) >= float(case.get("threshold", 0.5)):
            return True
    return False


def evaluate(port: int, cases: list[dict[str, Any]], *, model: str | None = None, concurrency: int = 4,
             timeout: float = 120, host: str = HOST, ask: Callable[[dict[str, Any]], str] | None = None) -> dict[str, Any]:
    """Run every case; ``ask`` replaces the HTTP call (used by tests)."""
    if ask is None:
        if model is None:
            data = (http_json(port, "GET", "/v1/models", timeout=30, host=host) or {}).get("data") or []
            model = data[0]["id"] if data else None

        def ask(c: dict[str, Any]) -> str:
            body: dict[str, Any] = {"messages": [{"role": "user", "content": c["prompt"]}], "temperature": 0,
                                    "max_tokens": int(c.get("max_tokens", 64))}
            if model:
                body["model"] = model
            r = http_json(port, "POST", "/v1/chat/completions", body, timeout=timeout, host=host)
            return str(r["choices"][0]["message"]["content"] or "")

    def one(c: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = ask(c)
        except Exception as e:  # a request that fails is a failed case, not a crashed run
            return {"id": c["id"], "metric": c["metric"], "passed": False, "response": "", "error": f"{type(e).__name__}: {e}"}
        return {"id": c["id"], "metric": c["metric"], "passed": score(c, resp), "response": resp}

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(one, cases))
    passed = sum(r["passed"] for r in results)
    by: dict[str, list[bool]] = {}
    for r in results:
        by.setdefault(r["metric"], []).append(r["passed"])
    return {"total": len(results), "passed": passed, "accuracy": passed / len(results),
            "errors": sum(1 for r in results if "error" in r),
            "by_metric": {k: {"total": len(v), "passed": sum(v)} for k, v in by.items()}, "results": results}
