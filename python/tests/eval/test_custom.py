"""Task-level JSONL checks (T-0801, no third-party framework): file loading, every metric, the HTTP path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mock_backend

from mqde.eval.custom import evaluate, load_cases, rouge_l, score

SMOKE = Path(__file__).resolve().parents[3] / "data" / "datasets" / "smoke_qa.jsonl"


def case(answer: object, metric: str, **kw: object) -> dict:
    return {"id": "c", "prompt": "q", "answer": answer, "metric": metric, **kw}


def test_the_shipped_smoke_set_loads_and_its_answers_are_self_consistent() -> None:
    cases = load_cases(SMOKE)
    assert len(cases) >= 10 and len({c["id"] for c in cases}) == len(cases)
    assert {c["metric"] for c in cases} <= {"exact_match", "contains", "regex"}
    # a reply that is exactly the (first) answer passes its own case; a reply with the wrong content does not
    for c in cases:
        ans = c["answer"][0] if isinstance(c["answer"], list) else c["answer"]
        right = ans.replace("\\b", "") if c["metric"] == "regex" else ans
        assert score(c, right), c["id"]
        assert not score(c, "I do not know"), c["id"]


@pytest.mark.parametrize(("c", "resp", "ok"), [
    (case("Paris", "exact_match"), "  paris. ", True),
    (case("Paris", "exact_match"), "It is Paris", False),
    (case("Paris", "contains"), "The capital is PARIS.", True),
    (case("Paris", "contains"), "Lyon", False),
    (case(r"\b42\b", "regex"), "the answer: 42", True),
    (case(r"\b42\b", "regex"), "142", False),
    (case(["12", "twelve"], "contains"), "There are twelve", True),
    (case("the cat sat on the mat", "rouge_l"), "a cat sat on the mat", True),
    (case("the cat sat on the mat", "rouge_l"), "dogs run fast", False),
    (case("the cat sat on the mat", "rouge_l", threshold=0.99), "a cat sat on the mat", False),
])
def test_metrics(c: dict, resp: str, ok: bool) -> None:
    assert score(c, resp) is ok


def test_rouge_l_values() -> None:
    assert rouge_l("a b c", "a b c") == 1.0
    assert rouge_l("a b c", "x y z") == 0.0 and rouge_l("", "a") == 0.0
    assert rouge_l("a b c d", "a c") == pytest.approx(2 * (2 / 4) * (2 / 2) / ((2 / 4) + (2 / 2)))  # LCS 2: P .5, R 1


def test_load_errors_name_the_line(tmp_path: Path) -> None:
    for text, msg in [('{"prompt": "q"}', "missing 'answer'"), ("not json", "not JSON"),
                      ('{"prompt": "q", "answer": "a", "metric": "bleu"}', "unknown metric 'bleu'"),
                      ('{"prompt": "q", "answer": "(", "metric": "regex"}', "bad regex"), ("# only a comment\n", "no cases")]:
        p = tmp_path / "x.jsonl"
        p.write_text(text)
        with pytest.raises(ValueError, match=msg):
            load_cases(p)
    p = tmp_path / "y.jsonl"
    p.write_text('# c\n\n{"prompt": "q", "answer": "a"}\n')
    (c,) = load_cases(p)
    assert c["metric"] == "contains" and c["id"] == "case-3"  # defaults; the id carries the line number


def test_evaluate_with_an_injected_asker_counts_passes_and_errors() -> None:
    cases = [{**case("Paris", "contains"), "id": "a"}, {**case("42", "exact_match"), "id": "b"}, {**case("x", "contains"), "id": "boom"}]

    def ask(c: dict) -> str:
        if c["id"] == "boom":
            raise TimeoutError("slow")
        return {"a": "It's Paris", "b": "forty-two"}[c["id"]]

    r = evaluate(0, cases, ask=ask, concurrency=2)
    assert (r["total"], r["passed"], r["errors"]) == (3, 1, 1) and r["accuracy"] == pytest.approx(1 / 3)
    assert r["by_metric"] == {"contains": {"total": 2, "passed": 1}, "exact_match": {"total": 1, "passed": 0}}
    assert [x["id"] for x in r["results"]] == ["a", "b", "boom"]  # input order is kept
    assert "TimeoutError: slow" in r["results"][2]["error"]


def test_evaluate_over_http_against_the_mock(tmp_path: Path) -> None:
    """The mock answers every chat request with 'Paris': the Paris case passes, an unrelated one does not."""
    cases = [{**case("Paris", "contains"), "id": "p"}, {**case("Tokyo", "contains"), "id": "t"}]
    with mock_backend.vllm(tmp_path) as srv:
        r = evaluate(srv.port, cases)
    assert [x["passed"] for x in r["results"]] == [True, False] and r["errors"] == 0
    assert json.dumps(r)  # JSON-serialisable: it goes into report.json


def test_unreachable_server_is_a_failed_run_not_a_crash() -> None:
    r = evaluate(1, [case("a", "contains")], model="m", timeout=2)
    assert r["passed"] == 0 and r["errors"] == 1
