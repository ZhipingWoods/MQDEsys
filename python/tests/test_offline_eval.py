"""Offline benchmark registry + scorer (no transformers model needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from mqde.eval import native, offline, registry
from mqde.models.base import data_dir


def test_registry_has_the_builtin_suites_and_benchmarks() -> None:
    names = {b["name"] for b in registry.catalog()}
    assert {"wikitext2_ppl", "smoke_qa", "smoke_mcq"} <= names
    assert registry.suite("quick") == ["wikitext2_ppl", "smoke_qa"]
    assert registry.suite("smoke_mcq") == ["smoke_mcq"]
    assert registry.suite("wikitext2_ppl,smoke_mcq") == ["wikitext2_ppl", "smoke_mcq"]
    meta = registry.get("smoke_mcq")
    assert meta.kind == "mcq" and meta.path and Path(meta.path).is_file()
    assert meta.group == "冒烟检查" and meta.offline is True and meta.runner == "offline"


def test_evalscope_catalog_is_metadata_only_and_grouped() -> None:
    doc = yaml.safe_load((data_dir() / "datasets" / "evalscope_catalog.yaml").read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in doc["datasets"]}
    assert len(rows) >= 100
    assert {"gsm8k", "mmlu", "ceval", "humaneval"} <= rows.keys()
    assert rows["gsm8k"]["group"] == "数学"
    assert rows["gsm8k"]["runner"] == "evalscope"
    assert rows["gsm8k"]["offline"] is False


def test_native_benchmarks_are_registered_and_normalize_dataset_schemas() -> None:
    names = {b["name"] for b in registry.catalog()}
    assert {"gsm8k", "mmlu", "arc_challenge", "hellaswag", "winogrande"} <= names
    assert registry.get("gsm8k").runner == "native"
    arc = native.normalize_mcq(registry.get("arc_challenge"), {
        "question": "pick", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "B"})
    assert arc["answer"] == 1 and arc["choices"] == ["a", "b"]
    hs = native.normalize_mcq(registry.get("hellaswag"), {"ctx": "ctx", "endings": ["e1", "e2"], "label": "1"})
    assert hs["choices"] == ["ctx e1", "ctx e2"] and hs["answer"] == 1
    wg = native.normalize_mcq(registry.get("winogrande"), {"sentence": "a _ b", "option1": "x", "option2": "y", "answer": "2"})
    assert wg["choices"] == ["a x b", "a y b"] and wg["answer"] == 1


def test_native_mcq_and_gsm8k_run_through_the_same_logits_and_generate_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"question": "pick one", "choices": ["3", "4"], "answer": 1}]
    monkeypatch.setattr(native, "load_native_rows", lambda meta, limit=None: rows[:limit] if limit else rows)

    def encode(s: str) -> list[int]:
        return [ord(s[-1]) % 20]

    def logits(ids: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((1, ids.shape[1], 20))
        if ids.shape[1] > 1:
            out[0, -2, ord("4") % 20] = 10.0
        return out

    mcq = native.run_native(registry.get("mmlu"), encode=encode, logits_fn=logits)
    assert mcq["metrics"]["accuracy"] == 1.0

    monkeypatch.setattr(native, "load_native_rows", lambda meta, limit=None: [{"question": "1+2?", "answer": "3"}])
    qa = native.run_native(registry.get("gsm8k"), generate=lambda prompts, n: ["#### 3" for _ in prompts])
    assert qa["metrics"]["accuracy"] == 1.0


def test_run_benchmarks_dispatches_native_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "load_native_rows", lambda meta, limit=None: [{"question": "1+2?", "answer": "3"}])
    out = offline.run_benchmarks(["gsm8k"], generate=lambda prompts, n: ["#### 3" for _ in prompts])
    assert out["benchmarks"]["gsm8k"]["runner"] == "native"
    assert out["benchmarks"]["gsm8k"]["metrics"]["accuracy"] == 1.0


def test_custom_jsonl_kind_is_inferred(tmp_path: Path) -> None:
    p = tmp_path / "my.jsonl"
    p.write_text(json.dumps({"question": "q", "choices": ["a", "b"], "answer": 1}) + "\n")
    meta = registry.get(str(p))
    assert meta.kind == "mcq" and meta.path == str(p)
    q = tmp_path / "qa.jsonl"
    q.write_text(json.dumps({"prompt": "q", "answer": "a"}) + "\n")
    assert registry.get(str(q)).kind == "qa"


def test_choice_index_accepts_index_letter_and_text() -> None:
    row = {"choices": ["red", "green", "blue"], "answer": 1}
    assert offline._choice_index(row) == 1
    row["answer"] = "blue"
    assert offline._choice_index(row) == 2
    row["answer"] = "B"
    assert offline._choice_index(row) == 1
    row["answer"] = "purple"
    with pytest.raises(ValueError):
        offline._choice_index(row)


def test_mcq_scoring_picks_the_highest_logprob_choice(tmp_path: Path) -> None:
    path = tmp_path / "mcq.jsonl"
    path.write_text(json.dumps({"id": "one", "subject": "s", "question": "Pick",
                                "choices": ["A", "B"], "answer": "A"}) + "\n")
    meta = registry.BenchmarkMeta("tmp_mcq", "tmp", "mcq", path=str(path))

    def encode(s: str) -> list[int]:
        return [1] if s.endswith("A") else [2] if s.endswith("B") else [0]

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(1, ids.shape[1], 8)
        out[0, 0, 1] = 20.0  # after the prompt, "1" (choice A) is almost certain
        return out

    r = offline.run_mcq(meta, encode, logits_fn)
    assert r["metrics"]["accuracy"] == 1.0
    assert r["categories"]["s"]["accuracy"] == 1.0
    assert r["results"][0]["predicted"] == 0


def test_qa_scoring_uses_the_existing_custom_metrics(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps({"id": "q1", "prompt": "capital?", "answer": "Paris",
                                "metric": "contains"}) + "\n")
    meta = registry.BenchmarkMeta("tmp_qa", "tmp", "qa", path=str(path))
    r = offline.run_qa(meta, lambda prompts, n: ["The answer is Paris." for _ in prompts])
    assert r["metrics"]["accuracy"] == 1.0


def test_run_benchmarks_reports_missing_adapters_instead_of_failing() -> None:
    out = offline.run_benchmarks(["smoke_mcq", "smoke_qa"])
    assert out["benchmarks"]["smoke_mcq"]["status"] == "skipped"
    assert out["benchmarks"]["smoke_qa"]["status"] == "skipped"


def test_ppl_benchmark_uses_the_shared_perplexity_protocol() -> None:
    ids = torch.arange(32, dtype=torch.long)

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, x.shape[1], 32)

    r = offline.run_ppl(ids, logits_fn, seq_len=16, max_windows=2)
    assert r["kind"] == "ppl" and r["windows"] == 2
    assert r["metrics"]["ppl"] == pytest.approx(32.0)
