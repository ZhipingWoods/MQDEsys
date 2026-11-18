"""EvalJob: evaluate an already-quantized checkpoint without quantizing or serving it again."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent / "export"))
import _tiny

from mqde.pipeline import eval_job
from mqde.pipeline.eval_job import EvalJob, EvalJobSpec, detect_format

VOCAB = 128


@pytest.fixture
def eval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_job, "load_tokenizer", lambda path: object())
    g = torch.Generator().manual_seed(11)
    monkeypatch.setattr(eval_job, "eval_ids",
                        lambda tok, **kw: torch.randint(0, VOCAB, (4096,), generator=g))
    monkeypatch.setattr(eval_job, "HFLogits",
                        lambda path, device="cpu", dtype=torch.float32, **kw: _tiny.tiny())


def checkpoint(tmp_path: Path, cfg: dict, name: str = "q4") -> Path:
    p = tmp_path / name
    p.mkdir()
    (p / "config.json").write_text(json.dumps(cfg))
    return p


def test_detect_format_reads_config_and_gguf(tmp_path: Path) -> None:
    assert detect_format(checkpoint(tmp_path, {"quantization_config": {"quant_method": "gptq"}})) == "gptq"
    p = checkpoint(tmp_path, {"quantization_config": {}}, "ct")
    assert detect_format(p) == "compressed-tensors"
    g = tmp_path / "model.gguf"
    g.write_bytes(b"")
    assert detect_format(g) == "gguf"


def test_eval_job_scores_a_quantized_checkpoint_without_quantizing(eval_env: None, tmp_path: Path) -> None:
    q = checkpoint(tmp_path, {"quantization_config": {"quant_method": "gptq"}})
    spec = EvalJobSpec(name="eval", model_path=str(tmp_path), quantized_path=str(q),
                       stages=["evaluate"], eval_seq_len=64, eval_windows=4, output_dir=str(tmp_path))
    rep = EvalJob(spec).run()
    assert rep["kind"] == "llm_eval"
    assert rep["checkpoint"]["format"] == "gptq"
    assert [s["stage"] for s in rep["stages"]] == ["validate", "eval_prepare", "eval_hf", "report"]
    assert rep["eval"]["hf"]["status"] == "succeeded"
    assert rep["eval"]["hf"]["ppl"] > 0
    assert "quantize" not in [s["stage"] for s in rep["stages"]]
    assert (tmp_path / rep["job_id"] / "report.json").is_file()


def test_eval_job_can_reuse_a_parent_export_without_a_serving_backend(eval_env: None, tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    ckpt = parent / "export" / "w4a16-awq"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "awq"}}))
    (parent / "report.json").write_text(json.dumps({
        "kind": "llm",
        "data": {"eval": {"windows": 4, "seq_len": 64}},
        "baseline": {"ppl": 10.0},
        "recipes": {"W4A16": {"algorithm": "rtn", "summary": {"ppl": 11.0},
                              "export": {"awq": {"status": "exported", "bytes": 1}}}},
    }))
    spec = EvalJobSpec(name="eval", model_path=str(tmp_path),
                       artifact={"job": "parent", "recipe": "W4A16", "format": "awq"},
                       stages=["evaluate"], eval_seq_len=64, eval_windows=4, output_dir=str(tmp_path))
    rep = EvalJob(spec).run()
    assert rep["checkpoint"]["source"] == "artifact"
    assert rep["checkpoint"]["job"] == "parent"
    assert rep["eval"]["hf"]["status"] == "succeeded"


def test_eval_job_runs_offline_benchmarks_without_a_server(eval_env: None, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    q = checkpoint(tmp_path, {"quantization_config": {"quant_method": "gptq"}})
    inner = _tiny.tiny()

    class Adapter:
        def logits(self, ids: torch.Tensor) -> torch.Tensor:
            return inner.logits(ids)

        def encode(self, text: str) -> list[int]:
            return [ord(c) % VOCAB for c in text] or [0]

        def generate(self, prompts: list[str], n: int) -> list[str]:
            return ["Paris" for _ in prompts]

    monkeypatch.setattr(eval_job, "HFLogits",
                        lambda path, device="cpu", dtype=torch.float32, **kw: Adapter())
    spec = EvalJobSpec(name="eval", model_path=str(tmp_path), quantized_path=str(q),
                       stages=["evaluate"], eval_seq_len=64, eval_windows=2,
                       benchmarks=["smoke_mcq"], benchmark_limit=2, output_dir=str(tmp_path))
    rep = EvalJob(spec).run()
    b = rep["eval"]["hf"]["benchmarks"]["benchmarks"]["smoke_mcq"]
    assert b["kind"] == "mcq"
    assert b["n"] == 2
    assert 0.0 <= b["metrics"]["accuracy"] <= 1.0
    assert b["categories"]


def test_an_unloadable_checkpoint_fails_with_an_actionable_message(eval_env: None, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    q = checkpoint(tmp_path, {"quantization_config": {"quant_method": "awq"}})

    def boom(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32, **kw: object):  # type: ignore[no-untyped-def]
        raise ImportError("autoawq is not installed")

    monkeypatch.setattr(eval_job, "HFLogits", boom)
    spec = EvalJobSpec(name="eval", model_path=str(tmp_path), quantized_path=str(q),
                       stages=["evaluate"], eval_seq_len=64, eval_windows=4, output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="could not evaluate"):
        EvalJob(spec).run()
