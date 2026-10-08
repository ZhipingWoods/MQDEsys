"""LLMJob end to end on a tiny random model with synthetic token data (no checkpoint, no dataset, CPU, seconds).

It runs the real pipeline: quantize, evaluate, export, fake/real check, sensitivity, the report, and the HTML pages
written next to report.json. Stages that need external binaries (vLLM, llama.cpp) are covered by their own tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
sys.path.insert(0, str(Path(__file__).parent / "export"))

import _tiny

from mqde.pipeline import llm_job
from mqde.pipeline.llm_job import LLMJob, LLMJobSpec

VOCAB = 128


@pytest.fixture
def job_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_hf(path: str, device: str = "cpu", **kw: object):  # type: ignore[no-untyped-def]
        ad = _tiny.tiny()
        ad.path = path
        return ad

    g = torch.Generator().manual_seed(7)
    monkeypatch.setattr(llm_job, "HFCausalLM", fake_hf)
    monkeypatch.setattr(llm_job, "calib_batches",
                        lambda tok, n, s, seed=0: [torch.randint(0, VOCAB, (1, s), generator=g) for _ in range(n)])
    monkeypatch.setattr(llm_job, "eval_ids", lambda tok: torch.randint(0, VOCAB, (4096,), generator=g))


def spec(tmp_path: Path, **kw: object) -> LLMJobSpec:
    base = {"name": "tiny", "model_path": "<tiny>", "recipes": ["w4a16"], "algorithm": "rtn", "calib_samples": 4, "calib_seq_len": 32,
            "eval_seq_len": 64, "eval_windows": 4, "output_dir": str(tmp_path)}
    return LLMJobSpec(**{**base, **kw})  # type: ignore[arg-type]


def test_full_job_writes_report_and_both_html_pages(job_env: None, tmp_path: Path) -> None:
    job = LLMJob(spec(tmp_path, export=["awq", "compressed-tensors"], sensitivity="block", sensitivity_windows=2), job_dir=tmp_path / "j")
    rep = job.run()
    assert "error" not in rep
    stages = [s["stage"] for s in rep["stages"]]
    assert stages[:4] == ["validate", "inspect", "calib_prepare", "eval_baseline"] and stages[-1] == "report"
    assert all(s["status"] == "succeeded" for s in rep["stages"])
    r = rep["recipes"]["W4A16"]
    assert r["export"]["awq"]["real_check"]["codes_identical"] and r["export"]["compressed-tensors"]["real_check"]["codes_identical"]
    assert r["sensitivity"]["groups"]
    # the JSON on disk equals the in-memory report, and the HTML pages exist next to it
    assert json.loads((tmp_path / "j" / "report.json").read_text())["job_id"] == rep["job_id"]
    sections = {"en": ("Quantization evaluation report", "Export and fake/real consistency"), "zh": ("量化评估报告", "导出与一致性")}
    for lang, (title, export_title) in sections.items():
        html = (tmp_path / "j" / f"report.{lang}.html").read_text(encoding="utf-8")
        assert title in html and export_title in html
        assert f'{r["summary"]["ppl"]:.3f}' in html and "compressed-tensors" in html


def test_a_failure_in_the_html_writer_does_not_fail_the_job(job_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mqde.report.html as html_mod

    def boom(report: dict, lang: str = "en") -> str:
        raise RuntimeError("renderer broke")

    monkeypatch.setattr(html_mod, "render_html", boom)
    job = LLMJob(spec(tmp_path), job_dir=tmp_path / "j")
    rep = job.run()
    assert rep["recipes"]["W4A16"]["summary"]["ppl"] > 0 and (tmp_path / "j" / "report.json").is_file()
    assert not (tmp_path / "j" / "report.en.html").exists()
    log = (tmp_path / "j" / "events.jsonl").read_text()
    assert "report.en.html not written: RuntimeError: renderer broke" in log


def test_job_without_export_has_a_report_page_with_no_export_section(job_env: None, tmp_path: Path) -> None:
    LLMJob(spec(tmp_path), job_dir=tmp_path / "j").run()
    html = (tmp_path / "j" / "report.en.html").read_text(encoding="utf-8")
    assert "Recipes" in html and "Export and fake/real" not in html and "vLLM deployment" not in html
