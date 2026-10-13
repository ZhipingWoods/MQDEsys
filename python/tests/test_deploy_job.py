"""B2: deploy / evaluate an exported checkpoint of a finished LLM job (another backend, device or eval set) without
quantizing again. Serving runs through the mock backends (tests/backend/fake_llama_server.py)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import mock_backend

from mqde.pipeline import deploy_job
from mqde.pipeline.deploy_job import DeployJob, DeployJobSpec

PARENT = "20261009-225558-b43b47"
TOKENIZER_FROM: list[str] = []


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A finished parent LLM job with an exported awq checkpoint and a GGUF, as llm_job lays them out."""
    r = tmp_path / "runs"
    p = r / PARENT
    (p / "export" / "w4a16-awq").mkdir(parents=True)
    (p / "export" / "w4a16-awq" / "config.json").write_text("{}")
    (p / "export" / "gguf").mkdir()
    (p / "export" / "gguf" / "model-q4_k.gguf").write_bytes(b"GGUF")
    (p / "report.json").write_text(json.dumps({
        "kind": "llm", "job_id": PARENT, "spec": {"model_path": "/models/qwen", "name": "parent"}, "baseline": {"ppl": 3.5},
        "stages": [{"stage": "report", "status": "succeeded"}],
        "recipes": {"W4A16": {"recipe": "W4A16", "algorithm": "gptq", "summary": {"ppl": 3.9, "ppl_delta_pct": 11.4},
                              "export": {"awq": {"status": "exported", "real_ppl": 4.0, "bytes": 449_000_000},
                                         "gptq": {"status": "unsupported", "problems": []}}}},
        "llamacpp": {"results": {"q4_k": {"type": "q4_k", "ppl": 14.7, "bytes": 4}}},
    }))
    monkeypatch.setattr(deploy_job, "load_tokenizer", lambda path: TOKENIZER_FROM.append(str(path)))
    monkeypatch.setattr(deploy_job, "eval_ids", lambda tok: torch.arange(1000, 1000 + 4 * 64))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # the stages only gate on it; the mocks need no GPU
    monkeypatch.setenv("FAKE_LOGPROB", "-1.5")
    return r


def spec(runs: Path, **kw: object) -> DeployJobSpec:
    base: dict = {"name": "derived", "model_path": "/models/qwen", "artifact": {"job": PARENT, "recipe": "W4A16", "format": "awq"},
                  "eval_seq_len": 64, "eval_windows": 2, "output_dir": str(runs)}
    return DeployJobSpec(**{**base, **kw})


def serve_cfg() -> dict:
    return {"requests": 4, "concurrency": 2, "prompt_len": 32, "n_predict": 4, "ppl_windows": 2}


def test_sglang_serves_the_parents_checkpoint_and_compares_with_its_real_weights(runs: Path, tmp_path: Path,
                                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(mock_backend.wrapper(tmp_path, "python")))
    before = sorted(p.name for p in (runs / PARENT / "export").iterdir())
    rep = DeployJob(spec(runs, sglang=serve_cfg()), job_id="child").run()
    assert rep["kind"] == "llm_deploy" and rep["verdict"] == "succeeded", rep.get("error")
    assert [s["stage"] for s in rep["stages"]] == ["validate", "eval_prepare", "deploy_sglang", "report"]
    a = rep["artifact"]
    assert a["path"] == str(runs / PARENT / "export" / "w4a16-awq") and a["parent_real_ppl"] == 4.0 and a["parent_baseline_ppl"] == 3.5
    s = rep["deploy"]["sglang"]
    assert s["status"] == "succeeded" and s["served_ppl"] == pytest.approx(math.exp(1.5))
    assert s["served_vs_real_ppl_pct"] == pytest.approx(100 * (math.exp(1.5) / 4.0 - 1))
    assert sorted(p.name for p in (runs / PARENT / "export").iterdir()) == before, "nothing is written into the parent job"
    assert list((runs / "child" / "serve").glob("sglang-*.log"))


def test_the_tokenizer_comes_from_the_checkpoint_or_for_a_gguf_from_the_original_model(runs: Path, tmp_path: Path,
                                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(mock_backend.wrapper(tmp_path, "python")))
    TOKENIZER_FROM.clear()
    DeployJob(spec(runs, sglang=serve_cfg()), job_id="a").run()
    monkeypatch.setattr(deploy_job, "perplexity", lambda *a, **k: {"ppl": 15.0})
    monkeypatch.setattr(deploy_job, "bench", lambda *a, **k: {})
    DeployJob(spec(runs, artifact={"job": PARENT, "format": "gguf", "type": "q4_k"}, llamacpp={"serve": None}), job_id="b").run()
    assert TOKENIZER_FROM == [str(runs / PARENT / "export" / "w4a16-awq"), "/models/qwen"]


def test_vllm_through_the_mock(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_VLLM_BIN", str(mock_backend.wrapper(tmp_path, "vllm")))
    rep = DeployJob(spec(runs, vllm=serve_cfg()), job_id="child").run()
    assert rep["verdict"] == "succeeded", rep.get("error")
    assert rep["deploy"]["vllm"]["status"] == "succeeded" and rep["deploy"]["vllm"]["load"]["ok"] == 4
    assert [s["stage"] for s in rep["stages"]] == ["validate", "eval_prepare", "deploy_vllm", "report"]


def test_gguf_on_llama_cpp_runs_perplexity_bench_and_the_server(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def ppl(path, text, ctx, chunks, threads, gpu_layers=None):  # type: ignore[no-untyped-def]
        seen["ppl"] = (Path(path).name, ctx, chunks, gpu_layers)
        return {"ppl": 15.0, "chunks": chunks}

    monkeypatch.setattr(deploy_job, "perplexity", ppl)
    monkeypatch.setattr(deploy_job, "bench", lambda *a, **k: {"backend": "CPU", "pp_tokens_per_s": 100.0, "tg_tokens_per_s": 20.0})
    monkeypatch.setenv("MQDE_LLAMA_SERVER_BIN", str(mock_backend.wrapper(tmp_path, "llama-server")))
    art = {"job": PARENT, "recipe": "", "format": "gguf", "type": "q4_k"}
    rep = DeployJob(spec(runs, artifact=art, llamacpp={"gpu_layers": 0, "serve": {"concurrency": 2, "requests": 4, "prompt_len": 32, "n_predict": 4}}),
                    job_id="child").run()
    assert rep["verdict"] == "succeeded", rep.get("error")
    assert seen["ppl"] == ("model-q4_k.gguf", 64, 2, 0), "the CPU run of the parent's q4_k, on this job's eval settings"
    l = rep["deploy"]["llamacpp"]
    assert l["ppl"] == 15.0 and l["parent_ppl"] == 14.7 and l["ppl_vs_parent_pct"] == pytest.approx(100 * (15.0 / 14.7 - 1))
    assert l["tg_tokens_per_s"] == 20.0 and l["serve"]["status"] in ("succeeded", "degraded")


@pytest.mark.parametrize(("art", "msg"), [
    ({"job": "nope", "recipe": "W4A16", "format": "awq"}, "not found"),
    ({"job": PARENT, "recipe": "W4A16", "format": "gptq"}, "unsupported"),           # the parent could not export it
    ({"job": PARENT, "recipe": "W8A16", "format": "awq"}, "W8A16"),                  # never quantized there
    ({"job": f"../{PARENT}", "recipe": "W4A16", "format": "awq"}, "job id"),         # no path tricks
    ({"job": PARENT, "recipe": "../W4A16", "format": "awq"}, "recipe"),
    ({"job": PARENT, "recipe": "W4A16", "format": "safetensors"}, "format"),
    ({"job": PARENT, "recipe": "", "format": "gguf", "type": "q2_k"}, "q2_k"),        # no such GGUF in the parent
])
def test_a_bad_artifact_fails_the_job_with_a_reason(runs: Path, art: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        DeployJob(spec(runs, artifact=art, sglang=serve_cfg()), job_id="child").run()
    rep = json.loads((runs / "child" / "report.json").read_text())
    assert rep["verdict"] == "failed" and msg in rep["error"]


def test_a_format_the_backend_cannot_load_is_refused(runs: Path) -> None:
    with pytest.raises(ValueError, match="llama.cpp"):
        DeployJob(spec(runs, llamacpp={}), job_id="child").run()  # an awq checkpoint on llama.cpp
    with pytest.raises(ValueError, match="GGUF"):
        DeployJob(spec(runs, artifact={"job": PARENT, "format": "gguf", "type": "q4_k"}, vllm={}), job_id="child3").run()
    with pytest.raises(ValueError, match="no serving backend"):
        DeployJob(spec(runs), job_id="child2").run()


def test_from_yaml_reads_the_wizards_deploy_job(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump({
        "metadata": {"name": "d", "parent_job": PARENT}, "model": {"type": "llm", "path": "/m"},
        "artifact": {"job": PARENT, "recipe": "W4A16", "format": "awq"},
        "evaluation": {"dataset": "wikitext-2-raw-v1:test", "seq_len": 256, "max_windows": 10},
        "deploy": {"device": "gpu", "sglang": {"mem_fraction_static": 0.4}}, "output_dir": "/runs"}))
    s = DeployJobSpec.from_yaml(p)
    assert (s.name, s.model_path, s.artifact["format"], s.eval_seq_len, s.eval_windows, s.sglang, s.vllm, s.output_dir) == \
        ("d", "/m", "awq", 256, 10, {"mem_fraction_static": 0.4}, None, "/runs")
    assert (s.device, s.parent_job) == ("cuda:0", PARENT)
    d = yaml.safe_load(p.read_text())
    d["deploy"] = {"device": "cpu", "llamacpp": {"gpu_layers": 0}}
    p.write_text(yaml.safe_dump(d))
    s = DeployJobSpec.from_yaml(p)
    assert s.device == "cpu" and s.llamacpp_gpu_layers == 0


def test_a_concurrency_sweep_is_run_against_the_served_checkpoint_when_asked(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(mock_backend.wrapper(tmp_path, "python")))
    cfg = {**serve_cfg(), "sweep": {"levels": [4, 1, 2], "requests_per_level": 2, "max_ttft_p99_ms": 5000}}
    rep = DeployJob(spec(runs, sglang=cfg), job_id="child").run()
    s = rep["deploy"]["sglang"]["sweep"]
    assert [p["concurrency"] for p in s["points"]] == [1, 2, 4] and s["peak"]["concurrency"] == 4
    assert s["saturation"]["reason"] in ("not-reached", "throughput") and s["max_ttft_p99_ms"] == 5000
    assert all(p["ok"] > 0 and p["itl_ms"]["p99"] > 0 for p in s["points"])
    off = DeployJob(spec(runs, sglang=serve_cfg()), job_id="child2").run()["deploy"]["sglang"]
    assert off["status"] == "succeeded" and "sweep" not in off, "off unless asked for (and its absence is not a crash)"
    ev = (runs / "child" / "events.jsonl").read_text()
    assert '"sweep.c1.tps"' in ev and '"sweep.c4.tps"' in ev, "each level's throughput is a metric event for the live log"


def test_a_sweep_with_no_levels_given_uses_the_default_ladder(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(mock_backend.wrapper(tmp_path, "python")))
    rep = DeployJob(spec(runs, sglang={**serve_cfg(), "sweep": {"requests_per_level": 2}}), job_id="child").run()
    assert [p["concurrency"] for p in rep["deploy"]["sglang"]["sweep"]["points"]] == [1, 2, 4, 8, 16]


def test_a_failing_sweep_does_not_throw_away_the_rest_of_the_serving_result(runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mqde.eval import loadtest

    monkeypatch.setenv("MQDE_VLLM_BIN", str(mock_backend.wrapper(tmp_path, "vllm")))
    monkeypatch.setattr(loadtest, "sweep", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sweep exploded")))
    rep = DeployJob(spec(runs, vllm={**serve_cfg(), "sweep": {"levels": [1, 2]}}), job_id="child").run()
    v = rep["deploy"]["vllm"]
    assert v["status"] == "succeeded" and v["served_ppl"] > 0 and v["sweep"] == {"error": "RuntimeError: sweep exploded"}
