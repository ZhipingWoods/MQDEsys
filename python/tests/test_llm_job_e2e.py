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
import yaml

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
                        lambda tok, n, s, seed=0, **kw: [torch.randint(0, VOCAB, (1, s), generator=g) for _ in range(n)])
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
    from mqde.report.schema import validate_report

    assert validate_report(json.loads((tmp_path / "j" / "report.json").read_text())) == []  # the schema describes what jobs really write
    sections = {"en": ("Quantization evaluation report", "Export and fake/real consistency"), "zh": ("量化评估报告", "导出与一致性")}
    for lang, (title, export_title) in sections.items():
        html = (tmp_path / "j" / f"report.{lang}.html").read_text(encoding="utf-8")
        assert title in html and export_title in html
        assert f'{r["summary"]["ppl"]:.3f}' in html and "compressed-tensors" in html


def test_quantize_only_skips_every_eval_export_and_deploy_stage(job_env: None, tmp_path: Path) -> None:
    job = LLMJob(spec(tmp_path, quantize_only=True, export=["awq"], sensitivity="block", llamacpp=["q4_k"],
                      logits_probe={}), job_dir=tmp_path / "j")
    rep = job.run()
    assert "error" not in rep
    stages = [s["stage"] for s in rep["stages"]]
    # the stages that need a baseline/reference (eval, export's real_check, llama.cpp deploy) never ran
    for skipped_stage in ("eval_baseline", "eval_fake_wo_kv", "eval_logits", "sensitivity", "export",
                          "real_check", "eval_real", "deploy_llamacpp", "lm_eval_baseline"):
        assert skipped_stage not in stages, f"{skipped_stage} ran despite quantize_only"
    assert stages == ["validate", "inspect", "calib_prepare", "quantize", "report"]
    assert "baseline" not in rep
    assert "llamacpp" not in rep
    r = rep["recipes"]["W4A16"]
    assert r["layers"], "the quantize-time trace is still produced"
    assert r["eval"] == {}
    assert r["summary"]["ppl"] is None and r["summary"]["ppl_delta_pct"] is None
    assert r["summary"]["worst_layers"], "worst_layers still comes from the quantize-time trace, not an eval stage"
    assert "export" not in r and "sensitivity" not in r


def test_quantize_only_still_quantizes_multiple_recipes(job_env: None, tmp_path: Path) -> None:
    job = LLMJob(spec(tmp_path, recipes=["w4a16", "w2a16"], quantize_only=True), job_dir=tmp_path / "j")
    rep = job.run()
    assert set(rep["recipes"]) == {"W4A16", "W2A16"}
    for r in rep["recipes"].values():
        assert r["layers"] and r["eval"] == {}


def test_from_yaml_parses_quant_device_and_eval_device_with_deploy_device_as_fallback(tmp_path: Path) -> None:
    base = {"metadata": {"name": "t"}, "model": {"path": "<tiny>"}, "quant": {"recipe": ["w4a16"]}}
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump({**base, "deploy": {"device": "cuda:1"}}))
    s = LLMJobSpec.from_yaml(p)
    assert s.device == s.quant_device == s.eval_device == "cuda:1", "deploy.device alone sets all three the same"

    p.write_text(yaml.safe_dump({**base, "deploy": {"device": "cpu"}, "quant_device": "cuda:0",
                                 "evaluation": {"device": "cuda:1"}}))
    s2 = LLMJobSpec.from_yaml(p)
    assert (s2.device, s2.quant_device, s2.eval_device) == ("cpu", "cuda:0", "cuda:1")

    p.write_text(yaml.safe_dump({**base, "deploy": {"device": "cpu"}, "quant_device": "cuda:0"}))
    s3 = LLMJobSpec.from_yaml(p)
    assert (s3.quant_device, s3.eval_device) == ("cuda:0", "cpu"), "eval_device with no evaluation.device falls back to deploy.device"


def test_quant_device_and_eval_device_default_to_the_shared_device_and_are_reported(job_env: None, tmp_path: Path) -> None:
    rep = LLMJob(spec(tmp_path), job_dir=tmp_path / "j").run()
    assert rep["env"]["quant_device"] == rep["env"]["eval_device"] == "cpu"


def test_hardware_and_real_device_names_are_recorded_for_cross_hardware_comparison(job_env: None, tmp_path: Path) -> None:
    rep = LLMJob(spec(tmp_path, hardware="nvidia/rtx4090"), job_dir=tmp_path / "j").run()
    assert rep["env"]["hardware"] == "nvidia/rtx4090"
    # this test runs with device="cpu" (job_env never touches torch.cuda), so the *real* device name is
    # "cpu" regardless of what hardware id was submitted -- device_name must reflect what actually ran,
    # not just echo the submitted label.
    assert rep["env"]["quant_device_name"] == rep["env"]["eval_device_name"] == "cpu"


def test_from_yaml_parses_deploy_sglang_like_deploy_vllm(tmp_path: Path) -> None:
    base = {"metadata": {"name": "t"}, "model": {"path": "<tiny>"}, "quant": {"recipe": ["w4a16"]}}
    p = tmp_path / "a.yaml"
    for dep, want in [({}, None), ({"sglang": True}, {}), ({"sglang": {"mem_fraction_static": 0.4}}, {"mem_fraction_static": 0.4}),
                      ({"sglang": False}, None)]:
        p.write_text(yaml.safe_dump({**base, "deploy": dep}))
        got = LLMJobSpec.from_yaml(p)
        assert got.sglang == want and got.vllm is None, dep


def test_sglang_stage_is_skipped_with_a_reason_when_the_environment_is_missing(job_env: None, tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from mqde.backend import envs

    monkeypatch.delenv("MQDE_SGLANG_PYTHON", raising=False)
    monkeypatch.setenv("MQDE_ENVS_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(envs, "repo_root", lambda: tmp_path)
    job = LLMJob(spec(tmp_path, sglang={}), job_dir=tmp_path / "j")
    r = job._sglang_serve(tmp_path, torch.zeros(8, dtype=torch.long), 10.0)
    assert r["status"] == "skipped" and "MQDE_SGLANG_PYTHON" in r["reason"]


def test_sglang_stage_serves_smokes_loads_and_scores_through_the_mock(job_env: None, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole stage against the mock SGLang server: served PPL is exp(-FAKE_LOGPROB) and is compared with the real PPL."""
    import math

    sys.path.insert(0, str(Path(__file__).parent))
    import mock_backend

    monkeypatch.setenv("MQDE_SGLANG_PYTHON", str(mock_backend.wrapper(tmp_path, "python")))
    monkeypatch.setenv("FAKE_LOGPROB", "-1.5")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # the stage only gates on it; the mock needs no GPU
    ckpt = tmp_path / "ckpt-awq"
    ckpt.mkdir()
    job = LLMJob(spec(tmp_path, sglang={"requests": 4, "concurrency": 2, "prompt_len": 32, "n_predict": 4, "ppl_windows": 2}),
                 job_dir=tmp_path / "j")
    r = job._sglang_serve(ckpt, torch.arange(1000, 1000 + 4 * 64), 4.0)
    assert r["status"] == "succeeded", r
    assert r["served_ppl"] == pytest.approx(math.exp(1.5)) and r["served_windows"] == 2
    assert r["served_vs_real_ppl_pct"] == pytest.approx(100 * (math.exp(1.5) / 4.0 - 1))
    assert r["engine"]["kv_cache_tokens"] == 12345 and r["load"]["ok"] == 4
    assert "--model-path" in r["args"] and str(ckpt) in r["args"]


def test_group_size_override_applies_to_per_group_recipes_and_is_a_logged_noop_for_others(job_env: None, tmp_path: Path) -> None:
    job = LLMJob(spec(tmp_path, recipes=["w4a16", "w8a16"], group_size_override=64), job_dir=tmp_path / "j")
    rep = job.run()
    assert rep["recipes"]["W4A16"]["weight"]["group_size"] == 64, "w4a16 is per_group (default 128): the override must win"
    assert "group_size" not in rep["recipes"]["W8A16"]["weight"], "w8a16 is per_channel: no group_size concept, override is a no-op"
    log = (tmp_path / "j" / "events.jsonl").read_text()
    assert "group_size override: W4A16 128 -> 64" in log
    assert "group_size override ignored for W8A16: granularity is per_channel, not per_group" in log


def test_from_yaml_parses_group_size_override(tmp_path: Path) -> None:
    base = {"metadata": {"name": "t"}, "model": {"path": "<tiny>"}, "quant": {"recipe": ["w4a16"]}}
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(base))
    assert LLMJobSpec.from_yaml(p).group_size_override is None, "absent quant.group_size must not turn into some default override"
    p.write_text(yaml.safe_dump({**base, "quant": {"recipe": ["w4a16"], "group_size": 256}}))
    assert LLMJobSpec.from_yaml(p).group_size_override == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real second device to exercise the move")
def test_quant_device_and_eval_device_can_differ_and_the_job_still_produces_a_consistent_report(job_env: None, tmp_path: Path) -> None:
    """Real cross-device round trip (see test_blockwise_llm.py's unit-level version of the same sequencing):
    quantize on cpu, move to cuda:0 for eval, move back before restore() so the detached FP originals
    (pinned on cpu since capture) match the live model's device again. Two recipes, not one: skipping the
    move-back is invisible on a single recipe (nothing forwards through the model again after its only
    restore()) and only breaks the *next* recipe's quantize, which forwards calibration data through a
    model now split across devices -- caught by hand by mutating moves_devices to False here and
    confirming this test (not the one-recipe version) goes from a clean pass to a device-mismatch error."""
    job = LLMJob(spec(tmp_path, recipes=["w4a16", "w2a16"], quant_device="cpu", eval_device="cuda:0"), job_dir=tmp_path / "j")
    rep = job.run()
    assert "error" not in rep
    assert rep["env"]["quant_device"] == "cpu" and rep["env"]["eval_device"] == "cuda:0"
    for name in ("W4A16", "W2A16"):
        r = rep["recipes"][name]
        assert r["summary"]["ppl"] > 0 and r["summary"]["ppl_delta_pct"] != 0, name


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


def test_logits_probe_is_off_by_default_and_reported_when_on(job_env: None, tmp_path: Path) -> None:
    off = LLMJob(spec(tmp_path), job_dir=tmp_path / "off").run()
    assert "logits_probe" not in off and "logits" not in off["recipes"]["W4A16"]
    on = LLMJob(spec(tmp_path, logits_probe={"windows": 2, "k": 16}), job_dir=tmp_path / "on").run()
    got = on["logits_probe"]
    assert got == {"windows": 2, "seq_len": 64, "k": 16, "exact": False, "reference_mib": got["reference_mib"]}
    m = on["recipes"]["W4A16"]["logits"]
    assert m["positions"] == 2 * 64 and m["k"] == 16 and m["exact"] is False
    assert 0 < m["kl"] < 5 and 0.0 <= m["top1_agreement"] <= 1.0
    stages = [s["stage"] for s in on["stages"]]
    assert stages.index("logits_reference") < stages.index("quantize") < stages.index("eval_logits")  # reference BEFORE quantization


def test_the_reference_is_captured_from_the_unquantized_model(job_env: None, tmp_path: Path) -> None:
    """If it were taken after quantization the KL of a recipe against itself would be ~0; it must not be."""
    s = spec(tmp_path, recipes=["w4a16", "w8a16"], logits_probe={"windows": 2, "k": 32, "exact": True})
    rep = LLMJob(s, job_dir=tmp_path / "j").run()
    kl4, kl8 = (rep["recipes"][r]["logits"]["kl"] for r in ("W4A16", "W8A16"))
    assert kl4 > 1e-4 and kl8 > 0
    assert kl4 > kl8  # more bits, closer to the unquantized distribution: both recipes compare against the same reference


def test_logits_probe_yaml_forms() -> None:
    from mqde.pipeline.llm_job import _opt_dict

    assert _opt_dict(None) is None and _opt_dict(False) is None
    assert _opt_dict(True) == {} and _opt_dict({"k": 8}) == {"k": 8}


def test_lm_eval_stage_records_results_skips_without_the_binary_and_never_fails_the_job(
    job_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(Path(__file__).parent / "eval"))
    from test_lm_eval_adapter import DOC, fake_cli

    job = LLMJob(spec(tmp_path), job_dir=tmp_path / "j")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setenv("MQDE_LM_EVAL_BIN", str(fake_cli(tmp_path, DOC)))
    ok = job._lm_eval(8123, ckpt, tmp_path / "lm-out", {"tasks": ["gsm8k"], "limit": 10, "num_fewshot": 5}, "the-served-name")
    assert ok["status"] == "succeeded" and ok["tasks"]["gsm8k"]["metrics"]["exact_match,strict-match"]["value"] == 0.5
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "10" and "gsm8k" in argv  # the config reaches the CLI
    # the name the server really serves under, not the directory name
    assert "model=the-served-name," in argv[argv.index("--model_args") + 1]
    monkeypatch.setenv("MQDE_LM_EVAL_BIN", str(fake_cli(tmp_path, None, code=2)))
    bad = job._lm_eval(8123, ckpt, tmp_path / "lm-out2", {}, "n")
    assert bad["status"] == "failed" and "exited with 2" in bad["error"]
    monkeypatch.setenv("MQDE_LM_EVAL_BIN", str(tmp_path / "missing"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert job._lm_eval(8123, ckpt, tmp_path / "lm-out2", {}, "n")["status"] == "skipped"


def test_vllm_context_grows_for_lm_eval_only() -> None:
    from mqde.pipeline.llm_job import served_max_len

    assert served_max_len({}, 512, 256, 64) == 1024
    assert served_max_len({}, 2048, 256, 64) == 2064, "PPL windows still decide when they are longer"
    assert served_max_len({"lm_eval": {"tasks": ["gsm8k"]}}, 512, 256, 64) == 4096, "5-shot gsm8k needs about 1.4k + 256 tokens"
    assert served_max_len({"lm_eval": {"max_model_len": 8192}}, 512, 256, 64) == 8192
    assert served_max_len({"lm_eval": {"max_model_len": 512}}, 512, 256, 64) == 1024, "never below what the other stages need"


def test_baseline_lm_eval_is_on_by_default_and_can_be_switched_off() -> None:
    from mqde.pipeline.llm_job import wants_lm_eval_baseline

    assert not wants_lm_eval_baseline(None) and not wants_lm_eval_baseline({}) and not wants_lm_eval_baseline({"lm_eval": None})
    assert wants_lm_eval_baseline({"lm_eval": {"tasks": ["gsm8k"]}})
    assert not wants_lm_eval_baseline({"lm_eval": {"tasks": ["gsm8k"], "baseline": False}})


def test_layers_carry_weight_and_activation_distributions(job_env: None, tmp_path: Path) -> None:
    rep = LLMJob(spec(tmp_path, algorithm="gptq"), job_dir=tmp_path / "j").run()
    layers = rep["recipes"]["W4A16"]["layers"]
    assert layers
    for lay in layers:
        wh, ah, ss = lay["w_hist"], lay["act_hist"], lay["w_scale_summary"]
        assert sum(wh["counts"]) == wh["n"] == lay["shape"][0] * lay["shape"][1], "the weight histogram covers the whole weight"
        assert ah is not None and sum(ah["counts"]) == ah["n"] > 0
        assert ss is not None and ss["min"] > 0 and ss["max"] >= ss["p99"] >= ss["p50"] >= ss["min"]
        # the histogram range agrees with the separately computed absmax
        assert max(ah["max"], -ah["min"]) == pytest.approx(lay["act_absmax"], rel=1e-5)
    from mqde.report.schema import validate_report

    assert validate_report(rep) == []


def test_data_free_rtn_has_weight_distributions_but_no_activation_histogram(job_env: None, tmp_path: Path) -> None:
    rep = LLMJob(spec(tmp_path), job_dir=tmp_path / "j2").run()
    for lay in rep["recipes"]["W4A16"]["layers"]:
        assert lay["w_hist"] is not None and lay["act_hist"] is None and lay["act_absmax"] is None


# ---------------------------------------------------------------- calibration data the user chose
class _CharTok:
    """One token per character (modulo the tiny model's vocabulary), so any text file tokenizes."""

    def __call__(self, text: str, return_tensors: str = "pt"):  # type: ignore[no-untyped-def]
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % VOCAB for c in text]]))


def test_the_spec_reads_the_calibration_dataset_from_yaml(tmp_path: Path) -> None:
    y = tmp_path / "j.yaml"
    y.write_text(yaml.safe_dump({"model": {"path": "<tiny>"}, "quant": {"recipe": ["w4a16"]}, "calibration": {"dataset": "/data/my_corpus", "n_samples": 8}}))
    assert LLMJobSpec.from_yaml(y).calib_dataset == "/data/my_corpus"
    y.write_text(yaml.safe_dump({"model": {"path": "<tiny>"}, "quant": {"recipe": ["w4a16"]}}))
    assert LLMJobSpec.from_yaml(y).calib_dataset == "wikitext-2-raw-v1:train", "unchanged default"


def test_the_job_calibrates_on_the_named_source_and_records_what_it_read(job_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(tok, n, s, seed=0, source="?", info=None):  # type: ignore[no-untyped-def]
        seen["source"] = source
        if info is not None:
            info.update(dataset=source, files=3, chars=1234, tokens=987, truncated=False)
        return [torch.randint(0, VOCAB, (1, s)) for _ in range(n)]

    monkeypatch.setattr(llm_job, "calib_batches", fake)
    rep = LLMJob(spec(tmp_path, calib_dataset="/data/corpus"), job_dir=tmp_path / "j").run()
    assert seen["source"] == "/data/corpus"
    assert rep["data"]["calib"] == {"dataset": "/data/corpus", "files": 3, "chars": 1234, "tokens": 987, "truncated": False, "n": 4, "seq_len": 32}


def test_the_default_calibration_is_still_reported_as_wikitext(job_env: None, tmp_path: Path) -> None:
    rep = LLMJob(spec(tmp_path), job_dir=tmp_path / "j").run()
    assert rep["data"]["calib"]["dataset"] == "wikitext-2 train"


def test_a_real_text_file_changes_what_gptq_learns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the real loader on two different corpora, the real GPTQ: the quantized model must depend on the calibration text
    def fake_hf(path: str, device: str = "cpu", **kw: object):  # type: ignore[no-untyped-def]
        ad = _tiny.tiny()
        ad.path = path
        ad.tokenizer = _CharTok()
        return ad

    g = torch.Generator().manual_seed(7)
    monkeypatch.setattr(llm_job, "HFCausalLM", fake_hf)
    monkeypatch.setattr(llm_job, "eval_ids", lambda tok: torch.randint(0, VOCAB, (4096,), generator=torch.Generator().manual_seed(9)))
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("the quick brown fox jumps over the lazy dog. " * 80)
    b.write_text("".join(chr(33 + (i * 7919) % 90) for i in range(3600)))
    ppl = {}
    for name, f in (("a", a), ("b", b)):
        rep = LLMJob(spec(tmp_path, algorithm="gptq", calib_dataset=str(f), calib_seq_len=64), job_dir=tmp_path / f"j{name}").run()
        assert rep["data"]["calib"]["dataset"] == str(f) and rep["data"]["calib"]["files"] == 1
        ppl[name] = rep["recipes"]["W4A16"]["eval"]["fake_quant_wo_kv"]["ppl"]
    assert ppl["a"] != ppl["b"], "GPTQ gave the same result on two different calibration texts: the source was ignored"


def test_a_missing_calibration_path_fails_the_job_at_calib_prepare_and_names_the_path(job_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mqde.data import text as T

    monkeypatch.setattr(llm_job, "calib_batches", T.calib_batches)
    job = LLMJob(spec(tmp_path, calib_dataset=str(tmp_path / "nope")), job_dir=tmp_path / "j")
    with pytest.raises(FileNotFoundError, match="nope"):
        job.run()
    rep = json.loads((tmp_path / "j" / "report.json").read_text())
    assert [x["stage"] for x in rep["stages"] if x["status"] == "failed"] == ["calib_prepare"]
    assert "nope" in rep["error"]
