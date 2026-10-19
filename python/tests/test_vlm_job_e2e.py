"""The VLM job end to end on a tiny stand-in model (CPU, seconds): real engine, real probe, real report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from vlm_fakes import FakeAdapter

from mqde.pipeline.vlm_job import VLMJob, VLMJobSpec, answer_matches


def spec(tmp_path: Path, **kw: object) -> VLMJobSpec:
    base = {
        "name": "tiny-vlm",
        "model_path": "<tiny>",
        "recipes": ["w4a16"],
        "algorithm": "rtn",
        "calib_samples": 4,
        "eval_samples": 6,
        "max_new_tokens": 4,
        "device": "cpu",
        "output_dir": str(tmp_path),
    }
    return VLMJobSpec(**{**base, **kw})  # type: ignore[arg-type]


def run(tmp_path: Path, **kw: object) -> tuple[dict, FakeAdapter]:
    holder: list[FakeAdapter] = []

    def factory(s: VLMJobSpec) -> FakeAdapter:
        holder.append(FakeAdapter())
        return holder[0]

    rep = VLMJob(spec(tmp_path, **kw), job_id="j", adapter_factory=factory).run()
    return rep, holder[0]


def test_full_job_reports_error_attribution_and_generation_quality(tmp_path: Path) -> None:
    rep, _ = run(tmp_path)
    assert rep["kind"] == "vlm" and rep["verdict"] == "succeeded"
    stages = [s["stage"] for s in rep["stages"]]
    assert stages == ["validate", "inspect", "calib_prepare", "reference", "attribution", "quantize", "eval_quantized", "report"]
    base, r = rep["baseline"], rep["recipes"]["W4A16"]
    assert base["answer_ppl"] > 0 and base["n_samples"] == 6 and base["n_answer_tokens"] >= 6
    s = r["summary"]
    assert s["kl"] > 0 and 0 <= s["top1_agreement"] <= 1 and s["answer_ppl"] > 0
    assert s["answer_ppl_delta_pct"] == pytest.approx(100 * (s["answer_ppl"] / base["answer_ppl"] - 1))
    assert 0 <= s["generation_exact_match"] <= 1 and 0 <= s["generation_prefix_match"] <= 1
    assert r["layers"] and {t["component"] for t in r["layers"]} == {"vision", "merger", "language"}
    assert json.loads((tmp_path / "j" / "report.json").read_text())["job_id"] == "j"


def test_attribution_quantizes_one_modality_at_a_time_and_names_the_worst(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, recipes=["w2a16"])
    rows = {a["modality"]: a for a in rep["recipes"]["W2A16"]["attribution"]}
    assert set(rows) == {"vision", "language"} and all(a["kl"] > 0 and a["units"] > 0 for a in rows.values())
    assert rep["recipes"]["W2A16"]["summary"]["worst_stage"] == max(rows, key=lambda m: rows[m]["kl"])
    full = rep["recipes"]["W2A16"]["summary"]["kl"]
    assert full >= max(a["kl"] for a in rows.values()) * 0.5, "quantizing everything is not much better than its worst part"


def test_more_bits_means_less_distribution_shift(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, recipes=["w8a16", "w4a16", "w2a16"], attribution=False)
    kl = [rep["recipes"][n]["summary"]["kl"] for n in ("W8A16", "W4A16", "W2A16")]
    assert kl[0] < kl[1] < kl[2], kl


def test_the_model_is_restored_after_every_recipe(tmp_path: Path) -> None:
    rep, ad = run(tmp_path, recipes=["w4a16", "w2a16"])
    from mqde.quant.modules import FakeQuantLinear

    assert not [m for m in ad.model.modules() if isinstance(m, FakeQuantLinear)]
    # the second recipe was measured against the same unquantized reference, not against the first one's leftovers
    again, _ = run(tmp_path / "again", recipes=["w2a16"])
    assert again["recipes"]["W2A16"]["summary"]["kl"] == pytest.approx(rep["recipes"]["W2A16"]["summary"]["kl"], rel=1e-6)


def test_quantize_only_skips_reference_attribution_and_evaluation(tmp_path: Path) -> None:
    rep, ad = run(tmp_path, quantize_only=True)
    assert [s["stage"] for s in rep["stages"]] == ["validate", "inspect", "calib_prepare", "quantize", "report"]
    assert "baseline" not in rep
    r = rep["recipes"]["W4A16"]
    assert r["layers"] and r["eval"] == {} and r["summary"]["kl"] is None and r["summary"]["worst_layers"]
    from mqde.quant.modules import FakeQuantLinear

    assert not [m for m in ad.model.modules() if isinstance(m, FakeQuantLinear)]


def test_attribution_can_be_switched_off(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, attribution=False)
    assert "attribution" not in [s["stage"] for s in rep["stages"]] and "attribution" not in rep["recipes"]["W4A16"]


def test_ground_truth_accuracy_exists_for_synthetic_questions(tmp_path: Path) -> None:
    rep, _ = run(tmp_path)
    assert 0 <= rep["baseline"]["gt_accuracy"] <= 1 and rep["baseline"]["gt_samples"] == 6
    assert rep["recipes"]["W4A16"]["summary"]["gt_accuracy"] is not None


def test_a_group_size_override_is_applied_and_logged(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, group_size_override=64, attribution=False)
    assert rep["recipes"]["W4A16"]["weight"]["group_size"] == 64
    assert "group_size override: W4A16 128 -> 64" in (tmp_path / "j" / "events.jsonl").read_text()


def test_a_failure_marks_the_job_failed_and_still_writes_the_report(tmp_path: Path) -> None:
    def boom(s: VLMJobSpec) -> FakeAdapter:
        raise RuntimeError("model would not load")

    with pytest.raises(RuntimeError, match="would not load"):
        VLMJob(spec(tmp_path), job_id="bad", adapter_factory=boom).run()
    rep = json.loads((tmp_path / "bad" / "report.json").read_text())
    assert rep["verdict"] == "failed" and "would not load" in rep["error"]


def test_only_answer_positions_are_scored_not_the_one_after_the_answer(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, attribution=False)
    assert rep["recipes"]["W4A16"]["eval"]["positions"] == rep["baseline"]["n_answer_tokens"]


def test_a_recipe_that_quantizes_nothing_is_skipped_like_in_the_other_jobs(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, recipes=["bf16", "w4a16"], attribution=False)
    assert list(rep["recipes"]) == ["W4A16"]


def test_a_group_size_override_leaves_per_channel_recipes_alone(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, recipes=["w8a16"], group_size_override=64, attribution=False)
    assert rep["recipes"]["W8A16"]["weight"]["granularity"] == "per_channel"
    assert rep["recipes"]["W8A16"]["weight"].get("group_size") != 64
    assert "group_size override ignored for W8A16" in (tmp_path / "j" / "events.jsonl").read_text()


class Scripted(FakeAdapter):
    """Reference answers [5,6,7,8] for every sample; the quantized model then 'answers' with a scripted list per sample."""

    REF: ClassVar[list[int]] = [5, 6, 7, 8]
    QUANT: ClassVar[list[list[int]]] = [[5, 6, 7, 8], [5, 6, 9, 9], [9, 6, 7, 8], [5, 6, 7], [5, 6, 7, 8, 9], [5, 6, 7, 9]]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, inputs: dict, max_new_tokens: int) -> list[int]:
        self.calls += 1
        return list(self.REF) if self.calls <= 6 else list(self.QUANT[self.calls - 7])


def test_generation_agreement_is_exact_match_and_fraction_of_the_reference_prefix(tmp_path: Path) -> None:
    rep = VLMJob(spec(tmp_path, attribution=False), job_id="g", adapter_factory=lambda s: Scripted()).run()
    e = rep["recipes"]["W4A16"]["eval"]
    assert e["generation_exact_match"] == pytest.approx(1 / 6)  # only sample 0; a longer answer is not an exact match
    assert e["generation_prefix_match"] == pytest.approx((1 + 0.5 + 0 + 0.75 + 1 + 0.75) / 6)


def test_skip_rules_keep_the_matching_layers_unquantized(tmp_path: Path) -> None:
    rules = [{"match": r"language_model\.layers\.0\.", "skip": True}, {"match": r"^model\.merger\.", "skip": True}]
    rep, _ = run(tmp_path, attribution=False, mixed_precision={"rules": rules})
    names = {t["name"] for t in rep["recipes"]["W4A16"]["layers"]}
    assert not [n for n in names if "language_model.layers.0." in n or n.startswith("model.merger.")]
    assert any("language_model.layers.1." in n for n in names) and any("vision_tower" in n for n in names)
    r = rep["recipes"]["W4A16"]
    assert sorted(r["skipped"]) == sorted(
        u.name for u in FakeAdapter().qmodel.units if "language_model.layers.0." in u.name or u.name.startswith("model.merger.")
    )
    full, _ = run(tmp_path / "full", attribution=False)
    assert r["summary"]["kl"] < full["recipes"]["W4A16"]["summary"]["kl"], "leaving layers in FP can only move the distribution less"


def test_skipped_layers_are_excluded_from_attribution_too(tmp_path: Path) -> None:
    rules = [{"match": r"vision_tower", "skip": True}]
    rep, _ = run(tmp_path, mixed_precision={"rules": rules})
    rows = {a["modality"]: a for a in rep["recipes"]["W4A16"]["attribution"]}
    assert "vision" not in rows or rows["vision"]["units"] == 1, rows  # only the merger (also modality vision) is left


def test_a_modality_skipped_entirely_is_still_listed_as_skipped_and_not_attributed(tmp_path: Path) -> None:
    rep, _ = run(tmp_path, mixed_precision={"rules": [{"match": r"vision_tower|\.merger\.", "skip": True}]})
    r = rep["recipes"]["W4A16"]
    assert any("vision_tower" in n for n in r["skipped"]) and any(n.startswith("model.merger.") for n in r["skipped"])
    assert [a["modality"] for a in r["attribution"]] == ["language"]


def test_a_bit_width_rule_is_refused_rather_than_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bits"):
        run(tmp_path, mixed_precision={"rules": [{"match": "layers", "bits": 8}]})


def test_every_recipe_summary_carries_a_skip_suggestion_in_the_job_rule_format() -> None:
    layers = [
        {
            "name": f"model.language_model.layers.{i}.mlp.up_proj",
            "component": "language",
            "shape": [256, 128],
            "act_kurtosis": 1.0,
            "out_sqnr_db": 27.0 if i == 5 else 38.0 + 0.1 * (i % 4),
        }
        for i in range(24)
    ]
    s = VLMJob._summarize({"layers": layers, "weight": {"bits": 4}, "eval": {}}, None)
    assert s["suggestion"]["rules"] == [{"match": r"^model\.language_model\.layers\.5\.mlp\.up_proj$", "skip": True}]
    assert s["suggestion"]["extra_weight_bytes"] == 256 * 128 * 12 // 8


def test_the_suggestion_costs_bytes_at_the_recipes_bit_width_and_assumes_4_when_it_is_missing() -> None:
    layers = [
        {
            "name": f"model.language_model.layers.{i}.mlp.up_proj",
            "component": "language",
            "shape": [256, 128],
            "act_kurtosis": 1.0,
            "out_sqnr_db": 27.0 if i == 5 else 38.0 + 0.1 * (i % 4),
        }
        for i in range(24)
    ]
    got = {
        b: VLMJob._summarize({"layers": layers, "weight": {"bits": b} if b else {}, "eval": {}}, None)["suggestion"]["extra_weight_bytes"]
        for b in (8, 4, None)
    }
    assert got == {8: 256 * 128, 4: 256 * 128 * 12 // 8, None: 256 * 128 * 12 // 8}


def test_from_yaml_reads_mixed_precision(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text(
        yaml.safe_dump(
            {"model": {"path": "/m"}, "quant": {"recipe": "w4a16", "mixed_precision": {"rules": [{"match": "x", "skip": True}]}}}
        )
    )
    assert VLMJobSpec.from_yaml(p).mixed_precision == {"rules": [{"match": "x", "skip": True}]}


@pytest.mark.parametrize(
    ("text", "gt", "ok"),
    [
        ("The color is red.", "red", True),
        ("RED", "red", True),
        ("blue", "red", False),
        ("3", "3", True),
        ("There are 33 shapes", "3", False),
        ("yes, there is", "yes", True),
        ("", "yes", False),
        ("no", "yes", False),
        # the model counts in words as often as in digits
        ("There are three shapes.", "3", True),
        ("Two shapes", "2", True),
        ("There are three shapes.", "2", False),
        ("one two three", "4", False),
        ("a third", "3", False),
        ("There are 3 shapes", "three", True),
    ],
)
def test_answer_matching_is_a_whole_word_case_insensitive_containment(text: str, gt: str, ok: bool) -> None:
    assert answer_matches(text, gt) is ok


def test_from_yaml_reads_the_wizards_job_shape(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "n"},
                "model": {"path": "/m", "type": "vlm"},
                "quant": {"recipe": ["w4a16", "w8a16"], "algorithm": "gptq", "group_size": 64, "also_quantize": []},
                "calibration": {"dataset": "synthetic", "n_samples": 8},
                "evaluation": {"dataset": "synthetic", "n_samples": 10, "max_new_tokens": 12, "attribution": False, "skip": True},
                "deploy": {"hardware": "nvidia/rtx4090", "device": "cuda:1"},
                "output_dir": "out",
            }
        )
    )
    s = VLMJobSpec.from_yaml(p)
    assert (s.name, s.model_path, s.recipes, s.algorithm, s.group_size_override) == ("n", "/m", ["w4a16", "w8a16"], "gptq", 64)
    assert (s.calib_samples, s.eval_samples, s.max_new_tokens, s.attribution, s.quantize_only) == (8, 10, 12, False, True)
    assert (s.device, s.hardware, s.output_dir) == ("cuda:1", "nvidia/rtx4090", "out")


def test_defaults_are_the_ones_the_wizard_sends_when_nothing_is_touched(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump({"model": {"path": "/m"}, "quant": {"recipe": ["w4a16"]}}))
    s = VLMJobSpec.from_yaml(p)
    assert (s.calib_source, s.eval_source, s.attribution, s.quantize_only, s.group_size_override, s.max_new_tokens) == (
        "synthetic",
        "synthetic",
        True,
        False,
        None,
        24,
    )


def test_from_yaml_reads_export_and_vllm_deploy_fields(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "n"},
                "model": {"path": "/m", "type": "vlm"},
                "quant": {"recipe": ["w4a16"], "algorithm": "rtn"},
                "calibration": {},
                "evaluation": {},
                "deploy": {"export": ["awq", "gptq"], "vllm": {"concurrency": 2, "requests": 8}},
            }
        )
    )
    s = VLMJobSpec.from_yaml(p)
    assert s.export == ["awq", "gptq"]
    assert s.vllm == {"concurrency": 2, "requests": 8}


def test_export_and_vllm_default_to_empty_and_none(tmp_path: Path) -> None:
    p = tmp_path / "j.yaml"
    p.write_text(yaml.safe_dump({"model": {"path": "/m"}, "quant": {"recipe": ["w4a16"]}}))
    s = VLMJobSpec.from_yaml(p)
    assert s.export == []
    assert s.vllm is None


def test_export_recipe_is_called_when_export_is_non_empty(tmp_path: Path) -> None:
    """_export_recipe must be invoked after eval_quantized; its return value is stored in rep['export']."""
    exported: list[str] = []

    def factory(s: VLMJobSpec) -> FakeAdapter:
        return FakeAdapter()

    job = VLMJob(spec(tmp_path, export=["awq"], vllm=None), job_id="j", adapter_factory=factory)

    # Patch _export_recipe to record that it was called and return a stub (no real exporter needed).
    def fake_export(ad: object, recipe_name: str) -> dict:  # type: ignore[type-arg]
        exported.append(recipe_name)
        return {"awq": {"status": "exported", "bytes": 1}}

    job._export_recipe = fake_export  # type: ignore[method-assign]
    rep = job.run()
    assert "export" in rep["recipes"]["W4A16"], "export key must be in recipe report"
    assert exported == ["W4A16"], "export was not called for the recipe"


def test_vllm_serve_is_called_only_when_vllm_is_set_and_export_succeeded(tmp_path: Path) -> None:
    """_vllm_serve must only be called when the export step produced status='exported'."""
    served: list[str] = []

    def factory(s: VLMJobSpec) -> FakeAdapter:
        return FakeAdapter()

    job = VLMJob(spec(tmp_path, export=["awq"], vllm={"concurrency": 1}), job_id="j", adapter_factory=factory)

    def fake_export(ad: object, recipe_name: str) -> dict:  # type: ignore[type-arg]
        return {"awq": {"status": "exported", "bytes": 1}}

    def fake_serve(ckpt: Path, evals: list) -> dict:  # type: ignore[type-arg]
        served.append(str(ckpt))
        return {"status": "succeeded"}

    job._export_recipe = fake_export  # type: ignore[method-assign]
    job._vllm_serve = fake_serve  # type: ignore[method-assign]
    job.run()
    assert len(served) == 1, "vllm serve must be called once for the one exported recipe"


def test_vllm_serve_is_not_called_when_export_failed(tmp_path: Path) -> None:
    served: list[str] = []

    def factory(s: VLMJobSpec) -> FakeAdapter:
        return FakeAdapter()

    job = VLMJob(spec(tmp_path, export=["awq"], vllm={"concurrency": 1}), job_id="j", adapter_factory=factory)

    def fake_export(ad: object, recipe_name: str) -> dict:  # type: ignore[type-arg]
        return {"awq": {"status": "unsupported", "n_problems": 1}}

    def fake_serve(ckpt: Path, evals: list) -> dict:  # type: ignore[type-arg]
        served.append(str(ckpt))
        return {"status": "succeeded"}

    job._export_recipe = fake_export  # type: ignore[method-assign]
    job._vllm_serve = fake_serve  # type: ignore[method-assign]
    job.run()
    assert served == [], "vllm serve must NOT be called when export status != 'exported'"
