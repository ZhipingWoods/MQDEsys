"""World-model / DiT quantization job end to end on the built-in tiny checkpoint (CPU, seconds)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mqde.models.base import QuantModel
from mqde.models.world import TinyDiTWorld, save_tiny_checkpoint
from mqde.pipeline.world_job import WorldJob, WorldJobSpec


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_tiny_checkpoint(tmp_path_factory.mktemp("tiny-dit"))


def spec(tmp_path: Path, checkpoint: Path, **kw: object) -> WorldJobSpec:
    base = {
        "name": "tiny-world",
        "model_path": str(checkpoint),
        "recipes": ["w4a16"],
        "algorithm": "rtn",
        "calib_samples": 3,
        "eval_samples": 3,
        "frames": 4,
        "sample_steps": 4,
        "device": "cpu",
        "output_dir": str(tmp_path),
    }
    return WorldJobSpec(**{**base, **kw})  # type: ignore[arg-type]


def run(tmp_path: Path, checkpoint: Path, **kw: object) -> dict:
    return WorldJob(spec(tmp_path, checkpoint, **kw), job_id="j").run()


def test_the_family_classifies_only_the_transformer_as_quantizable(checkpoint: Path) -> None:
    ad = TinyDiTWorld(checkpoint, device="cpu")
    comps = ad.qmodel.summary()
    assert set(comps) == {"patch_embed", "conditioner", "action_encoder", "transformer", "final"}
    assert comps["transformer"]["blocks"] == 2
    assert comps["transformer"]["quantizable"] == 12  # 4 attn + 2 mlp per block
    assert comps["transformer"]["linears"] == 2 * (4 + 2 + 2)  # + two adaLN modulation linears
    assert ad.qmodel.get_blocks("transformer")[0].modality == "world"
    assert all(u.keep_fp for u in ad.qmodel.units if u.component in ("patch_embed", "conditioner", "final"))


def test_generation_is_deterministic_for_a_seed_and_changes_with_another(checkpoint: Path) -> None:
    ad = TinyDiTWorld(checkpoint, device="cpu")
    from mqde.data.world import synthetic

    s = synthetic(1, seed=0, frames=4)[0]
    inp = ad.preprocess(s)
    a = ad.generate(s, seed=5, inputs=inp)
    b = ad.generate(s, seed=5, inputs=inp)
    c = ad.generate(s, seed=6, inputs=inp)
    assert a.shape == (4, 3, 32, 32) and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_full_job_reports_psnr_against_fp_and_ground_truth(tmp_path: Path, checkpoint: Path) -> None:
    rep = run(tmp_path, checkpoint)
    assert rep["kind"] == "world" and rep["verdict"] == "succeeded"
    assert [s["stage"] for s in rep["stages"]] == [
        "validate", "inspect", "calib_prepare", "eval_baseline", "attribution", "quantize", "eval_quantized", "report"
    ]
    base, r = rep["baseline"], rep["recipes"]["W4A16"]
    assert base["psnr_vs_gt"] > 0 and base["noise_floor"]["psnr"] > 0
    assert r["summary"]["psnr_vs_ref"] > 20
    assert r["summary"]["mse_vs_ref"] > 0
    assert r["summary"]["psnr_vs_gt"] > 0
    assert r["eval"]["per_sample"] and len(r["eval"]["per_sample"]) == 3
    assert r["summary"]["worst_stage"] in {"world", "vision", "action"}
    assert r["layers"] and {t["component"] for t in r["layers"]} == {"transformer"}
    assert r["summary"]["lpips_vs_ref"] is None  # optional runtime dependency, reported honestly
    assert json.loads((tmp_path / "j" / "report.json").read_text())["kind"] == "world"


def test_more_bits_are_closer_to_the_fp_output(tmp_path: Path, checkpoint: Path) -> None:
    rep = run(tmp_path, checkpoint, recipes=["w8a16", "w4a16"], attribution=False)
    assert rep["recipes"]["W8A16"]["summary"]["psnr_vs_ref"] > rep["recipes"]["W4A16"]["summary"]["psnr_vs_ref"]


def test_ewm_reports_action_conditioning_and_attributes_the_action_path(tmp_path: Path, checkpoint: Path) -> None:
    rep = run(tmp_path, checkpoint, action_conditioned=True, also_quantize=["action"], attribution=True)
    assert rep["kind"] == "ewm"
    assert rep["baseline"]["action_conditioned"] is True and rep["baseline"]["action_dim"] == 128
    rows = {a["modality"]: a for a in rep["recipes"]["W4A16"]["attribution"]}
    assert "action" in rows and rows["action"]["units"] == 1
    assert rep["data"]["eval"]["action_conditioned"] is True


def test_quantize_only_skips_baseline_attribution_and_eval(tmp_path: Path, checkpoint: Path) -> None:
    rep = run(tmp_path, checkpoint, quantize_only=True)
    assert [s["stage"] for s in rep["stages"]] == ["validate", "inspect", "calib_prepare", "quantize", "report"]
    assert "baseline" not in rep
    assert rep["recipes"]["W4A16"]["eval"] == {}
    assert rep["recipes"]["W4A16"]["summary"]["psnr_vs_ref"] is None
    assert rep["recipes"]["W4A16"]["layers"]


def test_a_failure_still_writes_a_failed_report(tmp_path: Path, checkpoint: Path) -> None:
    def boom(s: WorldJobSpec) -> TinyDiTWorld:
        raise RuntimeError("model would not load")

    with pytest.raises(RuntimeError, match="would not load"):
        WorldJob(spec(tmp_path, checkpoint), job_id="bad", adapter_factory=boom).run()
    rep = json.loads((tmp_path / "bad" / "report.json").read_text())
    assert rep["verdict"] == "failed" and "would not load" in rep["error"]


def test_group_size_override_is_applied_and_logged(tmp_path: Path, checkpoint: Path) -> None:
    rep = run(tmp_path, checkpoint, group_size_override=64, attribution=False)
    assert rep["recipes"]["W4A16"]["weight"]["group_size"] == 64
    assert "group_size override: W4A16 128 -> 64" in (tmp_path / "j" / "events.jsonl").read_text()


def test_native_continuous_serving_is_recorded_per_recipe(tmp_path: Path, checkpoint: Path) -> None:
    rep = run(tmp_path, checkpoint, native_serve={"requests": 1, "concurrency": 1},
              eval_samples=1, calib_samples=1, attribution=False)
    stages = [s["stage"] for s in rep["stages"]]
    assert "deploy_native" in stages
    d = rep["recipes"]["W4A16"]["deploy"]["native"]
    assert d["status"] == "succeeded" and d["protocol"] == "world/video-v1"
    assert d["load"]["ok"] == 1 and d["smoke"]["ok"] is True


def test_the_model_is_restored_between_recipes(checkpoint: Path, tmp_path: Path) -> None:
    ad = TinyDiTWorld(checkpoint, device="cpu")
    WorldJob(spec(tmp_path, checkpoint, recipes=["w4a16", "w2a16"], attribution=False), job_id="r",
             adapter_factory=lambda s: ad).run()
    from mqde.quant.modules import FakeQuantLinear

    assert not [m for m in ad.model.modules() if isinstance(m, FakeQuantLinear)]
