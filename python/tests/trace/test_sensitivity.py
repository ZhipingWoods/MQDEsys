"""Per-group FP fallback sensitivity (T-0902) and the mixed-precision suggestion (T-0904) on a tiny model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).parents[1] / "export"))

from _tiny import calib, tiny  # noqa: E402

from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy  # noqa: E402
from mqde.quant.engine import RecipeSpec  # noqa: E402
from mqde.quant.modules import FakeQuantLinear  # noqa: E402
from mqde.trace.sensitivity import fallback_sensitivity, group_key, suggest  # noqa: E402


def _setup():
    ad = tiny()
    ids = calib(1, 64)[0]
    ref = ad.logits(ids).clone()
    q = BlockwiseQuantizer(ad, RecipeSpec.load("w2a16"), "rtn")
    q.run(calib())
    err = lambda: float((ad.logits(ids) - ref).norm() / ref.norm())  # noqa: E731
    return ad, q, err


def test_group_keys() -> None:
    n = "model.layers.12.mlp.down_proj"
    assert group_key(n, "block") == "block:12" and group_key(n, "type") == "mlp.down_proj"


@pytest.mark.parametrize("by,n_groups", [("block", 2), ("type", 7)])
def test_fallback_restores_model_and_ranks(by: str, n_groups: int) -> None:
    ad, q, err = _setup()
    before = {n: m.weight.clone() for n, m in ad.model.named_modules() if isinstance(m, FakeQuantLinear)}
    e_q = err()
    s = fallback_sensitivity(ad.model, q.originals, err, by=by)
    assert len(s["groups"]) == n_groups and s["metric_quantized"] == e_q
    # errors of different groups can partly cancel, so a single fallback is not guaranteed to help;
    # overall, putting groups back to FP must recover error
    assert max(g["recovered"] for g in s["groups"]) > 0 and sum(g["recovered"] for g in s["groups"]) > 0
    assert s["ranking"][0] == max(s["groups"], key=lambda g: g["recovered"])["group"]
    after = {n: m.weight for n, m in ad.model.named_modules() if isinstance(m, FakeQuantLinear)}
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    assert err() == e_q


def test_suggestion_rules_apply_and_help() -> None:
    ad, q, err = _setup()
    s = fallback_sensitivity(ad.model, q.originals, err, by="block")
    sug = suggest(s, metric_fp=0.0, target_share=0.3, bits=8)
    assert sug["rules"] and sug["predicted_metric"] < s["metric_quantized"] and sug["extra_weight_bytes"] > 0
    q.restore()
    pol = MixedPrecisionPolicy.from_dict({"rules": sug["rules"]})
    BlockwiseQuantizer(ad, RecipeSpec.load("w2a16"), "rtn", policy=pol).run(calib())
    assert err() < s["metric_quantized"]
    budget = suggest(s, metric_fp=0.0, target_share=1.0, max_extra_bytes=1.0)
    assert budget["rules"] == []
    assert isinstance(ad.model.get_submodule("model.layers.0.mlp.down_proj"), (FakeQuantLinear, nn.Linear))
