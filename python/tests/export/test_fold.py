"""AWQ / SmoothQuant scale folding (T-0603): one scale per input group folded into the producing op, so
the transform is exact in FP, needs no runtime divisor, and the result exports like any GPTQ/RTN model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).parent))

from _tiny import calib, tiny  # noqa: E402

from mqde.export import get_exporter  # noqa: E402
from mqde.export.real_check import real_check  # noqa: E402
from mqde.quant.blockwise import BlockwiseQuantizer  # noqa: E402
from mqde.quant.engine import RecipeSpec  # noqa: E402
from mqde.quant.modules import FakeQuantLinear, replace_module  # noqa: E402


@pytest.mark.parametrize("algo", ["awq", "smoothquant", "awq+gptq"])
def test_fold_is_fp_exact_and_restorable(algo: str) -> None:
    ad = tiny()
    ids = calib(1, 48)[0]
    ref = ad.logits(ids).clone()
    q = BlockwiseQuantizer(ad, RecipeSpec.load("w4a8"), algo)
    q.run(calib())
    fq = {n: m for n, m in ad.model.named_modules() if isinstance(m, FakeQuantLinear)}
    assert fq and all(m.in_scale is None for m in fq.values())  # no runtime divisor left
    for n, lin in q.originals.items():  # FP weights of the folded model == original function
        replace_module(ad.model, n, lin)
    assert torch.allclose(ad.logits(ids), ref, rtol=1e-4, atol=1e-4)
    for n, m in fq.items():
        replace_module(ad.model, n, m)
    q.restore()
    assert torch.equal(ad.logits(ids), ref)  # norms / weights put back bit-exactly


def test_fold_targets_and_gqa_skip() -> None:
    ad = tiny()
    res = BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "awq").run(calib())
    by = {t.name: t.algo_info["smooth"] for t in res.traces}
    assert by["model.layers.0.self_attn.q_proj"]["folded"] in ("into input_layernorm", "no (identity scale was best)")
    assert by["model.layers.0.mlp.gate_proj"]["group"] == "mlp_in"
    assert by["model.layers.0.mlp.down_proj"]["folded"] in ("into mlp.up_proj", "no (identity scale was best)")
    # tiny model has 2 kv heads < 4 heads: v_proj rows cannot absorb o_proj's scale
    assert by["model.layers.0.self_attn.o_proj"]["folded"].startswith("no (self_attn.v_proj not foldable")


def test_folded_awq_exports_with_identical_codes(tmp_path: Path) -> None:
    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "awq+gptq").run(calib())
    assert get_exporter("awq").check(ad.model) == []
    get_exporter("awq").export(ad.model, tmp_path)
    r = real_check(ad.model, tmp_path)
    assert r["codes_identical"] and r["w_rel_l2_mean"] < 1e-3
