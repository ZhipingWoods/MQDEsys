"""Fake/real consistency (T-0702): exported integers are identical to the fake-quant ones; corruption is caught."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
sys.path.insert(0, str(Path(__file__).parent))

from _tiny import calib, tiny  # noqa: E402

from mqde.export import get_exporter  # noqa: E402
from mqde.export.real_check import real_check, use_real_weights  # noqa: E402
from mqde.quant.blockwise import BlockwiseQuantizer  # noqa: E402
from mqde.quant.engine import RecipeSpec  # noqa: E402


@pytest.mark.parametrize("fmt,recipe,algo", [("gptq", "w4a16", "gptq"), ("gptq", "w2a16", "gptq"), ("gptq", "w8a16", "rtn"), ("awq", "w4a16", "rtn")])
def test_codes_identical_and_weights_close(fmt: str, recipe: str, algo: str, tmp_path: Path) -> None:
    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load(recipe), algo).run(calib())
    get_exporter(fmt).export(ad.model, tmp_path)
    r = real_check(ad.model, tmp_path)
    assert r["codes_identical"] and r["code_mismatch"] == 0 and not r["missing"], r["per_layer"][:2]
    assert r["w_rel_l2_mean"] < 1e-3  # only fp16 scale storage separates real from fake
    ids = calib(1, 48)[0]
    fake = ad.logits(ids)
    with use_real_weights(ad.model, tmp_path):
        real = ad.logits(ids)
    assert float((real - fake).norm() / fake.norm()) < 1e-3
    assert torch.equal(ad.logits(ids), fake)  # weights restored


def test_corrupted_packing_is_detected(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "rtn").run(calib())
    get_exporter("gptq").export(ad.model, tmp_path)
    f = tmp_path / "model.safetensors"
    sd = load_file(str(f))
    k = "model.layers.1.self_attn.o_proj.qweight"
    sd[k][3, 7] ^= 1 << 13  # flip one bit of one 4-bit code
    save_file(sd, str(f))
    r = real_check(ad.model, tmp_path)
    assert not r["codes_identical"] and r["code_mismatch"] == 1
    bad = [x for x in r["per_layer"] if x["code_mismatch"]]
    assert [x["name"] for x in bad] == ["model.layers.1.self_attn.o_proj"]
