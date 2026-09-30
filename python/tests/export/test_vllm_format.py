"""GPTQ / AWQ checkpoint format (T-0701): tensor names, shapes, dtypes and quantization_config (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
sys.path.insert(0, str(Path(__file__).parent))

from _tiny import calib, tiny

from mqde.export import get_exporter
from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy
from mqde.quant.engine import RecipeSpec


@pytest.mark.parametrize("fmt,algo", [("gptq", "gptq"), ("gptq", "rtn"), ("awq", "rtn")])
def test_checkpoint_layout(fmt: str, algo: str, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), algo).run(calib())
    res = get_exporter(fmt).export(ad.model, tmp_path)
    assert len(res.quantized) == 14 and not res.kept_fp
    cfg = json.loads((tmp_path / "config.json").read_text())
    q = cfg["quantization_config"]
    assert q["quant_method"] == fmt and q["bits"] == 4 and q["group_size"] == 128
    sd = load_file(str(tmp_path / "model.safetensors"))
    n = "model.layers.0.mlp.down_proj"  # [256, 512]
    if fmt == "gptq":
        assert q["desc_act"] is False and q["checkpoint_format"] == "gptq"
        assert sd[f"{n}.qweight"].shape == (512 * 4 // 32, 256) and sd[f"{n}.qweight"].dtype == torch.int32
        assert sd[f"{n}.qzeros"].shape == (4, 256 * 4 // 32)
        assert torch.equal(sd[f"{n}.g_idx"], torch.arange(512, dtype=torch.int32) // 128)
    else:
        assert q["zero_point"] is True and q["version"] == "gemm"
        assert sd[f"{n}.qweight"].shape == (512, 256 * 4 // 32) and sd[f"{n}.qzeros"].shape == (4, 32)
    assert sd[f"{n}.scales"].shape == (4, 256) and sd[f"{n}.scales"].dtype == torch.float16
    assert f"{n}.weight" not in sd and "lm_head.weight" not in sd  # tied embeddings are not duplicated
    assert sd["model.embed_tokens.weight"].dtype == torch.float16 and "model.norm.weight" in sd


def test_mixed_precision_goes_into_dynamic(tmp_path: Path) -> None:
    ad = tiny()
    pol = MixedPrecisionPolicy.from_dict({"rules": [{"match": r"down_proj$", "skip": True}, {"match": "block:-1", "bits": 8}]})
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "rtn", policy=pol).run(calib())
    res = get_exporter("gptq").export(ad.model, tmp_path)
    dyn = res.quantization_config["dynamic"]
    assert res.kept_fp == ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]
    assert dyn[r"-:^model\.layers\.0\.mlp\.down_proj$"] == {}
    assert dyn[r"+:^model\.layers\.1\.self_attn\.q_proj$"]["bits"] == 8
    assert r"+:^model\.layers\.0\.self_attn\.q_proj$" not in dyn


def test_check_rejects_unexportable_layers(tmp_path: Path) -> None:
    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "awq", fold_scales=False).run(calib())  # per-layer divisors
    probs = get_exporter("gptq").check(ad.model)
    assert probs and all("fold" in r for _, r in probs)
    with pytest.raises(ValueError, match="cannot be exported"):
        get_exporter("gptq").export(ad.model, tmp_path)
    ad2 = tiny()
    BlockwiseQuantizer(ad2, RecipeSpec.load("fp8"), "rtn").run(calib())
    assert all("not an integer format" in r for _, r in get_exporter("awq").check(ad2.model))
