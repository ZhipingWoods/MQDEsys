"""compressed-tensors checkpoint format (T-0701): tensor names, shapes, dtypes and quantization_config (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
sys.path.insert(0, str(Path(__file__).parent))

from _tiny import calib, tiny

from mqde.export import get_exporter
from mqde.export.compressed_tensors import pack_int32, unpack_int32
from mqde.export.int_checkpoint import layer_codes
from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy
from mqde.quant.engine import RecipeSpec
from mqde.quant.modules import FakeQuantLinear

N = "model.layers.0.mlp.down_proj"  # [256, 512]


@pytest.mark.parametrize("bits", [4, 8])
def test_pack_int32_round_trip_and_bit_order(bits: int) -> None:
    rng = np.random.default_rng(0)
    v = rng.integers(0, 1 << bits, size=(16, 64), dtype=np.uint8)
    for dim in (0, 1):
        p = pack_int32(v, bits, dim)
        assert p.dtype == np.int32 and p.shape[dim] == v.shape[dim] * bits // 32
        assert np.array_equal(unpack_int32(p, bits, dim), v)
    # element 0 sits in the lowest bits of the word
    row = np.zeros((1, 32 // bits), dtype=np.uint8)
    row[0, 1] = 1
    assert int(pack_int32(row, bits, 1).view(np.uint32)[0, 0]) == 1 << bits


def test_w4a16_layout(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "gptq").run(calib())
    m = ad.model.get_submodule(N)
    assert isinstance(m, FakeQuantLinear)
    res = get_exporter("compressed-tensors").export(ad.model, tmp_path)
    assert len(res.quantized) == 14 and not res.kept_fp
    q = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert q["quant_method"] == "compressed-tensors" and q["format"] == "pack-quantized"
    assert list(q["config_groups"]) == ["group_0"] and q["ignore"] == ["lm_head"]  # as llm-compressor writes it
    w = q["config_groups"]["group_0"]["weights"]
    assert (w["num_bits"], w["type"], w["symmetric"], w["strategy"], w["group_size"]) == (4, "int", False, "group", 128)
    assert q["config_groups"]["group_0"]["input_activations"] is None
    sd = load_file(str(tmp_path / "model.safetensors"))
    assert sd[f"{N}.weight_packed"].shape == (256, 512 // 8) and sd[f"{N}.weight_packed"].dtype == torch.int32
    assert sd[f"{N}.weight_scale"].shape == (256, 4) and sd[f"{N}.weight_scale"].dtype == torch.float16
    assert sd[f"{N}.weight_zero_point"].shape == (256 // 8, 4)
    assert sd[f"{N}.weight_shape"].tolist() == [256, 512]
    assert f"{N}.weight" not in sd and "lm_head.weight" not in sd
    # dequantizing the checkpoint reproduces the fake-quant weight up to fp16 scale rounding
    codes = unpack_int32(sd[f"{N}.weight_packed"].numpy(), 4, 1).astype(np.float32)
    zeros = unpack_int32(sd[f"{N}.weight_zero_point"].numpy(), 4, 0).astype(np.float32)
    c0, z0, _, _ = layer_codes(m)
    assert np.array_equal(codes, c0) and np.array_equal(zeros, z0)
    sc = sd[f"{N}.weight_scale"].float().numpy()
    deq = (codes.reshape(256, 4, 128) - zeros[..., None]) * sc[..., None]
    assert np.allclose(deq.reshape(256, 512), m.weight.float().numpy(), atol=2e-3, rtol=1e-3)


def test_fp8_layout(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("fp8"), "rtn").run(calib())
    m = ad.model.get_submodule(N)
    res = get_exporter("compressed-tensors").export(ad.model, tmp_path)
    q = res.quantization_config
    assert q["format"] == "float-quantized"
    g = q["config_groups"]["group_0"]
    assert (g["weights"]["type"], g["weights"]["strategy"], g["weights"]["num_bits"]) == ("float", "channel", 8)
    assert g["input_activations"]["dynamic"] is True and g["input_activations"]["strategy"] == "token"
    sd = load_file(str(tmp_path / "model.safetensors"))
    assert sd[f"{N}.weight"].dtype == torch.float8_e4m3fn and sd[f"{N}.weight"].shape == (256, 512)
    assert sd[f"{N}.weight_scale"].shape == (256, 1) and sd[f"{N}.weight_scale"].dtype == torch.float32
    deq = sd[f"{N}.weight"].float() * sd[f"{N}.weight_scale"]
    assert torch.equal(deq, m.weight.float())  # FP8 codes and fp32 scales are stored exactly


def test_mixed_precision_groups_and_ignore(tmp_path: Path) -> None:
    ad = tiny()
    pol = MixedPrecisionPolicy.from_dict({"rules": [{"match": r"down_proj$", "skip": True}, {"match": "block:-1", "bits": 8}]})
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "rtn", policy=pol).run(calib())
    res = get_exporter("compressed-tensors").export(ad.model, tmp_path)
    q = res.quantization_config
    assert q["ignore"] == ["lm_head", "model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]
    assert res.kept_fp == ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]
    assert q["config_groups"]["group_0"]["weights"]["num_bits"] == 4
    g1 = q["config_groups"]["group_1"]
    assert g1["weights"]["num_bits"] == 8
    assert r"re:^model\.layers\.1\.self_attn\.q_proj$" in g1["targets"]
    assert not any("layers\\.0" in t for t in g1["targets"])


def test_check_rejects_unexportable_layers() -> None:
    ad = tiny()
    BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "awq", fold_scales=False).run(calib())
    probs = get_exporter("compressed-tensors").check(ad.model)
    assert probs and all("fold" in r for _, r in probs)
    ad2 = tiny()
    BlockwiseQuantizer(ad2, RecipeSpec.load("w2a16"), "rtn").run(calib())
    assert all("2-bit" in r for _, r in get_exporter("compressed-tensors").check(ad2.model))


def test_matches_compressed_tensors_library(tmp_path: Path) -> None:
    """Byte-exact against compressed-tensors' own packer (skips until the package is installed)."""
    ct = pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import pack_to_int32

    rng = np.random.default_rng(0)
    for bits in (4, 8):
        v = rng.integers(0, 1 << bits, size=(64, 128), dtype=np.uint8)
        signed = torch.from_numpy(v.astype(np.int64) - (1 << (bits - 1))).to(torch.int8)
        for dim in (0, 1):
            ref = pack_to_int32(signed, bits, packed_dim=dim)
            assert np.array_equal(pack_int32(v, bits, dim), ref.numpy()), (ct.__version__, bits, dim)
