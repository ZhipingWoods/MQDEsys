"""Block-wise FP8 checkpoints dequantized in plain PyTorch (T-0605: loading models that are already FP8).

The convention (``W = fp8 * scale_inv``, scale multiplied) was fixed by reading a real 128x128 block-quantized checkpoint:
the product has the size of a normal weight, dividing gives numbers around 1e5 to 1e6. The tests build checkpoints with
a known answer, so a wrong convention, a wrong block expansion or a partial edge block fails them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mqde.models.fp8_dequant import block_size, dequant_block, load_fp8_state_dict

FP8_MAX = 448.0


def quantize_blocks(w: torch.Tensor, block: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference quantizer: per-block amax / 448 scale, FP8 codes. Returns (fp8 [out, in], scale [ceil, ceil] bf16)."""
    bh, bw = block
    o, i = w.shape
    br, bc = (o + bh - 1) // bh, (i + bw - 1) // bw
    pad = torch.zeros(br * bh, bc * bw)
    pad[:o, :i] = w
    blk = pad.reshape(br, bh, bc, bw)
    scale = (blk.abs().amax(dim=(1, 3)).clamp(min=1e-12) / FP8_MAX).to(torch.bfloat16)
    q = (blk / scale.float()[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(br * bh, bc * bw)[:o, :i].contiguous(), scale


def test_round_trip_is_close_and_uses_multiplication() -> None:
    torch.manual_seed(0)
    w = torch.randn(256, 384) * 0.02
    q, s = quantize_blocks(w, (128, 128))
    back = dequant_block(q, s)
    # E4M3 has 3 mantissa bits: rounding to nearest is off by at most 2^-4 of a value, hence of the block's maximum.
    # Judged per block (blocks differ in magnitude); the worst case over 20 random draws is 3.6%, the bound is 6.25%.
    err = (back - w).abs().reshape(2, 128, 3, 128)
    amax = w.abs().reshape(2, 128, 3, 128).amax(dim=(1, 3), keepdim=True)
    assert (err / amax).max() <= 2 ** -4
    assert q.float().abs().max() > 100 and back.abs().max() < 1.0  # codes are big, the product is weight-sized
    wrong = q.float() / s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert wrong.abs().mean() > 1e3 * back.abs().mean()  # the divide convention is nowhere near: the test can tell them apart


@pytest.mark.parametrize(
    ("shape", "block"),
    [((256, 256), (128, 128)), ((300, 200), (128, 128)), ((130, 70), (64, 32)), ((17, 5), (128, 128))],
)
def test_partial_edge_blocks(shape: tuple[int, int], block: tuple[int, int]) -> None:
    torch.manual_seed(1)
    w = torch.randn(*shape)
    q, s = quantize_blocks(w, block)
    assert tuple(s.shape) == ((shape[0] + block[0] - 1) // block[0], (shape[1] + block[1] - 1) // block[1])
    back = dequant_block(q, s, block)
    assert back.shape == w.shape
    assert (back - w).abs().max() < 0.07 * w.abs().max()


def test_each_block_uses_its_own_scale() -> None:
    w = torch.zeros(256, 256)
    w[:128, :128] = 1.0
    w[:128, 128:] = 100.0
    w[128:, :128] = 0.01
    q, s = quantize_blocks(w, (128, 128))
    assert len({round(float(x), 9) for x in s.float().flatten()}) == 4
    back = dequant_block(q, s)
    for (r, c), want in {(0, 0): 1.0, (0, 1): 100.0, (1, 0): 0.01, (1, 1): 0.0}.items():
        assert float(back[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128].mean()) == pytest.approx(want, rel=0.05, abs=1e-6)


def test_shape_mismatch_and_bad_rank_are_errors() -> None:
    q = torch.zeros(256, 256).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="does not match"):
        dequant_block(q, torch.ones(3, 2), (128, 128))
    with pytest.raises(ValueError, match="2-D"):
        dequant_block(q.reshape(-1), torch.ones(2, 2))


def make_checkpoint(d: Path, text_config: bool = False, block: tuple[int, int] = (128, 128)) -> dict[str, torch.Tensor]:
    torch.manual_seed(2)
    sd: dict[str, torch.Tensor] = {}
    ref: dict[str, torch.Tensor] = {}
    for name, shape in {"layers.0.mlp.gate_proj.weight": (256, 384), "layers.0.self_attn.q_proj.weight": (130, 70)}.items():
        w = torch.randn(*shape) * 0.02
        q, s = quantize_blocks(w, block)
        sd[name], sd[name + "_scale_inv"] = q, s
        ref[name] = dequant_block(q, s, block)
    sd["layers.0.input_layernorm.weight"] = torch.ones(384, dtype=torch.bfloat16)
    ref["layers.0.input_layernorm.weight"] = torch.ones(384)
    sd["embed.weight"] = torch.randn(16, 8)
    ref["embed.weight"] = sd["embed.weight"].clone()
    save_file(sd, str(d / "model.safetensors"))
    q = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": list(block)}
    (d / "config.json").write_text(json.dumps({"text_config": {"quantization_config": q}} if text_config else {"quantization_config": q}))
    return ref


@pytest.mark.parametrize("text_config", [False, True])
def test_load_state_dict(tmp_path: Path, text_config: bool) -> None:
    ref = make_checkpoint(tmp_path, text_config)
    assert block_size(tmp_path) == (128, 128)
    out = load_fp8_state_dict(tmp_path)
    assert set(out) == set(ref)  # the _scale_inv tensors are consumed, nothing else is dropped
    for k, r in ref.items():
        assert out[k].dtype == torch.float32 and torch.equal(out[k], r.float()), k
    assert load_fp8_state_dict(tmp_path, torch.bfloat16)["layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16


def test_other_block_size(tmp_path: Path) -> None:
    ref = make_checkpoint(tmp_path, block=(64, 32))
    assert block_size(tmp_path) == (64, 32)
    out = load_fp8_state_dict(tmp_path)
    assert torch.equal(out["layers.0.self_attn.q_proj.weight"], ref["layers.0.self_attn.q_proj.weight"].float())


def test_not_a_blockwise_fp8_checkpoint_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq", "bits": 4}}))
    assert block_size(tmp_path) is None
    save_file({"w": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no block-wise fp8"):
        load_fp8_state_dict(tmp_path)


def test_fp8_weight_without_its_scale_is_an_error(tmp_path: Path) -> None:
    save_file({"a.weight": torch.zeros(128, 128).to(torch.float8_e4m3fn)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}))
    with pytest.raises(ValueError, match=r"without a\.weight_scale_inv"):
        load_fp8_state_dict(tmp_path)
