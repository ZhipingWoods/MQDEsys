"""GGUF requantize (T-0705): metadata is copied, blk.* linears are re-encoded, shape fallbacks apply."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from mqde import core
from mqde.export.gguf import dequantize_linear_weights, requantize


def _src(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = {
        "token_embd.weight": rng.normal(0, 0.02, (64, 512)).astype(np.float16),
        "blk.0.attn_q.weight": rng.normal(0, 0.02, (512, 512)).astype(np.float16),  # ne0 512: K-quant ok
        "blk.0.ffn_up.weight": rng.normal(0, 0.02, (256, 896)).astype(np.float16),  # ne0 896: needs fallback
        "blk.0.ffn_down.weight": rng.normal(0, 0.02, (64, 200)).astype(np.float16),  # ne0 200: no 32-block type
        "blk.0.attn_norm.weight": np.ones(512, np.float32),
    }
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", 1)
    w.add_string("general.name", "tiny")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    w.add_uint32("general.file_type", 1)
    for k, v in t.items():
        w.add_tensor(k, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return t


@pytest.mark.parametrize("ty", ["q8_0", "q5_0", "q4_0", "iq4_nl", "q4_k", "q2_k"])
def test_requantize(ty: str, tmp_path: Path) -> None:
    src, dst = tmp_path / "f16.gguf", tmp_path / f"{ty}.gguf"
    t = _src(src)
    info = requantize(src, dst, ty)
    r = gguf.GGUFReader(str(dst))
    assert r.fields["general.name"].contents() == "tiny" and r.fields["tokenizer.ggml.tokens"].contents() == ["a", "b", "c"]
    assert r.fields["llama.block_count"].contents() == 1
    types = {x.name: x.tensor_type.name.lower() for x in r.tensors}
    assert types["token_embd.weight"] == "f16" and types["blk.0.attn_norm.weight"] == "f32"
    assert types["blk.0.attn_q.weight"] == ty
    # llama-quantize tensor_type_fallback: 896 % 256 != 0 -> Q4_K uses Q5_0, Q2_K uses Q4_0
    expect_up = {"q4_k": "q5_0", "q2_k": "q4_0"}.get(ty, ty)
    assert types["blk.0.ffn_up.weight"] == expect_up
    assert types["blk.0.ffn_down.weight"] == "f16"  # 200 % 32 != 0: stays in the source type
    big = ty in ("q4_k", "q2_k")
    assert info["fallback"] == (["blk.0.ffn_up.weight"] if big else []) + ["blk.0.ffn_down.weight"]
    deq = dequantize_linear_weights(dst)
    for name in ("blk.0.attn_q.weight", "blk.0.ffn_up.weight"):
        x = t[name].astype(np.float32)
        ref = core.gguf_dequantize(core.gguf_quantize(x, types[name]), types[name], x.size).reshape(x.shape)
        assert np.array_equal(deq[name], ref)
        g = gguf.quants.dequantize(next(z for z in r.tensors if z.name == name).data, gguf.GGMLQuantizationType[types[name].upper()])
        assert np.array_equal(g.reshape(x.shape).astype(np.float32), ref)
