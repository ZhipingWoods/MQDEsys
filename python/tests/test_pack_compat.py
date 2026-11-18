"""Bit packing (T-0405) against upstream reference implementations.

* GGUF: bit-exact against ggml's own ``quantize_row_*_ref`` / ``dequantize_row_*`` (libggml-base via ctypes;
  set MQDE_GGML_LIB, otherwise a local llama.cpp build is searched), and readable by gguf-py
  (GGUFWriter -> GGUFReader -> gguf.quants.dequantize gives our values).
* AWQ: identical words to GPTQModel's ``pack_awq`` (llm-awq GEMM layout).
* GPTQ: identical to the AutoGPTQ/GPTQModel formula qweight[i // pf, o] |= q[o, i] << bits * (i % pf).
All reference dependencies are optional; the tests skip when they are absent.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from mqde import core

GGML_TYPES = {"q4_0": 2, "q5_0": 6, "q8_0": 8, "q2_k": 10, "q4_k": 12, "iq4_nl": 20}
GGML_SYM = {"q4_0": "q4_0", "q5_0": "q5_0", "q8_0": "q8_0", "q2_k": "q2_K", "q4_k": "q4_K", "iq4_nl": "iq4_nl"}
BLOCK = {"q4_0": (32, 18), "q5_0": (32, 22), "q8_0": (32, 34), "q2_k": (256, 84), "q4_k": (256, 144),
         "iq4_nl": (32, 18)}
REPO = Path(__file__).resolve().parents[2]


def _ggml() -> ctypes.CDLL | None:
    cands = [os.environ.get("MQDE_GGML_LIB", "")]
    for c in cands:
        if c and Path(c).is_file():
            return ctypes.CDLL(c)
    return None


def _data(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, n).astype(np.float32)
    x[:256] = 0.0  # all-zero super-block
    x[256:512] = 0.013  # constant block (max == min path)
    x[1000] = 0.9  # outlier
    x[2000:2256] = np.abs(x[2000:2256])  # all-positive block (min clamped to 0)
    return x


@pytest.mark.parametrize("t", sorted(GGML_TYPES))
def test_gguf_bit_exact_vs_ggml(t: str) -> None:
    lib = _ggml()
    if lib is None:
        pytest.skip("libggml-base.so not found (set MQDE_GGML_LIB)")
    x = _data(256 * 64, 1)
    be, bb = BLOCK[t]
    ref = np.zeros(x.size // be * bb, dtype=np.uint8)
    # ggml_quantize_chunk without an imatrix is exactly what llama-quantize runs per tensor (for IQ4_NL it is
    # quantize_iq4_nl's 7-step scale search, not quantize_row_iq4_nl_ref).
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    rows, ncols = 64, 256
    n = lib.ggml_quantize_chunk(
        ctypes.c_int(GGML_TYPES[t]), x.ctypes.data_as(ctypes.c_void_p), ref.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(0), ctypes.c_int64(rows), ctypes.c_int64(ncols), ctypes.c_void_p(None),
    )
    assert n == ref.size
    ours = core.gguf_quantize(x, t)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref), f"{t}: {(ours != ref).sum()} bytes differ"
    yr = np.zeros_like(x)
    getattr(lib, f"dequantize_row_{GGML_SYM[t]}")(
        ref.ctypes.data_as(ctypes.c_void_p), yr.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(x.size)
    )
    assert np.array_equal(core.gguf_dequantize(ref, t, x.size), yr)


@pytest.mark.parametrize("t", sorted(GGML_TYPES))
def test_gguf_file_readable_by_gguf_py(t: str, tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    x = _data(256 * 16, 2).reshape(16, 256)
    q = core.gguf_quantize(x, t)
    qt = gguf.GGMLQuantizationType(GGML_TYPES[t])
    path = tmp_path / f"t_{t}.gguf"
    w = gguf.GGUFWriter(str(path), "mqde")
    w.add_tensor("w", q.reshape(16, -1), raw_shape=q.reshape(16, -1).shape, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    r = gguf.GGUFReader(str(path))
    (tensor,) = r.tensors
    assert tensor.tensor_type == qt and list(tensor.shape) == [256, 16]  # GGUF stores ne0 first
    deq = gguf.quants.dequantize(tensor.data, qt).reshape(16, 256)
    assert np.array_equal(deq.astype(np.float32), core.gguf_dequantize(q, t, x.size).reshape(16, 256))


def test_gguf_dequant_real_file() -> None:
    gguf = pytest.importorskip("gguf")
    files = glob.glob(os.environ.get("MQDE_TEST_GGUF_GLOB", ""))
    if not files:
        pytest.skip("no local GGUF file")
    r = gguf.GGUFReader(files[0])
    checked = 0
    for tensor in r.tensors:
        name = tensor.tensor_type.name.lower()
        if name not in GGML_TYPES:
            continue
        n = int(np.prod(tensor.shape))
        ref = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(-1)
        ours = core.gguf_dequantize(np.asarray(tensor.data).reshape(-1).view(np.uint8), name, n)
        assert np.array_equal(ours, ref.astype(np.float32)), tensor.name
        checked += 1
    if not checked:
        pytest.skip("GGUF file has no Q4_0/Q8_0/Q2_K/Q4_K tensors")


def _gptqmodel_awq():
    p = REPO / "3rdpart/GPTQModel/gptqmodel/quantization/awq/utils/packing_utils.py"
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location("gptqmodel_awq_packing", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_awq_matches_gptqmodel() -> None:
    torch = pytest.importorskip("torch")
    ref = _gptqmodel_awq()
    if ref is None:
        pytest.skip("3rdpart/GPTQModel not present")
    rng = np.random.default_rng(3)
    out, inn, gs = 64, 256, 128
    codes = rng.integers(0, 16, (out, inn), dtype=np.uint8)
    zeros = rng.integers(0, 16, (out, inn // gs), dtype=np.uint8)
    scales = rng.uniform(1e-3, 1e-2, (out, inn // gs)).astype(np.float32)
    qw, qz, sc = core.pack_int("awq", codes, zeros, scales, 4)
    rqw, rqz = ref.pack_awq(torch.from_numpy(codes.T.astype(np.int32)), torch.from_numpy(zeros.T.astype(np.int32)), 4)
    assert np.array_equal(qw, rqw.numpy()) and np.array_equal(qz, rqz.numpy())
    assert np.array_equal(sc, scales.T.astype(np.float16))
    iw, iz = ref.reverse_awq_order(*ref.unpack_awq(torch.from_numpy(qw), torch.from_numpy(qz), 4), 4)
    assert np.array_equal(iw.numpy(), codes.T) and np.array_equal(iz.numpy(), zeros.T)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_gptq_formula_and_dequant(bits: int) -> None:
    rng = np.random.default_rng(bits)
    out, inn, gs = 64, 256, 64
    codes = rng.integers(0, 2**bits, (out, inn), dtype=np.uint8)
    zeros = rng.integers(0, 2**bits, (out, inn // gs), dtype=np.uint8)
    scales = rng.uniform(1e-3, 1e-2, (out, inn // gs)).astype(np.float32)
    qw, qz, sc = core.pack_int("gptq", codes, zeros, scales, bits, v1=True)
    assert qw.shape == (inn * bits // 32, out) and qz.shape == (inn // gs, out * bits // 32)
    if bits != 3:
        pf = 32 // bits
        ref = np.zeros((inn // pf, out), dtype=np.uint32)
        for i in range(inn):
            ref[i // pf] |= codes[:, i].astype(np.uint32) << np.uint32(bits * (i % pf))
        assert np.array_equal(qw.view(np.uint32), ref)
        refz = np.zeros((inn // gs, out // pf), dtype=np.uint32)
        z1 = (zeros.astype(np.int64) - 1) & (2**bits - 1)  # v1 checkpoints store zero - 1
        for o in range(out):
            refz[:, o // pf] |= z1[o].astype(np.uint32) << np.uint32(bits * (o % pf))
        assert np.array_equal(qz.view(np.uint32), refz)
    c2, z2 = core.unpack_int("gptq", qw, qz, out, inn, gs, bits, v1=True)
    assert np.array_equal(c2, codes) and np.array_equal(z2, zeros)
    # the checkpoint dequantizes to scale * (q - zero) with g_idx = i // group_size
    g = np.arange(inn) // gs
    w = sc.astype(np.float32).T[:, g] * (c2.astype(np.float32) - z2[:, g])
    assert np.allclose(w, scales.astype(np.float16).astype(np.float32)[:, g] * (codes.astype(np.float32) - zeros[:, g]))
