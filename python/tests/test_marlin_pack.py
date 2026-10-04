"""Marlin packing (T-0405) against vLLM v0.30's reference layout code.

The reference functions below are line-for-line numpy transcriptions of
vllm/model_executor/layers/quantization/utils/marlin_utils_test.py (get_weight_perm, marlin_permute_weights,
marlin_weights) and marlin_utils.py (get_scale_perms, marlin_permute_scales, marlin_zero_points) plus
quant_utils.pack_cols. Those modules import vllm itself, so they cannot be loaded without a vLLM install;
when vLLM is importable the test also checks against the real functions.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mqde import core

TILE = 16


# ---------------------------------------------------------------- vLLM v0.30 reference (transcribed)
def ref_weight_perm(num_bits: int) -> np.ndarray:
    perm_list: list[int] = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7]) if num_bits == 4 else np.array([0, 2, 1, 3])
    return perm.reshape((-1, len(interleave)))[:, interleave].ravel()


def ref_pack_cols(q: np.ndarray, num_bits: int) -> np.ndarray:
    pf = 32 // num_bits
    q = q.astype(np.uint32)
    res = np.zeros((q.shape[0], q.shape[1] // pf), dtype=np.uint32)
    for i in range(pf):
        res |= q[:, i::pf] << num_bits * i
    return res.astype(np.int32)


def ref_marlin_weights(q_w: np.ndarray, size_k: int, size_n: int, num_bits: int) -> np.ndarray:
    perm = ref_weight_perm(num_bits)
    q = q_w.reshape((size_k // TILE, TILE, size_n // TILE, TILE)).transpose((0, 2, 1, 3))
    q = q.reshape((size_k // TILE, size_n * TILE))
    q = q.reshape((-1, perm.size))[:, perm].reshape(q.shape)
    return ref_pack_cols(q, num_bits)


def ref_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    single: list[int] = []
    for i in range(4):
        single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, single


def ref_permute_scales(s: np.ndarray, size_k: int, size_n: int, group_size: int) -> np.ndarray:
    perm, single = ref_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(perm)))[:, perm]
    else:
        s = s.reshape((-1, len(single)))[:, single]
    return s.reshape((-1, size_n))


def ref_zero_points(zp: np.ndarray, size_n: int, num_bits: int) -> np.ndarray:
    perm, _ = ref_scale_perms()
    zp = zp.reshape((-1, len(perm)))[:, perm]
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7]) if num_bits == 4 else np.array([0, 2, 1, 3])
    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel().reshape((-1, size_n))
    return ref_pack_cols(zp, num_bits)


# ---------------------------------------------------------------- tests
CASES = [  # (out, in, group_size, bits, with_zeros)
    (128, 256, 128, 4, False),  # gptq_marlin uint4b8, grouped
    (256, 512, 512, 4, False),  # per-channel (single scale perm)
    (192, 256, 64, 4, True),  # awq_marlin uint4 with zero points
    (128, 256, 128, 8, False),  # uint8b128
    (64, 256, 32, 8, True),
]


def _inputs(out: int, inp: int, gs: int, bits: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = inp // gs
    codes = rng.integers(0, 1 << bits, (out, inp), dtype=np.uint8)
    zeros = rng.integers(0, 1 << bits, (out, g), dtype=np.uint8)
    scales = (np.abs(rng.normal(0, 0.02, (out, g))) + 1e-3).astype(np.float16).astype(np.float32)
    return codes, zeros, scales


@pytest.mark.parametrize(("out", "inp", "gs", "bits", "wz"), CASES)
def test_marlin_matches_vllm_reference(out: int, inp: int, gs: int, bits: int, wz: bool) -> None:
    codes, zeros, scales = _inputs(out, inp, gs, bits, seed=out + inp + bits)
    qw, qz, sc = core.pack_marlin(codes, zeros, scales, bits, wz)
    # vLLM works on q_w [size_k, size_n] = codes.T and s / zp [groups, size_n]
    np.testing.assert_array_equal(qw, ref_marlin_weights(codes.T.copy(), inp, out, bits))
    ref_s = ref_permute_scales(scales.T.astype(np.float16), inp, out, gs)
    np.testing.assert_array_equal(sc.view(np.uint16), ref_s.view(np.uint16))
    if wz:
        assert qz is not None
        np.testing.assert_array_equal(qz, ref_zero_points(zeros.T.copy(), out, bits))
    else:
        assert qz is None


@pytest.mark.parametrize(("out", "inp", "gs", "bits", "wz"), CASES)
def test_marlin_round_trip(out: int, inp: int, gs: int, bits: int, wz: bool) -> None:
    codes, zeros, scales = _inputs(out, inp, gs, bits, seed=1)
    qw, qz, sc = core.pack_marlin(codes, zeros, scales, bits, wz)
    c2, z2, s2 = core.unpack_marlin(qw, qz, sc, out, inp, gs, bits)
    np.testing.assert_array_equal(c2, codes)
    np.testing.assert_array_equal(s2, scales)
    if wz:
        np.testing.assert_array_equal(z2, zeros)
    else:
        assert z2 is None


def test_marlin_rejects_bad_shapes() -> None:
    codes, zeros, scales = _inputs(32, 128, 128, 4, seed=2)
    with pytest.raises(ValueError, match="out % 64"):
        core.pack_marlin(codes, zeros, scales, 4, False)
    codes, zeros, scales = _inputs(64, 128, 128, 3, seed=3)
    with pytest.raises(ValueError, match="4 or 8 bit"):
        core.pack_marlin(codes, zeros, scales, 3, False)


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vllm not installed")
def test_marlin_matches_installed_vllm() -> None:
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_zero_points
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights

    out, inp, gs, bits = 192, 256, 64, 4
    codes, zeros, scales = _inputs(out, inp, gs, bits, seed=4)
    qw, qz, sc = core.pack_marlin(codes, zeros, scales, bits, True)
    ref_qw = marlin_weights(torch.from_numpy(codes.T.astype(np.int32).copy()), inp, out, bits, get_weight_perm(bits))
    np.testing.assert_array_equal(qw, ref_qw.numpy())
    ref_s = marlin_permute_scales(torch.from_numpy(scales.T.astype(np.float16).copy()), inp, out, gs)
    np.testing.assert_array_equal(sc.view(np.uint16), ref_s.numpy().view(np.uint16))
    ref_z = marlin_zero_points(torch.from_numpy(zeros.T.astype(np.int32).copy()), inp // gs, out, bits)
    np.testing.assert_array_equal(qz, ref_z.numpy())
