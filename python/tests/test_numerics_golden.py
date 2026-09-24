"""The Torch quantization path must reproduce the C++ core (the numerical source of truth)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mqde import core
from mqde.quant.numerics import QuantSpec, fake_quant

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

MINMAX_SPECS = [
    {"format": "int", "bits": 4, "symmetric": True, "granularity": "per_channel"},
    {"format": "int", "bits": 4, "symmetric": False, "granularity": "per_group", "group_size": 128},
    {"format": "int", "bits": 2, "symmetric": False, "granularity": "per_group", "group_size": 64},
    {"format": "int", "bits": 8, "symmetric": True, "granularity": "per_tensor"},
    {"format": "int", "bits": 8, "symmetric": False, "granularity": "per_channel"},
    {"format": "int", "bits": 3, "symmetric": True, "granularity": "per_group", "group_size": 32},
    {"format": "fp8_e4m3", "bits": 8, "granularity": "per_channel"},
    {"format": "fp8_e5m2", "bits": 8, "granularity": "per_tensor"},
]


def _data(seed: int, rows: int = 64, cols: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 0.05
    x[rng.integers(0, rows, 8), rng.integers(0, cols, 8)] = 1.5  # outliers
    x[3] = 0.0  # an all-zero row exercises the scale fallback
    return x


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spec", MINMAX_SPECS, ids=lambda s: f"{s['format']}{s['bits']}-{s['granularity']}-{s.get('symmetric', True)}")
def test_minmax_bit_exact(spec: dict, device: str) -> None:
    for seed in range(3):
        x = _data(seed)
        ref, ref_scale, ref_zero = core.fake_quant(x, spec)
        out, scale, zero = fake_quant(torch.from_numpy(x).to(device), QuantSpec.from_dict(spec))
        np.testing.assert_array_equal(out.cpu().numpy(), ref)
        np.testing.assert_array_equal(scale.cpu().numpy().reshape(ref_scale.shape), ref_scale)
        if ref_zero is not None:
            np.testing.assert_array_equal(zero.cpu().numpy().reshape(ref_zero.shape), ref_zero)


@pytest.mark.parametrize("device", DEVICES)
def test_mse_close(device: str) -> None:
    spec = {"format": "int", "bits": 4, "symmetric": False, "granularity": "per_group", "group_size": 128, "calib": "mse"}
    x = _data(7)
    ref, _, _ = core.fake_quant(x, spec)
    out, _, _ = fake_quant(torch.from_numpy(x).to(device), QuantSpec.from_dict(spec))
    mismatch = (out.cpu().numpy() != ref).mean()
    assert mismatch < 0.01, f"{mismatch:.4%} of elements differ"
    assert core.measure(x, out.cpu().numpy())["mse"] <= core.measure(x, ref)["mse"] * 1.01


def test_fp8_round_matches_torch_cast() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(100_000).astype(np.float32) * s for s in (1e-3, 1.0, 100.0)])
    x = np.clip(x, -448, 448)
    ref = torch.from_numpy(x).to(torch.float8_e4m3fn).to(torch.float32).numpy()
    np.testing.assert_array_equal(core.fp8_round(x, "e4m3"), ref)
    x5 = np.clip(x * 100, -57344, 57344).astype(np.float32)
    ref5 = torch.from_numpy(x5).to(torch.float8_e5m2).to(torch.float32).numpy()
    np.testing.assert_array_equal(core.fp8_round(x5, "e5m2"), ref5)


def test_real_codes_dequantize_to_fake() -> None:
    spec = {"format": "int", "bits": 8, "symmetric": False, "granularity": "per_group", "group_size": 64}
    x = _data(1)
    fake, _, _ = core.fake_quant(x, spec)
    codes, scale, zero = core.quantize_int(x, spec)
    assert codes.min() >= 0 and codes.max() <= 255
    deq = (codes.reshape(64, -1, 64).astype(np.float32) - zero[..., None]) * scale[..., None]
    np.testing.assert_array_equal(deq.reshape(x.shape), fake)


def test_metrics_match_numpy() -> None:
    a = _data(2).ravel()
    b = a + np.float32(0.01) * np.sin(np.arange(a.size, dtype=np.float32))
    m = core.measure(a, b)
    d = a.astype(np.float64) - b
    assert m["mse"] == pytest.approx(np.mean(d * d), rel=1e-9)
    cos = np.dot(a.astype(np.float64), b.astype(np.float64)) / np.linalg.norm(a.astype(np.float64)) / np.linalg.norm(b.astype(np.float64))
    assert m["cosine"] == pytest.approx(cos, rel=1e-9)
