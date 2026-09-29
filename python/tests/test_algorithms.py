"""Algorithm sanity: each method must not be worse than RTN on output error, and exact transforms
must be FP-identity before quantization."""

from __future__ import annotations

import pytest
import torch

from mqde.quant import algorithms
from mqde.quant.numerics import QuantSpec, fake_quant, fake_quant_act

W4G = QuantSpec(format="int", bits=4, symmetric=False, granularity="per_group", group_size=32, calib="minmax")
W3C = QuantSpec(format="int", bits=3, symmetric=True, granularity="per_channel", calib="minmax")
A8 = QuantSpec(format="int", bits=8, symmetric=True, granularity="per_channel", calib="minmax")
NONE = QuantSpec()


def _layer(seed: int = 0, rows: int = 64, cols: int = 128, n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.05
    # correlated activations with a few outlier channels (the regime AWQ/SmoothQuant target)
    base = torch.randn(n, cols // 4, generator=g)
    x = base.repeat_interleave(4, dim=1) + 0.3 * torch.randn(n, cols, generator=g)
    x[:, :4] *= 20.0
    return w, x


def _out_mse(algo: str, w: torch.Tensor, x: torch.Tensor, ws: QuantSpec, as_: QuantSpec) -> float:
    r = algorithms.run(algo, w, x, ws, as_)
    wq = r.w if r.final else fake_quant(r.w, ws)[0]
    xs = x / r.in_scale if r.in_scale is not None else x
    xq = fake_quant_act(xs, as_) if as_.enabled else xs
    return float((x @ w.T - xq @ wq.T).pow(2).mean())


def test_registry_and_resolve() -> None:
    assert {"rtn", "gptq", "awq", "smoothquant"} <= set(algorithms.ALGO_REGISTRY)
    assert algorithms.resolve("awq+gptq") == ["awq", "gptq"]
    assert algorithms.resolve("smoothquant") == ["smoothquant", "rtn"]
    assert algorithms.resolve("none") == ["rtn"]
    with pytest.raises(ValueError):
        algorithms.resolve("foo")
    with pytest.raises(ValueError):
        algorithms.resolve("awq+smoothquant")
    names = {a["name"] for a in algorithms.algorithm_catalog()}
    assert {"rtn", "gptq", "awq", "smoothquant", "awq+gptq", "smoothquant+gptq"} <= names


@pytest.mark.parametrize("ws", [W4G, W3C], ids=["w4g32", "w3ch"])
def test_gptq_beats_rtn(ws: QuantSpec) -> None:
    w, x = _layer()
    rtn, gptq = _out_mse("rtn", w, x, ws, NONE), _out_mse("gptq", w, x, ws, NONE)
    assert gptq < 0.8 * rtn, (rtn, gptq)


def test_gptq_output_lies_on_quant_grid() -> None:
    w, x = _layer()
    r = algorithms.run("gptq", w, x, W4G, NONE)
    assert r.final
    # every group of 32 has at most 16 distinct values (4-bit)
    g = r.w.reshape(-1, 32)
    assert max(len(torch.unique(row)) for row in g) <= 16


def test_awq_not_worse_than_rtn() -> None:
    w, x = _layer()
    assert _out_mse("awq", w, x, W4G, NONE) <= _out_mse("rtn", w, x, W4G, NONE) * 1.0001


def test_smoothquant_is_fp_identity_and_helps_w8a8() -> None:
    w, x = _layer()
    r = algorithms.run("smoothquant", w, x, W4G, A8)
    assert r.in_scale is not None
    torch.testing.assert_close((x / r.in_scale) @ r.w.T, x @ w.T, rtol=1e-4, atol=1e-4)
    w8 = QuantSpec(format="int", bits=8, symmetric=True, granularity="per_channel", calib="minmax")
    a8t = QuantSpec(format="int", bits=8, symmetric=True, granularity="per_tensor", calib="minmax")
    assert _out_mse("smoothquant", w, x, w8, a8t) < _out_mse("rtn", w, x, w8, a8t)


def test_combo_awq_gptq_not_worse_than_awq() -> None:
    w, x = _layer(seed=1)
    assert _out_mse("awq+gptq", w, x, W4G, NONE) <= _out_mse("awq", w, x, W4G, NONE)


def test_fp8_gptq_falls_back() -> None:
    w, x = _layer()
    r = algorithms.run("gptq", w, x, QuantSpec(format="fp8_e4m3", bits=8, granularity="per_channel"), NONE)
    assert not r.final and "RTN" in str(r.info)
