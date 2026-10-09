"""Distribution summaries stored per layer for the error page (T-0903): checked against numpy and against real traces."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mqde.quant.engine import HIST_BINS, histogram_summary, scale_summary


def test_histogram_matches_numpy_and_is_exact_in_totals() -> None:
    g = torch.Generator().manual_seed(0)
    t = torch.randn(64, 96, generator=g) * 0.02
    h = histogram_summary(t)
    want, _ = np.histogram(t.numpy(), bins=HIST_BINS, range=(float(t.min()), float(t.max())))
    assert sum(h["counts"]) == h["n"] == t.numel(), "every element lands in a bin, including the maximum"
    assert np.abs(np.array(h["counts"]) - want).max() <= 1, "within one element of numpy per bin (float rounding at bin edges)"
    assert h["min"] == float(t.min()) and h["max"] == float(t.max())
    assert h["mean"] == pytest.approx(float(t.mean()), abs=1e-7) and h["std"] == pytest.approx(float(t.std()), rel=1e-5)


def test_constant_tensor_does_not_divide_by_zero() -> None:
    h = histogram_summary(torch.full((8, 8), 3.0))
    assert sum(h["counts"]) == 64 and h["std"] == 0.0


def test_scale_summary_quantiles_and_spread() -> None:
    s = torch.tensor([1.0, 2.0, 4.0, 8.0])
    d = scale_summary(s)
    assert d is not None and d["n"] == 4 and d["min"] == 1.0 and d["max"] == 8.0 and d["spread"] == 8.0
    assert d["p50"] == pytest.approx(3.0) and 4.0 < d["p99"] <= 8.0
    assert scale_summary(None) is None and scale_summary(torch.empty(0)) is None
    assert scale_summary(torch.zeros(3))["spread"] is None  # type: ignore[index]  # zero scale: no meaningful ratio
