"""Logits probe (T-0510): KL, Top-1 agreement and the top-k reference, checked against closed forms and invariants."""

from __future__ import annotations

import math

import pytest
import torch

from mqde.eval.logits_probe import (
    capture_reference,
    compare_to_reference,
    probe_error,
    split_windows,
)

V = 200
TOL = 1e-4  # float32 rounding of log-sum-exp


class Table:
    """A fake 'model': a fixed logits table per window index, so reference and quantized outputs are fully known."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits  # [n_windows, seq, vocab]
        self.calls = 0

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        i = int(w[0, 0])  # the first token id of a window is its index
        self.calls += 1
        return self.logits[i : i + 1]


def windows(n: int, s: int) -> torch.Tensor:
    w = torch.zeros(n, s, dtype=torch.long)
    w[:, 0] = torch.arange(n)
    return w


def pair(n: int = 3, s: int = 16, noise: float = 0.5, seed: int = 0) -> tuple[Table, Table, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n, s, V, generator=g) * 3.0
    return Table(r), Table(r + torch.randn(n, s, V, generator=g) * noise), windows(n, s)


def test_identical_models_have_zero_kl_and_full_agreement() -> None:
    r, _, w = pair()
    for exact in (True, False):
        ref = capture_reference(r, w, k=32, exact=exact)
        m = compare_to_reference(r, w, ref)
        assert abs(m["kl"]) < TOL and m["top1_agreement"] == 1.0 and abs(m["ref_top1_prob_drop"]) < TOL


def test_exact_kl_matches_the_closed_form() -> None:
    """For q = softmax(logits + shift on one token) the KL is known in closed form from log-sum-exps."""
    n, s = 2, 8
    g = torch.Generator().manual_seed(1)
    base = torch.randn(n, s, V, generator=g)
    shifted = base.clone()
    shifted[..., 7] += 2.0
    ref = capture_reference(Table(base), windows(n, s), exact=True)
    m = compare_to_reference(Table(shifted), windows(n, s), ref)
    # KL(p||q) = sum p (log p - log q);  log q = x + shift - lse_q ; log p = x - lse_p  =>  KL = lse_q - lse_p - 2 * p[7]
    p = torch.softmax(base, -1)
    want = (torch.logsumexp(shifted, -1) - torch.logsumexp(base, -1) - 2.0 * p[..., 7]).mean()
    assert m["kl"] == pytest.approx(float(want), abs=TOL)
    assert m["kl"] > 0


@pytest.mark.parametrize("k", [4, 16, 64])
def test_topk_kl_is_a_lower_bound_of_the_exact_kl(k: int) -> None:
    r, q, w = pair(noise=0.8)
    exact = compare_to_reference(q, w, capture_reference(r, w, k=k, exact=True))["kl"]
    approx = compare_to_reference(q, w, capture_reference(r, w, k=k, exact=False))["kl"]
    assert 0 < approx <= exact + TOL
    gap = probe_error(r, q, w, k=k)
    assert gap["exact_kl"] == pytest.approx(exact, abs=TOL) and gap["topk_kl"] == pytest.approx(approx, abs=TOL)
    assert gap["relative_gap"] == pytest.approx((exact - approx) / exact, abs=1e-3)


def test_the_gap_shrinks_as_k_grows_and_vanishes_at_the_vocabulary() -> None:
    r, q, w = pair(noise=0.8)
    gaps = [probe_error(r, q, w, k=k)["relative_gap"] for k in (4, 16, 64, V)]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0
    assert abs(gaps[3]) < 1e-3  # k = vocab: every token is its own bucket


def test_top1_agreement_counts_positions() -> None:
    n, s = 1, 10
    r = torch.zeros(n, s, V)
    r[..., 5] = 4.0  # reference always says token 5
    q = r.clone()
    q[0, :3, 5] = 0.0
    q[0, :3, 9] = 4.0  # the quantized model says 9 at the first three positions
    ref = capture_reference(Table(r), windows(n, s), exact=False)
    m = compare_to_reference(Table(q), windows(n, s), ref)
    assert m["top1_agreement"] == pytest.approx(0.7)
    assert m["positions"] == 10
    # the reference's top-1 probability drops only at the three disagreeing positions
    p_ref = math.exp(4.0) / (math.exp(4.0) + V - 1)
    p_q_wrong = 1.0 / (math.exp(4.0) + V - 1)
    assert m["ref_top1_prob_drop"] == pytest.approx(0.3 * (p_ref - p_q_wrong), abs=1e-4)


def test_kl_grows_with_the_perturbation() -> None:
    kls = []
    for noise in (0.0, 0.3, 1.0, 3.0):
        r, q, w = pair(noise=noise)
        kls.append(compare_to_reference(q, w, capture_reference(r, w, k=64))["kl"])
    assert kls[0] < TOL and kls == sorted(kls) and kls[3] > 10 * kls[1]


def test_reference_is_captured_once_and_reused() -> None:
    r, q, w = pair()
    ref = capture_reference(r, w, k=16)
    calls = r.calls
    for _ in range(3):
        compare_to_reference(q, w, ref)
    assert r.calls == calls  # comparing never runs the reference model again (it no longer exists after quantization)


def test_topk_reference_is_small_exact_reference_is_large() -> None:
    r, _, w = pair(n=2, s=16)
    small = capture_reference(r, w, k=8, exact=False).nbytes()
    big = capture_reference(r, w, k=8, exact=True).nbytes()
    assert small == 2 * 16 * 8 * (4 + 8) + 2 * 16 * 4  # fp32 values + int64 indices + fp32 lse
    assert big == small + 2 * 16 * V * 2  # the exact reference adds every logit as float16
    # at a real vocabulary the difference is what matters: 152k logits against 256 per position
    assert 151936 * 2 / (256 * (4 + 8) + 4) > 90


def test_windows_helper_and_shape_checks() -> None:
    ids = torch.arange(100)
    assert split_windows(ids, 30).shape == (3, 30) and split_windows(ids, 30, 2).shape == (2, 30)
    assert torch.equal(split_windows(ids, 30)[1], ids[30:60])  # non-overlapping, same windows as eval.ppl
    with pytest.raises(ValueError, match="at least 200"):
        split_windows(ids, 200)
    r, q, w = pair(n=3, s=16)
    ref = capture_reference(r, w, k=8)
    with pytest.raises(ValueError, match="differ from the reference"):
        compare_to_reference(q, windows(2, 16), ref)
    with pytest.raises(ValueError, match="differ from the reference"):
        compare_to_reference(q, windows(3, 8), ref)


def test_progress_callback() -> None:
    r, q, w = pair(n=3)
    seen: list[tuple[int, int]] = []
    compare_to_reference(q, w, capture_reference(r, w, k=8), progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------- per-sample (variable length) probe: VLM image+text inputs
from mqde.eval.logits_probe import capture_sample_reference, compare_to_sample_reference


class Samples:
    """A fake multimodal model: sample i has its own length and logits table; `fn(i)` returns [len_i, vocab]."""

    def __init__(self, tables: list[torch.Tensor]) -> None:
        self.tables = tables

    def __call__(self, i: int) -> torch.Tensor:
        return self.tables[i][None]


def tables(lens: list[int], noise: float = 0.0, seed: int = 0) -> tuple[Samples, Samples]:
    g = torch.Generator().manual_seed(seed)
    ref = [torch.randn(n, V, generator=g) * 2 for n in lens]
    q = [r + noise * torch.randn(r.shape, generator=g) for r in ref]
    return Samples(ref), Samples(q)


def masks(lens: list[int], start: list[int]) -> list[torch.Tensor]:
    return [torch.arange(n) >= s for n, s in zip(lens, start, strict=True)]


def test_variable_lengths_and_identical_models_give_zero_kl() -> None:
    lens = [5, 9, 3]
    ref, q = tables(lens)
    m = masks(lens, [0, 0, 0])
    r = capture_sample_reference(ref, len(lens), m)
    got = compare_to_sample_reference(q, r)
    assert got["kl"] == pytest.approx(0, abs=TOL) and got["top1_agreement"] == 1.0
    assert got["positions"] == sum(lens) and got["samples"] == 3


def test_only_masked_positions_are_compared() -> None:
    """The prompt (image + question) positions are not answers: perturbing them must not change the result."""
    lens = [8, 8]
    ref, q = tables(lens)
    for t in q.tables:
        t[:4] += 50.0 * torch.randn(4, V)  # destroy the first 4 positions of every sample
    m = masks(lens, [4, 4])
    got = compare_to_sample_reference(q, capture_sample_reference(ref, 2, m))
    assert got["kl"] == pytest.approx(0, abs=TOL) and got["top1_agreement"] == 1.0
    assert got["positions"] == 8, "only the 4 answer positions of each of the 2 samples"
    unmasked = compare_to_sample_reference(q, capture_sample_reference(ref, 2, masks(lens, [0, 0])))
    assert unmasked["kl"] > 1.0, "the same damage is visible once those positions are included"


def test_matches_the_windowed_probe_on_equal_length_inputs() -> None:
    a, b, w = pair(n=3, s=16, noise=0.7)
    base = compare_to_reference(b, w, capture_reference(a, w, k=32, exact=True))
    sa = Samples([a.logits[i] for i in range(3)])
    sb = Samples([b.logits[i] for i in range(3)])
    got = compare_to_sample_reference(sb, capture_sample_reference(sa, 3, masks([16] * 3, [0] * 3), k=32, exact=True))
    assert got["kl"] == pytest.approx(base["kl"], rel=1e-6) and got["top1_agreement"] == pytest.approx(base["top1_agreement"])
    assert got["ref_top1_prob_drop"] == pytest.approx(base["ref_top1_prob_drop"], abs=1e-6)


def test_topk_sample_probe_is_a_lower_bound_of_exact() -> None:
    lens = [7, 11]
    ref, q = tables(lens, noise=0.8)
    m = masks(lens, [2, 3])
    exact = compare_to_sample_reference(q, capture_sample_reference(ref, 2, m, k=16, exact=True))
    topk = compare_to_sample_reference(q, capture_sample_reference(ref, 2, m, k=16, exact=False))
    assert 0 < topk["kl"] <= exact["kl"] + TOL


def test_a_sample_whose_length_changed_is_rejected() -> None:
    ref, q = tables([6, 6])
    r = capture_sample_reference(ref, 2, masks([6, 6], [0, 0]))
    bad = Samples([q.tables[0], q.tables[1][:4]])  # sample 1 now has 4 positions instead of 6
    with pytest.raises(ValueError, match="sample 1"):
        compare_to_sample_reference(bad, r)


def test_an_empty_mask_is_rejected_instead_of_dividing_by_zero() -> None:
    ref, _ = tables([4])
    with pytest.raises(ValueError, match="no positions"):
        capture_sample_reference(ref, 1, [torch.zeros(4, dtype=torch.bool)])
