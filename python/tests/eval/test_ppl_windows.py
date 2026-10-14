"""perplexity() also reports each window's own mean loss (what the report's per-window review is drawn from), without
changing the PPL it has always returned."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mqde.eval.ppl import perplexity

VOCAB = 17


def logits_fn(w: torch.Tensor) -> torch.Tensor:
    """A deterministic 'model' whose logits depend on the previous token, so windows differ in difficulty."""
    g = torch.Generator().manual_seed(3)
    table = torch.randn(VOCAB, VOCAB, generator=g) * 2.0
    return table[w[0]].unsqueeze(0)


def ids(n: int = 7 * 32) -> torch.Tensor:
    return torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(11))


def reference(ids_: torch.Tensor, seq_len: int, windows: int) -> list[float]:
    out = []
    for i in range(windows):
        w = ids_[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
        lg = logits_fn(w).float()
        out.append(float(F.cross_entropy(lg[0, :-1], w[0, 1:], reduction="mean")))
    return out


def test_one_mean_loss_per_window_matching_an_independent_computation() -> None:
    x = ids()
    r = perplexity(logits_fn, x, seq_len=32)
    assert r["windows"] == 7 and len(r["window_nll"]) == 7
    for got, want in zip(r["window_nll"], reference(x, 32, 7), strict=True):
        assert math.isclose(got, want, rel_tol=1e-6)


def test_the_ppl_is_exactly_what_it_was_and_is_the_exp_of_the_mean_window_loss() -> None:
    x = ids()
    r = perplexity(logits_fn, x, seq_len=32)
    nll, count = 0.0, 0
    for i in range(7):
        w = x[i * 32 : (i + 1) * 32].unsqueeze(0)
        nll += float(F.cross_entropy(logits_fn(w).float()[0, :-1], w[0, 1:], reduction="sum"))
        count += 31
    assert r["ppl"] == math.exp(nll / count), "PPL must not change: reports written before this feature compare against it"
    assert math.isclose(r["ppl"], math.exp(sum(r["window_nll"]) / 7), rel_tol=1e-9)


def test_max_windows_limits_the_list() -> None:
    r = perplexity(logits_fn, ids(), seq_len=32, max_windows=3)
    assert r["windows"] == 3 and len(r["window_nll"]) == 3


def test_the_windows_are_not_all_alike() -> None:
    r = perplexity(logits_fn, ids(), seq_len=32)
    assert len({round(v, 6) for v in r["window_nll"]}) > 1, "the per-window list would be useless if it were constant"
