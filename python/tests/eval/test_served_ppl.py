"""Served-model perplexity (T-0802 / S12): protocol and error handling, against the mock backend."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mock_backend

from mqde.eval.served_ppl import served_perplexity, window_nll


def test_ppl_matches_the_known_logprob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LOGPROB", "-1.5")
    ids = list(range(1000, 1000 + 5 * 64 + 10))  # 5 full windows of 64 and a 10-token tail that must be ignored
    with mock_backend.vllm(tmp_path, max_model_len=256) as srv:
        r = served_perplexity(srv.port, ids, seq_len=64)
        two = served_perplexity(srv.port, ids, seq_len=64, max_windows=2, concurrency=1)
    assert r["windows"] == 5 and r["tokens"] == 5 * 63  # the first token of every window is not scored
    assert r["ppl"] == pytest.approx(math.exp(1.5), rel=1e-9)
    assert two["windows"] == 2 and two["tokens"] == 2 * 63


def test_window_nll_is_the_sum_over_tokens_after_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LOGPROB", "-0.25")
    with mock_backend.vllm(tmp_path, max_model_len=256) as srv:
        assert window_nll(srv.port, list(range(10, 18)), None, 30) == pytest.approx(7 * 0.25)


def test_too_few_tokens() -> None:
    with pytest.raises(ValueError, match="at least 64"):
        served_perplexity(1, list(range(10)), seq_len=64, model="x")


def test_backend_without_prompt_logprobs_is_reported(tmp_path: Path) -> None:
    """llama-server's /v1/completions has no prompt_logprobs: the caller must get a clear error, not a number."""
    with mock_backend.llama(tmp_path) as srv, pytest.raises(RuntimeError):
        served_perplexity(srv.port, list(range(200)), seq_len=64, model="m", timeout=10)
