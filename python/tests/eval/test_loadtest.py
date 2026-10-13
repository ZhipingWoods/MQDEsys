"""Load tester (T-0803) against the mock backend: latency accuracy, arrival modes, failures, prompts."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import mock_backend

from mqde.eval import loadtest
from mqde.eval.loadtest import LoadSpec, make_prompt, poisson_schedule, run_load

TTFT_MS, TPOT_MS = 100.0, 20.0  # the mock's ground truth
TOL_MS = 2.0  # T-0803 acceptance: measurement error < 2 ms (per-token timing: TPOT, ITL)
# TTFT / E2E also contain the mock's own request-start delay (a Python thread per connection, parsing the request
# before its clock starts). That is ~0.2 ms on Linux and up to ~4 ms on macOS, a shift of the whole stream that
# TPOT and ITL, which are differences between tokens, do not see.
START_TOL_MS = 2.0 if sys.platform.startswith("linux") else 5.0


@pytest.mark.parametrize("api", ["openai", "llama"])
def test_latency_accuracy_closed_loop(api: str, tmp_path: Path) -> None:
    n = 16
    with mock_backend.llama(tmp_path) as srv:
        r = run_load(srv.port, LoadSpec(api=api, concurrency=4, requests=8, prompt_len=32, n_predict=n, warmup=1))
    assert r["ok"] == 8 and r["failed"] == 0 and r["errors"] == []
    assert r["ttft_ms"]["p50"] == pytest.approx(TTFT_MS, abs=START_TOL_MS)
    assert r["tpot_ms"]["p50"] == pytest.approx(TPOT_MS, abs=TOL_MS)
    assert r["itl_ms"]["p50"] == pytest.approx(TPOT_MS, abs=TOL_MS)
    assert r["e2e_ms"]["p50"] == pytest.approx(TTFT_MS + (n - 1) * TPOT_MS, abs=START_TOL_MS * 2)
    assert r["output_tokens"] == 8 * n
    # 4 requests overlap, so output throughput is ~4x one stream's n / e2e
    assert r["output_tokens_per_s"] == pytest.approx(4 * n / ((TTFT_MS + (n - 1) * TPOT_MS) / 1e3), rel=0.25)
    assert r["client_cpu_frac"] < 0.2  # the generator mostly waits on I/O


def test_open_loop_poisson(tmp_path: Path) -> None:
    rate, n = 20.0, 30
    with mock_backend.vllm(tmp_path) as srv:
        r = run_load(srv.port, LoadSpec(mode="open", rate=rate, requests=n, prompt_len=16, n_predict=4, warmup=0, seed=3))
    assert r["ok"] == n and r["offered_rate"] == rate
    assert r["schedule_lag_ms"]["p99"] < 25  # starts keep to the schedule
    # requests finish ~ttft+3*tpot after arriving, so wall ~ last arrival + that
    expected_wall = poisson_schedule(rate, n, 3)[-1] + (TTFT_MS + 3 * TPOT_MS) / 1e3
    assert r["wall_s"] == pytest.approx(expected_wall, abs=0.15)
    assert r["requests_per_s"] == pytest.approx(n / expected_wall, rel=0.1)


def test_poisson_schedule_is_seeded_and_has_the_right_rate() -> None:
    a = poisson_schedule(50.0, 5000, seed=1)
    assert a == poisson_schedule(50.0, 5000, seed=1) and a != poisson_schedule(50.0, 5000, seed=2)
    assert all(y > x for x, y in itertools.pairwise(a))
    assert 5000 / a[-1] == pytest.approx(50.0, rel=0.05)
    gaps = [y - x for x, y in itertools.pairwise(a)]
    mean = sum(gaps) / len(gaps)
    cv = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean
    assert cv == pytest.approx(1.0, abs=0.1)  # exponential gaps: coefficient of variation 1


def test_failures_are_counted_not_raised(tmp_path: Path) -> None:
    with mock_backend.vllm(tmp_path, max_model_len=64) as srv:
        r = run_load(srv.port, LoadSpec(requests=3, concurrency=3, prompt_len=60, n_predict=16, warmup=0))
    assert r["ok"] == 0 and r["failed"] == 3
    assert any("HTTP 400" in e for e in r["errors"])
    assert "ttft_ms" not in r  # no successful request, no latency statistics


@pytest.mark.parametrize("model", ["x", None])  # None: the model name would be looked up on the dead server
def test_unreachable_server_is_a_failure_not_a_crash(model: str | None) -> None:
    r = run_load(1, LoadSpec(requests=2, concurrency=1, warmup=0, timeout=2, model=model))
    assert r["ok"] == 0 and r["failed"] == 2 and r["errors"]


def test_open_loop_needs_a_rate() -> None:
    with pytest.raises(ValueError, match="rate"):
        run_load(1, LoadSpec(mode="open", rate=0, requests=1, model="x"))


def test_prompts_have_exact_length_and_unique_prefixes() -> None:
    assert all(len(make_prompt(i, n, 1000)) == n for i in (0, 7, 12345) for n in (1, 4, 256, 4096))
    heads = {tuple(make_prompt(i, 16, 1000)[:4]) for i in range(5000)}
    assert len(heads) == 5000  # no two requests share a prefix, so prefix caching cannot help
    assert make_prompt(3, 100, 1000)[10:] == [1000] * 90


# ---- T-R12: concurrency sweep + saturation point ----
def _pt(c: int, tps: float, ttft99: float = 100.0) -> dict:
    return {"concurrency": c, "output_tokens_per_s": tps, "ok": 8, "failed": 0, "ttft_ms": {"p50": ttft99 / 2, "p99": ttft99}}


def test_sweep_runs_each_level_ascending_and_reports_per_level_percentiles(tmp_path: Path) -> None:
    with mock_backend.vllm(tmp_path) as srv:
        out = loadtest.sweep(srv.port, [4, 1, 2], requests_per_level=4, prompt_len=16, n_predict=4)
    assert [p["concurrency"] for p in out["points"]] == [1, 2, 4], "ascending, whatever order was asked for"
    for p in out["points"]:
        assert p["ok"] == max(4, 2 * p["concurrency"]) and p["failed"] == 0 and p["ttft_ms"]["p99"] >= p["ttft_ms"]["p50"] > 0 and p["tpot_ms"]["p50"] > 0
        assert p["itl_ms"]["p99"] >= p["itl_ms"]["p50"] > 0 and p["spec"]["concurrency"] == p["concurrency"]
    tps = [p["output_tokens_per_s"] for p in out["points"]]
    assert tps[2] > tps[0] * 1.5, "the mock's latency does not depend on load: more workers, more tokens per second"
    assert out["peak"]["concurrency"] == 4 and out["peak"]["output_tokens_per_s"] == tps[2]


def test_sweep_requests_scale_with_concurrency_so_every_worker_gets_work(tmp_path: Path) -> None:
    with mock_backend.vllm(tmp_path) as srv:
        out = loadtest.sweep(srv.port, [1, 4], requests_per_level=2, prompt_len=16, n_predict=2)
    assert [p["ok"] for p in out["points"]] == [2, 8], "max(requests_per_level, 2 x concurrency): 4 workers x 2 rounds"


def test_sweep_rejects_nonsense_levels() -> None:
    for bad in ([], [0], [-1, 2], [1, 1]):
        with pytest.raises(ValueError):
            loadtest.sweep(1, bad)


def test_saturation_is_where_another_doubling_stops_paying() -> None:
    pts = [_pt(1, 100), _pt(2, 195), _pt(4, 370), _pt(8, 400), _pt(16, 405)]
    s = loadtest.saturation(pts)
    assert s["concurrency"] == 4 and s["reason"] == "throughput"
    assert s["next_gain_pct"] == pytest.approx(100 * (400 / 370 - 1))


def test_saturation_also_stops_at_a_latency_blowup() -> None:
    pts = [_pt(1, 100, 80), _pt(2, 200, 90), _pt(4, 400, 700), _pt(8, 800, 2000)]
    s = loadtest.saturation(pts, max_ttft_p99_ms=500)
    assert s["concurrency"] == 2 and s["reason"] == "ttft", "throughput still doubles but TTFT p99 breaks the budget at 4"


def test_saturation_ignores_levels_with_failures_and_handles_degenerate_input() -> None:
    bad = {**_pt(4, 900), "failed": 3}
    assert loadtest.saturation([_pt(1, 100), _pt(2, 190), bad])["concurrency"] == 2
    assert loadtest.saturation([_pt(1, 100)]) == {"concurrency": 1, "reason": "single-level", "next_gain_pct": None}
    assert loadtest.saturation([]) is None
    never = loadtest.saturation([_pt(1, 100), _pt(2, 200), _pt(4, 400)])
    assert never["concurrency"] == 4 and never["reason"] == "not-reached", "still scaling at the top of the range"


def test_peak_is_the_best_throughput_not_the_highest_concurrency_and_a_dead_level_does_not_break_it(monkeypatch: pytest.MonkeyPatch) -> None:
    script = {1: {"output_tokens_per_s": 100.0, "ok": 4, "failed": 0}, 2: {"output_tokens_per_s": 300.0, "ok": 4, "failed": 0},
              4: {"output_tokens_per_s": 250.0, "ok": 8, "failed": 0}, 8: {"ok": 0, "failed": 16}}  # overloaded: nothing finished

    monkeypatch.setattr(loadtest, "run_load", lambda port, spec: dict(script[spec.concurrency]))
    out = loadtest.sweep(1, [1, 2, 4, 8])
    assert out["peak"]["concurrency"] == 2 and out["peak"]["output_tokens_per_s"] == 300.0
    assert out["saturation"]["concurrency"] == 2 and out["saturation"]["reason"] == "throughput", "4 is slower than 2"
    assert [p["concurrency"] for p in out["points"]] == [1, 2, 4, 8] and out["points"][-1]["failed"] == 16
    monkeypatch.setattr(loadtest, "run_load", lambda port, spec: {"ok": 0, "failed": 4})
    dead = loadtest.sweep(1, [1, 2])
    assert dead["peak"] is None and dead["saturation"] is None, "no level served anything: no peak, no saturation"
