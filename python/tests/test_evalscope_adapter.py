"""EvalScope adapter contract: exact CLI arguments and report normalization (no evalscope install needed)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from mqde.eval import evalscope_adapter as es


def test_build_command_uses_the_openai_compatible_endpoint() -> None:
    cmd = es.build_command("/opt/env/bin/evalscope", 18000, "served-model", ["gsm8k", "mmlu_pro"], "/tmp/out",
                           limit=100, generation_config="temperature=0,max_tokens=512", eval_batch_size=8, seed=7)
    assert cmd[:2] == ["/opt/env/bin/evalscope", "eval"]
    assert cmd[cmd.index("--api-url") + 1] == "http://127.0.0.1:18000/v1"
    assert cmd[cmd.index("--eval-type") + 1] == "openai_api"
    assert cmd[cmd.index("--datasets") + 1:cmd.index("--datasets") + 3] == ["gsm8k", "mmlu_pro"]
    assert cmd[cmd.index("--work-dir") + 1] == "/tmp/out"
    assert "--no-timestamp" in cmd
    assert cmd[cmd.index("--limit") + 1] == "100"
    assert cmd[cmd.index("--generation-config") + 1] == "temperature=0,max_tokens=512"
    assert cmd[cmd.index("--eval-batch-size") + 1] == "8"
    assert cmd[cmd.index("--seed") + 1] == "7"


def test_empty_datasets_are_rejected() -> None:
    with pytest.raises(ValueError):
        es.build_command("evalscope", 1, "m", [], "/tmp/out")


def test_parse_report_reads_legacy_and_current_metric_shapes() -> None:
    legacy = {
        "dataset_name": "gsm8k",
        "dataset_pretty_name": "GSM8K",
        "model_name": "m",
        "score": 0.42,
        "metrics": [{"name": "mean_acc", "num": 100, "score": 0.42}],
        "perf_metrics": {"summary": {"n_samples": 100, "latency": {"mean": 12.5}}},
    }
    current = {
        "dataset_name": "mmlu_pro",
        "metrics": [{"identity": {"key": "accuracy"}, "score": 0.61, "num": 50}],
    }
    a = es.parse_report(legacy)
    b = es.parse_report(current)
    assert a["dataset"] == "gsm8k" and a["pretty_name"] == "GSM8K"
    assert a["metrics"]["mean_acc"] == {"value": 0.42, "num": 100, "macro": None}
    assert a["n_samples"] == 100 and a["latency_ms_mean"] == 12.5
    assert b["metrics"]["accuracy"]["value"] == 0.61 and b["metrics"]["accuracy"]["num"] == 50


def test_find_reports_and_parse_results(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "m" / "gsm8k.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"dataset_name": "gsm8k", "metrics": [{"name": "acc", "score": 0.5, "num": 10}]}))
    (report.parent / "collection_report.json").write_text("{}")
    paths = es.find_reports(tmp_path)
    assert paths == [report]
    out = es.parse_results(paths)
    assert out["framework"] == "evalscope"
    assert out["datasets"]["gsm8k"]["metrics"]["acc"]["value"] == 0.5


def test_no_metrics_is_an_error_not_an_empty_success(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "m" / "x.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"dataset_name": "x"}))
    with pytest.raises(ValueError):
        es.parse_results(es.find_reports(tmp_path))


def test_run_evalscope_invokes_the_cli_and_normalizes_its_reports(tmp_path: Path) -> None:
    """A stand-in executable writes the report layout evalscope uses; the adapter must run it and parse it."""
    fake = tmp_path / "evalscope"
    fake.write_text(
        f"""#!{sys.executable}
import json, pathlib, sys
work = pathlib.Path(sys.argv[sys.argv.index('--work-dir') + 1])
assert '--api-url' in sys.argv and '--eval-type' in sys.argv
r = work / 'reports' / 'served' / 'gsm8k.json'
r.parent.mkdir(parents=True)
r.write_text(json.dumps({{'dataset_name': 'gsm8k', 'metrics': [{{'name': 'mean_acc', 'score': 0.75, 'num': 4}}]}}))
""",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    out = es.run_evalscope(18000, "served", ["gsm8k"], tmp_path / "out", limit=4, binary=fake)
    assert out["framework"] == "evalscope"
    assert out["datasets"]["gsm8k"]["metrics"]["mean_acc"]["value"] == 0.75
    assert out["limit"] == 4
    assert out["command"][0] == str(fake)
