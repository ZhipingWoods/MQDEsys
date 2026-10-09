"""lm-eval adapter (T-0801 / ADR-0002 A): command assembly, result parsing and failure handling, with a stand-in CLI.

The stand-in writes what the official CLI writes (<out>/<model>/results_<ts>.json) so the adapter's path handling and
parsing run for real. The real-output parsing test lives in test_lm_eval_real_output.py once a real file is recorded.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from mqde.eval.lm_eval_adapter import build_command, find_results, lm_eval_bin, parse_results, run_lm_eval

DOC: dict[str, Any] = {
    "results": {"gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.5, "exact_match_stderr,strict-match": 0.1,
                          "exact_match,flexible-extract": 0.6, "exact_match_stderr,flexible-extract": 0.11},
                "arc_easy": {"alias": "arc_easy", "acc,none": 0.75, "acc_stderr,none": "N/A", "acc_norm,none": 0.7}},
    "n-samples": {"gsm8k": {"original": 1319, "effective": 10}},
    "versions": {"gsm8k": 3.0}, "n-shot": {"gsm8k": 5}, "lm_eval_version": "0.4.14", "git_hash": "abc1234",
}


def fake_cli(tmp: Path, doc: dict[str, Any] | None, code: int = 0, sleep: float = 0) -> Path:
    """An executable that records its argv and writes `doc` the way lm-eval does."""
    p = tmp / "lm-eval"
    p.write_text(f"""#!{sys.executable}
import json, sys, time, pathlib
a = sys.argv[1:]
out = pathlib.Path(a[a.index("--output_path") + 1])
pathlib.Path(r"{tmp}/argv.json").write_text(json.dumps(a))
time.sleep({sleep})
print("log line from the fake cli")
doc = json.loads({json.dumps(json.dumps(doc))})
if doc is not None:
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "model" / "results_2026-10-09T00-00-00.json").write_text(json.dumps(doc))
sys.exit({code})
""")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_command_is_a_list_with_the_documented_flags() -> None:
    c = build_command("/x/lm-eval", 8123, "m", "/ckpt", ["mmlu", "gsm8k"], "/out", limit=20, num_fewshot=5, concurrency=4, seed=7,
                      max_gen_toks=128)
    assert c[0] == "/x/lm-eval" and c[1:3] == ["--model", "local-completions"]
    args = dict(zip(c[3::2], c[4::2], strict=False))
    assert args["--tasks"] == "mmlu,gsm8k" and args["--limit"] == "20" and args["--num_fewshot"] == "5" and args["--seed"] == "7"
    assert args["--gen_kwargs"] == "max_gen_toks=128" and args["--output_path"] == "/out"
    ma = dict(kv.split("=", 1) for kv in args["--model_args"].split(","))
    assert ma["base_url"] == "http://127.0.0.1:8123/v1/completions" and ma["model"] == "m" and ma["tokenizer"] == "/ckpt"
    assert ma["num_concurrent"] == "4" and ma["tokenizer_backend"] == "huggingface"
    bare = build_command("b", 1, "m", "t", ["a"], "o")
    assert "--limit" not in bare and "--num_fewshot" not in bare and "--gen_kwargs" not in bare
    assert build_command("b", 1, "m", "t", ["a"], "o", extra=["--apply_chat_template"])[-1] == "--apply_chat_template"
    with pytest.raises(ValueError, match="tasks is empty"):
        build_command("b", 1, "m", "t", [], "o")


def test_listen_address_is_loopback_only() -> None:
    assert "http://127.0.0.1:" in build_command("b", 9, "m", "t", ["a"], "o")[4]


def test_parse_results_metrics_stderr_and_samples() -> None:
    r = parse_results(DOC)
    g = r["tasks"]["gsm8k"]
    assert g["metrics"]["exact_match,strict-match"] == {"metric": "exact_match", "filter": "strict-match", "value": 0.5, "stderr": 0.1}
    assert g["metrics"]["exact_match,flexible-extract"]["value"] == 0.6
    assert g["samples"] == {"original": 1319, "effective": 10}
    a = r["tasks"]["arc_easy"]["metrics"]
    assert a["acc,none"]["stderr"] is None, "'N/A' stderr (lm-eval writes it for tiny samples) is None, not a crash"
    assert a["acc_norm,none"]["stderr"] is None and "alias" not in a
    assert r["version"] == "0.4.14" and r["git_hash"] == "abc1234" and r["n_shot"] == {"gsm8k": 5}
    with pytest.raises(ValueError, match="no 'results'"):
        parse_results({"nope": 1})


def test_run_records_command_and_results(tmp_path: Path) -> None:
    out = run_lm_eval(8123, "m", "/ckpt", ["gsm8k"], tmp_path / "o", limit=10, binary=fake_cli(tmp_path, DOC))
    assert out["tasks"]["gsm8k"]["metrics"]["exact_match,strict-match"]["value"] == 0.5
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv == out["command"][1:] and out["limit"] == 10 and out["seconds"] >= 0
    assert (tmp_path / "o" / "lm_eval.log").read_text().startswith("log line from the fake cli")


def test_failures_are_errors_with_the_log_tail(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"exited with 3.*log line from the fake cli"):
        run_lm_eval(1, "m", "t", ["a"], tmp_path / "o1", binary=fake_cli(tmp_path, None, code=3))
    with pytest.raises(FileNotFoundError, match="no results_"):
        run_lm_eval(1, "m", "t", ["a"], tmp_path / "o2", binary=fake_cli(tmp_path, None, code=0))
    with pytest.raises(RuntimeError, match="timed out"):
        run_lm_eval(1, "m", "t", ["a"], tmp_path / "o3", timeout=0.3, binary=fake_cli(tmp_path, DOC, sleep=5))


def test_the_first_real_problem_is_reported_not_the_traceback_tail(tmp_path: Path) -> None:
    from mqde.eval.lm_eval_adapter import first_problem

    log = ("INFO fine\nWARNING  [models.api_models:521] API request failed! Status code: 400, Response text: "
           "{\"error\":{\"message\":\"This model's maximum context length is 1024 tokens. However, you requested 256 output tokens\"}}\n"
           "Traceback ...\nRuntimeError: Session is closed\n")
    assert "maximum context length is 1024" in (first_problem(log) or "")
    assert first_problem("INFO all good\n") is None
    assert "No module named" in (first_problem("ModuleNotFoundError: No module named 'transformers'") or "")


def test_newest_results_file_wins(tmp_path: Path) -> None:
    import os
    d = tmp_path / "m"
    d.mkdir()
    old, new = d / "results_a.json", d / "results_b.json"
    old.write_text("{}")
    new.write_text("{}")
    os.utime(old, (1, 1))
    assert find_results(tmp_path) == new


def test_binary_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MQDE_LM_EVAL_BIN", str(fake_cli(tmp_path, DOC)))
    assert lm_eval_bin() == tmp_path / "lm-eval"
    monkeypatch.setenv("MQDE_LM_EVAL_BIN", str(tmp_path / "missing"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="MQDE_LM_EVAL_BIN"):
        lm_eval_bin()


REAL = Path(__file__).parent.parent / "fixtures" / "lm_eval" / "results_gsm8k_qwen05b_limit20.json"


def test_real_lm_eval_output_is_parsed() -> None:
    """A results file written by lm-eval 0.4.13 itself (Qwen2.5-0.5B-Instruct behind vLLM, gsm8k, 5-shot, limit 20)."""
    r = parse_results(json.loads(REAL.read_text()))
    g = r["tasks"]["gsm8k"]
    assert set(g["metrics"]) == {"exact_match,strict-match", "exact_match,flexible-extract"}, "sample_len is not a metric"
    assert g["metrics"]["exact_match,strict-match"]["value"] == 0.05 and g["metrics"]["exact_match,flexible-extract"]["value"] == 0.1
    assert g["metrics"]["exact_match,flexible-extract"]["stderr"] == pytest.approx(0.06882472016116853)
    assert g["samples"] == {"original": 1319, "effective": 20}
    assert r["version"] == "0.4.13" and r["git_hash"] is None and r["n_shot"] == {"gsm8k": 5}
    # 20 samples: the standard error is as large as the score itself, which is why the report says so
    assert g["metrics"]["exact_match,strict-match"]["stderr"] >= g["metrics"]["exact_match,strict-match"]["value"] * 0.99


def test_real_output_through_the_adapter_path(tmp_path: Path) -> None:
    out = run_lm_eval(1, "m", "t", ["gsm8k"], tmp_path / "o", binary=fake_cli(tmp_path, json.loads(REAL.read_text())))
    assert out["tasks"]["gsm8k"]["metrics"]["exact_match,flexible-extract"]["value"] == 0.1
