"""Report JSON schema (T-0806): real fixtures are valid; each kind of damage is reported with its path."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from mqde.report.__main__ import main
from mqde.report.schema import validate_report

FIX = Path(__file__).parent / "fixtures" / "reports"
NAMES = ["llm_full", "llm_vllm", "llm_gguf"]


def load(n: str) -> dict[str, Any]:
    return json.loads((FIX / f"{n}.json").read_text())


@pytest.mark.parametrize("n", NAMES)
def test_fixtures_are_valid(n: str) -> None:
    assert validate_report(load(n)) == []


def mutate(path: list[Any], value: Any) -> list[str]:
    r = load("llm_full")
    o = r
    for p in path[:-1]:
        o = o[p]
    if value is KeyError:
        del o[path[-1]]
    else:
        o[path[-1]] = value
    return validate_report(r)


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (["job_id"], KeyError, "missing required key 'job_id'"),
        (["baseline", "ppl"], "18.1", "$.baseline.ppl: expected number"),
        (["baseline", "ppl"], 0.5, "below the minimum"),
        (["kind"], "vla", "not one of"),
        (["stages", 1, "status"], "weird", "$.stages[1].status"),
        (["stages", 0, "seconds"], -1, "$.stages[0].seconds"),
        (["recipes", "W4A16", "layers", 0, "w_sqnr_db"], "x", "$.recipes.W4A16.layers[0].w_sqnr_db"),
        (["recipes", "W4A16", "layers", 0, "shape"], [896], "$.recipes.W4A16.layers[0].shape"),
        (["recipes", "W4A16", "algorithm"], KeyError, "$.recipes.W4A16: missing required key 'algorithm'"),
        (["spec", "recipes"], [], "$.spec.recipes"),
        (["model", "blocks"], True, "expected integer"),
        (["recipes", "W4A16", "layers", 0, "out_sqnr_db"], "high", "expected ['number', 'null']"),
    ],
)
def test_damage_is_reported_with_its_path(path: list[Any], value: Any, needle: str) -> None:
    errs = mutate(path, value)
    assert any(needle in e for e in errs), errs


def test_unmeasured_output_sqnr_is_null_not_an_error() -> None:
    assert mutate(["recipes", "W4A16", "layers", 0, "out_sqnr_db"], None) == []


def test_a_failed_job_needs_no_results() -> None:
    failed = {"job_id": "j", "kind": "llm", "spec": {"name": "n", "recipes": ["w4a16"]}, "stages": [], "error": "RuntimeError: boom"}
    assert validate_report(failed) == []
    del failed["error"]  # without an error the results are required
    errs = validate_report(failed)
    assert {"baseline", "recipes", "model", "env"} <= {e.split("'")[1] for e in errs if "missing required key" in e}


def test_every_problem_is_listed_not_just_the_first() -> None:
    r = copy.deepcopy(load("llm_full"))
    del r["job_id"]
    r["baseline"]["ppl"] = "x"
    r["stages"][0]["seconds"] = -1
    assert len(validate_report(r)) == 3


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "report.json"
    good.write_text((FIX / "llm_full.json").read_text())
    assert main(["validate", str(good)]) == 0 and "valid" in capsys.readouterr().out
    assert main(["html", str(good)]) == 0
    assert "<html lang=\"zh\"" in (tmp_path / "report.zh.html").read_text() and (tmp_path / "report.en.html").exists()
    (tmp_path / "b").mkdir()
    bad = tmp_path / "b" / "report.json"
    r = load("llm_full")
    del r["job_id"]
    bad.write_text(json.dumps(r))
    assert main(["validate", str(bad)]) == 1 and "INVALID" in capsys.readouterr().out
    assert main(["html", str(bad)]) == 1 and not list((tmp_path / "b").glob("*.html"))  # nothing written for an invalid report
    bad.write_text("not json")
    assert main(["validate", str(bad)]) == 2
