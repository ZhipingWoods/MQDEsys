"""T-0808: the M4 reference table (recipes x backends: accuracy, performance, estimate error) built from finished job reports.
Thresholds are ARCHITECTURE section 11.4: FP8 <= 1 %, W4A16 <= 5 %, W4A8 <= 8 % PPL degradation; W2* recorded only."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import bench_reference as br

FIX = Path(__file__).parent / "fixtures" / "reports"


def rep(**recipes: dict) -> dict:
    return {
        "job_id": "j1",
        "kind": "llm",
        "spec": {"model_path": "/m/Qwen2.5-0.5B-Instruct", "name": "m4"},
        "env": {"eval_device_name": "NVIDIA GeForce RTX 4090"},
        "baseline": {"ppl": 10.0, "windows": 40, "seq_len": 512},
        "recipes": recipes,
    }


def recipe(name: str, delta: float, algo: str = "gptq", export: dict | None = None) -> dict:
    return {
        "recipe": name,
        "algorithm": algo,
        "summary": {"ppl": 10.0 * (1 + delta / 100), "ppl_delta_pct": delta},
        "eval": {"fake_quant_wo_kv": {"ppl": 10.0 * (1 + delta / 100)}},
        **({"export": export} if export is not None else {}),
    }


@pytest.mark.parametrize(
    ("name", "delta", "want"),
    [
        ("FP8", 0.99, "PASS"),
        ("FP8", 1.0, "PASS"),
        ("FP8", 1.01, "FAIL"),
        ("W4A16", 5.0, "PASS"),
        ("W4A16", 14.1, "FAIL"),
        ("W4A16", -0.3, "PASS"),
        ("W4A16", 5.01, "FAIL"),
        ("W4A8", 8.0, "PASS"),
        ("W4A8", 8.01, "FAIL"),
        ("W4A8", 21.2, "FAIL"),
        ("W2A16", 900.0, "INFO"),
        ("W2A8", 5.0, "INFO"),
        ("W8A16", 99.0, "INFO"),
        ("BF16", 0.0, "INFO"),
    ],
)
def test_the_ppl_threshold_verdicts(name: str, delta: float, want: str) -> None:
    assert br.verdict(name, delta) == want


def test_a_missing_measurement_has_no_verdict() -> None:
    assert br.verdict("FP8", None) == "n/a"


def test_accuracy_table_lists_every_recipe_with_its_delta_threshold_and_verdict() -> None:
    md = br.render([rep(FP8=recipe("FP8", 0.5, "rtn"), W4A16=recipe("W4A16", 14.1), W2A16=recipe("W2A16", 900.0, "rtn"))], None)
    acc = md.split("## Accuracy")[1].split("## Deployment")[0]
    rows = [r for r in acc.splitlines() if r.startswith("| FP8 ") or r.startswith("| W4A16 ") or r.startswith("| W2A16 ")]
    assert len(rows) == 3
    assert "+0.50%" in rows[0] and "≤ 1%" in rows[0] and "PASS" in rows[0]
    assert "+14.10%" in rows[1] and "≤ 5%" in rows[1] and "FAIL" in rows[1]
    assert "+900.00%" in rows[2] and "record only" in rows[2] and "INFO" in rows[2]
    assert "RTX 4090" in md and "Qwen2.5-0.5B-Instruct" in md


def test_the_summary_line_counts_passes_and_failures() -> None:
    md = br.render([rep(FP8=recipe("FP8", 0.5), W4A16=recipe("W4A16", 14.1), W4A8=recipe("W4A8", 3.0), W2A16=recipe("W2A16", 9.0))], None)
    assert "2 passed, 1 failed" in md and "1 recorded only" in md


def served(ppl: float, pct: float | None, tps: float = 500.0, gib: float | None = 0.43, kernel: str = "MarlinLinearKernel") -> dict:
    return {
        "status": "succeeded",
        "served_ppl": ppl,
        "served_vs_real_ppl_pct": pct,
        "load": {"ok": 16, "output_tokens_per_s": tps, "ttft_ms": {"p50": 50.0}},
        "engine": {"weights_gib": gib, "kv_cache_tokens": 560000, "quant_kernel": kernel},
    }


def test_the_deployment_matrix_has_a_row_per_export_and_backend_with_its_numbers() -> None:
    exp = {
        "awq": {
            "status": "exported",
            "real_ppl": 11.4,
            "bytes": 470_000_000,
            "vllm": served(11.41, 0.1, 460.0),
            "sglang": served(11.40, 0.0, 900.0, None, ""),
        },
        "compressed-tensors": {"status": "exported", "real_ppl": 11.4, "bytes": 470_000_000, "vllm": served(11.41, 0.1, 470.0)},
    }
    md = br.render([rep(W4A16=recipe("W4A16", 14.1, export=exp))], None)
    sec = md.split("## Deployment")[1].split("##")[0]
    rows = [r for r in sec.splitlines() if r.startswith("| W4A16")]
    assert len(rows) == 3
    assert any("awq" in r and "vLLM" in r and "460" in r and "50 ms" in r and "Marlin" in r for r in rows)
    assert any("awq" in r and "SGLang" in r and "900" in r for r in rows)
    assert any("compressed-tensors" in r and "vLLM" in r for r in rows)


def test_a_recipe_that_cannot_be_exported_is_listed_with_why_not_dropped_silently() -> None:
    exp = {
        "awq": {"status": "unsupported", "problems": [{"layer": "x", "reason": "2-bit weights are not supported by awq"}], "n_problems": 3}
    }
    md = br.render([rep(W2A16=recipe("W2A16", 900.0, export=exp), W4A8=recipe("W4A8", 3.0))], None)
    gaps = md.split("## Not deployable")[1]
    assert "W2A16" in gaps and "2-bit weights are not supported by awq" in gaps
    assert "W4A8" in gaps and "not exported yet" in gaps


def test_a_failed_or_skipped_serving_run_shows_its_reason_in_the_matrix() -> None:
    exp = {
        "awq": {
            "status": "exported",
            "real_ppl": 11.4,
            "vllm": {"status": "skipped", "reason": "vllm not found"},
            "sglang": {"status": "failed", "error": "boom\nmore", "diagnosis": "CUDA out of memory"},
        }
    }
    md = br.render([rep(W4A16=recipe("W4A16", 14.1, export=exp))], None)
    assert "skipped" in md and "vllm not found" in md and "CUDA out of memory" in md


def test_a_served_ppl_that_is_not_comparable_shows_a_dash_not_a_made_up_percentage() -> None:
    exp = {"awq": {"status": "exported", "real_ppl": 11.4, "vllm": served(11.41, None)}}
    row = next(
        r for r in br.render([rep(W4A16=recipe("W4A16", 14.1, export=exp))], None).splitlines() if r.startswith("| W4A16") and "vLLM" in r
    )
    assert " – " in row


def gg(ty: str, ppl: float, b: int, pp: float, tg: float) -> dict:
    return {"type": ty, "bytes": b, "ppl": ppl, "pp512_tokens_per_s": pp, "tg64_tokens_per_s": tg}


def test_llama_cpp_is_one_table_and_compared_with_its_own_f16() -> None:
    r = rep(FP8=recipe("FP8", 0.5))
    r["llamacpp"] = {
        "protocol": "llama-perplexity ctx 512, 40 chunks",
        "backend": "CUDA0",
        "results": {
            "f16": gg("f16", 13.89, 948_000_000, 800.0, 26.0),
            "q4_k": gg("q4_k", 14.75, 511_000_000, 826.0, 48.0),
            "q2_k": gg("q2_k", 20.0, 330_000_000, 900.0, 60.0),
        },
    }
    sec = br.render([r], None).split("## llama.cpp")[1].split("##")[0]
    rows = [x for x in sec.splitlines() if x.startswith("| ")]
    q4 = next(x for x in rows if "q4_k" in x)
    assert "14.750" in q4 and "+6.19%" in q4 and "826" in q4 and "48.0" in q4
    assert "independent of the recipes" in sec.lower(), "GGUF re-quantizes the original weights, so it is not a column of the recipe table"
    assert "llama-perplexity ctx 512" in sec


def est(recipe_id: str, weights: int, qmeta: int = 0) -> dict:
    return {"recipe_id": recipe_id, "recipe": recipe_id.upper(), "breakdown": {"weights": weights, "qmeta": qmeta}}


def test_weight_memory_estimate_error_against_the_engines_own_report() -> None:
    exp = {"awq": {"status": "exported", "real_ppl": 11.4, "vllm": served(11.4, 0.0, gib=0.43), "sglang": served(11.4, 0.0, gib=None)}}
    md = br.render([rep(W4A16=recipe("W4A16", 3.0, export=exp))], [est("w4a16", int(0.42 * 2**30))])
    sec = md.split("## Memory estimate")[1].split("##")[0]
    row = next(x for x in sec.splitlines() if x.startswith("| W4A16") and "vLLM" in x)
    assert "| awq |" in row and "0.42" in row and "0.43" in row and "-2.3%" in row and "PASS" in row
    assert not any("SGLang" in x for x in sec.splitlines() if x.startswith("| W4A16")), (
        "an engine that reports no weight size has nothing to compare"
    )
    assert "weights only" in sec.lower()


def test_estimate_error_over_ten_percent_fails() -> None:
    exp = {"awq": {"status": "exported", "real_ppl": 11.4, "vllm": served(11.4, 0.0, gib=0.50)}}
    md = br.render([rep(W4A16=recipe("W4A16", 3.0, export=exp))], [est("w4a16", int(0.42 * 2**30))])
    row = next(x for x in md.split("## Memory estimate")[1].splitlines() if x.startswith("| W4A16"))
    assert "-16.0%" in row and "FAIL" in row
    assert "1 failed" in md.split("## Summary")[1].split("##")[0] or "estimate" in md.lower()


def test_several_reports_are_merged_and_the_real_fixtures_render() -> None:
    reps = [json.loads((FIX / "llm_vllm.json").read_text()), json.loads((FIX / "llm_gguf.json").read_text())]
    md = br.render(reps, None)
    assert "W4A16" in md and "awq" in md and "q4_k" in md
    assert br.render([], None).startswith("# ")


def test_main_reads_report_files_and_writes_markdown(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    a.write_text(json.dumps(rep(FP8=recipe("FP8", 0.5))))
    e = tmp_path / "e.json"
    e.write_text(json.dumps({"estimates": [est("fp8", 100)]}))
    out = tmp_path / "out.md"
    assert br.main(["--report", str(a), "--estimates", str(e), "--out", str(out)]) == 0
    assert out.read_text().startswith("# ") and "FP8" in out.read_text()
    assert br.main(["--report", str(tmp_path / "missing.json"), "--out", str(out)]) == 2


def test_rendering_does_not_modify_its_input() -> None:
    r = rep(W4A16=recipe("W4A16", 3.0))
    before = copy.deepcopy(r)
    br.render([r], None)
    assert r == before


def test_the_estimated_weight_size_includes_the_quantization_metadata() -> None:
    """Scales and zeros are part of what the engine loads: the estimator reports them as a separate component."""
    exp = {"awq": {"status": "exported", "real_ppl": 11.4, "vllm": served(11.4, 0.0, gib=0.43)}}
    md = br.render([rep(W4A16=recipe("W4A16", 3.0, export=exp))], [est("w4a16", int(0.35 * 2**30), int(0.07 * 2**30))])
    row = next(x for x in md.split("## Memory estimate")[1].splitlines() if x.startswith("| W4A16"))
    assert "0.42" in row and "PASS" in row, row  # 0.35 + 0.07; the weights alone (0.35) would be -18.6%


def test_the_f16_row_of_the_llama_cpp_table_is_the_reference_not_a_comparison_with_itself() -> None:
    r = rep(FP8=recipe("FP8", 0.5))
    r["llamacpp"] = {"results": {"f16": gg("f16", 13.89, 948_000_000, 800.0, 26.0), "q8_0": gg("q8_0", 13.93, 628_000_000, 880.0, 40.0)}}
    sec = br.render([r], None).split("## llama.cpp")[1]
    f16 = next(x for x in sec.splitlines() if x.startswith("| f16"))
    assert f16.split("|")[4].strip() == "–", f16
    assert "+0.29%" in next(x for x in sec.splitlines() if x.startswith("| q8_0"))


# ---- --suite m4: submit the reference jobs to a running server, wait, render ----
import http.server
import threading


class FakeServer:
    """Just enough of mqde-server: health, jobs (with ?dry_run), job detail with state progressing, estimate."""

    def __init__(self, *, invalid_name: str | None = None, fail_name: str | None = None, never_finish: bool = False) -> None:
        self.real: list[dict] = []
        self.dry: list[dict] = []
        self.polls = 0
        self.estimate_bodies: list[dict] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                pass

            def _send(self, code: int, obj: object) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _body(self) -> dict:
                return json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")

            def do_GET(self) -> None:
                if self.path == "/api/v1/health":
                    return self._send(200, {"status": "ok"})
                if self.path == "/api/v1/jobs":
                    outer.polls += 1
                    done = outer.polls >= 2 and not never_finish
                    return self._send(
                        200,
                        [
                            {
                                "job_id": j["id"],
                                "state": ("FAILED" if fail_name and j["metadata"]["name"] == fail_name else "SUCCEEDED")
                                if done
                                else "RUNNING",
                            }
                            for j in outer.real
                        ],
                    )
                if self.path.startswith("/api/v1/jobs/"):
                    jid = self.path.rsplit("/", 1)[1]
                    j = next(x for x in outer.real if x["id"] == jid)
                    failed = fail_name and j["metadata"]["name"] == fail_name
                    return self._send(
                        200,
                        {
                            "state": "FAILED" if failed else "SUCCEEDED",
                            "report": {
                                "job_id": jid,
                                "kind": "llm",
                                "spec": {"model_path": "/m/q", "name": j["metadata"]["name"]},
                                "baseline": {"ppl": 10.0},
                                "recipes": {r.upper(): recipe(r.upper(), 1.0) for r in j["quant"]["recipe"]},
                                **({"error": "RuntimeError: boom"} if failed else {}),
                            },
                        },
                    )
                self._send(404, {"code": "not_found"})

            def do_POST(self) -> None:
                body = self._body()
                if self.path.startswith("/api/v1/jobs"):
                    if invalid_name and body["metadata"]["name"] == invalid_name:
                        return self._send(
                            400,
                            {
                                "code": "invalid_argument",
                                "message": "/export/formats: recipe 'FP8' cannot be exported as awq",
                                "problems": [{"field": "/export/formats"}],
                            },
                        )
                    if "dry_run" in self.path:
                        outer.dry.append(body)
                        return self._send(200, {"valid": True})
                    body["id"] = f"job-{len(outer.real)}"
                    outer.real.append(body)
                    return self._send(201, {"job_id": body["id"]})
                if self.path == "/api/v1/estimate":
                    outer.estimate_bodies.append(body)
                    return self._send(200, {"estimates": [est(r, 1000) for r in body["recipes"]]})
                self._send(404, {"code": "not_found"})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    made: list[FakeServer] = []

    def make(**kw: object) -> FakeServer:
        s = FakeServer(**kw)  # type: ignore[arg-type]
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def test_the_suite_splits_recipes_so_that_every_export_format_suits_every_recipe_in_its_job() -> None:
    specs = br.suite_specs("/models/Qwen2.5-0.5B-Instruct")
    by = {s["metadata"]["name"]: s for s in specs}
    assert sorted(by) == ["m4-fp8", "m4-w4a16", "m4-w4a8-w2"]
    assert by["m4-fp8"]["quant"]["recipe"] == ["fp8"] and by["m4-fp8"]["export"]["formats"] == ["compressed-tensors"], (
        "fp8 is not an awq checkpoint"
    )
    assert by["m4-w4a16"]["export"]["formats"] == ["awq", "compressed-tensors"]
    assert by["m4-w4a8-w2"]["quant"]["recipe"] == ["w4a8", "w2a16", "w2a8"] and "export" not in by["m4-w4a8-w2"], (
        "no container holds these yet"
    )
    assert "llamacpp" in by["m4-fp8"]["deploy"] and "llamacpp" not in by["m4-w4a16"]["deploy"], (
        "GGUF does not depend on the recipe: measured once"
    )
    assert all(s["model"]["path"] == "/models/Qwen2.5-0.5B-Instruct" and s["evaluation"]["max_windows"] == 40 for s in specs)
    covered = [r for s in specs for r in s["quant"]["recipe"]]
    assert sorted(covered) == ["fp8", "w2a16", "w2a8", "w4a16", "w4a8"], "all five recipes of the suite, none twice"


def test_a_dry_run_validates_every_job_and_queues_none(server) -> None:  # type: ignore[no-untyped-def]
    s = server()
    out = br.run_suite(s.url, "/m/q", dry_run=True, poll_s=0.01)
    assert out is None and len(s.dry) == 3 and s.real == []


def test_every_job_is_validated_before_any_of_them_takes_the_gpu(server) -> None:  # type: ignore[no-untyped-def]
    s = server(invalid_name="m4-w4a16")
    with pytest.raises(br.SuiteError, match=r"m4-w4a16.*cannot be exported as awq"):
        br.run_suite(s.url, "/m/q", poll_s=0.01)
    assert s.real == [], "the fp8 job was valid but must not have been submitted: the suite is all-or-nothing"


def test_the_suite_submits_waits_collects_reports_and_asks_for_estimates(server) -> None:  # type: ignore[no-untyped-def]
    s = server()
    reports, estimates = br.run_suite(s.url, "/m/q", poll_s=0.01)
    assert [r["job_id"] for r in reports] == ["job-0", "job-1", "job-2"] and len(s.real) == 3
    assert s.polls >= 2, "it waited for the jobs to finish rather than reading half-written reports"
    assert s.estimate_bodies[0]["backend"] == "vllm" and sorted(s.estimate_bodies[0]["recipes"]) == [
        "fp8",
        "w2a16",
        "w2a8",
        "w4a16",
        "w4a8",
    ]
    assert {e["recipe_id"] for e in estimates} == {"fp8", "w4a16", "w4a8", "w2a16", "w2a8"}


def test_a_failed_job_stops_the_suite_with_its_error(server) -> None:  # type: ignore[no-untyped-def]
    s = server(fail_name="m4-w4a16")
    with pytest.raises(br.SuiteError, match=r"m4-w4a16.*FAILED.*boom"):
        br.run_suite(s.url, "/m/q", poll_s=0.01)


def test_a_job_that_never_finishes_times_out(server) -> None:  # type: ignore[no-untyped-def]
    s = server(never_finish=True)
    with pytest.raises(br.SuiteError, match="did not finish within"):
        br.run_suite(s.url, "/m/q", poll_s=0.01, timeout_s=0.2)


def test_an_unreachable_server_is_a_clear_error() -> None:
    with pytest.raises(br.SuiteError, match="cannot reach"):
        br.run_suite("http://127.0.0.1:1", "/m/q", poll_s=0.01)


def test_main_runs_the_suite_and_writes_the_page(server, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    s = server()
    out = tmp_path / "m4.md"
    assert br.main(["--suite", "m4", "--server", s.url, "--model", "/m/q", "--poll", "0.01", "--out", str(out)]) == 0
    md = out.read_text()
    assert md.startswith("# M4 reference table") and "FP8" in md and "## Memory estimate" in md
    assert br.main(["--suite", "m5", "--server", s.url, "--model", "/m/q", "--out", str(out)]) == 2, "only the m4 suite exists"
    assert br.main(["--suite", "m4", "--server", "http://127.0.0.1:1", "--model", "/m/q", "--out", str(out)]) == 2
    assert br.main(["--suite", "m4", "--server", s.url, "--model", "/m/q", "--dry-run"]) == 0


def test_the_throughput_columns_carry_the_warning_that_engines_are_not_configured_alike() -> None:
    """Real run: SGLang 1889 tok/s vs vLLM 200 on the same checkpoint. vLLM stages run enforce_eager (no CUDA graphs), SGLang with
    CUDA graphs: the gap is mostly that, so the page must say it next to the numbers."""
    exp = {"awq": {"status": "exported", "real_ppl": 11.4, "vllm": served(11.4, 0.0, 200.0), "sglang": served(11.4, 0.0, 1889.0)}}
    sec = (
        br.render([rep(W4A16=recipe("W4A16", 3.0, export=exp))], None).split("## Deployment")[1].split("## llama")[0].split("## Memory")[0]
    )
    assert (
        "enforce_eager=True` (no CUDA graphs)" in sec
        and "SGLang stage with CUDA graphs on" in sec
        and "not comparable across engines" in sec
    )


def test_recipes_that_have_no_container_say_why_in_the_plannersterms() -> None:
    md = br.render([rep(W4A8=recipe("W4A8", 3.0), W2A16=recipe("W2A16", 9.0), W2A8=recipe("W2A8", 9.0))], None)
    gaps = md.split("## Not deployable")[1]
    w4a8 = next(x for x in gaps.splitlines() if x.startswith("| W4A8"))
    assert "not exported yet" in w4a8 and "quantized activations" in w4a8
    for name in ("W2A16", "W2A8"):
        row = next(x for x in gaps.splitlines() if x.startswith(f"| {name}"))
        assert "2-bit" in row and "awq" in row and "compressed-tensors" in row
    assert "no export requested" not in gaps, "the known reasons replace the generic one"


def test_a_recipe_without_a_known_reason_keeps_the_generic_one() -> None:
    md = br.render([rep(W8A16=recipe("W8A16", 1.0))], None)
    assert "no export requested in this job" in md.split("## Not deployable")[1]


def test_the_page_states_its_scope() -> None:
    md = br.render([rep(FP8=recipe("FP8", 0.5))], None)
    assert (
        "Scope: Qwen2.5-0.5B-Instruct only (one model of ARCHITECTURE §11.3's reference set)" in md and "does not claim M4 acceptance" in md
    )
