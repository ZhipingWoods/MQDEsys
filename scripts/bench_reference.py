#!/usr/bin/env python3
"""M4 reference table (T-0808): one markdown page from finished LLM job reports -- accuracy against the ARCHITECTURE 11.4
thresholds, the recipe x export x backend deployment matrix, llama.cpp, and the estimator's weight-memory error against what
vLLM / SGLang report loading. Reads reports, runs nothing:

    python scripts/bench_reference.py --report runs/<job>/report.json [--report ...] [--estimates est.json] --out docs/reports/m4-reference.md

    python scripts/bench_reference.sh --suite m4 --server http://127.0.0.1:18080 --model /path/to/Qwen2.5-0.5B-Instruct --out docs/reports/m4-reference.md

`--suite m4` submits the reference jobs to a running server (all validated first, so nothing takes the GPU unless every job is
valid), waits for them, fetches their reports and the estimator's numbers, and renders the same page.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ORDER = ["BF16", "FP8", "W8A16", "W4A16", "W4A8", "W2A16", "W2A8"]
# ARCHITECTURE 11.4: relative wikitext-2 PPL degradation limits (percent). Anything else is recorded, not judged.
LIMITS = {"FP8": 1.0, "W4A16": 5.0, "W4A8": 8.0}
EST_LIMIT_PCT = 10.0  # T-0808: weight-memory estimate error
# Why a recipe has no vLLM / SGLang checkpoint: the planner's own refusals (POST /api/v1/jobs with export.formats), as seen on the
# dev machine; recipes not listed here get the generic note.
NO_CONTAINER = {
    "W4A8": "integer weights with quantized activations (W4A8 / W8A8-int) are not exported yet (compressed-tensors has no int-act writer)",
    "W2A16": "2-bit weights are not supported by awq (4-bit only) or compressed-tensors (4 / 8-bit)",
    "W2A8": "2-bit weights are not supported by awq (4-bit only) or compressed-tensors (4 / 8-bit)",
}


def verdict(recipe: str, delta_pct: float | None) -> str:
    if delta_pct is None:
        return "n/a"
    lim = LIMITS.get(recipe.upper())
    if lim is None:
        return "INFO"
    return "PASS" if delta_pct <= lim else "FAIL"


def _f(v: Any, nd: int = 2) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def _pct(v: float | None, nd: int = 2) -> str:
    return "–" if v is None else f"{v:+.{nd}f}%"


def _mib(b: float | None) -> str:
    return "–" if b is None else f"{b / 2**20:.0f} MiB"


def _limit_text(recipe: str) -> str:
    lim = LIMITS.get(recipe.upper())
    return "record only" if lim is None else f"≤ {lim:g}%"


def _recipes(reports: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    out = [(r, name, rc) for r in reports for name, rc in (r.get("recipes") or {}).items()]
    return sorted(out, key=lambda t: (ORDER.index(t[1].upper()) if t[1].upper() in ORDER else len(ORDER), t[1]))


def _reason(v: dict[str, Any]) -> str:
    why = v.get("diagnosis") or v.get("reason") or (v.get("error") or "").split("\n")[0]
    return f"{v.get('status')}: {why}".strip(": ")


def render(reports: list[dict[str, Any]], estimates: list[dict[str, Any]] | None) -> str:
    if not reports:
        return "# M4 reference table\n\nno reports given.\n"
    first = reports[0]
    model = Path((first.get("spec") or {}).get("model_path") or "?").name
    dev = next((e for r in reports for e in [(r.get("env") or {}).get("eval_device_name") or (r.get("env") or {}).get("device_name")] if e), "?")
    base = next((r["baseline"] for r in reports if r.get("baseline")), {})
    rows = _recipes(reports)
    verdicts = [verdict(n, (rc.get("summary") or {}).get("ppl_delta_pct")) for _, n, rc in rows]
    passed, failed, info = verdicts.count("PASS"), verdicts.count("FAIL"), verdicts.count("INFO")

    ests = {e["recipe_id"]: e for e in (estimates or [])}
    mem_rows: list[str] = []
    mem_ok = mem_bad = 0
    dep_rows: list[str] = []
    gaps: list[str] = []
    for r, name, rc in rows:
        exported = 0
        for fmt, ex in (rc.get("export") or {}).items():
            if ex.get("status") != "exported":
                prob = (ex.get("problems") or [{}])[0].get("reason", ex.get("status"))
                gaps.append(f"| {name} | {fmt} | {prob} |")
                continue
            exported += 1
            for key, label in (("vllm", "vLLM"), ("sglang", "SGLang")):
                v = ex.get(key)
                if not v:
                    continue
                if v.get("status") not in ("succeeded", "degraded"):
                    dep_rows.append(f"| {name} | {fmt} | {label} | {_reason(v)} | – | – | – | – | – | – |")
                    continue
                load, eng = v.get("load") or {}, v.get("engine") or {}
                ttft = (load.get("ttft_ms") or {}).get("p50")
                dep_rows.append(f"| {name} | {fmt} | {label} | {v['status']} | {_f(v.get('served_ppl'), 3)} | {_pct(v.get('served_vs_real_ppl_pct'))} | "
                                f"{_f(load.get('output_tokens_per_s'), 0)} | {'–' if ttft is None else f'{ttft:.0f} ms'} | {_f(eng.get('weights_gib'))} | {eng.get('quant_kernel') or '–'} |")
                e = ests.get(name.lower())
                if e and eng.get("weights_gib"):
                    bd = e.get("breakdown") or {}
                    est_gib = (bd.get("weights", 0) + bd.get("qmeta", 0)) / 2**30
                    err = 100.0 * (est_gib - eng["weights_gib"]) / eng["weights_gib"]
                    ok = abs(err) <= EST_LIMIT_PCT
                    mem_ok += ok
                    mem_bad += not ok
                    mem_rows.append(f"| {name} | {fmt} | {label} | {est_gib:.2f} | {eng['weights_gib']:.2f} | {err:+.1f}% | {'PASS' if ok else 'FAIL'} |")
        if not exported and not (rc.get("export") or {}):
            gaps.append(f"| {name} | – | {NO_CONTAINER.get(name.upper()) or 'no export requested in this job (the recipe may have none: see data/compat/matrix.yaml)'} |")

    lines = [f"# M4 reference table — {model} on {dev}", "",
             f"Jobs: {', '.join(r.get('job_id', '?') for r in reports)}. wikitext-2 test, {base.get('windows', '?')} windows × {base.get('seq_len', '?')} tokens; "
             f"unquantized PPL {_f(base.get('ppl'), 3)}.", "",
             f"Scope: {model} only (one model of ARCHITECTURE §11.3's reference set). This page does not claim M4 acceptance, which needs the whole set.", "",
             "## Summary", "",
             f"- Accuracy against the ARCHITECTURE §11.4 thresholds: **{passed} passed, {failed} failed**, {info} recorded only (no threshold).",
             f"- Weight-memory estimate within {EST_LIMIT_PCT:g}% of what the engine reports: **{mem_ok} within, {mem_bad} outside** (of {mem_ok + mem_bad} comparable).", "",
             "## Accuracy", "",
             "| Recipe | Algorithm | PPL (fake-quant W+A) | Δ vs unquantized | Threshold | Verdict | PPL of the exported weights |",
             "|---|---|---|---|---|---|---|"]
    for (r, name, rc), v in zip(rows, verdicts, strict=True):
        s = rc.get("summary") or {}
        real = [ex.get("real_ppl") for ex in (rc.get("export") or {}).values() if ex.get("real_ppl") is not None]
        lines.append(f"| {name} | {rc.get('algorithm', '–')} | {_f(s.get('ppl'), 3)} | {_pct(s.get('ppl_delta_pct'))} | {_limit_text(name)} | {v} | {_f(real[0], 3) if real else '–'} |")
    lines += ["", "## Deployment: vLLM and SGLang", ""]
    if dep_rows:
        lines += ["| Recipe | Export | Backend | Status | Served PPL | vs the exported weights | tok/s | TTFT p50 | Weights GiB | Kernel |", "|---|---|---|---|---|---|---|---|---|---|", *dep_rows]
    else:
        lines.append("no vLLM / SGLang runs in these reports.")
    lines += ["", "Served PPL comes from the engine's own prompt logprobs on the same windows as the exported weights; \"vs the exported weights\" is only given when the evaluation protocol matches.", "",
              "**tok/s and TTFT are not comparable across engines.** The vLLM stage runs `enforce_eager=True` (no CUDA graphs) and the SGLang stage with CUDA graphs on, "
              "so most of a gap between them is that configuration, not the engines (T-0704). Compare rows of the same engine.", ""]
    gg = next((r["llamacpp"] for r in reports if r.get("llamacpp")), None)
    if gg:
        res = gg.get("results") or {}
        f16 = (res.get("f16") or {}).get("ppl")
        backend_note = "; backend " + str(gg["backend"]) if gg.get("backend") else ""
        lines += ["## llama.cpp (GGUF; independent of the recipes)", "",
                  "GGUF re-quantizes the original FP weights with llama.cpp's own numerics, so it does not belong to any recipe row above.", "",
                  f"Protocol: {gg.get('protocol', '?')}{backend_note}.", "",
                  "| Type | Size | PPL | vs F16 | pp512 tok/s | tg64 tok/s |", "|---|---|---|---|---|---|"]
        for ty, x in res.items():
            d = None if (f16 is None or x.get("ppl") is None or ty == "f16") else 100.0 * (x["ppl"] / f16 - 1.0)
            lines.append(f"| {ty} | {_mib(x.get('bytes'))} | {_f(x.get('ppl'), 3)} | {_pct(d)} | {_f(x.get('pp512_tokens_per_s'), 0)} | {_f(x.get('tg64_tokens_per_s'), 1)} |")
        lines.append("")
    if estimates is not None:
        lines += ["## Memory estimate vs measured", "",
                  "Weights only: vLLM / SGLang pre-allocate the KV cache to their memory fraction, so total peak memory cannot be compared. "
                  "The engine reports weight size in GiB rounded to 0.01, so ±1% is rounding. The estimator has a vLLM profile only; SGLang is compared against it.", ""]
        if mem_rows:
            lines += ["| Recipe | Export | Backend | Estimated GiB | Measured GiB | Error | Verdict |", "|---|---|---|---|---|---|---|", *mem_rows]
        else:
            lines.append("nothing comparable (no engine reported a weight size).")
        lines.append("")
    lines += ["## Not deployable on vLLM / SGLang", ""]
    if gaps:
        lines += ["| Recipe | Format | Why |", "|---|---|---|", *gaps]
    else:
        lines.append("every recipe in these reports was exported.")
    return "\n".join(lines) + "\n"


class SuiteError(RuntimeError):
    pass


def suite_specs(model: str, hardware: str = "nvidia/rtx4090") -> list[dict[str, Any]]:
    """The m4 reference jobs for one model. Recipes are grouped so that every export format suits every recipe of its job (the
    server refuses a job otherwise): fp8 has no awq, W4A8 / W2* have no container yet. llama.cpp's GGUF re-quantizes the original
    weights, so it is measured once, in the first job."""
    common = {"domain": "llm", "model": {"source": "local", "path": model, "type": "llm"}, "quant_device": "gpu",
              "calibration": {"dataset": "wikitext-2-raw-v1:train", "n_samples": 32, "seq_len": 512},
              "evaluation": {"dataset": "wikitext-2-raw-v1:test", "seq_len": 512, "max_windows": 40, "device": "gpu"},
              "analysis": {"logits_probe": {}}}
    serve = {"requests": 16, "concurrency": 4}
    dep = {"hardware": hardware, "device": "gpu"}
    return [
        {**common, "metadata": {"name": "m4-fp8"}, "quant": {"recipe": ["fp8"]}, "export": {"formats": ["compressed-tensors"]},
         "deploy": {**dep, "vllm": {"gpu_memory_utilization": 0.3, **serve}, "sglang": {"mem_fraction_static": 0.3, **serve},
                    "llamacpp": {"types": ["q8_0", "q4_k", "q2_k"], "gpu_layers": 99}}},
        {**common, "metadata": {"name": "m4-w4a16"}, "quant": {"recipe": ["w4a16"]}, "export": {"formats": ["awq", "compressed-tensors"]},
         "deploy": {**dep, "vllm": {"gpu_memory_utilization": 0.3, **serve}, "sglang": {"mem_fraction_static": 0.3, **serve}}},
        {**common, "metadata": {"name": "m4-w4a8-w2"}, "quant": {"recipe": ["w4a8", "w2a16", "w2a8"]}, "deploy": dep},
    ]


def _http(server: str, method: str, path: str, body: Any = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(server.rstrip("/") + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except ValueError:
            return e.code, None
    except (urllib.error.URLError, OSError) as e:
        raise SuiteError(f"cannot reach {server}: {e}") from e


def run_suite(server: str, model: str, *, hardware: str = "nvidia/rtx4090", dry_run: bool = False, poll_s: float = 10.0,
              timeout_s: float = 3 * 3600) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    code, _ = _http(server, "GET", "/api/v1/health")
    if code != 200:
        raise SuiteError(f"{server} is not healthy (HTTP {code})")
    specs = suite_specs(model, hardware)
    for sp in specs:  # all of them, before any takes the GPU
        code, resp = _http(server, "POST", "/api/v1/jobs?dry_run=1", sp)
        if code != 200:
            raise SuiteError(f"{sp['metadata']['name']} is not a valid job: {(resp or {}).get('message', code)}")
    if dry_run:
        return None
    ids: dict[str, str] = {}
    for sp in specs:
        code, resp = _http(server, "POST", "/api/v1/jobs", sp)
        if code != 201:
            raise SuiteError(f"submitting {sp['metadata']['name']} failed: {(resp or {}).get('message', code)}")
        ids[resp["job_id"]] = sp["metadata"]["name"]
    deadline = time.monotonic() + timeout_s
    while True:
        _, jobs = _http(server, "GET", "/api/v1/jobs")
        state = {j["job_id"]: j["state"] for j in jobs or []}
        if any(state.get(i) in ("FAILED", "CANCELLED") for i in ids) or all(state.get(i) == "SUCCEEDED" for i in ids):
            break
        if time.monotonic() > deadline:
            raise SuiteError(f"jobs {', '.join(ids)} did not finish within {timeout_s:g} s")
        time.sleep(poll_s)
    reports = []
    for i, name in ids.items():
        _, detail = _http(server, "GET", f"/api/v1/jobs/{i}")
        if state.get(i) != "SUCCEEDED":
            err = ((detail or {}).get("report") or {}).get("error", "no error recorded")
            raise SuiteError(f"{name} ({i}) {state.get(i, 'is missing')}: {err}")
        reports.append(detail["report"])
    recipes = sorted({r for sp in specs for r in sp["quant"]["recipe"]})
    _, est = _http(server, "POST", "/api/v1/estimate", {"model": model, "hardware": hardware, "recipes": recipes, "backend": "vllm",
                                                        "runtime": {"concurrency": 1, "context_tokens": 1024}})
    return reports, (est or {}).get("estimates", [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="append", help="a finished LLM job's report.json (repeatable)")
    ap.add_argument("--suite", help="run a reference suite on a server instead of reading reports (only: m4)")
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--model", help="model directory the server can read (for --suite)")
    ap.add_argument("--hardware", default="nvidia/rtx4090")
    ap.add_argument("--dry-run", action="store_true", help="--suite: validate the jobs against the server, submit nothing")
    ap.add_argument("--poll", type=float, default=10.0, help="--suite: seconds between polls")
    ap.add_argument("--timeout", type=float, default=3 * 3600, help="--suite: give up after this many seconds")
    ap.add_argument("--estimates", help="JSON from POST /api/v1/estimate ({estimates: [...]}) for the memory-estimate comparison")
    ap.add_argument("--out", help="write markdown here instead of stdout")
    a = ap.parse_args(argv)
    try:
        if a.suite:
            if a.suite != "m4" or not a.model:
                print("bench_reference: --suite needs the name m4 and --model", file=sys.stderr)
                return 2
            got = run_suite(a.server, a.model, hardware=a.hardware, dry_run=a.dry_run, poll_s=a.poll, timeout_s=a.timeout)
            if got is None:
                print("bench_reference: all m4 jobs are valid; nothing was submitted (--dry-run)")
                return 0
            reports, est = got
        elif a.report:
            reports = [json.loads(Path(p).read_text()) for p in a.report]
            est = None
            if a.estimates:
                d = json.loads(Path(a.estimates).read_text())
                est = d["estimates"] if isinstance(d, dict) else d
        else:
            print("bench_reference: give --report FILE (repeatable) or --suite m4", file=sys.stderr)
            return 2
    except (OSError, ValueError, KeyError, SuiteError) as e:
        print(f"bench_reference: {e}", file=sys.stderr)
        return 2
    md = render(reports, est)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
