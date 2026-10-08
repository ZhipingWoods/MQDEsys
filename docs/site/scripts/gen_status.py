#!/usr/bin/env python3
"""Generate the DONE / PARTIAL / TODO tables of the Status & roadmap pages from ../TASKS.md.

Usage: python3 scripts/gen_status.py      (from docs/site/, or via `make status`)
Only the overview table of TASKS.md (rows starting with `| T-`) is read. The task titles in TASKS.md are
Chinese; the English page uses the EN map below (add an entry when a new task appears, otherwise the
Chinese title is used as a fallback and the script prints a warning).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE.parents[1] / "TASKS.md"
MAX_NOTE = 150

EN = {
    "T-0001": "Repository init and Git conventions", "T-0002": "C++ build system (CMake + presets)",
    "T-0003": "Python package skeleton and pybind11 build", "T-0004": "Web project skeleton",
    "T-0005": "Coding standards and pre-commit", "T-0006": "CI pipeline (PR gate)",
    "T-0007": "Contracts and schemas", "T-0008": "Development guide and ADRs",
    "T-0101": "common: errors, Result, logging", "T-0102": "common: config loading and schema validation",
    "T-0103": "common: registry, hashing, thread pool", "T-0104": "storage: SQLite metadata and migrations",
    "T-0105": "storage: content-addressed cache and GC", "T-0106": "CLI skeleton",
    "T-0201": "C hardware probe: CPU", "T-0202": "C hardware probe: GPU/NPU (dynamic vendor libs)",
    "T-0203": "Hardware profile library", "T-0204": "hub: HF/ModelScope resolution and metadata",
    "T-0205": "modelspec: ModelSpec from config + safetensors headers", "T-0206": "Architecture templates and QIR",
    "T-0207": "Quantization recipe library", "T-0208": "estimator: memory / KV / concurrency",
    "T-0209": "estimator: performance estimate and calibration store", "T-0210": "planner: compatibility matrix and JobSpec validation",
    "T-0211": "planner: recommendation and deployability matrix", "T-0212": "CLI: probe / model inspect / estimate / plan",
    "T-0301": "server: HTTP skeleton and read-only endpoints", "T-0302": "server: estimation endpoints",
    "T-0303": "web: API client, layout, routing", "T-0304": "web: new-job wizard with live estimate sidebar",
    "T-0305": "web: estimation center charts", "T-0306": "web: hardware page",
    "T-0401": "quant core: QuantSpec / QParams / granularity", "T-0402": "quant core: calibration (MinMax/MSE/percentile/EMA)",
    "T-0403": "quant core: INT/FP8 quantize, dequantize, fake-quant + golden", "T-0404": "quant core: error metrics",
    "T-0405": "quant core: bit packing (GPTQ/AWQ/Marlin/GGUF)", "T-0406": "quant core: SIMD/CUDA acceleration",
    "T-0407": "pybind11 binding mqde._core", "T-0408": "Torch fast path and dual-end golden test",
    "T-0501": "Worker RPC protocol and C++ WorkerClient", "T-0502": "Python worker process framework",
    "T-0503": "pipeline: DAG, stage interface, state machine", "T-0504": "pipeline: queue, scheduling, device allocation, retry",
    "T-0505": "Model download (fetch_model stage)", "T-0506": "Python model abstraction (BaseModel + HF decoder)",
    "T-0507": "Calibration data and first-block capture", "T-0508": "Blockwise engine + fake/real modules + RTN",
    "T-0509": "Error tracing (per layer / block)", "T-0510": "Evaluation: PPL + logits probe",
    "T-0511": "Stages S00-S08 + S14 wired together", "T-0512": "server: job endpoints + event stream",
    "T-0513": "web: job execution page", "T-0514": "web: quantization error page (initial)",
    "T-0515": "End-to-end minimal loop (M3)", "T-0601": "Algorithm: AWQ", "T-0602": "Algorithm: GPTQ",
    "T-0603": "Algorithm: SmoothQuant", "T-0604": "Activation quantization (dynamic/static): W4A8, W2A8",
    "T-0605": "FP8 recipe (E4M3/E5M2, per-channel/per-block)", "T-0606": "2-bit recipes (W2A16/W2A8) and HQQ",
    "T-0607": "KV cache quantization", "T-0608": "Mixed precision and keep-FP policy",
    "T-0609": "External algorithm adapters (GPTQModel / llm-compressor)", "T-0610": "Algorithm combos (AWQ+GPTQ) and QuaRot",
    "T-0701": "Exporter interface + vLLM export", "T-0702": "Fake/real consistency check (S10)",
    "T-0703": "Deploy backend interface + vLLM adapter (S11)", "T-0704": "SGLang export and adapter",
    "T-0705": "llama.cpp (GGUF) export and adapter", "T-0706": "TensorRT-LLM export and adapter",
    "T-0801": "Accuracy evaluation: lm-eval-harness / evalscope / custom", "T-0802": "Post-deployment accuracy evaluation (S12)",
    "T-0803": "Performance load tester (S13): TTFT/TPOT/throughput", "T-0804": "Memory sampling and OOM boundary search",
    "T-0805": "HAL micro-benchmarks", "T-0806": "Report generation (JSON + HTML)",
    "T-0807": "web: evaluation report and compare pages", "T-0808": "M4 acceptance: all recipes x multiple backends",
    "T-0901": "Stage attribution (waterfall) data path", "T-0902": "Sensitivity analysis (per-layer fallback)",
    "T-0903": "web: full error page", "T-0904": "Mixed-precision suggestion -> one-click new job",
    "T-0905": "Feedback loop (S15): automatic estimate calibration", "T-0906": "Error attribution acceptance (injection test)",
    "T-1001": "Multimodal framework: per-modality config", "T-1002": "VLM adapters (Qwen2-VL / LLaVA)",
    "T-1003": "MoE / MLA support (DeepSeek family)", "T-1004": "VLA adapter (experimental)",
    "T-1005": "World model / video generation (DiT) adapter", "T-1101": "HAL ABI and plugin loader + template plugin",
    "T-1102": "CPU / CUDA plugin", "T-1103": "Metal plugin", "T-1104": "ROCm plugin", "T-1105": "Ascend (CANN) plugin",
    "T-1106": "Plugin SDK docs and ABI compatibility check", "T-1201": "Docker image and compose",
    "T-1202": "pip wheel packaging and release", "T-1203": "Documentation site (GitHub Pages)",
    "T-1204": "Security hardening (credentials, path sandbox, auth)", "T-1205": "Observability (/metrics, reproducible replay)",
    "T-1206": "Nightly GPU regression and performance board", "T-1207": "v1.0 release acceptance",
}

TXT = {
    "en": {
        "intro": "Generated from the overview table of `docs/TASKS.md` by `docs/site/scripts/gen_status.py` ({d} DONE, {p} PARTIAL, {t} TODO; other states: {o}). The task list in the repository is the source of truth; run `make status` after it changes. Notes are shortened and copied verbatim (English) from the repository.",
        "done": "DONE", "partial": "PARTIAL", "todo": "TODO",
        "partial_intro": "Usable but not fully accepted. The note says what exists and what is missing.",
        "todo_intro": "Planned, not started. Nothing in this table works today.",
        "done_intro": "Implemented and verified as the task's acceptance describes.",
        "cols": ("Task", "Title"), "colsp": ("Task", "Title", "What exists / what is missing"),
    },
    "zh": {
        "intro": "本页由 `docs/site/scripts/gen_status.py` 根据 `docs/TASKS.md` 的总览表生成（DONE {d} 项，PARTIAL {p} 项，TODO {t} 项，其他状态 {o} 项）。以仓库里的任务清单为准；清单变化后运行 `make status` 重新生成。说明列从仓库原文截取（英文）。",
        "done": "DONE（已完成）", "partial": "PARTIAL（部分完成）", "todo": "TODO（计划中）",
        "partial_intro": "可以使用，但未完全验收。说明列写了已有的部分和缺口。",
        "todo_intro": "已规划、尚未开始。下表中的功能目前都不可用。",
        "done_intro": "已实现，并按任务的验收标准验证过。",
        "cols": ("任务", "标题"), "colsp": ("任务", "标题", "已有 / 缺口"),
    },
}


def parse() -> list[dict[str, str]]:
    rows = []
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| T-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        tid, title, stage, _dep, _lvl, status = cells[:6]
        kind = status.split()[0].rstrip(":").upper()
        m = re.search(r"\((.*)\)\s*$", status)
        note = m.group(1) if m else status[len(kind):].strip()
        rows.append({"id": tid, "title": title, "stage": stage, "kind": kind, "note": note})
    return rows


def short(s: str) -> str:
    s = s.replace("|", "/").replace("`", "").replace("git init on 4090", "git init on the dev server")
    s = re.sub(r"/cache/\S+", "<path>", s)
    return s if len(s) <= MAX_NOTE else s[: MAX_NOTE - 1].rstrip() + "…"


def render(lang: str, rows: list[dict[str, str]]) -> str:
    t = TXT[lang]
    by = {k: [r for r in rows if r["kind"] == k] for k in ("DONE", "PARTIAL", "TODO")}
    other = len(rows) - sum(len(v) for v in by.values())
    title = lambda r: EN.get(r["id"], r["title"]) if lang == "en" else r["title"]  # noqa: E731
    out = [t["intro"].format(d=len(by["DONE"]), p=len(by["PARTIAL"]), t=len(by["TODO"]), o=other), ""]
    for kind, key in (("PARTIAL", "partial"), ("TODO", "todo"), ("DONE", "done")):
        out += [f"## {t[key]}", "", t[key + "_intro"], ""]
        if kind == "PARTIAL":
            a, b, c = t["colsp"]
            out += [f"| {a} | {b} | {c} |", "|---|---|---|"]
            out += [f"| {r['id']} | {title(r)} | {short(r['note'])} |" for r in by[kind]]
        else:
            a, b = t["cols"]
            out += [f"| {a} | {b} |", "|---|---|"]
            out += [f"| {r['id']} | {title(r)} |" for r in by[kind]]
        out.append("")
    return "\n".join(out)


def main() -> None:
    rows = parse()
    for r in rows:
        if r["id"] not in EN:
            print(f"warning: no English title for {r['id']}", file=sys.stderr)
    for lang in ("en", "zh"):
        target = HERE.parent / lang / "_status_tables.md"
        target.write_text(render(lang, rows), encoding="utf-8")
        print(f"wrote {target.relative_to(HERE.parent)} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
