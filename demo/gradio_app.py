"""Operational Gradio demo for MQDEsys.

Unlike the static showcase, this page talks to a live `mqde serve` instance. It can:
  - inspect probe/backends/jobs
  - run a resource estimate
  - run deployment pre-evaluation over local or catalog models
  - validate (dry run) or submit a real job
  - search and download models from HuggingFace / ModelScope
  - browse the offline/native/EvalScope benchmark catalog

Start:
    MQDE_API=http://127.0.0.1:18081 python demo/gradio_app.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import gradio as gr
import pandas as pd
import requests

API = os.environ.get("MQDE_API", "http://127.0.0.1:18081").rstrip("/")
TIMEOUT = float(os.environ.get("MQDE_DEMO_TIMEOUT", "120"))


def _url(path: str) -> str:
    return f"{API}/api/v1{path}"


def _get(path: str, timeout: float = TIMEOUT) -> Any:
    r = requests.get(_url(path), timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()


def _post(path: str, body: Any, timeout: float = TIMEOUT) -> Any:
    r = requests.post(_url(path), json=body, timeout=timeout)
    if not r.ok:
        try:
            data = r.json()
            msg = data.get("message") or data.get("problems") or data
        except Exception:
            msg = r.text[:500]
        raise RuntimeError(f"POST {path} -> {r.status_code}: {msg}")
    return r.json() if r.text else {}


def _err(e: Exception) -> str:
    return f"**请求失败**：`{type(e).__name__}: {e}`\n\n请确认 `MQDE_API={API}` 指向正在运行的 `mqde serve`。"


def _gb(v: Any) -> str:
    try:
        return f"{float(v) / 2**30:.2f} GiB"
    except Exception:
        return "–"


def _load_meta():
    try:
        profiles = _get("/hardware/profiles")
        recipes = _get("/recipes")
        catalog = _get("/benchmarks")
        envs = _get("/backend-envs")["backends"]
        profile_ids = [p["id"] for p in profiles]
        recipe_ids = [r["id"] for r in recipes]
        try:
            detected = [d.get("profile") for d in _get("/hardware").get("devices", []) if d.get("profile")]
            default_hw = next((x for x in detected if x in profile_ids), None)
        except Exception:
            default_hw = None
        if not default_hw:
            default_hw = "nvidia/rtx4090" if "nvidia/rtx4090" in profile_ids else (profile_ids[0] if profile_ids else None)
        default_recipes = [x for x in ("bf16", "w4a16") if x in recipe_ids] or recipe_ids[:2]
        native = [b["name"] for b in catalog.get("native", [])]
        offline = [b["name"] for b in catalog.get("benchmarks", [])]
        benches = offline + native
        backends = [b["id"] for b in envs] + ["hf_transformers"]
        return (
            gr.update(choices=profile_ids, value=default_hw),
            gr.update(choices=recipe_ids, value=default_recipes),
            gr.update(choices=backends, value="hf_transformers"),
            gr.update(choices=benches, value=benches[:1]),
            gr.update(choices=["local", "catalog"], value="local"),
            f"已连接 `{API}` · {len(profile_ids)} 个硬件画像 · {len(recipe_ids)} 个配方 · {len(benches)} 个评测集",
        )
    except Exception as e:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), _err(e))


def load_hw():
    try:
        p = _get("/hardware/profiles")
        ids = [x["id"] for x in p]
        detected = [d.get("profile") for d in _get("/hardware").get("devices", []) if d.get("profile")]
        value = next((x for x in detected if x in ids), None)
        if not value:
            value = "nvidia/rtx4090" if "nvidia/rtx4090" in ids else (ids[0] if ids else None)
        return gr.update(choices=ids, value=value), gr.update(choices=ids, value=value)
    except Exception:
        return gr.update(), gr.update()


def load_job_meta():
    try:
        ids = [r["id"] for r in _get("/recipes")]
        value = [x for x in ("w4a16", "w4a8", "w2a16") if x in ids] or ids[:1]
        return gr.update(choices=ids, value=value)
    except Exception:
        return gr.update()


def overview():
    try:
        cfg = _get("/config")
        hw = _get("/hardware")
        envs = _get("/backend-envs")["backends"]
        jobs = _get("/jobs")
        devices = hw.get("devices", [])
        pool = hw.get("job_gpus", {}).get("pool", [])
        free = sum(1 for p in pool if not p.get("job"))
        md = [
            f"## MQDEsys Live Console\n\n**API**: `{API}`  \n**Runs**: `{cfg.get('runs_dir', '–')}`  \n**Allowed roots**: `{', '.join(cfg.get('allowed_roots', []))}`",
            f"\n### GPU pool\n\n**{free} / {len(pool)} 空闲**" if pool else "\n### GPU pool\n\n未配置 GPU 池",
        ]
        if devices:
            md.append("\n| kind | device | arch | target | memory free / total |\n|---|---|---|---|---|")
            for d in devices[:12]:
                md.append(
                    f"| {d.get('kind')} | {d.get('name')} | {d.get('arch')} | `{d.get('target', '–')}` | "
                    f"{_gb(d.get('mem_free_bytes'))} / {_gb(d.get('mem_total_bytes'))} |")
        md.append("\n### Inference backends\n\n| backend | installed | version | path |\n|---|---|---|---|")
        for b in envs:
            md.append(f"| {b['id']} | {'✓' if b['installed'] else '✗'} | {b.get('version') or '–'} | `{b.get('path') or '–'}` |")
        md.append(f"\n### Jobs\n\n共 **{len(jobs)}** 个作业，最近 8 个：")
        rows = [[j.get("job_id"), j.get("name"), j.get("state"), j.get("stage"), j.get("kind"), ", ".join(j.get("recipes") or [])] for j in jobs[:8]]
        return "\n".join(md), rows
    except Exception as e:
        return _err(e), []


def estimate(model: str, hardware: str, recipes: list[str], backend: str, concurrency: int, text_tokens: int,
             images: int, context: int, util: float):
    try:
        body = {
            "model": model.strip(), "hardware": hardware, "recipes": recipes, "backend": backend,
            "runtime": {"concurrency": int(concurrency), "images_per_step": int(images), "text_tokens": int(text_tokens),
                        "context_tokens": int(context), "image_resolution": 448,
                        "gpu_memory_utilization": float(util), "extra_modalities": []},
        }
        res = _post("/estimate", body)
        rows = []
        for e in res.get("estimates", []):
            rows.append([e.get("recipe"), _gb(e.get("peak_bytes")), _gb(e.get("capacity_bytes")),
                         "✓" if e.get("feasible") else "✗", e.get("max_concurrency"), e.get("perf", {}).get("latency_floor_ms"),
                         e.get("confidence")])
        md = f"### {model.split('/')[-1]} @ {hardware}\n\n{len(rows)} 个配方；峰值显存、可部署性和最大并发来自 C++ estimator。"
        return md, rows, res
    except Exception as e:
        return _err(e), [], {}


def fit(source: str, search: str, hardware: str, use_case: str, limit: int):
    try:
        body = {"source": source, "search": search.strip(), "hardware": hardware, "use_case": use_case,
                "limit": int(limit), "sort": "score"}
        res = _post("/fit", body)
        rows = []
        for m in res.get("models", []):
            rows.append([m.get("model"), m.get("fit_level"), m.get("run_mode"), m.get("best_quant"),
                         f"{m.get('memory_required_gb', 0):.2f}", f"{m.get('utilization_pct', 0):.0f}%",
                         f"{m.get('estimated_tps', 0):.2f}", f"{m.get('score', 0):.1f}"])
        md = f"### 部署预评估 · {len(rows)} 个模型\n\n来源：`{source}` · 硬件：`{hardware}` · 用途：`{use_case}`"
        return md, rows, res.get("errors", [])
    except Exception as e:
        return _err(e), [], []


def _job_body(mode: str, model: str, quantized_path: str, recipes: list[str], hardware: str,
              eval_dataset: str, benchmarks: list[str], limit: int, name: str, frames: int) -> dict[str, Any]:
    llm = {
        "domain": "llm",
        "metadata": {"name": name or f"gradio-{mode}-{int(time.time())}"},
        "model": {"type": "llm", "path": model.strip()},
        "deploy": {"device": "cpu", "hardware": hardware},
    }
    if mode == "evaluate":
        llm["stages"] = ["evaluate"]
        llm["model"]["quantized_path"] = quantized_path.strip()
        llm["evaluation"] = {"dataset": eval_dataset.strip(), "seq_len": 64, "max_windows": 1,
                             "benchmarks": benchmarks, "limit": int(limit)}
    elif mode == "quantize":
        llm["stages"] = ["quantize", "export"]
        llm["quant"] = {"recipe": recipes, "algorithm": "rtn"}
        llm["calibration"] = {"dataset": "synthetic", "n_samples": 8, "seq_len": 64}
        llm["export"] = {"formats": ["awq"]}
    else:
        llm["stages"] = ["quantize", "evaluate"]
        llm["quant"] = {"recipe": recipes, "algorithm": "rtn"}
        llm["calibration"] = {"dataset": "synthetic", "n_samples": 8, "seq_len": 64}
        llm["evaluation"] = {"dataset": eval_dataset.strip(), "seq_len": 64, "max_windows": 1,
                             "benchmarks": benchmarks, "limit": int(limit)}
    return llm


def submit_job(mode: str, model: str, quantized_path: str, recipes: list[str], hardware: str,
               eval_dataset: str, benchmarks: list[str], limit: int, name: str, dry: bool):
    try:
        body = _job_body(mode, model, quantized_path, recipes, hardware, eval_dataset, benchmarks, limit, name, 4)
        if dry:
            res = _post("/jobs?dry_run=1", body)
            return f"### Planner 校验通过\n\n```json\n{json.dumps(res, ensure_ascii=False, indent=2)}\n```", ""
        res = _post("/jobs", body)
        jid = res["job_id"]
        return f"### 作业已提交：`{jid}`\n\n在控制台打开：{API}/jobs/{jid}", jid
    except Exception as e:
        return _err(e), ""


def job_status(job_id: str):
    try:
        if not job_id.strip():
            return "请先填写作业 ID。", {}
        d = _get(f"/jobs/{job_id.strip()}")
        rep = d.get("report") or {}
        md = f"### `{d.get('job_id')}` · **{d.get('state')}** · stage `{d.get('stage')}`\n\n报告阶段：{', '.join(s.get('stage', '') for s in rep.get('stages', [])) or '尚未写入'}"
        return md, d
    except Exception as e:
        return _err(e), {}


def hub_search(provider: str, query: str, limit: int):
    try:
        if provider == "HuggingFace":
            res = _get(f"/hf/search?q={requests.utils.quote(query)}&limit={int(limit)}")
            rows = [[m["name"], m.get("provider"), m.get("pipeline_tag"), m.get("hf_downloads"), m.get("parameters_raw")] for m in res.get("models", [])]
        else:
            res = _get(f"/modelscope/search?q={requests.utils.quote(query)}&limit={int(limit)}")
            rows = [[m["name"], m.get("provider"), m.get("pipeline_tag"), m.get("downloads"), m.get("parameters_raw")] for m in res.get("models", [])]
        return f"### {provider} · {len(rows)} 条结果", rows, [m["name"] for m in res.get("models", [])]
    except Exception as e:
        return _err(e), [], []


def hub_download(provider: str, model: str, budget: str, list_only: bool):
    try:
        path = "/download" if provider == "HuggingFace" else "/modelscope/download"
        body = {"model": model.strip(), "list": bool(list_only)}
        if budget.strip():
            body["budget"] = budget.strip()
        res = _post(path, body)
        if list_only:
            rows = [[g["group"], g["bytes"], g["files"], g.get("quality")] for g in res.get("groups", [])]
            return f"### 可下载权重组\n\n推荐：`{res.get('chosen')}`", rows, ""
        return f"### 下载已启动：`{res.get('id')}`\n\n目录：`{res.get('dir')}`", [], res.get("id", "")
    except Exception as e:
        return _err(e), [], ""


def download_status(download_id: str):
    try:
        if not download_id.strip():
            return "请先填写下载 ID。"
        d = _get(f"/download/{download_id.strip()}/status")
        return f"### `{d.get('id')}` · **{d.get('state')}**\n\n{d.get('bytes_done')} / {d.get('bytes_total')} bytes · {d.get('files_done')}/{d.get('files_total')} files\n\n`{d.get('dir')}`\n\n{d.get('error', '')}"
    except Exception as e:
        return _err(e)


def benchmark_catalog():
    try:
        c = _get("/benchmarks")
        rows = []
        for source, key in (("offline", "benchmarks"), ("native", "native"), ("evalscope", "evalscope")):
            for b in c.get(key, []):
                rows.append([source, b["name"], b.get("label"), b.get("group"), b.get("runner"), ", ".join(b.get("metrics", [])), b.get("description", "")])
        return f"### 评测目录 · {len(rows)} 项\n\noffline / native / evalscope 三类；native 需要 `datasets` 包。", rows
    except Exception as e:
        return _err(e), []


CSS = """
.gradio-container {max-width: 1380px !important;}
.mqde-banner {padding: 14px 18px; border: 1px solid #7568ff55; border-radius: 12px; background: #7568ff12;}
"""

with gr.Blocks(title="MQDEsys Live Console") as demo:
    gr.Markdown(
        f"""<div class="mqde-banner"><b>MQDEsys Live Gradio Console</b><br>
        这不是静态截图：所有按钮都调用 <code>{API}</code> 的真实 REST API。提交作业会占用服务器资源；
        想先安全检查 JobSpec，请勾选 <b>Dry run</b>。静态展示页在 <code>http://127.0.0.1:19090/site/</code>。
        </div>"""
    )
    conn = gr.Markdown("正在连接 MQDEsys …")

    with gr.Tab("总览"):
        refresh_btn = gr.Button("刷新状态", variant="primary")
        overview_md = gr.Markdown()
        jobs_df = gr.Dataframe(headers=["作业", "名称", "状态", "阶段", "类型", "配方"], interactive=False, wrap=True)
        refresh_btn.click(overview, outputs=[overview_md, jobs_df])

    with gr.Tab("资源预估"):
        with gr.Row():
            est_model = gr.Textbox(label="模型路径", placeholder="<MODEL_DIR>/mqde-tiny-random-gpt2", scale=3)
            est_hw = gr.Dropdown(label="硬件画像", choices=[], scale=1)
            est_backend = gr.Dropdown(label="估算后端", choices=["hf_transformers"], value="hf_transformers", scale=1)
        est_recipes = gr.CheckboxGroup(label="配方", choices=[], value=[])
        with gr.Row():
            est_conc = gr.Number(label="并发", value=1, precision=0)
            est_text = gr.Number(label="文本 token", value=64, precision=0)
            est_images = gr.Number(label="图像数", value=0, precision=0)
            est_ctx = gr.Number(label="上下文", value=64, precision=0)
            est_util = gr.Slider(label="GPU 利用率", minimum=0.1, maximum=0.95, value=0.8, step=0.05)
        est_btn = gr.Button("运行预估", variant="primary")
        est_md = gr.Markdown()
        est_df = gr.Dataframe(headers=["配方", "峰值显存", "可用容量", "可部署", "最大并发", "延迟下界 ms", "置信度"], interactive=False)
        est_json = gr.JSON(label="原始响应")
        est_btn.click(estimate, inputs=[est_model, est_hw, est_recipes, est_backend, est_conc, est_text, est_images, est_ctx, est_util], outputs=[est_md, est_df, est_json])

    with gr.Tab("部署预评估"):
        with gr.Row():
            fit_source = gr.Radio(label="模型来源", choices=["local", "catalog"], value="local")
            fit_search = gr.Textbox(label="搜索", placeholder="qwen2.5 / llama / 空 = 全部")
            fit_hw = gr.Dropdown(label="硬件画像", choices=[])
            fit_use = gr.Dropdown(label="用途", choices=["auto", "general", "chat", "coding", "reasoning", "multimodal", "vla"], value="auto")
            fit_limit = gr.Number(label="模型上限", value=20, precision=0)
        fit_btn = gr.Button("运行预评估", variant="primary")
        fit_md = gr.Markdown()
        fit_df = gr.Dataframe(headers=["模型", "适配", "运行模式", "最佳量化", "显存 GiB", "占用", "速度", "评分"], interactive=False, wrap=True)
        fit_errors = gr.JSON(label="无法解析的模型")
        fit_btn.click(fit, inputs=[fit_source, fit_search, fit_hw, fit_use, fit_limit], outputs=[fit_md, fit_df, fit_errors])

    with gr.Tab("提交作业"):
        gr.Markdown("提交真实作业前建议先 **Dry run**。默认模型是 tiny 随机模型，只适合演示链路。")
        with gr.Row():
            job_mode = gr.Radio(label="模式", choices=["full", "quantize", "evaluate"], value="evaluate")
            job_name = gr.Textbox(label="作业名", value="gradio-demo")
            job_dry = gr.Checkbox(label="Dry run（只校验，不入队）", value=True)
        with gr.Row():
            job_model = gr.Textbox(label="基座模型路径", placeholder="<MODEL_DIR>/base-model", scale=2)
            job_qpath = gr.Textbox(label="已量化 checkpoint（仅评测）", placeholder="<MODEL_DIR>/quantized-checkpoint", scale=2)
        with gr.Row():
            job_recipes = gr.CheckboxGroup(label="量化配方", choices=[], value=[])
            job_hw = gr.Dropdown(label="目标硬件", choices=[])
        with gr.Row():
            job_dataset = gr.Textbox(label="评测语料", placeholder="<DATA_DIR>/evaluation.txt", scale=2)
            job_bench = gr.CheckboxGroup(label="Benchmark", choices=[])
            job_limit = gr.Number(label="Benchmark 样本上限", value=1, precision=0)
        submit_btn = gr.Button("提交 / 校验", variant="primary")
        submit_md = gr.Markdown()
        job_id_box = gr.Textbox(label="作业 ID")
        with gr.Row():
            status_btn = gr.Button("刷新作业状态")
            status_md = gr.Markdown()
        status_json = gr.JSON(label="作业详情")
        submit_btn.click(submit_job, inputs=[job_mode, job_model, job_qpath, job_recipes, job_hw, job_dataset, job_bench, job_limit, job_name, job_dry], outputs=[submit_md, job_id_box])
        status_btn.click(job_status, inputs=[job_id_box], outputs=[status_md, status_json])

    with gr.Tab("模型搜索 / 下载"):
        with gr.Row():
            hub_provider = gr.Radio(label="来源", choices=["HuggingFace", "ModelScope"], value="HuggingFace")
            hub_query = gr.Textbox(label="搜索", value="qwen2.5")
            hub_limit = gr.Number(label="结果数", value=10, precision=0)
            hub_search_btn = gr.Button("查询", variant="primary")
        hub_md = gr.Markdown()
        hub_df = gr.Dataframe(headers=["模型", "提供者", "任务", "下载量", "参数"], interactive=False, wrap=True)
        hub_choices = gr.Dropdown(label="选择模型", choices=[])
        with gr.Row():
            hub_budget = gr.Textbox(label="下载预算（如 12G，可空）")
            hub_list = gr.Checkbox(label="只列出权重组", value=True)
            hub_download_btn = gr.Button("开始 / 列出")
        hub_dl_md = gr.Markdown()
        hub_groups = gr.Dataframe(headers=["权重组", "字节", "文件数", "质量"], interactive=False)
        download_id = gr.Textbox(label="下载 ID")
        dl_status_btn = gr.Button("刷新下载状态")
        dl_status_md = gr.Markdown()
        hub_search_btn.click(hub_search, inputs=[hub_provider, hub_query, hub_limit], outputs=[hub_md, hub_df, hub_choices])
        hub_download_btn.click(hub_download, inputs=[hub_provider, hub_choices, hub_budget, hub_list], outputs=[hub_dl_md, hub_groups, download_id])
        dl_status_btn.click(download_status, inputs=[download_id], outputs=[dl_status_md])

    with gr.Tab("评测目录"):
        bench_btn = gr.Button("刷新目录", variant="primary")
        bench_md = gr.Markdown()
        bench_df = gr.Dataframe(headers=["来源", "名称", "显示名", "类别", "Runner", "指标", "说明"], interactive=False, wrap=True)
        bench_btn.click(benchmark_catalog, outputs=[bench_md, bench_df])

    demo.load(_load_meta, outputs=[est_hw, est_recipes, est_backend, job_bench, fit_source, conn])
    demo.load(load_hw, outputs=[job_hw, fit_hw], show_progress=False)
    demo.load(load_job_meta, outputs=[job_recipes], show_progress=False)
    demo.load(overview, outputs=[overview_md, jobs_df])
    demo.load(benchmark_catalog, outputs=[bench_md, bench_df])


if __name__ == "__main__":
    host = os.environ.get("GRADIO_HOST", "127.0.0.1")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, show_error=True, quiet=False, inbrowser=False, css=CSS, theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"))
