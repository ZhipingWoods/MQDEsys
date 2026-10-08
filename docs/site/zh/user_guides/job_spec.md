# 作业配置参考

**状态：已实现。** 作业由 YAML 文件或 JSON 请求体（`POST /api/v1/jobs`）描述；LLM 流水线把它读成 `LLMJobSpec`（`python/mqde/pipeline/llm_job.py`）。服务端补上的字段（`apiVersion: mqde/v1`、`kind: Job`）会被流水线忽略。

## 最小示例

```yaml
metadata: {name: qwen05b-w4a16}
model: {path: <MODEL_DIR>}
quant: {recipe: [w4a16], algorithm: gptq}
deploy: {device: cpu}
output_dir: runs
```

仓库示例 `examples/jobs/qwen25_05b_wikitext.yaml` 还设置了 `act_static`、`kv`、`mixed_precision`、`calibration` 和 `evaluation`。

## YAML 字段（LLM 作业）

| 键 | Spec 字段 | 默认值 | 含义 |
|---|---|---|---|
| `metadata.name` | `name` | 文件名 | 界面上显示的作业名 |
| `model.path` | `model_path` | 必填 | 本地 Hugging Face 模型目录。服务端要求它位于某个 `--allow` 根目录下 |
| `model.type` | （服务端） | `vla` | `llm` 或 `vla`；服务端据此把作业交给对应流水线。使用 LLM 流水线时请设为 `llm` |
| `quant.recipe` | `recipes` | 必填 | 一个配方或配方列表（`w4a16`、`w8a16`、`fp8`、`w4a8`、`w2a16`、`w2a8`、`bf16`） |
| `quant.algorithm` | `algorithm` | `recipe` | `rtn`、`gptq`、`awq`、`smoothquant`、`awq+gptq`、`smoothquant+gptq`，或 `recipe`（各配方用自己的默认算法） |
| `quant.act_static` | `act_static` | `null` | `minmax`、`ema` 或 `percentile` 表示静态 per-tensor 激活 scale；`null` 为动态 per-token |
| `quant.kv` | `kv_override`、`kv_method`、`kv_granularity` | 关闭 | `{format, bits, symmetric, method, granularity}`；见 [量化](quantization.md) |
| `quant.mixed_precision` | `mixed_precision` | 无 | `{rules: [{match, bits | skip}]}` |
| `calibration.n_samples` | `calib_samples` | 32 | 校准窗口数 |
| `calibration.seq_len` | `calib_seq_len` | 512 | 每个校准窗口的 token 数 |
| `evaluation.seq_len` | `eval_seq_len` | 512 | 每个 PPL 窗口的 token 数 |
| `evaluation.max_windows` | `eval_windows` | 40 | PPL 窗口数 |
| `deploy.device` | `device` | `cpu` | `cpu`，或量化时使用的 CUDA 设备。经服务端提交时，`cpu` 使用唯一的 CPU 槽位，其他值需要 `--gpus` 中的一张 GPU |
| `deploy.llamacpp.types` | `llamacpp` | `[]` | 要部署的 GGUF 类型：`q8_0`、`q5_0`、`q4_0`、`iq4_nl`、`q4_k`、`q2_k` |
| `deploy.llamacpp.gpu_layers` | `llamacpp_gpu_layers` | llama.cpp 默认 | 即 `-ngl`；0 为 CPU，99 为全部卸载（Metal / CUDA 构建） |
| `deploy.llamacpp.serve` | `llamacpp_serve` | `{concurrency: 4, requests: 16, prompt_len: 256, n_predict: 64}` | 对每种 GGUF 类型的 `llama-server` 压测；还可设 `ctx`。`null` 表示关闭 |
| `deploy.vllm` | `vllm` | 关闭 | `true` 或一个映射：用 vLLM 部署每个导出的 checkpoint（需要 CUDA）。键：`max_model_len`（1024）、`gpu_memory_utilization`（0.3）、`enforce_eager`（true）、`concurrency`（4）、`requests`（16）、`prompt_len`（256）、`n_predict`（64）、`ppl_windows` |
| `export.formats` | `export` | `[]` | `gptq`、`awq`、`compressed-tensors` |
| `analysis.sensitivity` | `sensitivity` | 关闭 | `block` 或 `type` |
| `analysis.sensitivity_windows` | `sensitivity_windows` | 8 | 每次敏感度探测使用的 PPL 窗口数 |
| `output_dir` | `output_dir` | `runs` | 作业写入 `<output_dir>/<job-id>/` |

`seed`（0）和 `suggest_target`（0.5）是 spec 的字段，但不从 YAML 读取。

## 命令行

`python -m mqde.pipeline.llm_job [spec] [options]`（需要 `PYTHONPATH=python`）。YAML `spec` 可省略；命令行选项会覆盖它。

| 选项 | 含义 |
|---|---|
| `--model` | 模型目录 |
| `--recipes` | 逗号分隔的配方（没有 YAML 时默认 `w4a16`） |
| `--algorithm` | 见上表 |
| `--device` | `cpu` 或 `cuda` |
| `--act-static {minmax,ema,percentile}` | 静态激活 scale |
| `--kv` | `int8`、`int4` 或 `fp8`：覆盖配方的 KV 格式 |
| `--kv-method {kivi,naive}` | KV 方法 |
| `--eval-windows`、`--calib-samples` | 评估与校准规模 |
| `--mixed-precision` | 内联 JSON，或含 `{rules: [...]}` 的 YAML/JSON 文件 |
| `--export` | 逗号分隔：`gptq`、`awq`、`compressed-tensors` |
| `--sensitivity {block,type}`、`--sensitivity-windows` | 敏感度分析 |
| `--llamacpp` | GGUF 类型，例如 `q8_0,q4_k` |
| `--llamacpp-gpu-layers` | llama.cpp 的 `-ngl` |
| `--llamacpp-serve` | JSON 压测设置，或 `off` |
| `--vllm [JSON]` | 用 vLLM 部署导出的 checkpoint；可选 JSON，例如 `'{"gpu_memory_utilization": 0.4}'` |
| `--job-dir` | 输出目录（默认 `<output_dir>/<job-id>`） |
| `--job-id` | 服务端分配的 id |

命令结束时会打印一个 JSON 摘要（作业 id、基线 PPL、各配方 PPL 与算法）。作业目录里有 `report.json`、`events.jsonl`，若有导出或 GGUF 文件还有 `export/`。

## VLA 作业

`python -m mqde.pipeline.vla_job <spec> [--algorithm ...]` 运行 MiniCPM-RobotManip 的动作误差评估（仅伪量化，需要 GPU）；模板是 `examples/jobs/minicpm_vla_libero.yaml` 和 `minicpm_vla_smoke.yaml`。VLA 流水线报告相对 BF16 模型的动作误差，属于实验性质，这里不再展开。

## 事件与报告

`events.jsonl` 每行一个 JSON 对象：`job_state`（`RUNNING`、`SUCCEEDED`、`FAILED`）、`stage_started`、`stage_finished`（含 `seconds`、`status`）、`log`、`metric`（`name`、`value`）、`progress`（`value` 取 0..1）和 `trace`（逐层误差）。`report.json` 的键包括 `baseline`、`recipes.<NAME>`（含 `eval`、`summary`、`layers`、`export`、`sensitivity`）、`llamacpp`、`env` 和 `stages`。请把报告视为仍在演进的格式：它还没有版本号，也没有 schema 校验（T-0007）。
