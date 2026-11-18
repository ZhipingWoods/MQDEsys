# REST API 参考

服务（`mqde serve`，cpp-httplib）在 `/api/v1` 下提供 JSON over HTTP，作业进度通过 Server-Sent Events 推送。接口定义位于 `cpp/src/server/server.cpp`，Web 客户端类型位于 `web/src/api.ts`。以下示例均可对运行中的服务直接执行。

约定：请求体是 JSON（空请求体视为 `{}`）；错误形如 `{"code": "...", "message": "..."}`，状态码为 400（`invalid_argument`、`parse`）、403（`forbidden`：路径超出 `--allow`）、404、409、500 或 503。模型和数据集路径必须位于允许的根目录下。自定义动作用冒号表示（`/jobs/<id>:cancel`）。服务默认仅监听回环地址，不启用认证。

## 系统

| 方法与路径 | 作用 |
|---|---|
| `GET /health` | `{status, version}` |
| `GET /config` | `{allowed_roots, gpus, runs_dir}` |
| `GET /hardware` | 探测到的设备（`devices`，含匹配的 `profile`）和作业 GPU 池（`job_gpus`） |
| `GET /hardware/profiles` | 硬件画像库 |
| `GET /benchmarks` | 离线 benchmark 元数据、套件和 EvalScope 数据集目录（`evalscope`，仅元数据） |
| `GET /datasets` | 向导可用的内置数据集与允许的根目录 |
| `GET /datasets/inspect?path=&kind=` | 只读检查一个数据文件 / 目录是否可用 |
| `GET /recipes` | 量化配方（含 `id`、规格、`summary`） |
| `GET /algorithms` | 量化算法（来自 `data/algorithms.yaml`） |
| `GET /installed` | 本机发现的推理运行时（缓存 30 s） |

```bash
curl -s localhost:18080/api/v1/health
# {"status":"ok","version":"0.1.0"}
```

## 模型与预估

| 方法与路径 | 请求体 | 作用 |
|---|---|---|
| `GET /models` | | 允许根目录下的模型（含 `config.json` 的目录），每个以解析后的 spec 表示 |
| `POST /models:inspect` | `{path, tensors?}` | 解析 config 与 safetensors 头 |
| `POST /models:graph` | `{path, focus?, depth?}` | 类 netron 的层级结构 |
| `POST /models:tensor` | `{path, tensor, recipe?}` | 单个张量的统计、64 bin 直方图和误差预览 |
| `POST /estimate` | `{model, hardware?, backend?, recipes?, runtime?, concurrency_curve?, images_curve?}` | 各配方的预估与曲线；`runtime` 的键：`concurrency`、`images_per_step`、`text_tokens`、`context_tokens`、`image_resolution`、`gpu_memory_utilization`、`extra_modalities` |
| `POST /estimate:matrix` | `{model, backend?, runtime?}` | 配方 × 硬件的可部署性格子 |
| `POST /fit` | `{models?, source?, search?, hardware?, gpu_count?, ram_gb?, vram_gb?, use_case?, context?, concurrency?, quant?, sort?, backend?}` | 排序后的适配分析；`source: "catalog"` 对 Hub 目录里的模型排序 |
| `POST /fit:plan` | `{model, ...}` | 每个硬件画像上的最佳配方 |
| `POST /fit:claim` | `{model, hardware?, min_tps?}` | 吞吐上下界 |
| `POST /fit:concurrency` | `{model, hardware?, recipe?, users?, context?, curve?}` | 能容纳的用户数及曲线 |

默认值：`hardware` 为 `nvidia/rtx4090`，`backend` 为 `hf_transformers`，`recipes` 为 `["bf16","w4a16"]`。

```bash
curl -s -X POST localhost:18080/api/v1/estimate \
  -d '{"model": "<MODEL_DIR>", "hardware": "nvidia/rtx4090", "recipes": ["w4a16"]}'
```

## 模型目录与下载

| 方法与路径 | 作用 |
|---|---|
| `GET /catalog?search=&provider=&capability=&limit=&include_demoted=` | 本地模型目录 |
| `GET /hf/search?q=&limit=` | 在线 Hugging Face 搜索（需要网络） |
| `GET /modelscope/search?q=&limit=` | 在线 ModelScope 搜索（需要网络） |
| `POST /download` | `{model: "<org>/<name>", quant?, budget?, output_dir?, list?}`：开始下载（`list: true` 则只列出文件分组）；返回下载 id |
| `POST /modelscope/download` | 同上，从 ModelScope 下载；状态中的 `provider` 为 `modelscope` |
| `GET /download/<id>/status` | 进度 |
| `POST /download/<id>:cancel` | 取消 |

## 作业

| 方法与路径 | 作用 |
|---|---|
| `POST /jobs` | 提交 [作业配置](job_spec.md)；返回 `201 {job_id}`。会校验路径、配方、`quant.algorithm`、`model.type`（默认 `vla`，或 `llm`），以及 CPU 作业必须是 LLM 作业 |
| `GET /jobs` | 全部作业（id、类型、状态、阶段、配方、设备、模型、名称），按时间倒序 |
| `GET /jobs/<id>` | 状态、阶段、事件数、`report`（作业的 `report.json`，运行中为空）、提交的 `job` 配置 |
| `POST /jobs/<id>:cancel` | 取消排队中或运行中的作业（其他状态返回 409） |
| `GET /jobs/<id>/log` | 执行器日志（最后 256 KiB，纯文本） |
| `GET /jobs/<id>/events?after_seq=N` | `seq > N` 的事件，JSON 数组 |
| `GET /jobs/<id>/stream?after_seq=N` | 同样的事件，以 Server-Sent Events 推送，持续跟随到作业结束；用最后的 `seq` 续传 |

状态：`QUEUED`、`RUNNING`、`SUCCEEDED`、`FAILED`、`CANCELLED`。事件类型：`job_state`、`stage_started`、`stage_finished`、`log`、`metric`、`progress`、`trace`；每个事件有单调递增的 `seq`。

```bash
curl -s -X POST localhost:18080/api/v1/jobs -d '{
  "metadata": {"name": "demo"}, "model": {"type": "llm", "path": "<MODEL_DIR>"},
  "quant": {"recipe": ["w4a16"], "algorithm": "gptq"}, "deploy": {"device": "cpu"}}'
curl -N "localhost:18080/api/v1/jobs/<job-id>/stream?after_seq=0"
```

`/api/` 之外的其他 `GET` 请求返回 Web 界面（单页应用，回退到 `index.html`）。
