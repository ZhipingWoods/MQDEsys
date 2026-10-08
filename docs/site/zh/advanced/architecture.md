# 架构

本页概括 `docs/ARCHITECTURE.md` 中的设计，并标出哪些已经存在。原设计中的若干部分（C++ DAG 引擎、JSON-RPC worker、SQLite 存储、硬件插件 ABI）**尚未实现**；当前系统是一个端到端可用的纵向切片。

## 今天实际在跑的东西

```text
浏览器 ──HTTP/SSE──▶ mqde serve（C++ 23，cpp-httplib）
终端 ─ mqde CLI ─▶   ├─ REST API、静态网页、事件流
                     ├─ 预估器、fit 评分、模型解析、硬件探针
                     └─ 作业运行器：队列、GPU 池、每个作业一个子进程
                              │ stdout/stderr、events.jsonl
                              ▼
                     python -m mqde.pipeline.llm_job | vla_job   （Python 执行器）
                              │ 通过 127.0.0.1 上的 HTTP 控制的子进程
                              ▼
                     llama-server、vllm serve、llama-perplexity、llama-bench
```

- **每个作业一个进程。** C++ 服务从池里分配一张 GPU（或唯一的 CPU 槽位），带着作业配置启动 Python 流水线，并跟随它的 `events.jsonl`；崩溃或显存不足不会影响服务本身。作业状态保存在运行目录（`<runs>/<job-id>/`：`job.yaml`、`events.jsonl`、`report.json`、`worker.log`、`export/`）里，所以从命令行启动的作业也会出现在界面里。服务重启后，运行中作业的状态由事件文件推断（已知限制，B-5）。
- **事件流。** 执行器每行写一个 JSON 事件；服务从 `after_seq` 起回放，然后持续跟随（Server-Sent Events）。设计原本是 WebSocket，现选用 SSE 因为已足够，这是登记在案的偏离。
- **数值只有一个定义。** `cpp/src/quant/ops.cpp` 是规范；`python/mqde/quant/numerics.py` 用 PyTorch 镜像它，并由对拍测试（`python/tests/test_numerics_golden.py`，与 `mqde._core` 比较）要求两者一致。GPTQ、AWQ、Marlin 和 GGUF 的位打包在 C++（`cpp/src/quant/pack.cpp`）中，并作为 `mqde._core` 暴露。
- **数据优先于代码。** 配方、硬件画像、后端画像、算法列表、模型架构模板和 fit 常数都是 `data/` 下的 YAML。预估器、服务、界面和 Python 执行器读的是同一批文件。

## C++ 模块与依赖规则

`common` ← `modelspec`、`quant` ← `estimator` ← `fit`、`hub` ← `server` ← apps/bindings。下层绝不 include 上层；`scripts/check_deps.py` 强制检查（它是本地门禁的一部分）。

| 模块 | 职责 |
|---|---|
| `common` | `Result<T>` 错误、YAML/JSON 配置加载、日志 |
| `modelspec` | safetensors 头解析、`ModelSpec`、架构模板、层图 |
| `quant` | 量化规格、fake-quant、误差指标、位打包 |
| `estimator` | 显存、KV/状态、FLOPs、延迟下界、可部署性 |
| `fit` | 仿 llmfit 的排序、速度模型、本机基线索引 |
| `hub` | Hugging Face 目录、搜索与下载 |
| `server` | HTTP 端点、作业运行器 |
| `probe_c` | C99 硬件探针（动态加载 NVML；CPU） |

## Python 包

`quant/`（数值、算法、逐块引擎、KV 量化、fake-quant 模块）、`models/`（模型适配器）、`data/`（校准与评估数据）、`eval/`（PPL、压测、部署后 PPL、资源采样、边界搜索）、`export/`（导出器、real check）、`backend/`（llama.cpp 与 vLLM 适配器）、`pipeline/`（作业流水线与事件输出）、`trace/`（敏感度分析）。

## 与书面设计的偏离

设计文档描述的系统更大。截至今天：HTTP 用的是 cpp-httplib 而不是 Drogon；事件用 SSE 而不是 WebSocket；C++ 依赖来自 conda 或 Homebrew 而不是 vcpkg；作业 DAG 在 Python 里；C++ 以 C++23 构建；没有 SQLite 存储、没有内容寻址缓存、没有 JSON-RPC worker，也没有硬件插件加载器。见 [状态与路线图](../status.md) 以及 `docs/PROGRESS.md` 中的决策清单。
