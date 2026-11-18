# 系统架构

MQDEsys 采用控制面与执行器分离的架构：C++ 控制面负责模型解析、资源预估、硬件探测、API 服务与作业调度；Python 执行器负责 PyTorch 相关的量化算法、评测、导出与部署适配。两部分通过进程边界与事件流解耦。

## 运行时结构

```text
浏览器 ──HTTP/SSE──▶ mqde serve（C++23，cpp-httplib）
终端 ─ mqde CLI ─▶   ├─ REST API、静态页面、事件流
                     ├─ 预估器、fit 评分、模型解析、硬件探针
                     └─ 作业运行器：队列、GPU 池、每作业一个子进程
                              │ stdout/stderr、events.jsonl
                              ▼
                     python -m mqde.pipeline.{llm_job,vla_job,...}（Python 执行器）
                              │ 通过 127.0.0.1 上的 HTTP 控制子进程
                              ▼
                     llama-server、vllm serve、sglang.launch_server、llama-perplexity、llama-bench
```

### 进程模型

每个作业由独立子进程执行。C++ 服务从 GPU 池中分配设备（或唯一的 CPU 槽位），以作业配置启动 Python 流水线，并持续读取其 `events.jsonl`。作业崩溃或显存不足不会影响服务进程。

作业状态持久化在运行目录 `<runs>/<job-id>/`，包含 `job.yaml`、`events.jsonl`、`report.json`、`worker.log` 与 `export/`。因此由命令行启动的作业同样会出现在 Web 界面中。

### 事件流

执行器按行写入 JSON 事件。服务端从 `after_seq` 开始回放，并通过 Server-Sent Events 持续推送至作业结束。事件类型包括 `job_state`、`stage_started`、`stage_finished`、`log`、`metric`、`progress` 与 `trace`。

### 数值一致性

`cpp/src/quant/ops.cpp` 是量化算术的规范实现；`python/mqde/quant/numerics.py` 提供等价的 PyTorch 实现，并通过与 `mqde._core` 对拍的 golden 测试保证一致性。GPTQ、AWQ、Marlin 与 GGUF 的位打包实现位于 `cpp/src/quant/pack.cpp`，通过 `mqde._core` 暴露给 Python。

### 数据驱动

配方、硬件画像、后端画像、算法列表、模型架构模板与 fit 常数均定义在 `data/` 下的 YAML 文件中。预估器、服务端、Web 界面与 Python 执行器读取同一份数据。

## C++ 模块

模块依赖方向为：`common` ← `modelspec`、`quant` ← `estimator` ← `fit`、`hub` ← `server` ← 应用与绑定。下层模块不引用上层模块，该规则由 `scripts/check_deps.py` 校验。

| 模块 | 职责 |
|---|---|
| `common` | `Result<T>` 错误类型、YAML/JSON 配置加载、日志 |
| `modelspec` | safetensors 头部解析、`ModelSpec`、架构模板、层图 |
| `quant` | 量化规格、伪量化、误差指标、位打包 |
| `estimator` | 显存、KV/状态、FLOPs、延迟下界、可部署性 |
| `fit` | 模型与硬件适配评分、速度模型、本机基线索引 |
| `hub` | Hugging Face 与 ModelScope 目录、搜索与下载 |
| `server` | HTTP 端点与作业运行器 |
| `probe_c` | C99 硬件探针（动态加载 NVML；CPU 探测） |

## Python 包

| 包 | 职责 |
|---|---|
| `quant/` | 数值实现、量化算法、逐块引擎、KV 量化、伪量化模块 |
| `models/` | 模型适配器 |
| `data/` | 校准与评测数据加载 |
| `eval/` | PPL、压测、部署后 PPL、资源采样、边界搜索、数据集评测 |
| `export/` | checkpoint 导出器与真实性校验 |
| `backend/` | llama.cpp、vLLM、SGLang 适配器 |
| `pipeline/` | 作业流水线与事件输出 |
| `trace/` | 敏感度分析 |

## 技术选型

| 领域 | 实现 |
|---|---|
| C++ 标准 | C++23 |
| HTTP 服务 | cpp-httplib |
| 事件传输 | Server-Sent Events |
| 依赖管理 | conda（Linux）、Homebrew（macOS） |
| 作业编排 | Python 流水线 |
| 状态存储 | 运行目录中的作业文件与事件流 |
| 硬件探测 | 动态加载 NVML 的 C 探针 |

## 部署拓扑

服务默认绑定 `127.0.0.1`，Web 静态资源由同一进程托管。模型权重与作业目录保留在部署主机本地，不经过外部服务。远程访问通过 SSH 隧道实现。
