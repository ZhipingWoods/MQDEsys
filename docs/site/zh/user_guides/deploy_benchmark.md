# 部署与压测

**状态：部分实现。** llama.cpp（GGUF，CPU 与 Metal）和 vLLM 已在 Python 中端到端跑通。C++ 后端接口、SGLang、TensorRT-LLM、lm-eval/evalscope 精度评测，以及边界搜索的报告页尚未完成。

导出之后，作业可以用真实引擎部署每个 checkpoint，确认它能应答、给它加压并给它打分。找不到引擎二进制时该阶段记为 `skipped`（原因写在报告里）；启动失败记为 `failed`，并附上从引擎日志识别出的诊断（显存不足、不支持的量化配置、KV Cache 太小、端口被占用、文件缺失等）。

服务始终绑定 `127.0.0.1`，因为它们没有认证。

## llama.cpp（GGUF）

在源码树之外构建 llama.cpp（CPU，或 Apple 芯片上的 Metal），并让 MQDEsys 找到二进制（见 [安装](../get_started/installation.md)）。Metal 用 `-DGGML_METAL=ON` 配置；开发服务器上的 CPU 构建用 `-DGGML_CUDA=OFF`。`-DLLAMA_CURL=OFF` 可免去一个依赖。需要的目标是 `llama-perplexity`、`llama-quantize`、`llama-bench` 和 `llama-server`。llama.cpp 的 CUDA 构建尚未验证。

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --llamacpp q8_0,q4_k,iq4_nl --llamacpp-gpu-layers 99 \
  --llamacpp-serve '{"concurrency": 4, "requests": 16}' --job-dir runs/gguf-demo
```

对 F16 和每个请求的类型，作业依次运行 `llama-perplexity`（上下文取 `evaluation.seq_len`，`evaluation.max_windows` 个 chunk，只对每个 chunk 的后半段计分）、`llama-bench`（prefill pp 与 decode tg 的 tok/s），然后启动 `llama-server`，发一个聊天请求（“What is the capital of France?” 必须回答 Paris），再做闭环流式压测。

Apple M5（16 GB）上的记录，Qwen2.5-0.5B-Instruct，4 线程，40 个 512 的 chunk（`docs/PROGRESS.md`）：

| 类型 | PPL Metal | PPL CPU | decode tok/s Metal / CPU | 4 并发 serve tok/s | TTFT p50 |
|---|---|---|---|---|---|
| f16 | 13.889 | 13.889 | 121 / 82 | 327 | 118 ms |
| q8_0 | 13.904 | 13.923 | 170 / 122 | 412 | 117 ms |
| q4_k | 14.718 | 14.742 | 194 / 137 | 401 | 123 ms |
| iq4_nl | 15.082 | 15.073 | 200 / 163 | 423 | 116 ms |

使用 `-ngl 99` 时日志显示 `offloaded 25/25 layers to GPU`；`-ngl 0` 时所有层留在 CPU。在 Linux 开发服务器（CPU，48 线程）上，同一模型 Q4_K 的 PPL 为 14.75，4 并发 serve 吞吐 113 tok/s。Web 作业页展示这张表（见 [界面导览](web_ui.md) 中的截图）。

## vLLM

vLLM 需要 CUDA GPU，通常装在自己的环境里：MQDEsys 以子进程方式启动 `vllm serve`（`MQDE_VLLM_BIN` 或 `PATH` 中的 `vllm`）并通过 HTTP 交互，不导入 vLLM 的任何代码。

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq --export awq,compressed-tensors \
  --vllm '{"gpu_memory_utilization": 0.3}' --job-dir runs/vllm-demo
```

对每个导出的 checkpoint，`deploy_vllm` 阶段启动服务、发冒烟请求、运行压测、从启动日志读取引擎信息（KV Cache 容量、权重显存、量化 kernel），并计算**部署后 PPL**（served PPL）。

RTX 4090、vLLM 0.30.0、Qwen2.5-0.5B-Instruct 上的记录，W4A16 / FP8 均由 MQDEsys 导出（`docs/PROGRESS.md`；前两行对应作业 20261008-101629-62da92）：

| Checkpoint | vLLM 选用的 kernel | 真实权重 PPL | 部署后 PPL | 差 | TTFT p50 | TPOT p50 | tok/s（4 并发） |
|---|---|---|---|---|---|---|---|
| W4A16 AWQ | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 52 ms | 16.2 ms | 240 |
| W4A16 compressed-tensors | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 55 ms | 16.9 ms | 225 |
| FP8 compressed-tensors | CutlassFP8ScaledMMLinearKernel | 18.333 | 18.382 | +0.27% | 65 ms | 19.6 ms | 199 |

## 基础组件（Python）

这些组件也可以单独使用，都在 `python/mqde/` 下。

### `eval/loadtest.py`：TTFT / TPOT / 吞吐

`run_load(port, LoadSpec(...))` 对 `POST /v1/completions`（`api="openai"`：vLLM、SGLang、llama-server）或 llama-server 的原生 `/completion`（`api="llama"`）发流式请求。两种模式：**闭环**（`concurrency` 个 worker 连续发请求）和**开环**（按 `rate` req/s 的泊松到达，同时最多 `max_inflight` 个；`schedule_lag_ms` 告诉你瓶颈是发压端还是服务端）。输入是长度恰为 `prompt_len` 的 token id 列表，且每个请求不同，因此 prefix caching 无法掩盖实际计算量；生成时忽略 EOS，恰好产生 `n_predict` 个 token。

定义：TTFT = 首个 token 到达时刻减去请求发出时刻；TPOT = （最后一个与第一个 token 到达时刻之差）/（token 数 - 1）；ITL = 相邻 token 到达的间隔；E2E = 最后一个 token 到达时刻减去请求发出时刻；吞吐 = 成功请求的输出 token 数 / 墙钟时间。报告 p50、p90 和 p99。对 TTFT 100 ms、TPOT 20 ms 的 mock 服务器，测量误差小于 2 ms。

### `eval/served_ppl.py`：部署后模型的精度

`served_perplexity(port, ids, seq_len, max_windows)` 用引擎自己的 `prompt_logprobs` 计算 PPL，窗口与伪量化、真实权重 PPL 相同，因此差值正好反映推理栈（kernel、dtype）带来的变化。它要求引擎返回 prompt 的对数概率：vLLM 可以，llama-server 不行。

### `eval/resources.py`：资源采样

`ResourceSampler` 用一个常驻的 `nvidia-smi` 进程每 100 ms 采样 GPU 显存、功耗和利用率，并用 `ps` 采样主机内存（服务进程树）。GPU 显存是整卡数值，所以摘要里还会给出基线和高于基线的峰值。没有 `nvidia-smi` 时 GPU 部分被禁用。

### `eval/boundary.py`：能服务的最大并发或上下文

对并发（固定上下文）或上下文（固定并发）做二分；每个探测点都启动全新的后端，所以某一点的崩溃不会污染下一点；只有服务启动成功、所有请求成功且进程仍然存活时才算通过。

```bash
PYTHONPATH=python python -m mqde.eval.boundary vllm <CHECKPOINT_DIR> --gpu 0 --max-len 1024 --lo 1 --hi 256
```

选项：`--axis concurrency|context`、`--concurrency`（上下文轴使用）、`--lo`、`--hi`、`--step`、`--util`、`--workdir`。它在 mock 后端上测试过；**还没有在真实 vLLM 上运行过**，每个探测点约需 30 s 启动时间，结果也尚未与预估器并排展示在报告里。请把它当作工具，而不是已验证的结果。

## 预估与实测

对 Qwen2.5-0.5B FP8，vLLM 报告权重显存 0.59 GiB，与预估一致。最大并发原先不一致（1024 上下文、0.3 显存利用率下预估 74，vLLM 为 509），因为预估器按序列计激活；使用 `vllm` 后端画像后预估为 488，是 vLLM 数值的 0.96（见 [预估指南](estimate.md)）。把实测反馈给预估器（S15 阶段）为计划中。
