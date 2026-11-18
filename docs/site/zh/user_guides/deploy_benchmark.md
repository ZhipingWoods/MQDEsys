# 部署与压测

导出完成后，作业可使用真实推理引擎部署每个 checkpoint，验证服务可用性，施加负载并评估部署后精度。未找到引擎二进制时，该阶段标记为 `skipped` 并在报告中记录原因；启动失败时标记为 `failed`，并附带从引擎日志中识别出的诊断信息（显存不足、不支持的量化配置、KV Cache 过小、端口占用、文件缺失等）。

所有推理服务均绑定 `127.0.0.1`。

## llama.cpp（GGUF）

在源码树之外构建 llama.cpp（CPU，或 Apple 芯片上的 Metal），并确保 MQDEsys 能找到二进制（见[安装](../get_started/installation.md)）。Metal 构建使用 `-DGGML_METAL=ON`；CPU 构建使用 `-DGGML_CUDA=OFF`；`-DLLAMA_CURL=OFF` 可减少一个依赖。需要构建的目标为 `llama-perplexity`、`llama-quantize`、`llama-bench` 与 `llama-server`。

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --llamacpp q8_0,q4_k,iq4_nl --llamacpp-gpu-layers 99 \
  --llamacpp-serve '{"concurrency": 4, "requests": 16}' --job-dir runs/gguf-demo
```

作业对 F16 及每种请求的 GGUF 类型依次执行 `llama-perplexity`（上下文取 `evaluation.seq_len`，共 `evaluation.max_windows` 个 chunk，仅对每个 chunk 的后半段计分）、`llama-bench`（prefill 与 decode 的 tok/s），随后启动 `llama-server`，发送冒烟请求，并执行闭环流式压测。

Apple M5（16 GB）、Qwen2.5-0.5B-Instruct、4 线程、40 个 512-token chunk 的实测结果：

| 类型 | PPL Metal | PPL CPU | decode tok/s Metal / CPU | 4 并发 serve tok/s | TTFT p50 |
|---|---|---|---|---|---|
| f16 | 13.889 | 13.889 | 121 / 82 | 327 | 118 ms |
| q8_0 | 13.904 | 13.923 | 170 / 122 | 412 | 117 ms |
| q4_k | 14.718 | 14.742 | 194 / 137 | 401 | 123 ms |
| iq4_nl | 15.082 | 15.073 | 200 / 163 | 423 | 116 ms |

使用 `-ngl 99` 时日志显示 `offloaded 25/25 layers to GPU`；`-ngl 0` 时所有层保留在 CPU。Linux CPU 服务器（48 线程）上，同一模型 Q4_K 的 PPL 为 14.75，4 并发 serve 吞吐为 113 tok/s。Web 作业页会展示该表（见[界面导览](web_ui.md)）。

## vLLM

vLLM 需要 CUDA GPU，通常安装于独立环境。MQDEsys 以子进程方式启动 `vllm serve`（`MQDE_VLLM_BIN` 或 `PATH` 中的 `vllm`），通过 HTTP 交互，不导入 vLLM 代码。

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq --export awq,compressed-tensors \
  --vllm '{"gpu_memory_utilization": 0.3}' --job-dir runs/vllm-demo
```

对每个导出的 checkpoint，部署阶段会启动服务、发送冒烟请求、执行压测、从启动日志读取引擎信息（KV Cache 容量、权重显存、量化 kernel），并计算部署后 PPL。

RTX 4090、vLLM 0.30.0、Qwen2.5-0.5B-Instruct 的实测结果（W4A16 与 FP8 均由 MQDEsys 导出）：

| Checkpoint | vLLM kernel | 真实权重 PPL | 部署后 PPL | 差值 | TTFT p50 | TPOT p50 | tok/s（4 并发） |
|---|---|---|---|---|---|---|---|
| W4A16 AWQ | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 52 ms | 16.2 ms | 240 |
| W4A16 compressed-tensors | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 55 ms | 16.9 ms | 225 |
| FP8 compressed-tensors | CutlassFP8ScaledMMLinearKernel | 18.333 | 18.382 | +0.27% | 65 ms | 19.6 ms | 199 |

## SGLang

SGLang 以独立解释器运行，通过 `MQDE_SGLANG_PYTHON` 指定环境，或使用 `scripts/setup_backend.py` 创建受管环境。部署流程与 vLLM 一致：启动 `sglang.launch_server`、发送冒烟请求、执行压测并读取启动日志中的权重显存与 KV Cache 容量。

## 基础组件

以下组件位于 `python/mqde/`，可独立使用。

### `eval/loadtest.py`：TTFT、TPOT 与吞吐

`run_load(port, LoadSpec(...))` 向 `POST /v1/completions`（`api="openai"`，适用于 vLLM、SGLang、llama-server）或 llama-server 原生 `/completion`（`api="llama"`）发送流式请求。支持两种模式：

- **闭环**：`concurrency` 个 worker 连续发送请求；
- **开环**：按 `rate` req/s 的泊松到达发送，同时最多 `max_inflight` 个请求；`schedule_lag_ms` 用于判断瓶颈位于发压端还是服务端。

输入为长度恰为 `prompt_len` 的 token id 列表，每个请求内容不同，因此 prefix caching 不会掩盖实际计算量；生成时忽略 EOS，恰好产生 `n_predict` 个 token。

指标定义：TTFT 为首个 token 到达时刻减去请求发出时刻；TPOT 为（最后一个与第一个 token 到达时刻之差）/（token 数 - 1）；ITL 为相邻 token 到达间隔；E2E 为最后一个 token 到达时刻减去请求发出时刻；吞吐为成功请求的输出 token 数除以墙钟时间。报告输出 p50、p90 与 p99。对 TTFT 100 ms、TPOT 20 ms 的 mock 服务，测量误差小于 2 ms。

### `eval/lm_eval_adapter.py`：任务精度评测

部署后模型可通过 lm-evaluation-harness 评测任务精度。MQDEsys 不导入 lm-eval，而是调用已安装的 `lm-eval` 命令（`MQDE_LM_EVAL_BIN`，否则 `PATH` 中的 `lm-eval`），通过其 `local-completions` 模型连接本阶段启动的 vLLM 服务，并读取其输出的 JSON。建议将 lm-eval 安装在独立环境中（分词器需要 `transformers`，不需要 torch）。数据集来自 Hugging Face 或镜像（`HF_ENDPOINT`）。

```bash
PYTHONPATH=python MQDE_LM_EVAL_BIN=/path/to/envs/lmeval/bin/lm-eval python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --export awq --device cuda \
  --vllm '{"gpu_memory_utilization": 0.4, "lm_eval": {"tasks": ["gsm8k"], "limit": 50, "num_fewshot": 5}}'
```

`lm_eval` 选项包括：`tasks`、`limit`（默认 50）、`num_fewshot`、`concurrency`、`seed`、`max_gen_toks`、`max_model_len`（默认 4096）、`timeout` 与 `baseline`（默认 true）。启用 `baseline` 时，相同评测会在未量化模型上以相同配置执行，并作为报告精度表的第一行。未找到 `lm-eval` 时该阶段标记为 `skipped`；运行失败时标记为 `failed` 并记录服务端返回的首条错误，作业仍正常结束。

RTX 4090、lm-eval 0.4.13、Qwen2.5-0.5B-Instruct、gsm8k、5-shot、50 个样本的实测结果：

| 模型 | strict-match | flexible-extract |
|---|---:|---:|
| 未量化（FP16，vLLM） | 0.20 ± 0.057 | 0.28 ± 0.064 |
| W4A16 GPTQ，导出为 AWQ | 0.02 ± 0.020 | 0.06 ± 0.034 |

同一 checkpoint 的 PPL 从 18.10 变为 20.66。50 个样本的标准误较大，该结果用于说明小模型上的明显退化，不代表精确数值。以相同参数直接运行官方命令行，所得分数与适配器一致。

### `eval/served_ppl.py`：部署后精度

`served_perplexity(port, ids, seq_len, max_windows)` 使用引擎返回的 prompt log-probability 计算 PPL，窗口与伪量化、真实权重 PPL 相同，因此差值反映推理栈（kernel、dtype）带来的变化。该功能要求引擎返回 prompt 对数概率：vLLM 支持，llama-server 不支持。

### `eval/resources.py`：资源采样

`ResourceSampler` 使用常驻的 `nvidia-smi` 进程每 100 ms 采样 GPU 显存、功耗与利用率，并通过 `ps` 采样主机内存（服务进程树）。GPU 显存为整卡数值，摘要中同时给出基线与高于基线的峰值。未安装 `nvidia-smi` 时禁用 GPU 采样。

### `eval/boundary.py`：最大并发或上下文边界

对并发（固定上下文）或上下文（固定并发）执行二分探测。每个探测点启动全新后端，因此单点崩溃不会影响后续探测；仅当服务启动成功、全部请求成功且进程仍存活时，该点判定为通过。

```bash
PYTHONPATH=python python -m mqde.eval.boundary vllm <CHECKPOINT_DIR> --gpu 0 --max-len 1024 --lo 1 --hi 256
```

选项：`--axis concurrency|context`、`--concurrency`（上下文轴使用）、`--lo`、`--hi`、`--step`、`--util`、`--workdir`。

## 预估与实测对照

以 Qwen2.5-0.5B FP8 为例，vLLM 报告的权重显存为 0.59 GiB，与预估一致。最大并发在 1024 上下文、0.3 显存利用率下，按序列计激活时预估为 74，vLLM 报告 509；改用 `vllm` 后端画像按批计激活后，预估为 488，为 vLLM 数值的 0.96（见[资源预估](estimate.md)）。
