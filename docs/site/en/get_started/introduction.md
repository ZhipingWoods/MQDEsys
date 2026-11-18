# Introduction

## Purpose

MQDEsys (Model Quantization & Deployment Evaluation System) is an integrated system for large-model quantization and deployment evaluation. Given a model, a quantization recipe, evaluation datasets and target hardware, the system performs resource estimation, quantization, dataset evaluation, real checkpoint export, inference serving and load testing, and records the complete process and metrics in a single job.

The objective is not merely to produce a smaller weight file. The system answers the deployment questions that matter: whether the quantized model retains sufficient accuracy, whether it fits on the target device, how it performs under inference, and where the accuracy loss originates.

## Core workflow

1. **Resource estimation**: Without loading weights, the estimator reads `config.json` and safetensors headers to compute weight size, KV cache or linear-attention state, activations, runtime overhead, single- and multi-concurrency peak memory, a latency floor, and a recipe-by-hardware deployability matrix.
2. **Quantization and evaluation**: Baseline evaluation, fake quantization with per-layer error tracing, and perplexity or dataset metrics at multiple evaluation positions.
3. **Export and verification**: Produce GPTQ, AWQ, compressed-tensors or GGUF checkpoints and verify that the exported integer codes and zero points match those used during fake-quant evaluation.
4. **Deployment and benchmarking**: Serve the checkpoint with llama.cpp, vLLM or SGLang and collect TTFT, TPOT, throughput and served accuracy.
5. **Visualization**: Inspect stage pipelines, live logs, per-layer error heat maps, worst layers, sensitivity analysis and recipe comparisons in the web workspace.

## Supported scope

| Dimension | Support |
|---|---|
| Model domains | LLM, VLM, VLA, world models (WM) and event world models (EWM) |
| Quantization recipes | BF16, W8A16, FP8, W4A16, W4A8, W2A16, W2A8 |
| Quantization algorithms | RTN, GPTQ, HQQ, AWQ, SmoothQuant and combinations |
| Export formats | GPTQ, AWQ, compressed-tensors, GGUF |
| Serving backends | llama.cpp, vLLM, SGLang |
| Evaluation | Offline PPL, QA and MCQ, plus the EvalScope dataset catalogue |
| Hardware profiles | NVIDIA (Jetson Orin, Jetson Thor, DGX Spark, RTX, A100, H100), AMD, Intel, Ascend and MetaX |

Model architectures are integrated through declarative descriptions in `data/model_families/`. Any architecture that Hugging Face transformers can load as a causal language model runs through the generic decoder adapter; specialized structures use dedicated adapters.

## Components

- **C++ control plane** (`cpp/`): model parsing, the normative quantization arithmetic, resource estimation, fit scoring, hardware probing, REST/SSE server and CLI.
- **Python executor** (`python/mqde/`): PyTorch numerics, quantization algorithms, model adapters, evaluation, exporters, backend adapters and job pipelines. The PyTorch numerics are bit-exact with the C++ core, enforced by golden tests.
- **Web workspace** (`web/`): React, TypeScript and ECharts, served by the C++ server.
- **Data layer** (`data/`): recipes, hardware profiles, backend profiles, model-family templates and scoring constants as YAML, extensible without code changes.

## Job modes

The system provides three decoupled job modes:

- **Full pipeline**: estimation, quantization, evaluation, export, deployment and load testing in sequence.
- **Quantization job**: quantization and export only, producing reusable checkpoints.
- **Evaluation job**: evaluates an existing quantized model or checkpoint against selected datasets without re-quantizing.

All modes share the same job records, event stream, report structure and web interface.

See [Architecture](../advanced/architecture.md) for details.
