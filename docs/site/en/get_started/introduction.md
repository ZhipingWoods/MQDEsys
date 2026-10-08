# Introduction

## What it is and why

Quantizing a model is easy; knowing whether the result is *good enough, fits, and is fast* on your hardware is the hard part. Typical tools stop at "a smaller checkpoint". MQDEsys closes the loop:

1. **Estimate** (C++, no weights loaded): weights, KV cache or linear-attention state, activations, peak memory for single and multiple concurrent requests, a latency floor, and a recipe x hardware deployability matrix. Only `config.json` and the safetensors headers are read.
2. **Quantize and evaluate** (Python executor scheduled by the C++ server): a baseline, fake-quantization with per-layer error tracing, and perplexity at several evaluation positions.
3. **Export and verify**: write a real GPTQ, AWQ, compressed-tensors or GGUF checkpoint and check that it carries exactly the integers the fake-quant evaluation used.
4. **Deploy and benchmark**: serve the checkpoint with llama.cpp or vLLM, send load, and measure TTFT, TPOT and throughput; score the *served* model's perplexity.
5. **Visualize** (web UI): live stages and logs, per-layer error heat maps, worst layers, sensitivity analysis and recipe comparison.

The first model family wired end to end is the LLM path (any Hugging Face causal LM that dispatches attention through transformers; verified on Qwen2.5-0.5B-Instruct). A vision-language-action path exists for MiniCPM-RobotManip (fake-quant evaluation only).

## Architecture in one picture

- **C++20 core** (`cpp/`): model parsing, numerics (the single normative definition of quantization arithmetic), estimator, fit scoring, hardware probe, REST/SSE server, CLI.
- **Python executor** (`python/mqde/`): PyTorch mirror of the numerics (bit-exact with C++, enforced by golden tests), algorithms, evaluation, exporters, backend adapters, the job pipelines.
- **Web** (`web/`): React + TypeScript + ECharts, served by the C++ server.
- **Data** (`data/`): recipes, hardware profiles, backend profiles, model-family templates and scoring constants, as YAML. Adding one needs no code change.

See [Architecture](../advanced/architecture.md) for details.

## Pipeline stages and their status

Each job is a sequence of stages. The IDs come from the design (`docs/ARCHITECTURE.md` section 5.1); the status column says what exists in the repository today.

| Stage | Name | Status | Notes |
|---|---|---|---|
| S00 | validate | implemented | Job spec and recipe validation (Python pipeline; the server validates the request first). |
| S01 | preflight (estimate) | partial | The estimator and `mqde fit` exist as a CLI, REST endpoints and UI pages; a job does not yet run it as a gating stage. |
| S02 | fetch_model | planned | `mqde download` and the model catalog page download weights, but it is not a job stage (T-0505). |
| S03 | inspect | implemented | Model load and block/linear-layer inventory. |
| S04 | calib_prepare | implemented | Calibration windows from wikitext-2 train (LLM); LeRobot v3 or synthetic frames (VLA). |
| S05 | eval_baseline | implemented | Baseline perplexity ("pretrain" position). |
| S06 | transform | partial | AWQ and SmoothQuant scaling run inside the quantize stage, not as a separate cached stage. |
| S07 | quantize | implemented | Blockwise quantization with per-layer error traces; RTN, GPTQ, AWQ, SmoothQuant and combinations. |
| S08 | eval_fake | implemented | Perplexity with fake-quantized weights, with and without KV cache quantization. |
| S09 | export | partial | GPTQ v1, AWQ GEMM, compressed-tensors, GGUF. SGLang and TensorRT-LLM exports are planned. |
| S10 | real_check | implemented | Reads the checkpoint back, compares codes and zero points, re-measures PPL with the real weights. |
| S11 | deploy | partial | llama.cpp (`llama-server`) and vLLM, implemented in Python. The C++ backend interface is not done. |
| S12 | eval_deployed | partial | PPL from the served model's own prompt log-probabilities, vLLM only. lm-eval and evalscope adapters are planned. |
| S13 | perf_bench | partial | Python load tester (closed and open loop) and OOM boundary search; not yet in the report beside the estimator. |
| S14 | report | partial | `report.json` per job and the UI pages; a standalone HTML report is planned. |
| S15 | feedback | planned | Automatic calibration of the estimator from measurements. |

## What is not there (yet)

Multi-node or multi-GPU quantization, SGLang and TensorRT-LLM deployment, Ascend/ROCm/Metal hardware plugins, authentication, Docker images and pip wheels, and accuracy suites beyond wikitext-2 perplexity are all planned and listed in [Status & roadmap](../status.md). Do not rely on them.
