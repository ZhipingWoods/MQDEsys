# Deployment and benchmarking

**Status: partial.** llama.cpp (GGUF, CPU and Metal) and vLLM run end to end in Python. The C++ backend interface, SGLang, TensorRT-LLM, lm-eval/evalscope accuracy suites and the report page for boundary search are not done.

After export, a job can serve each checkpoint with a real engine, check it answers, load it, and score it. A missing engine binary makes the stage `skipped` (reason in the report); a failed start is `failed` with a diagnosis recognised from the engine's log (out of memory, unsupported quantization config, KV cache too small, port in use, missing file ...).

The servers always bind to `127.0.0.1` because they have no authentication.

## llama.cpp (GGUF)

Build llama.cpp out of tree (CPU, or Metal on Apple silicon) and point MQDEsys at the binaries (see [Installation](../get_started/installation.md)). For Metal, configure with `-DGGML_METAL=ON`; the dev-server CPU build uses `-DGGML_CUDA=OFF`. `-DLLAMA_CURL=OFF` avoids a dependency. The needed targets are `llama-perplexity`, `llama-quantize`, `llama-bench` and `llama-server`. The CUDA build of llama.cpp has not been exercised.

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --llamacpp q8_0,q4_k,iq4_nl --llamacpp-gpu-layers 99 \
  --llamacpp-serve '{"concurrency": 4, "requests": 16}' --job-dir runs/gguf-demo
```

For F16 and every requested type the job runs `llama-perplexity` (context = `evaluation.seq_len`, `evaluation.max_windows` chunks, scoring the second half of each chunk), `llama-bench` (prefill pp and decode tg tok/s), then starts `llama-server`, sends a chat request ("What is the capital of France?" must answer Paris) and runs a closed-loop streaming load.

Recorded on an Apple M5 (16 GB), Qwen2.5-0.5B-Instruct, 4 threads, 40 chunks of 512 (`docs/PROGRESS.md`):

| Type | PPL Metal | PPL CPU | decode tok/s Metal / CPU | serve tok/s at 4 concurrent | TTFT p50 |
|---|---|---|---|---|---|
| f16 | 13.889 | 13.889 | 121 / 82 | 327 | 118 ms |
| q8_0 | 13.904 | 13.923 | 170 / 122 | 412 | 117 ms |
| q4_k | 14.718 | 14.742 | 194 / 137 | 401 | 123 ms |
| iq4_nl | 15.082 | 15.073 | 200 / 163 | 423 | 116 ms |

With `-ngl 99` the log shows `offloaded 25/25 layers to GPU`; with `-ngl 0` all layers stay on the CPU. On the Linux dev server (CPU, 48 threads) the same model gave Q4_K PPL 14.75 and 113 tok/s served at 4 concurrent. The web job page shows this table (see the screenshot in the [UI tour](web_ui.md)).

## vLLM

vLLM needs a CUDA GPU and normally lives in its own environment: MQDEsys launches `vllm serve` as a subprocess (`MQDE_VLLM_BIN` or `vllm` on `PATH`) and talks HTTP; nothing from vLLM is imported.

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq --export awq,compressed-tensors \
  --vllm '{"gpu_memory_utilization": 0.3}' --job-dir runs/vllm-demo
```

For each exported checkpoint the `deploy_vllm` stage starts the server, sends a smoke request, runs a load test, reads engine facts from the startup log (KV cache capacity, weight memory, quantization kernel), and computes **served PPL**.

Recorded on an RTX 4090, vLLM 0.30.0, Qwen2.5-0.5B-Instruct, W4A16 / FP8 exported by MQDEsys (`docs/PROGRESS.md`, job 20261008-101629-62da92 for the first two rows):

| Checkpoint | Kernel chosen by vLLM | real PPL | served PPL | delta | TTFT p50 | TPOT p50 | tok/s (4 conc.) |
|---|---|---|---|---|---|---|---|
| W4A16 AWQ | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 52 ms | 16.2 ms | 240 |
| W4A16 compressed-tensors | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 55 ms | 16.9 ms | 225 |
| FP8 compressed-tensors | CutlassFP8ScaledMMLinearKernel | 18.333 | 18.382 | +0.27% | 65 ms | 19.6 ms | 199 |

## Building blocks (Python)

You can use the pieces directly. All live under `python/mqde/`.

### `eval/loadtest.py`: TTFT / TPOT / throughput

`run_load(port, LoadSpec(...))` streams requests against `POST /v1/completions` (`api="openai"`: vLLM, SGLang, llama-server) or llama-server's native `/completion` (`api="llama"`). Two modes: **closed loop** (`concurrency` workers back to back) and **open loop** (Poisson arrivals at `rate` req/s, at most `max_inflight` at once; `schedule_lag_ms` tells you if the generator, not the server, was the bottleneck). Prompts are token-id lists of exactly `prompt_len` tokens that differ per request, so prefix caching cannot hide the work; generation ignores EOS and produces exactly `n_predict` tokens.

Definitions: TTFT = first token arrival minus request sent; TPOT = (last minus first arrival) / (tokens - 1); ITL = gaps between token arrivals; E2E = last arrival minus request sent; throughput = output tokens of successful requests / wall time. p50, p90 and p99 are reported. Against a mock server with TTFT 100 ms and TPOT 20 ms the measured error was under 2 ms.

### `eval/served_ppl.py`: accuracy of the deployed model

`served_perplexity(port, ids, seq_len, max_windows)` computes PPL from the engine's own `prompt_logprobs`, with the same non-overlapping windows as the fake and real PPL, so differences isolate what the serving stack (kernels, dtype) adds. It needs an engine that returns prompt log-probabilities: vLLM does, llama-server does not.

### `eval/resources.py`: memory sampling

`ResourceSampler` samples GPU memory, power and utilization every 100 ms from one long-running `nvidia-smi` process, and host memory (the server's process tree) with `ps`. GPU memory is whole-device, so the summary also reports the baseline and the peak above it. Without `nvidia-smi` the GPU part is disabled.

### `eval/boundary.py`: largest concurrency or context that is served

Bisection over concurrency (fixed context) or context (fixed concurrency); every probe starts a fresh backend so a crash at one point cannot poison the next, and passes only if the server started, every request succeeded and the process is alive.

```bash
PYTHONPATH=python python -m mqde.eval.boundary vllm <CHECKPOINT_DIR> --gpu 0 --max-len 1024 --lo 1 --hi 256
```

Options: `--axis concurrency|context`, `--concurrency` (for the context axis), `--lo`, `--hi`, `--step`, `--util`, `--workdir`. It has been tested against the mock backend; **it has not yet been run against a real vLLM**, each probe costs about 30 s of startup, and the result is not yet shown next to the estimator in a report. Treat it as a tool, not a verified result.

## Estimator versus measurement

vLLM reported a weight memory of 0.59 GiB for Qwen2.5-0.5B FP8, equal to the estimate. Maximum concurrency is where they disagree (estimate 74, vLLM 509 at 1024 context, 0.3 memory utilization) for the activation-model reason explained in [Estimation](estimate.md). Feeding measurements back into the estimator (stage S15) is planned.
