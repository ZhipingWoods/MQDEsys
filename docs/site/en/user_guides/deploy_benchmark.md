# Deployment and benchmarking

After export, a job can deploy each checkpoint with a real inference engine, verify that it responds, apply load and measure served accuracy. When an engine binary is missing, the stage is marked `skipped` and the reason is recorded in the report. A failed start is marked `failed` with diagnostics identified from the engine log (out of memory, unsupported quantization configuration, KV cache too small, port in use, missing files, and so on).

All inference servers bind to `127.0.0.1`.

## llama.cpp (GGUF)

Build llama.cpp outside the source tree (CPU, or Metal on Apple silicon) and make the binaries discoverable by MQDEsys (see [Installation](../get_started/installation.md)). Metal builds use `-DGGML_METAL=ON`; CPU builds use `-DGGML_CUDA=OFF`; `-DLLAMA_CURL=OFF` removes one dependency. Required targets are `llama-perplexity`, `llama-quantize`, `llama-bench` and `llama-server`.

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --llamacpp q8_0,q4_k,iq4_nl --llamacpp-gpu-layers 99 \
  --llamacpp-serve '{"concurrency": 4, "requests": 16}' --job-dir runs/gguf-demo
```

For F16 and each requested GGUF type, the job runs `llama-perplexity` (context from `evaluation.seq_len`, `evaluation.max_windows` chunks, scoring only the second half of each chunk), `llama-bench` (prefill and decode tok/s), then starts `llama-server`, sends a smoke request and runs a closed-loop streaming load test.

Measurements on an Apple M5 (16 GB) with Qwen2.5-0.5B-Instruct, 4 threads and 40 chunks of 512:

| Type | PPL Metal | PPL CPU | decode tok/s Metal / CPU | serve tok/s at 4 concurrency | TTFT p50 |
|---|---|---|---|---|---|
| f16 | 13.889 | 13.889 | 121 / 82 | 327 | 118 ms |
| q8_0 | 13.904 | 13.923 | 170 / 122 | 412 | 117 ms |
| q4_k | 14.718 | 14.742 | 194 / 137 | 401 | 123 ms |
| iq4_nl | 15.082 | 15.073 | 200 / 163 | 423 | 116 ms |

With `-ngl 99` the log shows `offloaded 25/25 layers to GPU`; with `-ngl 0` all layers remain on the CPU. On a Linux CPU server (48 threads), the same model at Q4_K gives PPL 14.75 and 113 tok/s at 4-way concurrency. The job page presents this table (see the [UI guide](web_ui.md)).

## vLLM

vLLM requires a CUDA GPU and is normally installed in its own environment. MQDEsys starts `vllm serve` as a subprocess (`MQDE_VLLM_BIN` or `vllm` on `PATH`), communicates over HTTP, and does not import vLLM code.

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq --export awq,compressed-tensors \
  --vllm '{"gpu_memory_utilization": 0.3}' --job-dir runs/vllm-demo
```

For each exported checkpoint, the deployment stage starts the server, sends a smoke request, runs the load test, reads engine information from the startup log (KV cache capacity, weight memory, quantization kernel), and computes served PPL.

Measurements on an RTX 4090 with vLLM 0.30.0 and Qwen2.5-0.5B-Instruct (W4A16 and FP8 exported by MQDEsys):

| Checkpoint | vLLM kernel | Real-weight PPL | Served PPL | Delta | TTFT p50 | TPOT p50 | tok/s (4 concurrent) |
|---|---|---|---|---|---|---|---|
| W4A16 AWQ | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 52 ms | 16.2 ms | 240 |
| W4A16 compressed-tensors | MarlinLinearKernel | 20.656 | 20.658 | +0.01% | 55 ms | 16.9 ms | 225 |
| FP8 compressed-tensors | CutlassFP8ScaledMMLinearKernel | 18.333 | 18.382 | +0.27% | 65 ms | 19.6 ms | 199 |

## SGLang

SGLang runs under a separate interpreter, specified with `MQDE_SGLANG_PYTHON` or created by `scripts/setup_backend.py` as a managed environment. The deployment flow matches vLLM: start `sglang.launch_server`, send a smoke request, run the load test and read weight memory and KV cache capacity from the startup log.

## Standalone components

The following components live under `python/mqde/` and can be used independently.

### `eval/loadtest.py`: TTFT, TPOT and throughput

`run_load(port, LoadSpec(...))` sends streaming requests to `POST /v1/completions` (`api="openai"`, applicable to vLLM, SGLang and llama-server) or llama-server's native `/completion` (`api="llama"`). Two modes are supported:

- **Closed loop**: `concurrency` workers send requests continuously.
- **Open loop**: Poisson arrivals at `rate` req/s with at most `max_inflight` requests in flight; `schedule_lag_ms` indicates whether the bottleneck is the load generator or the server.

Inputs are token-id lists of exactly `prompt_len` tokens, different for every request, so prefix caching cannot hide the real computation. Generation ignores EOS and produces exactly `n_predict` tokens.

Definitions: TTFT is the time from request dispatch to the first token; TPOT is (last token time minus first token time) / (token count - 1); ITL is the interval between consecutive tokens; E2E is the time from request dispatch to the last token; throughput is successful output tokens divided by wall-clock time. The report includes p50, p90 and p99. Against a mock server with TTFT 100 ms and TPOT 20 ms, measurement error is below 2 ms.

### `eval/lm_eval_adapter.py`: task accuracy

Served models can be scored on lm-evaluation-harness tasks. MQDEsys does not import lm-eval; it invokes the installed `lm-eval` command (`MQDE_LM_EVAL_BIN`, otherwise `lm-eval` on `PATH`), connects through lm-eval's `local-completions` model to the vLLM server started by the stage, and reads the resulting JSON. Install lm-eval in a separate environment (the tokenizer requires `transformers`, not torch). Datasets are fetched from Hugging Face or a mirror (`HF_ENDPOINT`).

```bash
PYTHONPATH=python MQDE_LM_EVAL_BIN=/path/to/envs/lmeval/bin/lm-eval python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --export awq --device cuda \
  --vllm '{"gpu_memory_utilization": 0.4, "lm_eval": {"tasks": ["gsm8k"], "limit": 50, "num_fewshot": 5}}'
```

Options under `lm_eval`: `tasks`, `limit` (default 50), `num_fewshot`, `concurrency`, `seed`, `max_gen_toks`, `max_model_len` (default 4096), `timeout` and `baseline` (default true). With `baseline` enabled, the same evaluation is repeated on the unquantized model and appears as the first row of the accuracy table. When `lm-eval` is missing, the stage is marked `skipped`; a failed run is marked `failed` with the first server error recorded, and the job still completes.

Measurements on an RTX 4090 with lm-eval 0.4.13, Qwen2.5-0.5B-Instruct, gsm8k, 5-shot, 50 samples:

| Model | strict-match | flexible-extract |
|---|---:|---:|
| Unquantized (FP16, vLLM) | 0.20 ± 0.057 | 0.28 ± 0.064 |
| W4A16 GPTQ exported as AWQ | 0.02 ± 0.020 | 0.06 ± 0.034 |

PPL for the same checkpoint changes from 18.10 to 20.66. The standard error at 50 samples is large; the result indicates a substantial degradation on a small model rather than a precise value. Running the official CLI directly with the same parameters produces identical scores.

### `eval/served_ppl.py`: served accuracy

`served_perplexity(port, ids, seq_len, max_windows)` computes PPL from the engine's prompt log-probabilities, using the same windows as fake-quantized and real-weight PPL, so the difference reflects the inference stack (kernels, dtype). The feature requires the engine to return prompt log-probabilities: vLLM supports this, llama-server does not.

### `eval/resources.py`: resource sampling

`ResourceSampler` runs a persistent `nvidia-smi` process sampling GPU memory, power and utilization every 100 ms, and samples host memory (server process tree) through `ps`. GPU memory is the whole-device value, so the summary also reports the baseline and the peak above baseline. GPU sampling is disabled when `nvidia-smi` is unavailable.

### `eval/boundary.py`: maximum concurrency or context

Performs a binary search over concurrency (fixed context) or context (fixed concurrency). Each probe starts a fresh backend, so a crash at one point does not affect later probes; a point passes only when the server starts successfully, all requests succeed and the process remains alive.

```bash
PYTHONPATH=python python -m mqde.eval.boundary vllm <CHECKPOINT_DIR> --gpu 0 --max-len 1024 --lo 1 --hi 256
```

Options: `--axis concurrency|context`, `--concurrency` (for the context axis), `--lo`, `--hi`, `--step`, `--util`, `--workdir`.

## Estimate versus measurement

For Qwen2.5-0.5B FP8, vLLM reports 0.59 GiB of weight memory, matching the estimate. At 1024 context and 0.3 memory utilization, maximum concurrency is estimated at 74 when activations are counted per sequence, while vLLM reports 509. With the `vllm` backend profile, which counts activations per batch, the estimate is 488, or 0.96 of vLLM's value (see [Resource estimation](estimate.md)).
