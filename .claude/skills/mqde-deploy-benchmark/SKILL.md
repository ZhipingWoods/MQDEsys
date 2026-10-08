---
name: mqde-deploy-benchmark
description: >
  Export a quantized model and deploy it for measurement in MQDEsys: GPTQ / AWQ / compressed-tensors checkpoints
  served by vLLM, GGUF files served by llama-server (CPU or Metal), then load-test it (TTFT / TPOT / throughput),
  score the deployed model's PPL, and find the maximum concurrency or context it serves. Use when the user asks to
  deploy, serve, benchmark, load-test, compare served accuracy with the fake-quant result, or find an OOM boundary
  with this repository. Do NOT use for choosing a recipe or reading error analysis (use mqde-quantize-eval).
argument-hint: model_path [export formats] [vllm|llamacpp]
user-invocable: true
allowed-tools: Bash(python:*) Bash(python3:*) Bash(build/*:*) Bash(ls:*) Bash(cat:*) Bash(nvidia-smi:*) Read Grep Glob
---

# Deploy and benchmark

All Python commands run from `<REPO>` with `PYTHONPATH=python`. A GPU that other people use is shared: check
`nvidia-smi` first, pin one device with `CUDA_VISIBLE_DEVICES`, and never start a server on a card you did not pick.
Servers listen on 127.0.0.1 only and have no authentication.

## One job, everything chained

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python MQDE_VLLM_BIN=<VLLM_ENV>/bin/vllm \
python -m mqde.pipeline.llm_job --model <MODEL_DIR> --recipes w4a16 --algorithm gptq --device cuda \
  --export awq,compressed-tensors --vllm --job-dir runs/deploy-demo
```

Per exported checkpoint this adds a `deploy_vllm` stage: start `vllm serve`, smoke request, closed-loop load test,
engine facts from the log, and PPL computed from vLLM's prompt log-probabilities on the same windows as the
fake-quant / real-weight PPL. Result: `report.json → recipes.<R>.export.<fmt>.vllm`
(`served_ppl`, `served_vs_real_ppl_pct`, `engine.quant_kernel`, `load.ttft_ms`, ...).
GGUF: add `--llamacpp q8_0,q4_k,iq4_nl` (`--llamacpp-gpu-layers 99` for Metal or CUDA builds).

Quantizing on CUDA and exporting works; the quantizer moves its inputs to the model's device.

## Binaries (none are in the repository)

| Backend | Where it is found | Build / install |
|---|---|---|
| vLLM | `MQDE_VLLM_BIN`, `vllm` on PATH, `build/vllm/bin/vllm` | install vLLM in its **own** environment (it pins torch and transformers) |
| llama-server | `MQDE_LLAMA_SERVER_BIN`, `build/llamacpp-server/bin/llama-server`, `build/llamacpp-metal/bin/llama-server` | upstream llama.cpp (the omni fork has no server target): `cmake -S <llama.cpp> -B build/llamacpp-server -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF` then `--target llama-server`; add `-DGGML_METAL=ON` on Apple Silicon |
| llama-perplexity, llama-bench | `MQDE_LLAMACPP_BIN`, `build/llamacpp/bin`, `build/llamacpp-metal/bin` | same build, targets `llama-perplexity llama-bench` |
| GGUF converter | `MQDE_LLAMACPP_SRC` (needs `convert_hf_to_gguf.py` and `gguf-py`) | a llama.cpp checkout |

If a binary is missing the stage is recorded as `skipped` with the reason; it does not fail the job.

## Pieces you can call directly

```python
from mqde.backend.vllm_server import VllmServer          # also LlamaServer in mqde.backend.llamacpp_server
from mqde.backend.base import openai_smoke
from mqde.eval.loadtest import LoadSpec, run_load
from mqde.eval.served_ppl import served_perplexity

with VllmServer(ckpt, "runs/x/vllm.log", gpu=3, max_model_len=1024, gpu_memory_utilization=0.3,
                enforce_eager=True, max_num_seqs=8) as srv:
    openai_smoke(srv.port)
    r = run_load(srv.port, LoadSpec(concurrency=4, requests=16, prompt_len=256, n_predict=64))
    print(r["ttft_ms"]["p50"], r["tpot_ms"]["p50"], r["output_tokens_per_s"], srv.engine_info())
```

Load test options (`LoadSpec`): `mode="closed"` with `concurrency`, or `mode="open"` with a Poisson `rate` (req/s) and
`max_inflight`; `api="openai"` (vLLM, SGLang, llama-server `/v1/completions`) or `"llama"` (native). Prompts are
exact-length token ids with unique prefixes. Definitions: TTFT = first token minus request sent; TPOT = (last − first
token) / (tokens − 1); ITL = gaps between tokens. In open-loop mode check `schedule_lag_ms`: a large p99 means the
generator was the bottleneck, not the server.

Boundary search (restarts the backend for every probe, ~30 s each for vLLM):

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=python python -m mqde.eval.boundary vllm <CKPT> --gpu 3 --axis concurrency --max-len 1024 --lo 1 --hi 256
```

Each probe records resource peaks (`eval.resources`: device memory, power, utilisation, host RSS at 100 ms) and the
failure reason from the backend's own log.

## Reading and trusting the numbers

- `served_vs_real_ppl_pct` compares the deployed model with the checkpoint's own dequantized weights. Measured so far
  (Qwen2.5-0.5B, vLLM 0.30): AWQ and compressed-tensors W4A16 +0.01%, FP8 compressed-tensors +0.27%.
- Asymmetric GPTQ checkpoints are rejected by vLLM 0.30's config validator (`Unsupported quantization config:
  bits=4, sym=False`): export AWQ or compressed-tensors instead. This is a vLLM restriction, not an export defect.
- vLLM reports its own KV capacity (`engine.kv_cache_tokens`, `max_concurrency_at_len`). The estimator's
  `max_concurrency` for vLLM is a conservative lower bound (docs/PROGRESS.md D-9); do not present it as measured.
- A load-test number is only meaningful with the exact settings that produced it: always quote concurrency,
  prompt/output lengths, device and whether `enforce_eager` was on.
- llama.cpp `llama-server` has no `prompt_logprobs`; served PPL is vLLM-only. llama.cpp PPL comes from
  `llama-perplexity` with its own protocol and is not comparable with the job's PPL.

See `references/failures.md` for startup errors and what to do about each.
