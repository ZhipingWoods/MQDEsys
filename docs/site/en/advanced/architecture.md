# Architecture

MQDEsys separates the control plane from the executor. The C++ control plane handles model parsing, resource estimation, hardware probing, the API server and job scheduling; the Python executor handles PyTorch-based quantization algorithms, evaluation, export and deployment adapters. The two communicate across a process boundary and through an event stream.

## Runtime structure

```text
browser ──HTTP/SSE──▶ mqde serve (C++23, cpp-httplib)
terminal ─ mqde CLI ─▶ ├─ REST API, static pages, event stream
                       ├─ estimator, fit scoring, model parsing, hardware probe
                       └─ job runner: queue, GPU pool, one subprocess per job
                                │ stdout/stderr, events.jsonl
                                ▼
                       python -m mqde.pipeline.{llm_job,vla_job,...} (Python executor)
                                │ controls child processes over 127.0.0.1 HTTP
                                ▼
                       llama-server, vllm serve, sglang.launch_server, llama-perplexity, llama-bench
```

### Process model

Each job runs in a dedicated subprocess. The C++ server allocates a device from the GPU pool (or the single CPU slot), starts the Python pipeline with the job specification, and follows its `events.jsonl`. A crash or out-of-memory condition in a job does not affect the server.

Job state is persisted in the run directory `<runs>/<job-id>/`, which contains `job.yaml`, `events.jsonl`, `report.json`, `worker.log` and `export/`. As a result, jobs started from the command line also appear in the web UI.

### Event stream

The executor writes one JSON event per line. The server replays events from `after_seq` and streams subsequent events over Server-Sent Events until the job completes. Event types include `job_state`, `stage_started`, `stage_finished`, `log`, `metric`, `progress` and `trace`.

### Numerical consistency

`cpp/src/quant/ops.cpp` is the normative implementation of quantization arithmetic. `python/mqde/quant/numerics.py` provides the equivalent PyTorch implementation, and golden tests compare it against `mqde._core` to enforce bit-exactness. GPTQ, AWQ, Marlin and GGUF bit packing live in `cpp/src/quant/pack.cpp` and are exposed to Python through `mqde._core`.

### Data-driven design

Recipes, hardware profiles, backend profiles, the algorithm list, model-family templates and fit constants are defined as YAML under `data/`. The estimator, server, web UI and Python executor read the same files.

## C++ modules

Module dependencies follow `common` ← `modelspec`, `quant` ← `estimator` ← `fit`, `hub` ← `server` ← applications and bindings. Lower layers never include upper layers; `scripts/check_deps.py` enforces this rule.

| Module | Responsibility |
|---|---|
| `common` | `Result<T>` errors, YAML/JSON configuration loading, logging |
| `modelspec` | safetensors header parsing, `ModelSpec`, architecture templates, layer graph |
| `quant` | quantization specifications, fake quantization, error metrics, bit packing |
| `estimator` | memory, KV/state, FLOPs, latency floor, deployability |
| `fit` | model-hardware fit scoring, throughput model, local baseline index |
| `hub` | Hugging Face and ModelScope catalogue, search and download |
| `server` | HTTP endpoints and job runner |
| `probe_c` | C99 hardware probe (dynamically loads NVML; CPU probing) |

## Python packages

| Package | Responsibility |
|---|---|
| `quant/` | numerics, quantization algorithms, blockwise engine, KV quantization, fake-quant modules |
| `models/` | model adapters |
| `data/` | calibration and evaluation data loading |
| `eval/` | PPL, load testing, served PPL, resource sampling, boundary search, dataset evaluation |
| `export/` | checkpoint exporters and authenticity checks |
| `backend/` | llama.cpp, vLLM and SGLang adapters |
| `pipeline/` | job pipelines and event output |
| `trace/` | sensitivity analysis |

## Technology choices

| Area | Implementation |
|---|---|
| C++ standard | C++23 |
| HTTP server | cpp-httplib |
| Event transport | Server-Sent Events |
| Dependency management | conda (Linux), Homebrew (macOS) |
| Job orchestration | Python pipelines |
| State storage | job files and event streams in the run directory |
| Hardware probing | C probe with dynamically loaded NVML |

## Deployment topology

The server binds to `127.0.0.1` by default and serves the web assets from the same process. Model weights and job directories remain on the deployment host and are not sent to external services. Remote access is provided through an SSH tunnel.
