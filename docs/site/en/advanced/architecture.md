# Architecture

This page summarizes the design in `docs/ARCHITECTURE.md` and marks what exists. Several parts of the original design (a C++ DAG engine, a JSON-RPC worker, SQLite storage, a hardware plugin ABI) are **not implemented**; the current system is a vertical slice that works end to end.

## What runs today

```text
browser ──HTTP/SSE──▶ mqde serve (C++ 23, cpp-httplib)
terminal ─ mqde CLI ─▶   ├─ REST API, static web files, event stream
                         ├─ estimator, fit scoring, model parsing, hardware probe
                         └─ job runner: queue, GPU pool, one subprocess per job
                                  │ stdout/stderr, events.jsonl
                                  ▼
                         python -m mqde.pipeline.llm_job | vla_job   (Python executor)
                                  │ child processes over HTTP on 127.0.0.1
                                  ▼
                         llama-server, vllm serve, llama-perplexity, llama-bench
```

- **One process per job.** The C++ server assigns a GPU from the pool (or the single CPU slot), launches the Python pipeline with the job spec, and follows its `events.jsonl`; crashes or out-of-memory exits do not affect the server. Job state lives in the run directory (`<runs>/<job-id>/`: `job.yaml`, `events.jsonl`, `report.json`, `worker.log`, `export/`), so jobs started from the command line appear in the UI too. After a server restart, the state of a running job is inferred from its event file (a known limitation, B-5).
- **Event stream.** Executors write one JSON event per line; the server replays from `after_seq` and then follows (Server-Sent Events). The design said WebSocket; SSE was chosen as sufficient and is a recorded deviation.
- **Numerics have one definition.** `cpp/src/quant/ops.cpp` is the specification; `python/mqde/quant/numerics.py` mirrors it in PyTorch and a cross-check test (`python/tests/test_numerics_golden.py`, comparing against `mqde._core`) requires agreement. Bit packing for GPTQ, AWQ, Marlin and GGUF is in C++ (`cpp/src/quant/pack.cpp`) and exposed as `mqde._core`.
- **Data over code.** Recipes, hardware profiles, backend profiles, algorithm list, model-family templates and fit constants are YAML under `data/`. The estimator, server, UI and Python executor all read the same files.

## C++ modules and their dependency rule

`common` ← `modelspec`, `quant` ← `estimator` ← `fit`, `hub` ← `server` ← apps/bindings. Lower layers never include higher ones; `scripts/check_deps.py` enforces it (it is part of the local gate).

| Module | Does |
|---|---|
| `common` | `Result<T>` errors, YAML/JSON config loading, logging |
| `modelspec` | safetensors header parsing, `ModelSpec`, family templates, the layer graph |
| `quant` | quantization specs, fake-quant, error metrics, bit packing |
| `estimator` | memory, KV/state, FLOPs, latency floor, deployability |
| `fit` | llmfit-style ranking, speed model, local benchmark index |
| `hub` | Hugging Face catalog, search and download |
| `server` | HTTP endpoints, job runner |
| `probe_c` | C99 hardware probe (NVML loaded dynamically; CPU) |

## Python packages

`quant/` (numerics, algorithms, blockwise engine, KV quantization, fake-quant modules), `models/` (model adapters), `data/` (calibration and evaluation data), `eval/` (PPL, load test, served PPL, resources, boundary search), `export/` (exporters, real check), `backend/` (llama.cpp and vLLM adapters), `pipeline/` (the job pipelines and event sink), `trace/` (sensitivity analysis).

## Deviations from the written design

The design documents describe a larger system. As of today: HTTP is cpp-httplib rather than Drogon; events are SSE rather than WebSocket; C++ dependencies come from conda or Homebrew rather than vcpkg; the job DAG is in Python; C++ is built as C++23; there is no SQLite storage, no content-addressed cache, no JSON-RPC worker and no hardware plugin loader. See [Status & roadmap](../status.md) and the decision list in `docs/PROGRESS.md`.
