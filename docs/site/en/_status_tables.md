Generated from the overview table of `docs/TASKS.md` by `docs/site/scripts/gen_status.py` (30 DONE, 32 PARTIAL, 41 TODO; other states: 0). The task list in the repository is the source of truth; run `make status` after it changes. Notes are shortened and copied verbatim (English) from the repository.

## PARTIAL

Usable but not fully accepted. The note says what exists and what is missing.

| Task | Title | What exists / what is missing |
|---|---|---|
| T-0001 | Repository init and Git conventions | 2026-09-24, git init on the dev server; remote/branch protection pending |
| T-0004 | Web project skeleton | 2026-09-24, Vite+React+ECharts (no AntD/tests yet) |
| T-0204 | hub: HF/ModelScope resolution and metadata | 2026-09-29, @revision, HF_ENDPOINT, retries, Range headers, inspect_remote; ModelScope + offline cache pending |
| T-0212 | CLI: probe / model inspect / estimate / plan | 2026-09-24, probe/inspect/estimate/recipes/hardware |
| T-0301 | server: HTTP skeleton and read-only endpoints | 2026-09-24, cpp-httplib instead of Drogon (see PROGRESS) |
| T-0302 | server: estimation endpoints | 2026-09-24 |
| T-0303 | web: API client, layout, routing | 2026-09-24, hand-written typed client; generated client pending T-0007 |
| T-0304 | web: new-job wizard with live estimate sidebar | 2026-09-24, form, no live estimate sidebar |
| T-0305 | web: estimation center charts | 2026-09-24 |
| T-0306 | web: hardware page | 2026-09-24 |
| T-0405 | quant core: bit packing (GPTQ/AWQ/Marlin/GGUF) | 2026-09-30, GPTQ v1/v2 2/3/4/8, AWQ GEMM, GGUF Q8_0/Q5_0/Q4_0/IQ4_NL/Q4_K/Q2_K byte-exact vs ggml; Marlin 4/8-bit pack/unpack 2026-10-05, round-trip… |
| T-0503 | pipeline: DAG, stage interface, state machine | 2026-09-24, Python-side stages; C++ DAG pending |
| T-0504 | pipeline: queue, scheduling, device allocation, retry | 2026-09-24, GPU pool + queue; no retry |
| T-0510 | Evaluation: PPL + logits probe | 2026-09-29, wikitext-2 PPL 3 positions; logits probe only in tests |
| T-0511 | Stages S00-S08 + S14 wired together | 2026-09-24, VLA stages S00-S08,S14 |
| T-0512 | server: job endpoints + event stream | 2026-09-29, SSE; model.type vla/llm, CPU slot |
| T-0513 | web: job execution page | 2026-09-24 |
| T-0514 | web: quantization error page (initial) | 2026-09-24, heatmap + Top-K + attribution |
| T-0515 | End-to-end minimal loop (M3) | 2026-09-30, docs/guides/quickstart.md + LLM 全功能作业跑通; VLA 演示待 GPU |
| T-0603 | Algorithm: SmoothQuant | 2026-09-30, per-group scales folded into the previous op (FP-exact, exportable); transformed eval position not yet |
| T-0605 | FP8 recipe (E4M3/E5M2, per-channel/per-block) | 2026-09-24, FP8 E4M3/E5M2 per-channel/per-tensor fake-quant; no per-block |
| T-0606 | 2-bit recipes (W2A16/W2A8) and HQQ | 2026-09-24, W2A16/W2A8 fake-quant via RTN-MSE; no GPTQ/HQQ |
| T-0610 | Algorithm combos (AWQ+GPTQ) and QuaRot | 2026-09-29, awq+gptq / smoothquant+gptq; no QuaRot |
| T-0701 | Exporter interface + vLLM export | 2026-10-08: the exporter's own AWQ / compressed-tensors W4A16 / compressed-tensors FP8 checkpoints of Qwen2.5-0.5B load in vLLM 0.30.0 and serve (ser… |
| T-0703 | Deploy backend interface + vLLM adapter (S11) | 2026-10-08, Python only: backend/base.py ServerBackend + vllm_server.py (launch, health, log diagnosis, engine_info) + llm_job deploy_vllm stage + da… |
| T-0802 | Post-deployment accuracy evaluation (S12) | 2026-10-08, eval/served_ppl.py: PPL from the backend's own prompt_logprobs, same windows as the fake/real PPL; vLLM only. lm-eval / evalscope adapter… |
| T-0803 | Performance load tester (S13): TTFT/TPOT/throughput | 2026-10-08, Python eval/loadtest.py: closed loop + Poisson open loop, exact-length token prompts, TTFT/TPOT/ITL/E2E p50/p90/p99; against the mock (TT… |
| T-0804 | Memory sampling and OOM boundary search | 2026-10-08, eval/resources.py 100 ms GPU/host sampler, eval/boundary.py bisection over concurrency / context with a fresh backend per probe; tested o… |
| T-0807 | web: evaluation report and compare pages | 2026-09-30, LLM/VLA compare tabs + llama.cpp table; standalone report page pending T-0806 |
| T-0901 | Stage attribution (waterfall) data path | 2026-09-24, modality-level attribution only |
| T-0902 | Sensitivity analysis (per-layer fallback) | 2026-09-29, block/type FP fallback in llm_job; 7B GPU timing pending |
| T-0904 | Mixed-precision suggestion -> one-click new job | 2026-09-29, suggest() + UI one-click clone; Playwright spec pending |

## TODO

Planned, not started. Nothing in this table works today.

| Task | Title |
|---|---|
| T-0006 | CI pipeline (PR gate) |
| T-0007 | Contracts and schemas |
| T-0008 | Development guide and ADRs |
| T-0103 | common: registry, hashing, thread pool |
| T-0104 | storage: SQLite metadata and migrations |
| T-0105 | storage: content-addressed cache and GC |
| T-0106 | CLI skeleton |
| T-0209 | estimator: performance estimate and calibration store |
| T-0210 | planner: compatibility matrix and JobSpec validation |
| T-0211 | planner: recommendation and deployability matrix |
| T-0406 | quant core: SIMD/CUDA acceleration |
| T-0501 | Worker RPC protocol and C++ WorkerClient |
| T-0502 | Python worker process framework |
| T-0505 | Model download (fetch_model stage) |
| T-0609 | External algorithm adapters (GPTQModel / llm-compressor) |
| T-0704 | SGLang export and adapter |
| T-0706 | TensorRT-LLM export and adapter |
| T-0801 | Accuracy evaluation: lm-eval-harness / evalscope / custom |
| T-0805 | HAL micro-benchmarks |
| T-0806 | Report generation (JSON + HTML) |
| T-0808 | M4 acceptance: all recipes x multiple backends |
| T-0903 | web: full error page |
| T-0905 | Feedback loop (S15): automatic estimate calibration |
| T-0906 | Error attribution acceptance (injection test) |
| T-1001 | Multimodal framework: per-modality config |
| T-1002 | VLM adapters (Qwen2-VL / LLaVA) |
| T-1003 | MoE / MLA support (DeepSeek family) |
| T-1005 | World model / video generation (DiT) adapter |
| T-1101 | HAL ABI and plugin loader + template plugin |
| T-1102 | CPU / CUDA plugin |
| T-1103 | Metal plugin |
| T-1104 | ROCm plugin |
| T-1105 | Ascend (CANN) plugin |
| T-1106 | Plugin SDK docs and ABI compatibility check |
| T-1201 | Docker image and compose |
| T-1202 | pip wheel packaging and release |
| T-1203 | Documentation site (GitHub Pages) |
| T-1204 | Security hardening (credentials, path sandbox, auth) |
| T-1205 | Observability (/metrics, reproducible replay) |
| T-1206 | Nightly GPU regression and performance board |
| T-1207 | v1.0 release acceptance |

## DONE

Implemented and verified as the task's acceptance describes.

| Task | Title |
|---|---|
| T-0002 | C++ build system (CMake + presets) |
| T-0003 | Python package skeleton and pybind11 build |
| T-0005 | Coding standards and pre-commit |
| T-0101 | common: errors, Result, logging |
| T-0102 | common: config loading and schema validation |
| T-0201 | C hardware probe: CPU |
| T-0202 | C hardware probe: GPU/NPU (dynamic vendor libs) |
| T-0203 | Hardware profile library |
| T-0205 | modelspec: ModelSpec from config + safetensors headers |
| T-0206 | Architecture templates and QIR |
| T-0207 | Quantization recipe library |
| T-0208 | estimator: memory / KV / concurrency |
| T-0401 | quant core: QuantSpec / QParams / granularity |
| T-0402 | quant core: calibration (MinMax/MSE/percentile/EMA) |
| T-0403 | quant core: INT/FP8 quantize, dequantize, fake-quant + golden |
| T-0404 | quant core: error metrics |
| T-0407 | pybind11 binding mqde._core |
| T-0408 | Torch fast path and dual-end golden test |
| T-0506 | Python model abstraction (BaseModel + HF decoder) |
| T-0507 | Calibration data and first-block capture |
| T-0508 | Blockwise engine + fake/real modules + RTN |
| T-0509 | Error tracing (per layer / block) |
| T-0601 | Algorithm: AWQ |
| T-0602 | Algorithm: GPTQ |
| T-0604 | Activation quantization (dynamic/static): W4A8, W2A8 |
| T-0607 | KV cache quantization |
| T-0608 | Mixed precision and keep-FP policy |
| T-0702 | Fake/real consistency check (S10) |
| T-0705 | llama.cpp (GGUF) export and adapter |
| T-1004 | VLA adapter (experimental) |
