# Adding a deployment backend

A deployment backend serves a quantized checkpoint over an OpenAI-compatible HTTP API as a child process
(T-0703, S11). Two already exist: `python/mqde/backend/vllm_server.py` (`VllmServer`) and
`python/mqde/backend/llamacpp_server.py` (`LlamaServer`); SGLang (`sglang_server.py`, `SglangServer`) is a third, and shows what a
new backend with its own protocol needs beyond the process wrapper: a scoring path in `eval/served_ppl.py`, log parsing for
`engine_info`, a mock mode in `tests/backend/fake_llama_server.py`, and a measured `backends:` entry in `data/compat/matrix.yaml`.

**This is not pluggable the way algorithms are (see `add-algorithm.md`) or exporters are
(`python/mqde/export/base.py`'s `EXPORTER_REGISTRY`).** There is no `@register()` decorator and no registry
for backends today: `python/mqde/pipeline/llm_job.py` imports `VllmServer`/`LlamaServer` by name and
dispatches on which JobSpec field is set (`deploy.vllm` vs `deploy.llamacpp`). Adding a backend means adding
a new branch there by hand. T-1302 (this guide's own task) did not build a backend registry — that is a
real architectural gap, not an oversight to paper over; if you need true pluggability here, that is its own
task, not something to improvise as a side effect of adding one backend.

## 1. Write the `ServerBackend` subclass

Subclass `python/mqde/backend/base.py`'s `ServerBackend`. It already handles the process lifecycle: launching
with its own session (so `stop()` can take down worker processes the server spawned), polling for
readiness, capturing combined stdout/stderr to a log file, and turning an early exit or a startup timeout
into an error with a diagnosis. You implement:

- `__init__`: build `self.args` (the command line) and call `super().__init__(args, log_path, port=..., env=...)`.
- `ready() -> bool`: one HTTP call that returns `True` once the server can serve requests (raise/return
  `False` while it's still loading; `start()` polls this every 0.25s against `startup_timeout`).
- `diagnoses: ClassVar[list[tuple[str, str]]]`: `(needle, human-readable reason)` pairs checked against the
  log on failure — `VllmServer`'s list recognizes OOM, unsupported quantization format, and port-in-use
  messages from real vLLM output; write yours from real failure logs, not guessed strings.

Read `VllmServer` end to end before starting (it is not long); it is the more complex of the two existing
examples (health check via `/health`, log-based diagnosis, `engine_info()` reading `/v1/models`).

## 2. Add the `data/backends/<name>.yaml` entry (only if it affects estimation)

```yaml
id: my_backend
runtime_overhead_bytes: 536870912   # 512 MiB: fixed engine overhead beyond weights + KV + activations
allocator_factor: 1.05              # fragmentation/allocator slack multiplier
eta_bw: 0.6                         # fraction of nominal memory bandwidth this engine actually achieves
mfu: 0.35                           # fraction of nominal FLOPs actually achieved (compute-bound phases)
activation_model: per_seq           # or "batched" (vLLM-style: one shared batch, see ADR-0003)
max_batched_tokens: 8192            # only used when activation_model is "batched"
```

This file feeds the **C++ roofline estimator** (`estimator::load_backend`, `GET /api/v1/backends`) — it is
about predicting memory/throughput before running anything, and has no connection to the Python
`ServerBackend` subclass's actual network calls. A backend only needs this file if you want the estimator
and the new-job live-estimate sidebar to model it; `data/backends/hf_transformers.yaml` and
`data/backends/vllm.yaml` are the two real examples. There is currently no `llamacpp.yaml` despite
llama.cpp being a real, shipped backend (T-0705) — the GGUF/llama.cpp deployment path has never been wired
into the estimator's backend-selector, a pre-existing gap this guide does not fix.

## 3. Wire the dispatch (today: by hand)

In `python/mqde/pipeline/llm_job.py`, find how `deploy.vllm` / `deploy.llamacpp` are read from the JobSpec
and how each constructs its `ServerBackend` subclass and runs smoke/benchmark/eval stages against it
(`_vllm_serve`/`_llamacpp` methods). Add an equivalent branch for your backend's JobSpec field. There is no
shortcut here yet; see the warning at the top of this guide.

## 4. Export format compatibility

If your backend only accepts certain export formats, add it to `data/compat/matrix.yaml`'s `backends:`
section (L2 — propose the rule, cite where you verified it, same as the existing vLLM entries) so
`planner::check_deploy` rejects an incompatible recipe/format combination before a job wastes GPU time
trying to serve it. llama.cpp/GGUF has no entry there today either (see T-1304's plan to add one when the
new-job wizard gains backend-aware export-format suggestion).

## 5. Test it

Follow `tests/integration/deploy_llamacpp_cpu.sh` or the real vLLM verification documented in
`internal/PROGRESS.md`'s T-0701/T-0703 entries: a tiny real checkpoint, real deploy, a smoke request, and (where
feasible) a served-PPL check against the unquantized baseline. A backend that was never run against a real
server of its kind is not verified — do not mark the task DONE on mocked HTTP calls alone.
