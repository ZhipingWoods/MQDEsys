# REST API reference

The server (`mqde serve`, cpp-httplib) exposes JSON over HTTP under `/api/v1`, with job progress delivered as Server-Sent Events. Endpoint definitions live in `cpp/src/server/server.cpp`; typed web client definitions live in `web/src/api.ts`. The examples below can be executed directly against a running server.

Conventions: request bodies are JSON (an empty body means `{}`); errors are `{"code": "...", "message": "..."}` with status 400 (`invalid_argument`, `parse`), 403 (`forbidden`: path outside `--allow`), 404, 409, 500 or 503. Paths for models and datasets must lie under an allowed root. Custom verbs use a colon (`/jobs/<id>:cancel`). The server listens on loopback and does not enable authentication.

## System

| Method and path | Purpose |
|---|---|
| `GET /health` | `{status, version}` |
| `GET /config` | `{allowed_roots, gpus, runs_dir}` |
| `GET /hardware` | detected devices (`devices`, with matched `profile`) and the job GPU pool (`job_gpus`) |
| `GET /hardware/profiles` | the hardware profile library |
| `GET /benchmarks` | offline benchmark metadata, suites and the EvalScope dataset catalogue (`evalscope`, metadata only) |
| `GET /datasets` | built-in datasets and allowed roots offered by the wizard |
| `GET /datasets/inspect?path=&kind=` | read-only check whether a data file / directory is usable |
| `GET /recipes` | quantization recipes (with `id`, specs, `summary`) |
| `GET /algorithms` | quantization algorithms (from `data/algorithms.yaml`) |
| `GET /installed` | inference runtimes found on this machine (cached for 30 s) |

```bash
curl -s localhost:18080/api/v1/health
# {"status":"ok","version":"0.1.0"}
```

## Models and estimation

| Method and path | Body | Purpose |
|---|---|---|
| `GET /models` | | models found under the allowed roots (directories with `config.json`), each as an inspected spec |
| `POST /models:inspect` | `{path, tensors?}` | parse config and safetensors headers |
| `POST /models:graph` | `{path, focus?, depth?}` | netron-style hierarchy |
| `POST /models:tensor` | `{path, tensor, recipe?}` | statistics, 64-bin histogram and error preview of one tensor |
| `POST /estimate` | `{model, hardware?, backend?, recipes?, runtime?, concurrency_curve?, images_curve?}` | per-recipe estimate with curves; `runtime` keys: `concurrency`, `images_per_step`, `text_tokens`, `context_tokens`, `image_resolution`, `gpu_memory_utilization`, `extra_modalities` |
| `POST /estimate:matrix` | `{model, backend?, runtime?}` | recipe x hardware deployability cells |
| `POST /fit` | `{models?, source?, search?, hardware?, gpu_count?, ram_gb?, vram_gb?, use_case?, context?, concurrency?, quant?, sort?, backend?}` | ranked fit analysis; `source: "catalog"` ranks Hub catalog models |
| `POST /fit:plan` | `{model, ...}` | best recipe on every hardware profile |
| `POST /fit:claim` | `{model, hardware?, min_tps?}` | throughput bounds |
| `POST /fit:concurrency` | `{model, hardware?, recipe?, users?, context?, curve?}` | users that fit, with curves |

Defaults: `hardware` is `nvidia/rtx4090`, `backend` is `hf_transformers`, `recipes` is `["bf16","w4a16"]`.

```bash
curl -s -X POST localhost:18080/api/v1/estimate \
  -d '{"model": "<MODEL_DIR>", "hardware": "nvidia/rtx4090", "recipes": ["w4a16"]}'
```

## Catalog and downloads

| Method and path | Purpose |
|---|---|
| `GET /catalog?search=&provider=&capability=&limit=&include_demoted=` | local model catalog |
| `GET /hf/search?q=&limit=` | live Hugging Face search (needs network) |
| `GET /modelscope/search?q=&limit=` | live ModelScope search (needs network) |
| `POST /download` | `{model: "<org>/<name>", quant?, budget?, output_dir?, list?}`: start a download (or `list: true` to list file groups); returns a download id |
| `POST /modelscope/download` | same, from ModelScope; the download status carries `provider: modelscope` |
| `GET /download/<id>/status` | progress |
| `POST /download/<id>:cancel` | cancel |

## Jobs

| Method and path | Purpose |
|---|---|
| `POST /jobs` | submit a [job spec](job_spec.md); returns `201 {job_id}`. Validates paths, recipes, `quant.algorithm`, `model.type` (`vla` default, or `llm`) and that CPU jobs are LLM jobs |
| `GET /jobs` | all jobs (id, kind, state, stage, recipes, device, model, name) newest first |
| `GET /jobs/<id>` | state, stage, event counts, `report` (the job's `report.json`, or empty while running), the submitted `job` spec |
| `POST /jobs/<id>:cancel` | cancel a queued or running job (409 otherwise) |
| `GET /jobs/<id>/log` | the executor's log (last 256 KiB, plain text) |
| `GET /jobs/<id>/events?after_seq=N` | JSON array of events with `seq > N` |
| `GET /jobs/<id>/stream?after_seq=N` | the same events as Server-Sent Events, following until the job ends; resume with the last `seq` |

States: `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`. Event types: `job_state`, `stage_started`, `stage_finished`, `log`, `metric`, `progress`, `trace`; each has a monotonically increasing `seq`.

```bash
curl -s -X POST localhost:18080/api/v1/jobs -d '{
  "metadata": {"name": "demo"}, "model": {"type": "llm", "path": "<MODEL_DIR>"},
  "quant": {"recipe": ["w4a16"], "algorithm": "gptq"}, "deploy": {"device": "cpu"}}'
curl -N "localhost:18080/api/v1/jobs/<job-id>/stream?after_seq=0"
```

Any other `GET` outside `/api/` serves the web UI (single-page fallback to `index.html`).
