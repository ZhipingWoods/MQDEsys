# Job spec reference

**Status: implemented.** A job is described by a YAML file or JSON body (`POST /api/v1/jobs`); the LLM pipeline reads it into `LLMJobSpec` (`python/mqde/pipeline/llm_job.py`). Fields the server adds (`apiVersion: mqde/v1`, `kind: Job`) are ignored by the pipeline.

## Minimal example

```yaml
metadata: {name: qwen05b-w4a16}
model: {path: <MODEL_DIR>}
quant: {recipe: [w4a16], algorithm: gptq}
deploy: {device: cpu}
output_dir: runs
```

The repository example `examples/jobs/qwen25_05b_wikitext.yaml` additionally sets `act_static`, `kv`, `mixed_precision`, `calibration` and `evaluation`.

## YAML fields (LLM jobs)

| Key | Spec field | Default | Meaning |
|---|---|---|---|
| `metadata.name` | `name` | file name | job name shown in the UI |
| `model.path` | `model_path` | required | local Hugging Face model directory. The server requires it to lie under an `--allow` root |
| `model.type` | (server) | `vla` | `llm` or `vla`; the server routes the job to the matching pipeline. Set `llm` for the LLM pipeline |
| `quant.recipe` | `recipes` | required | one recipe or a list (`w4a16`, `w8a16`, `fp8`, `w4a8`, `w2a16`, `w2a8`, `bf16`) |
| `quant.algorithm` | `algorithm` | `recipe` | `rtn`, `gptq`, `awq`, `smoothquant`, `awq+gptq`, `smoothquant+gptq`, or `recipe` (use each recipe's default) |
| `quant.act_static` | `act_static` | `null` | `minmax`, `ema` or `percentile` for static per-tensor activation scales; `null` is dynamic per-token |
| `quant.kv` | `kv_override`, `kv_method`, `kv_granularity` | off | `{format, bits, symmetric, method, granularity}`; see [Quantization](quantization.md) |
| `quant.mixed_precision` | `mixed_precision` | none | `{rules: [{match, bits | skip}]}` |
| `calibration.n_samples` | `calib_samples` | 32 | calibration windows |
| `calibration.seq_len` | `calib_seq_len` | 512 | tokens per calibration window |
| `evaluation.seq_len` | `eval_seq_len` | 512 | tokens per PPL window |
| `evaluation.max_windows` | `eval_windows` | 40 | number of PPL windows |
| `deploy.device` | `device` | `cpu` | `cpu`, or a CUDA device for the quantization run. Through the server, `cpu` uses the single CPU slot and anything else needs a GPU from `--gpus` |
| `deploy.llamacpp.types` | `llamacpp` | `[]` | GGUF types to deploy: `q8_0`, `q5_0`, `q4_0`, `iq4_nl`, `q4_k`, `q2_k` |
| `deploy.llamacpp.gpu_layers` | `llamacpp_gpu_layers` | llama.cpp default | `-ngl`; 0 CPU, 99 offload all (Metal / CUDA builds) |
| `deploy.llamacpp.serve` | `llamacpp_serve` | `{concurrency: 4, requests: 16, prompt_len: 256, n_predict: 64}` | `llama-server` load test per GGUF type; also `ctx`. `null` disables |
| `deploy.vllm` | `vllm` | off | `true` or a mapping: serve each exported checkpoint with vLLM (needs CUDA). Keys: `max_model_len` (1024), `gpu_memory_utilization` (0.3), `enforce_eager` (true), `concurrency` (4), `requests` (16), `prompt_len` (256), `n_predict` (64), `ppl_windows` |
| `export.formats` | `export` | `[]` | `gptq`, `awq`, `compressed-tensors` |
| `analysis.sensitivity` | `sensitivity` | off | `block` or `type` |
| `analysis.sensitivity_windows` | `sensitivity_windows` | 8 | PPL windows per sensitivity probe |
| `output_dir` | `output_dir` | `runs` | the job writes to `<output_dir>/<job-id>/` |

`seed` (0) and `suggest_target` (0.5) are fields of the spec but are not read from YAML.

## Command line

`python -m mqde.pipeline.llm_job [spec] [options]` (run with `PYTHONPATH=python`). A YAML `spec` is optional; options override it.

| Option | Meaning |
|---|---|
| `--model` | model directory |
| `--recipes` | comma-separated recipes (default `w4a16` when no YAML) |
| `--algorithm` | see the table above |
| `--device` | `cpu` or `cuda` |
| `--act-static {minmax,ema,percentile}` | static activation scales |
| `--kv` | `int8`, `int4` or `fp8`: overrides the recipe KV format |
| `--kv-method {kivi,naive}` | KV method |
| `--eval-windows`, `--calib-samples` | evaluation and calibration sizes |
| `--mixed-precision` | inline JSON or a YAML/JSON file with `{rules: [...]}` |
| `--export` | comma-separated: `gptq`, `awq`, `compressed-tensors` |
| `--sensitivity {block,type}`, `--sensitivity-windows` | sensitivity analysis |
| `--llamacpp` | GGUF types, e.g. `q8_0,q4_k` |
| `--llamacpp-gpu-layers` | llama.cpp `-ngl` |
| `--llamacpp-serve` | JSON load-test settings, or `off` |
| `--vllm [JSON]` | serve exported checkpoints with vLLM; optional JSON such as `'{"gpu_memory_utilization": 0.4}'` |
| `--job-dir` | output directory (default `<output_dir>/<job-id>`) |
| `--job-id` | id assigned by the server |

The command prints a JSON summary (job id, baseline PPL, per-recipe PPL and algorithm). The directory holds `report.json`, `events.jsonl`, and `export/` when exports or GGUF files were produced.

## VLA jobs

`python -m mqde.pipeline.vla_job <spec> [--algorithm ...]` runs the MiniCPM-RobotManip action-error evaluation (fake-quant only, needs a GPU); `examples/jobs/minicpm_vla_libero.yaml` and `minicpm_vla_smoke.yaml` are the templates. The VLA pipeline reports action error against the BF16 model; it is experimental and not covered further here.

## Events and report

`events.jsonl` has one JSON object per line: `job_state` (`RUNNING`, `SUCCEEDED`, `FAILED`), `stage_started`, `stage_finished` (with `seconds`, `status`), `log`, `metric` (`name`, `value`), `progress` (`value` 0..1) and `trace` (per-layer error). `report.json` keys include `baseline`, `recipes.<NAME>` (with `eval`, `summary`, `layers`, `export`, `sensitivity`), `llamacpp`, `env` and `stages`. Treat the report as an evolving format: it is not yet versioned or schema-validated (T-0007).
