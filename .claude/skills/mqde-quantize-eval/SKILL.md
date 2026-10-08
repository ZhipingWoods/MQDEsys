---
name: mqde-quantize-eval
description: >
  Run and read a quantization evaluation job in MQDEsys: estimate memory first, pick a recipe and algorithm,
  quantize and evaluate an LLM (wikitext-2 PPL at the pretrain / fake_quant_wo_kv / fake_quant positions), then read
  the report for per-layer error and the mixed-precision suggestion. Use when the user asks to quantize a model with
  this repository, compare W4A16 / W4A8 / W8A16 / W2A16 / FP8, run GPTQ / AWQ / SmoothQuant, find which layers hurt
  accuracy, or write a job YAML. Do NOT use for deploying or benchmarking an exported checkpoint (use
  mqde-deploy-benchmark) or for adding a new algorithm (use mqde-extend).
argument-hint: model_path_or_hf_id [recipes] [algorithm]
user-invocable: true
allowed-tools: Bash(python:*) Bash(python3:*) Bash(build/*:*) Bash(ls:*) Bash(cat:*) Read Grep Glob
---

# Quantize and evaluate a model

The repository root is `<REPO>`. Python tools run as `PYTHONPATH=python python -m mqde.pipeline.llm_job ...` from
`<REPO>`. A model is a local directory of a Hugging Face checkpoint. The first run needs the wikitext-2 parquet files
in the Hugging Face cache (no network access is attempted).

## 1. Check that it fits, before loading any weights

```bash
build/dev/bin/mqde estimate -m <MODEL_DIR_OR_hf:org/name> -r w4a16,fp8 --conc 1,8
```

`hf:org/name` reads only the config and the safetensors header. The output lists weights, KV cache, activation peak
and the maximum concurrency per recipe. Recipes: `mqde recipes`. Hardware profiles: `mqde hardware`.
The estimator's `max_concurrency` charges every sequence its own activation memory, so for vLLM it is a conservative
lower bound (docs/PROGRESS.md, D-9).

## 2. Pick recipe and algorithm

| Recipe | Weights | Activations | Typical use |
|---|---|---|---|
| `w8a16`, `fp8` | int8 / FP8 E4M3 | none / dynamic FP8 | near-lossless |
| `w4a16` | int4, group 128 | none | memory-bound decoding (default choice) |
| `w4a8` | int4, group 128 | int8 dynamic per token | needs `smoothquant+gptq` to behave |
| `w2a16`, `w2a8` | int2 | none / int8 | research; large error |

Algorithms (`quant.algorithm`): `rtn`, `gptq`, `awq`, `smoothquant`, `awq+gptq`, `smoothquant+gptq`, `recipe`
(= the recipe's own default). `awq` and `smoothquant` fold their scales into the previous op so the result stays
exportable. The list the server accepts is `data/algorithms.yaml`.

## 3. Run

Command line (CPU is enough for a 0.5B model; `CUDA_VISIBLE_DEVICES=""` keeps GPUs free):

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16,w8a16 --algorithm gptq \
  --kv int8 --sensitivity block --job-dir runs/my-job
```

or from a YAML (see `examples/jobs/qwen25_05b_wikitext.yaml`):
`PYTHONPATH=python python -m mqde.pipeline.llm_job <job.yaml>`.

Flags: `--recipes`, `--algorithm`, `--device`, `--act-static {minmax,ema,percentile}`, `--kv int8|int4|fp8`,
`--kv-method {kivi,naive}`, `--eval-windows`, `--calib-samples`, `--mixed-precision <JSON|file>`,
`--export gptq,awq,compressed-tensors`, `--sensitivity {block,type}`, `--sensitivity-windows`, `--llamacpp`,
`--vllm`, `--job-dir`. Deployment flags are covered by mqde-deploy-benchmark.

Mixed precision: `{"rules": [{"match": "block:0", "bits": 8}, {"match": "mlp\\.down_proj$", "skip": true}]}`.
First match wins; `block:lo-hi` and negative indices (`block:-1`) select decoder blocks.

## 4. Read the result

`<job-dir>/report.json` (`kind: "llm"`). Keys that matter:

| Where | Meaning |
|---|---|
| `baseline.ppl` | BF16/FP32 reference PPL |
| `recipes.<R>.summary.ppl`, `ppl_delta_pct` | quantized PPL and its change versus the baseline |
| `recipes.<R>.eval.fake_quant_wo_kv` / `fake_quant` | PPL with weights+activations quantized, without / with KV quantization |
| `recipes.<R>.summary.worst_layers` | layers with the lowest output SQNR: mixed-precision candidates |
| `recipes.<R>.sensitivity` | per-block (or per-type) FP fallback ranking and a suggested `rules` list |
| `env.torch_threads` | GPTQ results differ in the last digits with the BLAS thread count; compare runs with the same value |

The web page for a job is `Job → <id>`; the same data is available from `GET /api/v1/jobs/<id>`.

## Rules of thumb

- Compare PPL only inside one job: the evaluation windows and seed are fixed there. The llama.cpp numbers use a
  different protocol (`ctx 512, 40 chunks, second half`).
- A W4A16 RTN result that is far worse than GPTQ/AWQ on the same model is normal; check `--sensitivity block` before
  concluding a model cannot be quantized.
- Never assert a number you did not read from a report. If a run did not finish, say so.

See `references/job-spec.md` for every YAML field and `references/troubleshooting.md` for the failures seen so far.
