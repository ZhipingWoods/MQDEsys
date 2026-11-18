# Web UI tour

The web workspace is built with React, TypeScript and ECharts, served by `mqde serve`, and presented in Chinese; the captions below translate the interface. Screenshots come from a running instance with completed jobs.

The left sidebar groups the pages: *before evaluation* (deployment pre-evaluation, model structure, resource estimate), *quantization and evaluation* (new job, jobs, recipe comparison) and *deployment* (hardware and backends, model catalog). A dark/light theme switch sits at the bottom.

## Deployment pre-evaluation

Route `/fit`. The target-system panel (hardware profile, GPU count, system RAM, concurrency, context, use case) drives a ranked table of the models in the library: parameter count, fit level (perfect / good / marginal / too tight), run mode, best quantization, memory, speed and score. Selecting a row shows the score breakdown (quality, speed, fit, context), key metrics and a cross-hardware plan. This is `mqde fit` with a GUI; see [Estimation](estimate.md).

```{figure} ../../_shared/img/fit.png
:alt: Deployment pre-evaluation page
:width: 100%

Deployment pre-evaluation (UI text in Chinese): the target system on top, the model ranking in the middle, score breakdown and cross-hardware plan at the bottom.
```

## Model structure

Route `/graph`. A netron-style browser: component, block, module, tensor. Node colours are operator types; clicking a node shows its tensors, value histogram and a quantization-error preview for a chosen recipe. It works on a local model or a Hub model (header only, no weights).

```{figure} ../../_shared/img/graph.png
:alt: Model structure page
:width: 100%

Model structure of a vision-language-action model: layer tree on the left, the forward graph of one block in the middle, properties on the right.
```

## Resource estimate

Route `/estimate`. One model times several recipes on one hardware profile, with memory composition (weights, quantization metadata, KV cache / state, activation, runtime) against available capacity, plus curves over concurrency and over input images. The same numbers as `mqde estimate`.

```{figure} ../../_shared/img/estimate.png
:alt: Resource estimate page
:width: 100%

Resource estimate (model path shown as a placeholder): stacked memory per recipe at a 4096-token context, then a table with deployability, maximum concurrency and latency floor.
```

## New job

Route `/new` is the full pipeline. LLM jobs have two additional dedicated entries:

- `/new/quantize`: quantize and export only. The common path is “primary recipe → optional comparison recipes → algorithm → calibration data → export format”; advanced options are folded away;
- `/new/eval`: evaluate an existing quantized checkpoint. The page only has “checkpoint → evaluation datasets → optional deployment”, and datasets are chosen through suite / category / dataset dropdowns.

The full pipeline page keeps model type, recipes, algorithm, evaluation data and target hardware. LLM static activation, KV-cache quantization, mixed precision, llama.cpp types and sensitivity analysis live under “advanced”. The **JobSpec preview** at the bottom is the exact JSON that "Submit" posts to `POST /api/v1/jobs`, which matches the [job spec](job_spec.md) format that can also be written by hand.

```{figure} ../../_shared/img/ui-new-eval.png
:alt: Evaluate-only job form
:width: 100%

Evaluate-only job: checkpoint, datasets and run settings in separate blocks; deployment and advanced options are folded away.
```

```{figure} ../../_shared/img/ui-new-quantize.png
:alt: Quantize-only job form
:width: 100%

Quantize-only job: primary recipe, comparison recipes, calibration data and export format on one screen.
```

## Jobs and a job page

Route `/jobs` lists every job in the `--runs` directory, including those started from the command line, with its mode (full / quantize-only / evaluate-only / deploy), state, current stage, progress, recipes, device and duration. Clicking a row opens the job.

```{figure} ../../_shared/img/ui-jobs.png
:alt: Job list
:width: 100%

Job list: running jobs show their current stage and a progress bar; the header offers one-click full, quantize-only and evaluate-only jobs.
```

A job page (`/jobs/<job-id>`) has, from top to bottom:

- **Job header**: state, mode, device, completed stages, elapsed time and a progress ring, plus cancel / download log / bilingual HTML report actions;
- **Live monitor**: current stage, progress, latest metric, elapsed time, and a log panel with all / metrics / errors filters, follow-tail, copy and download;
- **Execution** (stage pipeline, coloured by state, with duration and repeat counts) and a **live log** fed by the event stream;
- summary cards: baseline PPL, evaluated tokens, calibration windows, model blocks and linear layers;
- tabs: **Overview** and one **error analysis** tab per recipe (layer-error heat map by block and layer type in SQNR dB, per-block mean SQNR, error versus activation kurtosis, the 15 worst layers, and sensitivity results with the "create a new job from the suggestion" button);
- **PPL chart** at the three evaluation positions, **recipe summary** (algorithm, spec, PPL, delta against the original model, KV delta, quantized/skipped layers, time, worst layer);
- **real low-bit export (S09) and fake/real consistency (S10)** table;
- **vLLM deployment (S11) and served accuracy (S12)** table, and/or the **llama.cpp deployment** chart and table.

```{figure} ../../_shared/img/ui-job-llm.png
:alt: LLM job detail page
:width: 100%

Job detail: state, stages, elapsed time and progress ring at the top; stage pipeline, live metrics and a filterable log in the middle; report charts and per-recipe error analysis below.
```

```{figure} ../../_shared/img/job-vllm.png
:alt: Job page of a W4A16 job exported and served by vLLM
:width: 100%

Job 20261008-101629-62da92 (Qwen2.5-0.5B-Instruct, W4A16 GPTQ): the AWQ and compressed-tensors checkpoints are exported, all integer codes identical, and vLLM serves them with MarlinLinearKernel at served PPL +0.01% against the real weights.
```

```{figure} ../../_shared/img/job-llamacpp.png
:alt: Job page with a llama.cpp GGUF deployment
:width: 100%

Job 20261001-004149-75d020: GGUF F16, Q8_0, Q4_K and IQ4_NL on a 48-thread CPU with llama-server (serve tok/s column); 144 tensors of Q4_K fall back to Q5_0.
```

```{figure} ../../_shared/img/job-full.png
:alt: Full-featured LLM job page with two recipes
:width: 100%

Job 20260930-190720-c80eda: W4A16 and W4A8 with GPTQ, INT8 KIVI KV quantization, sensitivity analysis, AWQ and GPTQ exports (4 checkpoints, all "fully consistent") and the llama.cpp table.
```

## Recipe comparison

Route `/compare`. All finished jobs on one plot: PPL against average weight bit width (a log axis so W2 collapses stay visible), and a per-job table with original, fake-quant, with-KV and real-export PPL. Only jobs on the same model and evaluation window are directly comparable. A second tab compares VLA action errors.

```{figure} ../../_shared/img/compare.png
:alt: Recipe comparison page
:width: 100%

Recipe comparison: points are job/recipe pairs (W4A16 blue, FP8 orange, W4A8 green).
```

## Hardware and backends, and the model catalog

`/hardware` starts with inference-backend environment cards (vLLM / SGLang / llama.cpp versions and install state), followed by:

- target-device probe cards: device name, vendor, arch, FlagTree target, compute units, driver and memory usage;
- a compute-capability line chart with exact TFLOPS values;
- HAL plugin cards and micro-benchmarks;
- hardware profiles grouped into edge / workstation / data centre.

`/catalog` provides local models, the HF catalogue, and live HuggingFace / ModelScope search and download. The downloader supports budget filtering, weight-group selection and resumable transfers (the server needs network access, and a proxy if the host has none).

```{figure} ../../_shared/img/ui-hardware.png
:alt: Hardware and backends page
:width: 100%

Hardware and backends: backend cards at the top, then deployment targets, probed devices, capability lines and grouped profiles.
```
