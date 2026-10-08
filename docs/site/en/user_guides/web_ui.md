# Web UI tour

**Status: implemented** (React + TypeScript + ECharts, served by `mqde serve`). The UI text is **Chinese** (an i18n layer for zh-CN / en-US is a project rule but English strings are not shipped yet); the captions below translate what you see. Screenshots come from a running instance with finished jobs; model path fields are cropped out.

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

Resource estimate (cropped to hide the model path field): stacked memory per recipe against the 21.6 GiB capacity line, then a table with deployability, maximum concurrency and latency floor.
```

## New job

Route `/new`. Choose the model type (VLA or LLM), recipes (experimental ones are marked), algorithm (default: each recipe's default), evaluation data and target hardware. For LLM jobs the form also offers static activation, KV cache quantization, mixed precision, export formats, llama.cpp types and sensitivity analysis. The **JobSpec preview** at the bottom is the exact JSON that "Submit" posts to `POST /api/v1/jobs`, which is the [job spec](job_spec.md) you can also write by hand.

```{figure} ../../_shared/img/new-job.png
:alt: New job form, quantization recipe and algorithm section
:width: 100%

Recipe and algorithm selection (the model path field above it is not shown).
```

## Jobs and a job page

Route `/jobs` lists every job in the `--runs` directory, including those started from the command line, with state, kind (LLM/VLA), current stage, recipes, device and duration.

```{figure} ../../_shared/img/jobs.png
:alt: Job list
:width: 100%

Job list: seven LLM jobs and four VLA jobs; one failed job is shown in red.
```

A job page (`/jobs/<job-id>`) has, from top to bottom:

- **Execution** (stage pipeline, coloured by state, with duration and repeat counts) and a **live log** fed by the event stream;
- summary cards: baseline PPL, evaluated tokens, calibration windows, model blocks and linear layers;
- tabs: **Overview** and one **error analysis** tab per recipe (layer-error heat map by block and layer type in SQNR dB, per-block mean SQNR, error versus activation kurtosis, the 15 worst layers, and sensitivity results with the "create a new job from the suggestion" button);
- **PPL chart** at the three evaluation positions, **recipe summary** (algorithm, spec, PPL, delta against the original model, KV delta, quantized/skipped layers, time, worst layer);
- **real low-bit export (S09) and fake/real consistency (S10)** table;
- **vLLM deployment (S11) and served accuracy (S12)** table, and/or the **llama.cpp deployment** chart and table.

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

`/hardware` shows the devices found by the C probe (with job-pool state) and the hardware profile library; `/catalog` searches a local Hugging Face model catalog and the live Hub, and downloads weights within a budget (the server needs network access, and a proxy if the host has none). Both are plain tables and are not pictured here because they show details of the host machine.
