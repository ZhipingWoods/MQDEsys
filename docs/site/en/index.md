# MQDEsys

**MQDEsys** (Model Quantization & Deployment Evaluation System) evaluates how a model performs under a given quantization recipe and target hardware. Given a model, a quantization recipe (W8A16, FP8, W4A16, W4A8, W2A16, W2A8), evaluation data and target hardware, the system estimates resources, quantizes, evaluates, exports real low-bit checkpoints, serves them with an inference engine and runs load tests. The web UI presents the complete process, down to the layer with the largest accuracy loss.

```{note}
This documentation describes the architecture, interfaces and usage of the current MQDEsys release. The models, configurations and evaluation data in the examples illustrate typical workflows; replace them with the target model, dataset and hardware in production.
```

## UI preview

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Jobs and execution
:img-top: ../_shared/img/ui-jobs.png
Full, quantize-only, evaluate-only and deploy jobs in one list, with mode, stage and progress visible.
:::

:::{grid-item-card} Job detail
:img-top: ../_shared/img/ui-job-llm.png
Progress ring, stage pipeline, live log, metrics and per-recipe error analysis.
:::

:::{grid-item-card} Hardware and backends
:img-top: ../_shared/img/ui-hardware.png
Backend environments, target-device probes, capability lines and grouped hardware profiles.
:::

:::{grid-item-card} Evaluate-only job
:img-top: ../_shared/img/ui-new-eval.png
Checkpoint, datasets and run settings in three blocks; advanced options folded away.
:::
::::

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Get started
:link: get_started/quickstart
:link-type: doc
Estimate a model, run a quantization job, read the results in the UI.
:::

:::{grid-item-card} Estimate and fit
:link: user_guides/estimate
:link-type: doc
`mqde estimate` and `mqde fit`: memory, KV cache, concurrency, deployability, without loading weights.
:::

:::{grid-item-card} Quantize
:link: user_guides/quantization
:link-type: doc
RTN, GPTQ, AWQ, SmoothQuant, combinations, FP8, mixed precision, KV cache quantization.
:::

:::{grid-item-card} Export and verify
:link: user_guides/export
:link-type: doc
GPTQ, AWQ, compressed-tensors, GGUF, Marlin packing, and the fake/real consistency check.
:::

:::{grid-item-card} Deploy and benchmark
:link: user_guides/deploy_benchmark
:link-type: doc
llama.cpp (CPU and Metal), vLLM, load tests, served PPL, OOM boundary search.
:::

:::{grid-item-card} Web UI and REST API
:link: user_guides/web_ui
:link-type: doc
A tour of every page, plus the HTTP API behind it.
:::

:::{grid-item-card} Dataset evaluation
:link: user_guides/evaluation
:link-type: doc
Offline PPL / QA / MCQ, the 144-dataset EvalScope catalogue, metrics and reports.
:::
::::

```{toctree}
:maxdepth: 2
:caption: Get started

get_started/introduction
get_started/installation
get_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guides

user_guides/estimate
user_guides/quantization
user_guides/evaluation
user_guides/job_spec
user_guides/export
user_guides/deploy_benchmark
user_guides/web_ui
user_guides/rest_api
```

```{toctree}
:maxdepth: 2
:caption: Advanced

advanced/architecture
advanced/extending
advanced/testing
```

```{toctree}
:maxdepth: 1
:caption: Reference

faq
```
