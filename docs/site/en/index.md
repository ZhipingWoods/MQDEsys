# MQDEsys

**MQDEsys** (Model Quantization & Deployment Evaluation System) answers one question end to end: *if I quantize this model with this recipe and deploy it on this hardware, what do I get?* You name a model, a quantization recipe (W8A16, FP8, W4A16, W4A8, W2A16, W2A8), evaluation data and target hardware; the system estimates resources, quantizes, evaluates, exports real low-bit checkpoints, serves them with an inference engine, load-tests them, and shows every step in a web UI, down to the layer that loses the most accuracy.

```{note}
MQDEsys is under active development. Every page marks what is **implemented**, **partial** or **planned**; the [Status & roadmap](status.md) page lists all tasks. Numbers in these docs come from real runs recorded in the repository (`docs/PROGRESS.md`) or in the web UI, and the model used is almost always Qwen2.5-0.5B-Instruct.
```

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
status
```
