# Status and roadmap

MQDEsys is a working vertical slice, not a finished product. The tables below separate what is **DONE** (implemented and accepted), **PARTIAL** (usable, with named gaps) and **TODO** (planned, nothing to use yet). Milestones M0-M2 of the task list are mostly complete for the estimator and UI; M3 (minimal loop) is partial; M4-M7 (all recipes across several backends, error attribution, multi-hardware, 1.0 release with pip and Docker) are open.

The short summary of the most important gaps:

- **Not available**: SGLang and TensorRT-LLM deployment, lm-eval / evalscope accuracy suites (a dependency decision is pending), the estimator feedback loop, Ascend/ROCm/Metal hardware plugins, authentication and a path sandbox beyond `--allow`, CI, Docker image, pip wheel.
- **Partial**: vLLM and llama.cpp deployment (Python only; no C++ backend interface), load tests and boundary search (the boundary search is not yet run on real vLLM), SmoothQuant evaluation position, FP8 per-block, W2 with GPTQ is experimental, model download as a job stage, the HTML report (a generator exists in `python/mqde/report/html.py`; it is not part of the task acceptance yet).

```{include} _status_tables.md
```
