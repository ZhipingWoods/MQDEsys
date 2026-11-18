# Quick start

This guide covers the workflow from resource estimation to inspecting quantization results in the web UI. It assumes the build from [Installation](installation.md), a local Hugging Face model directory `<MODEL_DIR>` (Qwen2.5-0.5B-Instruct is used below) and `PYTHONPATH=<REPO>/python`.

## 1. Estimate resources

Add the CLI to `PATH` (the build directory depends on the CMake configuration):

```bash
export PATH=<REPO>/build/mac/bin:$PATH
```

```bash
mqde model inspect <MODEL_DIR>
mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --hw nvidia/rtx4090 --ctx 4096 --conc 1,8,64
```

For each recipe, `mqde estimate` reports weight bytes (including quantization metadata), KV cache, activations, runtime overhead, peak memory relative to device capacity, maximum concurrency and a latency floor. The process reads only model configuration and tensor headers; no weights are loaded.

For a model with the shape of Qwen2.5-0.5B (24 layers, hidden size 896, 14/2 heads, bf16, 4096 context, RTX 4090 profile), the W4A16 peak is 1.30 GiB versus 1.82 GiB for BF16. Metric definitions are given in [Resource estimation](../user_guides/estimate.md).

## 2. Run a quantization job

Jobs are executed by the Python pipeline and can also be submitted from the LLM entry of the **New job** page in the web UI.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --export awq --eval-windows 40 --job-dir runs/quickstart
```

A job specification file can be used instead (the template is `examples/jobs/qwen25_05b_wikitext.yaml`; set `model.path` to the local model directory):

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job examples/jobs/qwen25_05b_wikitext.yaml
```

The job loads the model, measures the unquantized model's perplexity (PPL) on wikitext-2, quantizes block by block while recording per-layer error, and measures PPL again with fake-quantized weights. With `--export awq`, it also writes an AWQ checkpoint, reads it back and re-measures PPL with the real weights. Results are written to `<job-dir>/report.json`; the live event stream is written to `<job-dir>/events.jsonl`.

As a reference, on Qwen2.5-0.5B-Instruct over wikitext-2 with 40 windows of 512 tokens: unquantized PPL is 18.10; W4A16 + GPTQ (INT4, group size 128) gives 20.41 on CPU and 20.66 on CUDA. GPTQ results depend on the BLAS thread count; the job records `env.torch_threads`, and runs should only be compared when that value matches.

## 3. Inspect results in the web UI

Start the server (see [Installation](installation.md)), open **Jobs & execution** and select the job. Command-line jobs started with `--job-dir` under the server's `--runs` directory also appear in the list.

```{figure} ../../_shared/img/jobs.png
:alt: Job list in the web UI
:width: 100%

Job list: each row shows state, current stage, recipes, device and duration.
```

The job detail page presents the stage pipeline, live log, perplexity at each evaluation position, recipe summary, export and consistency results, and the worst layers. See [Web UI guide](../user_guides/web_ui.md) for a complete walkthrough.

## 4. Further reading

- Recipe and algorithm selection: [Quantization](../user_guides/quantization.md)
- Job fields and CLI options: [Job specification](../user_guides/job_spec.md)
- Serving and load testing: [Deployment and benchmarking](../user_guides/deploy_benchmark.md)
- Dataset evaluation: [Dataset evaluation](../user_guides/evaluation.md)
