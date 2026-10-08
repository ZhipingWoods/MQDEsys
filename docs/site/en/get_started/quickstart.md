# Quick start

This walk-through goes from "which recipe fits?" to a finished job in the UI. It assumes the build from [Installation](installation.md), a local Hugging Face model directory `<MODEL_DIR>` (we use Qwen2.5-0.5B-Instruct) and `PYTHONPATH=<REPO>/python`.

## 1. Estimate without loading weights

Put the CLI on your path first (the build directory depends on how you configured CMake):

```bash
export PATH=<REPO>/build/mac/bin:$PATH
```

```bash
mqde model inspect <MODEL_DIR>
mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --hw nvidia/rtx4090 --ctx 4096 --conc 1,8,64
```

`mqde estimate` prints, for every recipe, the weight bytes (including quantization metadata), KV cache, activation and runtime overhead, the peak against the device capacity, the maximum concurrency and a latency floor. For a model with the shape of Qwen2.5-0.5B (24 layers, hidden 896, 14/2 heads, bf16, 4096 context, RTX 4090 profile) it reports a W4A16 peak of 1.30 GiB against 1.82 GiB for BF16. [Estimation guide](../user_guides/estimate.md) explains each number.

## 2. Run an LLM quantization job

From the command line the job is a single Python module; the same job can be submitted from the UI ("New job", LLM tab).

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --export awq --eval-windows 40 --job-dir runs/quickstart
```

The same thing as a job file (`examples/jobs/qwen25_05b_wikitext.yaml` is the template; its `model.path` must point at your `<MODEL_DIR>`):

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job examples/jobs/qwen25_05b_wikitext.yaml
```

The job loads the model, measures wikitext-2 perplexity (PPL) of the unquantized model, quantizes block by block with per-layer error tracing, measures PPL again with fake-quantized weights, and with `--export awq` writes a real AWQ checkpoint, reads it back and re-measures PPL with the real weights. Results are written to `<job-dir>/report.json`, and the live event stream to `<job-dir>/events.jsonl`.

For reference, a recorded job on Qwen2.5-0.5B-Instruct (CPU or GPU, wikitext-2, 40 windows of 512 tokens): unquantized PPL 18.10; W4A16 with GPTQ (int4, group size 128) 20.41 on a CPU run and 20.66 on a CUDA run (GPTQ is not bit-reproducible across devices); the AWQ checkpoint exported from the CUDA run measures 20.656 with the real weights, within 0.01% of the fake-quantized value.

```{warning}
GPTQ results depend slightly on the number of BLAS threads (the job records `env.torch_threads`); compare runs only with the same thread count.
```

## 3. Look at the results in the UI

Start the server (see [Installation](installation.md)), open the **Jobs** page and click your job. A job started from the command line with `--job-dir` under the server's `--runs` directory shows up in the list too.

```{figure} ../../_shared/img/jobs.png
:alt: Job list of the web UI
:width: 100%

The job list (UI text is Chinese). Each row shows state, current stage, recipes, device and duration.
```

The job page shows the stage pipeline, the live log, perplexity at three evaluation positions, the recipe summary table, the export and consistency table and the worst layers. The [Web UI tour](../user_guides/web_ui.md) walks through each block with screenshots.

## 4. Where to go next

- Choose recipes and algorithms: [Quantization](../user_guides/quantization.md).
- Every job field and flag: [Job spec reference](../user_guides/job_spec.md).
- Serve the exported checkpoint and benchmark it: [Deployment and benchmarking](../user_guides/deploy_benchmark.md).
