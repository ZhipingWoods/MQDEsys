# FAQ and troubleshooting

## Installation and start-up

**`mqde: command not found`.** The binary is `<build-dir>/bin/mqde`; put that directory on `PATH` or call it by path.

**CMake cannot find `nlohmann_json`, `yaml-cpp`, `fmt`, `CLI11` or `httplib`.** Install them (Homebrew, apt, conda-forge) and pass `-DCMAKE_PREFIX_PATH=<prefix>`. The `dev` preset looks under `$CONDA_PREFIX`, so activate your conda environment first or configure by hand as in [Installation](get_started/installation.md).

**The build fails on a warning.** The project builds with `-Wall -Wextra -Werror`; a new compiler can introduce a warning that is an error here. Report it rather than disabling the flag.

**`ImportError` for `mqde._core`.** The extension is built into `python/mqde/` only when CMake was configured with `-DMQDE_WITH_PYTHON=ON` using the same Python minor version you run (the file name carries it, for example `cpython-312`). Rebuild if you change Python.

**Server returns HTTP 403 `model path is outside allowed roots`.** Start the server with `--allow <parent-of-your-model>` (repeatable).

**Server returns 503 `server started without a GPU pool`.** Start it with `--gpus <ids>`, or submit an LLM job with `deploy.device: cpu`. VLA jobs always need a GPU.

**`deploy.device=cpu is only supported for model.type=llm`.** Set `model.type: llm` in the job (the default is `vla`).

**The job list is empty or a job is stuck "running" after a server restart.** The server rebuilds job state from each job's `events.jsonl`; a job whose process died with the server stays "running" (known limitation).

## Quantization results

**GPTQ results differ between runs or machines.** The Cholesky and matrix products depend on the BLAS thread count and the device, and GPTQ's sequential error feedback amplifies the difference (about 1% PPL for W4 on Qwen2.5-0.5B). The job records `env.torch_threads`; compare runs only with the same value. The recorded CPU run gave W4A16 PPL 20.41 and the CUDA run 20.66.

**W2 or W4A8 gives a huge PPL.** W2 is experimental and collapses on small models (the comparison page uses a log axis for this reason). Use mixed precision for the layers the sensitivity analysis flags.

**My numbers differ from the documentation.** Docs numbers are for Qwen2.5-0.5B-Instruct, wikitext-2 test, 40 windows of 512 tokens, unless stated. PPL from `llama-perplexity` uses a different protocol (it scores the second half of each chunk) and is only comparable within its own table.

**`unsupported` in the export table.** The exporter's `check()` found a layer it cannot represent (for example 3-bit weights for AWQ, activation quantization with compressed-tensors integer weights, or an AWQ/SmoothQuant scale that cannot be folded). The reasons are listed in `report.json` under the recipe's `export`.

## Deployment

**A stage shows `skipped`.** The engine binary was not found or there is no CUDA device (for vLLM). Set `MQDE_VLLM_BIN`, `MQDE_LLAMA_SERVER_BIN`, `MQDE_LLAMACPP_BIN` or build into the `build/...` directories listed in [Installation](get_started/installation.md).

**vLLM rejects the GPTQ checkpoint.** vLLM 0.30.0 refuses asymmetric 4-bit GPTQ (`bits=4, sym=False`), which the `w4a16` recipe produces. Export AWQ or compressed-tensors instead.

**vLLM fails to start (out of memory, KV cache too small).** The report carries a diagnosis taken from the vLLM log. Raise `deploy.vllm.gpu_memory_utilization` (default 0.3) if the GPU is free, lower `max_model_len`, or free the GPU.

**GGUF conversion fails: "llama.cpp source with convert_hf_to_gguf.py not found".** Set `MQDE_LLAMACPP_SRC` to a llama.cpp source tree.

**Q4_K shows "144 fallback layers".** Tensors whose row length is not a multiple of 256 fall back to Q5_0, as `llama-quantize` does; this is expected for Qwen2.5-0.5B.

**The estimator's maximum concurrency is much lower than vLLM's.** Use the `vllm` backend profile: it models vLLM's batched activations (ADR-0003) and lands within 0.83 to 0.95 of vLLM's own figure on the two models checked. The default `hf_transformers` profile charges per-sequence activations and is a conservative bound for batching engines.

**`mqde fit` speed for the VLA model is far too high.** The formula counts one weight read per step; a VLA step runs ten denoising passes and vision encoding. A local `mqde bench` measurement replaces the formula for the same hardware.

## Web UI

**The page shows old content after a rebuild.** Hard-refresh the browser, check that `web/dist` is newer than your change, and that the server started with the right `--web` directory.

**The UI is in Chinese.** English UI strings are not shipped yet; the [tour](user_guides/web_ui.md) translates each page.

**Can I expose the server to other machines?** Not safely: there is no authentication. Keep the default loopback bind and use an SSH tunnel.
