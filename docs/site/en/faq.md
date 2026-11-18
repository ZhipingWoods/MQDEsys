# FAQ and troubleshooting

## Installation and start-up

**`mqde: command not found`.** The binary is located at `<build-dir>/bin/mqde`; add that directory to `PATH` or invoke it by full path.

**CMake cannot find `nlohmann_json`, `yaml-cpp`, `fmt`, `CLI11` or `httplib`.** Install the dependencies (Homebrew, apt or conda-forge) and pass `-DCMAKE_PREFIX_PATH=<prefix>`. The `dev` preset resolves dependencies from `$CONDA_PREFIX`; activate the conda environment first or configure manually as described in [Installation](get_started/installation.md).

**The build fails on a warning.** The project builds with `-Wall -Wextra -Werror`, and a newer compiler may introduce a new warning. Please report the issue rather than disabling the flag.

**`ImportError` for `mqde._core`.** The extension is built into `python/mqde/` only when CMake is configured with `-DMQDE_WITH_PYTHON=ON` and the Python minor version matches the build (the file name carries the version, for example `cpython-312`). Rebuild after changing Python.

**The server returns HTTP 403 `model path is outside allowed roots`.** Start the server with `--allow <parent-of-model>` (repeatable).

**The server returns 503 `server started without a GPU pool`.** Start it with `--gpus <ids>`, or submit an LLM job with `deploy.device: cpu`. VLA, WM and EWM jobs require a GPU.

**`deploy.device=cpu is only supported for model.type=llm`.** Set `model.type: llm` in the job specification (the default is `vla`).

**The job list is empty after a restart, or a job remains in "running" state.** The server reconstructs state from each job's `events.jsonl`; a job that terminated with the server process retains its running state and must be resubmitted.

## Quantization results

**GPTQ results differ between runs or machines.** Cholesky factorization and matrix multiplication depend on BLAS thread count and device, and GPTQ's per-layer error feedback amplifies the difference (about 1% PPL at W4 on Qwen2.5-0.5B). The job records `env.torch_threads`; compare runs only when this value matches.

**PPL is unexpectedly high for W2 or W4A8.** W2 is an experimental recipe for very low bit widths and can degrade markedly on small models; the comparison page therefore uses a logarithmic axis. Use sensitivity analysis and mixed precision for sensitive layers.

**Documented numbers differ from local results.** Unless stated otherwise, documented numbers use Qwen2.5-0.5B-Instruct, the wikitext-2 test set and 40 windows of 512 tokens. `llama-perplexity` uses a different scoring protocol (only the second half of each chunk is scored) and is comparable only with its own results.

**An export entry reports `unsupported`.** The exporter's `check()` found a layer it cannot represent (for example 3-bit AWQ weights, compressed-tensors integer weights with quantized activations, or AWQ/SmoothQuant scaling that cannot be folded). The reason is recorded under the recipe's `export` field in `report.json`.

## Deployment

**A stage reports `skipped`.** The engine binary was not found, or no CUDA device is available (vLLM / SGLang). Set `MQDE_VLLM_BIN`, `MQDE_SGLANG_PYTHON`, `MQDE_LLAMA_SERVER_BIN` or `MQDE_LLAMACPP_BIN`, or build into the directories listed in [Installation](get_started/installation.md).

**vLLM rejects a GPTQ checkpoint.** vLLM 0.30.0 does not accept asymmetric 4-bit GPTQ (`bits=4, sym=False`), which is the format produced by the `w4a16` recipe. Use an AWQ or compressed-tensors export, or a symmetric recipe.

**vLLM fails to start (out of memory or KV cache too small).** The report includes diagnostics extracted from the vLLM log. Increase `deploy.vllm.gpu_memory_utilization` (default 0.3) when the GPU is idle, reduce `max_model_len`, or free device memory.

**GGUF conversion fails with `llama.cpp source with convert_hf_to_gguf.py not found`.** Point `MQDE_LLAMACPP_SRC` at the llama.cpp source directory.

**Q4_K reports "144 layer fallback".** Tensors whose row length is not a multiple of 256 fall back to Q5_0 following llama.cpp's rule; this is expected for Qwen2.5-0.5B.

**Estimated maximum concurrency is much lower than vLLM's figure.** Select the `vllm` backend profile, which accounts for activations per batch; on the models checked it lands within 0.83 to 0.95 of vLLM's reported value. The default `hf_transformers` profile accounts for activations per sequence and therefore provides a conservative lower bound.

**`mqde fit` overestimates VLA throughput.** The formula assumes one weight read per step, while a VLA step includes multiple denoising passes and visual encoding. Use the measured value from `mqde bench` on the same hardware.

## Web UI

**The page is stale after a rebuild.** Force-refresh the browser, confirm that `web/dist` is newer than the change, and verify the `--web` directory passed to the server.

**Interface language.** The web workspace is currently presented in Chinese; see the [UI guide](user_guides/web_ui.md) for the page structure.

**Can the server be exposed to other machines?** This is not recommended. The server does not enable authentication; keep it bound to loopback and use an SSH tunnel.
