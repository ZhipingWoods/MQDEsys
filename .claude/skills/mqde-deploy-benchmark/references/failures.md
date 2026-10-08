# Backend startup failures and what the adapters report

`ServerBackend.start()` raises with the process exit code, a diagnosis recognised from the server log, and the last
1500 characters of the log. The full log is written to the path you gave (`runs/.../vllm-<ckpt>.log`).

## vLLM (`mqde.backend.vllm_server.DIAGNOSES`)

| Log text | Diagnosis | What to do |
|---|---|---|
| `Unsupported quantization config` | quantization config not supported by this vLLM version (seen: asymmetric GPTQ) | export AWQ or compressed-tensors, or use a newer vLLM |
| `is less than desired GPU memory utilization` | other processes hold GPU memory | lower `gpu_memory_utilization`, or pick another card |
| `No available memory for the cache blocks` | nothing left for KV cache | raise `gpu_memory_utilization` or lower `max_model_len` |
| `larger than the maximum number of tokens that can be stored in KV cache` | `max_model_len` > KV capacity | lower `max_model_len` or raise utilisation |
| `CUDA out of memory`, `OutOfMemoryError` | GPU out of memory while loading or profiling | lower `max_model_len`, `max_num_seqs`, utilisation |
| `No CUDA GPUs are available` | no visible device | check `CUDA_VISIBLE_DEVICES` and the driver |
| `Address already in use` | port taken | none: the adapter picks a free port unless one is forced |
| `Invalid repository ID or local directory` | model path not found | check the path |
| `does not appear to have a file named` | checkpoint directory misses config / weights / tokenizer | re-export with `source_dir` so tokenizer files are copied |

Startup takes about 30 s for a 0.5B model with `enforce_eager=True`; the adapter waits up to 900 s.
Environment notes from the real runs: vLLM 0.30 needs transformers >= 5.10 and its own Python 3.12 environment; the
quantization kernel used is printed as `Using MarlinLinearKernel for ...` / `Selected CutlassFP8ScaledMMLinearKernel
for ...` and is returned as `engine_info()["quant_kernel"]`.

## llama-server (`mqde.backend.llamacpp_server.DIAGNOSES`)

| Log text | Diagnosis |
|---|---|
| `failed to open` | model file not found or unreadable |
| `unknown model architecture` | architecture not supported by this llama.cpp build |
| `failed to allocate`, `unable to allocate`, `out of memory` | out of memory (lower `-c` / parallel slots) |
| `invalid magic`, `tensor data is not within` | not a GGUF, or truncated |
| `couldn't bind` | port in use |

`llama-quantize --pure` of this repository's llama.cpp fork aborts on rows that are not a multiple of the block size
(it skips the shape fallback). Reference files for comparison need `--tensor-type "blk\\..*=<type>"` plus
`--token-embedding-type f16 --output-tensor-type f16` instead. Our `requantize()` follows llama-quantize's own
fallbacks (Q4_K → Q5_0, Q2_K → Q4_0, otherwise the tensor stays F16/F32) and is byte-identical to it.

## Load tester

Failed requests are counted (`failed`, `errors`) and never raise. If `ok` is 0, read `errors[0]`; HTTP 400 with
"maximum context length" means `prompt_len + n_predict` exceeds the server's `max_model_len`.
