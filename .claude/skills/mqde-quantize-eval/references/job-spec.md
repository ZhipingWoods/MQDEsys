# LLM job spec (YAML) — fields read by `LLMJobSpec.from_yaml`

Source of truth: `python/mqde/pipeline/llm_job.py`. Anything not listed here is ignored.

```yaml
metadata: {name: my-job}            # optional, default = file stem
model: {path: <MODEL_DIR>}          # required
quant:
  recipe: [w4a16, w8a16]            # required, string or list (names from `mqde recipes`)
  algorithm: gptq                   # rtn | gptq | awq | smoothquant | awq+gptq | smoothquant+gptq | recipe
  act_static: null                  # null = dynamic per token; minmax | ema | percentile = static per tensor
  kv: {format: int, bits: 8, symmetric: false, method: kivi}   # method: kivi | naive; or {method: ..., granularity: ...}
  mixed_precision: {rules: [{match: "block:0", bits: 8}]}      # rule keys: match, bits, skip
calibration: {n_samples: 32, seq_len: 512}
evaluation: {seq_len: 512, max_windows: 40}
export: {formats: [awq, compressed-tensors]}   # gptq | awq | compressed-tensors
deploy:
  device: cpu                       # cpu | cuda  (device used for quantization and the PPL evaluation)
  llamacpp:
    types: [q8_0, q4_k]             # f16 is always added as the reference
    gpu_layers: null                # llama.cpp -ngl: null = default, 0 = CPU, 99 = all layers on Metal/CUDA
    serve: {concurrency: 4, requests: 16, prompt_len: 256, n_predict: 64}   # llama-server load test; omit to use these defaults
  vllm: {}                          # present = serve every exported checkpoint with vLLM; keys below
                                    # max_model_len, gpu_memory_utilization (0.3), enforce_eager (true),
                                    # concurrency, requests, prompt_len, n_predict, ppl_windows
analysis: {sensitivity: block, sensitivity_windows: 8}      # block | type
output_dir: runs
```

Stages written to `events.jsonl` / `report.json.stages`: `validate`, `inspect`, `calib_prepare`, `eval_baseline`,
per recipe `quantize`, `eval_fake_wo_kv`, `eval_fake`, optionally `export`, `real_check`, `eval_real`, `deploy_vllm`,
`sensitivity`, then `deploy_llamacpp`, `report`.

Exporters: `gptq` (v1, bits 2/3/4/8), `awq` (4 bit, per-group), `compressed-tensors` (4/8-bit integer
`pack-quantized`, FP8 `float-quantized`). `Exporter.check()` explains why a layer cannot be exported; the job records
`status: unsupported` with the reasons instead of failing.
