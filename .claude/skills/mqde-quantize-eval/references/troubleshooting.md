# Failures seen so far, with the cause

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: wikitext-2-raw-v1 <split> not found under .../datasets--wikitext` | the evaluation data must already be in the Hugging Face cache; nothing is downloaded | put the cache in place or set `HF_HUB_CACHE` / `HF_HOME` |
| `can't convert cuda:0 device type tensor to numpy` during export | fixed in `layer_codes` (2026-10-08); older checkouts only | update the repository |
| export `status: unsupported`, reason says the smoothing scale "must be folded into the previous op first" | `BlockwiseQuantizer(..., fold_scales=False)` (Python API only, default is True) leaves a per-layer divisor | keep the default `fold_scales=True`; GQA `o_proj` layers are skipped by the folding instead |
| export `unsupported`, "not an integer format" | FP8 recipe sent to `gptq` / `awq` | export FP8 with `compressed-tensors` |
| export `unsupported`, "per-tensor weight scales" | per-tensor granularity cannot be stored in GPTQ/AWQ | use per-group or per-channel |
| GPTQ PPL differs by ~1% between machines | BLAS thread count changes the last digits | compare runs with equal `env.torch_threads` |
| `unknown quantization algorithm` | typo, or a new algorithm not imported | names are in `data/algorithms.yaml` and `ALGO_REGISTRY` |
| model type not recognised, wrong components quantized | no matching family template | add `data/model_families/<family>.yaml` (see mqde-extend) |
