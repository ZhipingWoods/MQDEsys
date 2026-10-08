# Export formats and the fake/real consistency check

**Status: partial.** GPTQ, AWQ, compressed-tensors and GGUF export work and are verified; SGLang and TensorRT-LLM targets are planned.

Fake quantization evaluates accuracy but does not shrink anything: weights stay in floating point on the quantization grid. An **exporter** turns the fake-quantized model into a checkpoint that a serving engine loads. Enable them with `export.formats` in the job YAML or `--export gptq,awq,compressed-tensors`. Each recipe writes `<job-dir>/export/<recipe>-<format>/` containing `model.safetensors`, `config.json` (with `quantization_config`), `quantize_config.json` (GPTQ/AWQ) and the tokenizer files.

Before writing, each exporter runs `check()`. A layer it cannot represent is *reported* (job report: `status: unsupported`, with the layer and reason) rather than exported wrongly.

## Formats

| Format (`--export`) | Layout | Supported weights | Loads in |
|---|---|---|---|
| `gptq` | GPTQ v1: `qweight`, `qzeros`, `scales`, `g_idx` (`desc_act=false`) | INT 2, 3, 4, 8 bit, per-group or per-channel, symmetric or asymmetric | vLLM, SGLang, transformers (by design) |
| `awq` | AWQ GEMM, `zero_point: true` | INT4, per-group only | vLLM (verified), SGLang, transformers (by design) |
| `compressed-tensors` | `pack-quantized` (`weight_packed`, `weight_scale`, `weight_shape`, packed zero points) for INT4/INT8; `float-quantized` for FP8 E4M3 | INT 4/8 weight-only; FP8 per-channel or per-tensor with dynamic per-token or static per-tensor activation scales | vLLM (verified) |
| GGUF (`--llamacpp` types) | llama.cpp's converter writes F16, then `blk.*.weight` tensors are re-encoded | `q8_0`, `q5_0`, `q4_0`, `iq4_nl`, `q4_k`, `q2_k` | llama.cpp |

Not exportable today: integer weights with quantized activations (W4A8, W8A8-int) in compressed-tensors, per-tensor integer scales, per-group FP8 scales, and layers whose AWQ/SmoothQuant scale could not be folded into the previous operation. GPTQ and AWQ checkpoints never contain activation quantization; the engine runs activations in fp16, and the job notes that.

### Smoothing scales are folded

AWQ and SmoothQuant produce one scale per group of layers that read the same input. The exporter requires them to have been folded into the operation that produces that input (RMSNorm weight, `up_proj` rows, `v_proj` rows), which is exact in floating point and costs no extra division at inference. In grouped-query-attention models `o_proj` is left unsmoothed.

### Mixed precision in checkpoints

Per-layer bit overrides are written as GPTQModel's `dynamic` config for GPTQ, as `modules_to_not_convert` for AWQ, and as extra `config_groups` plus `ignore` for compressed-tensors. Layers kept in FP (`skip: true`, `lm_head`) are listed as such.

### GGUF details

`convert_hf_to_gguf.py` from a llama.cpp source tree writes the F16 file (`MQDE_LLAMACPP_SRC` points at it). Linear tensors are then quantized with MQDEsys's own packer, whose output was compared byte for byte against `llama-quantize` on Qwen2.5-0.5B for all six types (290 tensors each). Tensors whose row length does not divide the block size follow llama.cpp's fallback: Q4_K falls back to Q5_0 and Q2_K to Q4_0; if that also does not divide, the tensor stays F16. Embeddings, the output head and norms stay F16/F32. GGUF quantizes the original weights with llama.cpp's numerics, independently of the job's recipes and algorithms, so compare it with the recipe results only through PPL.

### Marlin packing

`mqde.core.pack_marlin` / `unpack_marlin` implement the Marlin tile layout for 4- and 8-bit weights with 16-bit activations, ported from vLLM v0.30.0 and checked byte for byte against vLLM's reference functions for three shapes. The exporters do not write Marlin checkpoints; vLLM repacks GPTQ/AWQ/compressed-tensors weights into Marlin itself at load time (it selected `MarlinLinearKernel` for the 4-bit checkpoints below). A byte-for-byte check against `gptq_marlin_repack` of an installed vLLM is written but still pending.

## The fake/real consistency check (S10)

After each export the pipeline runs two stages:

1. `real_check`: read the checkpoint back with a reference unpacker, rebuild integer codes and zero points for every layer, and compare with the fake-quantized model. Codes and zero points must be **identical** (any difference is a packing bug); the dequantized weight may differ only through fp16 storage of the scales (reported as max-abs and relative L2).
2. `eval_real`: swap the dequantized real weights into the model and measure PPL again.

Result on Qwen2.5-0.5B-Instruct, W4A16 GPTQ (job 20261008-101629-62da92 in the UI): the AWQ and compressed-tensors checkpoints are 448 MiB each, all integer codes identical, weight relative L2 2.0e-4, and real-weight PPL 20.656 against fake-quantized 20.655 (+0.00%). For the FP8 compressed-tensors export of the same model the real-weight PPL is 18.333 and the vLLM-served PPL 18.382 (+0.27%).

```{note}
vLLM 0.30.0's validator rejects asymmetric 4-bit GPTQ (`bits=4, sym=False`), which is what the `w4a16` recipe produces. This is a vLLM constraint, not an export bug; the file is structurally correct. Use the AWQ or compressed-tensors export for vLLM, or a symmetric recipe.
```
