# Export formats and consistency checks

Fake quantization evaluates accuracy but does not reduce model size: weights remain floating-point values on the quantization grid. Exporters convert the fake-quantized model into checkpoints that inference engines can load directly. Enable them with `export.formats` in the job YAML or `--export gptq,awq,compressed-tensors` on the command line. Each recipe is written to `<job-dir>/export/<recipe>-<format>/`, containing `model.safetensors`, `config.json` (with `quantization_config`), `quantize_config.json` (GPTQ/AWQ) and tokenizer files.

Before writing, every exporter runs `check()`. Layers that cannot be represented are reported in the job report as `status: unsupported` with the layer name and reason, rather than being exported incorrectly.

## Supported formats

| Format (`--export`) | Layout | Supported weights | Compatible engines |
|---|---|---|---|
| `gptq` | GPTQ v1: `qweight`, `qzeros`, `scales`, `g_idx` (`desc_act=false`) | INT 2/3/4/8 bit, per-group or per-channel, symmetric or asymmetric | vLLM, SGLang, transformers |
| `awq` | AWQ GEMM, `zero_point: true` | INT4, per-group | vLLM, SGLang, transformers |
| `compressed-tensors` | `pack-quantized` for INT4/INT8 (`weight_packed`, `weight_scale`, `weight_shape`, packed zero points); `float-quantized` for FP8 E4M3 | INT 4/8 weight-only; FP8 per-channel or per-tensor, with dynamic per-token or static per-tensor activation scales | vLLM |
| GGUF | F16 written by the llama.cpp converter, then `blk.*.weight` tensors re-encoded | `q8_0`, `q5_0`, `q4_0`, `iq4_nl`, `q4_k`, `q2_k` | llama.cpp |

Unsupported combinations include compressed-tensors integer weights with quantized activations (W4A8, W8A8-int), per-tensor integer scales, per-group FP8 scales, and layers whose AWQ/SmoothQuant scaling cannot fold into the preceding operator. GPTQ and AWQ checkpoints do not contain activation quantization; engines compute activations in FP16, and the job notes this in the report.

## Smoothing factor folding

AWQ and SmoothQuant produce one scaling factor per group of layers that read the same input. Exporters require this factor to be folded into the operator producing that input (RMSNorm weights, rows of `up_proj`, rows of `v_proj`). The transformation is exact in floating point, and inference requires no extra division. In grouped-query attention models, `o_proj` is not smoothed.

## Mixed-precision checkpoints

Per-layer bit-width overrides are written into each format's configuration: GPTQ uses GPTQModel's `dynamic` config, AWQ uses `modules_to_not_convert`, and compressed-tensors adds `config_groups` plus `ignore`. Layers kept in floating point (`skip: true`, `lm_head`) are listed explicitly.

## GGUF details

`convert_hf_to_gguf.py` from the llama.cpp source directory writes an F16 file (the directory is set with `MQDE_LLAMACPP_SRC`). Linear-layer tensors are then quantized by MQDEsys's own packer. On Qwen2.5-0.5B, the output of six quantization types (290 tensors each) matches `llama-quantize` byte for byte. Tensors whose row length is not divisible by the block size follow llama.cpp's fallback rules: Q4_K falls back to Q5_0, Q2_K to Q4_0; if still indivisible, F16 is retained. Embeddings, output heads and norms remain F16/F32. GGUF quantizes raw weights with llama.cpp's own numerics, independent of the job's recipe and algorithm, and is therefore compared against recipe results through PPL only.

## Marlin packing

`mqde.core.pack_marlin` and `unpack_marlin` implement the Marlin tile layout for 4/8-bit weights with 16-bit activations, ported from vLLM v0.30.0, and match vLLM's reference functions byte for byte across three shapes. Exporters do not write Marlin checkpoints directly; vLLM repacks GPTQ/AWQ/compressed-tensors weights into Marlin layout at load time. The project includes a byte-for-byte comparison test against vLLM's `gptq_marlin_repack`.

## Consistency verification

After export, the pipeline performs two checks:

1. **Authenticity check**: read the checkpoint back with the reference unpacker, reconstruct each layer's integer codes and zero points, and compare them with the fake-quantized model. Codes and zero points must be identical; any difference is a packing defect. Dequantized weights may differ only because scales are stored in FP16, reported as max-abs and relative L2 error.
2. **Real-weight re-evaluation**: substitute the dequantized real weights into the model and measure PPL again.

For Qwen2.5-0.5B-Instruct with W4A16 GPTQ: both the AWQ and compressed-tensors checkpoints are 448 MiB, all integer codes match, the relative L2 weight error is 2.0e-4, and real-weight PPL is 20.656 versus 20.655 for fake quantization (+0.00%). For the same model, FP8 compressed-tensors gives real-weight PPL 18.333 and served PPL 18.382 (+0.27%).

```{note}
vLLM 0.30.0 rejects asymmetric 4-bit GPTQ (`bits=4, sym=False`), which is the format produced by the `w4a16` recipe. This is a vLLM restriction, not an export defect. Use an AWQ or compressed-tensors export, or a symmetric recipe, when targeting vLLM.
```
