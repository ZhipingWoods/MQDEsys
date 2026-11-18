# Quantization recipes and algorithms

A job combines a **recipe** (the data format for weights, activations and KV cache) with an **algorithm** (how quantized values are selected). Both are data-driven: recipes live in `data/recipes/*.yaml` and algorithms in `data/algorithms.yaml`.

## Recipes

`mqde recipes` lists the built-in recipes:

| Recipe | Weights | Activations | Default algorithm |
|---|---|---|---|
| `bf16` | none (baseline) | none | none |
| `w8a16` | INT8 symmetric, per-channel, min/max | 16-bit | rtn |
| `fp8` | FP8 E4M3, per-channel | FP8 E4M3, dynamic per-token | rtn |
| `w4a16` | INT4 asymmetric, per-group (128), MSE range search | 16-bit | gptq |
| `w4a8` | INT4 symmetric, per-group (128), MSE | INT8 symmetric, dynamic per-token | smoothquant+gptq |
| `w2a16` | INT2 asymmetric, group 64, MSE | 16-bit | gptq |
| `w2a8` | INT2 asymmetric, group 64, MSE | INT8 dynamic per-token | gptq |

Built-in recipes cover 8, 4 and 2-bit weights. Per-layer `mixed_precision` overrides can set integer layers to any integer bit width; the GPTQ exporter accepts 2, 3, 4 and 8 bit. FP8 recipes quantize both weights and activations. Native FP8 kernels require sm_89 or newer, or MI300 architecture.

W4A8, W2A8 and FP8 use dynamic per-token activation quantization by default. For static per-tensor activation scales, set `quant.act_static` to `minmax`, `ema` or `percentile` (CLI `--act-static`).

## Algorithms

`GET /api/v1/algorithms` or `data/algorithms.yaml` lists all algorithms; the server rejects unknown names.

| Name | Kind | Calibration | Description |
|---|---|---|---|
| `rtn` | quantizer | no | Round-to-nearest using the recipe's range search (min/max or MSE). |
| `gptq` | quantizer | yes | Per-layer second-order (Hessian) column-wise error compensation on captured FP activations (`block_size 128`, `damp 0.01`). |
| `hqq` | quantizer | no | Half-Quadratic Quantization: data-free proximal optimization of scale/zero, suited to 2-bit cases without calibration data. |
| `awq` | transform | yes | Searches per-input-channel scaling (20-step grid) to minimize output error, followed by RTN. |
| `smoothquant` | transform | yes | Migrates activation outliers into weights (`s = max|X|^a / max|W|^(1-a)`, `alpha 0.5`), primarily for W4A8. |
| `awq+gptq`, `smoothquant+gptq`, `awq+hqq` | combo | yes | Applies the transform first, then the corresponding quantizer to the scaled weights. |

Composition permits at most one transform (`awq` or `smoothquant`) joined with `+` to at most one quantizer (`rtn`, `gptq` or `hqq`). `algorithm: recipe` (the default) uses the recipe's default algorithm.

Blocks are quantized sequentially; the calibration input for block *i* is the output of quantized blocks 0..i-1. Calibration data comes from wikitext-2 train (`calibration.n_samples` windows of `seq_len` tokens). AWQ/SmoothQuant scaling is shared by layers reading the same input (q/k/v, gate/up) and folded into the preceding normalization or projection layer, so the model remains mathematically equivalent in floating point and can be exported (see [Export](export.md)). In grouped-query attention (GQA) models, the input width of `o_proj` differs from the output width of `v_proj`, so `o_proj` is not smoothed.

## Mixed precision

`quant.mixed_precision.rules` is an ordered list; the first matching rule applies. A rule contains `match` and either `bits` or `skip: true`:

- `match: "block:3"`, `"block:0-2"` or `"block:-1"` selects decoder blocks (negative indices count from the end);
- other `match` values are regular expressions searched in module paths, for example `"mlp\\.down_proj$"`.

```yaml
quant:
  recipe: [w4a16]
  algorithm: gptq
  mixed_precision:
    rules:
      - {match: "block:0", bits: 8}
      - {match: "mlp\\.down_proj$", bits: 8}
      - {match: "lm_head", skip: true}
```

Per-layer overrides carry into export: GPTQ writes GPTQModel's `dynamic` config, AWQ writes `modules_to_not_convert`, and compressed-tensors writes additional `config_groups`.

**Sensitivity analysis** (`analysis.sensitivity: block | type`) restores groups of layers to floating point one at a time and measures the recovered PPL degradation. The report ranks the groups and, within the weight-memory budget, suggests a rule expected to recover 50% of the degradation. The suggestion adds the independent effect of each group and is therefore optimistic; validate it with a new job. The web UI can generate that validation job directly from the suggestion.

## KV cache quantization

`quant.kv` fake-quantizes the keys and values entering attention to assess accuracy. The actual KV cache data type in the inference engine is determined by the engine's own settings.

```yaml
quant:
  kv: {format: int, bits: 8, symmetric: false, method: kivi}
```

`method: kivi` quantizes keys per-channel and values per-token; `method: naive` uses the same granularity for both (`granularity: per_token | per_head | per_channel`). Formats are `int` with `bits`, or `fp8_e4m3` (CLI `--kv int8`, `--kv int4`, `--kv fp8`). On Qwen2.5-0.5B, INT8 KIVI changes PPL by about -0.01% to -0.03% (W4A16 without KV quantization: 20.411; with KV quantization: 20.408); naive per-token key quantization performs worse because Qwen2 keys contain outlier channels.

## Evaluation positions

Each recipe reports PPL at up to three positions: **pretrain** (original model), **fake_quant_wo_kv** (quantized weights and activations) and **fake_quant** (adding KV cache quantization). Export adds **real** (weights read back from the checkpoint), and deployment adds **served** (the served model). All positions use the same 40 windows of 512 tokens from the wikitext-2 test set by default, so results within a job are directly comparable. PPL from llama.cpp's own tools uses a different protocol and is comparable only within its own table.
