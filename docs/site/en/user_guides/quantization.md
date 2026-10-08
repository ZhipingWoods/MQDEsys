# Quantization recipes and algorithms

**Status: implemented** for LLM jobs (fake-quantization evaluation plus real export, see [Export formats](export.md)). Experimental recipes are marked.

A job combines a **recipe** (what format the weights, activations and KV cache take) with an **algorithm** (how the quantized values are chosen). Both are data-driven: recipes live in `data/recipes/*.yaml`, algorithms in `data/algorithms.yaml`.

## Recipes

`mqde recipes` prints the shipped ones:

| Recipe | Weights | Activations | Default algorithm | Maturity |
|---|---|---|---|---|
| `bf16` | none (baseline) | none | none | production |
| `w8a16` | INT8 symmetric, per-channel, min/max | 16-bit | rtn | production |
| `fp8` | FP8 E4M3, per-channel | FP8 E4M3, dynamic per-token | rtn | production (native kernels need sm_89+ / MI300) |
| `w4a16` | INT4 asymmetric, per-group (128), MSE range search | 16-bit | gptq | production |
| `w4a8` | INT4 symmetric, per-group (128), MSE | INT8 symmetric, dynamic per-token | smoothquant+gptq | production |
| `w2a16` | INT2 asymmetric, group 64, MSE | 16-bit | gptq | experimental |
| `w2a8` | INT2 asymmetric, group 64, MSE | INT8 dynamic per-token | gptq | experimental |

Weight bit widths are 8, 4 and 2 in the shipped recipes. Other widths: the per-layer `mixed_precision` override (below) changes the bit width of integer layers to any integer, and the GPTQ exporter accepts 2, 3, 4 and 8 bit; a W3, W5 or W6 recipe file is not shipped and is not covered by the verified runs. FP8 recipes quantize both weights and activations; there is no FP8 per-block recipe yet (T-0605 is partial).

Activations in W4A8, W2A8 and FP8 are quantized dynamically per token by default. For static per-tensor activation scales set `quant.act_static` to `minmax`, `ema` or `percentile` (CLI `--act-static`).

## Algorithms

`GET /api/v1/algorithms` or `data/algorithms.yaml` lists them; the server rejects unknown names.

| Name | Kind | Calibration | What it does |
|---|---|---|---|
| `rtn` | quantizer | no | Round-to-nearest using the recipe's range search (min/max or MSE). |
| `gptq` | quantizer | yes | Second-order column-wise error compensation per layer, on captured FP activations (`block_size 128`, `damp 0.01`). |
| `awq` | transform | yes | Searches per-input-channel scales (grid of 20) that minimize output error, then RTN. |
| `smoothquant` | transform | yes | Moves activation outliers into the weights (`s = max|X|^a / max|W|^(1-a)`, `alpha 0.5`); mainly for W4A8. |
| `awq+gptq`, `smoothquant+gptq` | combo | yes | Transform first, then GPTQ on the scaled weights. |

A combination is *at most one transform* (`awq` or `smoothquant`) joined with `+` to *at most one quantizer* (`rtn` or `gptq`). `algorithm: recipe` (the default) uses each recipe's default algorithm from the table above.

Blocks are quantized in order, and the calibration input of block *i* is the output of the already-quantized blocks 0..i-1. Calibration data is wikitext-2 train (`calibration.n_samples` windows of `seq_len` tokens). The AWQ/SmoothQuant scales are shared by the layers reading the same input (q/k/v, gate/up) and folded into the preceding normalization or projection, so the model stays mathematically unchanged in FP and can be exported (see [Export](export.md)); in grouped-query-attention models `o_proj` is not smoothed because its input width differs from `v_proj`'s output.

```{note}
AWQ and SmoothQuant here are the equivalent-transform part of each method plus RTN or GPTQ. The "transformed" evaluation position (perplexity after the transform, before quantization) is not implemented (T-0603 partial), and QuaRot is not implemented. The external adapters for GPTQModel or llm-compressor are planned (T-0609).
```

## Mixed precision

`quant.mixed_precision.rules` is an ordered list; the first matching rule wins. A rule has `match` and either `bits` or `skip: true`:

- `match: "block:3"`, `"block:0-2"`, `"block:-1"` selects decoder blocks (negative counts from the end);
- any other `match` is a regular expression searched in the module path, for example `"mlp\\.down_proj$"`.

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

Per-layer overrides are carried into exports: GPTQ via GPTQModel's `dynamic` config, AWQ via `modules_to_not_convert`, compressed-tensors via extra `config_groups`.

**Sensitivity analysis** (`analysis.sensitivity: block | type`) restores one group at a time to FP and measures how much of the PPL degradation it wins back; the report ranks groups and suggests rules that win back 50% of the degradation within a weight-memory budget. The suggestion adds single-group effects, so it is optimistic; check it by running a new job. In the UI a button creates that job from the suggestion.

## KV cache quantization

`quant.kv` quantizes the keys and values entering attention (fake-quantization, so it measures accuracy; real KV cache dtype in an engine is that engine's setting):

```yaml
quant:
  kv: {format: int, bits: 8, symmetric: false, method: kivi}
```

`method: kivi` quantizes keys per channel and values per token; `method: naive` uses one granularity for both (`granularity: per_token | per_head | per_channel`). Formats: `int` with `bits`, or `fp8_e4m3` (CLI `--kv int8`, `--kv int4`, `--kv fp8`). On Qwen2.5-0.5B, INT8 KIVI changes PPL by about -0.01% to -0.03%, i.e. nothing measurable (job c80eda: W4A16 20.411 without and 20.408 with KV quantization); naive per-token key quantization is worse because Qwen2 keys have outlier channels.

## Reading the evaluation positions

Each recipe reports PPL at up to three places: **pretrain** (the original model), **fake_quant_wo_kv** (quantized weights and activations), **fake_quant** (the same plus KV cache quantization). Exports add **real** (checkpoint weights read back) and deployment adds **served** PPL. All use the same 40 windows of 512 tokens from the wikitext-2 test split by default, so the positions are directly comparable within a job; PPL from llama.cpp's own tool uses a different protocol and is only comparable inside its table.
