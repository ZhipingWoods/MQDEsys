# M4 reference table (small tier) — Qwen2.5-0.5B-Instruct on NVIDIA GeForce RTX 4090

Jobs: 20261013-175450-104b21, 20261013-175450-ebe7f8, 20261013-175450-13b73a. wikitext-2 test, 40 windows × 512 tokens; unquantized PPL 18.103.

Scope: Qwen2.5-0.5B-Instruct only (one model of ARCHITECTURE §11.3's reference set). This page does not claim M4 acceptance, which needs the whole set.

## Summary

- Accuracy against the ARCHITECTURE §11.4 thresholds: **0 passed, 3 failed**, 2 recorded only (no threshold).
- Weight-memory estimate within 10% of what the engine reports: **4 within, 2 outside** (of 6 comparable).

## Accuracy

| Recipe | Algorithm | PPL (fake-quant W+A) | Δ vs unquantized | Threshold | Verdict | PPL of the exported weights |
|---|---|---|---|---|---|---|
| FP8 | rtn | 18.333 | +1.27% | ≤ 1% | FAIL | 18.333 |
| W4A16 | gptq | 20.655 | +14.10% | ≤ 5% | FAIL | 20.656 |
| W4A8 | smoothquant+gptq | 23.559 | +30.14% | ≤ 8% | FAIL | – |
| W2A16 | gptq | 2848.778 | +15636.09% | record only | INFO | – |
| W2A8 | gptq | 2141.751 | +11730.61% | record only | INFO | – |

## Deployment: vLLM and SGLang

| Recipe | Export | Backend | Status | Served PPL | vs the exported weights | tok/s | TTFT p50 | Weights GiB | Kernel |
|---|---|---|---|---|---|---|---|---|---|
| FP8 | compressed-tensors | vLLM | succeeded | 18.382 | +0.27% | 200 | 61 ms | 0.59 | CutlassFP8ScaledMMLinearKernel |
| FP8 | compressed-tensors | SGLang | succeeded | 18.352 | +0.10% | 1889 | 25 ms | 0.67 | – |
| W4A16 | awq | vLLM | succeeded | 20.658 | +0.01% | 232 | 55 ms | 0.43 | MarlinLinearKernel |
| W4A16 | awq | SGLang | succeeded | 20.657 | +0.00% | 1900 | 22 ms | 0.46 | – |
| W4A16 | compressed-tensors | vLLM | succeeded | 20.659 | +0.01% | 229 | 55 ms | 0.43 | MarlinLinearKernel |
| W4A16 | compressed-tensors | SGLang | succeeded | 20.656 | +0.00% | 1883 | 24 ms | 0.48 | – |

Served PPL comes from the engine's own prompt logprobs on the same windows as the exported weights; "vs the exported weights" is only given when the evaluation protocol matches.

**tok/s and TTFT are not comparable across engines.** The vLLM stage runs `enforce_eager=True` (no CUDA graphs) and the SGLang stage with CUDA graphs on, so most of a gap between them is that configuration, not the engines (T-0704). Compare rows of the same engine.

## llama.cpp (GGUF; independent of the recipes)

GGUF re-quantizes the original FP weights with llama.cpp's own numerics, so it does not belong to any recipe row above.

Protocol: llama-perplexity ctx 512, 40 chunks, scores the second half of each chunk; backend CUDA.

| Type | Size | PPL | vs F16 | pp512 tok/s | tg64 tok/s |
|---|---|---|---|---|---|
| f16 | 948 MiB | 13.894 | – | 55961 | 571.8 |
| q8_0 | 628 MiB | 13.934 | +0.28% | 48989 | 677.3 |
| q4_k | 488 MiB | 14.771 | +6.31% | 48599 | 740.3 |
| q2_k | 434 MiB | 27.072 | +94.84% | 46433 | 775.0 |

## Memory estimate vs measured

Weights only: vLLM / SGLang pre-allocate the KV cache to their memory fraction, so total peak memory cannot be compared. The engine reports weight size in GiB rounded to 0.01, so ±1% is rounding. The estimator has a vLLM profile only; SGLang is compared against it.

| Recipe | Export | Backend | Estimated GiB | Measured GiB | Error | Verdict |
|---|---|---|---|---|---|---|
| FP8 | compressed-tensors | vLLM | 0.59 | 0.59 | -0.3% | PASS |
| FP8 | compressed-tensors | SGLang | 0.59 | 0.67 | -12.2% | FAIL |
| W4A16 | awq | vLLM | 0.43 | 0.43 | -0.7% | PASS |
| W4A16 | awq | SGLang | 0.43 | 0.46 | -7.2% | PASS |
| W4A16 | compressed-tensors | vLLM | 0.43 | 0.43 | -0.7% | PASS |
| W4A16 | compressed-tensors | SGLang | 0.43 | 0.48 | -11.1% | FAIL |

## Not deployable on vLLM / SGLang

| Recipe | Format | Why |
|---|---|---|
| W4A8 | – | integer weights with quantized activations (W4A8 / W8A8-int) are not exported yet (compressed-tensors has no int-act writer) |
| W2A16 | – | 2-bit weights are not supported by awq (4-bit only) or compressed-tensors (4 / 8-bit) |
| W2A8 | – | 2-bit weights are not supported by awq (4-bit only) or compressed-tensors (4 / 8-bit) |

## Known issues from this run

Tracked here because the acceptance (T-0808) asks that every missed threshold has an issue. None of these were "fixed" by moving a threshold.

1. **FP8 misses its 1% limit**: +1.27% PPL with plain RTN per-channel on a 0.5B model. A real result; the next thing to try is per-token activation scaling or keeping the most sensitive layers (T-R10 dynamic quantization does this for VLM jobs, not yet for LLM jobs).
2. **W4A16 (GPTQ) misses its 5% limit**: +14.10%. Qwen2.5-0.5B is far more sensitive than the 7B-class models the limit was written for; the exported AWQ / compressed-tensors checkpoints reproduce the fake-quant PPL to within 0.01%, so the loss is in the quantization, not in the export or the engines. Needs the 7B reference models to tell "small models are fragile" from "our GPTQ is weak".
3. **W4A8 misses its 8% limit**: +30.14% (smoothquant+gptq), and it has no deployable container yet.
4. **SGLang's weight memory is above the estimator's vLLM profile by 7–12%** (FP8 0.67 vs 0.59 GiB, W4A16 0.46–0.48 vs 0.43): 2 of 6 comparisons are outside the 10% limit, both SGLang. There is no SGLang estimator profile yet (needs fields in the L2 `BackendProfile`, see T-0704), so this is the expected gap, not a vLLM-profile bug.
5. **Not covered**: the medium (7B / 8B), MoE / MLA and VLM entries of §11.3, other hardware (A100 / H100 / Apple / CPU), and the throughput of vLLM with CUDA graphs. This run therefore does not satisfy M4.
