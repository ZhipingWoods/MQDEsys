# Resource estimation and deployability

**Status: implemented** (CLI, REST, web). Performance numbers are formula-based estimates unless a local measurement exists; they are not a promise.

Everything on this page reads only `config.json` and the safetensors *headers* of a model (a few KiB even for a remote model), never the weights. A model can be a local directory or `hf:<org>/<name>[@revision]`.

## Inspect a model

```bash
mqde model inspect <MODEL_DIR>            # table
mqde --json model inspect <MODEL_DIR>     # JSON; add --tensors for the per-tensor listing
```

You get the architecture family (matched against `data/model_families/*.yaml`), parameter count, per-component tensor and quantizable-layer counts, and the attention layout (full-attention versus linear-attention layers, KV heads, head dimension).

## `mqde estimate`

```bash
mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --hw nvidia/rtx4090 --ctx 4096 --conc 1,8,64
```

| Flag | Meaning |
|---|---|
| `-m, --model` (required) | local directory or `hf:org/name[@rev]` |
| `-r, --recipe` | comma-separated recipes (`mqde recipes` lists them) |
| `--hw` | hardware profile id (`mqde hardware` lists them) |
| `--backend` | backend profile (`hf_transformers` by default; `vllm` also exists) |
| `--conc` | concurrency; a list such as `1,8,32` also produces a curve |
| `--ctx` | language context tokens |
| `--images`, `--text-tokens`, `--resolution` | VLA / vision inputs |
| `--util` | usable fraction of device memory |
| `--also-quantize` | extra modalities to quantize, for example `action` |

Output per recipe, for a Qwen2.5-0.5B-shaped model at 4096 tokens on the RTX 4090 profile:

```text
== W4A16 on GeForce RTX 4090 (sm_89)  [W: INT4 asym g128 mse | A: FP16/BF16 | KV: FP16/BF16]
   weights 437.08 MiB (qmeta 6.67 MiB) | KV+state 48.00 MiB (12.00 KiB /token) | activation 293.58 MiB | runtime 512.00 MiB
   PEAK 1.30 GiB / capacity 21.59 GiB  ->  DEPLOYABLE   max concurrency 58
```

How to read it:

- **weights / qmeta**: bytes per tensor under the recipe, plus scales and zero points (qmeta). Layers a family template keeps in high precision (embeddings, `lm_head`, and so on) stay at original size.
- **KV+state**: KV cache counts full-attention layers only; linear-attention layers hold a constant recurrent and convolution state per sequence.
- **activation, runtime**: an empirical activation peak and the backend profile's fixed overhead and allocator factor (`data/backends/*.yaml`).
- **capacity** is device memory times the usable fraction (0.9 in the UI default; 23.99 GiB becomes 21.59 GiB); **PEAK** is the sum for the requested concurrency; **max concurrency** is the largest concurrency that still fits.
- **latency floor**: `max(FLOPs / (peak x MFU), bytes read / (bandwidth x eta))`; kernel launch overhead is not modeled. It is a floor, not a prediction.

```{warning}
The estimator charges every concurrent sequence its own activation memory, as Hugging Face eager execution does. For batching engines such as vLLM this makes `max concurrency` a conservative lower bound: on Qwen2.5-0.5B FP8 at 1024 context, vLLM reported 509 concurrent sequences where the estimate said 74. Weight memory agreed (0.59 GiB both). The fix is a recorded open decision (D-9 in `docs/PROGRESS.md`).
```

Related commands: `mqde recipes`, `mqde hardware`, `mqde probe` (this machine), and `mqde claim <MODEL_DIR> -H <hw> --min-tps 1` (conservative and optimistic throughput bounds, plus the longest context that still meets the threshold).

## `mqde fit`: is it deployable, and how well?

`mqde fit` ranks a library of local models against a target system and picks, for each model, the best recipe that fits.

```bash
mqde fit <LIBRARY_DIR_OR_MODEL_DIR>... --hw nvidia/rtx4090 --use-case chat --ctx 4096 --conc 1 --sort score
```

| Flag | Meaning |
|---|---|
| `models...` (required) | model directories or library roots (a directory whose children contain `config.json`) |
| `--hw` | hardware profile id |
| `--gpus` | number of identical GPUs |
| `--ram` | system RAM in GiB (default: probed) |
| `--use-case` | `auto`, `general`, `chat`, `coding`, `reasoning`, `multimodal`, `vla` |
| `--ctx`, `--conc` | context tokens, concurrent sequences |
| `--quant` | force a recipe instead of choosing the best that fits |
| `--sort` | `score`, `tps`, `params`, `mem` |

Each row has a fit level (**perfect, good, marginal, too tight**), a run mode (GPU, MoE offload, CPU offload ...), the chosen quantization, memory, estimated speed and a 0-100 score built from quality, speed, fit and context. The algorithm and all constants follow llmfit and live in `data/fit/fit.yaml`; changing a constant changes every verdict, so it is reviewed as an interface change.

Speed comes with a confidence label: `estimated` (bandwidth roofline) or `measured_local` / `measured_community` when `data/benchmarks/` contains a measurement for the same hardware (written by `mqde bench`, VLA models only for now). A known gap: for the VLA model the formula over-estimates the measured step rate by about 10.6x (4.41 steps/s measured), so do not trust formula speed for VLA.

The web page [Deployment pre-evaluation](web_ui.md#deployment-pre-evaluation) shows the same analysis plus a cross-hardware plan (`/api/v1/fit:plan`) that lists the best recipe on every hardware profile.

## Hardware and backend profiles

Profiles are YAML: `data/hardware_db/<vendor>/<id>.yaml` (memory, bandwidth, per-precision peak throughput, native formats) and `data/backends/<name>.yaml`. Shipped hardware: `nvidia/rtx4090`, `nvidia/a100-80g`, `nvidia/h100-sxm`, `nvidia/jetson-agx-orin-64g`. Only NVIDIA profiles exist; AMD, Apple GPU and Ascend profiles are planned (T-1103 to T-1105). `mqde probe` detects NVIDIA GPUs through NVML (loaded dynamically) and the host CPU; on a machine without a profile match the device simply shows no profile.
