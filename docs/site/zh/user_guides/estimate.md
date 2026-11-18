# 资源预估与可部署性

资源预估提供命令行、REST 接口与 Web 界面三种使用方式。性能指标为基于公式的估计，存在本机实测时以实测为准。

本页的所有内容都只读取模型的 `config.json` 和 safetensors *头部*（即使是远程模型也只有几 KiB），从不读取权重。模型可以是本地目录，也可以是 `hf:<org>/<name>[@revision]`。

## 查看模型

```bash
mqde model inspect <MODEL_DIR>            # 表格
mqde --json model inspect <MODEL_DIR>     # JSON；加 --tensors 列出逐张量信息
```

输出包括架构族（与 `data/model_families/*.yaml` 匹配）、参数量、各组件的张量数与可量化层数，以及注意力布局（全注意力层与线性注意力层、KV 头数、head 维度）。

## `mqde estimate`

```bash
mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --hw nvidia/rtx4090 --ctx 4096 --conc 1,8,64
```

| 参数 | 含义 |
|---|---|
| `-m, --model`（必填） | 本地目录或 `hf:org/name[@rev]` |
| `-r, --recipe` | 逗号分隔的配方（`mqde recipes` 可列出） |
| `--hw` | 硬件画像 id（`mqde hardware` 可列出） |
| `--backend` | 后端画像（默认 `hf_transformers`；另有 `vllm`） |
| `--conc` | 并发数；写成 `1,8,32` 这样的列表时还会输出曲线 |
| `--ctx` | 语言模型上下文 token 数 |
| `--images`、`--text-tokens`、`--resolution` | VLA / 视觉输入 |
| `--util` | 可用显存比例 |
| `--also-quantize` | 额外量化的模态，例如 `action` |

对一个与 Qwen2.5-0.5B 同形状的模型，在 RTX 4090 画像上、4096 上下文时，每个配方的输出如下：

```text
== W4A16 on GeForce RTX 4090 (sm_89)  [W: INT4 asym g128 mse | A: FP16/BF16 | KV: FP16/BF16]
   weights 437.08 MiB (qmeta 6.67 MiB) | KV+state 48.00 MiB (12.00 KiB /token) | activation 293.58 MiB | runtime 512.00 MiB
   PEAK 1.30 GiB / capacity 21.59 GiB  ->  DEPLOYABLE   max concurrency 58
```

怎么读：

- **weights / qmeta**：按配方计算的每个张量字节数，加上 scale 与零点（qmeta）。架构模板要求保持高精度的层（embedding、`lm_head` 等）按原始大小计。
- **KV+state**：KV Cache 只统计全注意力层；线性注意力层每条序列持有恒定大小的循环状态与卷积状态。
- **activation、runtime**：经验激活峰值，以及后端画像（`data/backends/*.yaml`）给出的固定开销和分配器系数。
- **capacity** 是设备显存乘以可用比例（界面默认 0.9；23.99 GiB 变成 21.59 GiB）；**PEAK** 是所请求并发下的总和；**max concurrency** 是仍能装下的最大并发。
- **latency floor（延迟下界）**：`max(FLOPs / (峰值算力 × MFU), 读取字节数 / (带宽 × eta))`，不含 kernel 启动开销。它是下界，不是预测值。

```{warning}
后端画像带有 `activation_model`。默认的 `per_seq` 按 Hugging Face eager 的做法，给每条并发序列都计一份激活内存。`vllm` 画像用 `batched`（ADR-0003）：vLLM 把所有序列拼成一个最多 `max_batched_tokens` 个 token 的批，所以激活内存只按 `min(上下文 × 并发, max_batched_tokens)` 个 token 计一次。对照 vLLM 0.30 自己的启动报告，没有任何拟合系数：`vllm` 画像在 Qwen2.5-7B（1k、8k、32k 上下文）上预估的最大并发是 vLLM 的 0.83 到 0.95，在 Qwen2.5-0.5B 上是 0.92 到 0.94；旧的 `per_seq` 在小模型上偏低 7 到 8.5 倍，在 7B 的 32k 上下文上连一条序列都放不下。目前的证据是一种 GPU、一个引擎版本、两个模型；`max_num_seqs` 没有建模，其他批处理引擎没有数据。
```

相关命令：`mqde recipes`、`mqde hardware`、`mqde probe`（本机），以及 `mqde claim <MODEL_DIR> -H <hw> --min-tps 1`（保守与乐观的吞吐上下界，以及仍满足阈值的最大上下文）。

## `mqde fit`：能不能部署，部署得多好

`mqde fit` 把一批本地模型对照目标系统排序，并为每个模型选出装得下的最优配方。

```bash
mqde fit <LIBRARY_DIR_OR_MODEL_DIR>... --hw nvidia/rtx4090 --use-case chat --ctx 4096 --conc 1 --sort score
```

| 参数 | 含义 |
|---|---|
| `models...`（必填） | 模型目录或模型库根目录（子目录里含 `config.json` 的目录） |
| `--hw` | 硬件画像 id |
| `--gpus` | 相同 GPU 的数量 |
| `--ram` | 系统内存 GiB（默认自动探测） |
| `--use-case` | `auto`、`general`、`chat`、`coding`、`reasoning`、`multimodal`、`vla` |
| `--ctx`、`--conc` | 上下文 token 数、并发序列数 |
| `--quant` | 强制使用某个配方，而不是选最优的 |
| `--sort` | `score`、`tps`、`params`、`mem` |

每一行包含适配等级（**perfect、good、marginal、too tight**）、运行模式（GPU、MoE 卸载、CPU 卸载……）、所选量化、显存、估计速度，以及由质量、速度、适配度和上下文构成的 0-100 分。算法和全部常数沿用 llmfit，放在 `data/fit/fit.yaml`；改动任何一个常数都会改变所有结论，所以按接口变更来评审。

速度附带置信度标签：`estimated`（带宽屋顶线），或在 `data/benchmarks/` 里存在相同硬件的实测时为 `measured_local` / `measured_community`（由 `mqde bench` 写入，目前只支持 VLA 模型）。一个已知缺口：对 VLA 模型，公式高估了实测步速约 10.6 倍（实测 4.41 steps/s），所以不要相信 VLA 的公式速度。

Web 页面 [部署预评估](web_ui.md#部署预评估) 展示同样的分析，并带一个跨硬件规划（`/api/v1/fit:plan`），列出每个硬件画像上的最佳配方。

## 硬件与后端画像

画像是 YAML：`data/hardware_db/<vendor>/<id>.yaml`（显存、带宽、各精度峰值算力、原生支持格式）和 `data/backends/<name>.yaml`。已有 NVIDIA 画像：`rtx4090`、`a100-80g`、`h100-sxm`、`jetson-agx-orin-32g/64g`、`jetson-orin-nx-16g`、`jetson-orin-nano-8g`、`jetson-agx-thor`、`dgx-spark-gb10`，以及 AMD MI300X、Intel Gaudi 2、MetaX C500、Ascend 910B。硬件页可卡片式选择这些画像并直接带入新建作业。`mqde probe` 通过 NVML（动态加载）探测 NVIDIA GPU 和主机 CPU；没有匹配画像的机器上，设备只是不显示画像。
