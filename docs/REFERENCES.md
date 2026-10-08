# 参考架构对照（REFERENCES）

> 各子系统**必须模仿**的开源项目，以及“模仿到什么程度、落在哪个模块”。文档级别 **L2**。
> 源码在 `3rdpart/`（git 子模块，只记录上游仓库与固定提交；开发机 `/cache/wuzhiping/MQDEsys/3rdpart/` 已检出）。只读参考，**不得修改、不得直接 import 进产品代码**；需要复用代码时按许可证单独引入并走 L3。

| 子系统 | 参考项目 | 许可证 | 我们的落点 |
|--------|---------|--------|-----------|
| 部署预评估（能不能跑、跑多快、选哪个量化） | **llmfit** (Rust) | MIT | `cpp/src/fit/`（C++ 重写算法与常数）+ Web「部署预评估」页 |
| 量化方法与流程 | **GPTQModel**（主）+ **unsloth**（动态量化思想） | Apache-2.0 / Apache-2.0 | `python/mqde/quant/`（looper / processor / module_tree）+ C++ `quant` 数值核心 |
| 权重/网络结构节点展示 | **netron** | MIT | Web「模型结构」页（C++ 出图数据，前端渲染） |
| 精度评测 + 性能压测 | **evalscope** | Apache-2.0 | `python/mqde/eval/`（registry / adapter / report）+ C++ `bench`（perf） |
| 多端 / 多硬件架构部署 | **FlagTree** | MIT | `cpp/src/hal/` 后端插件 + `data/backends/` |
| 前端视觉风格 | styles.refero.design 的 Raycast / Linear 类“深色指挥台” + evalscope `DESIGN.md` 的 token 体系 | — | `web/src/styles.css` 设计 token |

---

## 1. llmfit → `fit` 模块

**要照搬的**（常数已从 `llmfit-core/src/{fit,models}.rs` 核对）：

- **显存公式**：`mem = params_B × bpp(quant) + KV + 0.5 GB`。KV = `2 × 全注意力层数 × kv_heads × head_dim × ctx × bytes`。我们额外加上线性注意力状态（llmfit 没有）。
- **每参数字节（显存）**：F32 4.0、F16/BF16 2.0、Q8_0 1.05、Q6_K 0.80、Q5_K_M 0.68、Q4_K_M 0.58、Q3_K_M 0.48、Q2_K 0.37、AWQ/GPTQ-4bit 0.5、8-bit 1.0。本系统的配方映射为：W8A16 1.0、FP8 1.0、W4A16/W4A8 0.5 + 组元数据、W2 0.25 + 组元数据。
- **Fit 等级**：`ratio = mem_required / mem_available`，≤0.60 Perfect、≤0.85 Good、≤0.98 Marginal，否则 TooTight。CPU 路径最高只给到 Good。
- **速度**：`tps = BW / (active_params_B × bytes_per_param) × 0.55 × run_mode_factor`。
  - run_mode_factor：GPU 1.0、TP 0.9、MoE offload 0.8、CPU offload 0.5、CPU only 0.3。
  - 不知道带宽时用经验常数 K：CUDA 220、Metal 160、ROCm 180、Vulkan 150、CPU-ARM 90、CPU-x86 70、Ascend 390。
  - Prefill 按算力受限估算，利用率 0.35。
- **评分**（0–100）：
  - speed = `tps / target × 100`。
  - fit = `100·exp(−0.5 z²)`，其中 `z = max(0, (ratio−0.70)/0.20)`。
  - context = 100 / 70 / 30 三档。
  - 总分按用途加权（Quality/Speed/Fit/Context）：General .45/.30/.15/.10，Chat .40/.35/.15/.10，Multimodal .50/.20/.15/.15。
  - **VLA 用途是我们新增的**：.40/.35/.15/.10，速度目标按“决策步/秒”算，10 Hz 记满分。
- **运行模式选择**：显存放得下选 GPU；MoE 放不下但活跃参数放得下选 MoeOffload；内存放得下选 CpuOffload；都不行则 CpuOnly。
- **量化选择**：沿量化层级从高精度往低精度走，选第一个“放得下”的（`choose_quant`）。
- **置信度**：`measured_local / measured_community / calibrated / estimated`。有实测时用实测中位数。社区基准存为 `data/benchmarks/community/<hw>/*.json`。
- **API 形状**：`GET /api/v1/system`、`GET /api/v1/fit/models?use_case=&sort=&min_fit=`、`GET /api/v1/fit/models/{name}`、`POST /api/v1/fit/plan`。每行字段对齐 llmfit：`fit_level, run_mode, best_quant, memory_required_gb, memory_available_gb, utilization_pct, estimated_tps, ttft_ms, score, score_components, estimate_confidence, notes`。
- **UI**：SystemPanel（本机硬件）、FilterBar（用途、fit、运行模式、量化、搜索、排序）、ModelTable（上面这些列）、DetailPanel、ComparePanel。

**不照搬的**：15k 条 HF 模型库。我们的模型库是本地 `hub` 目录扫描结果加上 `data/models/catalog.yaml`（手工维护的常用模型元数据，不下载权重）。

## 2. GPTQModel + unsloth → `quant`

**GPTQModel 的结构要照搬**：

- **QuantizeConfig**：`bits, group_size, sym, desc_act(act_order), damp_percent=0.01, damp_auto_increment=0.0015, static_groups, mse, true_sequential, lm_head, method(rtn|gptq|awq|fp8), format, dynamic`。
  - `dynamic` 是 `{regex: {bits, group_size, …} | "-" 表示跳过}` 形式的按层覆盖。它同时承载 **unsloth 动态量化**的思想：敏感层保持高精度。
- **module_tree**：每个架构一个定义，列出块路径、`#` 层号、子集顺序 `:N`、共享输入 `in=x`、不量化 `:!`。MiniCPM-VLA 需要三棵树：视觉塔、语言模型（区分全注意力层和线性注意力层）、DiT 动作头（可选）。
- **Looper / Processor 流程**：
  1. 捕获第 0 层输入。
  2. 逐层、逐子集做 hook，`add_batch` 累积 `H += XᵀX`。
  3. Processor 量化该子集。
  4. true_sequential：用量化后的层重新前向，得到下一层的输入。
- **GPTQ 算法**：
  - Cholesky 失败时按 `damp_auto_increment` 自增阻尼。
  - 支持 act_order / static_groups。
  - 返回 `Q, scale, zero, g_idx, loss, damp, nsamples, time`。
- **每模块日志表**（原样照搬列名）：`process, layer, module, loss, samples, damp, time, fwd_time`。它直接作为前端量化过程表的数据。
- **后端 / 格式**：导出 `quantize_config.json` 和 GPTQ v1/v2 布局。布局为 `qweight int32 [in/(32/bits), out]`、`qzeros int32 [in/g, out/(32/bits)]`、`scales fp16 [in/g, out]`、`g_idx`。

**unsloth 要照搬的**：

- **Dynamic quant**：用按层误差（我们已有的逐层追踪）自动生成 skip / 高精度名单，写进 `dynamic`。
- **导出菜单**：merged 16-bit、GGUF 的 q4_k_m / q8_0 等。GGUF 走 llama.cpp 的 convert 和 quantize。

## 3. netron → 模型结构页

- **数据**：C++ `modelspec` 从 safetensors 头 + family 模板生成层级图，节点为 模块 → 块 → 层 → 张量，边按 module_tree 的前向顺序连接。
- **展示**：
  - 左侧为层级树，可折叠。
  - 中间为按块分组的有向图，节点用 netron 的配色语义：Linear 蓝、Norm 灰、Attention 紫、Embedding 绿。
  - 右侧为属性面板：名称、dtype、shape、参数量、量化状态、该层的量化误差（SQNR / 离群比例），以及权重分布直方图（按需从 safetensors 读取该张量）。
- **交互**：搜索节点，点击节点联动误差分析。
- **不照搬**：netron 的 70 种格式解析，我们只做 safetensors 和 GGUF。

## 4. evalscope → `eval`

- **TaskConfig**：`model, eval_type(local|openai_api), datasets, dataset_args, limit, repeats, generation_config, eval_batch_size, work_dir, use_cache, seed`。
- **Registry**：`@register_benchmark(BenchmarkMeta(name, dataset_id, tags, subset_list, metric_list, …))`，Adapter 实现 `record_to_sample / predict / review`。
- **评估流程**：Evaluator 依次执行 collect → pool(predict) → review → aggregate → report。
- **输出目录**：`outputs/<ts>/{logs,predictions,reviews,reports,configs}`，可以断点续跑。
- **Report schema**：`Report{name, dataset_name, model_name, metrics:[Metric{name, num, score, categories:[Category{name, num, score, subsets:[Subset]}]}], perf_metrics}`。
- **本系统的 benchmark**：
  - `vla_action_consistency`：与 BF16 的动作一致性（现有）。
  - `vla_action_gt`：数据集真值 MAE，需要反归一化统计。
  - `general_vmcq` / `erqa`：VLM 与具身问答。
  - `ppl`：LLM。
  - 自定义 JSONL：格式与 evalscope 的 general_qa / general_mcq 相同。
- **Perf**：参数与指标名照搬（`parallel, number, rate, TTFT, TPOT, ITL, Latency, Output/Total Throughput` 及 p50/p90/p99）。并发扫描的子目录命名为 `parallel_{p}_number_{n}`。输出 `benchmark_summary.json` 和 `benchmark_percentile.json`。
- **UI**：Dashboard / Reports（Overview、Details、Predictions 逐样本）/ Compare（模型 × benchmark 矩阵）/ Perf 报告。

## 5. FlagTree → 多硬件部署（`hal`）

- **Target 描述**：`GPUTarget{backend, arch, warp_size}`，例如 `cuda:sm_89:32`、`maca:…`、`ascend:910b`。
- **后端接口**：
  - `supports_target(target)`：检查恰好一个后端匹配。
  - `is_active()`：驱动与设备存在时为真。
  - `get_current_target()`。
  - `capabilities()`：类似 FlagTree 的 `DotCap`，按 dtype 组合返回 `NATIVE / NON_NATIVE / UNSUPPORTED`。本系统用它判断“W4A8 在这块卡上有没有原生内核”。
  - `deploy(artifact) → endpoint`、`benchmark()`。
- **插件目录**：`cpp/plugins/<vendor>/`，包含 `backend.conf`、`capabilities.yaml`、驱动探测和部署适配。
  - 构建期：`MQDE_BACKENDS=nvidia;ascend;metax` 决定编译哪些插件。
  - 运行期：`MQDE_BACKEND` 可强制指定后端。未指定时，自动发现 `is_active()` 为真的后端。
- **按后端的 CI 与基准记录**：对照 FlagTree 的 `ref.txt`，记录硬件、镜像、驱动与包版本、冷/热启动时间，以及每个 workload 的吞吐、TTFT、TPOT，存为 `data/benchmarks/ref/<backend>/<model>.yaml`，作为回归基线。

## 6. 前端风格

refero 的样式详情页无法抓取 token（页面由 JS 渲染），因此以它的方向为准：Raycast 的“午夜指挥台”和 Linear 的深色高密度。具体 token 参考 evalscope 的 `DESIGN.md`，它已经给出一套深浅双主题的设计系统。

- **深色为默认**：底色 `#0b0b10`，卡片 `#13131a`，发丝边框 `rgba(255,255,255,.08)`，文字三级。
- **强调色**：单一品牌色，采用紫罗兰 `#816DF8`，与 evalscope 保持一致。
- **字体与尺寸**：等宽数字，12px 表格；圆角 8/12；阴影克制。
- **图表色**：继续用已校验的 dataviz 调色板，状态色保留给状态。
- **浅色主题**：暖白，同一套 token。

---

# 附录 A：llmfit 算法细节（可直接实现，无需再读源码）

> 2026-09-28 从 `3rdpart/llmfit` 源码逐行核对得到。文件行号为当时 commit `7f47fca`。
> 我们已实现的部分见 `cpp/src/fit/fit.cpp`；未实现的标注 **[待实现]** 并对应附录 R 任务号。

## A.1 质量分档（修正我们当前的数值）

按**活跃**参数量（`fit.rs:1969-1983`）：

| 活跃参数 | 分数 |
|---------|------|
| < 1 B | 30 |
| < 3 B | 45 |
| < 7 B | 60 |
| < 10 B | 75 |
| < 20 B | 82 |
| < 40 B | 89 |
| ≥ 40 B | 95 |

叠加项：

- **家族加分**：deepseek +3；qwen、llama +2；mistral、mixtral、gemma、starcoder +1；phi 0。
- **时效加分**：发布 < 3 个月 +3；< 9 个月 +1.5。
- **任务加分**：查 `use_case_benchmarks.json`，`((bench − 72) × 0.4)` 截断到 [−8, +9]。无记录时用启发式：名字含 coding 且用途为 coding +6；reasoning 且 ≥13B +5；带视觉 +6。
- 最后截断到 [0, 100]。

## A.2 量化质量惩罚（`models.rs:183`）

Q8_0 / F16 / MXFP4 / 8-bit 0；Q6_K −1；Q5_K_M −2；Q4_K_M −5；Q3_K_M −8；Q2_K −12；I2_S −6；mlx-4bit −4；AWQ / GPTQ / AutoRound-4bit −3；未知 −5。

**我们当前 W2A16 给了 −18，应改为 −12。**

## A.3 速度目标与上下文目标（`fit.rs:98-110`）

| 用途 | 权重 [质量,速度,适配,上下文] | 速度目标 tok/s | 上下文目标 |
|------|---------------------------|--------------|-----------|
| General | .45 / .30 / .15 / .10 | 40 | 4096 |
| Coding | .50 / .20 / .15 / .15 | 40 | 8192 |
| Reasoning | .55 / .15 / .15 / .15 | 25 | 8192 |
| Chat | .40 / .35 / .15 / .10 | 40 | 4096 |
| Multimodal | .50 / .20 / .15 / .15 | 40 | 4096 |
| Embedding | .30 / .40 / .20 / .10 | 200 | 512 |

适配分：ratio ≤ `COMFORT 0.70` 时为 100，之后高斯衰减 `100·exp(−0.5·((ratio−0.70)/0.20)²)`。
上下文分：≥ 目标 100；≥ 目标/2 为 70；否则 30。

## A.4 速度估算分支（`fit.rs:1623` estimate_tps）

按优先级：

1. **GPU 带宽屋顶线**：`bw / (params_B × bytes_per_param) × 0.55 × mode_factor`。
2. **按后端 K 常数**（GPU 带宽未知时）：BitNet 75、Metal+MLX 250、Metal+llama.cpp/vLLM 160、CUDA 220、ROCm 180、Vulkan 150、SYCL 100、CPU-ARM 90、CPU-x86 70、Ascend 390、不支持 0。再乘 `quant_speed_multiplier`，**核数 ≥ 8 再乘 1.1**。
3. **纯 CPU**：K = 70（x86）/ 90（aarch64）。
4. **MoE 卸载**：`1 / (active_gb / ddr_bw + active_gb / (bw × 0.55))`。

运行模式系数：gpu 1.0、tensor_parallel 0.9、moe_offload 0.8、cpu_offload 0.5、cpu_only 0.3。

DDR 带宽解析顺序：配置 → 环境变量 `LLMFIT_DDR_BANDWIDTH` → 实测 → 50 GB/s。

## A.5 MoE 两级速度模型 **[已实现，T-R18：`cpp/src/fit/fit.cpp`，常数在 `data/fit/fit.yaml` 的 `moe:` 段]**

我们当前只按 `active/total` 线性缩放，llmfit 分两级：

**Tier-1（架构已知，有专家维度信息）**：

```
active_ffn_params = experts_per_tok × (gate + up + down 单专家参数量)
fixed_params      = total_params − routed_expert_params
tps = K × bw / (active_ffn_GB + fixed_GB × moe_tier1_fixed_bpp)
K = 3.2（gpt_oss 架构为 1.43）
```

**Tier-2（回退）**：

```
active_GB = total_params × default_expert_ratio(0.50) × bpp
tps = bw / active_GB × efficiency × overhead
```

per-arch 校准：gpt_oss efficiency .72 / overhead .80；deepseek_v3、deepseek_v4 .62 / .70。
其它架构 efficiency 用默认 0.55，overhead 按专家数查表：

| 专家数 | overhead |
|-------|---------|
| ≤ 8 | 0.90 |
| ≤ 16 | 0.85 |
| ≤ 32 | 0.80 |
| ≤ 64 | 0.70 |
| ≥ 128 | 0.40 |
| 未知 | 0.60 |

**显存压力惩罚**：利用率超过 `VRAM_PRESSURE_UTIL_THRESHOLD 0.60` 后线性衰减，下限 `0.30`。

## A.6 预填充 / TTFT（`fit.rs:1564`）

`PREFILL_COMPUTE_UTILIZATION 0.35`，`flops_per_token = 2 × params`。
**算力未知或纯 CPU 时返回 null**，不给假数字。我们当前会算出一个值，应对齐为 null。

## A.7 置信度阶梯与本地校准 **[待实现，T-R03]**

阶梯：`measured_local > measured_community > calibrated > estimated > unsupported`。

`apply_local_calibration`（`analysis.rs`）：

1. 取锚点：稠密、≥1B、有**本地**实测（社区实测不算）的条目。
2. 锚点 < 2 个则放弃。
3. 算每个锚点的 `measured / estimated` 比值，取**中位数**，截断到 [0.05, 3.0]。
4. 该系数乘到所有**无实测**条目的 estimated_tps 上，置信度标为 `calibrated`。
5. 幂等：只改无实测项，系数每次都从锚点重算。

## A.8 实测数据格式与匹配（2026-09-28 按源码更正）

> 早先版本写的"三级硬件回退（精确 slug → 去厂商名前缀 → 硬件等级等价）"**在 llmfit 源码里不存在**，是调研摘要的误读。
> 以下按 `benchmarks.rs:395-650`、`sha
...[516 chars omitted]...
         "avgTtftMs": 180, "numRuns": 3, "source": "mqde bench"}]
}
```

## A.9 并发估算 **[待实现，T-R18]**

内存模型：`weights + users × kv_per_user + overhead`，反解 `max_users`。
另有 `max_context_for(users)`：给定并发数反解最大上下文。
结果字段含：请求并发、最大并发、上下文、各部分内存、利用率、是否放得下；利用率 > 90% 标记高压。

## A.10 claim 边界 **[待实现，T-R19]**

`claim::fit_bounds`：给定模型 + 硬件 + 最低 tok/s，产出「保守 / 乐观」两个估计（用不同效率假设），
判断保守值是否仍达标；再用二分找出仍达标的最大 2 的幂上下文长度（受模型上限约束）。
device-class 与 template 只影响输出文案，不参与计算。

## A.11 已安装运行时探测 **[待实现，T-R20]**

8 个来源，**并行**探测，单个 HTTP 超时 400 ms，整体有总deadline：

| 来源 | 方式 |
|------|------|
| ollama | HTTP `/api/tags` |
| LM Studio | HTTP API + 磁盘扫描 |
| llama.cpp | GGUF 目录扫描，深度 3 |
| vLLM | HTTP |
| MLX | 文件系统 |
| Docker Model Runner | `docker model ls` |
| RamaLama | 文件系统 |
| HuggingFace cache | 文件系统 |

名字归一化：转小写、去量化后缀、拆 ollama 的 `name:tag`。
`is_installed` 会额外尝试去掉 provider 前缀再匹配。

## A.12 HF 模型库 **[待实现，T-R21]**

`hf_models.json`：18 MB，15049 条。字段（★ 为必填）：

★`name` ★`provider` ★`parameter_count` `parameters_raw` `min_ram_gb` `recommended_ram_gb`
`min_vram_gb` ★`quantization` ★`format` ★`context_length` `use_case` `capabilities[]`
`pipeline_tag` `architecture` `hf_downloads` `hf_likes` `release_date`
`num_hidden_layers` `num_attention_heads` `num_key_value_heads` `head_dim` `hidden_size`
`vocab_size` `moe_intermediate_size` `shared_expert_intermediate_size` `license`
`gguf_sources[]` `attention_layout` `languages[]`

`llmfit update` 从 HF API 抓 trending / downloads 列表，再逐个取 config。
字段映射：`id`→name、`author`→provider、`safetensors.total`→parameter_count、
`config.architectures[0]`→architecture、`cardData.license`→license。
token 走环境变量或 `--token`，用 bearer auth。每次 config 请求间隔固定 **250 ms**。

**目录卫生**：draft / 投机解码模型（名字含 `-draft`、`-eagle`）、参数量与 raw 不一致、
缺尺寸或上下文长度的条目会被降级，不参与排名。

## A.13 下载文件选择（`download`）

列出 repo 文件后，按数字前缀把分片归组（下一片就下整组）。
文件大小取 HF 元数据，缺失时按 `params × bpp` 估。在预算内按质量排序取最优，
预算内无解时退化为最小文件。下载走 `resolve` 端点并跟随重定向。

## A.14 API 端点（`serve_api.rs:255-266`）

`GET /`、`GET /health`、`GET /api/v1/system`、`GET /api/v1/models`、
`GET /api/v1/models/top`（limit=5，排除 too_tight）、`GET /api/v1/models/{name}`、
`GET /api/v1/runtimes`、`GET /api/v1/installed`、
`POST /api/v1/download`、`GET /api/v1/download/{id}/status`、`POST /api/v1/plan`。

查询参数：`limit|n, perfect, min_fit, runtime, use_case, provider, search, sort,
include_too_tight, max_context, force_runtime`，以及 `ram_gb, vram_gb, cpu_cores` 覆盖项。

## A.15 ModelFit 字段全集（`fit.rs:331-380`）

`model, fit_level, run_mode, memory_required_gb, memory_available_gb, utilization_pct,
notes, moe_offloaded_gb, score, score_components{quality,speed,fit,context},
estimated_tps, best_quant, use_case, runtime, installed, fits_with_turboquant,
effective_context_length, usable_context,
estimate_basis{method, gpu_bandwidth_gbps, ddr_bandwidth_gbps, efficiency,
assumed_context, local_calibration},
measured_tps{tok_s, sample_count, hardware_label, source}, estimate_confidence,
prefill_tps, ttft_ms`

JSON 额外字段：`params_b, category, fit_label, run_mode_label, runtime_label,
capability_ids, disk_size_gb, ollama_name, supports_tp, verify_command,
estimate_confidence_label, gguf_sources, license`

**我们缺的字段**：`moe_offloaded_gb, runtime, installed, usable_context, estimate_basis,
measured_tps, disk_size_gb, supports_tp, verify_command, license, capability_ids`

## A.16 运行时选择（`fit.rs:475-511`）

`InferenceRuntime = LlamaCpp | Mlx | Vllm | BitNet | Unsupported`，按序判断：
ternary → BitNet（不接受强制覆盖）；cluster_mode → vLLM；已量化 → vLLM；
Metal + 统一内存 → MLX；否则 llama.cpp。

各运行时差异：量化层级、内存模型、K 常数、MLX vs llama.cpp 提速说明、
vLLM 固定开销（不重新量化）、TurboQuant 仅 vLLM+CUDA、BitNet 强制走 CPU/RAM。

`verify_command` 产出 `llama-bench` / `llama-cli` 参数，按运行模式给 `-ngl all|0|auto --fit on`。

**llmfit 没有的**：k8s / 调度器清单生成；除 runtime 标签和 pull tag 外没有「该装哪个运行时」的建议。

## A.17 存储估算（`storage.rs:144`）**[待实现，低优先级]**

`可用预算 = 空闲磁盘 − OS 保留(默认 20 GB) − scratch(默认 50 GB) − 总容量 × headroom(默认 10%)`，
与 keep 集合的实际占用比较判断是否放得下。
`parse_storage_size` 支持 `500GB` / `1.5TB`，区分十进制与二进制后缀。
单模型占用按 `params × bpp` 加元数据开销估算。

## A.18 我们不打算实现的部分

TUI（21 种输入模式、主题、下载管理器、排行榜视图）、MCP server（6 个 tool）、
NATS 事件、质量评测 rubric（`benchmarks.yaml` 19 个角色 + 路由矩阵 + frontier 对比）、
desktop（Tauri 5 个命令）、llmfit-python（只是二进制包装）、i18n（我们只做中文）。
