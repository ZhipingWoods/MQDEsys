# 参考架构对照（REFERENCES）

> 各子系统**必须模仿**的开源项目，以及“模仿到什么程度、落在哪个模块”。文档级别 **L2**。
> 源码在开发机 `/cache/wuzhiping/MQDEsys/3rdpart/`（只读参考，已在 `.gitignore`，**不得修改、不得直接 import 进产品代码**；需要复用代码时按许可证单独引入并走 L3）。

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
