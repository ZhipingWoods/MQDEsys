# MQDEsys 任务清单（TASKS）

> 按本清单**从上到下、按依赖**执行，即可构建出功能完整的系统。
> 每个任务包含：依赖 / 级别 / 产出 / 实现要点 / 验收标准 / 校验方法 / 状态。
>
> - 任务**定义**（产出、验收）的修改为 **L2**（需人工批准）；任务**状态**字段的更新为 **L0**（Agent 自行维护）。
> - 设计依据见 [ARCHITECTURE.md](ARCHITECTURE.md)（下文以 `A§x.y` 引用章节）；文件位置见 [STRUCTURE.md](STRUCTURE.md)。
> - 工作流程见 [../AGENTS.md](../AGENTS.md)。

---

## 0. 使用说明

### 0.1 状态标记（唯一事实来源 = 下方总览表的“状态”列）

| 状态 | 写法 | 含义 |
|------|------|------|
| 待办 | `TODO` | 未开始，依赖满足即可认领 |
| 进行中 | `DOING @agent-name feat/T-xxxx-slug` | 已认领，写明认领者与分支 |
| 评审中 | `REVIEW #PR号` | 已提 PR，等待评审 |
| 完成 | `DONE <commit短sha>` | 已合并到 main 且验收通过 |
| 部分完成 | `PARTIAL (日期, 说明)` | 最小可用实现已落地，验收未全部满足；说明里写缺什么 |
| 阻塞 | `BLOCKED: 原因` | 需人工决策或外部条件 |

### 0.2 认领规则

1. 只能认领 **所有依赖均为 DONE** 的任务。
2. 同一时间一个 Agent 最多持有 1 个 DOING 任务（除非任务被显式拆分）。
3. 若任务过大（预计 > 1 天工作量或 PR > 800 行非生成代码），先拆分为 `T-xxxx.a / .b …`，在本文件追加子任务（L2 修改，在 PR 中说明）。
4. 完成后必须：更新状态 → 更新 [PROGRESS.md](PROGRESS.md) → 在 PR 描述中贴出“校验方法”的执行结果。

### 0.3 全局完成定义（Definition of Done，适用于每个任务）

- [ ] 产出物全部存在于 STRUCTURE.md 规定的位置
- [ ] 任务“验收标准”逐条满足，“校验方法”逐条执行并在 PR 中贴出输出
- [ ] `scripts/check_all.sh` 本地通过（阶段 P0 完成后生效）
- [ ] 新增公共接口有文档注释；新增行为有测试
- [ ] 触及 L2/L3 路径已走对应评审流程
- [ ] 相关文档（guides、ARCHITECTURE 表格、README）已同步
- [ ] TASKS.md 状态、PROGRESS.md 已更新

---

## 1. 里程碑

| 里程碑 | 包含阶段 | 演示场景（验收即演示） |
|--------|---------|----------------------|
| **M0 骨架可构建** | P0 | 空仓库 clone 后 `scripts/dev_setup.sh && scripts/check_all.sh` 全绿；CI 跑通 |
| **M1 预评估可用** | P1、P2 | 无 GPU 机器上：`mqde probe`；`mqde estimate -m Qwen/Qwen2.5-7B-Instruct -r W4A16 --hw nvidia/a100-80g --ctx 8192 --conc 1,8,32` 输出显存构成、KV、最大并发、可部署性 |
| **M2 可视化预估** | P3 | 浏览器中新建作业向导实时预估；预估中心展示 5 类图表 |
| **M3 最小闭环** | P4、P5 | tiny-llama / Qwen2.5-0.5B：Web 提交 W4A16-RTN 作业 → 实时 DAG 与日志 → PPL 基线 vs fake → 逐层误差热力图 |
| **M4 全配方 + 部署评测** | P6、P7、P8 | Qwen2.5-7B：五种配方 × RTN/AWQ/GPTQ/SmoothQuant；vLLM/SGLang/llama.cpp 部署；TTFT/TPOT/吞吐/OOM 边界；预估 vs 实测对比 |
| **M5 误差归因 + 反馈闭环** | P9 | 阶段瀑布图定位最大误差环节；注入噪声层排 Top-1；预估校准后误差达标 |
| **M6 多模态 + 多硬件** | P10、P11 | Qwen2-VL 分模态量化；模板插件/ROCm/Metal/Ascend 插件接入 |
| **M7 发布 v1.0** | P12 | `pip install mqde`、`docker compose up`、文档站上线、v1.0.0 tag |

---

## 2. 任务总览

| ID | 标题 | 阶段 | 依赖 | 级别 | 状态 |
|----|------|------|------|------|------|
| T-0001 | 仓库初始化与 Git 规范 | P0 | — | L2 | PARTIAL (2026-09-24, git init on 4090; remote/branch protection pending) |
| T-0002 | C++ 构建系统（CMake + vcpkg + Presets） | P0 | T-0001 | L2 | DONE (2026-09-24, CMake+Presets(conda deps, not vcpkg yet)) |
| T-0003 | Python 包骨架与 pybind11 构建链 | P0 | T-0002 | L2 | DONE (2026-09-24, pybind11 _core in-tree build) |
| T-0004 | 前端工程骨架 | P0 | T-0001 | L2 | PARTIAL (2026-09-24, Vite+React+ECharts (no AntD/tests yet)) |
| T-0005 | 代码规范与 pre-commit | P0 | T-0002,T-0003,T-0004 | L2 | DONE (2026-09-30, check_all.sh + check_deps.py + clang-format/tidy + ruff/mypy clean + pre-commit) |
| T-0006 | CI 流水线（PR 门禁） | P0 | T-0005 | L2 | TODO |
| T-0007 | 契约与 Schema 初版 | P0 | T-0001 | L2 | TODO |
| T-0008 | 开发指南与 ADR 初始化 | P0 | T-0001 | L2 | TODO |
| T-0101 | common：错误、Result、日志 | P1 | T-0002 | L2 | DONE (2026-09-24) |
| T-0102 | common：配置加载与 Schema 校验 | P1 | T-0101,T-0007 | L2 | DONE (2026-09-24) |
| T-0103 | common：注册表、哈希、线程池 | P1 | T-0101 | L2 | TODO |
| T-0104 | storage：SQLite 元数据与迁移 | P1 | T-0101 | L2 | TODO |
| T-0105 | storage：内容寻址缓存池（CAS）与 GC | P1 | T-0104,T-0103 | L2 | TODO |
| T-0106 | CLI 骨架（CLI11） | P1 | T-0102 | L1 | TODO |
| T-0201 | C 硬件探针：CPU（三平台） | P2 | T-0002 | L2 | DONE (2026-09-24) |
| T-0202 | C 硬件探针：GPU/NPU（动态加载厂商库） | P2 | T-0201 | L1 | DONE (2026-09-24, NVML only) |
| T-0203 | 硬件画像库与 HardwareProfile | P2 | T-0102,T-0201 | L1 | DONE (2026-09-24, 4 NVIDIA profiles) |
| T-0204 | hub：HF/ModelScope 解析与元数据获取 | P2 | T-0102 | L2 | PARTIAL (2026-09-29, @revision, HF_ENDPOINT, retries, Range headers, inspect_remote; ModelScope + offline cache pending) |
| T-0205 | modelspec：ModelSpec 解析（config + safetensors 头） | P2 | T-0204 | L2 | DONE (2026-09-24) |
| T-0206 | 架构模板与 QIR | P2 | T-0205 | L2 | DONE (2026-09-24, data-driven families; QIR = UnitInfo) |
| T-0207 | 量化配方库 | P2 | T-0102 | L1 | DONE (2026-09-24) |
| T-0208 | estimator：显存/KV/并发预估 | P2 | T-0206,T-0207,T-0203 | L2 | DONE (2026-09-24, +linear-attention state) |
| T-0209 | estimator：性能预估与 CalibrationStore | P2 | T-0208,T-0104 | L2 | TODO |
| T-0210 | planner：兼容矩阵与 JobSpec 校验 | P2 | T-0207,T-0203 | L2 | TODO |
| T-0211 | planner：推荐与可部署矩阵 | P2 | T-0208,T-0210 | L1 | TODO |
| T-0212 | CLI：probe / model inspect / estimate / plan | P2 | T-0106,T-0211,T-0202 | L1 | PARTIAL (2026-09-24, probe/inspect/estimate/recipes/hardware) |
| T-0301 | server：Drogon 服务骨架与只读端点 | P3 | T-0212,T-0007 | L2 | PARTIAL (2026-09-24, cpp-httplib instead of Drogon (see PROGRESS)) |
| T-0302 | server：预估类端点 | P3 | T-0301 | L1 | PARTIAL (2026-09-24) |
| T-0303 | 前端：API 客户端生成、布局、路由 | P3 | T-0004,T-0301 | L1 | PARTIAL (2026-09-24, hand-written typed client; generated client pending T-0007) |
| T-0304 | 前端：新建作业向导 + 实时预估侧栏 | P3 | T-0303,T-0302 | L1 | PARTIAL (2026-10-09, form + live estimate sidebar: per-recipe weights / peak / feasible from the C++ estimator, 300 ms debounce, errors shown; Playwright web/e2e/newjob_live_estimate.mjs 10/10. Not done: five-step wizard layout, planner validation of illegal combinations, YAML download, `mqde run --dry-run` acceptance) |
| T-0305 | 前端：预估中心 5 类图表 | P3 | T-0303,T-0302 | L1 | PARTIAL (2026-09-24) |
| T-0306 | 前端：硬件页 | P3 | T-0303 | L1 | PARTIAL (2026-09-24) |
| T-0401 | quant 核心：QuantSpec/QParams/粒度重排 | P4 | T-0101 | L2 | DONE (2026-09-24) |
| T-0402 | quant 核心：标定算法 MinMax/MSE/百分位/MovingAvg | P4 | T-0401 | L1 | DONE (2026-09-24, MinMax/MSE) |
| T-0403 | quant 核心：INT/FP8 量化、反量化、假量化 + golden | P4 | T-0402 | L2 | DONE (2026-09-24, INT2-8/FP8 bit-exact) |
| T-0404 | quant 核心：误差度量 | P4 | T-0401 | L1 | DONE (2026-09-24) |
| T-0405 | quant 核心：位打包（GPTQ/AWQ/Marlin/GGUF） | P4 | T-0403 | L1 | PARTIAL (2026-09-30, GPTQ v1/v2 2/3/4/8, AWQ GEMM, GGUF Q8_0/Q5_0/Q4_0/IQ4_NL/Q4_K/Q2_K byte-exact vs ggml; Marlin 4/8-bit pack/unpack 2026-10-05, round-trip + port of vLLM v0.30.0 layout tested, byte-exact check vs installed vLLM pending) |
| T-0406 | quant 核心：SIMD/CUDA 加速 | P4 | T-0403 | L1 | TODO |
| T-0407 | pybind11 绑定 `mqde._core`（DLPack） | P4 | T-0403,T-0404,T-0003 | L2 | DONE (2026-09-24, buffer protocol (DLPack later)) |
| T-0408 | Python Torch 快速路径与双端 golden 对拍 | P4 | T-0407 | L1 | DONE (2026-09-24) |
| T-0501 | Worker RPC 协议与 C++ WorkerClient | P5 | T-0103,T-0007 | L2 | TODO |
| T-0502 | Python Worker 进程框架 | P5 | T-0501,T-0003 | L1 | TODO |
| T-0503 | pipeline：DAG、Stage 接口、状态机 | P5 | T-0104,T-0105 | L2 | PARTIAL (2026-09-24, Python-side stages; C++ DAG pending) |
| T-0504 | pipeline：队列、调度、设备分配、重试/恢复 | P5 | T-0503,T-0202 | L1 | PARTIAL (2026-09-24, GPU pool + queue; no retry) |
| T-0505 | 模型下载（fetch_model 阶段） | P5 | T-0204,T-0503 | L1 | TODO |
| T-0506 | Python 模型抽象层 BaseModel + HFDecoderModel | P5 | T-0502,T-0206 | L2 | DONE (2026-09-24, VLA adapter) |
| T-0507 | 校准数据与首块捕获 | P5 | T-0506 | L1 | DONE (2026-09-24, LeRobot v3 + synthetic) |
| T-0508 | Blockwise 引擎 + Fake/Real 模块包装 + RTN | P5 | T-0507,T-0408 | L2 | DONE (2026-09-29, quant/blockwise.py sequential blocks; real = export + use_real_weights) |
| T-0509 | 误差追踪 trace（逐层/逐块） | P5 | T-0508,T-0404 | L1 | DONE (2026-09-24) |
| T-0510 | 评测：PPL + logits 探针 | P5 | T-0506 | L1 | PARTIAL (2026-10-08, eval/logits_probe.py wired into llm_job (`--logits-probe`, YAML analysis.logits_probe) and the HTML report: KL / Top-1 agreement / drop of the reference's top-1 probability against the unquantized model, reference captured before quantization; top-k reference (default k=256) gives a KL LOWER BOUND, measured 10.8% / 10.6% low on Qwen2.5-0.5B W4A16 RTN / GPTQ vs the exact KL, Top-1 agreement unaffected; `exact` option keeps all logits (1199 MiB per 8x512 vs 12 MiB). Spec items not done: fixed 128-prompt set, saving top-k logits to disk, `eval.run` handler, and the acceptance (PPL within 1% of public / lm-eval values: not compared)) |
| T-0511 | 阶段实现 S00–S08 + S14 并串通 | P5 | T-0504,T-0505,T-0509,T-0510 | L1 | PARTIAL (2026-09-24, VLA stages S00-S08,S14) |
| T-0512 | server：作业端点 + WebSocket 事件流 | P5 | T-0511,T-0301 | L2 | PARTIAL (2026-09-29, SSE; model.type vla/llm, CPU slot) |
| T-0513 | 前端：作业执行页（DAG + 日志 + 资源） | P5 | T-0512,T-0303 | L1 | PARTIAL (2026-09-24) |
| T-0514 | 前端：量化误差页（初版：热力图 + Top-K） | P5 | T-0512,T-0509 | L1 | PARTIAL (2026-09-24, heatmap + Top-K + attribution) |
| T-0515 | 端到端最小闭环验收（M3） | P5 | T-0513,T-0514 | L1 | PARTIAL (2026-09-30, docs/guides/quickstart.md + LLM 全功能作业跑通; VLA 演示待 GPU) |
| T-0601 | 算法：AWQ | P6 | T-0508 | L1 | DONE (2026-09-29, b2248e4; per-layer scale search, VLA eval pending GPU) |
| T-0602 | 算法：GPTQ | P6 | T-0508 | L1 | DONE (2026-09-29, b2248e4; per-layer OBS on FP inputs, not sequential-block) |
| T-0603 | 算法：SmoothQuant（等效变换 + transformed 位点） | P6 | T-0508 | L1 | PARTIAL (2026-09-30, per-group scales folded into the previous op (FP-exact, exportable); transformed eval position not yet) |
| T-0604 | 激活量化（动态/静态）→ W4A8、W2A8 | P6 | T-0603 | L1 | DONE (2026-09-29, dynamic per-token + static per-tensor minmax/ema/percentile) |
| T-0605 | FP8 配方（E4M3/E5M2，per_channel/per_block） | P6 | T-0508 | L1 | PARTIAL (2026-09-24, FP8 E4M3/E5M2 per-channel/per-tensor fake-quant; 2026-10-08 added reading block-wise FP8 checkpoints as plain float weights: models/fp8_dequant.py, `W = fp8 * weight_scale_inv` (multiply, fixed from the real 27B checkpoint: product has weight-sized magnitude, divide gives ~6e5), 12 tests, magnitudes checked on 9 layers of the real checkpoint; NOT done: per-block *quantization* (needs a Granularity enum value in the public header quant_spec.h, L2), wiring the reader into HFCausalLM, and the acceptance run (FP8 7B PPL degradation <= 1%, no 7B FP8 checkpoint available and the 27B does not fit a 24 GB card)) |
| T-0606 | 2bit 配方（W2A16/W2A8）与 HQQ | P6 | T-0602,T-0604 | L1 | PARTIAL (2026-09-24, W2A16/W2A8 fake-quant via RTN-MSE; no GPTQ/HQQ) |
| T-0607 | KV Cache 量化（KV_REGISTRY） | P6 | T-0508 | L1 | DONE (2026-09-29, naive/KIVI; INT8 KIVI -0.03% PPL on Qwen2.5-0.5B) |
| T-0608 | 混合精度与 keep_fp 策略 | P6 | T-0509 | L1 | DONE (2026-09-29, regex/block rules, skip; exported as GPTQModel dynamic) |
| T-0609 | 外部算法适配器（GPTQModel / llm-compressor） | P6 | T-0508 | L3 | TODO |
| T-0610 | 算法组合（AWQ+GPTQ）与 QuaRot（扩展） | P6 | T-0601,T-0602 | L1 | PARTIAL (2026-09-29, awq+gptq / smoothquant+gptq; no QuaRot) |
| T-0701 | Exporter 接口 + vLLM 导出 | P7 | T-0405,T-0602 | L2 | PARTIAL (2026-10-08: the exporter's own AWQ / compressed-tensors W4A16 / compressed-tensors FP8 checkpoints of Qwen2.5-0.5B load in vLLM 0.30.0 and serve (served PPL within +0.01% / +0.01% / +0.27% of the real-weight PPL); asymmetric GPTQ is rejected by vLLM 0.30.0's validator (not an export bug); Marlin byte-exact vs vLLM ref; 7B W4A16 / FP8 acceptance run pending a 7B checkpoint) |
| T-0702 | Fake/Real 一致性检查（S10） | P7 | T-0701 | L1 | DONE (2026-09-29, 0 code mismatches; real PPL +0.0005% vs fake) |
| T-0703 | 部署后端接口 + vLLM 适配器（S11） | P7 | T-0701,T-0504 | L2 | PARTIAL (2026-10-08, Python only: backend/base.py ServerBackend + vllm_server.py (launch, health, log diagnosis, engine_info) + llm_job `deploy_vllm` stage + data/backends/vllm.yaml; verified on real vLLM 0.30.0; C++ `deploy_backend.h` (L2, needs approval) and `ctest -R backend_` not done) |
| T-0704 | SGLang 导出与适配 | P7 | T-0703 | L1 | TODO |
| T-0705 | llama.cpp（GGUF）导出与适配 | P7 | T-0703,T-0405 | L1 | DONE (2026-10-05, Q4_K deploy via llama-server on Linux CPU + Apple M5 CPU/Metal; tests/integration/deploy_llamacpp_cpu.sh; UI serve columns; CUDA build not exercised) |
| T-0706 | TensorRT-LLM 导出与适配（扩展） | P7 | T-0703 | L1 | TODO |
| T-0801 | 精度评测：lm-eval-harness / evalscope / 自定义 | P8 | T-0510 | L3 | PARTIAL (2026-10-08, custom JSONL checks done: eval/custom.py + data/datasets/smoke_qa.jsonl; 2026-10-09 ADR-0002 approved (subprocess, no new dependency): eval/lm_eval_adapter.py runs the official `lm-eval` CLI against the served model; run directly with identical arguments the CLI gave bit-identical gsm8k scores (20 samples); deploy_vllm option `lm_eval` plus an unquantized baseline run, shown first in the report. Measured: Qwen2.5-0.5B W4A16 gsm8k 5-shot, 50 samples, 0.02 / 0.06 against 0.20 / 0.28 unquantized (PPL 18.10 to 20.66). Not done: MMLU run (only gsm8k tried), evalscope adapter (later, per ADR), the spec's acceptance on a full-size run, VLA) |
| T-0802 | 部署后精度评测（S12） | P8 | T-0801,T-0703 | L1 | PARTIAL (2026-10-08, eval/served_ppl.py: PPL from the backend's own prompt_logprobs, same windows as the fake/real PPL; vLLM only. 2026-10-09: the lm-eval adapter is wired into deploy_vllm (see T-0801); evalscope adapter and llama-server (no prompt_logprobs; lm-eval generation tasks could work, untried) not done) |
| T-0803 | 性能压测器（S13）：TTFT/TPOT/吞吐 | P8 | T-0703 | L1 | PARTIAL (2026-10-08, Python eval/loadtest.py: closed loop + Poisson open loop, exact-length token prompts, TTFT/TPOT/ITL/E2E p50/p90/p99; against the mock (TTFT 100 ms, TPOT 20 ms) error < 2 ms. The spec's C++ implementation is not done) |
| T-0804 | 显存采样与 OOM 边界搜索 | P8 | T-0803,T-0202 | L1 | PARTIAL (2026-10-08, eval/resources.py 100 ms GPU/host sampler, eval/boundary.py bisection over concurrency / context with a fresh backend per probe; tested on the mock; run on real vLLM 0.30 (Qwen2.5-0.5B AWQ, GPU 3) on both axes: every probe passed up to the limits I set (1024 concurrent, 32768 context), so no true OOM boundary was found for this model on this card (vLLM queues rather than fails, and the model caps context at 32768); not yet in the report next to the estimator) |
| T-0805 | HAL 微基准（带宽/算力实测） | P8 | T-1101 | L1 | TODO |
| T-0806 | 报告生成（JSON + HTML） | P8 | T-0802,T-0804 | L1 | PARTIAL (2026-10-08, self-contained report.en.html / report.zh.html written by every LLM job and served at /api/v1/jobs/<id>/report.<lang>.html; 2026-10-09: report.schema.json (draft-07) + dependency-free validator `python -m mqde.report validate` and `html` subcommands, checked against the 3 real fixtures and a tiny-job report, 14 damage cases + 2 mutation checks; VLA job reports and a `mqde` console entry point not done; the spec asks for C++ `ctest -R report_`, this is Python) |
| T-0807 | 前端：评估报告页 + 对比页 | P8 | T-0806 | L1 | PARTIAL (2026-09-30, LLM/VLA compare tabs + llama.cpp table; standalone report page: job page links to the HTML report from 2026-10-08, VLA jobs not covered) |
| T-0808 | M4 验收：全配方 × 多后端参考集 | P8 | T-0807,T-0606,T-0705 | L1 | TODO |
| T-0901 | 阶段归因（瀑布）数据链路 | P9 | T-0702,T-0802 | L1 | PARTIAL (2026-09-24, modality-level attribution only) |
| T-0902 | 敏感度分析（逐层回退） | P9 | T-0509 | L1 | PARTIAL (2026-09-29, block/type FP fallback in llm_job; 7B GPU timing pending) |
| T-0903 | 前端：误差页完整版 | P9 | T-0901,T-0902 | L1 | PARTIAL (2026-10-09, layer detail card (metrics, ranks, weight/activation histograms, QParams summary) opened from a Top-15 row or by clicking a heatmap cell; VLA page states per recipe which modality's quantization hurts the action most (real attribution data, fixture vla_attribution.json); web/e2e/layer_detail.mjs 20/20 and attribution_highlight.mjs 8/8, three mutations caught. Not done: a module tree (modality > block > unit) linked to the heatmap; the 'stage waterfall' is the per-modality attribution bars plus this text, not a fake-to-real conversion waterfall (needs T-0901's finer attribution data); per-group QParams histogram; RTN has no activation histogram) |
| T-0904 | 混合精度建议 → 一键生成新作业 | P9 | T-0903,T-0608 | L1 | PARTIAL (2026-10-08, suggest() + UI one-click clone; Playwright-based UI test web/e2e/clone_from_suggestion.mjs, 14/14 on macOS + Chrome, mutation-checked, found and fixed a page crash on a report without `spec`; NOT in the gate: the dev server has no Playwright or Chrome, wiring it in is D-10) |
| T-0905 | 反馈闭环（S15）：预估自动校准 | P9 | T-0209,T-0804 | L1 | TODO |
| T-0906 | 误差归因验收（注入测试） | P9 | T-0903 | L1 | PARTIAL (2026-10-08, python/tests/test_error_attribution.py: injection via the product's own mixed-precision policy (one layer at 2 bit), the damaged layer is ranked worst by out SQNR in 4 of 4 victims with margins >= 34x the undamaged spread; corrupted packing is the largest fake-to-real error; mutation-checked. The spec's debug.inject_noise / debug.corrupt_pack switches were not added (not needed), A§10.2 acceptance was not available to compare; the FP-fallback sensitivity ranking is NOT asserted because it is noise-dominated on a tiny random model) |
| T-1001 | 多模态框架：per_modality 配置与 set_modality | P10 | T-0508 | L2 | TODO |
| T-1002 | VLM 适配（Qwen2-VL / LLaVA） | P10 | T-1001 | L1 | TODO |
| T-1003 | MoE / MLA 结构支持（DeepSeek 系） | P10 | T-0508,T-0208 | L1 | TODO |
| T-1004 | VLA 适配（实验） | P10 | T-1002 | L1 | DONE (2026-09-24, MiniCPM-RobotManip fake-quant) |
| T-1005 | 世界模型/视频生成（DiT）适配（实验） | P10 | T-1001 | L1 | TODO |
| T-1101 | HAL ABI 与插件加载器 + 模板插件 | P11 | T-0201 | L2 | TODO |
| T-1102 | CPU / CUDA 插件 | P11 | T-1101 | L1 | TODO |
| T-1103 | Metal 插件 | P11 | T-1101 | L1 | TODO |
| T-1104 | ROCm 插件 | P11 | T-1101 | L1 | TODO |
| T-1105 | Ascend（CANN）插件 | P11 | T-1101 | L1 | TODO |
| T-1106 | 插件 SDK 文档与 ABI 兼容检查 | P11 | T-1101 | L1 | TODO |
| T-1201 | Docker 镜像与 compose | P12 | T-0515 | L2 | TODO |
| T-1202 | pip wheel 打包与发布 | P12 | T-0515 | L2 | TODO |
| T-1203 | 文档站（GitHub Pages） | P12 | T-0008 | L1 | PARTIAL (2026-10-08, docs/site: Sphinx en + zh, -W clean; GitHub Pages deployment needs a workflow under .github/ (L2), kept as docs/site/pages-workflow.proposal.yml awaiting approval) |
| T-1204 | 安全加固（凭证、路径沙箱、认证） | P12 | T-0512 | L3 | TODO |
| T-1205 | 可观测性（/metrics、可复现重放） | P12 | T-0511 | L1 | TODO |
| T-1206 | 夜间 GPU 回归与性能看板 | P12 | T-0808 | L2 | TODO |
| T-1207 | v1.0 发布验收 | P12 | 全部 P0–P11 必选任务 | L2 | TODO |

> 标注“扩展/实验”的任务（T-0610、T-0706、T-1004、T-1005、T-1103~T-1105）不阻塞 v1.0，其余均为必选。
> 注意：T-0805 依赖 T-1101（HAL），可在 P8 前提前完成 T-1101。

---

## 3. 任务详情

### P0 —— 工程底座

#### T-0001 仓库初始化与 Git 规范
- **依赖**：— **级别**：L2
- **产出**：`.gitignore`、`.gitattributes`（LFS 规则：`*.safetensors *.gguf *.bin *.pt` 默认拒绝，仅 `tests/fixtures/**` 允许走 LFS）、`README.md`、`LICENSE`（待负责人确认，建议 Apache-2.0）、`CHANGELOG.md`、`.github/PULL_REQUEST_TEMPLATE.md`、`.github/ISSUE_TEMPLATE/*`、`.github/CODEOWNERS`、按 STRUCTURE.md 创建空目录（带 `.gitkeep` 或 README）。
- **实现要点**：远端 `github.com/zhipingwoods/MQDEsys`；`main` 分支保护（需 PR、需 CI、禁止 force push）；PR 模板包含：任务 ID、级别、变更摘要、验证证据、文档同步勾选、破坏性变更说明。
- **验收**：目录与 STRUCTURE.md 第 2 节一致；分支保护生效。
- **校验**：`git ls-files | head`、`tree -L 2`；尝试直接 push main 被拒。

#### T-0002 C++ 构建系统
- **依赖**：T-0001 **级别**：L2
- **产出**：顶层 `CMakeLists.txt`、`cpp/CMakeLists.txt`、`CMakePresets.json`（`dev`/`release`/`asan`/`cuda`）、`vcpkg.json`（初始依赖：nlohmann-json、yaml-cpp、spdlog、fmt、cli11、gtest、benchmark、sqlitecpp、curl、drogon、json-schema-validator、pybind11）、一个 `mqde_common` 占位库 + 一个 hello 测试。
- **实现要点**：C++20；选项 `MQDE_WITH_CUDA`、`MQDE_WITH_PYTHON`、`MQDE_BUILD_TESTS`；每个模块 `add_library(mqde_<module> STATIC)`，通过 `target_link_libraries` 表达依赖方向；警告即错误（`-Wall -Wextra -Werror`，MSVC `/W4 /WX`）；导出 `compile_commands.json`。
- **验收**：Linux(gcc 12+/clang 16+)、macOS(Apple clang) 编译通过；Windows(MSVC 2022) 至少 common 编译通过。
- **校验**：`cmake --preset dev && cmake --build --preset dev && ctest --preset dev` 输出 100% tests passed。

#### T-0003 Python 包骨架与 pybind11 构建链
- **依赖**：T-0002 **级别**：L2
- **产出**：`pyproject.toml`（scikit-build-core 构建 `mqde._core`）、`python/mqde/__init__.py`、`_core.pyi`、`python/tests/test_import.py`。
- **实现要点**：`pip install -e .` 即编译 C++ 扩展；可选依赖分组：`[quant]`（torch、transformers、safetensors）、`[eval]`（lm-eval、evalscope）、`[deploy]`（vllm 等不强装）、`[dev]`。
- **验收**：`python -c "import mqde, mqde._core; print(mqde._core.version())"` 输出版本号。
- **校验**：`pip install -e ".[dev]" && pytest python/tests -q`。

#### T-0004 前端工程骨架
- **依赖**：T-0001 **级别**：L2
- **产出**：`web/`：Vite + React 18 + TS（strict）+ Ant Design + ECharts + TanStack Query + Zustand + react-router + i18next；vitest、Playwright、eslint、prettier 配置；空白布局页。
- **验收**：`pnpm dev` 可访问；`pnpm build` 成功。
- **校验**：`cd web && pnpm i && pnpm lint && pnpm typecheck && pnpm test && pnpm build`。

#### T-0005 代码规范与 pre-commit
- **依赖**：T-0002, T-0003, T-0004 **级别**：L2
- **产出**：`.clang-format`（基于 Google，列宽 110）、`.clang-tidy`、`ruff`/`mypy` 配置（pyproject 内）、`.editorconfig`、`.pre-commit-config.yaml`、`scripts/check_all.sh`、`scripts/check_deps.py`、`docs/guides/coding-standards.md`。
- **实现要点**：`check_all.sh` 与 CI 门禁完全等价（A§11.2），失败时给出可读摘要；`check_deps.py` 解析 `#include <mqde/...>` 并对照 STRUCTURE.md 3.5 的依赖方向。
- **验收**：故意引入格式错误/反向依赖，`check_all.sh` 失败并指出位置。
- **校验**：`pre-commit run --all-files && scripts/check_all.sh`。

#### T-0006 CI 流水线
- **依赖**：T-0005 **级别**：L2
- **产出**：`.github/workflows/ci.yml`（矩阵：ubuntu-gcc、ubuntu-clang、macos；Windows 仅编译最小集），`nightly.yml`、`release.yml`、`docs.yml` 占位。
- **实现要点**：vcpkg 二进制缓存、pip/pnpm 缓存；任务：format → lint → build → test(C++/Py/Web) → contract-check → deps-check；PR 评论输出覆盖率摘要。
- **验收**：新 PR 自动触发，全绿后才允许合并；首次冷构建 < 30 分钟，缓存命中 < 10 分钟。
- **校验**：提交一个故意失败的测试 PR，确认被阻止合并。

#### T-0007 契约与 Schema 初版
- **依赖**：T-0001 **级别**：L2
- **产出**：`api/openapi.yaml`（A§6.14 全部端点的请求/响应骨架）、`api/events.schema.json`、`api/worker_rpc.schema.json`、`schemas/*.schema.json`（A§7 全部结构）、`examples/jobs/*.yaml`（≥ 5 个示例，覆盖五种配方）、`scripts/gen_api.sh`。
- **实现要点**：Schema 使用 JSON Schema 2020-12；每个结构带 `apiVersion`；枚举值（配方、粒度、阶段、事件类型）集中定义并被引用。
- **验收**：所有示例 JobSpec 通过 schema 校验；OpenAPI 通过 `redocly lint`。
- **校验**：`scripts/gen_api.sh --check`；`npx @redocly/cli lint api/openapi.yaml`；`python -m jsonschema` 校验 examples。

#### T-0008 开发指南与 ADR 初始化
- **依赖**：T-0001 **级别**：L2
- **产出**：`docs/adr/0001-cpp-control-plane-python-executor.md` … `0005-tech-stack.md`（将 A§2 展开为完整 ADR，状态 Proposed，由负责人改为 Accepted）、`docs/guides/dev-setup.md`、`scripts/dev_setup.sh`。
- **验收**：新人（新 Agent）按 dev-setup.md 从零在 macOS/Linux 搭好环境 ≤ 30 分钟。
- **校验**：在干净容器中执行 `scripts/dev_setup.sh && scripts/check_all.sh`。

---

### P1 —— 核心基础设施

#### T-0101 common：错误、Result、日志
- **依赖**：T-0002 **级别**：L2
- **产出**：`cpp/include/mqde/common/{error.h,result.h,log.h,types.h}` + 实现 + 单测。
- **实现要点**：见 A§6.1；`Result<T>` 用 `tl::expected`（C++20）并预留切换 `std::expected`；日志支持 console（彩色）+ JSON Lines 文件；日志上下文（job_id/stage）用 thread_local scope guard 注入。
- **验收**：错误可链式附加上下文；日志 JSON 每行可被 `jq` 解析。
- **校验**：`ctest -R common_` ；`./build/dev/bin/mqde_log_demo | jq .`。

#### T-0102 common：配置加载与 Schema 校验
- **依赖**：T-0101, T-0007 **级别**：L2
- **产出**：`common/config.h`：YAML/JSON 加载 → nlohmann::json → schema 校验；环境变量展开 `${VAR}`；`$MQDE_HOME/config.yaml` 服务配置。
- **验收**：≥ 20 个非法配置用例被拒，错误包含 JSON Pointer 路径与期望类型。
- **校验**：`ctest -R common_config`。

#### T-0103 common：注册表、哈希、线程池
- **依赖**：T-0101 **级别**：L2
- **产出**：`Registry<Base>`、`MQDE_REGISTER` 宏、`content_hash`（sha256，流式）、`ThreadPool`、`CancelToken`。
- **实现要点**：静态注册需防止链接器丢弃（使用 `WHOLE_ARCHIVE` 或显式 init 函数）——在测试中覆盖。
- **验收**：重复注册同名报错；未知名称返回 `kNotFound` 并列出可用名称。
- **校验**：`ctest -R common_registry`、`ctest -R common_hash`（与 `sha256sum` 对拍）。

#### T-0104 storage：SQLite 元数据与迁移
- **依赖**：T-0101 **级别**：L2
- **产出**：`storage/db.h`；迁移 `0001_init.sql`（表：jobs、stages、events、artifacts、hw_profiles、bench_history、calibration、cache_refs）；迁移执行器。
- **实现要点**：WAL 模式；单写多读；所有写操作经事务；`events` 表按 `(job_id, seq)` 主键。
- **验收**：空库自动迁移到最新版本；重复执行幂等；并发 8 线程写入无 `SQLITE_BUSY` 失败（带重试）。
- **校验**：`ctest -R storage_db`。

#### T-0105 storage：内容寻址缓存池与 GC
- **依赖**：T-0104, T-0103 **级别**：L2
- **产出**：`storage/cas.h`：`put(dir|file) → ArtifactRef`、`get`、`exists`、`pin/unpin`、`gc(max_size)`。
- **实现要点**：写入到临时目录后原子 rename；目录产物以 manifest（文件名+哈希）求整体哈希；引用计数防止 GC 删除运行中作业引用的产物。
- **验收**：相同内容二次 put 不重复占用空间；GC 后被 pin 产物仍存在。
- **校验**：`ctest -R storage_cas`（含 kill 中断写入后的恢复用例）。

#### T-0106 CLI 骨架
- **依赖**：T-0102 **级别**：L1
- **产出**：`cpp/apps/mqde_cli/main.cpp`：子命令框架、`--version`、`--log-level`、`--json` 全局输出选项。
- **校验**：`mqde --help`、`mqde --version`。

---

### P2 —— 预评估（不需要 GPU 即可交付价值）

#### T-0201 C 硬件探针：CPU
- **依赖**：T-0002 **级别**：L2（头文件）
- **产出**：`cpp/probe_c/`：`mqde_probe.h`、CPU 探测（型号、核心/线程、频率、缓存、NUMA、指令集 flags、内存总量/可用）。
- **验收**：Linux x86、Linux ARM（可用 CI arm runner 或 QEMU）、macOS Apple Silicon、Windows 输出正确。
- **校验**：`scripts/verify_probe.sh`（对比 `lscpu`/`sysctl -a`/`wmic`）。

#### T-0202 C 硬件探针：GPU/NPU
- **依赖**：T-0201 **级别**：L1
- **产出**：NVML、ROCm-SMI、Metal、Ascend DCMI 探测（全部 `dlopen`/弱链接）；输出架构代号（sm_xx/gfxXXX/apple-mX/910B）。
- **验收**：无厂商库时不崩溃；NVIDIA 机器上显存、架构、驱动版本与 `nvidia-smi` 一致。
- **校验**：`mqde probe --json`（T-0212 后）或 `probe_demo`；无 GPU 容器中运行返回 0 个 GPU 且退出码 0。

#### T-0203 硬件画像库与 HardwareProfile
- **依赖**：T-0102, T-0201 **级别**：L1
- **产出**：`data/hardware_db/**`（至少：A100-40/80G、H100-SXM、H20、L40S、RTX 4090、RTX 3090、MI300X、M2/M3 Max、910B、通用 x86/ARM CPU）；`HardwareProfile` 加载器；探测结果 → 画像匹配（按名称/架构/显存）。
- **实现要点**：规格值必须注明来源 URL（字段 `source`）；未匹配的设备生成“探测画像”（带 `confidence: low`）。
- **验收**：本机探测结果能匹配到画像或生成降级画像。
- **校验**：`ctest -R hwdb_`（schema 校验全部 YAML + 匹配用例）。

#### T-0204 hub：HF/ModelScope 解析与元数据获取
- **依赖**：T-0102 **级别**：L2
- **产出**：`hub/`：解析 `org/name[@revision]`、本地路径；获取文件列表与大小（HF `/api/models/{id}/revision/{rev}`）；下载单个小文件（config.json、tokenizer_config.json、`*.safetensors.index.json`）；HTTP Range 读取 safetensors 头；支持 `HF_ENDPOINT` 镜像、`HF_TOKEN`、代理、超时重试。
- **实现要点**：token 永不进日志（日志打印前脱敏）；离线模式读 `~/.cache/huggingface/hub` 已有文件。
- **验收**：对 Qwen2.5-7B-Instruct 在不下载权重的情况下获取全部 tensor 形状；gated 模型无 token 时给出明确错误。
- **校验**：`ctest -R hub_`（离线用例用本地 HTTP 桩）；`MQDE_ONLINE_TESTS=1 ctest -R hub_online`（可选）。

#### T-0205 modelspec：ModelSpec 解析
- **依赖**：T-0204 **级别**：L2
- **产出**：`modelspec/`：从 config.json + safetensors 头生成 ModelSpec（A§7.2）；支持 GQA/MQA/MLA/滑窗/MoE/tie_embeddings/多模态子配置（`text_config`/`vision_config`）。
- **验收**：参考集所有模型参数量与官方值误差 < 0.5%；`tests/fixtures/configs/` ≥ 12 个架构全部解析成功。
- **校验**：`ctest -R modelspec_golden`（对照 `data/golden/modelspec/*.json`）。

#### T-0206 架构模板与 QIR
- **依赖**：T-0205 **级别**：L2
- **产出**：`data/model_families/*.yaml`（llama、qwen2、qwen3、mistral、gemma2、deepseek_v2、qwen2_vl）；`ir/qir.h`：`QUnit`、`QIRGraph`（按 modality → block → unit 分层）。
- **实现要点**：模板 = 张量名正则 → `{kind, block_idx 捕获组, modality, role(q/k/v/o/gate/up/down/...)}`；未知架构走启发式并标记低置信度。
- **验收**：Llama-3.1-8B 生成 32 blocks × 7 linear + embedding + lm_head 的 QIR；前端层级树可直接使用。
- **校验**：`ctest -R qir_`。

#### T-0207 量化配方库
- **依赖**：T-0102 **级别**：L1
- **产出**：`data/recipes/{w4a16,w4a8,w2a16,w2a8,fp8}.yaml`（A§8 表格全部字段）、`Recipe` 加载器、`overrides` 合并逻辑。
- **验收**：五个配方通过 schema；overrides 合并后仍通过校验。
- **校验**：`ctest -R recipe_`。

#### T-0208 estimator：显存 / KV / 并发预估
- **依赖**：T-0206, T-0207, T-0203 **级别**：L2
- **产出**：`estimator/`：实现 A§9.1–9.4 全部公式；`estimate`、`kv_vs_context`、`memory_vs_concurrency`；EstimateReport 含 breakdown、feasible、reasons、max_concurrency、max_context、assumptions。
- **实现要点**：纯函数；MoE、MLA、滑窗、TP、paged block 取整、keep_fp 层全部覆盖；数值单位统一为字节，展示层再转换。
- **验收**：
  1. `data/golden/estimator/` 中 ≥ 15 个手算用例完全一致（字节级）。
  2. 参考集（需 GPU，可在 T-0808 回填）显存预估误差 ≤ 10%。
  3. 100 模型 × 20 硬件 × 5 配方矩阵计算 < 1s。
- **校验**：`ctest -R estimator_`；`./build/bench/estimator_bench`。

#### T-0209 estimator：性能预估与 CalibrationStore
- **依赖**：T-0208, T-0104 **级别**：L2
- **产出**：A§9.5 屋顶线模型（TTFT/TPOT/吞吐）；`CalibrationStore`（按 hw×backend×recipe 存 η_bw、MFU、overhead，EMA 更新，初值来自 `data/backends/*.yaml`）；报告中 `confidence` 字段。
- **验收**：无历史数据时 confidence=low；写入 3 次实测后 confidence=high 且系数更新符合 EMA 公式。
- **校验**：`ctest -R estimator_perf`、`ctest -R calibration_store`。

#### T-0210 planner：兼容矩阵与 JobSpec 校验
- **依赖**：T-0207, T-0203 **级别**：L2
- **产出**：`data/compat/matrix.yaml`（覆盖 5 配方 × 5 算法 × 4 后端 × 主要架构）；`planner/validate.h`：汇总全部静态规则（A§6.5）。
- **验收**：`data/golden/compat/` ≥ 50 条用例通过；每个拒绝都带修复建议。
- **校验**：`ctest -R planner_`；`mqde run examples/jobs/invalid_*.yaml --dry-run` 在 5s 内失败并提示。

#### T-0211 planner：推荐与可部署矩阵
- **依赖**：T-0208, T-0210 **级别**：L1
- **产出**：`recommend(input, constraints)` 返回 Top-K（配方, 算法, 后端, 硬件, TP）组合及理由；`estimate_matrix`。
- **验收**：给定“24GB 显存、上下文 8K、并发 8”能对 7B 模型推荐 W4A16/FP8 组合并排除不可行项。
- **校验**：`ctest -R planner_recommend`。

#### T-0212 CLI：probe / model inspect / estimate / plan
- **依赖**：T-0106, T-0211, T-0202 **级别**：L1
- **产出**：A§6.16 中前四个子命令；人类可读表格输出 + `--json`。
- **验收**：**M1 演示场景**通过。
- **校验**：`tests/integration/cli_estimate_test.sh`（离线 fixtures）；截图/输出贴 PR。

---

### P3 —— API 服务与预估可视化

#### T-0301 server：服务骨架与只读端点
- **依赖**：T-0212, T-0007 **级别**：L2
- **产出**：`mqde serve`；`/health`、`/version`、`/hardware`、`/hardware/profiles`、`/recipes`、`/algorithms`、`/backends`、`/datasets`；请求/响应按 OpenAPI 校验；统一错误体 `{code, message, details}`；CORS（仅开发）。
- **验收**：默认只监听 127.0.0.1；所有端点契约测试通过。
- **校验**：`ctest -R server_`；`scripts/contract_test.sh`。

#### T-0302 server：预估类端点
- **依赖**：T-0301 **级别**：L1
- **产出**：`/models:resolve`（带缓存）、`/estimate`、`/estimate:matrix`、`/estimate:curves`、`/plan:recommend`。
- **验收**：`/estimate` p99 < 50ms（ModelSpec 已缓存时）。
- **校验**：契约测试 + `wrk`/`hey` 压测结果贴 PR。

#### T-0303 前端：API 客户端、布局、路由
- **依赖**：T-0004, T-0301 **级别**：L1
- **产出**：`web/src/api/generated`（生成）、`client.ts`、全局布局（侧边导航：新建作业/预估中心/作业/对比/硬件/缓存）、i18n、暗色模式。
- **校验**：`pnpm typecheck`；`scripts/gen_api.sh --check`。

#### T-0304 前端：新建作业向导 + 实时预估侧栏
- **依赖**：T-0303, T-0302 **级别**：L1
- **产出**：A§6.15 “新建作业”五步向导：模型（HF id 输入 + 解析结果卡片）、配方（W4A8/W4A16/W2A16/W2A8/FP8 卡片式选择 + 算法 + 高级参数折叠）、评估（数据集/任务多选、自定义上传）、硬件与后端（本机探测 + 画像库）、运行参数与约束；右侧实时预估（debounce 300ms）；最终生成 JobSpec YAML 预览与下载。
- **验收**：非法组合即时提示（来自 planner）；生成的 YAML 可直接被 `mqde run --dry-run` 接受。
- **校验**：`pnpm test`；Playwright `wizard.spec.ts`。

#### T-0305 前端：预估中心 5 类图表
- **依赖**：T-0303, T-0302 **级别**：L1
- **产出**：`charts/`：MemoryStackBar（显存构成 vs 容量线）、KvContextCurve（多配方）、ConcurrencyCurve（标注最大并发）、FeasibilityHeatmap（模型×硬件，按配方切换）、PerfEstimateCard（TTFT/TPOT/吞吐 + 置信度）。
- **实现要点**：图表组件只接收纯数据 props；单位自动换算（B/MB/GB）；tooltip 显示公式分项。
- **验收**：与 CLI 同输入结果数值一致。
- **校验**：vitest 快照 + Playwright `estimate.spec.ts`。

#### T-0306 前端：硬件页
- **依赖**：T-0303 **级别**：L1
- **产出**：本机设备卡片、画像库表格、（T-0805 后）微基准按钮与结果。
- **校验**：Playwright `hardware.spec.ts`。

---

### P4 —— 量化数值核心（C++）

#### T-0401 QuantSpec / QParams / 粒度重排
- **依赖**：T-0101 **级别**：L2
- **产出**：`quant/{quant_spec.h,qparams.h,tensor_view.h}`；`TensorView`（指针+shape+stride+dtype，支持 fp32/fp16/bf16/fp8/int8/int32）；粒度重排 per_tensor/channel/group/head/block/token（含 padding 策略）。
- **验收**：非整除 group_size 在 `allow_padding=false` 时返回明确错误。
- **校验**：`ctest -R quant_reshape`。

#### T-0402 标定算法
- **依赖**：T-0401 **级别**：L1
- **产出**：MinMax、MSE 网格搜索（maxshrink/grid 可配）、百分位、直方图（KL）、Moving Average（静态激活）。
- **验收**：MSE 标定误差 ≤ MinMax 标定误差（在含离群值的合成数据上显著更小）。
- **校验**：`ctest -R quant_calib`。

#### T-0403 INT/FP8 量化、反量化、假量化 + golden
- **依赖**：T-0402 **级别**：L2（golden 数据）
- **产出**：`quant/ops.h`；对称/非对称；round-half-to-even；FP8 E4M3/E5M2（饱和与 NaN 规则与 PyTorch 一致）；`scripts/gen_golden_quant.py`（用独立 numpy 参考实现生成）+ `data/golden/quant/*.npz`；`docs/guides/quant-numerics.md`（数值规范）。
- **验收**：C++ 输出与 golden **逐位一致**（INT 路径 q 值与 scale 完全相等；FP8 位模式一致）。
- **校验**：`ctest -R quant_golden`。

#### T-0404 误差度量
- **依赖**：T-0401 **级别**：L1
- **产出**：`quant/metrics.h`：mse、rmse、sqnr_db、cosine、max_abs、rel_fro、outlier_ratio（|x| > kσ）、kurtosis、直方图。
- **校验**：`ctest -R quant_metrics`（与 numpy 对拍，相对误差 < 1e-6）。

#### T-0405 位打包
- **依赖**：T-0403 **级别**：L1
- **产出**：`quant/pack.h`：GPTQ int4（int32 打包，列序）、AWQ int4（交错顺序）、Marlin 布局（参考 vLLM 实现）、GGUF Q4_0/Q4_K/Q8_0/Q2_K 块格式、通用 int2 打包；每种格式的解包函数。
- **验收**：pack→unpack round-trip 无损；GGUF 产物能被 `llama.cpp` 的 `gguf-py` 读取且反量化结果一致。
- **校验**：`ctest -R quant_pack`；`pytest python/tests/test_pack_compat.py`（与上游参考实现对拍，可选依赖）。

#### T-0406 SIMD/CUDA 加速
- **依赖**：T-0403 **级别**：L1
- **产出**：`quant/kernels/`：AVX2/AVX-512/NEON 运行时分派；`MQDE_WITH_CUDA` 下 CUDA 实现（qdq、MSE 标定、打包）。
- **验收**：结果与标量实现逐位一致；4096×4096 per_group W4 MSE 标定 CPU 单线程 < 2s，CUDA < 50ms（A100）。
- **校验**：`ctest -R quant_kernels`；`build/bench/quant_bench`。

#### T-0407 pybind11 绑定 `mqde._core`
- **依赖**：T-0403, T-0404, T-0003 **级别**：L2
- **产出**：`cpp/bindings/python/core_module.cpp`；通过 DLPack 接收/返回 torch/numpy 张量（零拷贝）；暴露 compute_qparams / quantize / dequantize / fake_quant / pack / measure / estimate（便于 Python 侧复用预估）；`_core.pyi` 类型存根。
- **验收**：torch CPU 张量调用无拷贝（地址校验）；GIL 在长计算中释放。
- **校验**：`pytest python/tests/test_core_binding.py`。

#### T-0408 Torch 快速路径与双端 golden 对拍
- **依赖**：T-0407 **级别**：L1
- **产出**：`python/mqde/quant/numerics.py`（纯 torch，GPU 可用）；`quantizer.py` 按设备选择 `_core` 或 torch 路径。
- **验收**：torch 路径（CPU 与 CUDA）与 golden 逐位一致；与 `_core` 对拍 1000 组随机张量一致。
- **校验**：`pytest python/tests/test_numerics_golden.py`（CUDA 用例在无 GPU 时 skip 并标注）。

---

### P5 —— 编排、执行器与最小闭环

#### T-0501 Worker RPC 协议与 C++ WorkerClient
- **依赖**：T-0103, T-0007 **级别**：L2
- **产出**：`worker/transport.h`（`IWorkerTransport`，stdio 实现）、`worker_client.h`（请求/响应关联、通知回调、超时、取消、心跳、版本协商）。
- **验收**：A§6.9 验收（10 万事件无丢失无乱序；kill -9 3 秒内感知）。
- **校验**：`ctest -R worker_`（使用 `echo_worker.py`）。

#### T-0502 Python Worker 进程框架
- **依赖**：T-0501, T-0003 **级别**：L1
- **产出**：`python -m mqde.worker`：RPC 循环、handler 注册、`EventEmitter`（progress/log/metric/trace/artifact）、取消检查点、stderr 日志、异常 → RPC 错误（含 traceback 摘要）。
- **校验**：`pytest python/tests/worker`；`ctest -R worker_python_integration`。

#### T-0503 pipeline：DAG、Stage 接口、状态机
- **依赖**：T-0104, T-0105 **级别**：L2
- **产出**：`pipeline/{job.h,stage.h,orchestrator.h}`；JobSpec → DAG 构建（按 `evaluation.stages` 裁剪）；阶段指纹与缓存命中跳过；状态持久化；EventSink（写库 + 广播）。
- **验收**：相同 JobSpec 二次运行所有可缓存阶段命中；中途失败可 `retry --from-stage`。
- **校验**：`ctest -R pipeline_`（FakeStage）。

#### T-0504 队列、调度、设备分配、重试/恢复
- **依赖**：T-0503, T-0202 **级别**：L1
- **产出**：优先级队列；调度器依据 EstimateReport 的显存需求与设备空闲显存分配 GPU（设置 `CUDA_VISIBLE_DEVICES`）；重试策略（可重试错误码、最大次数、退避）；服务重启恢复。
- **校验**：`tests/integration/fault_injection_test.cpp`（worker 崩溃、超时、服务重启）。

#### T-0505 模型下载（S02）
- **依赖**：T-0204, T-0503 **级别**：L1
- **产出**：并发分块下载、断点续传、sha256 校验（HF 提供 etag/LFS oid 时）、进度事件、兼容 HF 缓存目录布局（与 `huggingface_hub` 互通）；失败回退到 Python `huggingface_hub`（Xet 等特殊存储）。
- **验收**：中断后续传；下载进度在事件流中每秒更新。
- **校验**：`ctest -R hub_download`（本地 HTTP 桩，模拟断线）。

#### T-0506 Python 模型抽象层
- **依赖**：T-0502, T-0206 **级别**：L2
- **产出**：`models/base.py`（`get_blocks / get_block_linears / get_attn_in_block / replace_module_subset / collect_first_block_input / set_modality / find_blocks`）、`hf_decoder.py`（读 model_families YAML 通用适配）、`catcher.py`（自定义 `CatcherStop` 异常，不吞其他异常）、`_ALL_LINEAR_TYPES_` 注册机制；`model.inspect` handler。
- **验收**：Llama/Qwen2/Mistral/Gemma2 的 tiny 随机模型均能正确枚举 block 与 linear，且与 C++ QIR 的单元名一致。
- **校验**：`pytest python/tests/models`（含与 QIR 对照测试）。

#### T-0507 校准数据与首块捕获
- **依赖**：T-0506 **级别**：L1
- **产出**：`data/`：wikitext2、c4、pileval、自定义 JSONL 校准集；chat template 选项；bs=-1/1/N 组织方式；`first_block_input` 持久化为 safetensors（可缓存）；data-free 模式。
- **校验**：`pytest python/tests/data`。

#### T-0508 Blockwise 引擎 + Fake/Real 模块 + RTN
- **依赖**：T-0507, T-0408 **级别**：L2
- **产出**：`quant/base.py`（BaseBlockwiseQuantization：`set_quant_config / w_qdq / a_qdq / w_q / get_replacement_params / block_forward / deploy_all_modality`）、`blockwise.py`（逐块、CPU offload、多卡按块放置）、`modules/`（FakeQuantLinear、EfficientFakeQuantLinear、OriginFloatLinear、RealQuantLinear 基类）、`algorithms/rtn.py`、`quant.run` handler；QParams 导出为 safetensors artifact。
- **实现要点**：严格遵循原设想文档中的七层设计；替换操作可重复（幂等）；每个 block 处理后释放显存。
- **验收**：tiny-llama RTN-W4A16 fake 模型前向与“手动逐层 fake_quant 权重”的参考结果一致（allclose 1e-6）；Qwen2.5-0.5B CPU 完成 < 10 分钟。
- **校验**：`pytest python/tests/quant/test_rtn.py`；`pytest python/tests/quant/test_replace_idempotent.py`。

#### T-0509 误差追踪 trace
- **依赖**：T-0508, T-0404 **级别**：L1
- **产出**：`trace/`：A§10.1 维度 B 全部指标；列式写出；`event.trace` 摘要（每 block 一条）。
- **验收**：开启 trace 额外耗时 ≤ 量化总时间 15%；指标与离线重算一致。
- **校验**：`pytest python/tests/trace`。

#### T-0510 评测：PPL + logits 探针
- **依赖**：T-0506 **级别**：L1
- **产出**：`eval/ppl.py`（滑窗 PPL，stride 可配）、`eval/logits_probe.py`（固定 128 条 prompt，保存 top-k logits；计算 KL、Top-1 一致率）、`eval.run` handler。
- **验收**：Qwen2.5-0.5B wikitext2 PPL 与公开值/`lm-eval` 结果偏差 < 1%。
- **校验**：`pytest python/tests/eval`。

#### T-0511 阶段实现 S00–S08 + S14 并串通
- **依赖**：T-0504, T-0505, T-0509, T-0510 **级别**：L1
- **产出**：`pipeline/stages/`：validate、preflight、fetch_model、inspect、calib_prepare、eval_baseline、quantize、eval_fake、report(JSON)；`mqde run job.yaml`。
- **验收**：`mqde run examples/jobs/tiny_w4a16_rtn.yaml` 在 CPU 上端到端成功；二次运行全部缓存命中且 < 10s。
- **校验**：`pytest tests/integration/test_e2e_tiny.py`。

#### T-0512 server：作业端点 + WebSocket
- **依赖**：T-0511, T-0301 **级别**：L2
- **产出**：`/jobs*`、`/jobs/{id}/events`、`/jobs/{id}/error-trace`、`/report`、`/artifacts`；WS `/jobs/{id}/stream`（after_seq 续传）、`/stream`。
- **验收**：断开 WS 5 秒后重连，事件无缺失无重复；推送延迟 < 200ms。
- **校验**：`ctest -R server_jobs`；`tests/integration/ws_resume_test.py`。

#### T-0513 前端：作业执行页
- **依赖**：T-0512, T-0303 **级别**：L1
- **产出**：作业列表；作业详情：DAG 进度图（状态色、耗时、缓存命中徽标）、实时日志面板（级别/阶段过滤、搜索、自动滚动）、资源实时曲线、取消/重试按钮。
- **校验**：Playwright `job_run.spec.ts`（mock WS 回放事件）。

#### T-0514 前端：量化误差页（初版）
- **依赖**：T-0512, T-0509 **级别**：L1
- **产出**：LayerMetricHeatmap（层×指标）、BlockErrorCurve、Top-K 最差层表格、基线 vs fake 指标对比卡。
- **校验**：vitest + Playwright `error_trace.spec.ts`。

#### T-0515 M3 最小闭环验收
- **依赖**：T-0513, T-0514 **级别**：L1
- **产出**：`docs/guides/quickstart.md`；演示录屏或截图集（放 PR）。
- **验收**：M3 演示场景完整走通。
- **校验**：按 quickstart 在干净环境执行。

---

### P6 —— 算法与配方全覆盖

#### T-0601 AWQ
- **依赖**：T-0508 **级别**：L1
- **产出**：`algorithms/awq.py`：激活感知 scale 搜索（grid）、`scale_ln_fcs / scale_fc_fc` 等效变换工具（放 `quant/transforms.py`）、可选 clip 搜索。
- **验收**：Llama-3.1-8B / Qwen2.5-7B W4A16 g128 wikitext2 PPL 与 AutoAWQ 参考结果差 < 1%。
- **校验**：`pytest python/tests/quant/test_awq.py`（tiny 正确性）+ GPU 参考结果贴 PR。

#### T-0602 GPTQ
- **依赖**：T-0508 **级别**：L1
- **产出**：`algorithms/gptq.py`：Hessian 累积 hook、阻尼、act_order、静态分组、块内列更新；显存优化（逐 subset 处理）。
- **验收**：7B W4A16 单卡 A100 < 30 分钟；PPL 劣化 < 5%；与 GPTQModel 参考结果差 < 1%。
- **校验**：同上。

#### T-0603 SmoothQuant
- **依赖**：T-0508 **级别**：L1
- **产出**：`algorithms/smoothquant.py`（alpha 可配）；启用 S06 transformed 位点。
- **验收**：transformed 模型与原模型输出 max_abs 差 < 1e-3（fp32）——即等效性验证。
- **校验**：`pytest python/tests/quant/test_smoothquant_equiv.py`。

#### T-0604 激活量化 → W4A8、W2A8
- **依赖**：T-0603 **级别**：L1
- **产出**：动态 per_token、静态 per_tensor（校准收集 + buffer）；`a_qdq` 全路径；W4A8 = SmoothQuant + GPTQ(W4) + A8 per_token 默认组合。
- **验收**：W4A8 7B PPL 劣化 ≤ 8%。
- **校验**：`pytest python/tests/quant/test_act_quant.py` + GPU 结果。

#### T-0605 FP8 配方
- **依赖**：T-0508 **级别**：L1
- **产出**：FloatQuantizer（E4M3/E5M2），per_channel / per_block(128×128) 权重，动态 per_token 激活；`Fp8Linear` 模块；已是 FP8 的模型加载（`weight_scale_inv`）。
- **验收**：FP8 7B PPL 劣化 ≤ 1%。
- **校验**：`pytest python/tests/quant/test_fp8.py`。

#### T-0606 2bit 配方与 HQQ
- **依赖**：T-0602, T-0604 **级别**：L1
- **产出**：W2A16/W2A8 配方联调；`algorithms/hqq.py`（半二次近端优化，data-free）；UI/报告实验标识。
- **验收**：W2A16 在 7B 上能跑完并产出完整误差报告（不设精度门槛）。
- **校验**：`pytest python/tests/quant/test_hqq.py`。

#### T-0607 KV Cache 量化
- **依赖**：T-0508 **级别**：L1
- **产出**：`quant/kv/`：KV_REGISTRY、Naive（int8/int4/fp8，per_head/per_token）、Prefill/Decode 分支；`fake_quant_wo_kv` 位点；预估器 KV 字节数联动。
- **验收**：KV int8 下 PPL 劣化 < 1%；`fake_quant` vs `fake_quant_wo_kv` 差值出现在误差页。
- **校验**：`pytest python/tests/quant/test_kv.py`。

#### T-0608 混合精度与 keep_fp
- **依赖**：T-0509 **级别**：L1
- **产出**：`mixed_precision.py`（MixedPrecisionPolicy：按层名/块索引/敏感度指定 bit）；改 bit 时同步 qmin/qmax（加断言测试）。
- **校验**：`pytest python/tests/quant/test_mixed_precision.py`。

#### T-0609 外部算法适配器
- **依赖**：T-0508 **级别**：L3（引入外部依赖，需 ADR）
- **产出**：ADR；`algorithms/external.py`：将 GPTQModel / llm-compressor 包装为注册算法（`external:gptqmodel`），输出统一 QParams 与 trace。
- **验收**：同配置下与自研 GPTQ 结果在报告中可并排对比。
- **校验**：`pytest python/tests/quant/test_external.py`（依赖缺失时 skip）。

#### T-0610 算法组合与 QuaRot（扩展）
- **依赖**：T-0601, T-0602 **级别**：L1
- **产出**：算法链 `awq+gptq`（顺序执行多个 blockwise opt）；`algorithms/quarot.py` + `RotateLinear`（Hadamard，在线旋转）。
- **校验**：`pytest python/tests/quant/test_chain.py`、`test_quarot_equiv.py`。

---

### P7 —— 导出与部署后端

#### T-0701 Exporter 接口 + vLLM 导出
- **依赖**：T-0405, T-0602 **级别**：L2
- **产出**：`export/base.py`（Exporter 接口 + EXPORTER_REGISTRY + 导出前后端约束检查）；`export/vllm.py`：compressed-tensors 格式（W4A16/W8A8/FP8/W4A8 视 vLLM 支持）与 GPTQ/AWQ 兼容格式；`export.run` handler（S09）。
- **验收**：导出的 Qwen2.5-7B W4A16 / FP8 可被 `vllm serve` 直接加载。
- **校验**：`pytest python/tests/export/test_vllm_format.py`（格式/config 检查，CPU）+ GPU 加载 smoke。

#### T-0702 Fake/Real 一致性检查（S10）
- **依赖**：T-0701 **级别**：L1
- **产出**：`real.check` handler：用参考反量化（`_core.dequantize` + unpack）重建权重，与 fake 权重逐层比较；在 logits 探针集上比较 fake 与 real 参考前向。
- **验收**：一致时 max_abs 差 = 0（INT 路径）；人为破坏打包（测试开关）必被检出。
- **校验**：`pytest python/tests/export/test_real_check.py`。

#### T-0703 部署后端接口 + vLLM 适配器（S11）
- **依赖**：T-0701, T-0504 **级别**：L2
- **产出**：`backend/deploy_backend.h`；`backend/vllm/`：生成启动参数、子进程启动、健康检查、日志采集、超时/失败诊断（OOM、不支持的量化格式等关键字识别）、关闭；`data/backends/vllm.yaml`。
- **验收**：部署 7B W4A16 与 FP8 成功，smoke 请求返回合理文本；启动参数写入报告。
- **校验**：`ctest -R backend_`（桩）；`tests/integration/gpu/deploy_vllm.sh`。

#### T-0704 SGLang
- **依赖**：T-0703 **级别**：L1
- **产出**：`export/sglang.py`（多数情况复用 vLLM 格式）、`backend/sglang/`。
- **校验**：`tests/integration/gpu/deploy_sglang.sh`。

#### T-0705 llama.cpp（GGUF）
- **依赖**：T-0703, T-0405 **级别**：L1
- **产出**：`export/gguf.py`（使用 `_core.pack` 的 GGUF 块格式 + gguf 元数据写出）、`backend/llamacpp/`（`llama-server`，支持 CPU / Metal / CUDA）。
- **验收**：在 Apple Silicon 与纯 CPU 上部署 W4A16(Q4_K) 成功；这也是**无 NVIDIA GPU 环境下的完整闭环路径**。
- **校验**：`tests/integration/deploy_llamacpp_cpu.sh`（CI 可跑 tiny 模型）。

#### T-0706 TensorRT-LLM（扩展）
- **依赖**：T-0703 **级别**：L1
- **产出**：`export/trtllm.py`（checkpoint 格式，W4A16/W4A8/FP8）、`backend/trtllm/`（build engine + serve）。
- **校验**：`tests/integration/gpu/deploy_trtllm.sh`。

---

### P8 —— 评估与压测

#### T-0801 精度评测适配
- **依赖**：T-0510 **级别**：L3（引入 lm-eval/evalscope 依赖，需 ADR）
- **产出**：`eval/lm_eval_adapter.py`、`eval/evalscope_adapter.py`、`eval/custom.py`（JSONL：prompt/answer/metric ∈ exact_match/contains/regex/rouge）；`data/datasets/eval_suites.yaml`（quick/standard 两档）。
- **验收**：MMLU/GSM8K 结果与 lm-eval 官方 CLI 直接运行结果一致。
- **校验**：`pytest python/tests/eval/test_adapters.py`（小 limit）。

#### T-0802 部署后精度评测（S12）
- **依赖**：T-0801, T-0703 **级别**：L1
- **产出**：评测适配器支持 OpenAI 兼容端点作为模型（lm-eval `local-completions`）；logits 探针在后端支持 logprobs 时使用 top-logprobs 计算近似 KL。
- **校验**：`tests/integration/gpu/eval_deployed.sh`；CPU 上用 llama.cpp tiny 模型跑通。

#### T-0803 性能压测器（S13）
- **依赖**：T-0703 **级别**：L1
- **产出**：`eval/bench`（C++）：流式 SSE 客户端、并发控制（固定并发 / 泊松到达率）、输入长度控制（按 tokenizer 构造定长 prompt）、指标 TTFT/TPOT/ITL/吞吐/req/s 的 p50/p90/p99。
- **验收**：对 mock backend（固定 TTFT=100ms、TPOT=20ms）测量误差 < 2ms；自身开销 < 1%。
- **校验**：`ctest -R bench_`（使用 `tests/fixtures/mock_backend`）。

#### T-0804 显存采样与 OOM 边界搜索
- **依赖**：T-0803, T-0202 **级别**：L1
- **产出**：压测期间 100ms 采样设备显存/主机内存/功耗；OOM 边界：固定上下文二分最大并发、固定并发二分最大稳定上下文（每次失败自动重启后端）。
- **验收**：OOM 边界结果与预估器 max_concurrency 同时出现在报告中。
- **校验**：`tests/integration/gpu/oom_boundary.sh`。

#### T-0805 HAL 微基准
- **依赖**：T-1101 **级别**：L1
- **产出**：插件 `profile()`：显存带宽（大块 memcpy）、各 dtype GEMM TFLOPS；结果写入硬件画像 `measured` 字段与 CalibrationStore。
- **校验**：与厂商标称值比较（带宽应在标称 70–95% 之间），结果贴 PR。

#### T-0806 报告生成
- **依赖**：T-0802, T-0804 **级别**：L1
- **产出**：JobReport JSON（schema 校验）+ 自包含 HTML（内嵌图表，可离线打开）；`mqde report`。
- **校验**：`ctest -R report_`；HTML 在无网络下可打开。

#### T-0807 前端：评估报告页 + 对比页
- **依赖**：T-0806 **级别**：L1
- **产出**：报告页（精度雷达、性能曲线、OOM 边界、预估 vs 实测、环境指纹）；对比页（帕累托散点：精度–显存–吞吐，多选作业）。
- **校验**：Playwright `report.spec.ts`、`compare.spec.ts`。

#### T-0808 M4 验收：全配方 × 多后端参考集
- **依赖**：T-0807, T-0606, T-0705 **级别**：L1
- **产出**：`docs/reports/m4-reference.md`：参考集（A§11.3 中、小模型）× 五配方 × (vLLM, SGLang, llama.cpp) 的精度/性能/预估误差总表。
- **验收**：A§11.4 精度阈值达标；显存预估误差 ≤ 10%；不达标项有 Issue 跟踪。
- **校验**：`scripts/bench_reference.sh --suite m4`。

---

### P9 —— 误差归因与反馈闭环

#### T-0901 阶段归因数据链路
- **依赖**：T-0702, T-0802 **级别**：L1
- **产出**：各阶段统一在同一评估集与 logits 探针上产出 `StageMetric`；报告与 API 输出 Δ_stage 及自动判定（等效变换异常 / 导出异常 / 后端内核误差）。
- **校验**：`pytest python/tests/trace/test_stage_attribution.py`。

#### T-0902 敏感度分析
- **依赖**：T-0509 **级别**：L1
- **产出**：可选的逐层回退 FP 评估（按 block 或按 unit，支持采样与并行），输出 sensitivity 排序。
- **验收**：7B 按 block 粒度敏感度分析 < 1 小时（单卡）。
- **校验**：`pytest python/tests/trace/test_sensitivity.py`（tiny）。

#### T-0903 前端：误差页完整版
- **依赖**：T-0901, T-0902 **级别**：L1
- **产出**：StageWaterfall（自动高亮最大 Δ）、层级树（modality→block→unit）联动热力图、单元详情抽屉（权重/激活直方图、离群值、QParams 分布）、敏感度排序视图。
- **校验**：Playwright `error_trace_full.spec.ts`。

#### T-0904 混合精度建议 → 一键新作业
- **依赖**：T-0903, T-0608 **级别**：L1
- **产出**：根据 Top-K 敏感层生成 `mixed_precision` 覆盖配置，预估其显存增量，一键克隆作业并提交。
- **校验**：Playwright `mixed_precision_suggest.spec.ts`。

#### T-0905 反馈闭环（S15）
- **依赖**：T-0209, T-0804 **级别**：L1
- **产出**：S15 阶段：用实测（峰值显存、TTFT、TPOT、吞吐）反推 overhead/η/MFU，更新 CalibrationStore；前端展示“预估误差随历史次数下降”曲线。
- **验收**：同一 (hw, backend, recipe) 跑 3 个不同模型后，第 4 个模型的吞吐预估误差 ≤ 25%。
- **校验**：`ctest -R feedback_`；参考集结果贴 PR。

#### T-0906 误差归因验收（注入测试）
- **依赖**：T-0903 **级别**：L1
- **产出**：`debug.inject_noise` 与 `debug.corrupt_pack` 测试开关（仅测试构建/显式开启时生效）；集成测试。
- **验收**：A§10.2 两条验收均通过（注入层 Top-1；破坏打包时 fake→real 为最大误差环节）。
- **校验**：`pytest tests/integration/test_error_attribution.py`。

---

### P10 —— 多模态与复杂结构

#### T-1001 多模态框架
- **依赖**：T-0508 **级别**：L2
- **产出**：JobSpec `quant.per_modality`；BaseModel `set_modality`；校准数据支持图文对；QIR 多模态分组；预估器分模态统计。
- **校验**：`pytest python/tests/models/test_modality.py`。

#### T-1002 VLM 适配
- **依赖**：T-1001 **级别**：L1
- **产出**：Qwen2-VL、LLaVA 适配器；视觉塔默认 W8A16/RTN 或保持 FP；VLM 评测（MMMU/自定义图文 JSONL，经 evalscope）。
- **验收**：Qwen2-VL-2B language=W4A16-GPTQ + vision=RTN-W8 全流程跑通，并可 vLLM 部署。
- **校验**：GPU 集成脚本 + 报告。

#### T-1003 MoE / MLA 支持
- **依赖**：T-0508, T-0208 **级别**：L1
- **产出**：MoE 专家线性层枚举与校准（按路由统计专家样本不足时的处理策略）；MLA KV 预估与 KV 量化适配；DeepSeek-V2-Lite 端到端。
- **校验**：`pytest python/tests/models/test_moe.py`；GPU 集成。

#### T-1004 VLA 适配（实验）
- **依赖**：T-1002 **级别**：L1
- **产出**：以 OpenVLA 类模型为例：视觉编码器 + LLM 主干 + 动作头（动作头默认保持 FP）；评测：离线动作预测误差（数据集回放）。
- **校验**：实验报告 `docs/reports/vla-experiment.md`。

#### T-1005 世界模型 / 视频生成（DiT）适配（实验）
- **依赖**：T-1001 **级别**：L1
- **产出**：DiT block 适配（时间步条件、adaLN）；评测：生成质量代理指标（与 FP 输出的 PSNR/LPIPS）、显存/速度。
- **校验**：实验报告 `docs/reports/dit-experiment.md`。

---

### P11 —— 硬件插件

#### T-1101 HAL ABI 与插件加载器 + 模板插件
- **依赖**：T-0201 **级别**：L2
- **产出**：`hal/mqde_hal.h`（A§6.11）、`plugin_loader`（扫描、版本检查、符号检查、隔离失败）、`cpp/plugins/template/`（模拟设备插件）。
- **验收**：模板插件不改上层代码即出现在 `mqde probe` 与前端硬件页。
- **校验**：`ctest -R hal_`；`scripts/check_abi.sh`。

#### T-1102 CPU / CUDA 插件
- **依赖**：T-1101 **级别**：L1
- **产出**：`plugins/cpu`（带宽/GEMM 微基准）、`plugins/cuda`（`dlopen` libcudart/libcublas，微基准）。
- **校验**：T-0805 校验方式。

#### T-1103 / T-1104 / T-1105 Metal / ROCm / Ascend 插件（扩展）
- **依赖**：T-1101 **级别**：L1
- **产出**：对应插件 + 硬件画像 + 兼容矩阵条目 PR（矩阵部分 L2）。
- **校验**：在对应硬件上 `mqde probe --json` 与微基准结果贴 PR。

#### T-1106 插件 SDK 文档与 ABI 兼容检查
- **依赖**：T-1101 **级别**：L1
- **产出**：`docs/guides/add-hardware-plugin.md`（从模板到上线的完整步骤）、CI 中 `check_abi.sh`。
- **校验**：由一个未参与开发的 Agent 按文档新建一个模拟插件并通过测试。

---

### P12 —— 交付与发布

#### T-1201 Docker 镜像与 compose
- **依赖**：T-0515 **级别**：L2
- **产出**：`deploy/docker/Dockerfile.cpu`、`Dockerfile.cuda`、`compose/docker-compose.yml`；多阶段构建；非 root 用户运行。
- **验收**：`docker compose up` 后浏览器访问即可跑通 M3 演示；cpu 镜像 < 3GB。
- **校验**：`release.yml` 中镜像 smoke 测试。

#### T-1202 pip wheel 打包与发布
- **依赖**：T-0515 **级别**：L2
- **产出**：cibuildwheel 构建 manylinux x86_64/aarch64、macOS arm64 wheel；`mqde` CLI 随 wheel 安装。
- **校验**：干净 venv 中 `pip install dist/*.whl && mqde estimate ...`。

#### T-1203 文档站
- **依赖**：T-0008 **级别**：L1
- **产出**：mkdocs-material 站点（介绍、快速开始、架构、配方说明、各类 add-* 指南、API 参考），发布到 `zhipingwoods.github.io/MQDEsys`。
- **校验**：`mkdocs build --strict`。

#### T-1204 安全加固
- **依赖**：T-0512 **级别**：L3
- **产出**：ADR；凭证存储与脱敏审计（grep 测试确认日志/报告/DB 无 token）；`allowed_paths` 沙箱；server token 认证（对外监听时强制）；`trust_remote_code` 二次确认。
- **校验**：`tests/integration/test_security.py`；`/security-review` 结果。

#### T-1205 可观测性
- **依赖**：T-0511 **级别**：L1
- **产出**：`/metrics`（Prometheus）；`mqde run --reproduce <job_id>`。
- **校验**：重放作业的报告与原报告（除时间戳外）一致。

#### T-1206 夜间 GPU 回归与性能看板
- **依赖**：T-0808 **级别**：L2
- **产出**：`nightly.yml`（自托管 GPU runner）；结果写入 `bench_history` 并生成趋势页（前端或静态页）。
- **校验**：连续 3 晚产出结果。

#### T-1207 v1.0 发布验收
- **依赖**：全部必选任务 **级别**：L2
- **产出**：`CHANGELOG.md` v1.0.0、发布说明、tag `v1.0.0`。
- **验收**：M0–M7 全部演示场景复现通过；A§11.2 CI 门禁全绿；A§11.4 精度阈值达标；已知问题列表。
- **校验**：发布检查清单（`docs/guides/release-checklist.md`）逐项勾选。

---

## 附录 R：参考架构对齐任务

> 依据 [REFERENCES.md](REFERENCES.md)。llmfit 相关任务的公式与常数已抄录在 REFERENCES.md **附录 A**，
> 实现时**不需要再读 3rdpart 源码**，按附录 A 的章节号对照即可。
> 状态更新为 L0；任务定义（产出/验收）修改为 L2。

### R.1 总览

| ID | 标题 | 参考 | 级别 | 状态 |
|----|------|------|------|------|
| T-R01 | `fit` 模块：适配等级/量化层级/运行模式/速度/评分 | llmfit | L2 | DONE (2141a7e) |
| T-R02 | `/fit`、`/fit:plan` API + 部署预评估页 | llmfit-web | L1 | DONE (7f2c8a7) |
| T-R04 | netron 式模型结构：`models:graph` / `models:tensor` + 结构页 | netron | L1 | DONE (7f2c8a7) |
| T-R16 | 前端深色控制台风格（token 化、主题切换） | refero / evalscope | L1 | DONE (7f2c8a7) |
| **T-R17** | **评分常数校准**：质量分档、量化惩罚、家族/时效/任务加分、TTFT 返回 null | llmfit A.1–A.3, A.6 | L2 | DONE (2026-09-28, 39/39 ctest) |
| **T-R18** | **MoE 两级速度模型** + 显存压力惩罚 | llmfit A.5 | L1 | DONE (2026-09-28, 45/45 ctest; 256 专家 MoE Tier-1) |
| **T-R19** | **置信度阶梯 + 本地校准**（calibrated 档） | llmfit A.7 | L1 | DONE (2026-09-28, d9f51d8; 63/63 ctest) |
| **T-R20** | **实测基线库**：格式、相同硬件匹配、中位数归并 | llmfit A.8 | L1 | DONE (2026-09-28, 56/56 ctest; 4090 实测 4.41 steps/s) |
| **T-R21** | **并发反解**：max_users / max_context_for | llmfit A.9 | L1 | DONE (2026-09-28, 69/69 ctest) |
| **T-R22** | **完整 plan**：run paths、最低/推荐硬件、升级增量、KV 替代 | llmfit A.10 | L1 | DONE (2026-09-28, a4c0299) |
| **T-R23** | **claim 边界**：保守/乐观双估 + 二分最大上下文 | llmfit A.10 | L1 | DONE (2026-09-28, 8cda15b) |
| **T-R24** | **已安装运行时探测**（8 来源并行） | llmfit A.11 | L1 | DONE (2026-09-29, 5862e3c/4767c41) |
| **T-R25** | **HF 模型库 + 搜索/下载** | llmfit A.12–A.13 | L2 | DONE (2026-09-29, b45a46c; 200-entry catalog, search, GGUF/safetensors download verified) |
| **T-R26** | **ModelFit 字段补全** + 运行时选择 | llmfit A.15–A.16 | L2 | DONE (2026-09-29, 9715e98; license field still empty for local models) |
| **T-R27** | **前端补全**：硬件模拟、对比模式、完整筛选、导出 | llmfit-web | L1 | TODO |
| T-R05 | GGUF 结构解析 | netron gguf.js | L1 | TODO |
| T-R06 | QuantizeConfig + module_tree | GPTQModel | L2 | TODO |
| T-R07 | Looper/Processor + 每模块日志表 | GPTQModel | L2 | TODO |
| T-R08 | GPTQ 算法 + AWQ Processor | GPTQModel | L1 | TODO |
| T-R09 | 导出 GPTQ v1/v2 / merged 16bit / GGUF | GPTQModel / unsloth | L2 | TODO |
| T-R10 | 动态量化（按误差自动生成 skip 名单） | unsloth | L1 | TODO |
| T-R11 | eval：TaskConfig + registry + Evaluator + Report | evalscope | L2 | TODO |
| T-R12 | perf：并发扫描 + TTFT/TPOT/ITL 百分位 | evalscope | L1 | TODO |
| T-R13 | 前端评测页：Reports / Compare / Perf | evalscope | L1 | TODO |
| T-R14 | hal：Target + Backend 接口 + 插件发现 | FlagTree | L3 | TODO |
| T-R15 | 后端插件 + 回归基线 | FlagTree | L1 | TODO |
| T-R28 | 存储估算 | llmfit A.17 | L1 | 暂不做 |

**执行顺序**：T-R17 → T-R18 → T-R20 → T-R19 → T-R21 → T-R26 → T-R22 → T-R23 → T-R24 → T-R25 → T-R27，之后进入 GPTQ（T-R06 起）。

---

### R.2 任务详情

#### T-R17 评分常数校准
- **依赖**：T-R01 **级别**：L2（改 `data/fit/fit.yaml` 与评分逻辑）
- **产出**：
  - `data/fit/fit.yaml`：质量分档改为 30/45/60/75/82/89/95（按**活跃**参数，见 A.1）；W2 惩罚 −18 → −12（A.2）；新增 `embedding` 用途（权重 .30/.40/.20/.10，速度目标 200，上下文目标 512）；新增 `family_bonus`、`recency_bonus` 表。
  - `cpp/src/fit/fit.cpp`：叠加家族/时效/任务加分并截断 [0,100]；核数 ≥ 8 时 K 常数路径乘 1.1（A.4）。
  - **TTFT**：算力未知或纯 CPU 时 `ttft_ms` / `prefill_tps` 返回 null 而非算出假值（A.6）。`ModelFit` 对应字段改 `std::optional<double>`。
- **验收**：
  - 质量分档按 llmfit 的**严格小于**：0.9B → 30，1B → 45，7B → 75，27B → 89（再加家族加分）。
  - `fit_test.cpp` 新增：质量分档边界（0.9B/1B/3B/7B/10B/20B/40B/70B）、W2 惩罚值、embedding 用途、TTFT 在无算力画像时为 null。
  - 现有 6 个 fit 测试中因分档变化而失效的断言同步更新，并在 PR 中说明新旧值。
- **校验**：`ctest --preset dev`；`mqde fit /cache/wuzhiping/hub --hw nvidia/rtx4090` 前后对比表贴进 PR。

#### T-R18 MoE 两级速度模型
- **依赖**：T-R17 **级别**：L1
- **产出**：
  - `data/fit/fit.yaml`：`moe:` 段 —— `tier1_k: 3.2`、`tier1_k_by_arch: {gpt_oss: 1.43}`、`tier1_fixed_bpp`、`tier2_by_arch`（gpt_oss .72/.80，deepseek_v3/v4 .62/.70）、`tier2_overhead_by_experts`（≤8 .90 / ≤16 .85 / ≤32 .80 / ≤64 .70 / ≥128 .40 / 未知 .60）、`default_expert_ratio: 0.50`、`vram_pressure_threshold: 0.60`、`vram_pressure_floor: 0.30`。
  - `cpp/src/modelspec/model_spec.cpp`：从 config 提取单专家 FFN 维度（`moe_intermediate_size`、`shared_expert_intermediate_size`），算出 `active_ffn_params`、`fixed_params`、`routed_expert_params` 并存入 `ModelSpec`。
  - `cpp/src/fit/fit.cpp`：按 A.5 实现 Tier-1 / Tier-2 分支与显存压力惩罚；`ModelFit` 新增 `moe_offloaded_gb`、`moe_tier`（1/2）。
- **验收**：
  - 256 专家、top-8 的 MoE 走 Tier-1；缺专家维度信息的 MoE 走 Tier-2 并按专家数取 overhead。
  - 显存利用率 0.60 以下无惩罚；0.60→1.00 线性衰减到 0.30 不再继续下降。
  - 新增测试：Tier-1 公式值、Tier-2 各 overhead 档、压力惩罚边界（0.59/0.60/0.80/1.00/1.20）。
- **校验**：`ctest`；该 MoE 模型在 4090 上的 tok/s 变化在 PR 中说明合理性。

#### T-R20 实测基线库
- **依赖**：T-R01 **级别**：L1
- **定义变更（2026-09-28）**：原"三级硬件回退"在 llmfit 源码中不存在（调研误读），改为 llmfit 实际行为：**仅相同硬件**匹配。见 REFERENCES.md A.8。
- **产出**：
  - 目录 `data/benchmarks/local/<hw-slug>/*.json`（本机实测）、`dat
...[516 chars omitted]...
配规则的偏差说明写进 PR。
- **校验**：`ctest`；在 4090 上对 MiniCPM-RobotManip 实跑一次，`mqde fit` 与前端置信度显示 `measured_local`。

#### T-R19 置信度阶梯与本地校准
- **依赖**：T-R20 **级别**：L1
- **产出**：
  - `Confidence` 增加 `kCalibrated`，阶梯 `measured_local > measured_community > calibrated > estimated > unsupported`。
  - `cpp/src/fit/fit.cpp`：`apply_local_calibration(std::vector<ModelFit>&)` —— 锚点为稠密、≥1B、有**本地**实测的条目；< 2 个锚点放弃；取 `measured/estimated` 比值中位数并截断 [0.05, 3.0]；乘到所有无实测条目上并标 `calibrated`；幂等。
  - `ModelFit.estimate_basis{method, gpu_bandwidth_gbps, ddr_bandwidth_gbps, efficiency, assumed_context, local_calibration}`，前端详情面板展示"这个数字怎么来的"。
- **验收**：
  - 锚点 < 2 时所有条目仍为 `estimated`。
  - 连续调用两次结果一致（幂等）。
  - 校准系数超出 [0.05, 3.0] 时被截断。
  - 测试覆盖上述三条 + `estimate_basis` 各字段非空。
- **校验**：`ctest`。

#### T-R21 并发反解
- **依赖**：T-R01 **级别**：L1
- **产出**：
  - `cpp/include/mqde/fit/concurrency.h` + `src/fit/concurrency.cpp`：`estimate_concurrency(spec, recipe, system, context, users)` → `ConcurrencyEstimate{requested_users, max_users, context, weights_gb, kv_per_user_gb, overhead_gb, total_gb, utilization_pct, fits, high_pressure}`；`max_context_for(users)` 反解最大上下文；利用率 > 90% 标 `high_pressure`。
  - API `POST /api/v1/fit:concurrency`；前端在详情面板加"并发能力"块（最大并发、该并发下最大上下文、内存构成）。
- **验收**：
  - 与 `estimate()` 的显存分解一致（同配方同上下文下 `total_gb` 对齐）。
  - `max_users` 反解结果代回正向计算后仍放得下，+1 则放不下。
  - `max_context_for` 同理（2 的幂步进）。
- **校验**：`ctest` 新增 concurrency 测试；前端手工核对一个例子。

#### T-R22 完整 plan
- **依赖**：T-R18, T-R21 **级别**：L1
- **产出**：`POST /api/v1/fit:plan` 扩展为返回：
  - `run_paths[]`：每种可行运行模式的量化、显存、速度、适配等级。
  - `minimum_hardware` / `recommended_hardware`：从硬件库里找出最低可跑与推荐配置。
  - `upgrades[]`：升级到某硬件后的增量（显存余量、速度提升倍数、可用更高精度量化）。
  - `kv_alternatives[]`：KV 量化换成 fp8 / q8_0 / q4_0 后的上下文与显存变化。
  - 前端详情面板内嵌 plan 表单与结果表（仿 llmfit DetailPanel 的 plan 区）。
- **验收**：三个本地模型各给出 ≥ 2 条 run path；升级增量方向正确（更大显存 → 更高精度或更大上下文）。
- **校验**：curl 三个模型的 plan 结果贴进 PR。

#### T-R23 claim 边界
- **依赖**：T-R18 **级别**：L1
- **产出**：
  - `cpp/src/fit/claim.cpp`：`fit_bounds(spec, system, min_tps)` → `{conservative_tps, optimistic_tps, holds, best_quant, max_context_at_min_tps, memory_gb, efficiency_used}`。保守/乐观用不同带宽效率假设；二分找仍达标的最大 2 的幂上下文（受模型上限约束）。
  - CLI `mqde claim <model> --min-tps N [--quant Q --device-class NAME]`；输出一句可直接用的结论。
  - API `POST /api/v1/fit:claim`。
- **验收**：保守值 ≤ 乐观值；`holds` 当且仅当保守值 ≥ min_tps；上下文二分结果 +1 档即不达标。
- **校验**：`ctest`；CLI 对 MiniCPM-RobotManip 跑一次贴输出。

#### T-R24 已安装运行时探测
- **依赖**：无 **级别**：L1
- **产出**：
  - `cpp/src/fit/installed.cpp`：按 A.11 并行探测 8 来源（ollama / LM Studio / llama.cpp / vLLM / MLX / Docker Model Runner / RamaLama / HF cache），单个 HTTP 超时 400 ms，整体总 deadline（建议 2 s）；名字归一化后建集合；`is_installed(name, ollama_name)`、`installed_providers(name)`。
  - `data/installed.yaml`：各来源的探测路径与端口（可配置，不写死）。
  - `ModelFit.installed`、`installed_providers[]`；API `GET /api/v1/installed`；前端模型表加"已安装"标记，排序时已安装优先。
- **验收**：
  - 开发机上能正确探测到实际存在的运行时；不存在的来源不报错、不拖慢（总耗时 < 2.5 s）。
  - 名字归一化能把 `qwen2.5:7b-instruct-q4_K_M` 匹配到目录里的 `Qwen2.5-7B-Instruct`。
- **校验**：`GET /api/v1/installed` 在开发机的输出贴进 PR；`time curl` 证明耗时。

#### T-R25 HF 模型库 + 搜索/下载
- **依赖**：T-R17 **级别**：L2（引入外部数据源与网络访问）
- **产出**：
  - `data/models/catalog.json`：按 A.12 的字段（7 个必填 + 可选）。**不照搬 llmfit 的 15k 条**，先用 `mqde catalog update` 抓 trending + downloads 前 N 条（N 可配，默认 500），需要时再扩。文件 > 5 MB 时走 LFS 或压缩存储。
  - `cpp/src/fit/catalog.{h,cpp}`：加载、按名字/provider 查询、目录卫生（draft/eagle 降级、参数量与 raw 不一致降级、缺尺寸或上下文降级，不参与排名）。
  - `mqde catalog update [--trending --downloads --token --limit N]`：HF API 抓列表 + 逐个取 config，字段映射见 A.12，请求间隔 250 ms，token 走环境变量 `HF_TOKEN` 或 `--token`（**禁止写入仓库**）。开发机需走代理（见 DEVELOPMENT.md）。
  - `mqde hf-search <query> [-n N]`；`mqde download <model> [--quant Q --budget SIZE --list --output-dir DIR]`：分片按数字前缀归组整下，文件大小取 HF 元数据、缺失时按 `params × bpp` 估，预算内按质量排序取最优、无解时退最小文件。
  - API `GET /api/v1/catalog?search=&provider=&limit=`、`POST /api/v1/download`、`GET /api/v1/download/{id}/status`；`/fit` 支持对目录里的模型（无本地权重）做评估。
  - 前端：模型来源切换（本地库 / HF 目录）、搜索、下载按钮与进度。
- **验收**：
  - `mqde catalog update --limit 50` 成功写入 50 条，字段完整，无凭证入库。
  - 目录里的模型能参与 `/fit` 评估（用 catalog 的层数/隐藏维等元数据，不需要下载权重）。
  - 降级规则命中示例：至少 1 条 draft 模型被排除在排名外。
  - 下载分片模型时整组下载，单文件模型按预算选中期望的量化。
- **校验**：CLI 三条命令各跑一次贴输出；`grep -r "hf_[A-Za-z0-9]" data/` 为空。

#### T-R26 ModelFit 字段补全与运行时选择
- **依赖**：T-R18, T-R20, T-R24, T-R25 **级别**：L2（公共结构与 API 契约）
- **产出**：
  - 补齐 A.15 中我们缺的字段：`moe_offloaded_gb`、`runtime`、`installed`、`usable_context`、`estimate_basis`、`measured_tps`、`disk_size_gb`、`supports_tp`、`verify_command`、`license`、`capability_ids`、`fit_label`、`run_mode_label`、`estimate_confidence_label`。
  - 运行时选择（A.16）：`InferenceRuntime = {VllmT, HfTransformers, LlamaCpp, TrtEdgeLlm, Unsupported}`（按我们的实际后端调整，不照搬 MLX/BitNet）；已量化 → vLLM；边缘设备 → TRT-Edge-LLM；否则 HF Transformers。各运行时的量化层级与固定开销从 `data/backends/*.yaml` 读。
  - `verify_command`：按运行模式产出可直接复制执行的验证命令。
  - `supports_tp`：按注意力头数算合法 TP 尺寸，不是简单按 GPU 数。
- **验收**：`/fit` 返回的每个字段都有值或明确 null；`supports_tp` 对 GQA 模型（kv_heads=4）只给出 1/2/4；`verify_command` 可直接执行。
- **校验**：`ctest`；curl 输出逐字段核对。

#### T-R27 前端补全
- **依赖**：T-R22, T-R24, T-R26 **级别**：L1
- **产出**：
  - **硬件模拟**：显存/内存/核数/带宽输入框（API 已支持，只需暴露）；内置画像下拉；"模拟中"标记与重置按钮。
  - **对比模式**：模型表多选（上限 3），并排对比面板（评分构成、显存构成、速度、适配）。
  - **完整筛选**：provider、capability、license、量化、运行模式、参数档（tiny/small/medium/large/xl）、TP 尺寸、最大上下文；折叠为"高级筛选"。
  - **导出**：当前表格导出 JSON / CSV（前端生成，不加后端端点）。
  - 列补全：运行时、已安装、磁盘占用、上下文、置信度。
- **验收**：筛选任意组合后表格与 URL query 同步（可分享链接）；导出的 CSV 能在 Excel 打开且列名为中文；对比面板在 3 个模型下不溢出。
- **校验**：`npm run build`；截图贴进 PR。
