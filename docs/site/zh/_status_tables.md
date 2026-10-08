本页由 `docs/site/scripts/gen_status.py` 根据 `docs/TASKS.md` 的总览表生成（DONE 30 项，PARTIAL 32 项，TODO 41 项，其他状态 0 项）。以仓库里的任务清单为准；清单变化后运行 `make status` 重新生成。说明列从仓库原文截取（英文）。

## PARTIAL（部分完成）

可以使用，但未完全验收。说明列写了已有的部分和缺口。

| 任务 | 标题 | 已有 / 缺口 |
|---|---|---|
| T-0001 | 仓库初始化与 Git 规范 | 2026-09-24, git init on the dev server; remote/branch protection pending |
| T-0004 | 前端工程骨架 | 2026-09-24, Vite+React+ECharts (no AntD/tests yet) |
| T-0204 | hub：HF/ModelScope 解析与元数据获取 | 2026-09-29, @revision, HF_ENDPOINT, retries, Range headers, inspect_remote; ModelScope + offline cache pending |
| T-0212 | CLI：probe / model inspect / estimate / plan | 2026-09-24, probe/inspect/estimate/recipes/hardware |
| T-0301 | server：Drogon 服务骨架与只读端点 | 2026-09-24, cpp-httplib instead of Drogon (see PROGRESS) |
| T-0302 | server：预估类端点 | 2026-09-24 |
| T-0303 | 前端：API 客户端生成、布局、路由 | 2026-09-24, hand-written typed client; generated client pending T-0007 |
| T-0304 | 前端：新建作业向导 + 实时预估侧栏 | 2026-09-24, form, no live estimate sidebar |
| T-0305 | 前端：预估中心 5 类图表 | 2026-09-24 |
| T-0306 | 前端：硬件页 | 2026-09-24 |
| T-0405 | quant 核心：位打包（GPTQ/AWQ/Marlin/GGUF） | 2026-09-30, GPTQ v1/v2 2/3/4/8, AWQ GEMM, GGUF Q8_0/Q5_0/Q4_0/IQ4_NL/Q4_K/Q2_K byte-exact vs ggml; Marlin 4/8-bit pack/unpack 2026-10-05, round-trip… |
| T-0503 | pipeline：DAG、Stage 接口、状态机 | 2026-09-24, Python-side stages; C++ DAG pending |
| T-0504 | pipeline：队列、调度、设备分配、重试/恢复 | 2026-09-24, GPU pool + queue; no retry |
| T-0510 | 评测：PPL + logits 探针 | 2026-09-29, wikitext-2 PPL 3 positions; logits probe only in tests |
| T-0511 | 阶段实现 S00–S08 + S14 并串通 | 2026-09-24, VLA stages S00-S08,S14 |
| T-0512 | server：作业端点 + WebSocket 事件流 | 2026-09-29, SSE; model.type vla/llm, CPU slot |
| T-0513 | 前端：作业执行页（DAG + 日志 + 资源） | 2026-09-24 |
| T-0514 | 前端：量化误差页（初版：热力图 + Top-K） | 2026-09-24, heatmap + Top-K + attribution |
| T-0515 | 端到端最小闭环验收（M3） | 2026-09-30, docs/guides/quickstart.md + LLM 全功能作业跑通; VLA 演示待 GPU |
| T-0603 | 算法：SmoothQuant（等效变换 + transformed 位点） | 2026-09-30, per-group scales folded into the previous op (FP-exact, exportable); transformed eval position not yet |
| T-0605 | FP8 配方（E4M3/E5M2，per_channel/per_block） | 2026-09-24, FP8 E4M3/E5M2 per-channel/per-tensor fake-quant; no per-block |
| T-0606 | 2bit 配方（W2A16/W2A8）与 HQQ | 2026-09-24, W2A16/W2A8 fake-quant via RTN-MSE; no GPTQ/HQQ |
| T-0610 | 算法组合（AWQ+GPTQ）与 QuaRot（扩展） | 2026-09-29, awq+gptq / smoothquant+gptq; no QuaRot |
| T-0701 | Exporter 接口 + vLLM 导出 | 2026-10-08: the exporter's own AWQ / compressed-tensors W4A16 / compressed-tensors FP8 checkpoints of Qwen2.5-0.5B load in vLLM 0.30.0 and serve (ser… |
| T-0703 | 部署后端接口 + vLLM 适配器（S11） | 2026-10-08, Python only: backend/base.py ServerBackend + vllm_server.py (launch, health, log diagnosis, engine_info) + llm_job deploy_vllm stage + da… |
| T-0802 | 部署后精度评测（S12） | 2026-10-08, eval/served_ppl.py: PPL from the backend's own prompt_logprobs, same windows as the fake/real PPL; vLLM only. lm-eval / evalscope adapter… |
| T-0803 | 性能压测器（S13）：TTFT/TPOT/吞吐 | 2026-10-08, Python eval/loadtest.py: closed loop + Poisson open loop, exact-length token prompts, TTFT/TPOT/ITL/E2E p50/p90/p99; against the mock (TT… |
| T-0804 | 显存采样与 OOM 边界搜索 | 2026-10-08, eval/resources.py 100 ms GPU/host sampler, eval/boundary.py bisection over concurrency / context with a fresh backend per probe; tested o… |
| T-0807 | 前端：评估报告页 + 对比页 | 2026-09-30, LLM/VLA compare tabs + llama.cpp table; standalone report page pending T-0806 |
| T-0901 | 阶段归因（瀑布）数据链路 | 2026-09-24, modality-level attribution only |
| T-0902 | 敏感度分析（逐层回退） | 2026-09-29, block/type FP fallback in llm_job; 7B GPU timing pending |
| T-0904 | 混合精度建议 → 一键生成新作业 | 2026-09-29, suggest() + UI one-click clone; Playwright spec pending |

## TODO（计划中）

已规划、尚未开始。下表中的功能目前都不可用。

| 任务 | 标题 |
|---|---|
| T-0006 | CI 流水线（PR 门禁） |
| T-0007 | 契约与 Schema 初版 |
| T-0008 | 开发指南与 ADR 初始化 |
| T-0103 | common：注册表、哈希、线程池 |
| T-0104 | storage：SQLite 元数据与迁移 |
| T-0105 | storage：内容寻址缓存池（CAS）与 GC |
| T-0106 | CLI 骨架（CLI11） |
| T-0209 | estimator：性能预估与 CalibrationStore |
| T-0210 | planner：兼容矩阵与 JobSpec 校验 |
| T-0211 | planner：推荐与可部署矩阵 |
| T-0406 | quant 核心：SIMD/CUDA 加速 |
| T-0501 | Worker RPC 协议与 C++ WorkerClient |
| T-0502 | Python Worker 进程框架 |
| T-0505 | 模型下载（fetch_model 阶段） |
| T-0609 | 外部算法适配器（GPTQModel / llm-compressor） |
| T-0704 | SGLang 导出与适配 |
| T-0706 | TensorRT-LLM 导出与适配（扩展） |
| T-0801 | 精度评测：lm-eval-harness / evalscope / 自定义 |
| T-0805 | HAL 微基准（带宽/算力实测） |
| T-0806 | 报告生成（JSON + HTML） |
| T-0808 | M4 验收：全配方 × 多后端参考集 |
| T-0903 | 前端：误差页完整版 |
| T-0905 | 反馈闭环（S15）：预估自动校准 |
| T-0906 | 误差归因验收（注入测试） |
| T-1001 | 多模态框架：per_modality 配置与 set_modality |
| T-1002 | VLM 适配（Qwen2-VL / LLaVA） |
| T-1003 | MoE / MLA 结构支持（DeepSeek 系） |
| T-1005 | 世界模型/视频生成（DiT）适配（实验） |
| T-1101 | HAL ABI 与插件加载器 + 模板插件 |
| T-1102 | CPU / CUDA 插件 |
| T-1103 | Metal 插件 |
| T-1104 | ROCm 插件 |
| T-1105 | Ascend（CANN）插件 |
| T-1106 | 插件 SDK 文档与 ABI 兼容检查 |
| T-1201 | Docker 镜像与 compose |
| T-1202 | pip wheel 打包与发布 |
| T-1203 | 文档站（GitHub Pages） |
| T-1204 | 安全加固（凭证、路径沙箱、认证） |
| T-1205 | 可观测性（/metrics、可复现重放） |
| T-1206 | 夜间 GPU 回归与性能看板 |
| T-1207 | v1.0 发布验收 |

## DONE（已完成）

已实现，并按任务的验收标准验证过。

| 任务 | 标题 |
|---|---|
| T-0002 | C++ 构建系统（CMake + vcpkg + Presets） |
| T-0003 | Python 包骨架与 pybind11 构建链 |
| T-0005 | 代码规范与 pre-commit |
| T-0101 | common：错误、Result、日志 |
| T-0102 | common：配置加载与 Schema 校验 |
| T-0201 | C 硬件探针：CPU（三平台） |
| T-0202 | C 硬件探针：GPU/NPU（动态加载厂商库） |
| T-0203 | 硬件画像库与 HardwareProfile |
| T-0205 | modelspec：ModelSpec 解析（config + safetensors 头） |
| T-0206 | 架构模板与 QIR |
| T-0207 | 量化配方库 |
| T-0208 | estimator：显存/KV/并发预估 |
| T-0401 | quant 核心：QuantSpec/QParams/粒度重排 |
| T-0402 | quant 核心：标定算法 MinMax/MSE/百分位/MovingAvg |
| T-0403 | quant 核心：INT/FP8 量化、反量化、假量化 + golden |
| T-0404 | quant 核心：误差度量 |
| T-0407 | pybind11 绑定 `mqde._core`（DLPack） |
| T-0408 | Python Torch 快速路径与双端 golden 对拍 |
| T-0506 | Python 模型抽象层 BaseModel + HFDecoderModel |
| T-0507 | 校准数据与首块捕获 |
| T-0508 | Blockwise 引擎 + Fake/Real 模块包装 + RTN |
| T-0509 | 误差追踪 trace（逐层/逐块） |
| T-0601 | 算法：AWQ |
| T-0602 | 算法：GPTQ |
| T-0604 | 激活量化（动态/静态）→ W4A8、W2A8 |
| T-0607 | KV Cache 量化（KV_REGISTRY） |
| T-0608 | 混合精度与 keep_fp 策略 |
| T-0702 | Fake/Real 一致性检查（S10） |
| T-0705 | llama.cpp（GGUF）导出与适配 |
| T-1004 | VLA 适配（实验） |
