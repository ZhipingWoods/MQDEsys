# MQDEsys 目录结构与文件职责（STRUCTURE）

> 本文件告诉每一个 Agent：**每个目录/文件是干什么的、谁能改、改了要谁审**。
> 文档级别：**L2**。新增顶层目录、调整目录职责必须先改本文件并经人工评审。
>
> 相关文档：[ARCHITECTURE.md](ARCHITECTURE.md)（设计）· [TASKS.md](TASKS.md)（任务）· [../AGENTS.md](../AGENTS.md)（工作规则）

---

## 1. 变更评审级别（Review Level）

| 级别 | 名称 | 适用范围 | Agent 可以做什么 | 合并条件 |
|------|------|---------|-----------------|---------|
| **L0** | 自由修改 | 模块内部实现（`src/` 下非公开代码）、单元测试、私有工具函数、注释、示例 | 直接修改，自测通过即可提 PR | CI 全绿 + 自查清单 |
| **L1** | 常规评审 | 新增文件/子模块、新增数据条目（新硬件画像/新配方/新数据集）、新增算法/模型/后端插件实现、前端页面与组件 | 可修改，PR 中说明设计 | CI 全绿 + 1 名评审（其他 Agent 或人） |
| **L2** | 接口评审（**必须人工**） | 公共头文件 `cpp/include/**`、`api/**`、`schemas/**`、HAL ABI、Worker RPC 协议、数据库迁移、`data/compat/**`、`data/golden/**`、顶层构建/CI/工具链配置、`docs/ARCHITECTURE.md`/`STRUCTURE.md`/`AGENTS.md` | **先提方案（PR 描述或 Issue），获批后再实现**；不得在同一 PR 中夹带无关修改 | CI 全绿 + 人工评审批准 |
| **L3** | 架构决策（**ADR + 负责人批准**） | 新增/替换第三方依赖（`vcpkg.json`、`pyproject.toml` 依赖、`web/package.json` 依赖）、语言/框架变更、许可证、安全相关（凭证处理、认证、路径沙箱）、破坏性变更（ABI/协议/契约主版本）、删除功能、`third_party/` | **禁止直接实现**；先写 `docs/adr/NNNN-*.md`（状态 Proposed），等待负责人改为 Accepted | ADR Accepted + L2 流程 |

> 判定原则：**一个 PR 的级别 = 它触碰到的最高级别路径**。拿不准就按更高一级处理。

### 绝对禁止（任何级别都不允许）

- 提交任何凭证（HF token、API key、密码）、模型权重、数据集原始文件（> 5MB 的非 LFS 文件）。
- `git push --force` 到 `main`；绕过 CI 合并；删除或跳过他人编写的测试以让 CI 通过。
- 手工编辑生成代码（`web/src/api/generated/**`、`*.pb.*`、`_generated` 后缀文件）——只能改源契约后重新生成。
- 修改已合并的数据库迁移文件（只能新增迁移）。
- 修改 golden 数据以“让测试通过”而不在 PR 中给出数值依据。

---

## 2. 顶层目录树

```
MQDEsys/
├── AGENTS.md                     [L2] Agent 工作手册（所有 Agent 的入口文件）
├── CLAUDE.md                     [L2] 引用 AGENTS.md（Claude Code 自动加载）
├── README.md                     [L1] 项目介绍、快速开始
├── CHANGELOG.md                  [L1] 版本变更记录（发版时更新）
├── LICENSE                       [L3] 许可证（建议 Apache-2.0）
├── CMakeLists.txt                [L2] 顶层 CMake：选项、子目录、全局编译标志
├── CMakePresets.json             [L2] dev / release / asan / cuda 等预设
├── vcpkg.json                    [L3] C++ 依赖清单（新增依赖需 ADR）
├── pyproject.toml                [L2] Python 包定义（scikit-build-core 构建 _core）；依赖段为 [L3]
├── .github/                      [L2] CI/CD 与协作模板
├── .clang-format / .clang-tidy   [L2] C++ 代码风格与静态检查
├── .editorconfig                 [L2]
├── .pre-commit-config.yaml       [L2] 提交前检查
├── .gitignore / .gitattributes   [L2] 忽略规则、LFS 规则
├── api/                          [L2] 对外契约（OpenAPI、事件、Worker RPC）
├── schemas/                      [L2] 核心数据结构 JSON Schema
├── data/                         [L1/L2] 数据驱动的知识库（配方、硬件、兼容矩阵、架构模板、golden）
├── cpp/                          C++ 后端（系统主干）
├── python/                       Python 执行器（mqde 包）
├── web/                          前端（React + TS）
├── deploy/                       [L2] Docker、compose、部署脚本
├── scripts/                      [L1] 开发/校验/生成脚本
├── examples/                     [L0] 示例 JobSpec、使用示例
├── tests/                        跨语言集成测试、端到端测试、共享夹具
├── docs/                         文档（设计、任务、进度、ADR、指南、站点）
└── third_party/                  [L3] 必须 vendoring 的第三方代码（尽量不用，优先 vcpkg）
```

---

## 3. 各目录详细说明

### 3.1 `.github/` —— 协作与 CI  [L2]

```
.github/
├── workflows/
│   ├── ci.yml                 # PR 门禁：C++ 构建+测试、Python 测试、Web 测试、格式/静态检查、契约检查
│   ├── nightly.yml            # 夜间：GPU 集成测试、参考集预估对比、性能回归
│   ├── release.yml            # 打 tag 触发：构建 wheel、Docker 镜像、发布
│   └── docs.yml               # 构建并发布 GitHub Pages（zhipingwoods.github.io/MQDEsys）
├── PULL_REQUEST_TEMPLATE.md   # PR 模板：任务 ID、级别、变更说明、验证证据、文档同步勾选
├── ISSUE_TEMPLATE/            # bug / feature / adr-proposal
└── CODEOWNERS                 # L2/L3 路径指定人工评审人
```

### 3.2 `api/` —— 对外契约  [L2]

| 文件 | 作用 |
|------|------|
| `openapi.yaml` | REST API 唯一事实来源；C++ 路由与 TS 客户端据此实现/生成 |
| `events.schema.json` | WebSocket / 事件流消息格式 |
| `worker_rpc.schema.json` | C++ ↔ Python Worker JSON-RPC 方法、参数、通知格式 |
| `README.md` | 契约版本规则、如何生成代码 |

### 3.3 `schemas/` —— 核心数据结构  [L2]

`jobspec.schema.json`、`recipe.schema.json`、`model_spec.schema.json`、`hardware_profile.schema.json`、`estimate_report.schema.json`、`error_trace.schema.json`、`job_report.schema.json`、`compat_matrix.schema.json`、`model_family.schema.json`。
C++ 用 json-schema-validator 在加载时校验；Python 用 `jsonschema` 校验；前端从 OpenAPI 引用。

### 3.4 `data/` —— 数据驱动知识库

```
data/
├── recipes/                  [新增 L1 / 修改 L2] 量化配方
│   ├── w4a16.yaml  w4a8.yaml  w2a16.yaml  w2a8.yaml  fp8.yaml
│   └── README.md             # 字段说明
├── hardware_db/              [新增 L1 / 修改 L2] 硬件画像（规格参数：显存、带宽、各精度峰值算力、支持格式）
│   ├── nvidia/ a100-80g.yaml h100-sxm.yaml h20.yaml l40s.yaml rtx4090.yaml …
│   ├── amd/    mi300x.yaml …
│   ├── apple/  m2-max.yaml m3-max.yaml …
│   ├── ascend/ 910b.yaml …
│   └── cpu/    generic-x86-avx512.yaml generic-arm-neon.yaml
├── backends/                 [L2] 各推理后端经验参数（运行时开销初值、支持的量化格式、启动参数模板）
│   ├── vllm.yaml  sglang.yaml  llamacpp.yaml  trtllm.yaml
├── compat/                   [L2] 兼容性矩阵
│   └── matrix.yaml           # recipe × algorithm × backend × hw_arch → native/emulated/unsupported + 内核 + 约束
├── model_families/           [新增 L1 / 修改 L2] 架构模板：张量名规则 → QUnit 类型/Block/模态
│   ├── llama.yaml  qwen2.yaml  mistral.yaml  gemma2.yaml  deepseek_v2.yaml  qwen2_vl.yaml …
├── datasets/                 [L1] 校准/评估数据集注册（名称 → 来源、切分、预处理），不存原始数据
│   ├── calib.yaml  eval_suites.yaml
└── golden/                   [L2] 冻结的对拍数据（修改必须给出数值依据）
    ├── quant/                # 量化数值 golden（.npz），由 scripts/gen_golden_quant.py 生成
    ├── modelspec/            # 参考模型 config.json + safetensors 头 → 期望 ModelSpec
    ├── estimator/            # 手算的预估用例
    └── compat/               # 兼容性判定用例表
```

### 3.5 `cpp/` —— C++ 后端（系统主干）

```
cpp/
├── CMakeLists.txt                    [L2]
├── include/mqde/                     [L2] 公共头文件 = 模块对外接口（改动需人工评审）
│   ├── common/   error.h result.h log.h config.h registry.h hash.h thread_pool.h types.h
│   ├── hub/      model_source.h downloader.h
│   ├── modelspec/ model_spec.h arch_template.h
│   ├── ir/       qir.h
│   ├── estimator/ estimator.h calibration_store.h
│   ├── planner/  compat.h planner.h validate.h
│   ├── quant/    quant_spec.h qparams.h ops.h pack.h metrics.h tensor_view.h
│   ├── pipeline/ job.h stage.h orchestrator.h scheduler.h
│   ├── worker/   transport.h worker_client.h
│   ├── backend/  deploy_backend.h
│   ├── hal/      mqde_hal.h (纯 C ABI) plugin_loader.h
│   ├── eval/     bench.h
│   ├── storage/  cas.h db.h
│   └── server/   server.h
├── src/                              [L0 内部实现 / 新增文件 L1]
│   ├── common/ hub/ modelspec/ ir/ estimator/ planner/ quant/ pipeline/ worker/ backend/ hal/ eval/ storage/ server/
│   │   # 每个子目录：实现文件 + CMakeLists.txt（生成静态库 mqde_<module>）
│   ├── quant/kernels/                # CPU SIMD 分派（avx2/avx512/neon）与可选 CUDA 内核（.cu）
│   ├── backend/vllm/ sglang/ llamacpp/ trtllm/   # 各部署后端适配器 [L1]
│   ├── pipeline/stages/              # S00..S15 各阶段实现 [L1]
│   └── storage/migrations/           # [L2] NNNN_*.sql，只增不改
├── probe_c/                          硬件探针（纯 C99）
│   ├── include/mqde_probe.h          [L2]
│   └── src/ probe.c cpu_x86.c cpu_arm.c os_linux.c os_macos.c os_windows.c gpu_nvml.c gpu_rocm.c gpu_metal.m npu_ascend.c  [L0/L1]
├── plugins/                          HAL 硬件插件（独立动态库）[L1]
│   ├── template/                     # 新硬件接入模板（含 README）
│   ├── cpu/  cuda/  rocm/  metal/  ascend/
├── bindings/python/                  pybind11 模块 → mqde._core  [接口 L2 / 实现 L0]
│   └── core_module.cpp
├── apps/
│   ├── mqde_cli/main.cpp             # mqde 命令行 [L1]
│   └── mqde_server/main.cpp          # mqde-server [L1]
├── tests/                            [L0/L1]
│   ├── unit/<module>/*_test.cpp
│   ├── golden/                       # 读取 data/golden 的对拍测试
│   └── integration/                  # 跨模块、故障注入测试
└── bench/                            # Google Benchmark 微基准 [L0]
```

**C++ 模块依赖方向**（由 `scripts/check_deps.py` 强制）：

```
common ← probe_c(独立) ← hal
common ← hub ← modelspec ← ir ← estimator ← planner
common ← quant
common ← storage
common ← worker
{planner, quant, storage, worker, backend, hal, eval} ← pipeline ← server / apps
```

### 3.6 `python/` —— Python 执行器（`mqde` 包）

```
python/
├── mqde/
│   ├── __init__.py
│   ├── _core.pyi                   [L2] C++ 绑定的类型存根（随 bindings 同步）
│   ├── registry.py                 [L2] ALGO/MODEL/KV/EXPORTER/DATASET 注册表
│   ├── config.py                   [L1] JobSpec/Recipe 解析（jsonschema 校验）
│   ├── worker/                     Worker 进程
│   │   ├── main.py                 [L1] RPC 循环入口：python -m mqde.worker
│   │   ├── protocol.py             [L2] 与 api/worker_rpc.schema.json 一一对应
│   │   ├── handlers/               [L1] model_inspect.py calib.py quant.py eval.py export.py real_check.py
│   │   └── echo_worker.py          [L0] 测试桩
│   ├── hub/                        [L1] 备用下载（huggingface_hub/modelscope），主路径在 C++
│   ├── data/                       [L1] 校准数据集构建、预处理（与推理一致的 tokenize）
│   ├── models/                     模型抽象层
│   │   ├── base.py                 [L2] BaseModel 接口
│   │   ├── hf_decoder.py           [L1] 通用 HF Decoder 适配（读 data/model_families）
│   │   ├── llama.py qwen2.py deepseek_v2.py …    [L1]
│   │   ├── vlm/ vla/ world/        [L1] 多模态适配器
│   │   └── catcher.py              [L0] 首块输入捕获
│   ├── quant/
│   │   ├── quantizer.py            [L1] 包装 _core；IntegerQuantizer / FloatQuantizer
│   │   ├── numerics.py             [L1] Torch GPU 快速路径（必须通过 golden）
│   │   ├── base.py                 [L2] BaseBlockwiseQuantization 接口
│   │   ├── blockwise.py            [L1] BlockwiseOpt 调度循环
│   │   ├── modules/                [L1] fake_linear.py real_linear.py fp8_linear.py rotate_linear.py quant_attention.py
│   │   ├── algorithms/             [L1] rtn.py awq.py gptq.py smoothquant.py hqq.py quarot.py external.py
│   │   ├── kv/                     [L1] naive.py kivi.py
│   │   └── mixed_precision.py      [L1]
│   ├── trace/                      [L1] 误差追踪 hooks、记录器、列式写出
│   ├── eval/                       [L1] ppl.py lm_eval_adapter.py evalscope_adapter.py custom.py logits_probe.py
│   └── export/                     [L1] base.py(接口 L2) vllm.py sglang.py gguf.py trtllm.py
└── tests/                          [L0/L1] pytest；fixtures 引用 tests/fixtures
```

### 3.7 `web/` —— 前端

```
web/
├── package.json                    [依赖 L3 / 脚本 L2]
├── vite.config.ts tsconfig.json    [L2]
├── src/
│   ├── api/
│   │   ├── generated/              [禁止手改] openapi-typescript 生成
│   │   ├── client.ts               [L1] fetch 封装、错误处理
│   │   └── ws.ts                   [L1] WebSocket 断线重连 + after_seq 补齐
│   ├── pages/                      [L1] NewJob/ Estimate/ JobRun/ ErrorTrace/ Report/ Compare/ Hardware/ Cache/
│   ├── components/                 [L0/L1] 通用组件（表单、DAG 视图、日志面板）
│   ├── charts/                     [L1] 纯数据驱动图表：MemoryStackBar KvContextCurve ConcurrencyCurve FeasibilityHeatmap
│   │                               #   StageWaterfall LayerMetricHeatmap BlockErrorCurve HistogramPanel ParetoScatter PerfCurves
│   ├── stores/                     [L1] Zustand 状态
│   ├── hooks/                      [L0]
│   ├── i18n/                       [L0] zh-CN / en-US
│   └── theme/                      [L1]
└── tests/
    ├── unit/                       [L0] vitest
    └── e2e/                        [L1] Playwright（对 mock server）
```

### 3.8 `deploy/` —— 交付  [L2]

```
deploy/
├── docker/
│   ├── Dockerfile.cpu              # server + worker(CPU torch) + web 静态文件
│   ├── Dockerfile.cuda             # 基于 nvidia/cuda，含 vLLM（可选层）
│   └── entrypoint.sh
├── compose/docker-compose.yml      # server + web(nginx)
└── systemd/mqde-server.service     # 裸机部署示例
```

### 3.9 `scripts/` —— 脚本  [L1]

| 脚本 | 作用 |
|------|------|
| `dev_setup.sh` | 一键安装开发环境（vcpkg bootstrap、pip -e、pnpm i、pre-commit） |
| `check_all.sh` | **本地全量校验**（= CI 门禁）：格式、静态检查、构建、全部测试、契约、依赖方向 |
| `gen_api.sh [--check]` | 由 OpenAPI/Schema 生成 TS 类型与 C++ DTO 校验表；`--check` 用于 CI 确认无漂移 |
| `gen_golden_quant.py` | 由参考实现生成量化 golden（L2 用途） |
| `check_deps.py` | C++ 模块 include 依赖方向检查 |
| `check_abi.sh` | HAL ABI 兼容检查 |
| `verify_probe.sh` | 探针结果与系统工具对比 |
| `bench_reference.sh` | 参考集“预估 vs 实测”报告（需要 GPU） |
| `contract_test.sh` | 启动 server 并用 schemathesis 做契约模糊测试 |

### 3.10 `tests/` —— 跨语言测试

```
tests/
├── fixtures/
│   ├── tiny-llama/                 # 2 层随机权重 Llama（< 5MB，可进仓库），CI 端到端用
│   ├── configs/                    # 参考模型 config.json（离线 ModelSpec 测试）
│   └── mock_backend/               # 模拟 OpenAI 兼容推理服务（固定延迟），压测器测试用
├── integration/
│   ├── test_e2e_tiny.py            # CLI 跑完整 Job（CPU）
│   └── gpu/                        # 需 GPU 的部署/压测脚本（夜间）
└── e2e_web/                        # 前后端联调（可选）
```

### 3.11 `docs/` —— 文档

```
docs/
├── ARCHITECTURE.md                 [L2] 系统设计
├── STRUCTURE.md                    [L2] 本文件
├── TASKS.md                        [任务状态字段 L0 / 任务定义 L2] 任务清单
├── PROGRESS.md                     [L0] 进度与交接日志（每次会话结束必须更新）
├── adr/                            [L3] 架构决策记录 NNNN-title.md（Proposed/Accepted/Superseded）
├── guides/                         [L1] 开发指南
│   ├── dev-setup.md  coding-standards.md  quant-numerics.md
│   ├── add-algorithm.md  add-model-adapter.md  add-backend.md  add-hardware-plugin.md  add-recipe.md
└── site/                           [L1] GitHub Pages 文档站（mkdocs-material）
```

---

## 4. 运行时目录（不进 Git）

```
$MQDE_HOME（默认 ~/.mqde）
├── config.yaml          # 服务配置：端口、allowed_paths、缓存上限、插件目录
├── credentials          # 凭证（权限 600），不进日志
├── db/mqde.sqlite       # 作业、阶段、事件、产物索引、硬件画像、基准历史、预估校准
├── cas/                 # 内容寻址产物库
├── runs/<job_id>/       # 作业工作目录：logs/ report.json report.html artifacts→cas
└── plugins/             # 第三方 HAL 插件
```

构建产物：`build/`（CMake）、`web/dist/`、`python/*.egg-info`、`.venv/` —— 全部在 `.gitignore` 中。

---

## 5. 命名与放置规则（新增文件前先查这里）

| 要新增的东西 | 放在哪里 | 还要同步什么 | 级别 |
|------------|---------|-------------|------|
| 新量化算法 | `python/mqde/quant/algorithms/<name>.py` | 注册、单测、`data/compat/matrix.yaml`（L2）、`docs/guides/add-algorithm.md` 示例表 | L1（+矩阵 L2） |
| 新模型架构 | `data/model_families/<family>.yaml`（优先）或 `python/mqde/models/<family>.py` | `data/golden/modelspec/` 用例 | L1 |
| 新量化配方 | `data/recipes/<name>.yaml` | 兼容矩阵、前端下拉自动出现（来自 `/recipes`） | L1（+矩阵 L2） |
| 新硬件（规格） | `data/hardware_db/<vendor>/<model>.yaml` | — | L1 |
| 新硬件（插件） | `cpp/plugins/<vendor>/` 复制 `template/` | `docs/guides/add-hardware-plugin.md` 支持列表 | L1 |
| 新部署后端 | `cpp/src/backend/<name>/` + `python/mqde/export/<name>.py` + `data/backends/<name>.yaml` | 兼容矩阵、集成测试 | L1（+矩阵 L2） |
| 新 REST 端点 | 先改 `api/openapi.yaml` | `scripts/gen_api.sh`、server 路由、前端 | L2 |
| 新流水线阶段 | `cpp/src/pipeline/stages/` | ARCHITECTURE 5.1 表、事件类型 | L2 |
| 新前端页面 | `web/src/pages/<Name>/` | 路由、i18n | L1 |
| 新 C++ 第三方库 | — | **先写 ADR** | L3 |

文件命名：C++ `snake_case.cpp/.h`，类 `PascalCase`，函数/变量 `snake_case`，常量 `kPascalCase`，命名空间 `mqde::<module>`；Python 遵循 PEP 8；TS 组件 `PascalCase.tsx`，其他 `camelCase.ts`。


## 附：2026-09-28 新增路径

| 路径 | 职责 | 级别 |
|------|------|------|
| `cpp/src/fit/`、`cpp/include/mqde/fit/` | llmfit 式部署预评估（适配等级、运行模式、速度、评分） | L2 |
| `data/fit/fit.yaml` | llmfit 常数表（bpp、阈值、效率、权重）。改一个值会改变所有结论 | L2 |
| `cpp/src/modelspec/graph.cpp` | netron 式层级图（组件→块→模块→张量），只读 safetensors 头 | L1 |
| `data/model_families/{qwen3_5,minicpmo}.yaml` | 新增架构模板 | L1 |
| `web/src/pages/{Fit,Graph}.tsx` | 部署预评估页、模型结构页 | L0 |
| `docs/REFERENCES.md` | 各子系统对应的开源参考与模仿边界 | L2 |
| `3rdpart/`（仅开发机，git 忽略） | 参考源码只读副本，禁止修改、禁止被产品代码 import | L3 |
