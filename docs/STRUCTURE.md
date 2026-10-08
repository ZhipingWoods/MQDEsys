# MQDEsys 目录结构与文件职责（STRUCTURE）

> 本文件告诉每一个 Agent：**每个目录/文件是干什么的、谁能改、改了要谁审**。
> 文档级别：**L2**。新增顶层目录、调整目录职责必须先改本文件并经人工评审。
>
> 相关文档：[ARCHITECTURE.md](ARCHITECTURE.md)（设计）· [TASKS.md](TASKS.md)（任务）·
> [REFERENCES.md](REFERENCES.md)（参考架构）· [DEVELOPMENT.md](DEVELOPMENT.md)（环境与流程）·
> [../AGENTS.md](../AGENTS.md)（工作规则）

---

## 1. 变更评审级别（Review Level）

| 级别 | 名称 | 适用范围 | Agent 可以做什么 | 合并条件 |
|------|------|---------|-----------------|---------|
| **L0** | 自由修改 | 模块内部实现（`src/` 下非公开代码）、单元测试、私有工具函数、注释、示例 | 直接修改，自测通过即可提 PR | CI 全绿 + 自查清单 |
| **L1** | 常规评审 | 新增文件/子模块、新增数据条目（新硬件画像/新配方/新数据集）、新增算法/模型/后端插件实现、前端页面与组件 | 可修改，PR 中说明设计 | CI 全绿 + 1 名评审（其他 Agent 或人） |
| **L2** | 接口评审（**必须人工**） | 公共头文件 `cpp/include/**`、`api/**`、`schemas/**`、HAL ABI、Worker RPC 协议、数据库迁移、`data/compat/**`、`data/golden/**`、`data/fit/**`、顶层构建/CI/工具链配置、`docs/ARCHITECTURE.md`/`STRUCTURE.md`/`REFERENCES.md`/`AGENTS.md` | **先提方案（PR 描述或 Issue），获批后再实现**；不得在同一 PR 中夹带无关修改 | CI 全绿 + 人工评审批准 |
| **L3** | 架构决策（**ADR + 负责人批准**） | 新增/替换第三方依赖（`vcpkg.json`、`pyproject.toml` 依赖、`web/package.json` 依赖）、语言/框架变更、许可证、安全相关（凭证处理、认证、路径沙箱）、破坏性变更（ABI/协议/契约主版本）、删除功能、`third_party/`、`3rdpart/` 的引入方式 | **禁止直接实现**；先写 `docs/adr/NNNN-*.md`（状态 Proposed），等待负责人改为 Accepted | ADR Accepted + L2 流程 |

> 判定原则：**一个 PR 的级别 = 它触碰到的最高级别路径**。拿不准就按更高一级处理。

### 绝对禁止（任何级别都不允许）

- 提交任何凭证（HF token、API key、密码）、模型权重、数据集原始文件（> 5MB 的非 LFS 文件）。
- `git push --force` 到 `main`；绕过 CI 合并；删除或跳过他人编写的测试以让 CI 通过。
- 手工编辑生成代码（`*.pb.*`、`_generated` 后缀文件）——只能改源契约后重新生成。
- 修改已合并的数据库迁移文件（只能新增迁移）。
- 修改 golden 数据以"让测试通过"而不在 PR 中给出数值依据。
- 修改 `3rdpart/` 下的参考源码，或让产品代码 `import` / `#include` 它。

---

## 2. 当前实际目录树

> 这是**当前仓库里真实存在的文件**（2026-09-28）。计划中但尚未创建的目录见 §7。

```
MQDEsys/
├── AGENTS.md                     [L2] Agent 工作手册（入口文件，先读它）
├── CLAUDE.md                     [L2] 引用 AGENTS.md（Claude Code 自动加载）
├── README.md                     [L1] 项目介绍、快速开始
├── CMakeLists.txt                [L2] 顶层 CMake：选项、子目录、全局编译标志
├── CMakePresets.json             [L2] dev / release 等预设
├── .gitignore                    [L2] 忽略规则（含 build/、runs/、模型权重、*.gguf）
├── .gitmodules                   [L3] 第三方参考库的上游地址（子模块）
├── cpp/                          C++ 后端（系统主干）
├── data/                         [L1/L2] 数据驱动的知识库
├── python/                       Python 执行器（mqde 包）
├── web/                          前端（React + TS + Vite）
├── docs/                         文档
├── examples/                     [L0] 示例 JobSpec
├── tests/                        跨语言测试夹具
└── 3rdpart/                      [L3] 参考源码（git 子模块：只存上游链接与提交，源码不入库）
```

---

## 3. `cpp/` —— C++ 后端

### 3.1 公共头文件 `cpp/include/mqde/`（全部 **L2**）

| 文件 | 职责 |
|------|------|
| `common/result.h` | `Result<T>` = `std::expected<T, Error>`；`Errc` 错误码；`MQDE_ASSIGN_OR_RETURN` 宏；`make_error()` |
| `common/config.h` | `Json` 类型别名；`load_structured_file()`（YAML/JSON 统一入口）；`resolve_data_file()`；`get_or<T>()` 安全取值 |
| `common/log.h` | 结构化日志（级别、模块标签） |
| `modelspec/safetensors.h` | `TensorInfo{name,dtype,shape,numel,nbytes,file,data_offset}`；`read_safetensors_headers()`；`read_tensor_f32()`（按需读单个张量并转 fp32，支持 BF16/F16/F32/F8_E4M3/F8_E5M2，抽样上限保护） |
| `modelspec/model_spec.h` | `ModelSpec` 全模型描述；`TextSpec`（层数/隐藏维/注意力布局/MoE 专家数）；`VisionSpec`；`ActionSpec`；`UnitInfo`（逐张量的组件归属、块号、可量化性、逻辑形状、是否为 aux 元数据）；`CheckpointQuant`（已量化 checkpoint 的方法/算法/位宽/组大小/忽略名单）；`inspect_local()`；`to_json()`；`to_graph_json()`（netron 式层级图） |
| `estimator/estimator.h` | `Recipe`（量化配方）；`HardwareProfile`；`BackendProfile`；`RuntimeParams`；`Estimate`（显存分解、FLOPs、字节数、延迟下界、可行性、理由列表）；`estimate()`；`load_recipe()` / `load_hardware()` / `load_backend()` |
| `fit/fit.h` | **部署预评估**（仿 llmfit）：`FitLevel`、`RunMode`、`Confidence`、`UseCase`、`FitConfig`、`System`、`Request`、`ScoreComponents`、`ModelFit`；`analyze()`、`rank()`、`load_fit_config()`、`system_from_profile()` |
| `fit/benchmarks.h` | **实测基线**：`MeasuredTps`、`HardwareId`、`BenchmarkIndex`（load / lookup）、`Submission`、`write_local_submission()`、`canonical_slug()` |
| `quant/quant_spec.h` | `QuantSpec`（位宽、粒度、对称性、格式）；解析与校验 |
| `quant/ops.h` | `fake_quant()`（伪量化，C++ 权威实现） |
| `quant/metrics.h` | `tensor_stats()`（min/max/absmax/mean/std/峰度/离群比例）；`measure()`（SQNR/cosine/rel_l2/max_abs/mse/rmse） |
| `server/server.h` | HTTP 服务选项与启动入口 |
| `probe_c/include/mqde_probe.h` | **C ABI** 硬件探测接口（`mqde_device_info`、`mqde_probe_all`）；跨语言边界，改动需格外谨慎 |

### 3.2 实现 `cpp/src/`（**L0/L1**）

| 文件 | 职责 | 级别 |
|------|------|------|
| `common/config.cpp` | YAML/JSON 加载、数据文件路径解析 | L1 |
| `common/log.cpp` | 日志实现 | L0 |
| `modelspec/safetensors.cpp` | safetensors 头解析、分片索引、单张量读取与 dtype 转换 | L1 |
| `modelspec/model_spec.cpp` | 从 config.json + 张量头构建 `ModelSpec`；逐张量归类到组件/块；预量化 checkpoint 的逻辑参数还原；MoE 活跃参数计算 | L1 |
| `modelspec/model_family.cpp` | 架构模板匹配（`data/model_families/*.yaml`） | L1 |
| `modelspec/graph.cpp` | netron 式层级图生成：组件 → 块 → 模块 → 张量，按名字/形状推断算子类型 | L1 |
| `estimator/estimator.cpp` | 显存与性能估算核心：逐张量按配方算量化后字节、量化元数据、KV/线性注意力状态、激活、MoE 感知的 bytes/FLOPs | L1 |
| `estimator/hardware.cpp` | 硬件画像加载 | L1 |
| `estimator/recipe.cpp` | 配方加载与校验 | L1 |
| `fit/fit.cpp` | 适配等级判定、量化层级搜索、运行模式选择、速度估算（带宽屋顶线 / K 常数 / 卸载）、四要素评分、notes 生成、排序 | L1 |
| `fit/benchmarks.cpp` | 实测基线库：读 `data/benchmarks/{local,community}`，相同硬件匹配，同模型取中位数；`mqde bench` 写入 | L1 |
| `quant/ops.cpp` | 伪量化数值实现（**权威**，Python 必须逐位对齐） | L1 |
| `quant/metrics.cpp` | 统计与误差指标 | L1 |
| `quant/quant_spec.cpp` | QuantSpec 解析 | L1 |
| `server/server.cpp` | 所有 HTTP 端点 | L1 |
| `server/job_runner.{h,cpp}` | 作业队列、GPU 池、子进程管理、SSE 事件流 | L1 |
| `probe_c/src/probe.c` | NVML/CPU 探测实现 | L1 |

### 3.3 应用与绑定

| 文件 | 职责 | 级别 |
|------|------|------|
| `apps/mqde_cli/main.cpp` | CLI：`model inspect`、`estimate`、`fit`、`list`、`serve` | L1 |
| `bindings/python/core_module.cpp` | pybind11 绑定，把 C++ 量化/指标暴露给 Python | L2（跨语言边界） |

### 3.4 测试 `cpp/tests/unit/`（**L0**，但删除他人测试禁止）

`common_test.cpp`、`modelspec_test.cpp`、`estimator_test.cpp`、`fit_test.cpp`、`quant_test.cpp`、`probe_test.cpp`

---

## 4. `data/` —— 数据驱动的知识库

**铁律**：模型结构、量化配方、硬件参数、评分常数只能出现在 YAML 里，不得硬编码进代码。

| 路径 | 内容 | 级别 |
|------|------|------|
| `fit/fit.yaml` | **llmfit 常数表**：每参数字节（显存/速度两套）、量化层级、质量惩罚、适配阈值、带宽效率、运行模式系数、后端 K 常数、用途权重与目标、质量分档。改任何一个值都会改变所有评估结论 | **L2** |
| `recipes/*.yaml` | 量化配方：`bf16`、`fp8`、`w8a16`、`w4a16`、`w4a8`、`w2a16`、`w2a8`。含权重/激活/KV 三段 QuantSpec、适用模态、成熟度 | L1 |
| `hardware_db/<vendor>/<id>.yaml` | 硬件画像：显存、带宽、各精度算力、原生支持格式。现有 `nvidia/{rtx4090,a100-80g,h100-sxm,jetson-agx-orin-64g}` | L1 |
| `model_families/*.yaml` | 架构模板：匹配规则、config 路径映射、组件前缀与块正则、默认保持高精度的层。现有 `generic_decoder`、`minicpm_vla`、`minicpmo`、`qwen3_5` | L1 |
| `backends/*.yaml` | 推理后端能力：支持的量化格式、固定开销。现有 `hf_transformers` | L1 |

---

## 5. `python/` —— 执行器

| 路径 | 职责 | 级别 |
|------|------|------|
| `mqde/core.py` | 加载 C++ `_core` 扩展，暴露量化与指标 | L1 |
| `mqde/quant/numerics.py` | 数值实现，**必须与 C++ 逐位一致**（有交叉验证测试） | L1 |
| `mqde/quant/engine.py` | 伪量化引擎：按配方替换模块、逐层追踪 | L1 |
| `mqde/quant/modules.py` | 可量化模块的包装与替换 | L1 |
| `mqde/models/base.py` | 模型适配器基类 | L1 |
| `mqde/models/minicpm_vla.py` | MiniCPM-VLA 适配器：加载、前向、动作预测 | L1 |
| `mqde/data/vla.py` | VLA 数据集读取与预处理 | L1 |
| `mqde/pipeline/vla_job.py` | 作业主流程：量化 → 评测 → 报告 | L1 |
| `mqde/eval/bench.py` | 实测吞吐（由 `mqde bench` 以子进程调用；stdout 最后一行为 JSON 结果） | L1 |
| `mqde/pipeline/events.py` | 结构化事件输出（被 C++ job_runner 解析为 SSE） | L2（协议） |
| `tests/test_numerics_golden.py` | C++/Python 数值一致性黄金测试 | L1（改动需给数值依据） |

---

## 6. `web/` —— 前端

| 路径 | 职责 | 级别 |
|------|------|------|
| `src/main.tsx` | 应用外壳：侧栏分组导航、路由、深浅主题切换 | L1 |
| `src/styles.css` | **设计 token 唯一来源**：深浅两套配色、算子色、图表色、所有组件样式。方向为 Raycast/Linear 式深色控制台 | L1 |
| `src/api.ts` | 后端类型定义 + API 客户端（唯一网络出口） | L1 |
| `src/charts.tsx` | ECharts 封装、`baseOption()`、色板取值 | L1 |
| `src/hooks.ts` | `useAsync`、`useDebounced` | L0 |
| `src/pages/Fit.tsx` | **部署预评估页**：目标系统面板、筛选排序、模型表、详情（评分构成/显存构成）、跨硬件规划 | L1 |
| `src/pages/Graph.tsx` | **模型结构页**：层级树、块内前向图（算子着色）、属性面板（张量/直方图/量化误差预览） | L1 |
| `src/pages/Estimate.tsx` | 资源预估页：单模型 × 多配方 × 硬件 | L1 |
| `src/pages/NewJob.tsx` | 新建作业表单 | L1 |
| `src/pages/Jobs.tsx` / `Job.tsx` | 作业列表 / 作业详情（阶段流水线、实时日志、逐层误差） | L1 |
| `src/pages/Compare.tsx` | 配方对比矩阵 | L1 |
| `src/pages/Hardware.tsx` | 硬件与后端 | L1 |
| `package.json` | 依赖清单（**新增依赖为 L3**） | L1/L3 |
| `vite.config.ts` / `tsconfig.json` | 构建配置 | L2 |

---

## 7. `docs/` —— 文档

| 文件 | 内容 | 级别 |
|------|------|------|
| `ARCHITECTURE.md` | 系统设计：分层、模块边界、数据流、关键决策 | L2 |
| `STRUCTURE.md` | **本文件**：目录与文件职责、评审级别 | L2 |
| `REFERENCES.md` | 各子系统模仿哪个开源项目、照搬边界；附录 A 为 llmfit 算法细节（公式与常数，可直接实现） | L2 |
| `TASKS.md` | 任务清单：ID、标题、级别、验收标准、状态 | L1 |
| `PROGRESS.md` | 会话日志：每次会话的完成项、校验方式、下一步、坑 | L1（**每次会话结束必须更新**） |
| `DEVELOPMENT.md` | 环境搭建、构建、测试、调试、提交流程 | L1 |
| `adr/NNNN-*.md` | 架构决策记录：背景、决策、替代方案、影响（**L3**，状态 Proposed → Accepted → Done） | L1 |

### 计划中但尚未创建

`api/`（OpenAPI 契约）、`schemas/`（JSON Schema）、`deploy/`、`scripts/`、
`data/benchmarks/`（实测基线）、`data/models/`（模型目录）、`data/compat/`、`data/golden/`、
`cpp/src/hal/`（后端插件）、`cpp/plugins/<vendor>/`、`python/mqde/eval/`（评测）、
`CHANGELOG.md`、`LICENSE`、`vcpkg.json`、`pyproject.toml`、`.github/`、`.clang-format`

创建它们时按本文件 §1 判定级别，并回来更新本表。

---

## 8. `3rdpart/` —— 参考源码（git 子模块，源码不入库）

每个目录是一个子模块：仓库里只有「上游地址 + 固定提交」（见 `.gitmodules`），GitHub 上点击目录会跳转到上游仓库的对应提交。
开发机 `/cache/wuzhiping/MQDEsys/3rdpart/` 已检出源码；其他机器按需 `git submodule update --init <目录>`。

| 目录 | 参考对象 | 许可证 |
|------|---------|-------|
| `llmfit/` | 部署预评估算法与 UI | MIT |
| `GPTQModel/` | 量化流程架构 | Apache-2.0 |
| `unsloth/` | 动态量化思想 | Apache-2.0 |
| `netron/` | 模型结构可视化 | MIT |
| `evalscope/` | 评测与压测框架 | Apache-2.0 |
| `lm-evaluation-harness/` | 精度评测（lm-eval） | MIT |
| `FlagTree/` | 多硬件后端插件 | MIT |

**规则**：只读。不得修改、不得被产品代码 import/include。需要复用代码时按许可证单独引入并走 L3 流程（写 ADR）。

---

## 9. 新增文件放哪里

| 你要加的东西 | 放这里 | 级别 |
|------------|-------|------|
| 新的量化配方 | `data/recipes/<name>.yaml` | L1 |
| 新的硬件 | `data/hardware_db/<vendor>/<id>.yaml` | L1 |
| 新的模型架构支持 | `data/model_families/<family>.yaml`（+ 必要时 `python/mqde/models/<family>.py`） | L1 |
| 新的评分常数 | `data/fit/fit.yaml` | **L2** |
| 新的 C++ 模块 | `cpp/include/mqde/<mod>/`（头，L2）+ `cpp/src/<mod>/`（实现，L1）+ `cpp/tests/unit/<mod>_test.cpp` + 在 `cpp/CMakeLists.txt` 注册 | 混合 |
| 新的 HTTP 端点 | `cpp/src/server/server.cpp` + `web/src/api.ts` 同步类型 | L1 |
| 新的前端页面 | `web/src/pages/<Name>.tsx` + `main.tsx` 注册路由与导航 | L1 |
| 新的样式 | 改 `web/src/styles.css` 的 token / 组件类，**不要写内联魔法数值** | L1 |
| 新的第三方依赖 | 先写 `docs/adr/NNNN-*.md` | **L3** |

