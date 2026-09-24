# MQDEsys 系统架构设计（ARCHITECTURE）

> Model Quantization & Deployment Evaluation System —— 模型量化部署评估系统
> 一条命令/一次点击：任意模型 × 任意量化配方 × 任意硬件，从「预估」到「部署」到「评估」到「反馈」。
>
> 文档级别：**L2（修改需人工评审）**。本文件描述“系统应该是什么样”；任务拆解见 [TASKS.md](TASKS.md)，目录职责见 [STRUCTURE.md](STRUCTURE.md)，Agent 工作规则见 [../AGENTS.md](../AGENTS.md)。

---

## 0. 目录

1. 目标、范围与非目标
2. 关键架构决策（ADR 摘要）
3. 总体分层架构
4. 进程拓扑与部署形态
5. 作业生命周期与主数据流
6. 模块设计（每块：职责 / 接口 / 实现要点 / 验收 / 校验）
7. 核心数据结构与契约
8. 量化配方定义（W4A16 / W4A8 / W2A16 / W2A8 / FP8）
9. 资源预估模型（模型占用 / KV Cache / 并发）
10. 量化误差追踪与归因
11. 质量保障体系
12. 可观测性、安全与版本演进

---

## 1. 目标、范围与非目标

### 1.1 目标

| # | 目标 | 可度量的完成标志 |
|---|------|------------------|
| G1 | 指定模型（HuggingFace / ModelScope / 本地路径）自动获取与解析 | 输入 `Qwen/Qwen2.5-7B-Instruct`，无需下载权重即可得到完整 ModelSpec |
| G2 | 指定量化配方 W4A8 / W4A16 / W2A16 / W2A8 / FP8，可选算法 RTN/AWQ/GPTQ/SmoothQuant/HQQ… | 五种配方均可跑通 fake_quant；W4A16/W4A8/FP8 可真实导出部署 |
| G3 | 指定评估数据 / 评估类型（PPL、MMLU、GSM8K、自定义数据集、性能压测） | 同一 Job 输出精度 + 性能双报告 |
| G4 | 指定部署硬件架构（CUDA/ROCm/Metal/Ascend/CPU/…，插件化接入） | 新增硬件只需实现 HAL 插件，上层代码零修改 |
| G5 | **部署前预估**：模型占用、KV Cache 占用、单/多并发资源、最大上下文、可部署性 | 显存预估与实测误差 ≤ 10%（参考集） |
| G6 | 前端可视化整体执行过程、实时日志、量化误差（逐层/逐阶段） | Web 上能定位“误差最大的层”和“误差最大的转换阶段” |
| G7 | 缓存池：模型、校准统计、量化产物、硬件画像、基准历史可复用 | 相同输入二次运行命中缓存，跳过已完成阶段 |
| G8 | 支持 LLM → VLM → VLA → 世界模型 的渐进扩展 | 模型适配器插件化，新增模态不改调度层 |
| G9 | 后端 C++ 实现，便于扩展与接入其他后端 | 控制面、预估、探针、数值核心、API 全部 C++ |

### 1.2 v1 范围（首个可发布版本）

- 模型：LLM（Llama / Qwen2/2.5/3 / Mistral / Gemma2 / DeepSeek-V2/V3(MoE, MLA)），VLM（Qwen2-VL / LLaVA）实验性支持。
- 配方：W4A16、W4A8、FP8 生产级；W2A16、W2A8 **实验级**（保证 fake_quant 评估，真实部署视后端支持）。
- 算法：RTN、AWQ、GPTQ、SmoothQuant；HQQ、QuaRot 作为扩展。
- 后端：vLLM、SGLang、llama.cpp（GGUF）；TensorRT-LLM 作为扩展。
- 硬件：NVIDIA CUDA、x86/ARM CPU、Apple Metal；ROCm、Ascend 作为插件扩展。

### 1.3 非目标（明确不做）

- 不做 QAT（量化感知训练），只做 PTQ。
- v1 不自研推理 Runtime（架构图第 6 层“Runtime / Compiler Core”列为长期研究方向，v1 通过适配现有推理引擎实现部署）。
- 不追求 2bit SOTA；优先保证 W4A16 / W4A8 / FP8 稳定可上线。
- 不做分布式多机训练/推理编排（单机多卡支持，多机留接口）。

---

## 2. 关键架构决策（ADR 摘要）

完整 ADR 放在 `docs/adr/`，此处为摘要。**任何推翻以下决策的修改都是 L3 级，需要项目负责人批准。**

### ADR-0001：C++ 控制面 + 数值核心，Python 执行面

**背景**：要求后端用 C++ 以便扩展与接入其他后端；但 HF 模型加载、GPTQ/AWQ 校准、lm-evaluation-harness/evalscope、vLLM 导出格式均深度绑定 PyTorch 生态，纯 C++ 重写成本极高且会与上游脱节。

**决策**：
- **C++（系统主干）**：API 服务、作业编排/调度、硬件探针、ModelSpec/IR、资源预估、兼容性规划、量化数值核心（QParams/量化/反量化/打包/误差度量）、部署后端适配、性能压测、HAL 插件 ABI、缓存池与元数据存储、CLI。
- **Python（执行器，受 C++ 调度）**：`mqde-worker` 子进程，负责需要 PyTorch 的工作：模型加载、校准前向、Blockwise 量化算法、精度评测、导出为后端格式。
- **边界**：C++ ↔ Python 通过 **Worker RPC 协议**（JSON-RPC 2.0 over stdio，行分隔）通信；大张量永不走 RPC，只通过文件系统中的 artifact（safetensors）交换。
- **一致性保证**：量化数值语义以 C++ `mqde::quant` 为唯一规范（通过 pybind11 暴露为 `mqde._core`）；Python/Torch 侧 GPU 快速路径必须通过同一组 **golden 向量**逐位一致校验。

**后果**：Python 是可替换的执行器，未来可替换为 libtorch C++ 执行器或远程 GPU 集群执行器，而不影响控制面。

### ADR-0002：前后端契约先行（OpenAPI + JSON Schema）

`api/openapi.yaml` 与 `schemas/*.schema.json` 是唯一事实来源；C++ DTO 校验、TS 客户端均由其生成/校验。修改契约为 L2。

### ADR-0003：硬件接入走稳定 C ABI 插件

硬件插件为独立动态库（`.so/.dylib/.dll`），导出 `mqde_hal_get_plugin_v1()`，结构体带 `abi_version`。上层只依赖 ABI，不依赖任何厂商 SDK。

### ADR-0004：存储 = 内容寻址文件缓存 + SQLite 元数据

所有阶段产物以 `sha256(输入指纹 + 配置 + 代码版本)` 为键存入缓存池，天然支持复用、断点续跑、可复现。

### ADR-0005：技术栈

| 层 | 选型 | 说明 |
|----|------|------|
| C++ 标准/构建 | C++20、CMake ≥ 3.25、CMakePresets、vcpkg manifest | 跨平台 |
| HTTP/WebSocket | Drogon | 异步 HTTP + WS，一套库 |
| 序列化 | nlohmann/json、yaml-cpp、json-schema-validator | |
| 日志 | spdlog + fmt | 结构化 JSON 日志 |
| 存储 | SQLite（SQLiteCpp） | 单机零运维 |
| 网络下载 | libcurl | HF/ModelScope 下载、Range 请求读 safetensors 头 |
| CLI | CLI11 | |
| 测试 | GoogleTest、Google Benchmark | |
| Python 绑定 | pybind11 + DLPack | 零拷贝张量 |
| 硬件探针 | C99（厂商库一律 `dlopen` 动态加载） | 无厂商 SDK 也能编译 |
| Python 执行器 | Python ≥ 3.10、PyTorch、transformers、safetensors、lm-eval、evalscope | |
| 前端 | React 18 + TypeScript + Vite + Ant Design + ECharts + TanStack Query + Zustand | |
| 前端契约 | openapi-typescript 生成类型 | |
| 质量 | clang-format/clang-tidy、ruff/mypy、eslint/prettier、pre-commit | |
| 交付 | Docker（cpu / cuda 镜像）、pip wheel（scikit-build-core） | |

---

## 3. 总体分层架构

与需求图的 8 层 + 缓存池一一对应：

```
┌───────────────────────────────────────────────────────────────────────────┐  ┌──────────────────────┐
│ L1 用户/API 层        CLI (mqde) │ REST + WebSocket │ Web UI (React)          │  │  Profile / Artifact  │
├───────────────────────────────────────────────────────────────────────────┤  │     / Cache Pool     │
│ L2 控制面/编排层      JobSpec 解析 → Pipeline DAG → 任务队列 → 调度器          │  │                      │
│                      → 设备分配 → 重试/恢复 → 可复现性（日志/产物/环境）         │◄─┤ 模型元数据缓存        │
├───────────────────────────────────────────────────────────────────────────┤  │ HF 原始模型缓存       │
│ L3 注册表 + 预检规划   Model/Dataset/Recipe/Hardware/Algorithm/Backend Registry │◄─┤ 校准统计缓存          │
│                      兼容性矩阵 │ 资源预估器 │ 可部署性判定（Can it run?）       │  │ 量化产物缓存          │
├───────────────────────────────────────────────────────────────────────────┤  │ 硬件画像             │
│ L4 模型前端 & 统一IR   HF/ModelScope 下载器 │ config/tokenizer 解析              │◄─┤ 基准历史             │
│                      架构适配器 │ ModelSpec │ 规范量化 IR（QIR）                  │  │ 兼容性矩阵           │
├───────────────────────────────────────────────────────────────────────────┤  │ 部署配方             │
│ L5 量化编译层          校准引擎 │ Observer │ 算法插件 │ 混合精度规划              │◄─┤ 预估校准模型          │
│   [C++ 数值核心 + Py] Scale/ZP 计算 │ 权重打包 │ 激活量化 │ 量化校验 │ 误差追踪    │  │                      │
├───────────────────────────────────────────────────────────────────────────┤  │  Estimate → Deploy   │
│ L6 部署后端适配层      vLLM │ SGLang │ llama.cpp │ TRT-LLM │ (长期: 自研 Runtime) │◄─┤  → Measure → Update  │
├───────────────────────────────────────────────────────────────────────────┤  │  → Better Estimate   │
│ L7 硬件插件层 (HAL)    稳定 C ABI: device_probe/supports/alloc/launch/profile  │◄─┤                      │
│                      CUDA │ ROCm │ Metal │ SYCL │ CANN │ MUSA │ CPU │ +插件     │  │                      │
├───────────────────────────────────────────────────────────────────────────┤  │                      │
│ L8 评估与反馈          精度(PPL/任务) │ TTFT/TPOT/吞吐 │ 峰值显存 │ 并发压测       │──►                      │
│                      OOM 边界 │ 最大稳定上下文 │ 结果回写 → 校准预估器            │  └──────────────────────┘
└───────────────────────────────────────────────────────────────────────────┘
```

**依赖规则（强制）**：上层可以依赖下层，下层不得依赖上层；同层模块之间只通过 `include/mqde/<module>/` 公共头文件交互。`common` 被所有模块依赖，`common` 不依赖任何模块。CI 用 `scripts/check_deps.py` 检查 include 依赖图。

---

## 4. 进程拓扑与部署形态

```
                ┌──────────────┐        HTTP/WS         ┌─────────────────────────────┐
  浏览器 ───────►│  web (静态)   │ ─────────────────────► │ mqde-server (C++, Drogon)    │
                └──────────────┘                        │  ├─ API / WS 事件推送         │
  终端 ── mqde CLI ── (本地库调用 或 HTTP 调用 server) ──►│  ├─ Orchestrator + Scheduler │
                                                        │  ├─ Estimator / Planner       │
                                                        │  ├─ Storage (SQLite + CAS)    │
                                                        │  └─ HAL plugin loader         │
                                                        └───┬───────────┬──────────────┘
                                      JSON-RPC over stdio    │           │ 子进程 + HTTP
                                              ┌──────────────▼──┐   ┌────▼──────────────────┐
                                              │ mqde-worker (Py) │   │ 推理引擎进程            │
                                              │ quant/eval/export│   │ vllm serve / sglang /  │
                                              └──────────────────┘   │ llama-server …         │
                                                                     └────────────────────────┘
```

- **单机模式（默认）**：`mqde-server` 同时承担 API 与调度；每个 GPU 作业启动独立 worker 子进程（进程级隔离，崩溃/OOM 不影响主服务）。
- **纯 CLI 模式**：`mqde estimate / mqde run job.yaml` 直接链接核心库，无需启动 server。
- **Docker**：`mqde:cpu`、`mqde:cuda` 两个镜像；`docker compose up` 启动 server + web。
- **pip**：`pip install mqde` 安装 Python 执行器 + `mqde._core` + `mqde` CLI 二进制。

---

## 5. 作业生命周期与主数据流

### 5.1 Pipeline DAG（标准阶段）

每个阶段是 DAG 中的一个节点，具有：输入指纹、输出 artifact、可缓存、可重试、可取消、事件上报。

| 阶段 ID | 名称 | 执行者 | 输入 | 输出 | 可缓存 |
|---------|------|--------|------|------|--------|
| S00 | validate | C++ | JobSpec | 规范化 JobSpec | — |
| S01 | preflight | C++ | ModelSpec + Recipe + HardwareProfile | EstimateReport + 可部署性结论 | ✓ |
| S02 | fetch_model | C++（libcurl） | model id/revision | 本地模型目录（HF 缓存） | ✓ |
| S03 | inspect | Py | 模型目录 | 层结构清单、QIR 描述 | ✓ |
| S04 | calib_prepare | Py | 数据集 + tokenizer | 校准样本张量 | ✓ |
| S05 | eval_baseline | Py | FP 模型 | 基线精度 + 探针集 logits | ✓ |
| S06 | transform（可选） | Py | FP 模型 | 等效变换后模型（SmoothQuant/QuaRot） | ✓ |
| S07 | quantize | Py（调用 `_core`） | 模型 + 校准数据 + 配方 | QParams + fake 模型状态 + **逐层误差 trace** | ✓ |
| S08 | eval_fake | Py | fake_quant 模型 | 精度 + logits KL | ✓ |
| S09 | export | Py（调用 `_core` 打包） | QParams + 权重 | 后端格式产物（safetensors/GGUF…） | ✓ |
| S10 | real_check | Py | 真实量化产物 | Fake/Real 一致性报告 | ✓ |
| S11 | deploy | C++（后端适配器） | 产物 + 硬件 | 运行中的推理端点 | — |
| S12 | eval_deployed | Py/C++ | 推理端点 | 部署后精度 | ✓ |
| S13 | perf_bench | C++ | 推理端点 | TTFT/TPOT/吞吐/峰值显存/OOM 边界 | ✓ |
| S14 | report | C++ | 以上全部 | JobReport（JSON + HTML） | — |
| S15 | feedback | C++ | 预估 vs 实测 | 更新预估校准系数 | — |

JobSpec 中 `evaluation.stages` 决定启用哪些节点；DAG 构建器按依赖自动裁剪。

### 5.2 数据流要点

1. **预估不下载权重**：S01 仅用 `config.json` + safetensors **头部**（HTTP Range 读取前 8 字节 + header JSON）即可得到精确张量形状/dtype。
2. **大数据只走文件**：权重、校准张量、QParams 全部落盘为 safetensors，RPC 只传路径与元数据。
3. **事件流**：所有阶段通过 `EventSink` 发出结构化事件 → 写入 SQLite（可回放）→ WebSocket 推送前端。
4. **误差 trace 在量化时即时记录**（S07），不需要事后重跑。

---

## 6. 模块设计

> 每个模块统一按「职责 / 公共接口 / 实现要点 / 验收标准 / 校验方法」描述。接口为示意，最终以 `cpp/include/mqde/**` 头文件为准（L2）。

### 6.1 `common` —— 基础设施

- **职责**：错误类型、`Result<T>`、日志、配置加载（YAML/JSON + Schema 校验）、注册表模板、线程池、路径/哈希工具、时间与 ID 生成。
- **公共接口**：
  ```cpp
  namespace mqde {
  enum class Errc { kOk, kInvalidArgument, kNotFound, kUnsupported, kResourceExhausted, kIo, kTimeout, kCancelled, kInternal, kExternal };
  struct Error { Errc code; std::string message; std::string context; };
  template <class T> using Result = std::expected<T, Error>;   // C++23 或 tl::expected

  template <class Base> class Registry {           // 字符串 → 工厂
   public:
    using Factory = std::function<std::unique_ptr<Base>()>;
    void add(std::string_view name, Factory f);
    Result<std::unique_ptr<Base>> create(std::string_view name) const;
    std::vector<std::string> names() const;
  };
  Result<nlohmann::json> load_and_validate(const std::filesystem::path&, std::string_view schema_id);
  std::string content_hash(std::span<const std::byte>);  // sha256 hex
  }
  ```
- **实现要点**：不抛异常跨模块边界（内部可抛，边界转为 `Result`）；日志统一带 `job_id / stage / component` 字段；注册通过静态注册宏 `MQDE_REGISTER(Registry, "name", Class)`。
- **验收**：所有错误码有文字说明；Schema 校验错误能指出 JSON Pointer 路径。
- **校验**：`ctest -R common_`；非法配置用例 ≥ 20 个全部被拒且错误信息含字段路径。

### 6.2 `probe`（C99）—— 硬件探针

- **职责**：跨平台（Linux / macOS / Windows）自动识别 CPU、GPU、NPU、TPU、DPU 等计算设备，输出 `HardwareInventory`。
- **公共接口**（`cpp/probe_c/include/mqde_probe.h`，L2）：
  ```c
  typedef enum { MQDE_DEV_CPU, MQDE_DEV_GPU, MQDE_DEV_NPU, MQDE_DEV_TPU, MQDE_DEV_DPU, MQDE_DEV_OTHER } mqde_dev_kind;
  typedef struct {
    mqde_dev_kind kind; char vendor[32]; char name[128]; char arch[32];   /* 如 "sm_90" "gfx942" "apple-m3" */
    uint64_t mem_total_bytes, mem_free_bytes; uint32_t compute_units; uint32_t clock_mhz;
    char driver_version[64]; char runtime_version[64]; int pci_bus; int numa_node;
    uint32_t feature_flags;   /* AVX2/AVX512/AMX/NEON/SVE/FP8/BF16/INT8-TC… 位标志 */
  } mqde_device_info;
  int  mqde_probe_all(mqde_device_info* out, int max_devices, int* n_found);
  const char* mqde_probe_last_error(void);
  ```
- **实现要点**：
  - CPU：Linux `/proc/cpuinfo` + `sysfs`；macOS `sysctl`；Windows `GetLogicalProcessorInformationEx`；x86 用 `cpuid`，ARM 读 `HWCAP`。
  - NVIDIA：`dlopen("libnvidia-ml.so")` 调 NVML；AMD：`dlopen` `librocm_smi64`；Apple：IOKit / `MTLCopyAllDevices`（`.m` 文件）；Ascend：`dlopen` DCMI；其他厂商由 HAL 插件的 `device_probe` 补充。
  - **任何厂商库缺失都不能编译失败或崩溃**，只返回“未检测到”。
- **验收**：在 Linux+NVIDIA、macOS Apple Silicon、纯 CPU 容器三种环境正确输出设备列表。
- **校验**：`mqde probe --json` 结果与 `nvidia-smi -q` / `sysctl hw` / `lscpu` 对比（脚本 `scripts/verify_probe.sh`）；无 GPU 容器中运行不报错。

### 6.3 `hub` + `modelspec` + `ir` —— 模型前端与统一 IR

- **职责**：
  1. `hub`：从 HuggingFace / ModelScope / 本地路径解析与下载模型；支持镜像（`HF_ENDPOINT`）、token、断点续传、只下载元数据。
  2. `modelspec`：从 `config.json` + safetensors header 生成 **ModelSpec**（层数、hidden、heads、kv_heads、head_dim、intermediate、vocab、MoE 专家数/激活专家数、注意力类型 MHA/GQA/MQA/MLA/滑窗/线性注意力、模态组件、每个张量形状与 dtype）。
  3. `ir`：规范量化 IR（QIR）——以“可量化单元”为粒度描述模型：`QUnit{ name, kind(linear/embedding/attn_matmul/kv_cache/conv), shape, block_idx, modality, sensitivity_hint }`，以及量化元数据与张量布局描述。QIR 是预估器、规划器、误差追踪、前端层级视图的共同语言。
- **公共接口**：
  ```cpp
  Result<ModelRef>  resolve_model(const ModelSource&);                       // id/revision → 文件清单与大小
  Result<ModelSpec> inspect_remote(const ModelRef&);                          // 仅读 config + safetensors 头
  Result<std::filesystem::path> fetch(const ModelRef&, FetchOptions, EventSink&);
  Result<QIRGraph>  build_qir(const ModelSpec&, const ArchTemplate&);         // 架构模板来自 data/model_families/*.yaml
  ```
- **实现要点**：
  - 架构适配器数据驱动：`data/model_families/<family>.yaml` 描述张量名正则 → QUnit 类型/所属 block/模态，新增主流架构只加 YAML；特殊结构（MLA、Mamba、DiT）才写 C++ 适配器。
  - 未知架构降级：按张量名启发式归类 + 标记 `confidence: low`，前端显示警告。
  - `trust_remote_code` 默认关闭；开启需要 JobSpec 显式声明并在 UI 二次确认。
- **验收**：参考模型集（见 11.3）全部能生成 ModelSpec，参数量与官方公布值误差 < 0.5%。
- **校验**：`ctest -R modelspec_golden`：对 `tests/golden/modelspec/*.json` 快照比对；离线用例使用仓库内置的 config.json + 伪造 safetensors 头，不依赖网络。

### 6.4 `estimator` —— 资源预估器

- **职责**：给定 ModelSpec/QIR + Recipe + HardwareProfile + 运行参数（上下文、并发、TP、后端），输出 **EstimateReport**：权重显存、量化元数据、KV Cache、激活/工作区、运行时开销、峰值显存、最大并发、最大上下文、TTFT/TPOT/吞吐预估、可部署性判定及原因。公式见第 9 章。
- **公共接口**：
  ```cpp
  struct EstimateInput { ModelSpec model; Recipe recipe; HardwareProfile hw; RuntimeParams rt; BackendId backend; };
  Result<EstimateReport> estimate(const EstimateInput&);
  Result<FeasibilityMatrix> estimate_matrix(std::span<const ModelSpec>, std::span<const Recipe>, std::span<const HardwareProfile>, RuntimeParams);
  Result<Curve> kv_vs_context(const EstimateInput&, std::span<const int> ctx_points);
  Result<Curve> memory_vs_concurrency(const EstimateInput&, std::span<const int> conc_points);
  ```
- **实现要点**：纯函数、无 I/O、可在毫秒级完成（矩阵 100 模型 × 20 硬件 × 5 配方 < 1s）；所有经验系数（运行时开销、带宽效率 η、MFU）来自 `CalibrationStore`，由 S15 反馈更新，初始值来自 `data/hardware_db` 与 `data/backends`。
- **验收**：参考集上 **显存预估误差 ≤ 10%**；校准后 **decode 吞吐误差 ≤ 25%**、TTFT 误差 ≤ 30%。
- **校验**：`ctest -R estimator_`（公式单测 + 手算用例）；`scripts/bench_reference.sh` 在真机上跑参考集并输出“预估 vs 实测”表，CI 夜间任务存档。

### 6.5 `planner` —— 兼容性与可部署性规划

- **职责**：维护并查询兼容性矩阵（配方 × 算法 × 后端 × 硬件架构），在 S00/S01 阶段**提前**拦截不合法组合，并给出替代推荐（“W4A8 在 sm_80 上 vLLM 不支持原生内核 → 推荐 W4A16-Marlin 或 TRT-LLM”）。
- **公共接口**：
  ```cpp
  Result<CompatVerdict> check(const Recipe&, AlgorithmId, BackendId, const HardwareProfile&);   // native / emulated / unsupported + 原因 + 内核名
  std::vector<Recommendation> recommend(const EstimateInput&, const Constraints&);            // 满足约束的 Top-K 组合
  Result<void> validate_jobspec(const JobSpec&);                                               // 汇总全部静态校验
  ```
- **实现要点**：矩阵数据在 `data/compat/matrix.yaml`（L2），规则按“最具体匹配优先”；校验包括：`per_group` 需 `group_size>0`、静态激活只允许 per_tensor、AutoAWQ 仅 W4A16 非对称、FP8 需 sm_89+/MI300/…、`hidden % group_size == 0` 等（覆盖原设想文档中 `check_config` 的全部规则）。
- **验收**：非法 JobSpec 在 **启动 5 秒内**被拒绝，并给出可执行的修复建议。
- **校验**：`ctest -R planner_`；`tests/golden/compat/` 用例表驱动（≥ 50 条组合）。

### 6.6 `quant`（C++ 数值核心，经 pybind11 暴露为 `mqde._core`）

- **职责**：量化的**唯一数值规范**：张量重排（粒度）、范围标定（MinMax/MSE/百分位/直方图/Moving Average）、QParams 计算、量化/反量化/假量化、FP8（E4M3/E5M2）转换、位打包（GPTQ/AWQ/Marlin/GGUF Q4_K/Q2_K…）、误差度量。
- **公共接口**：
  ```cpp
  enum class Granularity { kPerTensor, kPerChannel, kPerGroup, kPerHead, kPerBlock, kPerToken };
  enum class CalibAlgo { kMinMax, kMse, kPercentile, kHistogram, kMovingAvg };
  struct QuantSpec { NumFormat fmt; /* int2/3/4/8, fp8_e4m3, fp8_e5m2 */ bool symmetric; Granularity gran;
                     int group_size = 0; std::array<int,2> block{0,0}; CalibAlgo calib; float maxshrink = 0.8f; int grid = 100; };
  struct QParams { Tensor scale; std::optional<Tensor> zero; int32_t qmin, qmax; };

  Result<QParams> compute_qparams(TensorView x, const QuantSpec&);
  Result<void>    quantize(TensorView x, const QParams&, const QuantSpec&, TensorView q_out);
  Result<void>    dequantize(TensorView q, const QParams&, const QuantSpec&, TensorView x_out);
  Result<void>    fake_quant(TensorView x, const QuantSpec&, TensorView x_out, QParams* qp_out = nullptr);
  Result<Packed>  pack(TensorView q, const QParams&, PackFormat);
  ErrorMetrics    measure(TensorView ref, TensorView approx);   // mse, rmse, sqnr_db, cosine, max_abs, rel_fro, outlier_ratio
  ```
- **实现要点**：
  - Fake 与 Real 共用 `compute_qparams`（原设想文档“问题 3：Fake/Real 一致性”的根治方案）。
  - 舍入模式统一为 round-half-to-even，并写入规范文档 `docs/guides/quant-numerics.md`。
  - CPU 实现（AVX2/AVX-512/NEON 分派）+ 可选 CUDA 实现；Python 通过 DLPack 零拷贝调用。
  - Torch GPU 快速路径（`python/mqde/quant/numerics.py`）允许存在，但必须通过同一组 golden 向量。
- **验收**：INT 路径与 golden **逐位一致**；FP8 与 `torch.float8_e4m3fn` 转换逐位一致；打包后解包 round-trip 无损；4096×4096 per_group W4 MSE 标定 CPU 单线程 < 2s。
- **校验**：`ctest -R quant_`；`pytest python/tests/test_numerics_golden.py`（C++ 与 Torch 路径双跑对拍）；`data/golden/quant/*.npz` 由参考实现生成并冻结（L2）。

### 6.7 Python 执行器框架（`python/mqde`）

即原设想文档中的“七层量化框架”，作为 worker 内部结构：

| 子包 | 对应原设计层 | 职责 |
|------|-------------|------|
| `worker/` | 入口层 | RPC 循环、方法分发、取消、心跳、事件上报 |
| `registry.py` | 配置与注册层 | `ALGO_REGISTRY / MODEL_REGISTRY / KV_REGISTRY / EXPORTER_REGISTRY` |
| `models/` | 模型抽象层 | `BaseModel`：`get_blocks / get_block_linears / replace_module_subset / collect_first_block_input / set_modality` |
| `quant/blockwise.py` | 调度层 | `BlockwiseOpt.run_block_loop`，按 Block 处理控制显存，支持 offload |
| `quant/base.py` + `quant/quantizer.py` | 量化引擎层 | `BaseBlockwiseQuantization`：`w_qdq / a_qdq / w_q / get_replacement_params`；Quantizer 包装 `_core` |
| `quant/modules/` | Module 包装层 | `FakeQuantLinear / EfficientFakeQuantLinear / RealQuantLinear* / Fp8Linear / RotateLinear / OriginFloatLinear / QuantAttention*` |
| `quant/algorithms/` | 算法插件层 | RTN / AWQ / GPTQ / SmoothQuant / HQQ / QuaRot …，只实现 `block_opt` / `subset_transform` |
| `quant/kv/` | KV 量化 | `KV_REGISTRY`：Naive / KIVI …，区分 Prefill/Decode |
| `trace/` | 误差追踪 | 逐层/逐块 hook 记录误差（第 10 章） |
| `eval/` | 评测层 | PPL、lm-eval-harness 适配、evalscope 适配、logits 探针 |
| `export/` | 导出层 | vLLM(compressed-tensors/GPTQ/AWQ)、SGLang、GGUF、TRT-LLM 检查点 |

- **实现要点**：
  - 新增算法 = 继承 `BaseBlockwiseQuantization` + `@ALGO_REGISTRY.register("name")`，不改主流程。
  - 新增模型 = 继承 `BaseModel` 实现 Block 访问接口；主流架构可通过通用 `HFDecoderModel` + 架构 YAML 覆盖。
  - 新增后端 = 实现 `Exporter` + C++ 侧 `IDeployBackend`。
  - 支持 `data_free`（无校准数据）、多模态 `quant.language / quant.vision` 分别配置。
  - v1 允许 `ExternalAlgorithm` 适配器包装成熟库（GPTQModel / llm-compressor），以便早期端到端打通；自研实现逐步替换，且两者结果须在误差报告中可对比。
- **验收**：Qwen2.5-0.5B 在 CPU 上 RTN-W4A16 端到端 < 10 分钟；GPTQ-W4A16 在 7B/单卡 A100 上 < 30 分钟且 wikitext2 PPL 劣化 < 5%。
- **校验**：`pytest python/tests`（含小模型 `tests/fixtures/tiny-llama` 随机权重模型，CI 无 GPU 可跑）；`tests/integration/test_e2e_tiny.py`。

### 6.8 `pipeline` —— 编排与调度（C++）

- **职责**：JobSpec → DAG；任务队列（优先级、资源需求）；调度器（按设备空闲显存与作业预估需求分配）；阶段执行、重试（指数退避，可配置可重试错误码）、取消、断点续跑（基于缓存命中）；可复现性记录（环境指纹：代码 git sha、依赖版本、驱动版本、随机种子）。
- **公共接口**：
  ```cpp
  class IStage { public: virtual StageId id() const = 0; virtual Fingerprint fingerprint(const StageContext&) const = 0;
                 virtual Result<StageOutput> run(StageContext&, EventSink&, CancelToken) = 0; };
  class Orchestrator { public: Result<JobId> submit(JobSpec); Result<void> cancel(JobId); Result<void> retry(JobId, std::optional<StageId> from);
                       Result<JobStatus> status(JobId) const; };
  ```
- **实现要点**：状态机 `PENDING → PREFLIGHT → QUEUED → RUNNING(stage) → SUCCEEDED | FAILED | CANCELLED`；所有状态迁移持久化到 SQLite，服务重启后可恢复；单设备同时只运行一个 GPU 独占阶段（可配置）。
- **验收**：杀掉 server 后重启，未完成作业从最后一个成功阶段继续；worker 崩溃被检测并按策略重试。
- **校验**：`ctest -R pipeline_`（用 `FakeStage` 模拟失败/超时/取消）；故障注入集成测试 `tests/integration/fault_injection_test.cpp`。

### 6.9 `worker` —— C++ ↔ Python 执行器桥

- **职责**：启动/监控 Python worker 子进程，JSON-RPC 通信，传递取消，收集 stdout/stderr，资源限制（CUDA_VISIBLE_DEVICES、超时）。
- **协议**（`api/worker_rpc.schema.json`，L2）：
  - 请求：`worker.hello`、`model.inspect`、`calib.prepare`、`eval.run`、`quant.run`、`export.run`、`real.check`、`worker.shutdown`
  - 通知（worker → C++）：`event.progress`、`event.log`、`event.metric`、`event.trace`、`event.artifact`
  - 取消：`$/cancel`；心跳：`$/ping`（10s 无响应视为挂死）
  - 版本协商：`worker.hello` 交换 `protocol_version`，主版本不一致直接拒绝。
- **实现要点**：`IWorkerTransport` 抽象（v1 stdio；预留 gRPC/ZeroMQ 远程 worker）；stdout 只走协议帧，Python 日志一律写 stderr 并被转发为 `event.log`。
- **验收**：10 万条事件无丢失无乱序；worker 被 `kill -9` 时 C++ 3 秒内感知。
- **校验**：`ctest -R worker_`（使用 `python/mqde/worker/echo_worker.py` 桩）；协议 schema 双端校验测试。

### 6.10 `backend` —— 部署后端适配（C++）

- **职责**：把量化产物部署到推理引擎并返回可访问端点；报告后端实际内存占用。
- **公共接口**：
  ```cpp
  class IDeployBackend {
   public:
    virtual BackendId id() const = 0;
    virtual CompatVerdict supports(const Recipe&, const HardwareProfile&) const = 0;
    virtual ExportFormat required_format(const Recipe&) const = 0;            // 告诉 S09 用哪个 Exporter
    virtual Result<Endpoint> launch(const DeploySpec&, const ArtifactRef&, EventSink&, CancelToken) = 0;
    virtual Result<RuntimeMemory> query_memory(const Endpoint&) = 0;
    virtual Result<void> shutdown(const Endpoint&) = 0;
  };
  ```
- **实现要点**：v1 适配 vLLM / SGLang / llama.cpp（均为 OpenAI 兼容 HTTP 服务），以子进程方式启动，等待 `/health` 就绪，超时即失败并收集日志；启动参数（`--quantization`、`--kv-cache-dtype`、`--max-model-len`、`--gpu-memory-utilization`、`-tp`）由适配器根据 Recipe/DeploySpec 生成并写入报告以便复现。
- **验收**：三种后端均能部署 Qwen2.5-7B W4A16 并通过 smoke 请求。
- **校验**：`ctest -R backend_`（桩后端）；GPU 集成测试 `tests/integration/gpu/deploy_*.sh`（手动/夜间）。

### 6.11 `hal` —— 硬件插件层（稳定 C ABI）

- **职责**：让任意硬件以插件形式接入：设备发现、能力查询、内存、微基准（带宽/算力实测），长期支持编译与内核启动。
- **ABI**（`cpp/include/mqde/hal/mqde_hal.h`，**L2，破坏性修改为 L3**）：
  ```c
  #define MQDE_HAL_ABI_VERSION 1
  typedef struct mqde_hal_plugin_v1 {
    uint32_t abi_version; const char* name; const char* vendor; const char* plugin_version;
    int  (*device_probe)(mqde_device_info* out, int max, int* n);
    int  (*supports)(const mqde_capability_query* q, mqde_capability_result* r);   /* dtype/算子/内核能力 */
    int  (*alloc)(int dev, size_t bytes, void** ptr);
    int  (*free)(int dev, void* ptr);
    int  (*profile)(int dev, const mqde_profile_request* req, mqde_profile_result* res); /* 带宽、各 dtype GEMM TFLOPS */
    int  (*compile)(const mqde_compile_request*, mqde_compile_result*);             /* 可选，v1 可为 NULL */
    int  (*launch)(const mqde_launch_request*);                                     /* 可选，v1 可为 NULL */
    const char* (*last_error)(void);
  } mqde_hal_plugin_v1;
  MQDE_HAL_EXPORT const mqde_hal_plugin_v1* mqde_hal_get_plugin_v1(void);
  ```
- **实现要点**：插件目录 `$MQDE_HOME/plugins/` + 内置插件；加载失败只告警；`profile` 结果写入硬件画像缓存，作为预估器的实测带宽/算力来源；提供 `cpp/plugins/template/` 作为新硬件接入模板和 `docs/guides/add-hardware-plugin.md`。
- **验收**：用模板插件（模拟设备）不改任何上层代码即出现在 `mqde probe` 与前端硬件页中。
- **校验**：`ctest -R hal_`（ABI 版本不符/缺符号/空函数指针均被安全处理）；`scripts/check_abi.sh` 用 `abidiff` 防止 ABI 意外变化。

### 6.12 `eval` —— 评估与压测

- **精度评估（Python）**：PPL（wikitext2/c4，滑窗）、lm-eval-harness 任务（MMLU、GSM8K、ARC、HellaSwag…）、evalscope 适配、自定义数据集（JSONL：`prompt/answer/metric`）；**logits 探针集**（固定 128 条 prompt，逐阶段记录 top-k logits，用于 KL/Top-1 一致率）。
- **性能压测（C++）**：对 OpenAI 兼容端点发起流式请求，按 `concurrency × context × input/output_len` 网格压测，指标：TTFT(p50/p90/p99)、TPOT、tokens/s、req/s、峰值显存（HAL/NVML 采样 100ms）、主机内存、功耗（若可得）；**OOM 边界**（二分搜索最大并发与最大稳定上下文）。
- **验收**：压测器自身开销 < 1%（用桩服务器验证）；同配置 3 次重复吞吐方差 < 5%。
- **校验**：`ctest -R bench_`（本地 mock server 注入固定延迟，验证 TTFT/TPOT 计算正确）；`pytest python/tests/eval`。

### 6.13 `storage` —— 缓存池与元数据

- **职责**：内容寻址存储（CAS）+ SQLite 元数据；管理：模型元数据缓存、HF 原始模型缓存（复用 `~/.cache/huggingface`）、校准统计、量化产物、编译产物、硬件画像、基准历史、兼容性观测、部署配方、预估校准系数。
- **布局**：
  ```
  $MQDE_HOME (默认 ~/.mqde)
  ├── db/mqde.sqlite            # jobs, stages, events, artifacts, hw_profiles, bench_history, calibration
  ├── cas/sha256/ab/cdef…/      # 产物本体（不可变）
  ├── runs/<job_id>/            # 作业工作目录（日志、报告、指向 CAS 的软链接）
  ├── plugins/
  └── config.yaml
  ```
- **实现要点**：数据库迁移脚本 `cpp/src/storage/migrations/NNNN_*.sql`（L2，只增不改）；LRU + 引用计数 GC，`mqde cache gc --max-size 500G`；写入原子（临时目录 + rename）。
- **验收**：并发写入无损坏；GC 不删除被运行中作业引用的产物。
- **校验**：`ctest -R storage_`；迁移前后数据兼容测试。

### 6.14 `server` —— API 服务

- **REST（`/api/v1`）**：

  | 方法 | 路径 | 说明 |
  |------|------|------|
  | GET | `/health`、`/version` | 健康与版本 |
  | GET | `/hardware`、`/hardware/profiles` | 本机探测结果、硬件画像库 |
  | POST | `/hardware/{id}:profile` | 触发微基准测量 |
  | GET | `/recipes`、`/algorithms`、`/backends`、`/datasets` | 注册表内容（前端下拉框数据源） |
  | POST | `/models:resolve` | 模型 id → ModelSpec（不下载权重） |
  | POST | `/estimate`、`/estimate:matrix`、`/estimate:curves` | 预估、可部署矩阵、曲线 |
  | POST | `/plan:recommend` | 按约束推荐配方/后端/硬件 |
  | POST/GET | `/jobs`、`/jobs/{id}` | 提交/列表/详情 |
  | POST | `/jobs/{id}:cancel`、`/jobs/{id}:retry` | 控制 |
  | GET | `/jobs/{id}/events?after_seq=` | 事件回放 |
  | GET | `/jobs/{id}/error-trace`、`/report`、`/artifacts` | 误差、报告、产物 |
  | GET | `/compare?jobs=a,b,c` | 多作业对比 |
  | GET/DELETE | `/cache` | 缓存池 |

- **WebSocket**：`/api/v1/jobs/{id}/stream`（事件推送，支持 `after_seq` 断线续传）、`/api/v1/stream`（全局作业状态）。
- **验收**：OpenAPI 中每个端点都有契约测试；事件推送延迟 < 200ms。
- **校验**：`ctest -R server_`；`scripts/contract_test.sh`（schemathesis 对 OpenAPI 做模糊测试）。

### 6.15 `web` —— 前端

| 页面 | 核心内容 | 关键可视化 |
|------|---------|-----------|
| 新建作业（向导） | ① 模型（HF id 搜索/本地路径/上传 config）② 配方 W4A8/W4A16/W2A16/W2A8/FP8 + 算法 + 高级参数 ③ 评估数据/类型 ④ 硬件与后端 ⑤ 上下文/并发/约束 → 实时预估侧栏 | 右侧实时预估卡片（随参数变化 debounce 刷新） |
| 预估中心 | 模型占用、KV Cache、单/多并发资源、可部署性 | ① 显存构成堆叠柱（权重/量化元数据/KV/激活/运行时）vs 设备容量线 ② KV Cache–上下文长度曲线（多配方对比）③ 显存–并发曲线 + 最大并发标注 ④ 模型×硬件可部署热力矩阵 ⑤ 预估 TTFT/TPOT/吞吐 |
| 作业执行 | 整体执行过程 | DAG 进度图（阶段状态/耗时/缓存命中）、实时日志（按级别/阶段过滤）、资源实时曲线 |
| 量化误差 | 哪一层、哪一环节误差大 | ① 阶段误差瀑布图（baseline→transformed→fake W→fake W+A→+KV→real→deployed 的精度/KL 逐级变化）② 层 × 指标热力图（SQNR/cosine/MSE）③ Top-K 最差层列表（可点进看权重/激活分布直方图、离群值比例）④ Block 输出误差累积曲线 |
| 评估报告 | 精度 + 性能 | 雷达图（任务精度 vs 基线）、TTFT/TPOT–并发曲线、OOM 边界、预估 vs 实测对比 |
| 对比 | 多作业横向对比 | 精度-显存-吞吐 帕累托散点 |
| 硬件 | 本机设备、画像库、插件 | 设备卡片、微基准结果 |
| 缓存池 | 产物、占用、命中率 | 占用树图 |

- **实现要点**：API 类型由 `openapi-typescript` 生成（禁止手写 DTO）；WS 断线自动重连并按 `after_seq` 补齐；所有图表组件放在 `web/src/charts/`，输入为纯数据；中英文 i18n；暗色模式。
- **验收**：完整走通“新建 → 预估 → 运行 → 看误差 → 看报告”；Lighthouse 性能 ≥ 80。
- **校验**：`pnpm lint && pnpm typecheck && pnpm test`（vitest）；`pnpm e2e`（Playwright，对 mock server 运行关键流程）。

### 6.16 `cli` —— 命令行

```
mqde probe [--json]                                  # 硬件探测
mqde model inspect Qwen/Qwen2.5-7B-Instruct          # 仅元数据解析
mqde estimate -m <id> -r W4A16 --hw nvidia/a100-80g --ctx 8192 --conc 1,8,32
mqde plan --model <id> --max-mem 24G --min-acc 0.97  # 推荐
mqde run job.yaml [--from-stage S07] [--dry-run]     # 执行作业（本地）
mqde submit job.yaml --server http://host:8080       # 提交到服务
mqde serve --port 8080                               # 启动 API 服务
mqde cache ls|gc|rm
mqde report <job_id> --format html|json
```

---

## 7. 核心数据结构与契约

所有结构有对应 JSON Schema（`schemas/`），`apiVersion` 字段用于演进。

### 7.1 JobSpec（用户输入）

```yaml
apiVersion: mqde/v1
kind: Job
metadata: { name: qwen7b-w4a16-gptq, tags: [demo] }
model:
  source: huggingface              # huggingface | modelscope | local
  id: Qwen/Qwen2.5-7B-Instruct
  revision: main
  modality: llm                    # llm | vlm | vla | world
  trust_remote_code: false
quant:
  recipe: W4A16                    # W4A8 | W4A16 | W2A16 | W2A8 | FP8
  algorithm: gptq                  # rtn | awq | gptq | smoothquant | hqq | awq+gptq …
  overrides: { weight: { group_size: 128, calib: mse } }
  per_modality: {}                 # 例：{ vision: { recipe: W8A16, algorithm: rtn } }
  mixed_precision: { keep_fp: [lm_head, embed_tokens] }
  kv_cache: { dtype: fp16 }        # fp16 | fp8_e4m3 | int8 | int4
calibration: { dataset: wikitext2, n_samples: 128, seq_len: 2048, batch: 1 }   # 省略 = data-free
evaluation:
  stages: [baseline, fake_quant, real_quant, deployed]
  accuracy:
    - { task: ppl, dataset: wikitext2 }
    - { task: mmlu, limit: 500 }
    - { task: custom, path: ./my_eval.jsonl, metric: exact_match }
  performance: { context_lengths: [4096, 8192], concurrency: [1, 8, 32], input_len: 1024, output_len: 256, find_oom_boundary: true }
deploy:
  backend: vllm                    # vllm | sglang | llamacpp | trtllm
  hardware: { target: auto }       # auto | 本机设备序号 | 画像名如 nvidia/a100-80g
  tensor_parallel: 1
  gpu_memory_utilization: 0.9
constraints: { min_accuracy_ratio: 0.97, max_ttft_ms: 500, max_memory_gb: 24 }
```

### 7.2 其他核心结构（字段摘要）

| 结构 | 关键字段 |
|------|---------|
| `ModelSpec` | family, arch, n_layers, hidden, n_heads, n_kv_heads, head_dim, intermediate, vocab, attn_type, mla{kv_lora_rank, rope_dim}, sliding_window, moe{n_experts, top_k, shared}, modalities[], tensors[{name, shape, dtype}], total_params, active_params |
| `Recipe` | name, weight QuantSpec, act QuantSpec \| none, kv QuantSpec \| none, default_algorithm, keep_fp_patterns, maturity(production/experimental) |
| `HardwareProfile` | id, vendor, arch, mem_gb, mem_bw_gbps, peak_tflops{fp32,fp16,bf16,fp8,int8,int4}, supported_formats[], interconnect, measured{bw, tflops, ts} |
| `EstimateReport` | breakdown{weights, qmeta, kv, activation, runtime}, peak, capacity, feasible, reasons[], max_concurrency, max_context, perf{ttft_ms, tpot_ms, tok_s}, assumptions{η, mfu, overhead}, confidence |
| `ErrorTrace` | job_id, stage, unit(QUnit name), block_idx, modality, metrics{mse, sqnr_db, cosine, max_abs, rel_fro, outlier_ratio, act_range}, qparams_ref(artifact) |
| `StageMetric` | stage, dataset/task, value, baseline_value, delta, kl_vs_baseline, top1_agree |
| `Event` | job_id, seq(单调递增), ts, type(stage_started/progress/finished/log/metric/trace/artifact/job_state), stage, level, payload |
| `JobReport` | spec, env_fingerprint, stages[], estimate, measured, error_summary{worst_units[], worst_stage}, verdict |

---

## 8. 量化配方定义

配方文件位于 `data/recipes/*.yaml`（新增 L1，修改默认值 L2）。

| 配方 | 权重 | 激活 | 默认算法 | 成熟度 | 典型后端/内核 | 硬件要求 |
|------|------|------|---------|--------|---------------|---------|
| **W4A16** | INT4, per_group g128, 对称/非对称 | FP16/BF16 | GPTQ / AWQ | 生产 | vLLM(Marlin/GPTQ/AWQ)、SGLang、llama.cpp(Q4_K)、TRT-LLM | 通用；Marlin 需 sm_80+ |
| **W4A8** | INT4, per_group g128 / per_channel | INT8 per_token 动态（或 FP8） | SmoothQuant+GPTQ / QServe 类 | 生产（后端受限） | TRT-LLM(W4A8-AWQ, FP8 act)、LMDeploy/QServe | Ampere(INT8)/Hopper(FP8) |
| **W2A16** | INT2, per_group g64/g32 | FP16 | GPTQ / HQQ | 实验 | llama.cpp(Q2_K)、自定义插件 | 通用（性能视内核） |
| **W2A8** | INT2, per_group g64 | INT8 per_token | SmoothQuant+GPTQ/HQQ | 实验（默认仅 fake_quant 评估） | 需硬件插件提供内核 | — |
| **FP8** | FP8 E4M3, per_channel（或 block 128×128） | FP8 E4M3 per_token 动态 | RTN（FP8 通常无需复杂校准） | 生产 | vLLM、SGLang、TRT-LLM | sm_89+/sm_90、MI300、Gaudi |

规则：
- 每个配方 = 权重 QuantSpec + 激活 QuantSpec + KV QuantSpec（可空）+ 默认算法 + 保持浮点的层模式（默认 `lm_head`、`embed_tokens`、视觉 patch embedding）。
- 实验级配方在 UI 上显示警告徽标，并在报告中标注“真实部署内核不可用时仅为仿真结果”。

---

## 9. 资源预估模型

记号：`b_w/b_a/b_kv` 为位宽（bit），`g` 为 group size，`L` 层数，`H_kv` KV 头数，`d_h` 头维度，`C` 上下文长度，`B` 并发数，`TP` 张量并行度。

### 9.1 模型占用（权重）

```
W_bytes  = Σ_{u ∈ 量化单元} n_u · b_w / 8                        # 量化权重
         + Σ_{u ∈ 保持浮点单元} n_u · b_orig / 8                   # lm_head/embedding 等
Q_meta   = Σ_{u ∈ 量化单元} (n_u / g) · (s_bytes + z_bytes)       # scale(2B) + zero(非对称，按打包格式 0.5~2B)
         # per_channel: out_features · s_bytes ; per_block(128×128): n_u/16384 · 4B
Weights  = (W_bytes + Q_meta) / TP  (+ 未切分部分如 embedding 视后端策略)
```

MoE：总显存按全部专家计；计算/带宽按激活专家（top_k + shared）计。

### 9.2 KV Cache 占用

```
kv_per_token = Σ_{l=1..L} 2 · H_kv · d_h · b_kv / 8   (+ KV 量化 scale 开销)
  · MLA（DeepSeek）：   Σ_l (kv_lora_rank + qk_rope_head_dim) · b_kv / 8        # 无 ×2
  · 滑窗层：            有效长度 = min(C, window)
  · 线性注意力/SSM 层： 常数状态大小，与 C 无关
KV_total     = kv_per_token · C_eff · B / TP
  · Paged 后端：C_eff 按 block_size（如 16）向上取整
```

（与需求图一致：`KV Cache ≈ Concurrency × Context × Layers × KV Heads × Head Dim × [Bytes(K)+Bytes(V)]`）

### 9.3 激活 / 工作区 / 运行时

```
Act_peak ≈ T_batch · (k1·hidden + k2·intermediate/TP) · b_act/8  +  T_logits · vocab · 4
           # T_batch = max_num_batched_tokens（prefill 分块大小）；k1,k2 为架构/后端经验系数
Runtime  = backend_overhead(hw, backend)   # CUDA context、cuBLAS workspace、CUDA Graph、通信 buffer
Peak     = Weights + KV_total + Act_peak + Runtime
Feasible ⇔ Peak ≤ mem_total · gpu_memory_utilization
```

### 9.4 并发与上下文上限

```
KV_budget        = mem_total · util − Weights − Act_peak − Runtime
Max_concurrency  = ⌊ KV_budget / (kv_per_token · C) ⌋
Max_context(B)   = ⌊ KV_budget / (kv_per_token · B) ⌋   ∧ ≤ model.max_position_embeddings
```

### 9.5 性能预估（屋顶线 + 经验效率）

```
Decode（访存受限）： TPOT ≈ (W_active_bytes + KV_read_bytes(B, C)) / (BW · η_bw) + t_launch
                    吞吐 ≈ B / TPOT，且 ≤ FLOPS_fmt · η_c / (2 · N_active)
Prefill（算力受限）：TTFT ≈ 2 · N_active · P / (FLOPS_fmt · MFU) + 注意力项 + 排队时间
TP>1：再加 all-reduce 通信项 2·(TP−1)/TP · hidden · b_act/8 · tokens / 链路带宽
```

`η_bw、MFU、k1、k2、backend_overhead` 按 **(硬件, 后端, 配方)** 三元组存入 `CalibrationStore`，初值来自 `data/hardware_db` 与 `data/backends`，每次 S13 实测后由 S15 用指数滑动平均更新，形成“Estimate → Deploy → Measure → Update → Better Estimate”闭环。报告中给出 `confidence`（有无该三元组的历史数据）。

---

## 10. 量化误差追踪与归因

目标：直观清晰地知道**哪一个节点（层）**量化效果差、**哪一个环节（转换阶段）**引起的误差大。

### 10.1 两个维度

**维度 A：阶段归因（哪一环节）** —— 对每个启用阶段，在同一评估集 + logits 探针集上测量：

```
S05 baseline(FP) → S06 transformed → S07a fake(W only) → S07b fake(W+A) → S07c fake(W+A+KV)
               → S10 real_quant(参考内核) → S12 deployed(真实后端)
Δ_stage = metric(stage) − metric(previous_stage)
```

- `transformed` 与 baseline 差异应≈0（验证等效变换）；非零即标红“等效变换异常”。
- `fake → real` 差异应≈0（验证 Fake/Real 一致性）；超过阈值即标红“导出/打包问题”。
- `real → deployed` 差异反映后端内核/格式转换误差。
- 前端用**瀑布图**展示每个 Δ，自动高亮最大 Δ 的环节。

**维度 B：单元归因（哪一个节点）** —— 在 S07 的 Blockwise 循环中，`trace/` 为每个 QUnit 记录：

| 指标 | 含义 |
|------|------|
| `w_sqnr_db`、`w_mse`、`w_rel_fro` | 权重量化误差 |
| `out_cosine`、`out_rel_err` | 该层输出（真实校准输入下）与 FP 输出的相似度 |
| `act_range`、`act_outlier_ratio`、`act_kurtosis` | 激活分布特征（解释为什么误差大） |
| `block_out_mse`、`block_out_cosine` | 整个 Block 输出与 FP Block 输出（累积误差） |
| `sensitivity` | 可选：单层回退为 FP 时的 PPL/KL 改善（敏感度分析，开销大，按需开启） |

- 记录方式：每个 Block 处理前后跑一次 FP 参考前向与量化前向（复用 first_block_input 流水），开销控制在量化总时间的 +15% 以内；数据以列式（Parquet 或 safetensors + JSON）写入 artifact，事件流只推送摘要。
- 前端：层 × 指标热力图、Block 累积误差曲线、Top-K 最差层（附分布直方图），并支持“一键将 Top-K 层设为高精度重新量化”（混合精度建议 → 生成新 JobSpec）。

### 10.2 验收

- 人为在某一层注入噪声（测试用 `debug.inject_noise: {unit: model.layers.5.mlp.down_proj}`），报告必须把它排在 Top-1。
- 人为破坏导出打包（测试开关），`fake → real` 阶段必须被标记为最大误差环节。

---

## 11. 质量保障体系

### 11.1 测试金字塔

| 层 | 工具 | 范围 | 运行时机 |
|----|------|------|---------|
| 单元 | GoogleTest / pytest / vitest | 每个模块公共接口 | 每次 PR（必须） |
| Golden | 冻结的 `.npz/.json` 快照 | 量化数值、ModelSpec、预估公式、兼容矩阵 | 每次 PR（必须） |
| 契约 | JSON Schema、schemathesis | OpenAPI、Worker RPC、事件 | 每次 PR（必须） |
| 集成（CPU） | tiny 随机模型 | 端到端 Job：S00→S14（跳过 GPU 阶段或用 llama.cpp CPU） | 每次 PR（必须） |
| 集成（GPU） | 自托管 runner | 参考集真实量化/部署/压测 | 夜间 / 发布前 |
| E2E 前端 | Playwright | 关键用户流程 | 每次 PR（mock server） |
| 性能回归 | Google Benchmark / 参考集 | 数值核心速度、预估精度 | 夜间 |

### 11.2 CI 门禁（合并到 `main` 的硬性条件）

1. `cmake --build` 在 Linux(gcc/clang)、macOS(clang) 通过；Windows 编译通过（probe/common/estimator 最小集）。
2. `ctest`、`pytest`、`pnpm test` 全绿；覆盖率：`quant`/`estimator`/`planner` ≥ 90%，其他 C++ ≥ 70%，Python ≥ 70%。
3. `clang-format`、`clang-tidy`、`ruff`、`mypy`、`eslint`、`tsc` 零错误。
4. 契约检查：OpenAPI / schemas 变更必须同时更新生成代码（`scripts/gen_api.sh --check` 无 diff）。
5. 依赖方向检查 `scripts/check_deps.py`、ABI 检查 `scripts/check_abi.sh`。
6. 文档同步：修改 L2 路径时 PR 必须勾选“已更新相关文档”。

### 11.3 参考集（Reference Set）

| 类别 | 模型 | 用途 |
|------|------|------|
| tiny | `tests/fixtures/tiny-llama`（随机权重，2 层） | CI 无 GPU 端到端 |
| 小 | Qwen2.5-0.5B-Instruct、Llama-3.2-1B | CPU/单卡快速验证 |
| 中 | Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct | 主力验收 |
| MoE/MLA | DeepSeek-V2-Lite、Qwen1.5-MoE-A2.7B | 结构覆盖 |
| VLM | Qwen2-VL-2B-Instruct | 多模态 |

参考硬件：NVIDIA A100-80G / H100 / RTX 4090、Apple M 系列、x86 CPU。

### 11.4 精度验收阈值（默认，可在 JobSpec.constraints 覆盖）

| 配方 | wikitext2 PPL 相对劣化 | 任务精度保持率 |
|------|------------------------|----------------|
| FP8 | ≤ 1% | ≥ 99% |
| W4A16 (GPTQ/AWQ) | ≤ 5% | ≥ 97% |
| W4A8 | ≤ 8% | ≥ 95% |
| W2A16 / W2A8 | 仅记录，不设门槛（实验） | 仅记录 |

---

## 12. 可观测性、安全与版本演进

### 12.1 可观测性
- 结构化日志（JSON Lines）：`ts, level, job_id, stage, component, msg, kv…`；每个作业日志归档到 `runs/<job_id>/logs/`。
- 指标：`/metrics`（Prometheus 格式）暴露队列长度、阶段耗时、缓存命中率、worker 重启次数。
- 可复现：每份报告包含环境指纹（git sha、C++/Python 依赖版本、驱动/CUDA 版本、硬件、随机种子、完整 JobSpec），`mqde run --reproduce <job_id>` 可重放。

### 12.2 安全
- HF/ModelScope token 只从环境变量或 `$MQDE_HOME/credentials`（600 权限）读取，**绝不写入日志、报告、事件、数据库**。
- `trust_remote_code` 默认关闭；开启时 worker 在独立进程中运行并在报告标注。
- 本地路径访问限定在允许目录列表内（`config.yaml: allowed_paths`），防止 API 被用来读任意文件。
- server 默认只监听 `127.0.0.1`；对外暴露需显式配置并启用 token 认证。

### 12.3 版本与演进
- **代码版本**：Git，主干开发（`main` 受保护）+ 短生命周期特性分支；Conventional Commits；SemVer 发版打 tag（`v0.x` 阶段允许破坏性变更但必须写入 CHANGELOG）。详见 [../AGENTS.md](../AGENTS.md) 第 5 节。
- **契约版本**：`apiVersion: mqde/v1`、`protocol_version`、`MQDE_HAL_ABI_VERSION`、数据库迁移号 —— 四者独立演进，破坏性变更必须升主版本并提供兼容期。
- **数据版本**：`data/` 下 recipes/hardware_db/compat 变更随代码走 Git；golden 数据变更必须在 PR 中说明原因（L2）。
- **大文件**：模型权重、数据集不进 Git；小型测试夹具（< 5MB）可进仓库，更大的使用 Git LFS（L2 审核）。
