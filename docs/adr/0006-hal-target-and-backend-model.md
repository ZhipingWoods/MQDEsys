# ADR-0006: HAL 的 Target / Backend 模型（FlagTree 对齐）——不需要 ABI v2

| 字段 | 内容 |
|------|------|
| 状态 | **Proposed**（未批准；本文档只起草，**不实现任何代码**，见第 6 节） |
| 日期 | 2026-10-13 |
| 作者 | Claude（Sonnet 5.5） |
| 审核人 | 负责人（zhipingwu）—— 待定 |
| 依赖 | T-1101（HAL ABI v1）、T-1102/1103/1104（cuda / metal / rocm 插件）已 DONE 或 PARTIAL |
| 影响任务 | T-R14（L3，本 ADR 由其起草）、T-R15（L1，回归基线） |

---

## 1. 背景与结论先行

T-R14 要求"hal：Target + Backend 接口 + 插件发现"，对照 `docs/REFERENCES.md` §5 的 FlagTree 设计，并标为 L3（因为动 HAL ABI 就是破坏性变更）。
逐条对照后的结论是：**FlagTree 设计里的东西，ABI v1 已经覆盖了大部分，剩下的几项都不需要改 ABI。** 推荐的结论是**不发布 ABI v2**，把剩余项拆成两个 L1 小任务。本 ADR 的价值就在于把这件事说清楚，免得为了"对齐"而做一次不必要的破坏性变更。

## 2. 逐项对照

| REFERENCES §5 里的设计 | ABI v1 / 现有代码 | 缺口 |
|------|------|------|
| `GPUTarget{backend, arch, warp_size}`（如 `cuda:sm_89:32`） | `mqde_hal_device` 有 `vendor`、`arch`（"sm_89" / "gfx1100" / "ascend910b"）、`kind`；插件有唯一 `name` | **无**：Target 字符串可以由 `<plugin>:<arch>` 拼出，不需要新字段。`warp_size` 没有任何消费者（估算器、规划器都不用），不建议为它扩 ABI |
| `is_active()`（驱动与设备存在时为真） | `device_probe` 返回 0 台设备 = 运行时不存在，这是约定的正常结果（头文件规则第 4 条） | **无** |
| `supports_target(target)` / `get_current_target()` | 由 `device_probe` 的结果派生 | 加载层缺一个按 Target 查询的小函数（L1，不碰 ABI） |
| `capabilities()`：NATIVE / NON_NATIVE / UNSUPPORTED | `supports()` → `{supported, native, peak_tflops, note}`，两个布尔恰好表达三态 | **无**。已被使用：规划器据此回答"W4A8 在这块卡上有没有原生内核" |
| `benchmark()` | `profile()`：`bandwidth` / `gemm`，已在 4090 上实测 886–919 GB/s（标称 1008 的 88–91%）（T-0805） | **无** |
| `deploy(artifact) → endpoint` | v1 **没有**；`alloc` / `free` / `compile` / `launch` 是占位 | 见 D1：**不应放进 HAL** |
| 插件目录 `cpp/plugins/<vendor>/`、`backend.conf`、`capabilities.yaml` | 每厂商一个目录已成立（cuda / cpu / metal / rocm / template）；运行期 `$MQDE_PLUGIN_DIR` → `$MQDE_HOME/plugins` → `<exe>/plugins` 发现；静态能力数据在 `data/hardware_db/**`（`native_formats`、峰值算力） | `backend.conf` 与每插件 `capabilities.yaml` 重复了 `hardware_db` 已有的信息（见 D3） |
| 构建期 `MQDE_BACKENDS=nvidia;ascend;metax` | `mqde_add_plugin()` 对每个插件目录生成独立的可加载模块；加载失败只影响该插件 | 没有"只编某几个"的开关（见 D4，价值小） |
| 运行期 `MQDE_BACKEND` 强制指定后端 | **没有**（代码里搜不到） | 见 D2 |
| `data/benchmarks/ref/<backend>/<model>.yaml` 回归基线 | `data/benchmarks/{local,community}/<hw-slug>/` + `schema.json`（T-0905/T-R20，"相同硬件才匹配"） | 见 D5：布局已有，缺的是"每次发布与上一次对比"的流程（T-R15，L1） |

## 3. 决策点

### D1 `deploy(artifact) → endpoint` 放不放进 HAL

**推荐：不放。** 理由：

1. HAL 回答"**这台机器上有什么硬件、它原生支持什么**"；部署回答"**哪个推理引擎在它上面跑**"。后者已经有自己的层：`data/backends/<name>.yaml`（估算画像）、`python/mqde/backend/*`（vLLM / SGLang / llama.cpp 适配器）、`data/compat/matrix.yaml`（引擎能加载什么格式）。
2. 一个 C ABI 的 `deploy()` 只能是"启动一个子进程并返回端口"，这正是 Python 适配器已经在做、且需要大量引擎特定的诊断（日志解析、启动超时、显存不足的原因）的事。放进 C ABI 会让每个硬件厂商插件都要实现一遍引擎启动逻辑，且无法测试（`ServerBackend` 现在可以用 mock 服务器端到端测）。
3. FlagTree 的 backend 是**编译器后端**（Triton → 各厂商代码生成），`deploy` 在它那里的含义和我们不同。照搬名字会误导。

### D2 运行期强制后端 `MQDE_BACKEND`

**推荐：做，且只在加载层（L1，不改 ABI）。** 用途：在装了多种硬件的机器上，让探测、微基准和估算对"只这一个插件"生效（排查插件问题、做可复现的对比）。语义：值是插件名（逗号分隔可多个），未列出的插件不加载；值里有不存在的插件名要明确报错而不是静默忽略。

### D3 `backend.conf` / 每插件 `capabilities.yaml`

**推荐：不加。** 静态能力已在 `data/hardware_db/**`，那里的 `native_formats` 与峰值算力被估算器和规划器使用，是单一事实来源。再让每个插件目录带一份会产生两处需要同步的数据（`scripts/check_*` 已经为类似问题存在）。插件的动态能力走 `supports()`。

### D4 构建期 `MQDE_BACKENDS`

**推荐：不做。** 每个插件是独立的 `MODULE`，加载失败互不影响；不想要某个插件时不把 `.so` 放进插件目录即可。为"少编几个"加一个构建开关，换来的只是节省几秒编译时间，却多一条需要测试的配置路径。

### D5 回归基线（T-R15，L1）

布局已经有了。缺的是流程：一个脚本把当前实测与同硬件上一次的基线对比，超出容差时失败。放在 T-R15，不属于本 ADR。

## 4. 推荐的结论

1. **不发布 HAL ABI v2**；`abi_v1.golden` 保持不变，`scripts/check_abi.sh` 继续守护。
2. 拆成两个 L1 小任务（批准后即可做）：
   - **T-R14a**：`MQDE_BACKEND` 过滤 + 按 Target（`<plugin>:<arch>`）查询的加载层函数，带测试（含"不存在的名字报错"）；
   - **T-R14b**：把 REFERENCES §5 里已满足的条目标注"由 v1 满足"，把没采纳的（`warp_size`、`deploy`、`backend.conf`、`MQDE_BACKENDS`）写明理由，避免下一个人重复提议。
3. T-R14 本身在 T-R14a/b 完成后记为 DONE，并注明与原任务书的偏离。

## 5. 不做的事

- 不改 `mqde_hal.h` 的任何结构体，不改 `MQDE_HAL_ABI_VERSION`。
- 不为没有消费者的字段（`warp_size`）扩展 ABI。
- 不在没有对应硬件的情况下新增插件（Ascend 仍是 TODO：本机与 4090 都没有 CANN 头文件、库或设备）。

## 6. 待批准

请负责人确认第 4 节的结论（推荐：不发布 ABI v2，批准 T-R14a / T-R14b），并把状态改为 Accepted。若负责人仍希望 HAL 提供 `deploy`，请先说明它要服务的具体场景，因为 D1 的第 2 点（无法测试、重复引擎逻辑）需要有对策才能成立。
