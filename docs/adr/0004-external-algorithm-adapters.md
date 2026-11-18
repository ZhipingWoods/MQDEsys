# ADR-0004: 外部量化算法适配器（GPTQModel / llm-compressor）

| 字段 | 内容 |
|------|------|
| 状态 | **Proposed**（未批准；本文档仅起草，不实现任何代码，见第 5 节） |
| 日期 | 2026-10-10 |
| 作者 | Claude（Sonnet 5） |
| 审核人 | 负责人（zhipingwu）—— 待定 |
| 依赖 | T-0508（Blockwise 引擎 + Fake/Real 模块包装）已 DONE |
| 影响任务 | T-0609（L3）、T-1302（本 ADR 由其起草） |

---

## 1. 背景

T-0609 要求把 GPTQModel 和/或 llm-compressor 包装成一个注册算法（`external:gptqmodel` 等），输出统一的 QParams 与 trace，验收标准是"同配置下与自研 GPTQ 结果在报告中可并排对比"。引入这两个包都是"新增第三方运行时依赖"，按 AGENTS.md 是 L3，必须先有 ADR 并获批。

现状（2026-10-10）：

- 本项目**已经有自研、已验证的 GPTQ 和 AWQ 实现**（`python/mqde/quant/algorithms.py`，T-0601/T-0602，DONE），在真实模型（Qwen2.5-0.5B 等）上跑通并导出部署过（见 `internal/PROGRESS.md` 多条记录）。T-0609 不是"缺少 GPTQ 能力"，而是"想要一个独立实现做交叉验证/信任背书"。
- `3rdpart/GPTQModel` 已经作为**只读参考源码的 git 子模块**存在（仅供照抄结构/公式用，见 `docs/REFERENCES.md` §2），从未作为运行时依赖被 import 过。这次要讨论的是另一件事：把它（或 llm-compressor）变成一个**真正在 `mqde` 环境里跑起来的 Python 依赖**。
- 本项目已经有过一次几乎相同形状的决策（ADR-0002，评测框架 lm-eval/evalscope）：结论是"子进程调用官方产物，不把框架 import 进 `mqde` 环境"，理由是依赖版本冲突（vLLM 固定的 `transformers` 版本与量化主流程环境不一致）。这次的约束非常相似。

## 2. 许可证与依赖（已核实，不是猜测）

| 项目 | 许可证 | 来源 | 备注 |
|------|--------|------|------|
| GPTQModel (ModelCloud) | **Apache-2.0**（核心代码） | GitHub 仓库 license 字段 + `pyproject.toml` 的 `license = "Apache-2.0"` + 源文件 SPDX 头 | **例外**：可选的 Swordfish 内核（`gptqmodel_ext/swordfish/`）是 **AGPL-3.0-or-later**，运行时 JIT 编译链接，不是默认必需组件；若启用，AGPL 的传染性条款需要单独评估。Fast Hadamard Transform 代码（`gptqmodel_ext/hadamard/`）是 BSD-3-Clause。 |
| llm-compressor (vllm-project / Red Hat AI) | **Apache-2.0** | GitHub 仓库 license 字段 | 面向 compressed-tensors 格式，和本项目 T-0701 已经在用的导出格式（`compressed-tensors`）是同一家族，潜在集成摩擦可能比 GPTQModel 更小。 |

两者核心都是 Apache-2.0，商用友好，可以直接依赖（前提：不触发 GPTQModel 的 AGPL Swordfish 可选内核）。

依赖冲突风险（与 ADR-0002 同一类问题）：两者都会拉入 `torch`/`transformers`/`accelerate` 等重量级包，版本可能与本项目 `mqde` 环境（Python 3.11，torch 2.11+cu128，transformers 5.7.0）或 vLLM 专用环境（Python 3.12，transformers >= 5.10）冲突，具体冲突程度需要在批准后实际尝试安装才能确认（本 ADR 起草阶段未安装验证）。

## 3. 方案

### A. 子进程调用（与 ADR-0002 一致的模式）

把 GPTQModel/llm-compressor 装进**独立的 Python 环境**（类似 ADR-0002 给 lm-eval 用的那个），写一个小命令行脚本或用它们自带的 CLI/示例脚本做"量化一个 checkpoint，落盘"，`mqde` 这边只是 `subprocess` 调用 + 读落盘产物（权重 + `quantize_config.json`），不 import 它们的 Python API。

- 优点：零依赖冲突风险，`pyproject.toml` 不改；复用 ADR-0002 已经验证过的"按版本做解析测试、记录工具版本"的工程模式。
- 缺点：GPTQModel/llm-compressor 的输出是**导出格式**（量化后的 checkpoint），不是本项目内部的 `QParams`/trace 数据结构；要做"同配置下并排对比"，需要额外写一个"从它们的导出产物反推出可比较指标（如权重误差、PPL）"的适配层，而不是直接拿到和自研算法同样粒度的逐层 trace。T-0609 原始验收标准("统一 QParams 与 trace")在子进程模式下只能部分满足。

### B. 作为可选依赖 import

`pyproject.toml` 加 `[project.optional-dependencies] external-quant = ["gptqmodel", "llmcompressor"]`，直接 `import gptqmodel` 拿到它的内部数据结构（更细粒度，理论上能对上 T-0609"统一 QParams"的原始要求）。

- 优点：能拿到逐层 trace，满足最初验收标准的字面要求。
- 缺点：依赖冲突风险最高（和 ADR-0002 否决掉的"评测框架直接 import"选项是同一类风险，而且 GPTQModel/llm-compressor 对 torch/transformers 的版本要求目前未知，需要先验证）；GPTQModel 的 Swordfish 可选内核若被间接触发编译，需要单独确认 AGPL 不会被触发。

### C. 只做方案 A 的"对比"部分，放弃"统一数据结构"

子进程量化出一个 checkpoint，只对比最终产出的量化误差指标（PPL、logits KL，复用本项目已有的 `real.check`/PPL/logits probe 能力），不要求拿到它们内部的逐层 trace。T-0609 的验收标准相应降级为"同配置下量化误差指标可并排对比"，不是"统一 QParams 数据结构"。

## 4. 建议

**倾向方案 A + C 的组合，但最终取舍应由负责人决定**（这正是本 ADR 停在 Proposed、不往下做的原因）：

- 先用子进程模式跑通 GPTQModel（Apache-2.0 核心，风险最低，注意不要触发 Swordfish 可选内核），验收标准降级为"量化误差指标并排对比"，不强求统一 trace 数据结构。
- llm-compressor 作为第二个适配器，视是否有实际需要（它和本项目已用的 compressed-tensors 格式更贴近，可能后续更有价值）。
- 不把两者 import 进 `mqde` 或 vLLM 任一现有环境；需要用到时让用户/部署方装进独立环境，通过环境变量指向可执行文件（同 `MQDE_LM_EVAL_BIN` 的既有约定）。

## 5. 需要负责人决定（本 ADR 停在这里，未实现任何代码）

1. 是否批准引入 GPTQModel 和/或 llm-compressor 作为运行时依赖（哪怕只是子进程调用，也需要用户在自己的环境里 `pip install` 它们）？
2. 方案 A（子进程，验收标准降级）还是方案 B（直接 import，更高依赖冲突风险，但满足原始"统一 QParams"要求）？
3. 两个都做，还是只做 GPTQModel？
4. 若选方案 B：谁来验证 GPTQModel/llm-compressor 与本项目现有 torch/transformers 版本的兼容性（需要真实安装尝试，本 ADR 起草阶段未做）？

按 AGENTS.md 的 L3 规则：**本 ADR 写完即停，不实现 `algorithms/external.py`，不改 `pyproject.toml`，等待负责人批复后再继续 T-0609。**
