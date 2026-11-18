# ADR-0001: 引入 flash-linear-attention 和 causal-conv1d 加速 VLA 推理

| 字段 | 内容 |
|------|------|
| 状态 | **Proposed** |
| 日期 | 2026-09-28 |
| 作者 | Claude（Opus 5） |
| 审核人 | 负责人（zhipingwu） |
| 依赖 | 无 |
| 影响任务 | T-R19（置信度与本地校准）、T-R26（运行时选择） |

---

## 背景

MiniCPM-RobotManip 的语言模型核心是 Qwen3.5（混合 GQA + GatedDeltaNet 线性注意力）。
GatedDeltaNet 需要 `flash-linear-attention`（FLA）和 `causal-conv1d` 提供的 CUDA 内核才能走快路径；
若缺失，`transformers` 回退到 pure-Python 慢路径，日志中会出现：

```
UserWarning: The fast path is not available for this model.
Falling back to the slow path.
```

在 GPU 3 上对 BF16 模型的剖析数据（`/tmp/vla_breakdown.py`，2026-09-28）：

| 阶段 | Wall ms | GPU busy ms | 内核数 |
|------|--------:|------------:|------:|
| vision + LM | 190.5 | 37.6 | 12,359 |
| DiT 动作头 | 40.9 | 10.4 | 1,906 |
| **完整一步** | **236** | **~48** | **~14,000** |

- GPU 利用率仅 ~20%（48/236 ms），剩余 ~80% 为 CPU 内核发射开销。
- 每内核主机侧开销约 12–16 µs（视觉/LM：12.4 µs；动作头：16.0 µs）。
- 带宽屋顶线公式预测 46.9 steps/s，实测 **4.41 steps/s**（10.6 倍差距）。
- **根本原因推断**：~12k 个小内核来源于慢路径展开；FLA/causal-conv1d 快路径会将
  多步 GatedDeltaNet 合并为单次大内核，从而将内核数量和启动开销下降 1–2 个数量级。
  ⚠️ **此推断尚未逐模块验证**；当前只有宏观剖析数据，未在每层单独计时。

---

## 决策

提议：将以下两个包作为**可选运行时依赖**引入 `pyproject.toml`（extra `[vla-fast]`），
并在 `python/mqde/models/minicpm_vla.py` 的加载路径中自动检测并启用：

| 包 | 仓库 | 许可证 | 作用 |
|----|------|-------|------|
| `flash-linear-attention` (FLA) | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT | GatedDeltaNet / RetNet / Mamba2 等线性注意力的 Triton 内核 |
| `causal-conv1d` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | BSD-3-Clause | 线性注意力所需的因果卷积 CUDA 内核 |

引入方式：
- `pyproject.toml` 增加 `[project.optional-dependencies]` 中的 `vla-fast` 组，
  依赖 `flash-linear-attention>=0.3` 和 `causal-conv1d>=1.4`。
- 产品代码仅在 `importlib.util.find_spec("fla")` 成功时才启用快路径，
  否则以 warning 降级运行（现有行为），不抛出异常。
- 不将任何 FLA/causal-conv1d 源码复制到仓库；使用 pip 安装。
- `data/benchmarks/` 中已有的 4.41 steps/s 慢路径基线继续保留，
  安装快路径后重新跑 `mqde bench` 更新基线，二者在数据文件中均记录。

---

## 替代方案

| 方案 | 评估 |
|------|------|
| A. 保持现状，只标注公式为上限 | **已实施**（T-R20 后续，commit 待提交）。不解决根本性能问题，10.6× 差距持续 |
| B. 迁移到 vLLM / SGLang | 解决调度开销，但引入更大依赖（完整推理服务器），超出当前阶段范围 |
| C. 手写 PyTorch eager fused 内核 | 工程成本高，且不如 Triton 内核灵活 |
| **D. 本方案（FLA + causal-conv1d）** | 针对性最强；是 Qwen3.5 官方推荐路径；依赖小且均为宽松许可证 |

---

## 实施步骤（Accepted 后执行）

1. `pyproject.toml`（**L3**）：增加 `vla-fast` optional extra。
2. `python/mqde/models/minicpm_vla.py`（**L1**）：加载时检测 FLA，记录 warning/info。
3. 在 GPU 3 上安装并重跑 `mqde bench`，记录新基线。
4. 重新剖析，**逐模块计时**，验证内核数量变化。
5. 若快路径下步速 ≥15 steps/s（保守估计），将 `speed_is_upper_bound` 标志保留，
   但更新公式注释说明误差已收窄；若 ≥30 steps/s，考虑重新校准公式斜率。
6. 更新 `internal/PROGRESS.md` 和本 ADR 状态为 Accepted/Done。

---

## 影响范围

| 类别 | 影响 |
|------|------|
| 新增依赖 | `flash-linear-attention`、`causal-conv1d`（均为可选，`[vla-fast]` extra） |
| 变更文件 | `pyproject.toml`（L3）、`python/mqde/models/minicpm_vla.py`（L1） |
| 服务端安装 | 需在 conda env `mqde` 中 `pip install "mqde[vla-fast]"` |
| 破坏性变更 | 无；慢路径保留为降级 |
| 数值一致性 | 快路径改变运行速度，不改变权重数值；fake-quant 误差测量不受影响 |
| 许可证 | MIT + BSD-3-Clause，与项目兼容 |

---

## 待决策

- [ ] 负责人批准本 ADR（将状态改为 **Accepted**）
- [ ] 确认 conda env `mqde` 中 CUDA 版本（当前 cu128）与 FLA wheel 的兼容性
- [ ] 确认是否需要同时更新 `data/model_families/minicpm_vla.yaml` 中的 `fast_path_requires` 字段
