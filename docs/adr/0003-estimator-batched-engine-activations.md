# ADR-0003: 预估器为批处理推理引擎（vLLM）使用“按批”而不是“按序列”的激活内存

| 字段 | 内容 |
|------|------|
| 状态 | **Approved**（2026-10-09 负责人批准；0.5B 与 7B 的验证均通过，见第 7 节） |
| 日期 | 2026-10-08 |
| 作者 | Claude（Sonnet 5.5） |
| 审核人 | 负责人（zhipingwu） |
| 依赖 | 无 |
| 影响任务 | T-0209（预估器与校准库）、T-0905（反馈闭环）、T-0211（可部署矩阵）；PROGRESS.md D-9 |
| 级别 | **L2**：公共头文件 `cpp/include/**` |

---

## 1. 问题

预估器对 vLLM 给出的最大并发明显偏低，并且上下文越长偏得越多。同一模型、同一设置下，对照 vLLM 自己在启动日志里报告的容量：

| 模型 / 检查点 | 上下文 | 预估并发 | vLLM 实测并发 | 预估 / 实测 | 权重：预估 / 实测 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B AWQ，利用率 0.3，RTX 4090 | 1024 | 76 | 547.03 | 0.139 | 0.978 |
| 同上 | 32768 | 2 | 17.09 | 0.117 | 0.978 |

权重估得准（0.978），并发偏低 7 到 8.5 倍。（预估值取自 `core.estimate(...)` 的 `max_concurrency`，实测值取自 vLLM 日志里的 `Maximum concurrency for N tokens per request`；两者的复现命令见 `internal/PROGRESS.md` 2026-10-08 日志。）

## 2. 原因

`cpp/src/estimator/estimator.cpp` 第 208 到 211 行：

```
per_seq = (kv_seq + max(act_vision, act_lang, act_action)) * allocator_factor
budget  = capacity - runtime_overhead - weights * allocator_factor
max_concurrency = floor(budget / per_seq)
```

`act_lang` 随上下文增长（第 184 行附近，按 `ctx` 线性）。这个公式对应 HF eager 的逐序列前向：每多一个并发序列，就多一份完整的激活。

vLLM 不这样工作：它把所有序列的 token 拼成一个批，一步最多处理 `max_num_batched_tokens` 个 token（默认 8192，chunked prefill 打开时），激活内存由这个上限决定，**不随并发数增长**。因此 vLLM 里并发的瓶颈是 KV 缓存，而不是激活。上下文变长时 `act_lang` 变大，旧公式因此偏得更多，与上表一致。

### 2.1 手算的初步证据（不是预估器输出）

按预估器的公式在 Python 里手算 Qwen2.5-0.5B（hidden 896、intermediate 4864、24 层、vocab 151936、KV 每 token 12288 字节）：

| 上下文 | 每序列 KV | 每序列激活 | 激活占每序列开销 | 现公式并发 | 激活只计一次（按 `max_batched_tokens` = 8192 封顶）的并发 | vLLM 实测 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 12.0 MiB | 73.8 MiB | 86% | 76 | 540 | 547 |
| 32768 | 384.0 MiB | 2344.6 MiB | 86% | 2 | 15 | 17.09 |

两点：激活占了每序列开销的 86%，所以并发几乎完全由它决定；而激活里有 59% 来自 `(n_layers + 1) × hidden` 这一项（估算“保留每一层隐状态输出”，见估算器第 184 到 188 行），vLLM 并不保留这些。所以 `batched` 模式既要去掉“乘并发数”，也要去掉这一项，不能只改乘数。
去掉后手算值与实测的比约为 0.99 和 0.88，**没有任何拟合系数**。这只是一个模型上的纸面计算，不是修改后的预估器的输出，不能当作验证；它的作用是说明方案值得做，验证见第 4 节。

## 3. 方案

给后端画像加两个字段（`BackendProfile`，公共头文件，所以是 L2）：

```yaml
# data/backends/vllm.yaml
activation_model: batched          # per_seq（默认，HF eager）| batched
max_batched_tokens: 8192           # 仅 batched 时使用
```

估算规则：

- `per_seq`：与现在完全相同（默认值，所有现有后端行为不变）。
- `batched`：激活内存按 `min(ctx × 并发, max_batched_tokens)` 个 token 计一次，不再乘并发数，也不再计“每层隐状态保留”那一项；`per_seq` 项只剩 KV（含线性注意力状态）：
  `per_seq = kv_seq × allocator_factor`，`activation = act(min(ctx × concurrency, max_batched_tokens))`，从 `budget` 里一次性扣除。

对头文件的改动只有 `BackendProfile` 增加两个成员，其余接口不变，`load_backend` 读两个新键，缺省时取默认值，所以旧的画像文件不需要修改。

## 4. 验证（批准后按此验收）

1. 已有的 100 个 C++ 测试不变（默认 `per_seq`）。
2. 新增估算器单元测试：同一模型在 `per_seq` 与 `batched` 下，权重和 KV 项相同，`max_concurrency` 只在 `batched` 下变大，且随 `max_batched_tokens` 单调。
3. **回归对照真实数据**：用本 ADR 第 1 节的两个点作为验收，`batched` 下的预估并发相对实测应落在 [0.5, 1.5]（现在是 0.12 到 0.14）。不达标则方案不成立，需要重新考虑，而不是调系数去凑。
4. 不同模型规模各取一点验证（至少再加一个 ≥ 7B 的检查点，需要 GPU 与检查点；0.5B 的一个点不足以证明公式在大模型上成立）。
5. 边界搜索（`eval/boundary.py`）里的 `compare_with_estimator` 把每次真实测量与预估并排写进报告。

## 5. 风险与未知

- **只有一个模型的数据**。公式的方向（激活不随并发增长）是 vLLM 的已知行为，但“偏低倍数”是否在大模型上相同，我没有数据。第 4 条的第 4 点是必须项，不是可选项。
- 并发的真正上限还受 `max_num_seqs`（vLLM 默认 1024）限制，预估器目前不建模它；本 ADR 不处理，只在报告里注明。
- 上表的“实测并发”是 vLLM 对 KV 池的容量计算，而不是“并发 N 个请求一定都能同时跑”：并发 1024、上下文 1024 的真实压测全部成功，但 vLLM 对超出 KV 容量的请求是排队而不是失败（见 PROGRESS 2026-10-08 边界搜索），所以这里的“容量”是同时驻留的上限，不是失败边界。
- SGLang、llama.cpp 等其他批处理后端要不要同样标 `batched`：需要各自的实测，本 ADR 只处理 vLLM。

## 6. 需要负责人决定

1. 是否批准在 `BackendProfile` 增加 `activation_model` 与 `max_batched_tokens`（L2 头文件改动）？
2. 验收阈值 [0.5, 1.5] 是否合适，还是要更严？
3. 是否同意把“≥ 7B 检查点上的验证”作为合并前的硬条件（需要先准备检查点与空闲 GPU）？

## 7. 实施记录（2026-10-09）

负责人批准了方案，并允许使用 GPU 4 到 7 做 ≥ 7B 的验证。验收阈值与“7B 验证是合并前硬条件”按第 6 节原样采用。

0.5B 的两个点（`estimator_batched.regression_against_what_vllm_reported`，用仓库自己的 `vllm` 画像，没有拟合系数）：

| 上下文 | 预估并发 | vLLM 实测 | 比值 | 旧模型预估 |
|---:|---:|---:|---:|---:|
| 1024 | 501 | 547.03 | 0.92 | 76 |
| 32768 | 16 | 17.09 | 0.94 | 2 |

第 2.1 节纸面手算预测 0.99 和 0.88，实际 0.92 和 0.94：差别来自实现里多算的每序列 logits 行和 `max_batched_tokens` 封顶的细节，方向和量级与手算一致。
破坏性验证：把 `vllm.yaml` 改回 `per_seq`，回归测试变红且预估值恰为 76 和 2，与第 1 节表格一致；让二分查找忽略 KV，两个测试变红。

实现范围：`BackendProfile` 增加 `activation_model`、`max_batched_tokens`（缺省值保持旧行为）；`load_backend` 校验取值；`batched` 下语言模型激活只计一次、不计逐层隐状态保留，视觉/动作阶段仍按序列计；最大并发由二分查找得到。`max_num_seqs` 仍未建模。

### 7.1 ≥ 7B 验证（第 4 节第 4 点，硬条件）

Qwen2.5-7B-Instruct（bf16，约 14.2 GiB 权重），RTX 4090（GPU 5），vLLM 0.30.0，`gpu_memory_utilization` 0.9，`enforce_eager`。预估取自仓库的 `vllm` 画像，没有任何拟合系数；vLLM 一列取自它启动日志里的 `Maximum concurrency for N tokens per request`。

| 上下文 | 预估并发 | vLLM 实测 | 比值 | 旧的按序列模型 |
|---:|---:|---:|---:|---:|
| 1024 | 110 | 115.66 | 0.95 | 16 |
| 8192 | 13 | 14.46 | 0.90 | 2 |
| 32768 | 3 | 3.61 | 0.83 | 0 |

- 三个点都在 [0.5, 1.5] 内。预估值是向下取整的整数，所以在实测值很小时（3 对 3.61）比值显得偏低，这是取整，不是系统偏差。
- 权重：预估 14.19 GiB，vLLM 日志 14.29 GiB（0.7% 偏低）。
- 旧模型在 32768 上给出 0，也就是判定一个序列都放不下，而 vLLM 实际能放 3.6 个；这正是问题的最坏表现。
- 数据点：0.5B 两个、7B 三个，两种规模，都不是为拟合而选的。只验证了 vLLM；SGLang 等其他批处理后端没有数据，仍按第 5 节不处理。
- 原始数据：`/cache/wuzhiping/adr3_7b/`（4090），复现脚本思路见本节：对每个上下文起 vLLM、读日志、与 `core.estimate(model, "bf16", "nvidia/rtx4090", "vllm", context_tokens=N, gpu_memory_utilization=0.9)` 比较。
