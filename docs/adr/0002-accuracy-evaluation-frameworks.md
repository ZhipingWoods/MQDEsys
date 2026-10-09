# ADR-0002: 精度评测框架（lm-evaluation-harness / evalscope）以子进程方式接入

| 字段 | 内容 |
|------|------|
| 状态 | **Approved**（2026-10-09 负责人批准：采纳方案 A，先做 lm-eval；`pyproject.toml` 不改） |
| 日期 | 2026-10-08 |
| 作者 | Claude（Sonnet 5.5） |
| 审核人 | 负责人（zhipingwu） |
| 依赖 | 无 |
| 影响任务 | T-0801（L3）、T-0802（部署后精度评测，S12）、T-0808（M4 验收） |

---

## 1. 背景

T-0801 要求“MMLU / GSM8K 结果与 lm-eval 官方 CLI 直接运行的结果一致”，并提供 `eval/lm_eval_adapter.py`、`eval/evalscope_adapter.py`、`eval/custom.py`。
引入 lm-eval 或 evalscope 属于“新增第三方依赖”，按 AGENTS.md 是 L3，必须先有 ADR。

现状（2026-10-08）：

- S12 的“部署后精度”已有一个不依赖第三方的指标：用 vLLM 返回的 `prompt_logprobs` 算部署后 PPL（`eval/served_ppl.py`），实测与真实权重 PPL 相差 +0.01% 到 +0.27%。
- PPL 不能替代任务准确率：低比特量化常见的是 PPL 变化很小而推理类任务明显变差。MMLU、GSM8K 这类任务才是验收需要的数字。
- 仓库已通过子模块链接了两个参考实现（`3rdpart/lm-evaluation-harness`、`3rdpart/evalscope`），只读，不能被产品代码 import。

## 2. 约束

1. **依赖冲突**：lm-eval 与 evalscope 会拉入 `transformers`、`datasets`、`torch`、`evaluate` 等。vLLM 0.30 要求 `transformers >= 5.10` 并且必须放在独立的 Python 3.12 环境里；本项目的 `mqde` 环境固定在 Python 3.11。把评测框架装进 `mqde` 环境，等于把另一套 `transformers` 版本约束带进量化主流程。
2. **离线与网络**：开发机不能直连 Hugging Face；MMLU / GSM8K 的数据集需要经过镜像或代理下载，或预先放进缓存。
3. **可复现**：评测结果必须能对应到框架的确切版本、任务配置和样本数。
4. **验收**：“与官方 CLI 一致”最稳妥的做法，是让我们的适配器调用的就是官方 CLI。

## 3. 方案

### A. 以子进程调用官方 CLI（推荐）

适配器只做三件事：拼命令行、运行 `lm-eval`（或 `evalscope eval`）、解析它写出的 JSON 结果。我们的 Python 包**不 import** 评测框架，`pyproject.toml` **不新增依赖**。

- 被测模型是一个已经启动的 OpenAI 兼容服务（vLLM 或 llama-server，S11 已经负责启动和关闭）：`lm-eval --model local-completions --model_args model=<name>,base_url=http://127.0.0.1:<port>/v1/completions,num_concurrent=<n> --tasks mmlu,gsm8k --limit <k> --output_path <dir>`。
- 用户把评测框架装进**自己选的环境**（可以是 vLLM 那个环境，或一个单独的环境），用环境变量 `MQDE_LM_EVAL_BIN` / `MQDE_EVALSCOPE_BIN` 指向可执行文件，与 `MQDE_VLLM_BIN` 同一套约定。
- 找不到可执行文件时，该阶段记录为 `skipped` 并写明原因，不使作业失败。
- 因为运行的就是官方 CLI，“与官方 CLI 直接运行一致”在构造上成立；测试只需验证命令拼装和结果解析（用假的 CLI 脚本），真实运行放在有网络和数据集的机器上做一次验收。

### B. 作为可选依赖 import（不推荐）

在 `pyproject.toml` 加 `eval = ["lm-eval", ...]` 并 `import lm_eval`。能拿到结构化对象，但会把上面的依赖冲突带进 `mqde` 环境，并且 lm-eval 的 Python API 在版本间变化较多。

### C. 只做自有评测（不引入任何框架）

保留 `eval/custom.py`（JSONL：prompt / answer / metric ∈ exact_match / contains / regex / rouge_l）和部署后 PPL。零依赖，但无法满足 T-0801 的 MMLU / GSM8K 一致性验收。

## 4. 建议

**采纳 A，并同时保留 C。**

- C 立即可用，已经不需要批准：`eval/custom.py` 与 `data/datasets/smoke_qa.jsonl` 给每次部署一个便宜的任务级健全性检查。
- A 在批准后实现：`eval/lm_eval_adapter.py`（先做，lm-eval 是 MIT 许可，支持 `local-completions` 与基于对数概率的选择题任务）；evalscope（Apache-2.0）作为第二个适配器，只在有需要时再做。
- 不修改 `pyproject.toml`，不把任何评测框架装进 `mqde` 环境。

## 5. 影响

| 项目 | 影响 |
|------|------|
| 新增依赖 | 无（运行时可选的外部可执行文件） |
| 许可证 | lm-eval MIT、evalscope Apache-2.0；我们不分发它们的代码 |
| 安全 | 子进程只连 `127.0.0.1` 上刚启动的服务；参数用列表而非拼接 shell 字符串 |
| 成本 | MMLU 全量对 0.5B 模型约几十分钟量级；默认 `--limit` 取小值，完整评测需显式开启 |
| 风险 | 数据集需网络或镜像；lm-eval 输出的 JSON 字段随版本变化，适配器按版本做解析测试，并在结果里记录框架版本 |

## 6. 需要负责人决定

1. 是否批准方案 A（子进程调用，不改 `pyproject.toml`）？
2. 是否同意先只做 lm-eval 适配器，evalscope 之后再说？
3. MMLU / GSM8K 的数据集在开发机上的来源（镜像、代理，或手动放进缓存目录）。

批准后的验收：在有数据集的机器上，对同一已部署模型分别用我们的适配器和直接运行官方 CLI（相同 `--limit`、相同种子），两者的 `acc` / `exact_match` 数值逐位一致，结果写入报告并出现在 HTML 报告的 S12 部分。
