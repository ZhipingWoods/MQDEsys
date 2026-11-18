# 数据集评测

MQDEsys 将评测分为两个层次，以区分“可在本地直接运行”与“目录中提供”的数据集：

1. **离线 benchmark**：直接加载 checkpoint，由内置评测器评分，无需部署服务；
2. **EvalScope 数据集**：通过 EvalScope 官方 CLI 及其数据集执行，模型需先部署为 OpenAI 兼容服务。

## 离线 benchmark

内置三个数据集，定义于 `data/datasets/benchmarks.yaml`：

| 名称 | 类型 | 指标 | 用途 |
|---|---|---|---|
| `wikitext2_ppl` | PPL | perplexity | 与量化报告使用相同的窗口协议，适合量化前后对比 |
| `smoke_qa` | QA | accuracy | 短问答检查，用于快速验证 checkpoint 可用性 |
| `smoke_mcq` | MCQ | accuracy | 按答案对数概率评分，不依赖生成式判定 |

离线 benchmark 既可用于“仅评测”作业，也可用于全流程作业的评估阶段，不依赖 vLLM、SGLang 或 llama.cpp。

## 原生数据集 runner

框架内置原生数据集 runner，覆盖以下任务：

- GSM8K：生成式作答并抽取数字答案；
- MMLU：多选题对数概率；
- ARC-Challenge：多选题；
- HellaSwag：续写选择；
- Winogrande：指代消解选择。

上述任务通过可选的 `datasets` 包加载 Hugging Face 数据集，并在 `python/mqde/eval/native.py` 中完成归一化与评分，不依赖 lm-eval 或 EvalScope CLI。当 `datasets` 未安装或数据集加载失败时，该 benchmark 在报告中标记为 failed，不影响作业整体执行。

## EvalScope 数据集目录

`data/datasets/evalscope_catalog.yaml` 由 `scripts/gen_evalscope_catalog.py` 基于 EvalScope 支持的 LLM 数据集列表生成，包含 144 个数据集的名称、显示名、类别与标签。该文件为**元数据目录**，实际评测由 EvalScope 执行。

类别覆盖：

- 知识与推理：MMLU、MMLU-Pro、C-Eval、CMMLU、ARC、HellaSwag、GPQA-Diamond；
- 数学：GSM8K、MATH-500、AIME、Minerva-Math；
- 代码：HumanEval、MBPP、LiveCodeBench、BigCodeBench；
- 中文：C-Eval、CMATH、CMMLU、IQuiz；
- 长上下文：LongBench-v2、FRAMES、Needle-in-a-Haystack；
- 指令遵循、Agent、工具调用、问答与真实性等。

## 界面流程

“新建评测作业”页面（`/new/eval`）由三个区块组成：

1. **评测对象**：已量化 checkpoint、量化格式，以及可选的基座模型与 tokenizer；
2. **评测数据集**：依次选择评测套件（快速检查、标准评测或自定义）、类别与数据集，查看指标、样本量、语言、标签与文档后加入本次运行；
3. **可选：部署与压测**：在需要运行 EvalScope 或测量吞吐时展开。

数据集选择采用“套件 → 类别 → 数据集”的逐级下拉结构，已选数据集以紧凑列表呈现，详情卡仅显示当前数据集。

## 作业配置

离线 benchmark 配置位于 `evaluation.benchmarks`：

```yaml
evaluation:
  dataset: wikitext-2-raw-v1:test
  seq_len: 512
  max_windows: 40
  benchmarks: [wikitext2_ppl, smoke_qa]
  limit: 50
```

EvalScope 数据集配置位于部署段：

```yaml
deploy:
  vllm:
    evalscope:
      datasets: [gsm8k, ceval]
      limit: 100
      eval_batch_size: 8
```

## 报告

评测报告按数据集展示以下内容：

- 指标（PPL 或 accuracy）与样本数；
- 运行耗时；
- 按 `subject` 分类的准确率；
- 跳过原因（例如 checkpoint 格式不支持离线加载）；
- 部署后的 EvalScope 指标。

## 新增离线数据集

1. 在 `data/datasets/` 中添加 JSONL 文件；
2. 在 `benchmarks.yaml` 中增加条目，填写 `kind`、`metrics`、`group`、`language` 与 `source`；
3. 如需纳入套件，在 `eval_suites.yaml` 中注册；
4. 运行 `PYTHONPATH=python pytest python/tests -q`。

新增 EvalScope 数据集只需更新 `evalscope_catalog.yaml`（或重新运行生成脚本），无需修改 C++ 或前端代码。
