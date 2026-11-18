# 快速开始

本节介绍从资源预估到在 Web 界面中查看量化结果的完整流程。开始前需完成[安装](installation.md)，准备一个本地 Hugging Face 模型目录 `<MODEL_DIR>`（示例使用 Qwen2.5-0.5B-Instruct），并设置 `PYTHONPATH=<REPO>/python`。

## 1. 资源预估

将命令行工具加入 `PATH`（构建目录取决于 CMake 配置）：

```bash
export PATH=<REPO>/build/mac/bin:$PATH
```

```bash
mqde model inspect <MODEL_DIR>
mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --hw nvidia/rtx4090 --ctx 4096 --conc 1,8,64
```

`mqde estimate` 针对每个配方输出权重字节数（含量化元数据）、KV Cache、激活、运行时开销、相对设备容量的峰值、最大并发与延迟下界。该过程仅读取模型配置与张量头部，不加载权重。

以 Qwen2.5-0.5B 同形状模型（24 层、hidden 896、14/2 头、bf16、4096 上下文、RTX 4090 画像）为例，W4A16 的峰值显存为 1.30 GiB，BF16 为 1.82 GiB。各指标定义见[资源预估](../user_guides/estimate.md)。

## 2. 运行量化作业

作业由 Python 流水线执行，也可通过 Web 界面“新建作业”中的 LLM 入口提交。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --export awq --eval-windows 40 --job-dir runs/quickstart
```

也可使用作业配置文件（模板为 `examples/jobs/qwen25_05b_wikitext.yaml`，需将 `model.path` 改为实际模型目录）：

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job examples/jobs/qwen25_05b_wikitext.yaml
```

作业将加载模型，测量未量化模型在 wikitext-2 上的困惑度（PPL），逐块量化并记录逐层误差，再以伪量化权重测量 PPL；指定 `--export awq` 后还会写出 AWQ checkpoint，读回并以真实权重复测。结果写入 `<job-dir>/report.json`，实时事件流写入 `<job-dir>/events.jsonl`。

作为参考，Qwen2.5-0.5B-Instruct 在 wikitext-2、40 个 512-token 窗口上的结果为：未量化 PPL 18.10；W4A16 + GPTQ（INT4，group size 128）在 CPU 上为 20.41，在 CUDA 上为 20.66。GPTQ 结果与 BLAS 线程数相关，作业会记录 `env.torch_threads`，仅在线程数相同时比较不同运行。

## 3. 在 Web 界面中查看结果

启动服务（见[安装](installation.md)），打开“作业与执行”页面并选择对应作业。以 `--job-dir` 指向服务 `--runs` 目录下启动的命令行作业同样会出现在列表中。

```{figure} ../../_shared/img/jobs.png
:alt: Web 界面的作业列表
:width: 100%

作业列表：每行显示状态、当前阶段、配方、设备与耗时。
```

作业详情页依次展示阶段流水线、实时日志、各评估位点的困惑度、配方汇总、导出与一致性检查结果，以及误差最大的层。完整说明见[Web 界面导览](../user_guides/web_ui.md)。

## 4. 延伸阅读

- 配方与算法选择：[量化](../user_guides/quantization.md)
- 作业字段与命令行参数：[作业配置参考](../user_guides/job_spec.md)
- 部署与压测：[部署与压测](../user_guides/deploy_benchmark.md)
- 数据集评测：[数据集评测](../user_guides/evaluation.md)
