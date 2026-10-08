# 快速开始

本节从“哪个配方装得下？”一路走到在界面里看到一个完成的作业。前提：已按 [安装](installation.md) 构建好，有一个本地 Hugging Face 模型目录 `<MODEL_DIR>`（这里用 Qwen2.5-0.5B-Instruct），并设置了 `PYTHONPATH=<REPO>/python`。

## 1. 不加载权重的预估

先把命令行工具放进 `PATH`（构建目录取决于你怎么配置的 CMake）：

```bash
export PATH=<REPO>/build/mac/bin:$PATH
```

```bash
mqde model inspect <MODEL_DIR>
mqde estimate -m <MODEL_DIR> -r bf16,w4a16,fp8 --hw nvidia/rtx4090 --ctx 4096 --conc 1,8,64
```

`mqde estimate` 会针对每个配方打印权重字节数（含量化元数据）、KV Cache、激活和运行时开销、相对设备容量的峰值、最大并发和延迟下界。对一个与 Qwen2.5-0.5B 同形状的模型（24 层、hidden 896、14/2 个头、bf16、4096 上下文、RTX 4090 画像），W4A16 的峰值为 1.30 GiB，BF16 为 1.82 GiB。每个数字的含义见 [预估指南](../user_guides/estimate.md)。

## 2. 运行一次 LLM 量化作业

命令行下，作业就是一个 Python 模块；同样的作业也可以在界面（“新建作业”，LLM 标签）里提交。

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=python python -m mqde.pipeline.llm_job \
  --model <MODEL_DIR> --recipes w4a16 --algorithm gptq \
  --export awq --eval-windows 40 --job-dir runs/quickstart
```

等价的作业文件写法（模板是 `examples/jobs/qwen25_05b_wikitext.yaml`，其中的 `model.path` 需要改成你的 `<MODEL_DIR>`）：

```bash
PYTHONPATH=python python -m mqde.pipeline.llm_job examples/jobs/qwen25_05b_wikitext.yaml
```

作业会加载模型，测量未量化模型在 wikitext-2 上的困惑度（PPL），逐块量化并记录逐层误差，用伪量化权重再测一次 PPL；加上 `--export awq` 后还会写出真实的 AWQ checkpoint，读回并用真实权重再测 PPL。结果写在 `<job-dir>/report.json`，实时事件流在 `<job-dir>/events.jsonl`。

作为参照，Qwen2.5-0.5B-Instruct 上记录的一次作业（wikitext-2，40 个 512 token 窗口）：未量化 PPL 18.10；W4A16 + GPTQ（int4，group size 128）在 CPU 上为 20.41，在 CUDA 上为 20.66（GPTQ 在不同设备上不能逐位复现）；用 CUDA 那次导出的 AWQ checkpoint 以真实权重测得 20.656，与伪量化值相差不到 0.01%。

```{warning}
GPTQ 的结果与 BLAS 线程数略有关系（作业会记录 `env.torch_threads`）；只在线程数相同的情况下比较不同运行。
```

## 3. 在界面中查看结果

启动服务（见 [安装](installation.md)），打开**作业与执行**页并点击你的作业。命令行用 `--job-dir` 指向服务 `--runs` 目录之下启动的作业也会出现在列表里。

```{figure} ../../_shared/img/jobs.png
:alt: Web 界面的作业列表
:width: 100%

作业列表（界面文字为中文）。每行显示状态、当前阶段、配方、设备和耗时。
```

作业页依次展示阶段流水线、实时日志、三个评估位点的困惑度、配方汇总表、导出与一致性表以及最差层。[Web 界面导览](../user_guides/web_ui.md) 带截图逐块介绍。

## 4. 下一步

- 选择配方和算法：[量化](../user_guides/quantization.md)。
- 全部作业字段和命令行参数：[作业配置参考](../user_guides/job_spec.md)。
- 部署导出的 checkpoint 并压测：[部署与压测](../user_guides/deploy_benchmark.md)。
