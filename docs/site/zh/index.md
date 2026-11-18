# MQDEsys

**MQDEsys**（Model Quantization & Deployment Evaluation System，模型量化部署评估系统）用于评估模型在指定量化配方与目标硬件上的部署效果。系统以模型、量化配方（W8A16、FP8、W4A16、W4A8、W2A16、W2A8）、评测数据和目标硬件为输入，完成资源预估、量化、数据集评测、真实低比特 checkpoint 导出、推理部署与压力测试，并在 Web 界面中呈现完整过程，直至定位精度损失最大的层。

```{note}
本文档描述 MQDEsys 当前发布版本的系统架构、接口和使用方式。示例中的配置、模型和评测数据用于说明典型工作流；实际部署时请替换为目标模型、数据集和硬件。
```

## 界面预览

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 作业与执行
:img-top: ../_shared/img/ui-jobs.png
全流程、仅量化、仅评测、部署评测统一管理；阶段、进度、模式一眼可见。
:::

:::{grid-item-card} 作业详情
:img-top: ../_shared/img/ui-job-llm.png
进度环、阶段流水线、实时日志、指标和逐配方误差分析。
:::

:::{grid-item-card} 硬件与后端
:img-top: ../_shared/img/ui-hardware.png
后端环境、目标设备探针、算力折线图和分组硬件画像。
:::

:::{grid-item-card} 新建评测作业
:img-top: ../_shared/img/ui-new-eval.png
checkpoint、数据集、运行参数三段式；高级选项折叠。
:::
::::

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} 快速开始
:link: get_started/quickstart
:link-type: doc
预估模型、运行一次量化作业、在界面中查看结果。
:::

:::{grid-item-card} 预估与适配
:link: user_guides/estimate
:link-type: doc
`mqde estimate` 与 `mqde fit`：不加载权重即可得到显存、KV Cache、并发和可部署性。
:::

:::{grid-item-card} 量化
:link: user_guides/quantization
:link-type: doc
RTN、GPTQ、AWQ、SmoothQuant、组合算法、FP8、混合精度、KV Cache 量化。
:::

:::{grid-item-card} 导出与校验
:link: user_guides/export
:link-type: doc
GPTQ、AWQ、compressed-tensors、GGUF、Marlin 打包，以及伪量化/真实量化一致性检查。
:::

:::{grid-item-card} 部署与压测
:link: user_guides/deploy_benchmark
:link-type: doc
llama.cpp（CPU 与 Metal）、vLLM、压测、部署后 PPL、OOM 边界搜索。
:::

:::{grid-item-card} Web 界面与 REST API
:link: user_guides/web_ui
:link-type: doc
逐页介绍界面，以及背后的 HTTP 接口。
:::

:::{grid-item-card} 数据集评测
:link: user_guides/evaluation
:link-type: doc
离线 PPL / QA / MCQ、EvalScope 144 个数据集目录、指标与报告。
:::
::::

```{toctree}
:maxdepth: 2
:caption: 入门

get_started/introduction
get_started/installation
get_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: 用户指南

user_guides/estimate
user_guides/quantization
user_guides/evaluation
user_guides/job_spec
user_guides/export
user_guides/deploy_benchmark
user_guides/web_ui
user_guides/rest_api
```

```{toctree}
:maxdepth: 2
:caption: 进阶

advanced/architecture
advanced/extending
advanced/testing
```

```{toctree}
:maxdepth: 1
:caption: 参考

faq
```
