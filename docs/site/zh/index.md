# MQDEsys

**MQDEsys**（Model Quantization & Deployment Evaluation System，模型量化部署评估系统）要回答的是一个完整的问题：*用这个量化配方量化这个模型，部署到这块硬件上，结果到底怎么样？* 你指定模型、量化配方（W8A16、FP8、W4A16、W4A8、W2A16、W2A8）、评估数据和目标硬件，系统会预估资源、量化、评估、导出真实的低比特 checkpoint、用推理引擎部署并压测，并在 Web 界面中展示每一步，直到定位出精度损失最大的那一层。

```{note}
MQDEsys 仍在积极开发中。每个页面都会标明功能是**已实现**、**部分实现**还是**计划中**；[状态与路线图](status.md) 列出了全部任务。文档中的数字来自仓库里记录的真实运行（`docs/PROGRESS.md`）或 Web 界面，所用模型几乎都是 Qwen2.5-0.5B-Instruct。
```

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
status
```
