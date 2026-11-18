# 系统简介

## 定位

MQDEsys（Model Quantization & Deployment Evaluation System）是面向大模型量化与部署评估的一体化系统。系统以模型、量化配方、评测数据集和目标硬件为输入，完成资源预估、量化、数据集评测、真实 checkpoint 导出、推理部署与压力测试，并将全部过程与指标记录在同一个作业中。

系统的核心目标不是生成一个更小的权重文件，而是回答部署决策所关心的问题：量化后的模型精度是否满足要求、能否装入目标设备、推理性能如何，以及精度损失出现在哪些层。

## 核心流程

1. **资源预估**：在未加载权重的情况下，依据 `config.json` 与 safetensors 头部计算权重、KV Cache 或线性注意力状态、激活、运行时开销、单并发与多并发峰值显存、延迟下界，并生成“配方 × 硬件”的可部署矩阵。
2. **量化与评估**：执行基线评估、带逐层误差追踪的伪量化，以及多个评估位点上的困惑度或数据集指标。
3. **导出与校验**：生成 GPTQ、AWQ、compressed-tensors 或 GGUF checkpoint，并校验导出的整数码与零点同伪量化阶段完全一致。
4. **部署与压测**：以 llama.cpp、vLLM 或 SGLang 启动服务，采集 TTFT、TPOT、吞吐与部署后精度。
5. **可视化**：在 Web 工作台中查看阶段流水线、实时日志、逐层误差热力图、最差层、敏感度分析与配方对比。

## 支持范围

| 维度 | 支持内容 |
|---|---|
| 模型领域 | LLM、VLM、VLA、世界模型（WM）与事件世界模型（EWM） |
| 量化配方 | BF16、W8A16、FP8、W4A16、W4A8、W2A16、W2A8 |
| 量化算法 | RTN、GPTQ、HQQ、AWQ、SmoothQuant 及组合算法 |
| 导出格式 | GPTQ、AWQ、compressed-tensors、GGUF |
| 推理后端 | llama.cpp、vLLM、SGLang |
| 评测数据 | 离线 PPL、QA、MCQ，以及 EvalScope 数据集目录 |
| 硬件适配 | NVIDIA（Jetson Orin、Jetson Thor、DGX Spark、RTX、A100、H100）、AMD、Intel、昇腾、沐曦 |

模型架构通过 `data/model_families/` 中的声明式描述接入。凡能由 Hugging Face transformers 作为因果语言模型加载的架构，均可通过通用解码器适配器运行；结构特殊的模型由专用适配器处理。

## 系统组成

- **C++ 控制面**（`cpp/`）：模型解析、量化数值规范实现、资源预估、适配评分、硬件探针、REST/SSE 服务与命令行工具。
- **Python 执行器**（`python/mqde/`）：PyTorch 数值实现、量化算法、模型适配器、评测、导出器、后端适配器与作业流水线。PyTorch 数值与 C++ 核心逐位一致，由 golden 测试保证。
- **Web 工作台**（`web/`）：React、TypeScript 与 ECharts 实现，由 C++ 服务托管。
- **数据层**（`data/`）：配方、硬件画像、后端画像、模型架构模板与评分常数，均以 YAML 描述，新增条目无需修改代码。

## 作业模式

系统提供三种作业模式，彼此解耦：

- **全流程作业**：预估、量化、评测、导出、部署与压测依次执行。
- **量化作业**：仅执行量化与导出，输出可复用的 checkpoint。
- **评测作业**：直接加载已有的量化模型或 checkpoint，按所选数据集执行评测，无需重新量化。

各模式共享同一套作业记录、事件流、报告结构与 Web 界面。

详细的架构说明见[架构](../advanced/architecture.md)。
