# 简介

## 它是什么，为什么需要它

量化一个模型并不难；难的是知道量化之后*精度够不够、装不装得下、跑得快不快*。常见工具只给你一个更小的 checkpoint，MQDEsys 把整个闭环补齐：

1. **预估**（C++，不加载权重）：权重、KV Cache 或线性注意力状态、激活、单并发和多并发的峰值显存、延迟下界，以及“配方 × 硬件”的可部署矩阵。只读取 `config.json` 和 safetensors 头部。
2. **量化与评估**（由 C++ 服务调度的 Python 执行器）：基线、带逐层误差追踪的伪量化（fake-quant），以及多个评估位点上的困惑度（PPL）。
3. **导出与校验**：写出真实的 GPTQ、AWQ、compressed-tensors 或 GGUF checkpoint，并检查它保存的整数码与伪量化评估时使用的完全一致。
4. **部署与压测**：用 llama.cpp 或 vLLM 起服务、施加负载，测量 TTFT、TPOT 和吞吐；并计算*部署后*模型的 PPL。
5. **可视化**（Web 界面）：实时阶段与日志、逐层误差热力图、最差层、敏感度分析和配方对比。

目前完整打通的是 LLM 路径（任何通过 transformers 分发注意力的 Hugging Face 因果语言模型；在 Qwen2.5-0.5B-Instruct 上验证过）。另有针对 MiniCPM-RobotManip 的视觉-语言-动作（VLA）路径，只支持伪量化评估。

## 一张图看架构

- **C++20 核心**（`cpp/`）：模型解析、数值（量化算术的唯一规范实现）、预估器、fit 评分、硬件探针、REST/SSE 服务、CLI。
- **Python 执行器**（`python/mqde/`）：PyTorch 版数值镜像（与 C++ 逐位一致，由 golden 测试保证）、算法、评估、导出器、后端适配器、作业流水线。
- **Web**（`web/`）：React + TypeScript + ECharts，由 C++ 服务托管。
- **数据**（`data/`）：配方、硬件画像、后端画像、模型架构模板、评分常数，全部是 YAML；新增条目无需改代码。

详见 [架构](../advanced/architecture.md)。

## 流水线阶段及其状态

每个作业由一系列阶段组成。阶段编号来自设计文档（`docs/ARCHITECTURE.md` 第 5.1 节）；状态列说明仓库里今天实际有什么。

| 阶段 | 名称 | 状态 | 说明 |
|---|---|---|---|
| S00 | validate | 已实现 | 作业与配方校验（Python 流水线；服务端会先校验请求）。 |
| S01 | preflight（预估） | 部分 | 预估器和 `mqde fit` 已有 CLI、REST 和界面页面；作业尚不把它作为前置阶段运行。 |
| S02 | fetch_model | 计划中 | `mqde download` 和模型目录页可以下载权重，但还不是作业阶段（T-0505）。 |
| S03 | inspect | 已实现 | 加载模型，列出块与线性层。 |
| S04 | calib_prepare | 已实现 | 校准窗口取自 wikitext-2 train（LLM）；LeRobot v3 或合成帧（VLA）。 |
| S05 | eval_baseline | 已实现 | 基线 PPL（“pretrain”位点）。 |
| S06 | transform | 部分 | AWQ 与 SmoothQuant 的缩放在 quantize 阶段内完成，不是独立的可缓存阶段。 |
| S07 | quantize | 已实现 | 逐块量化并记录逐层误差；RTN、GPTQ、AWQ、SmoothQuant 及其组合。 |
| S08 | eval_fake | 已实现 | 伪量化权重的 PPL，分别不带与带 KV Cache 量化。 |
| S09 | export | 部分 | GPTQ v1、AWQ GEMM、compressed-tensors、GGUF。SGLang 与 TensorRT-LLM 导出为计划中。 |
| S10 | real_check | 已实现 | 读回 checkpoint，比对整数码与零点，并用真实权重重测 PPL。 |
| S11 | deploy | 部分 | llama.cpp（`llama-server`）与 vLLM，由 Python 实现。C++ 后端接口尚未完成。 |
| S12 | eval_deployed | 部分 | 用部署后模型自己的 prompt log-prob 计算 PPL，仅 vLLM。lm-eval 与 evalscope 适配器为计划中。 |
| S13 | perf_bench | 部分 | Python 压测器（闭环与开环）和 OOM 边界搜索；报告里还没有与预估器并排展示。 |
| S14 | report | 部分 | 每个作业有 `report.json` 和界面页面；独立的 HTML 报告为计划中。 |
| S15 | feedback | 计划中 | 用实测结果自动校准预估器。 |

## 还没有的东西

多机或多卡量化、SGLang 与 TensorRT-LLM 部署、Ascend/ROCm/Metal 硬件插件、认证、Docker 镜像与 pip wheel，以及 wikitext-2 PPL 之外的精度评测，全部是计划中，列在 [状态与路线图](../status.md) 里。请不要依赖它们。
