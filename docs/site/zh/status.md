# 状态与路线图

MQDEsys 是一个能用的纵向切片，不是成品。下面的表把任务分成 **DONE**（已实现并通过验收）、**PARTIAL**（可用但有明确缺口）和 **TODO**（计划中，目前没有可用的东西）。任务清单里的里程碑 M0-M2 对预估器和界面基本完成；M3（最小闭环）部分完成；M4-M7（多后端全配方、误差归因、多硬件、带 pip 和 Docker 的 1.0 发布）尚未完成。

最重要的缺口概述：

- **不可用**：SGLang 与 TensorRT-LLM 部署、lm-eval / evalscope 精度评测（依赖决策待定）、预估器反馈闭环、Ascend/ROCm/Metal 硬件插件、`--allow` 之外的认证与路径沙箱、CI、Docker 镜像、pip wheel。
- **部分实现**：vLLM 与 llama.cpp 部署（仅 Python；没有 C++ 后端接口）、压测与边界搜索（边界搜索尚未在真实 vLLM 上运行）、SmoothQuant 评估位点、FP8 per-block、W2 配合 GPTQ 属于实验、把模型下载作为作业阶段、HTML 报告（`python/mqde/report/html.py` 中有生成器，但还不属于任务验收的一部分）。

```{include} _status_tables.md
```
