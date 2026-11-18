# 常见问题与排查

## 安装与启动

**`mqde: command not found`。** 可执行文件位于 `<build-dir>/bin/mqde`；将该目录加入 `PATH`，或使用完整路径调用。

**CMake 找不到 `nlohmann_json`、`yaml-cpp`、`fmt`、`CLI11` 或 `httplib`。** 安装对应依赖（Homebrew、apt 或 conda-forge），并通过 `-DCMAKE_PREFIX_PATH=<prefix>` 指定位置。`dev` 预设从 `$CONDA_PREFIX` 查找依赖，需先激活 conda 环境，或按[安装](get_started/installation.md)手动配置。

**构建因警告失败。** 项目启用 `-Wall -Wextra -Werror`，较新的编译器可能引入新的警告。建议提交问题报告，而非关闭该选项。

**`mqde._core` 导入失败。** 该扩展仅在配置 `-DMQDE_WITH_PYTHON=ON` 且 Python 次版本与构建时一致（扩展文件名包含版本，如 `cpython-312`）时可用。更换 Python 版本后需重新构建。

**服务返回 HTTP 403 `model path is outside allowed roots`。** 通过 `--allow <模型上级目录>`（可重复）启动服务。

**服务返回 503 `server started without a GPU pool`。** 使用 `--gpus <ids>` 启动服务，或提交 `deploy.device: cpu` 的 LLM 作业。VLA、WM、EWM 作业需要 GPU。

**`deploy.device=cpu is only supported for model.type=llm`。** 在作业配置中设置 `model.type: llm`（默认值为 `vla`）。

**服务重启后作业列表为空，或作业始终显示“运行中”。** 服务依据每个作业的 `events.jsonl` 重建状态；随服务进程一同终止的作业会保留运行中状态，需要重新提交。

## 量化结果

**GPTQ 结果在不同运行或不同机器之间不一致。** Cholesky 分解与矩阵乘依赖 BLAS 线程数和设备，GPTQ 的逐层误差反馈会放大差异（Qwen2.5-0.5B 上 W4 约 1% PPL）。作业会记录 `env.torch_threads`，仅在该值相同时比较结果。

**W2 或 W4A8 的 PPL 异常偏高。** W2 属于实验性配方，在小模型上可能出现明显退化，对比页面因此使用对数轴。可结合敏感度分析，对敏感层使用混合精度。

**文档中的数值与实际结果不一致。** 除特别说明外，文档数值均基于 Qwen2.5-0.5B-Instruct、wikitext-2 测试集与 40 个 512-token 窗口。`llama-perplexity` 使用不同的计分协议（仅对每个 chunk 的后半段计分），只能与其自身结果比较。

**导出表中出现 `unsupported`。** 导出器的 `check()` 检测到无法表示的层（例如 AWQ 的 3-bit 权重、带量化激活的 compressed-tensors 整数权重、无法折叠的 AWQ/SmoothQuant 缩放）。具体原因记录在 `report.json` 中对应配方的 `export` 字段下。

## 部署

**阶段显示 `skipped`。** 未找到引擎二进制，或缺少 CUDA 设备（vLLM / SGLang）。请设置 `MQDE_VLLM_BIN`、`MQDE_SGLANG_PYTHON`、`MQDE_LLAMA_SERVER_BIN`、`MQDE_LLAMACPP_BIN`，或按[安装](get_started/installation.md)构建至对应目录。

**vLLM 拒绝 GPTQ checkpoint。** vLLM 0.30.0 不接受非对称 4-bit GPTQ（`bits=4, sym=False`），而 `w4a16` 配方会产生该格式。请改用 AWQ 或 compressed-tensors 导出，或使用对称配方。

**vLLM 启动失败（显存不足或 KV Cache 过小）。** 报告会附带从 vLLM 日志提取的诊断信息。可在 GPU 空闲时提高 `deploy.vllm.gpu_memory_utilization`（默认 0.3）、降低 `max_model_len`，或释放显存。

**GGUF 转换失败：`llama.cpp source with convert_hf_to_gguf.py not found`。** 将 `MQDE_LLAMACPP_SRC` 指向 llama.cpp 源码目录。

**Q4_K 显示“144 层回退”。** 行长不是 256 倍数的张量会按 llama.cpp 规则回退到 Q5_0；对 Qwen2.5-0.5B 属于预期行为。

**预估的最大并发显著低于 vLLM 的结果。** 请选用 `vllm` 后端画像：其按批计算激活内存，在已核对的模型上处于 vLLM 自报数值的 0.83 至 0.95 区间。默认的 `hf_transformers` 画像按序列计算激活，结果为保守下限。

**`mqde fit` 对 VLA 模型给出的速度估计偏高。** 该公式按每步读取一次权重计算；VLA 单步包含多次去噪与视觉编码。应以同硬件上的 `mqde bench` 实测值为准。

## Web 界面

**重新构建后页面仍为旧版本。** 强制刷新浏览器，确认 `web/dist` 新于改动时间，并检查服务启动时 `--web` 指向的目录。

**界面语言。** Web 工作台当前以中文呈现，页面结构见[界面导览](user_guides/web_ui.md)。

**是否可将服务暴露给其他机器？** 不建议。服务默认不启用认证，应保持回环地址绑定，并通过 SSH 隧道访问。
