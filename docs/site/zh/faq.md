# 常见问题与排查

## 安装与启动

**`mqde: command not found`。** 可执行文件是 `<build-dir>/bin/mqde`；把该目录加入 `PATH`，或直接用路径调用。

**CMake 找不到 `nlohmann_json`、`yaml-cpp`、`fmt`、`CLI11` 或 `httplib`。** 安装它们（Homebrew、apt、conda-forge），并传入 `-DCMAKE_PREFIX_PATH=<prefix>`。`dev` 预设在 `$CONDA_PREFIX` 下查找，所以请先激活 conda 环境，或按 [安装](get_started/installation.md) 手动配置。

**构建因为警告失败。** 项目使用 `-Wall -Wextra -Werror`；新编译器可能引入在这里被当作错误的新警告。请反馈问题，而不是去掉这个选项。

**`mqde._core` 导入失败。** 只有用 `-DMQDE_WITH_PYTHON=ON` 配置、且 Python 小版本与你运行的相同（文件名里带有版本，如 `cpython-312`）时，扩展才会被构建到 `python/mqde/`。换了 Python 就要重新构建。

**服务返回 HTTP 403 `model path is outside allowed roots`。** 用 `--allow <模型的上级目录>`（可重复）启动服务。

**服务返回 503 `server started without a GPU pool`。** 用 `--gpus <ids>` 启动，或提交 `deploy.device: cpu` 的 LLM 作业。VLA 作业始终需要 GPU。

**`deploy.device=cpu is only supported for model.type=llm`。** 在作业里设置 `model.type: llm`（默认是 `vla`）。

**服务重启后作业列表为空，或某个作业一直显示“运行中”。** 服务根据每个作业的 `events.jsonl` 重建状态；进程随服务一起死掉的作业会一直显示运行中（已知限制）。

## 量化结果

**GPTQ 结果在不同运行或机器之间不同。** Cholesky 和矩阵乘依赖 BLAS 线程数和设备，GPTQ 的逐层误差反馈会放大差异（Qwen2.5-0.5B 上 W4 约 1% PPL）。作业会记录 `env.torch_threads`；只在该值相同的情况下比较不同运行。记录里的 CPU 运行给出 W4A16 PPL 20.41，CUDA 运行为 20.66。

**W2 或 W4A8 的 PPL 极大。** W2 是实验性的，在小模型上会崩溃（对比页因此使用对数轴）。对敏感度分析指出的层使用混合精度。

**我的数字与文档不一样。** 除非另有说明，文档里的数字都是 Qwen2.5-0.5B-Instruct、wikitext-2 测试集、40 个 512 token 窗口。`llama-perplexity` 的 PPL 使用不同协议（只对每个 chunk 的后半段计分），只能在它自己的表内比较。

**导出表里出现 `unsupported`。** 导出器的 `check()` 发现了无法表示的层（例如 AWQ 的 3 bit 权重、compressed-tensors 整数权重加激活量化、无法折叠的 AWQ/SmoothQuant 缩放）。原因列在 `report.json` 中该配方的 `export` 下。

## 部署

**某个阶段显示 `skipped`。** 找不到引擎二进制，或没有 CUDA 设备（vLLM）。设置 `MQDE_VLLM_BIN`、`MQDE_LLAMA_SERVER_BIN`、`MQDE_LLAMACPP_BIN`，或构建到 [安装](get_started/installation.md) 中列出的 `build/...` 目录。

**vLLM 拒绝 GPTQ checkpoint。** vLLM 0.30.0 拒绝非对称 4 位 GPTQ（`bits=4, sym=False`），而 `w4a16` 配方恰好产生这种格式。请改用 AWQ 或 compressed-tensors 导出。

**vLLM 启动失败（显存不足、KV Cache 太小）。** 报告里带有从 vLLM 日志得到的诊断。GPU 空闲时可以调大 `deploy.vllm.gpu_memory_utilization`（默认 0.3），也可以降低 `max_model_len`，或腾出 GPU。

**GGUF 转换失败：“llama.cpp source with convert_hf_to_gguf.py not found”。** 把 `MQDE_LLAMACPP_SRC` 设为 llama.cpp 源码目录。

**Q4_K 显示 “144 层回退”。** 行长不是 256 倍数的张量会像 `llama-quantize` 一样回退到 Q5_0；这对 Qwen2.5-0.5B 是预期行为。

**预估器的最大并发远低于 vLLM 的结果。** 请使用 `vllm` 后端画像：它按批计激活内存（ADR-0003），在已核对的两个模型上落在 vLLM 自报数值的 0.83 到 0.95 之内。默认的 `hf_transformers` 画像按序列计激活，对批处理引擎只是保守下限。

**`mqde fit` 对 VLA 模型给出的速度高得离谱。** 公式只算每步读一遍权重；而 VLA 一步包含十次去噪和视觉编码。同一硬件上的本机 `mqde bench` 实测会取代公式值。

## Web 界面

**重新构建后页面还是旧的。** 在浏览器里强制刷新，确认 `web/dist` 比你的改动新，并确认服务启动时 `--web` 目录正确。

**界面是中文的。** 英文界面文案尚未提供；[导览](user_guides/web_ui.md) 逐页做了说明。

**能把服务暴露给其他机器吗？** 不安全：没有认证。保持默认的回环绑定，用 SSH 隧道。
