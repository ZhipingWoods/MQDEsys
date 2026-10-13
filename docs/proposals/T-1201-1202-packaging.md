# 方案：T-1201 Docker 镜像 / T-1202 pip wheel（L2，等待评审，未实现）

日期 2026-10-13 · 作者 Claude（Sonnet 5.5） · 状态 **Proposed** · 触及：`deploy/**`、构建/发布配置（L2）；wheel 的构建依赖（`cibuildwheel`）与镜像基础镜像按 `docs/STRUCTURE.md` 属 L3（新增第三方依赖），**批准本方案之前不要开始实现**

## 1. 现状

- **wheel 的一半已经有了**：`python/pyproject.toml` 已经用 `scikit-build-core` 构建 `mqde` Python 包和 `_core` 扩展（`cmake.source-dir = ".."`，`-DMQDE_WITH_PYTHON=ON`）。**缺的是服务本体**：`mqde`（命令行 + `mqde serve`）是 C++ 可执行文件，不在 wheel 里；`data/`（配方、硬件库、模型家族、兼容矩阵）和 `web/dist`（前端）也不在。没有它们，装了 wheel 也跑不起 `mqde serve`，也没有 `mqde estimate`。
- **没有 `deploy/` 目录**；4090 和 Mac 上都装有 docker，但仓库里没有 Dockerfile。
- 服务有几个必须随发行版带上的运行时事实（都来自本轮真机经验）：
  - GCC ≥ 12 才能编（系统 g++ 11 编不过 `std::async`）；C++20（`std::expected`）。
  - 作业用的是 Python 执行器 + PyTorch；vLLM / SGLang 是**独立的 venv**（各自要特定的 torch 版本，装进同一环境会冲突），llama.cpp 是单独的二进制。`scripts/setup_backend.py` + 前端"硬件与后端"页已经负责把它们装到 `$MQDE_ENVS_DIR`。**镜像不应该把它们打进去**，应该让用户在容器里点按钮装（或挂载一个持久卷），否则镜像会 >15 GB。

## 2. T-1201 Docker

| 项 | 建议 |
|---|---|
| 镜像 | `Dockerfile.cpu`（`python:3.11-slim` 多阶段：构建阶段 conda-forge 工具链编译，运行阶段只带 `mqde`、`data/`、`web/dist`、Python 依赖）；`Dockerfile.cuda`（基础镜像 `nvidia/cuda:*-runtime`） |
| 内容 | `mqde` 二进制、`data/`、`web/dist`、`python/mqde`、CPU 版 torch（cpu 镜像）。**不含** vLLM / SGLang / llama.cpp / 任何模型权重 |
| 运行 | 非 root；`--allow` 的目录与 `$MQDE_ENVS_DIR` 做成卷；默认监听 `0.0.0.0` 在容器里是必须的，所以**必须同时有 ADR-0005 D2 的 token**（否则容器端口一映射就是未认证的服务，并且受 ADR-0005 第 2.1 节的跨站问题影响）——这是本方案对 ADR-0005 的硬依赖 |
| 体积 | 任务书"cpu 镜像 < 3 GB"：CPU 版 torch + transformers 约 1.5–2 GB，可行但没量过 |
| 验证 | CI 里 `docker run` 起容器，`curl /api/v1/health`，再 `POST /api/v1/estimate` 对一个小模型目录 |

## 3. T-1202 wheel

1. **把 C++ 服务打进 wheel**：`scikit-build-core` 已经在跑 CMake，让它把 `mqde` 可执行文件、`data/`、`web/dist` 作为包数据装进 `mqde/_bin`、`mqde/data`、`mqde/web`，并加一个 console script `mqde` 去 `exec` 这个二进制（`MQDE_DATA_DIR` 指向包内的 `data`）。**不新增依赖**。
2. **manylinux**：`cibuildwheel` 的 `manylinux_2_28` 镜像带 `gcc-toolset-12+`，可编 C++20；macOS arm64 在 `macos-14` 上。`cibuildwheel` 是**新的构建期依赖，按 STRUCTURE 是 L3，需要先有 ADR**。
3. **torch 不进依赖**：保持现在的 `[quant]` extra（`torch>=2.4`），否则 wheel 会被迫选一个 CUDA 版本。
4. aarch64 Linux 目前**没有任何机器可以验证**（没有 ARM 服务器，QEMU 编译 C++20 很慢）；建议先只发 x86_64 Linux + macOS arm64。

## 4. 我能先验证的部分

不需要批准、不改仓库任何受控路径，我在 4090 的干净环境里试：`pip wheel python/` 能不能出 wheel，装进干净 venv 后 `import mqde._core` 是否成功——这只验证"现有的一半"，结果写在下面"实测"一节（见本文件末尾，若有）。

## 5. 需要负责人决定

1. 同意第 2 节的镜像内容边界（**不带** vLLM/SGLang/llama.cpp）吗？
2. 批准 `cibuildwheel` 作为构建期依赖（需另写 ADR）吗？还是先只发源码包 + conda 配方？
3. aarch64 先不做可以吗？
4. 确认 Docker 方案依赖 ADR-0005 的 token（先批 ADR-0005，再做镜像）。
