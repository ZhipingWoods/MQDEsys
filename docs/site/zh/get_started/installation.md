# 安装

MQDEsys 由三部分组成，需要分别构建：C++ 核心（`mqde` 命令行、服务，以及 `mqde._core` Python 扩展）、Python 执行器和 Web 界面。目前还没有 pip wheel 和 Docker 镜像（都在计划中，见 [状态](../status.md)），需要从源码目录使用。

```{note}
下面的命令在 macOS（Apple M5，Homebrew 依赖，Python 3.12）上运行过；按仓库开发记录，Linux（conda 依赖）上同样可用。`<REPO>` 和 `<MODEL_DIR>` 是占位符，分别代表你的代码目录和本地模型目录。
```

## 1. 构建 C++ 核心

要求：支持 C++23 的编译器（Apple clang 或 gcc 12+）、CMake 3.25+、Ninja，以及这些库：nlohmann_json、yaml-cpp、fmt、CLI11、cpp-httplib。构建 Python 扩展还需要 Python 3.10+ 和 pybind11；单元测试需要 GoogleTest。构建使用 `-Wall -Wextra -Werror`。

仓库提供两个预设：`dev`（RelWithDebInfo，带测试和 Python 扩展，依赖从 `$CONDA_PREFIX` 查找）和 `asan`（带 sanitizer 的 Debug）：

```bash
cd <REPO>
cmake --preset dev
cmake --build --preset dev -j 8
ctest --preset dev
```

使用 Homebrew（macOS）或系统包时，不用预设，手动指定依赖位置：

```bash
cmake -S . -B build/mac -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DMQDE_WITH_PYTHON=ON \
  -DPython_EXECUTABLE=<REPO>/build/venv-mac/bin/python \
  -Dpybind11_DIR=/opt/homebrew/share/cmake/pybind11
cmake --build build/mac -j 8
ctest --test-dir build/mac
```

CMake 选项：`MQDE_BUILD_TESTS`（默认 ON）和 `MQDE_WITH_PYTHON`（默认 OFF；ON 时把 `mqde._core` 直接构建到 `python/mqde/`）。检查结果：

```bash
build/mac/bin/mqde --version      # mqde 0.1.0
build/mac/bin/mqde probe          # 本机的 CPU / GPU
```

## 2. Python 环境

执行器需要 Python 3.10+、`numpy` 和 `pyyaml`；量化流水线还需要 `torch>=2.4`、`transformers>=4.44`、`safetensors` 和 `pyarrow`（`python/pyproject.toml` 的 `quant` extra），GGUF 导出需要 `gguf`（`export` extra）。开发工具是 `pytest`、`ruff` 和 `mypy`（`dev` extra）。

```bash
python3 -m venv build/venv && source build/venv/bin/activate
pip install numpy pyyaml torch transformers safetensors pyarrow gguf pytest
export PYTHONPATH=<REPO>/python        # 直接使用源码目录中的包
python -c "from mqde import core; print(core.__name__)"
```

仓库的运行方式是 `PYTHONPATH=python`；把它安装成包（`python/pyproject.toml` 使用 scikit-build-core）不在已验证的流程里。

外部工具是可选的，只在对应阶段才需要：

| 阶段 | 工具 | MQDEsys 如何找到它 |
|---|---|---|
| GGUF 导出 | llama.cpp 源码目录（提供 `convert_hf_to_gguf.py`） | `MQDE_LLAMACPP_SRC` |
| llama.cpp 困惑度 / bench | `llama-perplexity`、`llama-bench` | `MQDE_LLAMACPP_BIN`，否则 `build/llamacpp/bin`、`build/llamacpp-metal/bin` |
| llama.cpp 服务 | `llama-server` | `MQDE_LLAMA_SERVER_BIN`，否则 `build/llamacpp-server/bin/llama-server` |
| vLLM 服务 | `vllm`（需要 CUDA GPU） | `MQDE_VLLM_BIN`，否则 `PATH` 中的 `vllm`，否则 `build/vllm/bin/vllm` |

找不到二进制时，对应阶段记为 `skipped` 并在报告里写明原因，不会让作业失败。

## 3. Web 界面

```bash
cd <REPO>/web
npm install
npm run build        # tsc -b && vite build，输出在 web/dist
```

## 4. 启动服务

```bash
build/mac/bin/mqde serve --repo <REPO> --runs <REPO>/runs --web <REPO>/web/dist \
  --python <REPO>/build/venv/bin/python --allow <MODEL_PARENT_DIR> --port 18080
```

| 参数 | 含义 |
|---|---|
| `--repo`（必填） | 仓库根目录；服务在其中找 `python/` 和 `data/` |
| `--runs` | 作业目录（每个作业是 `<runs>/<job-id>/`） |
| `--web` | 已构建的前端（`web/dist`） |
| `--python` | 启动执行器所用的 Python 可执行文件 |
| `--gpus` | 作业使用的 GPU 池，例如 `2,3`。`deploy.device: cpu` 的 LLM 作业不占用 GPU 池 |
| `--allow` | 允许的模型/数据集路径根目录（可重复）；范围之外的请求返回 HTTP 403 |
| `--host`、`--port` | 服务默认只监听回环地址；远程机器请用 SSH 隧道（`ssh -N -L 18080:127.0.0.1:18080 <dev-server>`） |

服务没有认证（计划中，T-1204）：请保持只监听回环地址。

打开 `http://localhost:18080`。不带 `--gpus` 时 GPU 池为空：需要 GPU 的作业会被拒绝并返回 HTTP 503，所以在没有 GPU 的机器上请提交 `deploy.device: cpu` 的 LLM 作业（VLA 作业始终需要 GPU 池）。
