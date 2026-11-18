# 安装

MQDEsys 由三部分组成，分别构建：C++ 核心（`mqde` 命令行、服务及 `mqde._core` Python 扩展）、Python 执行器与 Web 界面。系统当前以源码方式分发。

```{note}
以下命令在 macOS（Apple M5，Homebrew 依赖，Python 3.12）与 Linux（conda 依赖）环境中验证。`<REPO>` 与 `<MODEL_DIR>` 为占位符，分别表示代码目录与本地模型目录。
```

## 1. 准备本机工具链

MQDEsys 使用目标设备上已经安装的工具，不要求用户复用开发机的 conda 环境，也不包含模型权重。在目标系统中准备支持 C++23 的编译器、CMake 3.25+、Ninja、Python 3.10+、Node.js 18+，以及下面列出的原生库，并使用本机包管理器完成安装。

仓库提供适用于 Linux、使用 Homebrew 的 macOS，以及已激活 conda 环境的一键脚本：

```bash
bash scripts/dev_setup.sh
```

如果原生库已经安装，可设置 `MQDE_INSTALL_SYSTEM_DEPS=0`。脚本会在仓库内创建 `.venv`，安装 Python 依赖，构建 C++ 核心和 Web 前端，并启动服务。使用 `bash scripts/dev_setup.sh --no-start` 可只准备并构建项目。

## 2. 构建 C++ 核心

依赖要求：支持 C++23 的编译器（Apple clang 或 GCC 12 及以上）、CMake 3.25 及以上、Ninja，以及 nlohmann_json、yaml-cpp、fmt、CLI11、cpp-httplib。构建 Python 扩展还需要 Python 3.10 及以上与 pybind11；单元测试需要 GoogleTest。编译选项包含 `-Wall -Wextra -Werror`。

仓库提供两个 CMake 预设：`dev`（RelWithDebInfo，含测试与 Python 扩展，依赖从本机工具链或 `$CONDA_PREFIX` 查找）和 `asan`（带 sanitizer 的 Debug）。

```bash
cd <REPO>
cmake --preset dev
cmake --build --preset dev -j 8
ctest --preset dev
```

使用 Homebrew（macOS）或系统包时，可直接指定依赖位置：

```bash
cmake -S . -B build/mac -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DMQDE_WITH_PYTHON=ON \
  -DPython_EXECUTABLE=<REPO>/build/venv-mac/bin/python \
  -Dpybind11_DIR=/opt/homebrew/share/cmake/pybind11
cmake --build build/mac -j 8
ctest --test-dir build/mac
```

CMake 选项：`MQDE_BUILD_TESTS`（默认 ON）、`MQDE_WITH_PYTHON`（默认 OFF；启用后将 `mqde._core` 构建至 `python/mqde/`）。验证安装：

```bash
build/mac/bin/mqde --version      # mqde 0.1.0
build/mac/bin/mqde probe          # 本机 CPU / GPU 信息
```

## 3. Python 环境

执行器需要 Python 3.10 及以上、`numpy` 与 `pyyaml`；量化流水线需要 `torch>=2.4`、`transformers>=4.44`、`safetensors` 与 `pyarrow`（`python/pyproject.toml` 的 `quant` extra）；GGUF 导出需要 `gguf`（`export` extra）。开发工具为 `pytest`、`ruff` 与 `mypy`（`dev` extra）。

```bash
python3 -m venv build/venv && source build/venv/bin/activate
pip install numpy pyyaml torch transformers safetensors pyarrow gguf pytest
export PYTHONPATH=<REPO>/python
python -c "from mqde import core; print(core.__name__)"
```

系统以 `PYTHONPATH=python` 方式运行源码；也可通过 `python/pyproject.toml` 中定义的 scikit-build-core 后端安装为包。

外部工具为可选项，仅在对应阶段使用：

| 阶段 | 工具 | 查找方式 |
|---|---|---|
| GGUF 导出 | llama.cpp 源码目录（提供 `convert_hf_to_gguf.py`） | `MQDE_LLAMACPP_SRC` |
| llama.cpp 困惑度 / 基准测试 | `llama-perplexity`、`llama-bench` | `MQDE_LLAMACPP_BIN`，否则 `build/llamacpp/bin`、`build/llamacpp-metal/bin` |
| llama.cpp 服务 | `llama-server` | `MQDE_LLAMA_SERVER_BIN`，否则 `build/llamacpp-server/bin/llama-server` |
| vLLM 服务 | `vllm`（需要 CUDA GPU） | `MQDE_VLLM_BIN`，否则 `PATH` 中的 `vllm`，否则 `build/vllm/bin/vllm` |
| SGLang 服务 | `sglang`（需要 CUDA GPU） | `MQDE_SGLANG_PYTHON`，否则自动管理的 `$MQDE_ENVS_DIR/sglang` |
| 任务精度评测 | `lm-eval`（独立环境，含 `transformers`） | `MQDE_LM_EVAL_BIN`，否则 `PATH` 中的 `lm-eval` |

未找到对应二进制时，该阶段标记为 `skipped` 并在报告中记录原因，不影响作业整体完成。

## 4. Web 界面

```bash
cd <REPO>/web
npm install
npm run build        # tsc -b && vite build，输出至 web/dist
```

## 5. 启动服务

```bash
build/mac/bin/mqde serve --repo <REPO> --runs <REPO>/runs --web <REPO>/web/dist \
  --python <REPO>/build/venv/bin/python --allow <MODEL_PARENT_DIR> --port 18080
```

| 参数 | 说明 |
|---|---|
| `--repo`（必填） | 仓库根目录；服务在此查找 `python/` 与 `data/` |
| `--runs` | 作业目录，每个作业位于 `<runs>/<job-id>/` |
| `--web` | 已构建的前端目录（`web/dist`） |
| `--python` | 启动执行器所使用的 Python 可执行文件 |
| `--gpus` | 作业使用的 GPU 池，例如 `2,3`；`deploy.device: cpu` 的 LLM 作业不占用 GPU 池 |
| `--allow` | 允许访问的模型与数据集根目录（可重复）；范围之外的请求返回 HTTP 403 |
| `--host`、`--port` | 服务默认仅监听回环地址；远程访问建议使用 SSH 隧道（`ssh -N -L 18080:127.0.0.1:18080 <host>`） |

服务默认不启用认证，仅监听回环地址。访问 `http://localhost:18080` 即可打开工作台。未配置 `--gpus` 时 GPU 池为空，需要 GPU 的作业将返回 HTTP 503；无 GPU 环境可提交 `deploy.device: cpu` 的 LLM 作业（VLA、WM、EWM 作业需要 GPU 池）。
