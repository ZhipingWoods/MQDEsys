# Installation

MQDEsys consists of three separately built components: the C++ core (`mqde` CLI, server and the `mqde._core` Python extension), the Python executor, and the web UI. The system is distributed as source.

```{note}
The commands below have been verified on macOS (Apple M5, Homebrew dependencies, Python 3.12) and on Linux (conda dependencies). `<REPO>` and `<MODEL_DIR>` are placeholders for the source checkout and a local model directory.
```

## 1. Prepare the local toolchain

MQDEsys uses the tools installed on the target device. It does not require the developer's conda environment and does not include model weights. Install a C++23-capable compiler, CMake 3.25+, Ninja, Python 3.10+, Node.js 18+ and the native libraries listed below with the package manager for your operating system.

The repository includes a one-command setup script for Linux, macOS with Homebrew, and an active conda environment:

```bash
bash scripts/dev_setup.sh
```

Set `MQDE_INSTALL_SYSTEM_DEPS=0` when the native libraries are already installed. The script creates `.venv` inside the checkout, installs the Python extras, builds the C++ core and Web frontend, and starts the service. Use `bash scripts/dev_setup.sh --no-start` to stop after building.

## 2. Build the C++ core

Requirements: a C++23 compiler (Apple clang or GCC 12+), CMake 3.25+, Ninja, and nlohmann_json, yaml-cpp, fmt, CLI11 and cpp-httplib. The Python extension additionally requires Python 3.10+ and pybind11; unit tests require GoogleTest. The build enables `-Wall -Wextra -Werror`.

Two CMake presets are provided: `dev` (RelWithDebInfo, tests and Python extension, dependencies resolved from the local toolchain or `$CONDA_PREFIX`) and `asan` (Debug with sanitizers).

```bash
cd <REPO>
cmake --preset dev
cmake --build --preset dev -j 8
ctest --preset dev
```

With Homebrew (macOS) or system packages, dependencies can be specified directly:

```bash
cmake -S . -B build/mac -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DMQDE_WITH_PYTHON=ON \
  -DPython_EXECUTABLE=<REPO>/build/venv-mac/bin/python \
  -Dpybind11_DIR=/opt/homebrew/share/cmake/pybind11
cmake --build build/mac -j 8
ctest --test-dir build/mac
```

CMake options: `MQDE_BUILD_TESTS` (default ON) and `MQDE_WITH_PYTHON` (default OFF; when enabled, `mqde._core` is built into `python/mqde/`). Verify the build:

```bash
build/mac/bin/mqde --version      # mqde 0.1.0
build/mac/bin/mqde probe          # local CPU / GPU information
```

## 3. Python environment

The executor requires Python 3.10+, `numpy` and `pyyaml`. The quantization pipeline requires `torch>=2.4`, `transformers>=4.44`, `safetensors` and `pyarrow` (the `quant` extra in `python/pyproject.toml`); GGUF export requires `gguf` (the `export` extra). Development tools are `pytest`, `ruff` and `mypy` (the `dev` extra).

```bash
python3 -m venv build/venv && source build/venv/bin/activate
pip install numpy pyyaml torch transformers safetensors pyarrow gguf pytest
export PYTHONPATH=<REPO>/python
python -c "from mqde import core; print(core.__name__)"
```

The system runs from the source tree via `PYTHONPATH=python`. It can also be installed as a package using the scikit-build-core backend defined in `python/pyproject.toml`.

External tools are optional and only required by their corresponding stages:

| Stage | Tool | Resolution |
|---|---|---|
| GGUF export | llama.cpp source directory (provides `convert_hf_to_gguf.py`) | `MQDE_LLAMACPP_SRC` |
| llama.cpp perplexity / benchmark | `llama-perplexity`, `llama-bench` | `MQDE_LLAMACPP_BIN`, otherwise `build/llamacpp/bin`, `build/llamacpp-metal/bin` |
| llama.cpp serving | `llama-server` | `MQDE_LLAMA_SERVER_BIN`, otherwise `build/llamacpp-server/bin/llama-server` |
| vLLM serving | `vllm` (CUDA GPU required) | `MQDE_VLLM_BIN`, otherwise `vllm` on `PATH`, otherwise `build/vllm/bin/vllm` |
| SGLang serving | `sglang` (CUDA GPU required) | `MQDE_SGLANG_PYTHON`, otherwise the managed `$MQDE_ENVS_DIR/sglang` environment |
| Task accuracy | `lm-eval` (separate environment with `transformers`) | `MQDE_LM_EVAL_BIN`, otherwise `lm-eval` on `PATH` |

When a binary is not found, the stage is marked `skipped` and the reason is recorded in the report; the job still completes.

## 4. Web UI

```bash
cd <REPO>/web
npm install
npm run build        # tsc -b && vite build, output in web/dist
```

## 5. Start the server

```bash
build/mac/bin/mqde serve --repo <REPO> --runs <REPO>/runs --web <REPO>/web/dist \
  --python <REPO>/build/venv/bin/python --allow <MODEL_PARENT_DIR> --port 18080
```

| Option | Description |
|---|---|
| `--repo` (required) | Repository root; the server locates `python/` and `data/` there |
| `--runs` | Job directory, one job per `<runs>/<job-id>/` |
| `--web` | Built frontend directory (`web/dist`) |
| `--python` | Python executable used to start the executor |
| `--gpus` | GPU pool for jobs, for example `2,3`; LLM jobs with `deploy.device: cpu` do not occupy the pool |
| `--allow` | Allowed model and dataset roots (repeatable); requests outside these roots return HTTP 403 |
| `--host`, `--port` | The server listens on loopback by default; use an SSH tunnel for remote access (`ssh -N -L 18080:127.0.0.1:18080 <host>`) |

The server does not enable authentication and binds to loopback only. Open `http://localhost:18080` to access the workspace. Without `--gpus`, the GPU pool is empty and GPU-requiring jobs return HTTP 503; on machines without a GPU, submit LLM jobs with `deploy.device: cpu` (VLA, WM and EWM jobs require a GPU pool).
