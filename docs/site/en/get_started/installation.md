# Installation

MQDEsys has three parts you build separately: the C++ core (`mqde` CLI, server, and the `mqde._core` Python extension), the Python executor, and the web UI. There is no pip wheel or Docker image yet (both planned, see [Status](../status.md)); you work from a source checkout.

```{note}
The commands below were run on macOS (Apple M5, Homebrew dependencies, Python 3.12) and, per the repository's development notes, on Linux (conda dependencies). Paths such as `<REPO>` and `<MODEL_DIR>` are placeholders for your checkout and a local model directory.
```

## 1. Build the C++ core

Requirements: a C++23 compiler (Apple clang or gcc 12+), CMake 3.25+, Ninja, and these libraries: nlohmann_json, yaml-cpp, fmt, CLI11, cpp-httplib. To build the Python extension you also need Python 3.10+ and pybind11. Unit tests need GoogleTest. The build uses `-Wall -Wextra -Werror`.

The repository ships two presets, `dev` (RelWithDebInfo, tests, Python extension; looks for dependencies under `$CONDA_PREFIX`) and `asan` (Debug with sanitizers):

```bash
cd <REPO>
cmake --preset dev
cmake --build --preset dev -j 8
ctest --preset dev
```

With Homebrew (macOS) or system packages, configure without the preset and point CMake at the dependencies:

```bash
cmake -S . -B build/mac -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DMQDE_WITH_PYTHON=ON \
  -DPython_EXECUTABLE=<REPO>/build/venv-mac/bin/python \
  -Dpybind11_DIR=/opt/homebrew/share/cmake/pybind11
cmake --build build/mac -j 8
ctest --test-dir build/mac
```

CMake options: `MQDE_BUILD_TESTS` (default ON) and `MQDE_WITH_PYTHON` (default OFF; ON builds `mqde._core` straight into `python/mqde/`). Check the result:

```bash
build/mac/bin/mqde --version      # mqde 0.1.0
build/mac/bin/mqde probe          # CPUs / GPUs found on this machine
```

## 2. Python environment

The executor needs Python 3.10+ with `numpy` and `pyyaml`; the quantization pipeline additionally needs `torch>=2.4`, `transformers>=4.44`, `safetensors` and `pyarrow` (extra `quant` in `python/pyproject.toml`), and GGUF export needs `gguf` (extra `export`). Development tools are `pytest`, `ruff` and `mypy` (extra `dev`).

```bash
python3 -m venv build/venv && source build/venv/bin/activate
pip install numpy pyyaml torch transformers safetensors pyarrow gguf pytest
export PYTHONPATH=<REPO>/python        # the package is used in place
python -c "from mqde import core; print(core.__name__)"
```

The repository is run with `PYTHONPATH=python`; installing it as a package (`python/pyproject.toml` uses scikit-build-core) is not part of the verified workflow.

External tools are optional and only needed for the matching stage:

| Stage | Tool | How MQDEsys finds it |
|---|---|---|
| GGUF export | a llama.cpp source tree (for `convert_hf_to_gguf.py`) | `MQDE_LLAMACPP_SRC` |
| llama.cpp perplexity / bench | `llama-perplexity`, `llama-bench` | `MQDE_LLAMACPP_BIN`, else `build/llamacpp/bin`, `build/llamacpp-metal/bin` |
| llama.cpp serving | `llama-server` | `MQDE_LLAMA_SERVER_BIN`, else `build/llamacpp-server/bin/llama-server` |
| vLLM serving | `vllm` (needs a CUDA GPU) | `MQDE_VLLM_BIN`, else `vllm` on `PATH`, else `build/vllm/bin/vllm` |

A missing binary turns the stage into `skipped` with the reason in the report; it does not fail the job.

## 3. Web UI

```bash
cd <REPO>/web
npm install
npm run build        # tsc -b && vite build, output in web/dist
```

## 4. Start the server

```bash
build/mac/bin/mqde serve --repo <REPO> --runs <REPO>/runs --web <REPO>/web/dist \
  --python <REPO>/build/venv/bin/python --allow <MODEL_PARENT_DIR> --port 18080
```

| Flag | Meaning |
|---|---|
| `--repo` (required) | repository root; the server finds `python/` and `data/` there |
| `--runs` | job directory (every job is `<runs>/<job-id>/`) |
| `--web` | built front end (`web/dist`) |
| `--python` | Python executable used to launch the executor |
| `--gpus` | GPU pool for jobs, for example `2,3`. LLM jobs with `deploy.device: cpu` do not use the pool |
| `--allow` | allowed root for model and dataset paths (repeatable); requests outside it get HTTP 403 |
| `--host`, `--port` | the server listens on loopback by default; for a remote machine use an SSH tunnel (`ssh -N -L 18080:127.0.0.1:18080 <dev-server>`) |

The server has no authentication (planned, T-1204): keep it on loopback.

Open `http://localhost:18080`. Without `--gpus` the GPU pool is empty: a job that needs a GPU is rejected with HTTP 503, so on a machine without a GPU submit LLM jobs with `deploy.device: cpu` (VLA jobs always need the pool).
