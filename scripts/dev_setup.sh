#!/usr/bin/env bash
# Prepare MQDEsys on the user's local device, using its own compiler, Python and package manager.
#
#   bash scripts/dev_setup.sh                 # install local prerequisites, build, start
#   MQDE_ALLOW=/data/models MQDE_GPUS=0 bash scripts/dev_setup.sh
#   bash scripts/dev_setup.sh --no-start      # prepare and build only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV_DIR="${MQDE_VENV_DIR:-$PWD/.venv}"
BUILD_DIR="${BUILD_DIR:-build/dev}"
MQDE_PORT="${MQDE_PORT:-18080}"
MQDE_ALLOW="${MQDE_ALLOW:-$PWD/models}"
MQDE_RUNS="${MQDE_RUNS:-$PWD/runs}"
START=1
[[ "${1:-}" == "--no-start" ]] && START=0

step() { printf '\033[1m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
run_root() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi; }

if [[ "${MQDE_INSTALL_SYSTEM_DEPS:-1}" == 1 ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]] && have conda; then
    step "installing native dependencies in the active conda environment"
    conda install -y -c conda-forge cmake ninja cxx-compiler nlohmann_json yaml-cpp fmt cli11 cpp-httplib gtest pybind11 libcurl openssl sqlite nodejs
  elif [[ "$(uname -s)" == Darwin ]] && have brew; then
    step "installing native dependencies with Homebrew"
    brew install cmake ninja fmt yaml-cpp cli11 cpp-httplib googletest nlohmann-json pybind11 node
  elif have apt-get; then
    step "installing native dependencies with apt"
    run_root apt-get update
    run_root apt-get install -y build-essential cmake ninja-build pkg-config python3-dev python3-venv nodejs npm \
      libfmt-dev libyaml-cpp-dev nlohmann-json3-dev libsqlite3-dev libcurl4-openssl-dev libgtest-dev \
      libcli11-dev libcpp-httplib-dev pybind11-dev
  elif have dnf; then
    step "installing native dependencies with dnf"
    run_root dnf install -y gcc-c++ cmake ninja-build python3-devel nodejs npm \
      fmt-devel yaml-cpp-devel nlohmann-json-devel sqlite-devel libcurl-devel gtest-devel \
      cli11-devel cpp-httplib-devel pybind11-devel
  else
    echo "No supported local package manager found. Install the dependencies listed in the README, then rerun with MQDE_INSTALL_SYSTEM_DEPS=0." >&2
    exit 1
  fi
fi

for tool in cmake ninja npm; do
  have "$tool" || { echo "missing required tool: $tool" >&2; exit 1; }
done
PYTHON="${MQDE_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
[[ -n "$PYTHON" ]] || { echo "Python 3.10+ is required" >&2; exit 1; }
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  step "creating local Python environment at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"

mkdir -p "$MQDE_ALLOW" "$MQDE_RUNS"
step "installing Python executor and quantization dependencies"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${CONDA_PREFIX:-}}" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --upgrade -e "python[dev,quant,export]"
step "building C++ control plane"
CMAKE_ARGS=(-DPython_EXECUTABLE="$PYTHON")
PYBIND11_CMAKE_DIR="$($PYTHON -m pybind11 --cmakedir 2>/dev/null || true)"
[[ -n "$PYBIND11_CMAKE_DIR" ]] && CMAKE_ARGS+=("-Dpybind11_DIR=$PYBIND11_CMAKE_DIR")
cmake --preset dev "${CMAKE_ARGS[@]}"
cmake --build --preset dev -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}"
step "building Web workspace"
(cd web && npm ci --no-audit --no-fund && npm run build)

if [[ -z "${MQDE_GPUS+x}" ]]; then MQDE_GPUS=; have nvidia-smi && MQDE_GPUS=0; fi
export MQDE_ROOT="$PWD" MQDE_BUILD="$BUILD_DIR" MQDE_RUNS MQDE_WEB="$PWD/web/dist"
export MQDE_PYTHON="$PYTHON" MQDE_PORT MQDE_ALLOW MQDE_GPUS
if [[ $START -eq 1 ]]; then
  step "starting MQDEsys on http://127.0.0.1:${MQDE_PORT}"
  scripts/serve_prod.sh
  echo "MQDEsys is ready at http://127.0.0.1:${MQDE_PORT}"
else
  echo "Build complete. Start with: MQDE_GPUS=${MQDE_GPUS:-} MQDE_ALLOW=$MQDE_ALLOW scripts/serve_prod.sh"
fi
