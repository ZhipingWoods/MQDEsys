#!/usr/bin/env bash
# One-shot developer environment (T-0008): conda toolchain -> Python dependency check -> CMake configure -> web deps.
# Idempotent: re-running updates the environment and reuses the build directory.
#
#   scripts/dev_setup.sh                 # Linux: conda env `mqde` from the toolchain file, then configure build/dev
#   ENV_NAME=mqde-ci BUILD_DIR=build/ci scripts/dev_setup.sh
#   scripts/dev_setup.sh --no-web        # skip `npm ci`
#   scripts/dev_setup.sh --check         # only report what is present, change nothing
#
# macOS: the toolchain file is Linux-only (GCC 12); install the C++ dependencies with brew
# (see docs/guides/dev-setup.md) and run this script with --no-env.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME=${ENV_NAME:-mqde}
ENV_FILE=${ENV_FILE:-docs/proposals/T-0006-ci/environment.ci.yml}
BUILD=${BUILD_DIR:-build/dev}
DO_ENV=1 DO_WEB=1 CHECK=0
for a in "$@"; do
  case $a in
    --no-env) DO_ENV=0 ;;
    --no-web) DO_WEB=0 ;;
    --check) CHECK=1 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

step() { printf '\033[1m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
act() { if [[ $CHECK -eq 1 ]]; then echo "(check) would run: $*"; else "$@"; fi; }

if [[ $DO_ENV -eq 1 ]]; then
  CONDA=$(command -v mamba || command -v conda || true)
  [[ -n $CONDA ]] || { echo "conda (or mamba) is required: https://github.com/conda-forge/miniforge" >&2; exit 1; }
  [[ -f $ENV_FILE ]] || { echo "toolchain file $ENV_FILE not found (set ENV_FILE=...)" >&2; exit 1; }
  if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    step "conda env '$ENV_NAME' exists: updating from $ENV_FILE"
    act "$CONDA" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
  else
    step "creating conda env '$ENV_NAME' from $ENV_FILE"
    act "$CONDA" env create -n "$ENV_NAME" -f "$ENV_FILE"
  fi
  echo "activate it:  conda activate $ENV_NAME"
fi

# the rest uses whatever is on PATH (the activated env, or brew on macOS)
for t in cmake ninja python3; do have "$t" || { echo "missing: $t (activate the environment first)" >&2; [[ $CHECK -eq 1 ]] || exit 1; }; done

# the Python package is used in place (PYTHONPATH=python, as check_all.sh does); the environment supplies its dependencies
step "python dependencies"
PYTHONPATH=python python3 -c "import torch, transformers, pyarrow, yaml, jsonschema, PIL; import mqde" \
  && echo ok || { echo "python dependencies missing: update the environment" >&2; [[ $CHECK -eq 1 ]] || exit 1; }

step "cmake configure -> $BUILD"
act cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release

if [[ $DO_WEB -eq 1 ]]; then
  if have npm; then step "web dependencies"; (cd web && act npm ci --no-audit --no-fund)
  else echo "npm not found: the web build will be skipped by check_all.sh"; fi
fi

step "done"
echo "next:  BUILD_DIR=$BUILD scripts/check_all.sh"
