#!/usr/bin/env bash
# T-0705 acceptance: deploy a W4A16 (GGUF Q4_K) model on llama.cpp, CPU only (-ngl 0).
# Tiny random Qwen2 by default (runs in CI in seconds); needs a tokenizer dir, the llama.cpp source
# (MQDE_LLAMACPP_SRC, for convert_hf_to_gguf.py), llama-perplexity/llama-bench (MQDE_LLAMACPP_BIN or
# build/llamacpp/bin) and llama-server (MQDE_LLAMA_SERVER_BIN or build/llamacpp-server/bin).
#
#   MQDE_TEST_TOKENIZER=/path/to/Qwen2.5-0.5B-Instruct tests/integration/deploy_llamacpp_cpu.sh
#   GPU_LAYERS=99 tests/integration/deploy_llamacpp_cpu.sh   # same check with Metal / CUDA offload
set -euo pipefail
cd "$(dirname "$0")/../.."
TOK=${MQDE_TEST_TOKENIZER:-/cache/wuzhiping/hub/Qwen2.5-0.5B-Instruct}
WD=$(mktemp -d "${TMPDIR:-/tmp}/mqde-llamacpp-XXXXXX")
trap 'rm -rf "$WD"' EXIT
python tests/integration/deploy_llamacpp.py --tokenizer "$TOK" --type q4_k \
  --gpu-layers "${GPU_LAYERS:-0}" --threads "${THREADS:-4}" --workdir "$WD"
