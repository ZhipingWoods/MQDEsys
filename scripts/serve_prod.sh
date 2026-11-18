#!/usr/bin/env bash
# Start a local MQDEsys service. Every path can be overridden for the target machine.
set -euo pipefail

MQDE_ROOT="${MQDE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MQDE_BUILD="${MQDE_BUILD:-$MQDE_ROOT/build/dev}"
MQDE_RUNS="${MQDE_RUNS:-$MQDE_ROOT/runs}"
MQDE_WEB="${MQDE_WEB:-$MQDE_ROOT/web/dist}"
MQDE_PYTHON="${MQDE_PYTHON:-python}"
MQDE_PORT="${MQDE_PORT:-18080}"
MQDE_ENVS_DIR="${MQDE_ENVS_DIR:-}"
MQDE_LM_EVAL_BIN="${MQDE_LM_EVAL_BIN:-}"
MQDE_ALLOW="${MQDE_ALLOW:-}"
MQDE_EXTRA_ALLOW="${MQDE_EXTRA_ALLOW:-}"

cd "$MQDE_ROOT"
pkill -x mqde 2>/dev/null || true
sleep 1

args=(
  "$MQDE_BUILD/bin/mqde" serve
  --repo "$MQDE_ROOT"
  --runs "$MQDE_RUNS"
  --web "$MQDE_WEB"
  --python "$MQDE_PYTHON"
  --port "$MQDE_PORT"
)

[[ -n "${MQDE_GPUS:-}" ]] && args+=(--gpus "$MQDE_GPUS")
[[ -n "$MQDE_ALLOW" ]] && args+=(--allow "$MQDE_ALLOW")
[[ -n "$MQDE_EXTRA_ALLOW" ]] && args+=(--allow "$MQDE_EXTRA_ALLOW")

[[ -n "$MQDE_ENVS_DIR" ]] && export MQDE_ENVS_DIR
[[ -n "$MQDE_LM_EVAL_BIN" ]] && export MQDE_LM_EVAL_BIN

nohup "${args[@]}" > "${MQDE_LOG:-/tmp/mqde-server.log}" 2>&1 &
sleep 5
curl -fsS "http://127.0.0.1:${MQDE_PORT}/api/v1/health"; echo
