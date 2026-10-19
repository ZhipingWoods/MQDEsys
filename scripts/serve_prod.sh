#!/usr/bin/env bash
# Production mqde serve on 4090 (port 18080). Single source of truth for the flags; see docs/DEVELOPMENT.md section 4.
set -u
cd /cache/wuzhiping/MQDEsys
pkill -x mqde 2>/dev/null; sleep 1
export MQDE_ENVS_DIR=/cache/wuzhiping/envs            # vllm / sglang / lmeval live here
export MQDE_LM_EVAL_BIN=/cache/wuzhiping/envs/lmeval/bin/lm_eval
nohup build/dev/bin/mqde serve \
  --repo /cache/wuzhiping/MQDEsys \
  --runs /cache/wuzhiping/MQDEsys/runs \
  --web /cache/wuzhiping/MQDEsys/web/dist \
  --python /root/miniconda3/envs/mqde/bin/python \
  --gpus "${MQDE_GPUS:-4,5,6,7}" \
  --allow /cache/wuzhiping/hub --allow /cache/wuzhiping/models \
  --port 18080 > /tmp/mqde_server.log 2>&1 &
sleep 5
curl -s http://127.0.0.1:18080/api/v1/health; echo
