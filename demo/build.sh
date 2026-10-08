#!/usr/bin/env bash
# Re-record both language versions. Usage: demo/build.sh [zh|en|all]   (default: all)
# Extra env: DEMO_BASE (default http://localhost:18080), DEMO_VOICE_ZH, DEMO_VOICE_EN, DEMO_RATE_ZH, DEMO_RATE_EN, BURN_SUBS=1
set -euo pipefail
cd "$(dirname "$0")"
BASE="${DEMO_BASE:-http://localhost:18080}"
for t in node npm ffmpeg ffprobe say; do command -v "$t" >/dev/null || { echo "missing: $t (brew install ffmpeg for ffmpeg/ffprobe)"; exit 1; }; done
[ -d node_modules/playwright ] || npm install --no-audit --no-fund
curl -sf -o /dev/null "$BASE/" || { echo "web UI not reachable at $BASE"; exit 1; }
which="${1:-all}"; langs=(zh en); [ "$which" != all ] && langs=("$which")
START=$(date +%s)
for l in "${langs[@]}"; do
  u=$(echo "$l" | tr a-z A-Z)
  vv="DEMO_VOICE_$u"; rr="DEMO_RATE_$u"
  args=(--lang "$l" --base "$BASE")
  [ -n "${!vv:-}" ] && args+=(--voice "${!vv}")
  [ -n "${!rr:-}" ] && args+=(--rate "${!rr}")
  node record.mjs "${args[@]}"
done
for l in "${langs[@]}"; do
  f="out/mqde-demo-$l.mp4"
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  s=$(du -m "$f" | cut -f1)
  echo "$f  ${d}s  ${s} MB  streams: $(ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$f" | tr '\n' ' ')"
  # Sidecar subtitles are always written to out/mqde-demo-$l.srt. Optional burned-in copy:
  if [ "${BURN_SUBS:-0}" = 1 ]; then
    ffmpeg -y -loglevel error -i "$f" -vf "subtitles=out/mqde-demo-$l.srt" -c:v libx264 -crf 28 -c:a copy "out/mqde-demo-$l.burned.mp4"
  fi
done
echo "build took $(( $(date +%s) - START )) s"
