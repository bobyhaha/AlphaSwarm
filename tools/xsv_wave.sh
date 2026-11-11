#!/bin/bash
# One wave of xsv societies, all at once, keeping the machine awake until the last ends.
#   OPENROUTER_API_KEY=... TAG=X1 SEED=1 ARMS="individual team society plain" tools/xsv_wave.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
TAG=${TAG:-X1}; SEED=${SEED:-1}; ARMS=${ARMS:-individual team society plain}
mkdir -p runs_xsv
pids=()
for arm in $ARMS; do
  log="runs_xsv/${TAG}_${arm}_s${SEED}.log"
  ( ARM=$arm TAG=$TAG SEED=$SEED tools/launch_xsv.sh > "$log" 2>&1; echo "[xsv] $arm exit=$? $(date +%T)" ) &
  pids+=($!)
  sleep 20
done
caffeinate -i -w "${pids[0]}" &
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[xsv] wave $TAG done $(date +%T)"
