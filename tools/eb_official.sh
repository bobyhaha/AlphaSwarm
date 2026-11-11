#!/bin/bash
# Official ExplorationBench single-model harness on the public demo worlds (calibration).  KEYFILE=... tools/eb_official.sh RUNTAG
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-bench/bin/python"
export OPENROUTER_API_KEY="$(cat "$KEYFILE")"
cd "$ROOT/ext/explorationbench/explorationbench"
tag="$1"
EVAL_TOOL_MODE=1 ALIENCODE_PROTOCOL_V2=1 "$PY" -u sandboxes/code/run_eval.py --model openrouter/deepseek/deepseek-v4.1-flash --short dsv41flash --run-id "code_$tag"
"$PY" -u scripts/answer_variance.py --run "code_$tag" --trials 3 --milestones 0,1,2,3,4
EVAL_TOOL_MODE=1 ALIENLOGIC_PROTOCOL_V2=1 "$PY" -u sandboxes/logic/run_eval.py --model openrouter/deepseek/deepseek-v4.1-flash --model-short dsv41flash --defer-tests --skip-pre --run-id "logic_$tag"
"$PY" -u scripts/logic_answer_variance.py --run "logic_$tag" --trials 3
echo "OFFICIAL HARNESS DONE"
