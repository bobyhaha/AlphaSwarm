#!/bin/bash
# Run one live alphateam experiment within the research budget.
#   KEYFILE=/path/to/key tools/run_live.sh RUN_CAP_USD [run_alphateam.py args...]
#   (or with OPENROUTER_API_KEY already in the environment and no KEYFILE)
# The run's --cost-budget is the smaller of RUN_CAP_USD and what is left of the session cap (tools/budget.py).
set -euo pipefail
cd "$(dirname "$0")/.."
cap="$1"; shift
if [ -n "${KEYFILE:-}" ]; then export OPENROUTER_API_KEY="$(cat "$KEYFILE")"; fi
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY or KEYFILE}"
left="$(python3 tools/budget.py --left)"
budget="$(python3 -c "print(f'{max(0.0, min($cap, $left - 2)):.2f}')")"  # keep $2 slack for in-flight calls
if python3 -c "import sys; sys.exit(0 if $budget >= 0.5 else 1)"; then
  echo "run budget \$$budget (session left \$$left)"
  exec .venv/bin/python -u run_alphateam.py --backend openrouter --cost-budget "$budget" --yes "$@"
else
  echo "not started: only \$$left left in the session budget"; exit 3
fi
