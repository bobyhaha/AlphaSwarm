#!/bin/bash
# The vc33 depth run (2026-11-01): AGENTS (default 20) agents of the full design on vc33 alone for 60 ticks, seed SEED
# (default 0), cap CAP dollars (default 20), with the guide
# merge, the IDEAS list, credit for new ideas and for understanding, the lessons guide (DESIGN.md section 13), and
# the whole board: every post delivered once, search by relevance and use, every piece listed (section 14).
# Asks for the OpenRouter key without echoing it, keeps it only in this process's environment (never on disk), and
# starts the run in the background; progress goes to runs_arc/H_vc33_<agents>_s<seed>.log.
# EVIDENCE=1 adds evidence=1 (section 15); START=l4 starts every copy at level 4 (tools/vc33_start_l4.json); TICKS sets
# the length (default 60); NAME sets the run's name (default H). START=l5 starts at level 5 (tools/vc33_start_l5.json: the
# same moves plus A3's published level-4 solution from E_vc33_20_s0). ISLANDS=1 adds islands=1 (DESIGN.md section 17)
# with EMBARGO (default 3 ticks), PUBLISH (default 1/3: one publish per agent per 3 ticks) and GROUP_CAP (default 5).
# TEAMING=1 adds teaming=1: the prompt says what working with others does; each turn has a MEMBERSHIP block.
# MESSAGES=1 (a message action), ISLAND_GUIDES=1 (a guide per island) and UNIQUE_CREDIT=1 (credit for ideas only one
# island tested) are the I7 additions (DESIGN.md section 17).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  read -rsp "OpenRouter API key (input hidden): " OPENROUTER_API_KEY; echo
fi
export OPENROUTER_API_KEY
AGENTS="${AGENTS:-20}"; SEED="${SEED:-0}"; CAP="${CAP:-20}"; TICKS="${TICKS:-60}"; NAME="${NAME:-H}"
OUT="runs_arc/${NAME}_vc33_${AGENTS}_s${SEED}"
EXTRA=()
if [ "${EVIDENCE:-0}" = "1" ]; then EXTRA+=(--world-opt evidence=1); fi
if [ "${START:-}" = "l4" ]; then EXTRA+=(--world-opt start=tools/vc33_start_l4.json); fi
if [ "${START:-}" = "l5" ]; then EXTRA+=(--world-opt start=tools/vc33_start_l5.json); fi
if [ "${START:-}" = "l5x" ]; then EXTRA+=(--world-opt start=tools/vc33_start_l5x.json); fi  # plus A3's level-4 explanation
if [ "${ISLANDS:-0}" = "1" ]; then EXTRA+=(--world-opt islands=1 --world-opt "embargo=${EMBARGO:-3}"
                                         --world-opt "publish=${PUBLISH:-1/3}" --world-opt "group_cap=${GROUP_CAP:-5}"); fi
if [ "${TEAMING:-0}" = "1" ]; then EXTRA+=(--world-opt teaming=1); fi
if [ "${MESSAGES:-0}" = "1" ]; then EXTRA+=(--world-opt messages=1); fi
if [ "${ISLAND_GUIDES:-0}" = "1" ]; then EXTRA+=(--world-opt island_guides=1); fi
if [ "${UNIQUE_CREDIT:-0}" = "1" ]; then EXTRA+=(--world-opt unique_credit=1); fi
if [ "${DIRECTIONS:-0}" = "1" ]; then EXTRA+=(--world-opt directions=1); fi  # DESIGN.md section 18
if [ "${TRUST:-0}" = "1" ]; then EXTRA+=(--world-opt trust=1); fi
if [ "${CHANGE_CREDIT:-0}" = "1" ]; then EXTRA+=(--world-opt change_credit=1); fi  # DESIGN.md section 18
echo "budget before the run: $(python3 tools/budget.py)"
ARGS=(--world arc --ticks "$TICKS" --actions 4 --calls 2 --attention 6000 --reasoning
      --world-opt steps=15 --world-opt substrate=1 --world-opt publish_solutions=1
      --world-opt explore=1 --world-opt delegation=1 --world-opt cycle=1 --world-opt sync=1 --world-opt carry=1
      --world-opt model=1 --world-opt continuity=2 --world-opt final_moves=1 --world-opt guide=1
      --world-opt hypotheses=1 --world-opt lessons=1 --world-opt board=1
      --agents "$AGENTS" --world-opt games=vc33 --seed "$SEED" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"})
nohup tools/run_live.sh "$CAP" "${ARGS[@]}" > "$OUT.log" 2>&1 &
echo "started (pid $!); log: $OUT.log. You can close this tab; the run keeps going."
