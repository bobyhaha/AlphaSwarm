#!/bin/bash
# AlphaTeam on FrontierScience (alphateam/science_world.py), the pilot on work that divides.
#   ARM=individual | team | society   ten agents, the goal fixed; the review may change anything else
#   ARM=free                          the review may change the goal too
#   ARM=plain                         ten agents, no credit, no review: the primitives alone
#   ARM=review                        the review, no credit
#   ARM=single                        one agent with the same replies
#   SCALE=3                           with ARM=single, three times the replies: does one agent saturate?
#   HARD=0                            every question, not only those one agent leaves room on (tools/screen_science.py)
#   WAYS=1                            any arm with organizational examples in the brief (default: none)
# Same questions in every arm (SET picks them). Audits use the run's own model, the final grade FINAL_MODEL at high effort, one judge for all arms.
# OPENROUTER_API_KEY comes from the environment and is never written anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
ARM=${ARM:-individual}; SEED=${SEED:-1}; TAG=${TAG:-F}; TICKS=${TICKS:-12}; CALLS=${CALLS:-2}; AGENTS=${AGENTS:-10}; CAP=${CAP:-6}; REFLECT=${REFLECT:-4}
OLY=${OLY:-40}; RES=${RES:-20}; SET=${SET:-0}; HARD=${HARD:-1}; SCALE=${SCALE:-1}; EFFORT=${EFFORT:-low}; MODEL=${MODEL:-openai/gpt-6-luna}; FINAL_MODEL=${FINAL_MODEL:-openai/gpt-5}; STEPS=${STEPS:-12}
CONDITION=self_org; LAW=(); REVIEW=(--reflect "$REFLECT")
case "$ARM" in
  individual|team|society) LAW=(--world-opt "goal=$ARM" --world-opt goal_locked=1) ;;
  free) ;;
  plain) LAW=(--world-opt credit=0); REVIEW=() ;;
  review) LAW=(--world-opt credit=0) ;;
  single) CONDITION=single; CALLS=$(( (CALLS + 1) * AGENTS * SCALE - 1 )); REVIEW=(); LAW=(--world-opt force_budget=1) ;;
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
mkdir -p runs_science
exec tools/run_live.sh "$CAP" --model "$MODEL" --reasoning-effort "$EFFORT" --provider-ignore openai/flex --max-tokens 16384 \
  --world science --condition "$CONDITION" --agents "$AGENTS" --ticks "$TICKS" --calls "$CALLS" --attention 6000 --seed "$SEED" \
  --out "runs_science/${TAG}_${ARM}${SCALE/#1/}_s${SEED}" "${REVIEW[@]+"${REVIEW[@]}"}" "${LAW[@]+"${LAW[@]}"}" \
  --world-opt sync=1 --world-opt board=1 --world-opt private=1 --world-opt teaming=1 --world-opt messages=1 --world-opt continuity=2 \
  --world-opt final_moves=1 --world-opt "steps=$STEPS" --world-opt "olympiad=$OLY" --world-opt "research=$RES" --world-opt "set_seed=$SET" --world-opt "hard=$HARD" \
  --world-opt "ways=${WAYS:-0}" --world-opt "final_model=$FINAL_MODEL" --world-opt final_effort=high
