#!/bin/bash
# AlphaTeam on ProgramBench xsv (alphateam/code_world.py): ten agents, the same model, probes, held-out set and hidden tests as the
# ten-agent baseline in ~/asp-runs (mean 780.7 of 1182), 720 replies in all. The arms differ in whose wealth credit is:
#   ARM=individual | team | society   the goal is fixed; the review may change anything else
#   ARM=free                          the review may change the goal too
#   ARM=fixed                         credit, no review: the law never changes
#   ARM=plain                         no credit and no review: the primitives alone
#   ARM=review                        the review, no credit
#   WAYS=1                            any arm with organizational examples in the brief (default: none)
#   STREAM=1                          work lands as soon as it is offered, as in the ten-agent baseline, not at tick end
#   ARM=free EPISODES=3 TICKS=12 tools/launch_xsv.sh   generations: each society starts from no code, with the law, norms and tools of the last
# OPENROUTER_API_KEY comes from the environment and is never written anywhere. Progress goes to runs_xsv/<TAG>_<ARM>_s<SEED>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
ARM=${ARM:-individual}; SEED=${SEED:-1}; TAG=${TAG:-X}; TICKS=${TICKS:-24}; CALLS=${CALLS:-2}; STEPS=${STEPS:-12}
CAP=${CAP:-5}; REFLECT=${REFLECT:-6}; EPISODES=${EPISODES:-1}; SANDBOXES=${SANDBOXES:-3}; AGENTS=${AGENTS:-10}; PROBES=${PROBES:-480}
MODEL=${MODEL:-openai/gpt-6-luna}; CONDITION=${CONDITION:-self_org}
LAW=(); REVIEW=(--reflect "$REFLECT")
case "$ARM" in
  individual|team|society) LAW=(--world-opt "goal=$ARM" --world-opt goal_locked=1) ;;
  free) ;;
  fixed) REVIEW=() ;;
  plain) LAW=(--world-opt credit=0); REVIEW=() ;;
  review) LAW=(--world-opt credit=0) ;;
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
mkdir -p runs_xsv
exec tools/run_live.sh "$CAP" --model "$MODEL" --reasoning-effort low --provider-ignore openai/flex --max-tokens 16384 \
  --world code --condition "$CONDITION" --agents "$AGENTS" --ticks "$TICKS" --calls "$CALLS" --attention 6000 --seed "$SEED" \
  --episodes "$EPISODES" --out "runs_xsv/${TAG}_${ARM}_s${SEED}" "${REVIEW[@]+"${REVIEW[@]}"}" "${LAW[@]+"${LAW[@]}"}" \
  --world-opt sync=1 --world-opt board=1 --world-opt private=1 --world-opt teaming=1 --world-opt messages=1 --world-opt continuity=2 \
  --world-opt final_moves=1 --world-opt "steps=$STEPS" --world-opt "sandboxes=$SANDBOXES" --world-opt "probes=$PROBES" --world-opt "ways=${WAYS:-0}" --world-opt "stream=${STREAM:-0}" --world-opt held_every=merge --world-opt "heritage=${HERITAGE:-all}"
