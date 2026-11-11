#!/bin/bash
# The review's experiment on the design world: does structure appear where work needs it, and does it change when the work does?
#   WORLD=independent | coupled | moving    the same knobs and blocks; only who shares what differs, and the brief never says which
#   ARM=free    equal rights, every institution offered: take_lead, own_knob, delegate, contracts, groups
#   ARM=flat    every shared knob held by one agent who uses it: clear rights with no hierarchy
#   ARM=ceo     a prescribed hierarchy that knows the modules holds the rights
#   ARM=single  one agent with every block and the same replies
#   ARM=alone   the same agents with no communication
# PAY=0 takes away the engine's pay for leading (project_mult), which M1 had. SHARE=1 pays every agent the fall in the whole design's
# cost, on top of its own block's: the society-wealth arm. SIMS is simulations per agent per tick; a smaller society keeps the same
# replies and simulations in total with AGENTS=8 CALLS=8 SIMS=9. FACTS=1 adds to each review which shared knobs keep moving and who
# moved them: a measured fact, no advice. OPENROUTER_API_KEY comes from the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
ARM=${ARM:-free}; WORLD=${WORLD:-coupled}; SEED=${SEED:-1}; TAG=${TAG:-D}; TICKS=${TICKS:-24}; CALLS=${CALLS:-2}; AGENTS=${AGENTS:-24}; CAP=${CAP:-4}
BLOCKS=${BLOCKS:-8}; SUPERS=${SUPERS:-2}; MODEL=${MODEL:-openai/gpt-6-luna}; EFFORT=${EFFORT:-low}; PAY=${PAY:-2}; SHARE=${SHARE:-0}; SIMS=${SIMS:-3}; FACTS=${FACTS:-0}
SHARED=(--world-opt sync=1 --world-opt messages=1)
CONDITION=self_org; OPTS=(--world-opt ownership=1 --world-opt leadership=1 --world-opt contracts=1 --world-opt "project_mult=$PAY")
case "$ARM" in
  free) ;;
  flat) OPTS=(--world-opt authority=agent_shared --world-opt contracts=1 --world-opt "project_mult=$PAY") ;;
  ceo) CONDITION=oracle; OPTS=(--world-opt authority=1 --world-opt contracts=1 --world-opt "project_mult=$PAY") ;;
  single) CONDITION=single; CALLS=$(( (CALLS + 1) * AGENTS - 1 )); OPTS=(--world-opt force_budget=1) ;;
  alone) CONDITION=isolated; OPTS=(); SHARED=() ;;  # a society with no communication has nothing to synchronize
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
mkdir -p runs_design
exec tools/run_live.sh "$CAP" --model "$MODEL" --reasoning-effort "$EFFORT" --provider-ignore openai/flex --max-tokens 8000 \
  --world design --condition "$CONDITION" --agents "$AGENTS" --ticks "$TICKS" --calls "$CALLS" --attention 6000 --seed "$SEED" \
  --out "runs_design/${TAG}_${WORLD}_${ARM}_s${SEED}" --world-opt "dependency=$WORLD" --world-opt "blocks=$BLOCKS" --world-opt "supers=$SUPERS" \
  --world-opt "sims=$SIMS" --world-opt "contest=$FACTS" --world-opt review_every=1 --world-opt credit_every=3 --world-opt economy=1 --world-opt "economy_share=$SHARE" --world-opt substrate=1 \
  --world-opt continuity=2 "${SHARED[@]+"${SHARED[@]}"}" "${OPTS[@]+"${OPTS[@]}"}"
