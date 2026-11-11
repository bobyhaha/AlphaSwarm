#!/bin/bash
# The design-world experiment: every arm in every world, a few at a time. Waits for all of them.
#   OPENROUTER_API_KEY=... TAG=D2 SEED=1 ARMS="flat ceo single alone" WORLDS="independent coupled moving" tools/design_wave.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
TAG=${TAG:-D2}; SEED=${SEED:-1}; ARMS=${ARMS:-flat ceo single alone}; WORLDS=${WORLDS:-independent coupled moving}; AT_ONCE=${AT_ONCE:-6}
mkdir -p runs_design
for world in $WORLDS; do
  for arm in $ARMS; do
    while [ "$(jobs -r | wc -l)" -ge "$AT_ONCE" ]; do sleep 2; done
    ( ARM=$arm WORLD=$world TAG=$TAG SEED=$SEED CAP=${CAP:-3} tools/launch_design.sh > "runs_design/${TAG}_${world}_${arm}_s${SEED}.log" 2>&1; echo "[design] $world $arm exit=$? $(date +%T)" ) &
  done
done
wait
echo "[design] wave $TAG seed $SEED done $(date +%T)"
