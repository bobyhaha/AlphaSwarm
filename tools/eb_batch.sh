#!/bin/bash
# ExplorationBench replication batch: seeds 1-3 for self_org and single, plus a per-agent-budget society.
cd "$(dirname "$0")/.."
for s in 1 2 3; do
  tools/run_live.sh 2 --world explorationbench --agents 20 --ticks 12 --seed $s --concurrency 64 --out runs_bench > logs/eb_self20_s$s.log 2>&1 &
  tools/run_live.sh 1 --world explorationbench --condition single --agents 20 --ticks 12 --seed $s --concurrency 32 --out runs_bench > logs/eb_single_s$s.log 2>&1 &
  sleep 2
done
tools/run_live.sh 3 --world explorationbench --agents 20 --ticks 12 --seed 0 --world-opt probe_budget=per_agent:12 --concurrency 64 --out runs_bench > logs/eb_self20_peragent.log 2>&1 &
wait
echo "EB BATCH DONE"
