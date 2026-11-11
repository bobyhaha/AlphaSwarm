#!/bin/bash
# DiG-bench runs (2026-11-01): the full design of the vc33 depth run, plus trust=1, on DiG-bench's public games.
#   ARM=society   one society, PER_GAME agents on each game, sharing everything (one run, cap CAP)
#   ARM=isolated  PER_GAME separate one-agent runs per game with the same tools, sharing nothing (all at once, CAP in all)
#   ARM=both      (default) both of the above, CAP each
# The arms use about the same agent-turns, so tools/dig_compare.py measures what sharing adds over more attempts.
#   GAMES=P-1,P-2,P-3 PER_GAME=4 TICKS=30 CAP=5 SEED=0 TAG=D_pilot ARM=society tools/launch_dig.sh
# COPIES=shared (2026-11-02): the society's agents on a game play one shared instance of it instead of one each
# (world option copies=shared), so the game is an object they act on together.
#   COPIES=shared ARM=society TAG=D_shared tools/launch_dig.sh
# LEASE=1 with COPIES=shared offers hold: an agent (or a group of its own) holds the shared instance for up to 2 ticks.
#   COPIES=shared LEASE=1 ARM=society TAG=D_lease tools/launch_dig.sh
# DIRECTIONS=1 offers directions (DESIGN.md section 18): lines of attack on a level, with headcounts, credit shared by
# those on a direction its solver cites, and each agent's own direction first in its deliveries.
# Asks for the OpenRouter key and the DiG-bench token without echoing them; both stay in this process's environment only,
# never on disk. Progress goes to runs_dig/<TAG>_<ARM>_s<SEED>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "${OPENROUTER_KEY_FILE:-}" ]; then  # one-use files (mode 600) the launcher reads and deletes
  OPENROUTER_API_KEY="$(cat "$OPENROUTER_KEY_FILE")"; rm -f "$OPENROUTER_KEY_FILE"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  read -rsp "OpenRouter API key (input hidden): " OPENROUTER_API_KEY; echo
fi
if [ -n "${DIG_TOKEN_FILE:-}" ]; then
  DIGBENCH_API_TOKEN="$(cat "$DIG_TOKEN_FILE")"; rm -f "$DIG_TOKEN_FILE"
fi
if [ -z "${DIGBENCH_API_TOKEN:-}" ]; then
  read -rsp "DiG-bench API token (input hidden): " DIGBENCH_API_TOKEN; echo
fi
export OPENROUTER_API_KEY DIGBENCH_API_TOKEN
ARM="${ARM:-both}"; GAMES="${GAMES:-P-1,P-2,P-3}"; PER_GAME="${PER_GAME:-4}"; TICKS="${TICKS:-30}"
CAP="${CAP:-5}"; SEED="${SEED:-0}"; TAG="${TAG:-D_pilot}"; COPIES="${COPIES:-own}"
case "$COPIES" in own|shared) ;; *) echo "COPIES must be own or shared"; exit 2;; esac
if [ "$COPIES" = shared ] && [ "$ARM" != society ]; then echo "COPIES=shared needs ARM=society"; exit 2; fi
LEASE="${LEASE:-0}"; DIRECTIONS="${DIRECTIONS:-0}"
case "$DIRECTIONS" in 0|1) ;; *) echo "DIRECTIONS must be 0 or 1"; exit 2;; esac
case "$LEASE" in 0|1) ;; *) echo "LEASE must be 0 or 1"; exit 2;; esac
if [ "$LEASE" = 1 ] && [ "$COPIES" != shared ]; then echo "LEASE=1 needs COPIES=shared"; exit 2; fi
SOC_LABEL=society; [ "$COPIES" = shared ] && SOC_LABEL=society-shared; [ "$LEASE" = 1 ] && SOC_LABEL=society-lease
OPTS=(--world dig --ticks "$TICKS" --actions 4 --calls 2 --attention 6000 --reasoning --world-opt copies="$COPIES"
      --world-opt lease="$LEASE" --world-opt directions="$DIRECTIONS"
      --world-opt steps=15 --world-opt substrate=1 --world-opt publish_solutions=1 --world-opt explore=1
      --world-opt delegation=1 --world-opt cycle=1 --world-opt sync=1 --world-opt carry=1 --world-opt continuity=2
      --world-opt final_moves=1 --world-opt guide=1 --world-opt hypotheses=1 --world-opt lessons=1 --world-opt board=1
      --world-opt trust=1)
mkdir -p runs_dig
echo "budget before: $(python3 tools/budget.py)"
n=$(echo "$GAMES" | tr ',' '\n' | grep -c .)
case "$ARM" in society|isolated|both) ;; *) echo "ARM must be society, isolated or both"; exit 2;; esac
if [ "$ARM" != isolated ]; then
  OUT="runs_dig/${TAG}_society_s${SEED}"
  nohup tools/run_live.sh "$CAP" "${OPTS[@]}" --agents $((PER_GAME * n)) --world-opt games="$GAMES" \
    --world-opt label="$SOC_LABEL" --seed "$SEED" --out "$OUT" > "$OUT.log" 2>&1 &
  echo "started the society (pid $!): $((PER_GAME * n)) agents, $PER_GAME on each of $GAMES ($COPIES instances); log $OUT.log"
fi
if [ "$ARM" != society ]; then
  OUT="runs_dig/${TAG}_isolated_s${SEED}"
  each="$(python3 -c "print(f'{max(0.6, $CAP / ($PER_GAME * $n)):.2f}')")"  # run_live.sh starts nothing under \$0.50
  mkdir -p "$OUT"
  (
    for g in $(echo "$GAMES" | tr ',' ' '); do
      for i in $(seq 1 "$PER_GAME"); do
        tools/run_live.sh "$each" "${OPTS[@]}" --agents 1 --world-opt games="$g" --world-opt label=isolated \
          --seed $((SEED * 100 + i)) --out "$OUT/$g-$i" > "$OUT/$g-$i.log" 2>&1 &
      done
    done
    wait
  ) > "$OUT.log" 2>&1 &
  echo "started the isolated arm (pid $!): $PER_GAME one-agent runs on each of $GAMES at once, \$$each cap each; logs in $OUT/"
fi
echo "You can close this tab; the runs keep going."
