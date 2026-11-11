"""Run an agent society.

    python run.py --backend mock --ticks 12                 # offline plumbing test
    python run.py --ticks 10                                # Claude agents (needs ANTHROPIC_API_KEY)
    python run.py --condition fixed --token-budget 3000000  # baseline at matched compute
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from society.runner import CONDITIONS, build_world, inherit, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="scenarios/solid_electrolyte.json")
    ap.add_argument("--condition", default="self_org", choices=list(CONDITIONS))
    ap.add_argument("--backend", default="claude", choices=["claude", "mock"])
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--agents", type=int, default=0, help="use only the first N people in the scenario")
    ap.add_argument("--actions-per-turn", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=0, help="stop when the society has used this many tokens (0 = none)")
    ap.add_argument("--blank-personas", action="store_true", help="identical agents with no specialty (stronger no-prior test)")
    ap.add_argument("--inherit", type=Path, help="run.json of a previous society: start with its custom actions and lessons")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    scenario = json.loads(Path(args.scenario).read_text())
    if args.agents:
        scenario["people"] = scenario["people"][: args.agents]
    world = build_world(scenario, args.condition, args.actions_per_turn, args.blank_personas)
    world.token_budget = args.token_budget or None
    if args.inherit:
        inherit(world, args.inherit)

    if args.backend == "mock":
        from society.mock_backend import MockBackend
        backend = MockBackend(seed=args.seed)
    else:
        import anthropic
        from society.claude_backend import ClaudeBackend
        backend = ClaudeBackend(model=args.model, effort=args.effort, concurrency=args.concurrency)

    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.condition}-{args.backend}"
    out_dir = args.out / name
    meta = {"condition": args.condition, "backend": args.backend, "model": args.model if args.backend == "claude" else None,
            "effort": args.effort, "scenario": args.scenario, "seed": args.seed, "blank_personas": args.blank_personas,
            "token_budget": args.token_budget or None, "condition_note": CONDITIONS[args.condition]}
    print(f"{len(world.agents)} agents · condition {args.condition} · backend {args.backend} · output {out_dir}")
    try:
        asyncio.run(run(world, backend, args.ticks, out_dir, meta, args.seed))
    except Exception as exc:
        if args.backend == "claude" and isinstance(exc, anthropic.AuthenticationError):
            sys.exit("No valid Anthropic credentials: set ANTHROPIC_API_KEY (or run `ant auth login`).")
        raise
    print(f"\nviewer: {out_dir / 'viewer.html'}\nreport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
