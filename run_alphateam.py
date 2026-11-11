#!/usr/bin/env python3
"""Run a alphateam: a self-organizing society of LLM agents on a hidden-law science world or a real benchmark.

Offline (scripted agents, no API calls):
  .venv/bin/python run_alphateam.py --backend mock --agents 1000 --ticks 12

OpenRouter (needs OPENROUTER_API_KEY; default model deepseek/deepseek-v4.1-flash):
  .venv/bin/python run_alphateam.py --backend openrouter --agents 50 --ticks 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alphateam.evolve import lineage  # noqa: E402
from alphateam.llm import AnthropicProvider, Fatal, LLMBackend, OpenAICompatible  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import CONDITIONS, rebuild_viewer  # noqa: E402

# Measured cost per agent-turn (USD) in a 10-agent, 6-tick run on 2026-10-30 (3.6 calls, ~17k prompt tokens of which
# half cached, ~680 output tokens per turn). Larger societies see bigger observations, so treat it as a lower bound.
# Only used for the up-front estimate; the run reports OpenRouter's actual per-call cost.
MEASURED_PER_TURN = {"deepseek/deepseek-v4.1-flash": 0.0013}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["openrouter", "deepseek", "claude", "mock"], default="mock")
    p.add_argument("--model", help="model id (defaults: openrouter deepseek/deepseek-v4.1-flash, deepseek deepseek-flash, claude claude-opus-5-5)")
    p.add_argument("--reasoning", action="store_true", help="turn on model reasoning for OpenRouter/DeepSeek (slower, costlier)")
    p.add_argument("--effort", default="medium", help="effort for Claude")
    p.add_argument("--reasoning-effort", help="OpenRouter: reasoning effort for models that take one (e.g. low)")
    p.add_argument("--provider-ignore", default="", help="OpenRouter: comma list of providers to skip")
    p.add_argument("--max-tokens", type=int, help="largest reply per call (OpenRouter/DeepSeek)")
    p.add_argument("--strong", type=int, default=0, help="mixed society: this many randomly chosen agents run with reasoning "
                   "on (OpenRouter/DeepSeek); the others as configured; nobody is told who")
    p.add_argument("--condition", choices=list(CONDITIONS), default="self_org")
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--ticks", type=int, default=20)
    p.add_argument("--actions", type=int, default=4, help="primitive actions per agent per tick")
    p.add_argument("--calls", type=int, help="limit replies (model calls) per tick instead of actions: each reply may "
                   "take any number of actions; --actions is then ignored")
    p.add_argument("--attention", type=int, default=3000, help="reading budget per agent per tick, in tokens")
    p.add_argument("--world", default="hidden", help="hidden (the hidden-law world), discoverybench, scigym or explorationbench")
    p.add_argument("--tasks", type=int, help="benchmark worlds: use a random subset of this many tasks")
    p.add_argument("--home", type=int, default=3, help="benchmark worlds: home tasks dealt to each agent")
    p.add_argument("--eval-at", default="", help="benchmark worlds: comma list of ticks at which to grade (always graded at the end)")
    p.add_argument("--world-opt", action="append", default=[], metavar="KEY=VALUE", help="world-specific option (repeatable)")
    p.add_argument("--structure", default="hierarchical",
                   help="world dependency structure: hierarchical, modular or flat; a comma list cycles across episodes")
    p.add_argument("--instruments", type=int, default=3, help="outputs each agent can measure")
    p.add_argument("--redundancy", type=int, help="agents per output (default grows with sqrt of the society size)")
    p.add_argument("--experiments", type=int, default=30, help="experiment budget per agent per episode")
    p.add_argument("--seed-runs", type=int, default=3, help="free observations seeded on each private board")
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--validate-every", type=int, default=3)
    p.add_argument("--shock", action="append", default=[], help="T:shift (rewrite one super-module's laws) or T:churn:FRACTION")
    p.add_argument("--org-learning", type=int, metavar="W", help="design world: every W ticks every agent reflects and the "
                   "society's teamwork strategy is revised and enforced (continual organizational learning)")
    p.add_argument("--org-stall", action="store_true", help="with --org-learning: a window that does not improve is "
                   "marked stalled and the proposer must change the strategy")
    p.add_argument("--rule-learning", type=int, metavar="W", help="ARC world with explore=1: every W ticks the credit "
                   "rules and what the substrate shows are revised from agents' proposals and what has paid")
    p.add_argument("--reflect", type=int, metavar="W", help="code world: every W ticks the society reviews itself: every agent "
                   "reflects, and any rule may change (the law, norms, teams); held-out accuracy judges a change")
    p.add_argument("--reflect-floor", type=float, default=0.5, help="with --reflect: revert a change when the gain after it falls "
                   "below this share of the gain before it")
    p.add_argument("--org-judge", choices=["level", "delta"], default="level", help="with --org-learning: judge a strategy by its "
                   "window's mean score against the previous window (level) or by the score change during its own window (delta)")
    p.add_argument("--episodes", type=int, default=1, help="successive episodes; each inherits the previous culture")
    p.add_argument("--population", type=int, default=1, help="societies per episode; the best one's culture is passed on")
    p.add_argument("--inherit", help="start from the culture saved in a previous run.json")
    p.add_argument("--token-budget", type=int, help="stop when a society has used this many tokens")
    p.add_argument("--cost-budget", type=float, help="stop when a society has spent this many US dollars (uses the provider's reported cost)")
    p.add_argument("--concurrency", type=int, default=128, help="simultaneous LLM requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs_colony")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation for large paid runs")
    p.add_argument("--viewer", metavar="RUN_DIR", help="only re-render RUN_DIR/viewer.html from its run.json and events.jsonl")
    a = p.parse_args()
    if a.viewer:
        rebuild_viewer(Path(a.viewer))
        print(f"wrote {Path(a.viewer) / 'viewer.html'}")
        return

    structures = [s.strip() for s in a.structure.split(",") if s.strip()]
    cfg = {"condition": a.condition, "agents": a.agents, "ticks": a.ticks, "actions": a.actions, "attention": a.attention,
           "instruments": a.instruments, "redundancy": a.redundancy, "experiments": a.experiments, "seed_runs": a.seed_runs,
           "noise": a.noise, "validate_every": a.validate_every, "shocks": a.shock, "episodes": a.episodes,
           "population": a.population, "structures": structures, "token_budget": a.token_budget,
           "cost_budget": a.cost_budget, "seed": a.seed, "world": a.world, "tasks": a.tasks, "home": a.home,
           "eval_at": [int(x) for x in a.eval_at.split(",") if x.strip()], "org_window": a.org_learning, "rule_window": a.rule_learning, "org_stall": a.org_stall, "org_judge": a.org_judge, "calls": a.calls,
           "reflect_window": a.reflect, "reflect_floor": a.reflect_floor,
           "world_opts": dict(kv.split("=", 1) for kv in a.world_opt)}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    turns = (1 if a.condition == "single" else a.agents) * a.ticks * a.episodes * a.population
    meta = {"condition": a.condition, "backend": a.backend, "agents": a.agents, "ticks": a.ticks, "config": cfg,
            "condition_note": CONDITIONS[a.condition], "started": stamp}

    try:
        if a.backend == "mock":
            backend = MockBackend(a.seed)
        elif a.backend == "claude":
            provider = AnthropicProvider(a.model or "claude-opus-5-5", a.effort)
            meta["model"] = provider.model
            backend = LLMBackend(provider, a.concurrency)
        else:
            ignore = [x.strip() for x in a.provider_ignore.split(",") if x.strip()]
            provider = OpenAICompatible(a.backend, a.model, a.reasoning, run_id=stamp, max_tokens=a.max_tokens,
                                        reasoning_effort=a.reasoning_effort, provider_ignore=ignore)
            meta.update(model=provider.model, preset=a.backend, reasoning=a.reasoning, reasoning_effort=a.reasoning_effort)
            strong, chosen = None, set()
            if a.strong:
                import random as _random
                strong = OpenAICompatible(a.backend, a.model, True, run_id=stamp)
                chosen = set(_random.Random(f"strong:{a.seed}").sample([f"A{i + 1}" for i in range(a.agents)], min(a.strong, a.agents)))
                meta["strong_agents"] = sorted(chosen, key=lambda x: int(x[1:]))
            backend = LLMBackend(provider, a.concurrency, strong, chosen)
    except Fatal as exc:
        sys.exit(f"error: {exc}")

    if a.backend != "mock":
        line = f"{turns:,} agent turns ({a.agents} agents x {a.ticks} ticks x {a.episodes} episodes x {a.population} societies)"
        per_turn = MEASURED_PER_TURN.get(meta.get("model", "")) if not a.reasoning else None
        if per_turn:
            line += (f"; roughly ${turns * per_turn * a.actions / 4:,.2f} or more at ${per_turn} per agent-turn measured "
                     f"in a small run (actual cost is reported as it runs)")
        print(line)
        if turns > 200 and not a.yes:
            if not sys.stdin.isatty() or input("Proceed? [y/N] ").strip().lower() != "y":
                sys.exit("not started (pass --yes to skip this question)")

    start_culture = json.loads(Path(a.inherit).read_text()).get("culture") if a.inherit else None
    world = "" if a.world == "hidden" else f"{a.world}-"
    out_root = Path(a.out) / f"{stamp}-{world}{a.condition}-{a.backend}-{a.agents}a-s{a.seed}"
    n = 1
    while out_root.exists():  # runs started in the same second must not share a directory
        n += 1
        out_root = out_root.with_name(f"{stamp}-{world}{a.condition}-{a.backend}-{a.agents}a-s{a.seed}-{n}")
    out_root.mkdir(parents=True)
    try:
        summary = asyncio.run(lineage(cfg, backend, out_root, meta, start_culture))
    except Fatal as exc:
        sys.exit(f"error: {exc}")
    for s in summary:
        cost = f", cost ${s['cost_usd']:.2f}" if s.get("cost_usd") is not None else ""
        print(f"{s['dir']}: score {s['score'] if s['score'] is None else round(s['score'], 4)}{cost}")
    print(f"open {out_root}/{'' if len(summary) == 1 else '<episode>/'}viewer.html to replay; report.md summarizes")


if __name__ == "__main__":
    main()
