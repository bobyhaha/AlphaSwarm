"""Shared by the recipe tools (recipe_evolve.py, scaling_grid.py): frames from the command line, backends, and the
guard on paid runs.

Offline is the default: scripted agents (--backend mock), no API calls, nothing spent. A paid run needs --backend
openrouter AND --live AND the OPENROUTER_API_KEY in the environment. It starts only if what the plan is expected to
cost fits in what is left of the research budget (spend.json, the same ledger tools/run_live.sh keeps to), with $2
kept back for calls in flight. With --cap it may start on less: it then runs until that cap and stops. The runner
gives every run it starts a share of what is left under the cap, stops a run that reaches its share, and uses no
score from a run that stopped.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphateam.mock import MockBackend  # noqa: E402
from alphateam.paired import frame  # noqa: E402

# measured on 2026-11-04 (RESEARCH_LOG.md): eight agents, 1,728 replies, about $0.30; one agent, the same replies, $0.55-0.90
USD_PER_REPLY, USD_PER_REPLY_SINGLE = 0.30 / 1728, 0.90 / 1728


def frame_args(p):
    p.add_argument("--arm", default="open", choices=["open", "chaired", "hierarchy", "single", "alone"])
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--replies", type=int, default=9, help="replies an agent a tick, its closing reply included")
    p.add_argument("--sims", type=int, default=9, help="simulations an agent a tick")
    p.add_argument("--blocks", type=int, help="blocks of the design (default: one an agent)")
    p.add_argument("--supers", type=int)
    p.add_argument("--dependency", default="coupled", choices=["coupled", "independent", "moving"])


def make_frame(a, **over) -> dict:
    kw = dict(arm=a.arm, agents=a.agents, ticks=a.ticks, replies=a.replies, sims=a.sims, blocks=a.blocks, supers=a.supers,
              dependency=a.dependency)
    kw.update(over)
    return frame(**kw)


def run_cost(f: dict) -> float:
    """What one run in a frame is expected to cost, from the replies it makes (an estimate from the November runs)."""
    if f["condition"] == "single":
        return (f["calls"] + 1) * f["ticks"] * USD_PER_REPLY_SINGLE
    return f["agents"] * (f["calls"] + 1) * f["ticks"] * USD_PER_REPLY


def backend_args(p):
    p.add_argument("--backend", default="mock", choices=["mock", "openrouter"])
    p.add_argument("--live", action="store_true", help="needed, with --backend openrouter, for a run that spends money")
    p.add_argument("--model", default="openai/gpt-6-luna")
    p.add_argument("--effort", default="low", help="reasoning effort")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--cap", type=float, help="dollars this command may spend; with it a plan that does not fit the budget "
                   "may still start and runs until the cap (default: the plan's expected cost, and the plan must fit)")


def budget_left() -> float:
    sys.path.insert(0, str(ROOT / "tools"))
    import budget
    led = json.loads((ROOT / "spend.json").read_text())
    return led["cap_usd"] - (budget.usage() - led["baseline_usage"])


def make_backend(a, planned_usd: float):
    """(a function seed -> backend for one society, metadata for the runs, a spending cap or None)."""
    if a.backend == "mock":
        return (lambda seed: MockBackend(seed)), {"backend": "mock"}, None
    if not a.live:
        sys.exit("not started: a paid run needs --live as well as --backend openrouter")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("not started: set OPENROUTER_API_KEY in the environment")
    left = budget_left()
    if a.cap is None and planned_usd > left - 2:
        sys.exit(f"not started: the plan is expected to cost ${planned_usd:.2f} and ${left:.2f} is left in the research "
                 f"budget (spend.json), of which $2 is kept back for calls in flight. --cap N runs until N dollars instead.")
    cap = min(a.cap if a.cap is not None else planned_usd, left - 2)
    if cap < 0.5:
        sys.exit(f"not started: ${left:.2f} is left in the research budget (spend.json), of which $2 is kept back")
    from alphateam.llm import LLMBackend, OpenAICompatible
    stamp = time.strftime("%Y%m%d-%H%M%S")
    provider = OpenAICompatible("openrouter", a.model, False, run_id=stamp, max_tokens=8000, reasoning_effort=a.effort,
                                provider_ignore=["openai/flex"])
    base = LLMBackend(provider, a.concurrency)

    def make(seed):
        b = copy.copy(base)  # one request limit across all runs; each society's backend keeps its own conversations
        b.memory = {}
        return b
    print(f"paid run: model {provider.model}, cap ${cap:.2f} (plan ${planned_usd:.2f}; ${left:.2f} left in the research budget)")
    return make, {"backend": "openrouter", "preset": "openrouter", "model": provider.model, "reasoning_effort": a.effort}, cap
