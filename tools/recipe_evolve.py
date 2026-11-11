#!/usr/bin/env python3
"""Evolve a society's rules: generations of recipes, each revision tested against its parent (alphateam/generations.py).

Offline, with scripted agents (free; it shows the machinery, not how societies of models would score):
  .venv/bin/python tools/recipe_evolve.py --generations 3 --out runs_recipes/demo

The control arm (the same generations, a random valid revision kept, nothing scored):
  .venv/bin/python tools/recipe_evolve.py --control --out runs_recipes/demo-control

A paid run (needs OPENROUTER_API_KEY, --backend openrouter and --live; starts only if the plan fits the budget left):
  .venv/bin/python tools/recipe_evolve.py --backend openrouter --live --agents 16 --generations 5 --out runs_recipes/E1

Let societies try their own proposals on a copy of themselves during generation runs (alphateam/trial.py):
  ... --trials 6,3,2      # a beat every 6 ticks, a 3-tick window, proposals with at least 2 backers

The archive (recipes, runs, pairs, decisions, lessons) is under OUT/archive and can be carried on with the same --out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _recipes import ROOT, backend_args, frame_args, make_backend, make_frame, run_cost  # noqa: E402

from alphateam.archive import Archive, ArchiveMismatch  # noqa: E402
from alphateam.generations import evolve, report  # noqa: E402
from alphateam.paired import AlphaTeamRunner, OverBudget, RunStopped  # noqa: E402
from alphateam.recipe import seed  # noqa: E402
from alphateam.selection import Policy  # noqa: E402


def plan_runs(a) -> float:
    """Full runs the plan could take at most, counting a branch as half a run."""
    per_gen = 1 + a.retest + a.per_generation * a.screen_pairs * 0.5 + a.screen_pairs * 0.5 + a.finalists * 2 * a.confirm_pairs
    if a.control:
        per_gen = 1
    return a.generations * per_gen + a.report_runs * 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    frame_args(p)
    backend_args(p)
    p.add_argument("--out", required=True, help="where the archive and the runs go")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--per-generation", type=int, default=4, help="candidate revisions a generation")
    p.add_argument("--screen-pairs", type=int, default=4)
    p.add_argument("--confirm-pairs", type=int, default=30)
    p.add_argument("--looks", type=int, nargs="*", default=[10, 20, 30])
    p.add_argument("--finalists", type=int, default=1)
    p.add_argument("--retest", type=int, default=3, help="runs a recipe has before it seeds a generation")
    p.add_argument("--control", action="store_true", help="the control arm: keep a random valid revision, score nothing")
    p.add_argument("--menu-only", action="store_true", help="a second control: the same ladder and budget, with every "
                   "candidate drawn from the menu of edits (what agents' own proposals add)")
    p.add_argument("--report-runs", type=int, default=0, help="score the seed and the best recipe on this many held-out instances")
    p.add_argument("--trials", help="EVERY,WINDOW,BACKERS: generation runs may try their own proposals on a copy of themselves")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="start from the seed recipe with this setting (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    f = make_frame(a)
    if not f["rules"]:
        sys.exit(f"the {a.arm} arm has no rules to evolve")
    planned = plan_runs(a) * run_cost(f)
    make, meta, cap = make_backend(a, planned)
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    archive = Archive(out / "archive")
    trials = None
    if a.trials:
        every, window, backers = (int(x) for x in a.trials.split(","))
        trials = {"every": every, "window": window, "backers": backers}
    runner = AlphaTeamRunner(f, make, out / "runs", trials=trials, cap_usd=cap, meta=meta,
                          est_cost=None if a.backend == "mock" else run_cost(f))
    policy = Policy(mode="random" if a.control else "paired", screen_pairs=a.screen_pairs, confirm_pairs=a.confirm_pairs,
                    looks=tuple(a.looks), finalists=a.finalists, retest_runs=a.retest)
    start = seed(f["world"], **dict(kv.split("=", 1) for kv in a.set))
    print(f"niche {runner.niche}; seed recipe {start.id}; up to {plan_runs(a):.0f} full runs"
          + (f", about ${planned:.2f}" if a.backend != "mock" else " with scripted agents (free)"))

    async def go():
        try:
            res = await evolve(archive, runner, start, a.generations, a.per_generation, policy, random.Random(a.seed),
                               agents=not a.menu_only)
            if a.report_runs:
                res["report"] = await report(archive, runner, {"seed": start, "evolved": archive.get(res["best"])}, a.report_runs)
        except (OverBudget, RunStopped) as exc:  # what was judged is on file; the same command carries on from it
            print(f"stopped: {exc}")
            return None
        return res

    try:
        res = asyncio.run(go())
    except ArchiveMismatch as exc:
        sys.exit(f"not started: {exc}")
    print()
    print(archive.table(runner.niche))
    if res:
        (out / "evolution.json").write_text(json.dumps(res, indent=1, default=list))
        print(f"\nbest recipe {res['best']} (lineage {' > '.join(res['lineage'])}); "
              + (f"runs cost ${runner.spent:.2f}; " if a.backend != "mock" else "") + f"archive in {out / 'archive'}")
    if a.backend == "mock":
        print("scripted agents: this shows the machinery only. Scripted agents ignore most rules and never propose.")


if __name__ == "__main__":
    main()
