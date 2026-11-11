#!/usr/bin/env python3
"""The scaling grid and the positive control: recipes compared across arms and team sizes on paired task instances.

The positive control (does a paired test detect the one effect we know, the holder rule at 8 agents?). Its two arms
are the free and flat arms of the November runs: equal rights with every institution on offer, against one holder for
each shared knob and no claims or leads:
  .venv/bin/python tools/scaling_grid.py --control --pairs 20 --out runs_recipes/control
A bigger task for a bigger team: open, chaired and hierarchy starting recipes and a single agent at 4, 16 and 64
agents, one block an agent, the same replies and simulations in total for every arm of a size:
  .venv/bin/python tools/scaling_grid.py --sizes 4 16 64 --runs 10 --out runs_recipes/grid
The same task for a bigger team (is a larger team faster, or does it fall apart as 24 agents did?): one 16-block
design and the same 192 replies and simulations a tick in total, shared among 16, 32 or 64 agents; --target reports
the tick a run first reaches a score:
  .venv/bin/python tools/scaling_grid.py --fixed-blocks 16 --per-tick 192 --sizes 16 32 64 --target 0.5 --out runs_recipes/speed
With evolved recipes (from tools/recipe_evolve.py archives), to compare starting and evolved rules in an arm:
  ... --recipe open=runs_recipes/E1/archive:R1a2b3c4d5e --recipe chaired=runs_recipes/E2/archive:R9f8e7d6c5b

Offline by default (scripted agents, free: the machinery only). A paid run needs --backend openrouter --live and the
key in the environment, and starts only if the plan fits the budget left (tools/_recipes.py). --plan prints the plan
and its expected cost and starts nothing.

Every cell of a size runs on the same report instances, so differences between cells are paired by instance. (On
the moving world a single agent's rewire differs from a society's, so that one comparison is not paired.)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _recipes import ROOT, backend_args, make_backend, run_cost  # noqa: E402

from alphateam.archive import Archive  # noqa: E402
from alphateam.paired import AlphaTeamRunner, OverBudget, RunStopped, frame  # noqa: E402
from alphateam.recipe import apply_edits, seed  # noqa: E402
from alphateam.stats import summary  # noqa: E402

FLAT = [{"op": "set", "key": "holders", "value": "shared"}, {"op": "set", "key": "claims", "value": "off"},
        {"op": "set", "key": "take_lead", "value": "off"}]  # the flat arm of tools/launch_design.sh, as a recipe


def cells(a) -> list[dict]:
    """The grid: one cell an (arm, recipe, team size)."""
    start = seed("design")
    named = {}
    for spec in a.recipe:
        arm, _, where = spec.partition("=")
        path, _, rid = where.rpartition(":")
        named[arm] = Archive(path if Path(path).is_absolute() else ROOT / path).get(rid)
    if a.control:
        held = apply_edits(start, FLAT, {"by": "known"})
        return [{"name": "free", "arm": "open", "recipe": start, "agents": 8, "runs": a.pairs},
                {"name": "holders", "arm": "open", "recipe": held, "agents": 8, "runs": a.pairs}]
    out = []
    for n in a.sizes:
        runs = a.runs if n < 64 else max(2, a.runs // 2)  # the largest size costs most a run
        for arm in a.arms:
            if arm in ("single", "alone"):
                out.append({"name": arm, "arm": arm, "recipe": None, "agents": n, "runs": max(2, runs // 3) if arm == "single" else runs})
                continue
            out.append({"name": arm, "arm": arm, "recipe": start, "agents": n, "runs": runs})
            if arm in named:
                out.append({"name": f"{arm}, evolved", "arm": arm, "recipe": named[arm], "agents": n, "runs": runs})
    return out


def frame_of(c: dict, a) -> dict:
    replies, sims, blocks = a.replies, a.sims, 8 if a.control else None
    if a.fixed_blocks and not a.control:  # one fixed task: the same replies and simulations a tick, shared among the agents
        n, blocks = c["agents"], a.fixed_blocks
        if n < blocks or a.per_tick % n or a.per_tick // n < 2:
            sys.exit(f"{n} agents cannot share {a.per_tick} replies a tick on {blocks} blocks: there must be at least one agent "
                     f"a block, and a whole number of replies (2 or more) an agent")
        replies = sims = a.per_tick // n
    return frame(c["arm"], agents=c["agents"], ticks=a.ticks, replies=replies, sims=sims, dependency=a.dependency, blocks=blocks)


def reached(scores: list, target: float) -> int | None:
    """The first tick at which a run's score is at or above the target (None: never)."""
    return next((t for t, s in enumerate(scores) if t and s is not None and s >= target), None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    backend_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--control", action="store_true", help="the positive control: the holder rule on and off at 8 agents")
    p.add_argument("--pairs", type=int, default=20, help="pairs of the positive control")
    p.add_argument("--sizes", type=int, nargs="*", default=[4, 16, 64])
    p.add_argument("--arms", nargs="*", default=["open", "chaired", "hierarchy", "single"],
                   choices=["open", "chaired", "hierarchy", "single", "alone"])
    p.add_argument("--runs", type=int, default=10, help="runs a cell (half as many at 64 agents, a third for a single agent)")
    p.add_argument("--recipe", action="append", default=[], metavar="ARM=ARCHIVE:ID", help="an evolved recipe for an arm")
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--replies", type=int, default=9)
    p.add_argument("--sims", type=int, default=9)
    p.add_argument("--dependency", default="coupled")
    p.add_argument("--fixed-blocks", type=int, help="one fixed task of this many blocks at every size (default: one block an agent)")
    p.add_argument("--per-tick", type=int, default=192, help="with --fixed-blocks: replies and simulations a tick in total")
    p.add_argument("--target", type=float, help="also report the tick at which a run first reaches this score")
    p.add_argument("--conf", type=float, default=0.9)
    p.add_argument("--plan", action="store_true", help="print the plan and its expected cost; start nothing")
    a = p.parse_args()
    grid = cells(a)
    planned = sum(c["runs"] * run_cost(frame_of(c, a)) for c in grid)
    print("| cell | agents | blocks | replies a run | runs | expected cost |\n|---|---|---|---|---|---|")
    for c in grid:
        f = frame_of(c, a)
        replies = (f["calls"] + 1) * f["ticks"] * (1 if f["condition"] == "single" else f["agents"])
        print(f"| {c['name']} | {c['agents']} | {f['world_opts']['blocks']} | {replies:,} | {c['runs']} | ${c['runs'] * run_cost(f):.2f} |")
    print(f"\n{sum(c['runs'] for c in grid)} runs, expected ${planned:.2f} at the November prices (an estimate)")
    if a.plan:
        return
    make, meta, cap = make_backend(a, planned)
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scores: dict[tuple, list] = {}
    spent = 0.0

    async def go():
        nonlocal spent
        with open(out / "runs.jsonl", "a") as log:
            for c in grid:
                f = frame_of(c, a)
                runner = AlphaTeamRunner(f, make, out / "runs" / f"{c['name'].replace(', ', '-')}-n{c['agents']}",
                                      cap_usd=None if cap is None else cap - spent, meta=meta,
                                      est_cost=None if a.backend == "mock" else run_cost(f))
                key = (c["name"], c["agents"])
                scores[key] = []
                try:
                    for i in range(c["runs"]):
                        r = await runner.score(c["recipe"], i, "report")
                        scores[key].append({"score": r.score, "progress": r.progress,
                                            "at": None if a.target is None else reached(r.scores, a.target)})
                        log.write(json.dumps({"cell": c["name"], "agents": c["agents"], "niche": runner.niche, "recipe": r.recipe,
                                              "seed": r.seed, "score": r.score, "progress": r.progress, "cost": r.cost,
                                              "sims": r.sims, "ticks": r.ticks, "dir": r.dir}) + "\n")
                        log.flush()
                except (OverBudget, RunStopped) as exc:
                    print(f"stopped: {exc}")
                    return
                finally:
                    spent += runner.spent

    asyncio.run(go())
    speed = "" if a.target is None else f" runs that reached {a.target:g} | median tick they reached it |"
    print(f"\n| cell | agents | runs | final score | lower bound | progress score |{speed}\n|---|---|---|---|---|---|" + ("---|---|" if speed else ""))
    table = {}
    for (name, n), rows in scores.items():
        s, g = summary([r["score"] for r in rows], a.conf), summary([r["progress"] for r in rows], a.conf)
        table[f"{name}/n{n}"] = {"final": s, "progress": g}
        if s["n"]:
            line = f"| {name} | {n} | {s['n']} | {s['mean']:+.3f} | {'-' if s['lcb'] is None else format(s['lcb'], '+.3f')} | {g['mean']:+.3f} |"
            if a.target is not None:
                hit = sorted(r["at"] for r in rows if r["at"] is not None)
                table[f"{name}/n{n}"]["reached"] = hit
                line += f" {len(hit)} of {len(rows)} | {hit[len(hit) // 2] if hit else '-'} |"
            print(line)
    print("\npaired differences on the same instances (final score):")
    sizes = [8] if a.control else a.sizes
    for n in sizes:
        names = [name for (name, m) in scores if m == n]
        for other in names[1:]:
            x, y = scores[(names[0], n)], scores[(other, n)]
            d = summary([b["score"] - c["score"] for c, b in zip(x, y) if b["score"] is not None and c["score"] is not None], a.conf)
            table[f"{other} - {names[0]}/n{n}"] = d
            if d["n"]:
                bound = "" if d["lcb"] is None else f", lower bound {d['lcb']:+.3f}"
                print(f"  n={n}: {other} - {names[0]}: {d['mean']:+.3f} over {d['n']} instances{bound}")
    (out / "grid.json").write_text(json.dumps(table, indent=1))
    if a.backend == "mock":
        print("\nscripted agents: this shows the machinery only, not how societies of models would score.")
    else:
        print(f"\nruns cost ${spent:.2f}")


if __name__ == "__main__":
    main()
