#!/usr/bin/env python3
"""How many pairs does selection need? The spread of saved runs, then the ladder's error rates by simulation.

  .venv/bin/python tools/selection_power.py                      # the spread of the saved design runs, then the table
  .venv/bin/python tools/selection_power.py --sd 0.3 0.45 0.6    # sweep the run-to-run spread yourself
  .venv/bin/python tools/selection_power.py --confirm 20 --looks 10 20

Part 1 reads the valid design-world runs (series D6 to D10, after the search fix of 2026-11-04) and, for every group
of runs that differ only by seed, gives the spread of the final score and of the progress score (the mean score over
the run's ticks). No run is started.

Part 2 simulates the ladder's two scored stages for one revision with a true gain: the screen (a few pairs as
branches of one run; passed when the mean gain is above zero) and the confirmation (fresh pairs; kept when the lower
bound of the mean paired gain is above zero; given up at a look when the mean is at or below zero). A share rho of
the noise variance belongs to the task instance and cancels in a pair; branches share more (rho_branch). Saved runs
cannot say what rho is (no two arms were run on enough common seeds), so it is swept. This is arithmetic on stated
assumptions, not a result about societies.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphateam.stats import area, summary, t_quantile  # noqa: E402

VALID = re.compile(r"^(D(?:6|7|8|9|10))_([a-z]+)_([a-z]+)_s(\d+)$")


def saved(root: Path) -> dict:
    groups = defaultdict(list)
    for d in sorted(root.iterdir()):
        m = VALID.match(d.name)
        if not m or not d.is_dir():
            continue
        for mj in d.glob("*/metrics.json"):
            scores = [x.get("score") for x in json.loads(mj.read_text())]
            meta = json.loads((mj.parent / "run.json").read_text())["meta"]
            if meta.get("stopped") or len(scores) < 3:
                continue
            groups[(m.group(1), m.group(2), m.group(3), meta.get("agents"))].append(
                {"seed": int(m.group(4)), "final": scores[-1], "progress": area(scores, 1), "cost": meta.get("cost_usd")})
    return groups


def spread(groups: dict) -> dict:
    print("| series | world | arm | agents | runs | final: mean | final: sd | progress: mean | progress: sd |")
    print("|---|---|---|---|---|---|---|---|---|")
    pooled = {"final": [0.0, 0], "progress": [0.0, 0]}
    for (tag, world, arm, agents), runs in sorted(groups.items()):
        f, p = summary([r["final"] for r in runs]), summary([r["progress"] for r in runs])
        sd = lambda s: "-" if s["sd"] is None else f"{s['sd']:.2f}"  # noqa: E731
        print(f"| {tag} | {world} | {arm} | {agents} | {len(runs)} | {f['mean']:+.2f} | {sd(f)} | {p['mean']:+.2f} | {sd(p)} |")
        for name, s in (("final", f), ("progress", p)):
            if s["sd"] is not None:
                pooled[name][0] += s["sd"] ** 2 * (s["n"] - 1)
                pooled[name][1] += s["n"] - 1
    out = {name: (math.sqrt(ss / df) if df else None) for name, (ss, df) in pooled.items()}
    df = pooled["final"][1]
    print(f"\npooled spread within groups ({df} degrees of freedom): final score sd {out['final']:.2f}, "
          f"progress score sd {out['progress']:.2f}" if df else "\nno group has two runs")
    return out


def confirm(effect, sd_diff, pairs, looks, conf, rng) -> tuple[bool, int]:
    diffs = []
    for j in range(1, pairs + 1):
        diffs.append(rng.gauss(effect, sd_diff))
        if j in looks:
            mean = sum(diffs) / j
            if j == looks[-1]:
                sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (j - 1))
                return mean - t_quantile(conf, j - 1) * sd / math.sqrt(j) > 0, j
            if mean <= 0:
                return False, j
    return False, pairs


def simulate(sd, rho, rho_branch, effects, a, reps, rng):
    looks = sorted({n for n in a.looks if n <= a.confirm} | {a.confirm})
    d_fresh, d_branch = sd * math.sqrt(2 * (1 - rho)), sd * math.sqrt(2 * (1 - rho_branch))
    rows = []
    for e in effects:
        passed = kept = pairs = 0
        for _ in range(reps):
            ok = a.screen == 0 or sum(rng.gauss(e, d_branch) for _ in range(a.screen)) / a.screen > 0
            passed += ok
            if ok:
                k, n = confirm(e, d_fresh, a.confirm, looks, a.conf, rng)
                kept += k
                pairs += n
        rows.append({"gain": e, "pass_screen": passed / reps, "kept": kept / reps,
                     "confirm_pairs": pairs / max(1, passed), "runs": a.screen + 1 + 2 * pairs / reps})
    return rows


def generation(sd, rho, rho_branch, a, reps, rng, k: int = 4) -> float:
    """How often a generation of k useless revisions ends with one kept: the best of k at the screen goes on to be
    confirmed. The k children are branches of one run of the parent, so their screening gains share its luck."""
    looks = sorted({n for n in a.looks if n <= a.confirm} | {a.confirm})
    d_fresh, s_branch = sd * math.sqrt(2 * (1 - rho)), sd * math.sqrt(1 - rho_branch)
    kept = 0
    for _ in range(reps):
        best = 0.0
        if a.screen:
            sums = [0.0] * k
            for _ in range(a.screen):
                parent = rng.gauss(0, s_branch)
                for j in range(k):
                    sums[j] += rng.gauss(0, s_branch) - parent
            best = max(sums) / a.screen
        if not a.screen or best > 0:
            kept += confirm(0.0, d_fresh, a.confirm, looks, a.conf, rng)[0]
    return kept / reps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_design", help="where the saved design runs are")
    p.add_argument("--sd", type=float, nargs="*", help="run-to-run spread of the score (default: the pooled spread of the saved runs)")
    p.add_argument("--rho", type=float, nargs="*", default=[0.0, 0.3, 0.6], help="share of the noise variance that a pair shares")
    p.add_argument("--rho-branch", type=float, default=0.7, help="the same for two branches of one run")
    p.add_argument("--gain", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    p.add_argument("--screen", type=int, default=4, help="screening pairs (0: no screen)")
    p.add_argument("--confirm", type=int, default=30, help="confirming pairs at most")
    p.add_argument("--looks", type=int, nargs="*", default=[10, 20, 30])
    p.add_argument("--conf", type=float, default=0.9)
    p.add_argument("--candidates", type=int, default=4, help="candidate revisions a generation")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write the tables here")
    a = p.parse_args()
    out = {"spread": None, "tables": []}
    root = ROOT / a.runs
    sds = a.sd
    if root.is_dir():
        print(f"## The spread of saved runs ({a.runs}, series D6-D10)\n")
        pooled = spread(saved(root))
        out["spread"] = pooled
        if not sds and pooled["final"]:
            sds = [round(pooled["final"], 2)]
    sds = sds or [0.45]
    rng = random.Random(a.seed)
    print(f"\n## What the ladder keeps ({a.screen} screening pairs, up to {a.confirm} confirming pairs, looks at "
          f"{', '.join(map(str, a.looks))}, lower bound at {a.conf:.0%}; {a.reps} simulated revisions a row)\n")
    print("| spread (sd) | shared by a pair (rho) | true gain | passes the screen | kept | confirming pairs used | runs a revision |")
    print("|---|---|---|---|---|---|---|")
    for sd in sds:
        for rho in a.rho:
            rows = simulate(sd, rho, max(rho, a.rho_branch), a.gain, a, a.reps, rng)
            out["tables"].append({"sd": sd, "rho": rho, "rows": rows})
            for r in rows:
                print(f"| {sd:.2f} | {rho:.1f} | {r['gain']:+.1f} | {r['pass_screen']:.0%} | {r['kept']:.1%} | "
                      f"{r['confirm_pairs']:.0f} | {r['runs']:.0f} |")
    print(f"\nA generation of {a.candidates} useless revisions ends with one kept (the best of them at the screen goes on):")
    for sd in sds:
        for rho in a.rho:
            g = generation(sd, rho, max(rho, a.rho_branch), a, a.reps, rng, a.candidates)
            out["tables"].append({"sd": sd, "rho": rho, "generation_false_keep": g})
            print(f"  spread {sd:.2f}, shared {rho:.1f}: {g:.1%}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
