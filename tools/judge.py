#!/usr/bin/env python3
"""Judge a round of candidates by the rules written down before they ran (DESIGN.md section 22).

  .venv/bin/python tools/judge.py --out runs_recipes/XSV2b --pool runs_recipes/XSV2a
  .venv/bin/python tools/judge.py --out runs_recipes/XSV2b --pool runs_recipes/XSV2a --pick 3

The judge is the hidden tests. A run that ends below 100 of them has collapsed: it is counted for its recipe and left
out of means (the mean over all runs is printed beside). A candidate's first run is what picks it for more runs, so a
re-run candidate is estimated from its later runs only. The baseline is the runs of the starting rules, taken two
ways: this round's own, and those pooled with the runs of --pool (the same texts shown, under earlier code). A
candidate is taken on when it has at least two later runs that did not collapse, their mean is above the baseline's,
and the one-sided 90% lower bound of the difference (Welch) is above zero against both baselines. Anything less is
"not shown", with its interval.

--factors A=x B=y judges rules that were run crossed (round 2): for each rule, every run with it against every run
without it, among the runs whose rules are made only of the named ones (the starting rules included), by the same
bound. --also ID adds a recipe's later runs from the --pool folders to its runs here (a candidate carried over).

--pick N names the N candidates with the highest hidden score in their first run (ties: more held-out checks): the
ones a second pass runs again. Only each run's result.json and job.json are read: never a work folder.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code_evolve import welch  # noqa: E402
from alphateam.recipe import seed  # noqa: E402
from alphateam.record import change  # noqa: E402

FLOOR = 100


def runs_of(out: Path) -> list[dict]:
    """Every finished run of an experiment, in the order the runs were made."""
    rows = []
    for res in sorted((out / "runs").glob("eval-*/result.json")):
        try:
            r, j = json.loads(res.read_text()), json.loads((res.parent / "job.json").read_text())
        except (ValueError, FileNotFoundError):
            continue
        rows.append({"run": res.parent.name, "recipe": j["recipe"]["id"], "rules": change(j["recipe"]),
                     "by": (j["recipe"].get("origin") or {}).get("by", ""), "hidden": r.get("hidden"),
                     "held": (r.get("facts") or {}).get("held_out"), "cost": r.get("cost")})
    return rows


def stood(xs: list, floor: int = FLOOR) -> list:
    return [x for x in xs if x is not None and x >= floor]


def fell(xs: list, floor: int = FLOOR) -> int:
    return sum(1 for x in xs if x is not None and x < floor)


def mean(xs: list):
    return statistics.mean(xs) if xs else None


def baseline(rows: list[dict], pool: list[dict], base: str) -> dict:
    own = [x["hidden"] for x in rows if x["recipe"] == base]
    return {"own": own, "pooled": own + [x["hidden"] for x in pool if x["recipe"] == base]}


def judge(rows: list[dict], pool: list[dict], base: str, floor: int = FLOOR, conf: float = 0.9) -> list[dict]:
    ref = baseline(rows, pool, base)
    by: dict[str, list[dict]] = {}
    for x in rows:
        if x["recipe"] != base:
            by.setdefault(x["recipe"], []).append(x)
    out = []
    for rid, xs in by.items():
        hidden = [x["hidden"] for x in xs]
        later = hidden[1:]
        kept = stood(later, floor)
        gaps = {name: welch(stood(vals, floor), kept, conf) for name, vals in ref.items()}
        if not later:
            verdict = "one run: not judged"
        elif len(kept) < 2:
            verdict = "not shown (fewer than two later runs that did not collapse)"
        elif all(low is not None and low > 0 for _, low in gaps.values()):
            verdict = "taken on"
        else:
            verdict = "not shown"
        if fell(hidden, floor):
            verdict += f"; flagged: {fell(hidden, floor)} of its {len(hidden)} runs collapsed"
        out.append({"recipe": rid, "rules": xs[0]["rules"], "by": xs[0]["by"], "first": hidden[0], "first_held": xs[0]["held"],
                    "later": later, "mean": mean(kept), "all_mean": mean([h for h in hidden if h is not None]),
                    "collapsed": fell(hidden, floor), "ungraded": sum(1 for h in hidden if h is None), "gaps": gaps, "verdict": verdict})
    return sorted(out, key=lambda c: (c["mean"] is None, -(c["mean"] or 0), -(c["first"] or 0)))


def pick(rows: list[dict], base: str, n: int) -> list[str]:
    """The candidates a second pass runs again: highest hidden score in the first run; ties, more held-out checks."""
    first: dict[str, dict] = {}
    for x in rows:
        if x["recipe"] != base:
            first.setdefault(x["recipe"], x)
    order = sorted(first.values(), key=lambda x: (-(x["hidden"] if x["hidden"] is not None else -1), -(x["held"] or 0)))
    return [x["recipe"] for x in order[:n]]


def factors(rows: list[dict], names: list[str], floor: int = FLOOR, conf: float = 0.9) -> list[dict]:
    """Rules run crossed: each one judged on all runs with it against all runs without it."""
    cell = lambda x: frozenset(s.strip() for s in x["rules"].split(",") if "=" in s)  # noqa: E731
    inside = [x for x in rows if cell(x) <= set(names)]
    out = []
    for name in names:
        w = [x["hidden"] for x in inside if name in cell(x)]
        wo = [x["hidden"] for x in inside if name not in cell(x)]
        d, low = welch(stood(wo, floor), stood(w, floor), conf)
        out.append({"rule": name, "with": w, "without": wo, "diff": d, "low": low,
                    "verdict": "shown" if low is not None and low > 0 and len(stood(w, floor)) >= 2 else "not shown"})
    return out


def gap_text(gap) -> str:
    d, low = gap
    return "-" if d is None else f"{d:+.0f}" + ("" if low is None else f" (bound {low:+.0f})")


def report(rows: list[dict], pool: list[dict], base: str, floor: int = FLOOR) -> str:
    ref = baseline(rows, pool, base)
    out = [f"{len(rows)} runs in this round, ${sum(x['cost'] or 0 for x in rows):.2f} by their own reports. Hidden tests of 1,182; "
           f"a run below {floor} has collapsed."]
    for name, vals in ref.items():
        kept = stood(vals, floor)
        sd = f", sd {statistics.stdev(kept):.0f}" if len(kept) > 1 else ""
        every = mean([v for v in vals if v is not None])
        out.append(f"starting rules, {name:6}: {len(kept)} runs that stood, mean {mean(kept):.0f}{sd}; {fell(vals, floor)} collapsed "
                   f"(mean over all {len([v for v in vals if v is not None])}: {every:.0f}); runs {vals}" if kept else
                   f"starting rules, {name}: no run yet")
    out.append("")
    out.append(f"{'recipe':12} {'by':9} {'first':>5} {'later runs':16} {'later mean':>10} {'all runs':>8}  {'vs own baseline':20} "
               f"{'vs pooled baseline':20} verdict | rules")
    for c in judge(rows, pool, base, floor):
        out.append(f"{c['recipe']:12} {str(c['by'])[:9]:9} {str(c['first']):>5} {str(c['later'])[:16]:16} "
                   f"{'-' if c['mean'] is None else f'{c['mean']:.0f}':>10} {'-' if c['all_mean'] is None else f'{c['all_mean']:.0f}':>8}  "
                   f"{gap_text(c['gaps']['own']):20} {gap_text(c['gaps']['pooled']):20} {c['verdict']} | {c['rules']}")
    out.append("")
    out.append("A difference is the mean of a candidate's later runs minus the baseline's, over runs that did not collapse; the bound "
               "is its one-sided 90% lower bound (Welch). The first run picked the candidate and is not in its estimate.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="the round's folder")
    p.add_argument("--pool", nargs="*", default=[], help="folders whose runs of the starting rules are pooled into the second baseline")
    p.add_argument("--factors", nargs="*", default=[], help="rules run crossed, as KEY=VALUE")
    p.add_argument("--also", nargs="*", default=[], help="recipes whose later runs in the pool folders count with their runs here")
    p.add_argument("--pick", type=int, help="only name the candidates a second pass runs again")
    a = p.parse_args()
    here = lambda s: (ROOT / s) if not Path(s).is_absolute() else Path(s)  # noqa: E731
    rows = runs_of(here(a.out))
    pool = [x for folder in a.pool for x in runs_of(here(folder))]
    base = seed("code").id
    if a.pick:
        print(" ".join(pick(rows, base, a.pick)))
        return
    print(report(rows, pool, base))
    for rid in a.also:
        earlier = [x["hidden"] for x in pool if x["recipe"] == rid][1:]
        mine = stood(earlier + [x["hidden"] for x in rows if x["recipe"] == rid])
        ref = baseline(rows, pool, base)
        gaps = {k: welch(stood(v), mine, 0.9) for k, v in ref.items()}
        ok = len(mine) >= 2 and all(low is not None and low > 0 for _, low in gaps.values())
        print(f"\n{rid} carried over: later runs {mine}, mean {mean(mine):.0f}; vs own {gap_text(gaps['own'])}, vs pooled "
              f"{gap_text(gaps['pooled'])}: {'taken on' if ok else 'not shown'}" if mine else f"\n{rid}: no runs")
    if a.factors:
        print()
        for f in factors(rows, a.factors):
            sw, so = stood(f["with"]), stood(f["without"])
            print(f"{f['rule']:24} with {f['with']} (mean {mean(sw):.0f}) without {f['without']} (mean {mean(so):.0f}): "
                  f"{gap_text((f['diff'], f['low']))}: {f['verdict']}" if sw and so else f"{f['rule']}: no runs yet")


if __name__ == "__main__":
    main()
