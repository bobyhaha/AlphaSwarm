#!/usr/bin/env python3
"""Compare ARC-AGI-3 runs: score, the score's area under the curve (mean over ticks; rewards reaching levels sooner),
first completions, how far solutions spread, and which offered actions each society used.
  python3 tools/arc_compare.py runs_arc/R4_* runs_arc/X* runs_arc/V*
Works on finished runs (run.json) and on runs in progress (metrics.json and events.jsonl).
"""
import json
import sys
from collections import Counter
from pathlib import Path

WIN = {"vc33": 7, "sb26": 8, "lp85": 8, "ls20": 7}  # levels to win, from the games' normal frames
ORG = ("plan", "finding", "focus", "define_kind", "adopt", "revise_kind", "propose", "contract", "lead", "challenge",
       "replay", "write", "link", "connect", "task", "design")


def load(d: Path) -> dict:
    out = {"name": d.parent.name, "done": (d / "run.json").exists()}
    metrics = json.loads((d / "metrics.json").read_text())
    out["t"] = metrics[-1]["t"]
    # scores recomputed from first completions and each game's true level count: GAME_OVER frames report "0/0", which
    # the world took as a level count before 2026-10-31 12:20 and which inflated per-tick scores while it lasted
    firsts, games = [], []
    if (d / "run.json").exists():
        w = json.loads((d / "run.json").read_text())["world"]
        firsts, games = w.get("firsts", []), w["games"]
    if games:
        best = {g: 0 for g in games}
        per_tick = []
        for t in range(1, out["t"] + 1):
            for f in firsts:
                if f["t"] == t:
                    best[f["game"]] = max(best[f["game"]], f["level"])
            per_tick.append(sum(best[g] / WIN.get(g.split("-")[0], 8) for g in games) / len(games))
        out["score"], out["auc"] = round(per_tick[-1], 4), sum(per_tick) / len(per_tick)
    else:  # in progress: the logged scores (may be inflated by the 0/0 frames)
        out["score"] = metrics[-1]["score"]
        out["auc"] = sum(m["score"] for m in metrics) / len(metrics)
    out["levels"] = metrics[-1].get("levels")
    out["metrics"] = metrics[-1]
    ops = Counter()
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("ok") and e.get("op") != "intent":
            op = e["op"]
            ops[op.split(".")[1] if op.startswith("lab.") else op.split(".")[0]] += 1
    out["ops"] = ops
    if out["done"]:
        r = json.loads((d / "run.json").read_text())
        w = r["world"]
        out["cost"] = r["meta"].get("cost_usd", 0)
        out["agents"] = len(r["agents"])
        out["best"] = {g.split("-")[0]: w["best"][g] for g in w["games"]}
        out["firsts"] = [(f["game"].split("-")[0], f["level"], f["t"]) for f in w.get("firsts", [])]
        reached = Counter()
        prev = {}
        for e in w["log"]:
            key = (e["agent"], e["game"])
            if e["levels"] > prev.get(key, 0):
                reached[(e["game"].split("-")[0], e["levels"])] += 1
            prev[key] = max(prev.get(key, 0), e["levels"])
        out["copies"] = dict(reached)
        out["paid"] = sorted(((c, round(v[0], 1), v[1]) for c, v in (w.get("combo_credit") or {}).items()), key=lambda x: -x[1])[:5]
        out["rules"] = r["meta"].get("rule_learning")
        out["kinds"] = w.get("kinds")
        out["teams"] = w.get("teams")
    return out


def main(paths):
    runs = []
    for p in paths:
        for f in sorted(Path(p).glob("**/metrics.json")):
            runs.append(load(f.parent))
    print(f"{'run':38s} {'t':>3s} {'score':>6s} {'AUC':>6s} {'lvls':>4s} {'cost':>6s}  first completions (game level tick)")
    for r in runs:
        fl = " ".join(f"{g}{lv}@{t}" for g, lv, t in r.get("firsts", []))
        print(f"{r['name'][:38]:38s} {r['t']:3d} {r['score']:6.3f} {r['auc']:6.3f} {r['levels'] or 0:4d} "
              f"{('$%.2f' % r['cost']) if r.get('cost') else '  -  ':>6s}  {fl}")
    print("\nOffered actions used (successful calls):")
    print(f"{'run':38s} " + " ".join(f"{o[:7]:>7s}" for o in ORG))
    for r in runs:
        print(f"{r['name'][:38]:38s} " + " ".join(f"{r['ops'].get(o, 0):7d}" for o in ORG))
    for r in runs:
        if r.get("paid"):
            print(f"\n{r['name']}: copies per level {r['copies']}; what paid {r['paid']}")
        if r.get("rules"):
            for h in r["rules"]:
                print(f"  rules t{h['t']}: {json.dumps(h['changed'])} because {h['why'][:200]}")
        if r.get("kinds"):
            print(f"  kinds {json.dumps(r['kinds'])[:300]}")
        if r.get("teams"):
            print(f"  teams {json.dumps(r['teams'])[:300]}")


if __name__ == "__main__":
    main(sys.argv[1:])
