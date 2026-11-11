#!/usr/bin/env python3
"""Compare DiG-bench arms: a society's best levels per game against the best of the isolated one-agent runs on that game
(a game counts as reached by the isolated arm if any of its runs reached it), with cost and agent-turns.

    python3 tools/dig_compare.py runs_dig/D_pilot_society_s0 runs_dig/D_pilot_isolated_s0
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def runs(folder: Path) -> list[dict]:
    out = []
    for p in sorted(folder.rglob("run.json")):
        d = json.loads(p.read_text())
        if d["world"].get("kind") != "dig":
            continue
        snaps = d.get("snapshots") or []
        m = snaps[-1]["metrics"] if snaps else {}
        cost = (d.get("usage") or {}).get("cost_microusd", 0) / 1e6
        out.append({"dir": str(p.parent), "world": d["world"], "metrics": m, "cost": cost,
                    "turns": len(d.get("agents") or []) * max(0, len(snaps) - 1)})
    return out


def arm(folder: Path) -> dict:
    rs = runs(folder)
    best, win, attempts, firsts = defaultdict(int), {}, defaultdict(int), {}
    for r in rs:
        w = r["world"]
        for g in w["games"]:
            players = sum(1 for gg in w["assigned"].values() if gg == g)
            attempts[g] += players
            if w["best"][g] > best[g]:
                best[g] = w["best"][g]
            win[g] = max(win.get(g, 0), w["win"].get(g, 0))
        for f in w.get("firsts", []):
            key = (f["game"], f["level"])
            firsts[key] = min(firsts.get(key, 10 ** 9), f["t"])
    return {"runs": len(rs), "best": dict(best), "win": win, "attempts": dict(attempts), "firsts": firsts,
            "cost": sum(r["cost"] for r in rs), "turns": sum(r["turns"] for r in rs),
            "tests": sum(r["metrics"].get("tests", 0) for r in rs)}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = arm(Path(sys.argv[1])), arm(Path(sys.argv[2]))
    games = sorted(set(a["best"]) | set(b["best"]), key=lambda g: (len(g), g))
    print(f"{'game':6} {'levels':>6}  {'society':>16}  {'isolated (best of runs)':>24}")
    for g in games:
        w = a["win"].get(g) or b["win"].get(g, "?")
        ta = a["firsts"].get((g, a["best"].get(g, 0)), "-")
        tb = b["firsts"].get((g, b["best"].get(g, 0)), "-")
        print(f"{g:6} {w!s:>6}  {a['best'].get(g, 0):>4} (t{ta}, {a['attempts'].get(g, 0)} agents)  "
              f"{b['best'].get(g, 0):>4} (t{tb}, {b['attempts'].get(g, 0)} runs)")
    score = lambda x: sum(x["best"].get(g, 0) / max(1, (a["win"].get(g) or b["win"].get(g) or 1)) for g in games) / max(1, len(games))
    for name, x in (("society", a), ("isolated", b)):
        print(f"{name:9} score {score(x):.3f} · levels {sum(x['best'].values())} · {x['runs']} run(s) · "
              f"{x['turns']} agent-turns · ${x['cost']:.2f} · tests {x['tests']}")


if __name__ == "__main__":
    main()
