#!/usr/bin/env python3
"""How broadly did each ARC society explore? From the event log (works for every condition, finished or not):
distinct moves tried per (game, level), with clicks binned into 8x8 regions, and how much copies on the same game
overlapped (the share of a copy's distinct moves that another copy on that game also tried at that level).
Event args are clipped, so long plays are undercounted the same way in every condition.
  python3 tools/arc_diversity.py runs_arc/R4_* runs_arc/V_*
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CLICK = re.compile(r"click(?:\(|\W+\[)\s*(\d+)\s*,\s*(\d+)", re.I)  # click(x,y) or {"click": [x, y]}
PLAIN = re.compile(r"ACTION[1-57]|RESET", re.I)
LEVEL = re.compile(r"level (\d+)/\d+")


def moves(args: dict) -> list[str]:
    text = str(args.get("actions") or "")
    out = [f"r{int(y) // 8}c{int(x) // 8}" for x, y in CLICK.findall(text)]
    out += [m.upper() for m in PLAIN.findall(CLICK.sub("", text))]
    return out


def analyze(d: Path) -> dict:
    assigned = {}
    run = d / "run.json"
    if run.exists():
        assigned = json.loads(run.read_text())["world"].get("assigned", {})
    tried = defaultdict(lambda: defaultdict(set))  # (game, level) -> agent -> moves
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("op") not in ("lab.play",) or not e.get("ok"):
            continue
        msg = e.get("msg", "")
        game = msg.split(",")[0].split()[-1] if msg.startswith("ran ") else msg.split(",")[0]
        lv = LEVEL.findall(msg)
        level = int(lv[0]) if lv else 0
        tried[(game, level)][e["agent"]].update(moves(e.get("args") or {}))
    rows = {}
    for (game, level), per in tried.items():
        union = set().union(*per.values())
        overlaps = []
        for a, m in per.items():
            others = set().union(*(v for b, v in per.items() if b != a)) if len(per) > 1 else set()
            if m and len(per) > 1:
                overlaps.append(len(m & others) / len(m))
        rows[(game, level)] = {"agents": len(per), "distinct": len(union),
                               "overlap": round(sum(overlaps) / len(overlaps), 2) if overlaps else None}
    return rows


def main(paths):
    for p in paths:
        for f in sorted(Path(p).glob("**/events.jsonl")):
            rows = analyze(f.parent)
            total = sum(r["distinct"] for r in rows.values())
            ov = [r["overlap"] for r in rows.values() if r["overlap"] is not None]
            print(f"{f.parent.parent.name[:36]:36s} distinct (game,level,move) {total:4d}; mean overlap between copies "
                  f"{(sum(ov) / len(ov)) if ov else float('nan'):.2f}; " +
                  " ".join(f"{g}L{lv + 1}:{r['distinct']}/{r['agents']}a" for (g, lv), r in sorted(rows.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
