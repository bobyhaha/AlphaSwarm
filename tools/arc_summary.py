#!/usr/bin/env python3
"""ARC-AGI-3 runs: who completed which level when, and whether solutions spread through posts.
  python3 tools/arc_summary.py runs_arc
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def summarize(d: Path):
    data = json.loads((d / "run.json").read_text())
    w, meta = data["world"], data["meta"]
    print(f"\n== {d.parent.name}/{d.name}  condition {meta.get('condition')}  score {data['snapshots'][-1]['metrics']['score']}  "
          f"cost ${meta.get('cost_usd', 0):.2f}")
    print("  best:", {g.split('-')[0]: f"{w['best'][g]}/{w['win'].get(g)}" for g in w["games"]})
    first = defaultdict(dict)  # game -> level -> [(tick, agent)]
    for e in w["log"]:
        pass
    reached = defaultdict(list)
    prev = {}
    for e in w["log"]:
        key = (e["agent"], e["game"])
        if e["levels"] > prev.get(key, 0):
            reached[(e["game"].split("-")[0], e["levels"])].append((e["t"], e["agent"]))
        prev[key] = max(prev.get(key, 0), e["levels"])
    for (g, lv), who in sorted(reached.items()):
        print(f"  {g} level {lv}: {len(who)} copies, " + ", ".join(f"{a} t{t}" for t, a in sorted(who)))
    entries = [json.loads(l) for l in open(d / "entries.jsonl")]
    shared = [e for e in entries if re.fullmatch(r"A\d+", e["author"]) and not e["board"].startswith("private")
              and re.search(r"level|complete|click|action", e["content"], re.I)]
    print(f"  agent posts on shared boards about levels/actions: {len(shared)}; citing posts: "
          f"{sum(bool(e.get('refs')) for e in shared)}")
    for e in shared[:5]:
        print(f"    t{e['tick']} {e['author']} [{e['board']}]: {e['content'][:200]}")


for root in sys.argv[1:]:
    for f in sorted(Path(root).glob("**/run.json")):
        summarize(f.parent)
