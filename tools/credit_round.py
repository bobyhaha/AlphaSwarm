#!/usr/bin/env python3
"""One line per credit-design run (single seed): score, structure, claims, information flow.
  python3 tools/credit_round.py runs_credit runs_incentives/design_calls_control runs_incentives/design_calls_continual runs_incentives/design_calls_durable
"""
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(d: Path) -> str:
    data = json.loads((d / "run.json").read_text())
    w, snaps, meta = data["world"], data["snapshots"], data["meta"]
    sc = [s["metrics"]["score"] for s in snaps[1:]]
    half = sum(sc[len(sc) // 2:]) / max(1, len(sc) - len(sc) // 2)
    ops = Counter()
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e.get("ok") and e["op"] != "intent":
            ops[e["op"].split(".")[0] if not e["op"].startswith("lab.") else e["op"]] += 1
    sims_shared = 0
    for line in open(d / "entries.jsonl"):
        e = json.loads(line)
        if e["content"].startswith("Simulation of") and not e["board"].startswith("private"):
            sims_shared += 1
    claims = w.get("claims", [])
    groups_max = max(s["metrics"]["groups"] for s in snaps)
    dc = w.get("durable_credit", {})
    group_credit = sum(v[0] for k, v in dc.items() if k.startswith("G"))
    return (f"{d.parent.name}/{d.name[:2] if d.parent.name == 'runs_credit' else meta.get('seed')}: seed {meta.get('seed')} "
            f"2nd-half {half:+.3f} final {sc[-1]:+.3f} | groups max {groups_max} | claims self {sum(c['holder'].startswith('A') for c in claims)} "
            f"group {sum(c['holder'].startswith('G') for c in claims)} | sets {len(w['sets'])} for groups {sum(bool(s['group']) for s in w['sets'])} "
            f"rev {snaps[-1]['metrics'].get('shared_reversals')} | writes {ops['write']} links {ops['link']} connect {ops['connect']} "
            f"| sims posted publicly {sims_shared} | group credit {group_credit:+.3f} | ${meta.get('cost_usd', 0):.2f}")


for root in sys.argv[1:]:
    for f in sorted(Path(root).glob("**/run.json")):
        if "seed 0" in summarize(f.parent) or True:
            print(summarize(f.parent))
