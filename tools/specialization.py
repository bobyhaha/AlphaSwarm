#!/usr/bin/env python3
"""Informal division of labor in a benchmark run, from what agents actually worked on (not from groups).

For each agent: the tasks it ran tools on or answered, and their families (e.g. DiscoveryBench datasets).
- purity: share of an agent's task-work in its most-worked family (1 = specialist), averaged over agents
- families/agent: distinct families an agent touched
- work mirroring: P(two agents share a task family focus | ...) is approximated by the family-level mirroring
  index on each agent's dominant family vs. a random baseline with the same family sizes.
- overlap: agents per task (duplication)
  python3 tools/specialization.py RUN_DIR...
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORK = ("lab.run_code", "lab.submit", "lab.experiment", "lab.open_task")


def analyze(d: Path) -> dict:
    data = json.loads((d / "run.json").read_text())
    fam = {t["id"]: t["family"] for t in data["world"]["tasks"]}
    by_agent = defaultdict(Counter)
    per_task = defaultdict(set)
    for line in open(d / "events.jsonl"):
        e = json.loads(line)
        if e["ok"] and e["op"] in WORK[:3] and e.get("target") in fam:
            by_agent[e["agent"]][fam[e["target"]]] += 1
            per_task[e["target"]].add(e["agent"])
    if not by_agent:
        return {"run": d.name}
    purity = [max(c.values()) / sum(c.values()) for c in by_agent.values()]
    nfam = [len(c) for c in by_agent.values()]
    # baseline: same number of work events per agent, families drawn in proportion to the tasks they worked on
    pool = [fam[t] for t, ags in per_task.items() for _ in ags]
    rng = random.Random(0)
    base = []
    for c in by_agent.values():
        n = sum(c.values())
        draw = Counter(rng.choice(pool) for _ in range(n))
        base.append(max(draw.values()) / n)
    return {"run": d.name, "cond": data["meta"].get("condition"), "home": data["meta"]["config"].get("home"),
            "agents_working": len(by_agent), "purity": round(sum(purity) / len(purity), 3),
            "purity_random": round(sum(base) / len(base), 3), "families_per_agent": round(sum(nfam) / len(nfam), 2),
            "agents_per_task": round(sum(len(v) for v in per_task.values()) / len(per_task), 2),
            "tasks_worked": len(per_task)}


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if (Path(a) / "run.json").exists():
            print(analyze(Path(a)))
