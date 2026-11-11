#!/usr/bin/env python3
"""Incentive experiments: score and how much each society organized, one row per run, then means per condition.

  python3 tools/incentive_summary.py runs_incentives            # markdown tables
  python3 tools/incentive_summary.py runs_incentives --json

Organizing actions are connect, link, protocol, delegate and design_action calls that succeeded. 'Mentions' counts
agents' own posts that talk about the incentive signal (credit/ledger in the design world, practice grades in benchmarks).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ORG = ("connect.", "link.", "protocol", "delegate.", "design.")


def second_half(snaps: list[dict]) -> float | None:
    vals = [s["metrics"]["score"] for s in snaps[1:] if s["metrics"].get("score") is not None]
    vals = vals[len(vals) // 2:]
    return sum(vals) / len(vals) if vals else None


def row(d: Path, root: Path) -> dict:
    data = json.loads((d / "run.json").read_text())
    meta, world, snaps = data["meta"], data["world"], data["snapshots"]
    rel = d.relative_to(root).parts
    ops, failed = Counter(), 0
    with open(d / "events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e.get("ok") and e["op"].startswith(ORG):
                ops[e["op"].split(".")[0]] += 1
            failed += not e.get("ok")
    signal = re.compile(r"credit|ledger" if world.get("kind") == "design" else r"practice", re.I)
    mentions = 0
    with open(d / "entries.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if re.fullmatch(r"A\d+", e["author"]) and e["board"] not in ("",) and signal.search(e["content"]):
                mentions += 1
    m = snaps[-1]["metrics"]
    r = {"condition": rel[0], "run": "/".join(rel[1:]), "seed": meta.get("seed"), "episode": meta.get("episode"),
         "society": meta.get("society"), "experiment": meta.get("experiment"), "score": m.get("score"),
         "groups_final": m.get("groups"), "groups_max": max(s["metrics"]["groups"] for s in snaps),
         "depth_max": max(s["metrics"]["depth"] for s in snaps), "org_actions": sum(ops.values()),
         "connect": ops["connect"], "link": ops["link"], "protocol": ops["protocol"], "delegate": ops["delegate"],
         "mentions": mentions, "cost": meta.get("cost_usd"), "stopped": meta.get("stopped")}
    if world.get("kind") == "design":
        claims = world.get("claims", [])
        r |= {"second_half": second_half(snaps), "group_claims": sum(c["holder"].startswith("G") for c in claims),
              "agent_claims": sum(c["holder"].startswith("A") for c in claims),
              "sets_for_groups": sum(bool(s["group"]) for s in world["sets"]), "knob_sets": len(world["sets"]),
              "shared_reversals": m.get("shared_reversals")}
    else:
        final = [e for e in world.get("evaluations", []) if e.get("final")]
        ev = final[-1] if final else {}
        r |= {"practice": ev.get("practice"), "held_out_answered": ev.get("held_out_answered"),
              "answered": ev.get("answered"), "practice_reviews": [x["mean"] for x in world.get("practice_reviews", [])]}
    return r


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    as_json = "--json" in sys.argv
    roots = [Path(a) for a in sys.argv[1:] if not a.startswith("--")] or [Path("runs_incentives")]
    rows = []
    for root in roots:
        for f in sorted(root.glob("**/run.json")):
            rows.append(row(f.parent, root))
    if as_json:
        print(json.dumps(rows, indent=1))
        return
    fmt = lambda v: "–" if v is None or v == "" else (f"{v:+.3f}" if isinstance(v, float) else str(v))
    for kind in ("design", "bench"):
        sub = [r for r in rows if ("second_half" in r) == (kind == "design")]
        if not sub:
            continue
        cols = (["condition", "run", "experiment", "score", "second_half", "groups_max", "depth_max", "org_actions",
                 "group_claims", "agent_claims", "sets_for_groups", "shared_reversals", "mentions", "cost"] if kind == "design"
                else ["condition", "run", "experiment", "score", "practice", "held_out_answered", "groups_max", "depth_max",
                      "org_actions", "mentions", "cost"])
        print(f"\n### {kind}: runs\n")
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for r in sub:
            vals = [(r.get(c) or "")[:60] if c == "experiment" and r.get(c) else r.get(c) for c in cols]
            print("| " + " | ".join(fmt(v) for v in vals) + " |")
        groups = defaultdict(list)
        for r in sub:
            key = r["condition"] + (f" ep{r['episode']}" if r["episode"] and "selection" in r["condition"] else "")
            groups[key].append(r)
        num = [c for c in cols if c not in ("condition", "run", "experiment")]
        print(f"\n### {kind}: means\n")
        print("| condition | n | " + " | ".join(num) + " |\n|" + "---|" * (len(num) + 2))
        for key, rs in sorted(groups.items()):
            print(f"| {key} | {len(rs)} | " + " | ".join(fmt(mean([r.get(c) for r in rs])) for c in num) + " |")


if __name__ == "__main__":
    main()
