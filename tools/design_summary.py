#!/usr/bin/env python3
"""Design-world results by condition as a markdown table (unclamped scores replayed from recorded knob changes).
  .venv/bin/python tools/design_summary.py [runs_design]"""
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_rescore import rescore  # noqa: E402

LABEL = {"single": "single agent", "isolated": "isolated (no communication)", "flat": "flat (no groups)",
         "self_org": "self-organizing", "oracle": "prescribed hierarchy, no rights",
         "oracle+authority": "prescribed hierarchy holding the rights",
         "self_org+knob-holders": "each knob held by one random agent that uses it (no groups)",
         "self_org+ownership": "self-organizing + own_knob primitive",
         "self_org+shared-knob-holders": "each shared knob held by one random agent that uses it; private knobs free",
         "self_org+ownership+reasoning": "self-organizing + own_knob, reasoning on",
         "self_org@40": "self-organizing, 40 ticks", "self_org/14a": "self-organizing, 14 agents (one per block)", "self_org+ownership@40": "self-organizing + own_knob primitive, 40 ticks"}


def tag(data: dict) -> str:
    opts = data["meta"]["config"].get("world_opts") or {}
    t = data["meta"]["condition"] + {"1": "+authority", "agent": "+knob-holders",
                                     "agent_shared": "+shared-knob-holders"}.get(opts.get("authority"), "")
    t += {"group": "+group-ownership"}.get(opts.get("ownership"), "+ownership" if opts.get("ownership") else "")
    t += "+reasoning" if data["meta"].get("reasoning") else ""
    ticks, agents = len(data["snapshots"]) - 1, len(data["agents"])
    return t + (f"@{ticks}" if ticks != 20 else "") + (f"/{agents}a" if agents not in (1, 50) else "")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs_design")
    by = collections.defaultdict(list)
    for d in sorted(root.iterdir()):
        if (d / "run.json").exists():
            data = json.loads((d / "run.json").read_text())
            r = rescore(d)
            r["groups"] = data["snapshots"][-1]["metrics"]["groups"]
            r["claims"] = len(data["world"].get("claims", []))
            r["cost"] = data["meta"].get("cost_usd") or 0
            by[tag(data)].append(r)
    mean = lambda v: sum(v) / len(v)
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
    print("| condition | runs | final score, mean ± sd | second-half mean ± sd | shared-knob reversals | groups | cost/run |")
    print("|---|---|---|---|---|---|---|")
    order = ["single", "isolated", "flat", "self_org", "oracle", "self_org+ownership", "self_org+ownership+reasoning",
             "oracle+authority", "self_org+knob-holders", "self_org+shared-knob-holders", "self_org@40",
             "self_org+ownership@40", "self_org/14a", "self_org+group-ownership"]
    for t in sorted(by, key=lambda t: order.index(t) if t in order else 99):
        rs = by[t]
        fin, half, rev = [r["final"] for r in rs], [r["mean_second_half"] for r in rs], [r["reversals"] for r in rs]
        print(f"| {LABEL.get(t, t)} | {len(rs)} | {mean(fin):+.2f} ± {sd(fin):.2f} | {mean(half):+.2f} ± {sd(half):.2f} | "
              f"{min(rev)}-{max(rev)} | {max(r['groups'] for r in rs)} | ${mean([r['cost'] for r in rs]):.2f} |")


if __name__ == "__main__":
    main()
