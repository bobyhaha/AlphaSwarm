#!/usr/bin/env python3
"""AlphaTeam runs on xsv against the ten-agent baseline and the single agent: accuracy by compute, by rounds, and the society's shape.

  python3 tools/xsv_compare.py [runs_xsv]

Reads run.json of AlphaTeam runs and events.jsonl and evaluation.json of te-next runs. Never opens native logs or *.eval.json.
Latency here means rounds in parallel: a AlphaTeam tick is three replies per agent, a te-next tick one action per agent, and a
single agent has no parallelism, so its rounds are its actions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TN = Path("/Users/baiyu/asp-runs/te-next/runs")


def events(run: Path) -> list[dict]:
    out = []
    for line in (run / "events.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def te_next(run: Path) -> dict:
    ev = events(run)
    n, curve = 0, []
    for e in ev:
        n += e.get("event") == "tool"
        if e.get("event") == "validation":
            curve.append((n, e.get("passed")))
    hidden = None
    if (run / "evaluation.json").exists():
        hidden = json.loads((run / "evaluation.json").read_text()).get("passed")
    return {"run": run.name, "actions": n, "curve": curve, "hidden": hidden}


def alphateam(path: Path) -> dict:
    d = json.loads(path.read_text())
    w, last, meta = d["world"], d["snapshots"][-1]["metrics"], d["meta"]
    s = w.get("structure_measures", {})
    return {"run": path.parent.name if path.parent.name.startswith("ep") else path.parent.parent.name, "goal": w["law"]["goal"], "hidden": w["hidden"].get("passed"),
            "held": [(h["t"], h["passed"]) for h in w["held_out"]], "cost": meta.get("cost_usd"), "merges": len(w["merges"]),
            "rejected": len(w["rejections"]), "law": last.get("law_version"), "calls": d["usage"].get("calls"), "tokens": last.get("tokens"),
            "builders": s.get("builders"), "top": s.get("top_builder_share"), "work": s.get("working_share"), "org": s.get("organizing_share"),
            "roles": s.get("roles_taken"), "teams": s.get("teams"), "depth": last.get("depth"), "left": s.get("work_not_at_head"),
            "tests": last.get("tests"), "claims": last.get("claims"), "props": last.get("proposals"), "tools": last.get("tools"),
            "tool_uses": last.get("tool_uses_by_others"), "contracts": last.get("contracts"), "reads": s.get("reads")}


def main():
    base = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs_xsv")
    c0 = [te_next(TN / "final" / f"c0-s{s}") for s in range(800, 806)]
    print("c0 (ten agents, te-next), hidden:", [r["hidden"] for r in c0], "held-out by 120 actions:")
    for r in c0:
        print("  ", r["run"], r["curve"])
    for run in sorted((TN / "sat").glob("x*-s*")):
        r = te_next(run)
        print(f"{r['run']} (single agent): {r['actions']} actions, hidden {r['hidden']}, held-out {r['curve']}")
    print()
    for path in sorted(base.glob("**/run.json")):
        r = alphateam(path)
        print(f"{r['run']}: goal {r['goal']}, hidden {r['hidden']}, ${r['cost']}, {r['calls']} calls, {r['tokens']:,} tokens, {r['merges']} merges, {r['rejected']} rejected, law v{r['law']}")
        print("   held-out:", r["held"])
        print(f"   shape: {r['builders']} builders, top {r['top']}, work {r['work']}, organizing {r['org']}, roles {r['roles']}, teams {r['teams']}, depth {r['depth']}, "
              f"tests {r['tests']}, claims {r['claims']}, proposals {r['props']}, tools {r['tools']} ({r['tool_uses']} uses), contracts {r['contracts']}, reads {r['reads']}, files never assembled {r['left']}")


if __name__ == "__main__":
    main()
