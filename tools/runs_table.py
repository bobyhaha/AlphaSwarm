#!/usr/bin/env python3
"""One row per finished run (benchmarks and hidden-law world), as markdown or JSON, straight from run.json files.

  python3 tools/runs_table.py runs_bench runs_hidden            # markdown table
  python3 tools/runs_table.py --json runs_bench > runs.json
"""
import json
import sys
from pathlib import Path


def row(d: Path) -> dict | None:
    f = d / "run.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    meta, world, cfg = data["meta"], data["world"], data["meta"].get("config", {})
    m = data["snapshots"][-1]["metrics"]
    opts = dict(cfg.get("world_opts") or {})
    opts.pop("judge_temperature", None)
    r = {"run": d.name, "world": world.get("benchmark") or world.get("kind") or "hidden", "condition": meta.get("condition"),
         "model": meta.get("model", "").split("/")[-1], "reasoning": bool(meta.get("reasoning")),
         "seed": meta.get("seed"), "agents": len(data["agents"]), "ticks": len(data["snapshots"]) - 1,
         "home": cfg.get("home") if "benchmark" in world else None, "tasks": world.get("n_tasks"),
         "opts": ",".join(f"{k}={v}" for k, v in sorted(opts.items())), "score": m.get("score"),
         "coverage": m.get("coverage"), "groups": m.get("groups"), "depth": m.get("depth"),
         "links": m.get("links"), "cost": meta.get("cost_usd"), "stopped": meta.get("stopped")}
    if "benchmark" in world:
        answered = [t for t in world["tasks"] if t["answer"]]
        r["answered"] = len(answered)
        r["per_answer"] = sum(t["score"] or 0 for t in answered) / len(answered) if answered else None
        judge_t = (cfg.get("world_opts") or {}).get("judge_temperature")
        r["judge"] = f"t{judge_t}" if judge_t is not None else ("official t1" if world["benchmark"] == "discoverybench" else "")
        final = [e for e in world.get("evaluations", []) if e.get("final")]
        r["by_domain"] = final[-1]["by_domain"] if final else None
    else:
        r.update(answered=None, per_answer=None, judge="", by_domain=None, composite=m.get("score_composite"),
                 mirror_l1=m.get("mirror_l1"), mirror_l2=m.get("mirror_l2"), hier_align=m.get("hier_align"))
    return r


def main():
    as_json = "--json" in sys.argv
    roots = [Path(a) for a in sys.argv[1:] if a != "--json"]
    rows = []
    for root in roots:
        dirs = [root] if (root / "run.json").exists() else sorted(p.parent for p in root.glob("**/run.json"))
        rows += [r for r in (row(d) for d in dirs) if r]
    rows.sort(key=lambda r: (r["world"], r["condition"] or "", r["run"]))
    if as_json:
        print(json.dumps(rows, indent=1))
        return
    cols = ["world", "condition", "model", "reasoning", "seed", "agents", "ticks", "home", "opts", "judge", "score",
            "coverage", "answered", "per_answer", "groups", "depth", "cost", "run"]
    fmt = lambda v: "–" if v is None or v == "" else (f"{v:.3f}" if isinstance(v, float) else str(v))
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")


if __name__ == "__main__":
    main()
