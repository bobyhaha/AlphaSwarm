#!/usr/bin/env python3
"""Compare finished alphateam runs side by side.

  python3 tools/compare.py runs_colony/*discoverybench*        # table of all matching run dirs
  python3 tools/compare.py --md runs_colony/A runs_colony/B    # markdown table
"""
import json
import sys
from pathlib import Path


def row(d: Path) -> dict | None:
    f = d / "run.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    meta, world = data["meta"], data["world"]
    m = data["snapshots"][-1]["metrics"]
    toks = {}
    for s in data["snapshots"]:
        for key, v in s["metrics"].items():
            if key.startswith("tok_"):
                toks[key[4:]] = toks.get(key[4:], 0) + (v or 0)
    total = sum(toks.values()) or 1
    final = [e for e in world.get("evaluations", []) if e.get("final")]
    return {"run": d.name, "world": world.get("benchmark", "hidden"), "cond": meta.get("condition"),
            "agents": len(data["agents"]), "ticks": len(data["snapshots"]) - 1, "score": m.get("score"),
            "cov": m.get("coverage"), "cost": meta.get("cost_usd"), "groups": m.get("groups"), "depth": m.get("depth"),
            "super": m.get("supergroups"), "in_group": m.get("in_group"), "links": m.get("links"),
            "mirror_l1": m.get("mirror_l1"), "mirror_l2": m.get("mirror_l2"),
            "comm%": round(100 * toks.get("communication", 0) / total, 1),
            "read%": round(100 * toks.get("read", 0) / total, 1),
            "delib%": round(100 * toks.get("deliberation", 0) / total, 1),
            "by_domain": final[-1]["by_domain"] if final else None, "stopped": meta.get("stopped")}


def main():
    md = "--md" in sys.argv
    dirs = [Path(a) for a in sys.argv[1:] if a != "--md"]
    rows = [r for r in (row(d) for d in dirs) if r]
    if not rows:
        sys.exit("no finished runs (run.json) in the given directories")
    cols = ["run", "world", "cond", "agents", "ticks", "score", "cov", "cost", "groups", "depth", "super", "in_group",
            "links", "mirror_l1", "mirror_l2", "delib%", "comm%", "read%"]
    fmt = lambda v: "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    if md:
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    else:
        w = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
        print("  ".join(c.ljust(w[c]) for c in cols))
        for r in rows:
            print("  ".join(fmt(r[c]).ljust(w[c]) for c in cols))
    for r in rows:
        if r["by_domain"] or r["stopped"]:
            print(f"{r['run']}: " + (", ".join(f"{d} {v:.3f}" for d, v in sorted(r["by_domain"].items())) if r["by_domain"] else "")
                  + (f" [stopped: {r['stopped']}]" if r["stopped"] else ""))


if __name__ == "__main__":
    main()
