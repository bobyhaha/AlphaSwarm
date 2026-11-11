#!/usr/bin/env python3
"""Organizational learning in the design world: each lineage episode vs a fresh society on the same world seed.
  .venv/bin/python tools/lineage_compare.py [runs_lineage]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_rescore import rescore  # noqa: E402


def facts(d: Path) -> dict:
    data = json.loads((d / "run.json").read_text())
    r = rescore(d)
    claims = data["world"].get("claims", [])
    shared = [c for c in claims if c["level"] != "private"]
    return {"seed": data["meta"].get("seed"), "episode": data["meta"].get("episode", 1),
            "inherited": bool(data["meta"].get("inherited")), "final": r["final"], "half": r["mean_second_half"],
            "reversals": r["reversals"], "claims": len(claims), "first_shared_claim": min((c["t"] for c in shared), default=None),
            "library_in": sum(1 for line in open(d / "entries.jsonl") if json.loads(line).get("author") == "heritage")}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs_lineage")
    rows = [facts(p.parent) for p in sorted(root.glob("**/run.json"))]
    fresh = {r["seed"]: r for r in rows if not r["inherited"]}
    print("| seed | episode | inherited | final | second half | reversals | claims | first shared claim | fresh society on same seed: second half / claims |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["seed"] % 1000, r["episode"], r["inherited"])):
        if not r["inherited"] and r["episode"] == 1 and r["seed"] >= 1000:
            continue  # controls are shown beside the episode they match
        c = fresh.get(r["seed"]) if r["inherited"] else None
        ctrl = f"{c['half']:+.2f} / {c['claims']}" if c else "–"
        print(f"| {r['seed']} | {r['episode']} | {'yes' if r['inherited'] else 'no'} | {r['final']:+.2f} | {r['half']:+.2f} | "
              f"{r['reversals']} | {r['claims']} | {r['first_shared_claim'] if r['first_shared_claim'] is not None else '–'} | {ctrl} |")


if __name__ == "__main__":
    main()
