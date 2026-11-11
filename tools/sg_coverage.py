#!/usr/bin/env python3
"""Generation vs selection in a finished SciGym run (the 'coverage' idea from Pappu et al. 2026): grade every model any
agent saved for a system and compare the best of them with the model the society ended up with.
  .venv/bin/python tools/sg_coverage.py RUN_DIR...      (no API calls; the grader is local)"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.bench.scigym import DATA, call_worker  # noqa: E402


async def coverage(run: Path) -> dict:
    data = json.loads((run / "run.json").read_text())
    w = data["world"]
    practice = set(w.get("practice", []))
    rows = []
    for t in w["tasks"]:
        files = [p for p in (run / "work" / t["id"]).glob("A*/*.xml")]
        system = DATA / t["family"]

        async def grade(p):
            m = await call_worker({"op": "grade", "dir": str(system), "sbml": p.read_text(), "rounds": 2, "noise": 0.2,
                                   "seed": 0}, 900)
            return float(m.get("rp_f1") or 0.0) if m.get("valid") else 0.0
        scores = await asyncio.gather(*(grade(p) for p in files)) if files else []
        rows.append({"task": t["id"], "practice": t["id"] in practice, "saved_models": len(files),
                     "agents": len({p.parent.name for p in files}), "best_saved": max(scores, default=0.0),
                     "society": t["score"] or 0.0})
    held = [r for r in rows if not r["practice"]]
    n = len(held)
    return {"run": run.name, "condition": run.parent.name, "seed": data["meta"].get("seed"),
            "society": round(sum(r["society"] for r in held) / n, 4), "best_saved": round(sum(r["best_saved"] for r in held) / n, 4),
            "systems_with_models": sum(r["saved_models"] > 0 for r in held), "held_out": n, "rows": rows}


async def main(runs):
    out = [await coverage(Path(r)) for r in runs]
    for o in out:
        print(f"{o['condition']:16s} s{o['seed']}: society {o['society']:.3f}  best saved model per system {o['best_saved']:.3f}  "
              f"(held-out systems with any saved model {o['systems_with_models']}/{o['held_out']})")
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main([a for a in sys.argv[1:] if not a.startswith("--")]))
