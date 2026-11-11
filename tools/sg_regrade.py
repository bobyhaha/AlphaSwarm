#!/usr/bin/env python3
"""Regrade a finished SciGym run, filling systems nobody submitted with the newest final_sbml.xml from the run's
agent workspaces (as SciGym's controller asks for a final model when iterations run out).
  .venv/bin/python tools/sg_regrade.py RUN_DIR...      (no API calls)"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.bench.base import Submission  # noqa: E402
from alphateam.bench.scigym import make  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402


async def regrade(run: Path):
    data = json.loads((run / "run.json").read_text())
    cfg, meta = data["meta"]["config"], data["meta"]
    w = make(cfg, len(data["agents"]) if meta.get("condition") != "single" else 1, seed=meta.get("seed", 0),
             single=meta.get("condition") == "single")
    w.workdir = str(run / "work")
    k = Kernel(1)
    k.env = w
    k.tick = len(data["snapshots"]) - 1
    for row in data["world"]["tasks"]:
        if row["answer"]:  # the export clips answers, so take the full model from the agent's workspace history
            files = sorted((run / "work" / row["id"]).glob(f"{row['answer']['author']}/*.xml"), key=lambda p: -p.stat().st_mtime)
            sbml = next((p.read_text() for p in files if p.name != "input.xml"), None)
            if sbml:
                w.submissions[row["id"]] = Submission(row["id"], sbml, "", row["answer"]["author"], None, row["answer"]["tick"])
    before = len(w.submissions)
    await w.finalize(k)
    m = w.score()
    print(f"{run.name}: score {m['score']:.4f} coverage {m['coverage']:.3f} (submitted {before}, auto-filled "
          f"{len(w.auto_submitted)}; original score {data['snapshots'][-1]['metrics']['score']})")


for r in sys.argv[1:]:
    asyncio.run(regrade(Path(r)))
