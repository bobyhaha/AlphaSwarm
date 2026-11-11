#!/usr/bin/env python3
"""Unclamped design-world score per tick for finished runs, by replaying their recorded knob changes.
(Early design runs reported scores clamped at 0, which hid designs worse than the start.)
  .venv/bin/python tools/design_rescore.py runs_design/RUN...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.runner import build  # noqa: E402


def rescore(run: Path) -> dict:
    data = json.loads((run / "run.json").read_text())
    meta, w0 = data["meta"], data["world"]
    k = build(meta["config"], meta.get("seed", 0), "design")
    env = k.env
    idx = {n: i for i, n in enumerate(env.name)}
    x = np.array([kn["start"] for kn in w0["knobs"]])
    assert np.allclose(x, env.x_start) and abs(env.c_start - w0["c_start"]) < 1e-6, f"{run}: world does not line up"
    ticks = len(data["snapshots"]) - 1
    by_tick, j, sets = [], 0, w0["sets"]
    for t in range(0, ticks + 1):
        while j < len(sets) and sets[j]["t"] <= t:
            x[idx[sets[j]["knob"]]] = sets[j]["to"]
            j += 1
        by_tick.append(round(env.normalized(x), 3))
    half = by_tick[ticks // 2 + 1:]
    return {"run": run.name, "condition": meta["condition"], "seed": meta.get("seed"), "final": by_tick[-1],
            "mean_second_half": round(sum(half) / len(half), 3), "clamped_feedback": "score_clamped" not in data["snapshots"][-1]["metrics"],
            "best": max(by_tick), "min": min(by_tick), "by_tick": by_tick, "sets": len(sets),
            "reversals": sum(s["reversal"] for s in sets if s["level"] != "private")}


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if (Path(a) / "run.json").exists():
            r = rescore(Path(a))
            print(f"{r['run']}: {r['condition']} seed {r['seed']} final {r['final']:+.3f} 2nd-half mean {r['mean_second_half']:+.3f} "
                  f"best {r['best']:+.3f} "
                  f"min {r['min']:+.3f} sets {r['sets']} shared reversals {r['reversals']}"
                  + (" (clamped feedback)" if r["clamped_feedback"] else "") + f"\n   {r['by_tick']}")
