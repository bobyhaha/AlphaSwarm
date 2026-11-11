#!/usr/bin/env python3
"""Regrade a finished DiscoveryBench run's final answers with a chosen judge setting.

The official judge samples at temperature 1, so the same answer can score differently between gradings (gold
hypotheses scored against themselves reach 1.0 only about half the time). Comparisons between conditions use
temperature 0, where gold scores 1.0 and another task's gold 0.0 on every checked task.

  OPENROUTER_API_KEY=... .venv/bin/python tools/db_regrade.py RUN_DIR [RUN_DIR...] [--temp 0] [--repeats 1]
Writes RUN_DIR/regrade_t<temp>.json and prints the score (unanswered tasks count 0).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.bench.base import Submission  # noqa: E402
from alphateam.bench.discoverybench import make  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.llm import LLMBackend, OpenAICompatible  # noqa: E402


async def regrade(run: Path, temp: float, repeats: int, backend) -> dict:
    data = json.loads((run / "run.json").read_text())
    cfg = data["meta"]["config"]
    seed = data["meta"].get("seed", cfg.get("seed", 0))
    opts = dict(cfg.get("world_opts") or {}, judge_temperature=str(temp))
    w = make(dict(cfg, world_opts=opts), 1, seed=seed)
    k = Kernel(1)
    k.env = w
    w.llm = lambda *a, **kw: backend.complete(k, *a, **kw)
    saved = {t["id"]: t for t in data["world"]["tasks"]}
    for tid, t in w.tasks.items():
        s = saved[tid]
        assert s["gold"] == t.meta["gold"], f"{run}: task {tid} does not line up with the saved run"
        if s["answer"]:
            a = s["answer"]
            w.submissions[tid] = Submission(tid, a["answer"], a.get("rationale", ""), a["author"], a.get("group"), a["tick"])
    answered = list(w.submissions)
    per = {t: [] for t in answered}
    for _ in range(repeats):
        w._gold_sub.clear()
        got = await w.grade(k, answered)
        for t, v in got.items():
            per[t].append(v)
    scores = {t: sum(v) / len(v) for t, v in per.items() if v}
    by_fam = {}
    for tid, t in w.tasks.items():
        by_fam.setdefault(t.family, []).append(scores.get(tid, 0.0))
    out = {"run": run.name, "condition": data["meta"].get("condition"), "temperature": temp, "repeats": repeats,
           "n_tasks": len(w.tasks), "answered": len(answered), "graded": len(scores),
           "score": sum(scores.values()) / len(w.tasks),
           "per_answer": sum(scores.values()) / len(scores) if scores else None,
           "by_family": {f: round(sum(v) / len(v), 4) for f, v in sorted(by_fam.items())},
           "tasks": {t: per[t] for t in answered}, "judge_cost_usd": k.usage["cost_microusd"] / 1e6}
    (run / f"regrade_t{temp:g}.json").write_text(json.dumps(out, indent=1))
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--repeats", type=int, default=1)
    a = p.parse_args()
    backend = LLMBackend(OpenAICompatible("openrouter"), 32)
    outs = await asyncio.gather(*(regrade(Path(r), a.temp, a.repeats, backend) for r in a.runs))
    for o in outs:
        print(f"{o['run']}: {o['condition']} score {o['score']:.4f} (per answer {o['per_answer'] or 0:.3f}, "
              f"{o['answered']}/{o['n_tasks']} answered, judge ${o['judge_cost_usd']:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
