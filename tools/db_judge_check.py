#!/usr/bin/env python3
"""Sanity check for the DiscoveryBench judge: gold hypotheses submitted as answers should score near 1,
and an unrelated answer near 0.  OPENROUTER_API_KEY=... .venv/bin/python tools/db_judge_check.py [N]"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.bench.base import Submission  # noqa: E402
from alphateam.bench.discoverybench import make  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.llm import LLMBackend, OpenAICompatible  # noqa: E402


async def main(n: int, temp: str):
    w = make({"tasks": 40, "home": 1, "eval_at": [], "world_opts": {"judge_temperature": temp}}, 1, seed=0)
    k = Kernel(1)
    k.env = w
    backend = LLMBackend(OpenAICompatible("openrouter"), 16)
    w.llm = lambda *a, **kw: backend.complete(k, *a, **kw)
    ids = list(w.tasks)[:n]
    for i, t in enumerate(ids):  # gold for all; a wrong answer (another task's gold) as control for one
        answer = w.tasks[t].meta["gold"]
        w.submissions[t] = Submission(t, answer, "", "A1", None, 0)
    gold = await w.grade(k, ids)
    swapped = {t: Submission(t, w.tasks[ids[(i + 1) % n]].meta["gold"], "", "A1", None, 1) for i, t in enumerate(ids)}
    w.submissions.update(swapped)
    w._gold_sub = w._gold_sub  # keep cached gold sub-hypotheses
    wrong = await w.grade(k, ids)
    for t in ids:
        print(f"{t} {w.tasks[t].family:45s} gold-as-answer {gold[t]:.2f}  other-task-gold {wrong[t]:.2f}")
    print(f"mean gold {sum(gold.values()) / n:.3f}, mean wrong {sum(wrong.values()) / n:.3f}, "
          f"cost ${k.usage['cost_microusd'] / 1e6:.2f}")


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6, sys.argv[2] if len(sys.argv) > 2 else "1.0"))
