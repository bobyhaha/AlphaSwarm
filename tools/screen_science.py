#!/usr/bin/env python3
"""How much headroom one agent leaves on FrontierScience: each question tried several times, one reply each, nothing shared.

  OPENROUTER_API_KEY=... .venv/bin/python tools/screen_science.py [trials]

A society can only show a gain where a single agent does not already saturate the task (the user, 2026-11-03). This writes
data/frontierscience/screen.json: for each question, how many one-shot attempts the judge passed. alphateam/science_world.py with
hard=1 keeps the olympiad questions one agent solves at most once in three. The judge is the run's cheap model, so this is a filter,
not a score; selected questions are then scored afresh by the final judge, in fresh attempts.
"""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openai  # noqa: E402

from alphateam.science_world import DATA, OLYMPIAD_JUDGE, RESEARCH_JUDGE, verdict  # noqa: E402

MODEL = "openai/gpt-6-luna"
client = openai.AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=600, max_retries=4)
slots = asyncio.Semaphore(24)
spent = [0.0]


async def ask(prompt: str, effort: str) -> str:
    async with slots:
        r = await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=16000,
                                                 extra_body={"reasoning": {"effort": effort}, "provider": {"ignore": ["openai/flex"]}})
    spent[0] += float((getattr(r.usage, "model_extra", None) or {}).get("cost") or 0)
    return r.choices[0].message.content or ""


async def trial(q: dict) -> float | None:
    answer = await ask(f"Solve this problem. End with your final answer.\n\n{q['problem']}", "low")
    research = q["track"] == "research"
    text = await ask((RESEARCH_JUDGE if research else OLYMPIAD_JUDGE).format(problem=q["problem"], reference=q["answer"], answer=answer[:9000]), "medium")
    return verdict(text, research)


async def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rows = []
    for track in ("olympiad", "research"):
        rows += [{**json.loads(line), "track": track} for line in (DATA / f"{track}.jsonl").read_text().splitlines() if line.strip()]
    results = await asyncio.gather(*(asyncio.gather(*(trial(q) for _ in range(trials)), return_exceptions=True) for q in rows))
    out = {}
    for q, marks in zip(rows, results):
        good = [m for m in marks if isinstance(m, float)]
        out[hashlib.sha1(q["problem"].encode()).hexdigest()[:12]] = {"track": q["track"], "subject": q["subject"], "marks": good}
    (DATA / "screen.json").write_text(json.dumps(out))
    oly = [v["marks"] for v in out.values() if v["track"] == "olympiad"]
    res = [v["marks"] for v in out.values() if v["track"] == "research"]
    print("olympiad one-shot mean", round(sum(map(sum, oly)) / max(1, sum(map(len, oly))), 3), "by questions solved n of", trials,
          {n: sum(sum(m) == n for m in oly) for n in range(trials + 1)})
    print("research mean points", round(10 * sum(map(sum, res)) / max(1, sum(map(len, res))), 2), "ever passed", sum(any(x >= .7 for x in m) for m in res), "of", len(res))
    print("cost", round(spent[0], 3))


asyncio.run(main())
