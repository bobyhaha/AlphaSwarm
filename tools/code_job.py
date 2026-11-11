#!/usr/bin/env python3
"""One code-world run in a process of its own (alphateam/code_runs.py starts these).

  .venv/bin/python tools/code_job.py RUN_DIR/job.json

Reads the job (frame, recipe, seed, budget, which backend), runs the society, and leaves result.json in the run's
directory. The OpenRouter key, when the backend needs one, comes from the environment and is never written anywhere.
"""
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(job["out"])
    from alphateam.paired import run_once
    from alphateam.recipe import Recipe
    recipe = Recipe.from_json(job["recipe"]) if job.get("recipe") else None
    spec = job.get("backend") or {"kind": "mock"}
    if spec.get("kind", "mock") == "mock":
        from alphateam.mock import MockBackend
        backend = MockBackend(job["seed"], proposes=spec.get("proposes"))
    else:
        from alphateam.llm import LLMBackend, OpenAICompatible
        provider = OpenAICompatible("openrouter", spec["model"], False, run_id=time.strftime("%Y%m%d-%H%M%S"),
                                    max_tokens=spec.get("max_tokens", 16384), reasoning_effort=spec.get("effort", "low"),
                                    provider_ignore=spec.get("provider_ignore", ["openai/flex"]))
        backend = LLMBackend(provider, spec.get("concurrency", 128))
    try:
        r = asyncio.run(run_once(job["frame"], recipe, backend, job["seed"], out, job.get("purpose", "run"), job.get("ticks"),
                                 meta=dict(job.get("meta") or {}), cost_budget=job.get("cost_budget")))
    except BaseException as exc:
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.json").write_text(json.dumps({"error": f"{exc.__class__.__name__}: {exc}",
                                                     "trace": traceback.format_exc()[-4000:]}))
        raise
    print(f"done: held-out score {r.score}, hidden {r.hidden}, cost {r.cost}, stopped {r.stopped}", flush=True)


if __name__ == "__main__":
    main()
