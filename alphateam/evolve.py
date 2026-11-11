"""Organizational learning across episodes.

Two improvement mechanisms, kept separate so they can be compared:

- Experience (inside the society): each episode faces a new instance of the world, and starts with the
  previous society's culture: the actions it invented, its library, its group protocols and its lessons.
  Nothing outside the agents decides what to keep; they wrote it.
- Selection (outside the society, optional): run a population of societies per generation and pass on the
  culture of the best-scoring one. This is an external optimizer, useful as a comparison.

With a population, every society of a generation faces the same world instance (so their scores are comparable) and
is told that only the best one's culture survives. After each generation a review compares the societies and proposes
new organizational experiments: in the next generation society 1 carries on the selected culture unchanged and each
other society is assigned one experiment, so a lineage keeps varying how it organizes instead of copying one habit.

`--structure` may list several world structures (e.g. hierarchical,flat); episodes cycle through them, which
tests whether a society learns when hierarchy helps and when to abandon it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .kernel import clip
from .llm import cost_usd
from .runner import build, inherit, run

STAKES = ("SELECTION: {p} societies, each with its own agents, work on this same {what} in parallel. The society with "
          "the best final score passes its culture (library, custom actions, group protocols and lessons) to the next "
          "generation; the other societies' cultures are discarded.")


def second_half(data: dict) -> float | None:
    """Mean score over the second half of the run (the design world's score moves a lot from tick to tick)."""
    vals = [s["metrics"]["score"] for s in data["snapshots"][1:] if s["metrics"].get("score") is not None]
    vals = vals[len(vals) // 2:]
    return round(sum(vals) / len(vals), 4) if vals else None


def org_stats(data: dict) -> dict:
    """How a society organized, for the generation review: structure metrics and world-specific decision records."""
    last = data["snapshots"][-1]["metrics"]
    keys = ("groups", "depth", "in_group", "links", "reorganizations", "protocol_changes", "custom", "sets_for_groups",
            "knobs_held_by_groups", "knobs_held_by_agents", "shared_reversals", "coverage")
    return {x: last[x] for x in keys if last.get(x) is not None}


async def lineage(cfg: dict, backend, out_root: Path, meta: dict, start_culture: dict | None = None) -> list[dict]:
    structures = cfg["structures"]
    episodes, population = cfg["episodes"], cfg["population"]
    nested = episodes > 1 or population > 1
    what = "design" if cfg.get("world") == "design" else "benchmark instance" if cfg.get("world", "hidden") != "hidden" else "world"
    culture = start_culture
    experiments: list[str] = []  # assigned to societies 2..P of the next generation
    summary, reviews = [], []
    for ep in range(episodes):
        structure = structures[ep % len(structures)]
        world_seed = cfg["seed"] + 1000 * ep  # every society of a generation faces the same instance

        async def one(i: int):
            k = build(cfg, world_seed, structure)
            if culture:
                inherit(k, culture)
            exp = experiments[i - 1] if 0 < i <= len(experiments) else None
            if population > 1:
                k.stakes = STAKES.format(p=population, what=what)
                if exp:
                    k.stakes += ("\nYOUR SOCIETY'S ORGANIZATIONAL EXPERIMENT (assigned by the review of the previous "
                                 f"generation; the other societies try different ones): {exp}")
                    k.post("global", "heritage", "proposal", f"Organizational experiment for this society: {exp}", notify=False)
                elif experiments:
                    k.stakes += ("\nYour society carries on the previous generation's best culture without a new "
                                 "experiment; each of the other societies tries one.")
            out = out_root / (f"ep{ep + 1:02d}-soc{i + 1:02d}" if nested else "")
            m = dict(meta, episode=ep + 1, society=i + 1, structure=structure, seed=world_seed + i, world_seed=world_seed,
                     inherited=bool(culture), experiment=exp)
            data = await run(k, backend, cfg["ticks"], out, m, world_seed + i, quiet=population > 1)
            return out, data, k

        results = await asyncio.gather(*(one(i) for i in range(population)))
        ranked = sorted(results, key=lambda r: -(r[1]["snapshots"][-1]["metrics"]["score"] or 0))
        for out, data, _ in results:
            last = data["snapshots"][-1]["metrics"]
            summary.append({"episode": ep + 1, "society": data["meta"]["society"], "structure": structure, "dir": str(out),
                            "score": last["score"], "second_half": second_half(data), "coverage": last["coverage"],
                            "depth": last["depth"], "groups": last["groups"], "supergroups": last["supergroups"],
                            "mirror_l1": last.get("mirror_l1"), "mirror_l2": last.get("mirror_l2"), "tokens": last["tokens"],
                            "cost_usd": data["meta"].get("cost_usd"), "experiment": data["meta"].get("experiment"),
                            "selected": out == ranked[0][0], "stopped": data["meta"].get("stopped")})
        best_out, best, best_k = ranked[0]
        print(f"episode {ep + 1} ({structure}): scores " + ", ".join(str(r[1]['snapshots'][-1]['metrics']['score']) for r in results)
              + (f"; passing on the culture of {best_out.name}" if episodes > 1 else ""), flush=True)
        if any(r[1]["meta"].get("stopped") for r in results):
            break
        culture = best["culture"]
        experiments = []
        if population > 1 and ep + 1 < episodes and hasattr(backend, "generation_review"):
            socs = [{"society": d["meta"]["society"],
                     "experiment": d["meta"].get("experiment") or "none (carried on the inherited culture, if any)",
                     "final_score": d["snapshots"][-1]["metrics"]["score"], "mean_score_second_half": second_half(d),
                     "selected": o == best_out, "organization": org_stats(d),
                     "own_retrospective": clip(d["meta"].get("lessons", ""), 1500)} for o, d, _ in results]
            spent = cost_usd(best_k, meta.get("preset", ""), meta.get("model", "")) or 0.0
            review = await backend.generation_review(best_k, socs, population - 1)
            spent = (cost_usd(best_k, meta.get("preset", ""), meta.get("model", "")) or 0.0) - spent
            experiments = review["experiments"]
            culture = dict(culture, lessons=f"Review of generation {ep + 1} (comparing its {population} societies):\n"
                           f"{review['lessons']}\n\nThe selected society's own retrospective:\n{clip(culture.get('lessons'), 2500)}")
            reviews.append({"episode": ep + 1, "selected": best_out.name, "cost_usd": round(spent, 4), **review})
            print(f"review of episode {ep + 1}: {len(experiments)} experiments for the next generation", flush=True)
        if nested:
            (out_root / "lineage.json").write_text(json.dumps({"summary": summary, "reviews": reviews}, indent=1))
    if nested:
        lines = ["| episode | society | experiment | score | 2nd half | coverage | depth | groups | supergroups | tokens | selected |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
        lines += [f"| {s['episode']} | {s['society']} | {clip(s['experiment'], 80) if s['experiment'] else '–'} | "
                  f"{f(s['score'])} | {f(s['second_half'])} | {f(s['coverage'])} | {s['depth']} | {s['groups']} | "
                  f"{s['supergroups']} | {s['tokens']:,} | {'yes' if s['selected'] else ''} |" for s in summary]
        for r in reviews:
            lines += ["", f"## Review of episode {r['episode']} (selected {r['selected']})", "", r["lessons"], "",
                      "Experiments for the next generation:", ""] + [f"{i + 2}. {e}" for i, e in enumerate(r["experiments"])]
        (out_root / "lineage.md").write_text("# Lineage\n\n" + "\n".join(lines) + "\n")
    return summary
