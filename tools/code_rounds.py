"""Rounds of candidate recipes on the code world, judged on the hidden tests (DESIGN.md section 22).

  add    file a candidate: a recipe that differs from a base by settings, or by lessons earlier societies left
  run    fresh full runs of recipes, side by side; each run is filed with both of its scores
  table  every recipe of the experiment: runs, hidden tests (mean, spread), held-out checks, and how far it is from
         a base with a one-sided 90% bound
  voices what the societies left after their runs: rules they suggested, values they proposed, lessons

  .venv/bin/python tools/code_rounds.py --out runs_recipes/XSV2 --stage add --base start --set hold_scope=named --by proposer --title "..."
  .venv/bin/python tools/code_rounds.py --out runs_recipes/XSV2 --stage run --recipes Rxxxx:4 Ryyyy --backend openrouter --live
  .venv/bin/python tools/code_rounds.py --out runs_recipes/XSV2 --stage table --base Rxxxx

Why this and not the ladder of tools/code_evolve.py: after ten runs on 2026-11-10 it was clear that one run cannot rank
recipes here (two runs of the same rules differ by about 12 of 179 held-out checks) and that candidates drawn from a
menu of switches have effects smaller than that. So candidates now come from whoever read the record of earlier runs
and can say what they expect to change (tools/proposer_brief.md, alphateam/record.py), there are few of them, and each
is run several times. Nothing is taken on from one run. The judge is the hidden tests: the held-out checks are told to
agents as counts during a run and paid as credit, so they are not independent of the rules being judged.

A paid stage needs --backend openrouter --live and the key in the environment (tools/keyed_runner.py holds it).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from code_evolve import HELD, est_run, log, make_runner, mean_sd, welch  # noqa: E402
from alphateam.archive import Archive, ArchiveMismatch  # noqa: E402
from alphateam.paired import OverBudget, RunStopped, code_frame  # noqa: E402
from alphateam.recipe import RecipeError, apply_edits, registry, seed  # noqa: E402
from alphateam.record import change  # noqa: E402

PURPOSE = "eval"
HIDDEN = 1182


def base_recipe(archive: Archive, name: str):
    if name == "start":
        return seed("code")
    if name not in archive.records:
        sys.exit(f"no recipe {name} in {archive.root} (or use --base start)")
    return archive.get(name)


def lesson_edits(base, source: Path, n: int, ids: list[str]) -> tuple[list[dict], list[str]]:
    """Lessons earlier societies left, as one body of advice: those named, or those that cite the most evidence first,
    as many as the recipe's size budget holds."""
    rows = {}
    for line in (source / "lessons.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("op") == "add":
            rows[row["id"]] = row
    order = [rows[i] for i in ids if i in rows] if ids else sorted(rows.values(), key=lambda x: -len(x.get("evidence") or []))
    edits, used = [], []
    for les in order:
        one = {"op": "advise", "text": les["text"], "when": les.get("when", ""), "evidence": les.get("evidence", []),
               "source": les.get("source", str(source))}
        try:
            apply_edits(base, edits + [one], {})
        except RecipeError:
            continue
        edits.append(one)
        used.append(les["id"])
        if len(edits) >= n:
            break
    return edits, used


async def do_add(a, archive: Archive, runner):
    base = base_recipe(archive, a.base)
    edits = []
    for kv in [x for part in a.set for x in part.split(",") if x.strip()]:
        key, _, value = kv.partition("=")
        if key.strip() not in registry("code"):
            sys.exit(f"--set: no setting '{key.strip()}' (settings: {', '.join(registry('code'))})")
        edits.append({"op": "set", "key": key.strip(), "value": value.strip()})
    origin = {"by": a.by, "title": a.title or "", "note": a.note or ""}
    if a.lessons or a.lesson_ids:
        more, used = lesson_edits(base, ROOT / a.lessons_from, a.lessons, a.lesson_ids)
        edits += more
        origin["lessons"] = used
    if not edits:  # the base itself, filed so that it can be run and compared
        archive.add(base, "seed", runner.niche, why="base of the rounds")
        print(f"{base.id}  {change(base.to_json())}")
        return
    try:
        child = apply_edits(base, edits, origin)
    except RecipeError as exc:
        sys.exit(f"not filed: {exc}")
    ok, why = await runner.dry(child)
    if not ok:
        sys.exit(f"not filed: {why}")
    archive.add(base, "seed" if a.base == "start" else archive.records[base.id]["status"], runner.niche)
    archive.add(child, "candidate", runner.niche, why=a.title or "")
    print(f"{child.id}  {change(child.to_json())}  (base {base.id})")


async def do_run(a, archive: Archive, runner, wanted: list[tuple[str, int]]):
    niche = runner.niche

    async def one(rid: str, n: int):
        recipe = archive.get(rid)
        idx = [archive.next(PURPOSE) for _ in range(n)]
        try:
            runs = await runner.score_many(recipe, idx, PURPOSE)
        except (RunStopped, OverBudget) as exc:
            log(f"{rid}: {exc}")
            return
        for r in runs:
            archive.record_run(r.recipe, niche, PURPOSE, r.seed, r.score, r.progress, r.cost, dir=r.dir, sims=r.sims,
                               hidden=r.hidden, held_out=(r.facts or {}).get("held_out"), ticks=r.ticks)
            archive.record_voices(r, niche)
            log(f"{rid} ({change(recipe.to_json())}): hidden {r.hidden}, held-out {(r.facts or {}).get('held_out')}, "
                f"${r.cost if r.cost is not None else float('nan'):.2f}, {Path(r.dir).name}"
                + ("" if r.hidden is not None else f"  [no hidden grade: {(r.facts or {}).get('grade')}]"))

    await asyncio.gather(*(one(rid, n) for rid, n in wanted))


def table(archive: Archive, base: str | None = None, niche: str | None = None):
    rows = archive.recipes(None, niche)
    runs = [x for x in archive.runs(None, niche, (PURPOSE,))]
    by = {}
    for x in runs:
        by.setdefault(x["recipe"], []).append(x)
    if not by:
        print("no runs filed yet")
        return
    base = base or next((r["id"] for r in rows if r["status"] == "seed" and r["id"] in by), None)
    ref = [x.get("hidden") for x in by.get(base, [])] if base else []
    ref_held = [x.get("score") for x in by.get(base, [])] if base else []
    print(f"{len(runs)} runs of {len(by)} recipes, ${sum(x.get('cost') or 0 for x in runs):.2f} by their own reports; base {base}")
    print(f"{'recipe':12} {'by':9} {'runs':>4} {'hidden':>7} {'sd':>4} {'held-out':>8} {'sd':>5}  {'vs base (hidden)':24} rules")
    order = sorted(by, key=lambda rid: -(mean_sd([x.get("hidden") for x in by[rid]])[0] or -1))
    for rid in order:
        rec = archive.records[rid]["recipe"]
        hm, hs, n = mean_sd([x.get("hidden") for x in by[rid]])
        m, s, _ = mean_sd([x.get("score") for x in by[rid]])
        gap = ""
        if base and rid != base:
            d, low = welch([v for v in ref if v is not None], [x.get("hidden") for x in by[rid]])
            gap = "-" if d is None else f"{d:+.0f}" + ("" if low is None else f" (90% bound {low:+.0f})")
        print(f"{rid:12} {str((rec.get('origin') or {}).get('by', ''))[:9]:9} {len(by[rid]):>4} "
              f"{'-' if hm is None else f'{hm:7.0f}':>7} {'' if hs is None else f'{hs:4.0f}':>4} "
              f"{'-' if m is None else f'{m * HELD:8.1f}':>8} {'' if s is None else f'{s * HELD:5.1f}':>5}  {gap:24} {change(rec)}"
              + (f"  [{(rec.get('origin') or {}).get('title')}]" if (rec.get('origin') or {}).get('title') else ""))
    print("\neach run, hidden/held-out:")
    for rid in order:
        print(f"  {rid}: " + ", ".join(f"{x.get('hidden')}/{'-' if x.get('score') is None else round(x['score'] * HELD)}" for x in by[rid]))
    if base and len([v for v in ref if v is not None]) > 1:
        sd = statistics.stdev([v for v in ref if v is not None])
        print(f"\nspread of the base on hidden tests: sd {sd:.0f} over {len(ref)} runs "
              f"(held-out: sd {statistics.stdev([v for v in ref_held if v is not None]) * HELD:.1f})")


def voices(archive: Archive):
    """What the societies themselves left after their runs: rules they suggested that are not settings yet, values they
    proposed for settings, and how many lessons. Read by whoever builds the next candidates."""
    path = archive.root / "suggestions.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    print(f"{len(rows)} rules suggested by agents after their runs (not settings yet):")
    for x in sorted(rows, key=lambda x: (x.get("run", ""), -x.get("backers", 0))):
        print(f"- [{Path(x.get('run', '')).name} {x.get('by')}, backed by {x.get('backers')}, under {x.get('recipe')}] {x.get('text')}"
              + (f"\n    when: {x['when']}" if x.get("when") else "") + (f"\n    evidence: {', '.join(x.get('evidence') or [])}" if x.get("evidence") else ""))
    path = archive.root / "proposals.jsonl"
    props = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    print(f"\n{len(props)} values proposed for settings:")
    for x in props:
        print(f"- [{Path(x.get('run', '')).name}, backed by {x.get('backers')}] {x.get('key')} = {x.get('value')}: {str(x.get('why', ''))[:300]}")
    print(f"\nlessons left: {len(archive.lessons())}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["add", "run", "table", "voices"])
    p.add_argument("--out", required=True)
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--replies", type=int, default=3)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--backend", default="mock", choices=["mock", "openrouter"])
    p.add_argument("--live", action="store_true")
    p.add_argument("--model", default="openai/gpt-6-luna")
    p.add_argument("--effort", default="low")
    p.add_argument("--cap", type=float)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--stagger", type=float, default=15.0)
    p.add_argument("--closing", action="store_true", help="after the last round each agent may still use the rules action")
    p.add_argument("--closing-for", nargs="*", default=[], metavar="ID", help="run: only these recipes' runs get the closing "
                   "round (it cannot change a score, and it costs a few more replies an agent)")
    p.add_argument("--propose", help=argparse.SUPPRESS)
    # add
    p.add_argument("--base", default="start", help="add: the recipe a candidate differs from (an id, or start); table: the base")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE[,KEY=VALUE]")
    p.add_argument("--lessons", type=int, default=0, help="add: also carry up to this many lessons from --lessons-from")
    p.add_argument("--lesson-ids", nargs="*", default=[])
    p.add_argument("--lessons-from", default="runs_recipes/XSV1/archive")
    p.add_argument("--by", default="proposer", help="who proposed it: proposer (read the record from outside), agents, control")
    p.add_argument("--title", default="")
    p.add_argument("--note", default="", help="where the diagnosis is written down")
    # run
    p.add_argument("--recipes", nargs="*", default=[], metavar="ID[:RUNS]")
    p.add_argument("--runs", type=int, default=1)
    a = p.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    archive = Archive(out / "archive")
    if a.stage == "table":
        return table(archive, None if a.base == "start" else a.base)
    if a.stage == "voices":
        return voices(archive)
    f = code_frame(agents=a.agents, ticks=a.ticks, replies=a.replies, fake=a.fake, closing=a.closing)
    wanted = []
    for item in a.recipes:
        rid, _, n = item.partition(":")
        wanted.append((rid, int(n) if n else a.runs))
    if a.stage == "add":  # filing a candidate needs no model: scripted agents try it for two rounds on the in-memory harness
        a.backend = "mock"
        return asyncio.run(do_add(a, archive, make_runner(a, f, out, 0.0)))
    planned = 1.6 * est_run(f) * sum(n for _, n in wanted)  # a run costs $0.61 to $0.89; a run is cut off at its share
    runner = make_runner(a, f, out, max(planned, 0.5))
    try:
        archive.bind(runner.niche, runner.fingerprint())
    except ArchiveMismatch as exc:
        sys.exit(f"not started: {exc}")
    for rid in a.closing_for:
        runner.frames[rid] = dict(f, closing=True)
    missing = [rid for rid, _ in wanted if rid not in archive.records]
    if missing or not wanted:
        sys.exit(f"--recipes: not on file: {missing}" if missing else "--recipes: name the recipes to run")
    log(f"niche {runner.niche}, code {runner.fingerprint()['code']}; {sum(n for _, n in wanted)} full runs: "
        + ", ".join(f"{rid} x{n}" for rid, n in wanted))
    asyncio.run(do_run(a, archive, runner, wanted))
    print()
    table(archive, niche=runner.niche)
    log(f"this stage's runs cost ${runner.spent:.2f}" + (" (scripted agents: nothing was spent)" if a.backend == "mock" else ""))


if __name__ == "__main__":
    main()
