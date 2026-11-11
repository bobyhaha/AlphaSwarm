#!/usr/bin/env python3
"""Evolve the rules of a society that rebuilds ProgramBench xsv (alphateam/code_runs.py, DESIGN.md section 21).

Offline rehearsal on the in-memory harness with scripted agents (free; it shows the machinery only):
  .venv/bin/python tools/code_evolve.py --fake --agents 4 --ticks 4 --generations 2 --out runs_recipes/rehearsal

A paid stage needs --backend openrouter, --live and OPENROUTER_API_KEY in the environment (tools/keyed_runner.py
holds it), and starts only inside what is left of the research budget (spend.json, $2 kept back):
  ... --stage smoke  --out runs_recipes/XSV1                       one short run: does the whole path work with models?
  ... --stage evolve --out runs_recipes/XSV1 --generations 1 --per-generation 1 --plant stream=on
                                                                    a generation; --plant adds a known-good control
  ... --stage evolve --out runs_recipes/XSV1 --screen-pairs 1 --confirm-pairs 0
                                                                    a first pass on single runs: one run of the parent and
                                                                    one of each candidate; the best that beat the parent
                                                                    goes on as a lead, not confirmed
  ... --stage report --out runs_recipes/XSV1 --report-runs 3        fresh runs of the seed and the best recipe
  ... --stage status --out runs_recipes/XSV1                        what the archive holds (free, no key)

Selection reads the held-out checks (179; nobody sees them). Every full run also ends with the hidden tests (1,182),
which selection never reads. One generation: a full run of the parent (its agents may propose rule changes), then each
candidate in `--screen-pairs` fresh runs, then the best of them against the parent in up to `--confirm-pairs` fresh
pairs, looked at after `--looks`. The archive under OUT/archive carries on with the same --out.
"""
import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from alphateam.archive import FRESH, PARENTS, Archive, ArchiveMismatch  # noqa: E402
from alphateam.code_runs import CodeRunner  # noqa: E402
from alphateam.generations import best, evolve, tip  # noqa: E402
from alphateam.paired import OverBudget, RunStopped, code_frame, instance  # noqa: E402
from alphateam.recipe import registry, seed, show  # noqa: E402
from alphateam.selection import Policy  # noqa: E402
from alphateam.stats import t_quantile  # noqa: E402

USD_PER_REPLY = 0.69 / 720  # ten agents, 24 ticks of three replies: $0.57-0.83 a run in the saved runs of 2026-11-04
HELD = 179  # held-out checks


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def est_run(f: dict) -> float:
    return f["agents"] * (f["calls"] + 1) * f["ticks"] * USD_PER_REPLY


def make_runner(a, f: dict, out: Path, planned: float) -> CodeRunner:
    if a.fake or a.backend == "mock":  # scripted agents: free. Without --fake they work on the real harness (Docker, the grader)
        spec = {"kind": "mock"}
        if a.propose:  # a rehearsal of the closing round: the first scripted agent proposes this after the last tick
            key, _, value = a.propose.partition("=")
            spec["proposes"] = {"key": key.strip(), "value": value.strip()}
        return CodeRunner(f, spec, out / "runs", jobs=a.jobs, stagger=0.0 if a.fake else a.stagger,
                          meta={"backend": "mock"}, log=log)
    if not a.live:
        sys.exit("not started: a paid run needs --live as well as --backend openrouter")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("not started: OPENROUTER_API_KEY is not in the environment (tools/keyed_runner.py holds it)")
    from _recipes import budget_left
    left = budget_left()
    cap = min(a.cap if a.cap is not None else planned, left - 2)
    if cap < 0.3:
        sys.exit(f"not started: ${left:.2f} is left in the research budget (spend.json), of which $2 is kept back")
    log(f"paid stage: model {a.model}, cap ${cap:.2f} (plan at most ${planned:.2f}; ${left:.2f} left in the research budget)")
    backend = {"kind": "openrouter", "model": a.model, "effort": a.effort, "max_tokens": 16384, "provider_ignore": ["openai/flex"]}
    meta = {"backend": "openrouter", "preset": "openrouter", "model": a.model, "reasoning_effort": a.effort,
            "max_tokens": 16384}
    return CodeRunner(f, backend, out / "runs", jobs=a.jobs, stagger=a.stagger, max_run_usd=3 * est_run(f), cap_usd=cap,
                      meta=meta, est_cost=est_run(f), log=log)


def edits_of(text: str, world: str) -> list[dict]:
    reg = registry(world)
    out = []
    for kv in text.split(","):
        key, _, value = kv.partition("=")
        if key.strip() not in reg:
            sys.exit(f"--plant: no setting '{key.strip()}' (settings: {', '.join(reg)})")
        out.append({"op": "set", "key": key.strip(), "value": value.strip()})
    return out


def checks(x) -> str:
    return "  -  " if x is None else f"{x * HELD:5.1f}"


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None), len(xs)


def welch(a: list, b: list, conf: float = 0.9):
    """b minus a for two sets of independent runs: the mean difference and its one-sided lower bound."""
    (ma, sa, na), (mb, sb, nb) = mean_sd(a), mean_sd(b)
    if na < 2 or nb < 2:
        return (None if ma is None or mb is None else mb - ma), None
    va, vb = sa * sa / na, sb * sb / nb
    if va + vb == 0:
        return mb - ma, mb - ma
    df = (va + vb) ** 2 / (va * va / (na - 1) + vb * vb / (nb - 1))
    return mb - ma, mb - ma - t_quantile(conf, max(1, int(df))) * math.sqrt(va + vb)


def results(out: Path) -> list[dict]:
    """Every run that left a result under OUT/runs, whatever the archive recorded of it (screens file only pairs)."""
    rows = []
    for path in sorted((out / "runs").glob("**/result.json")):
        if "dry" in path.relative_to(out).parts:
            continue
        try:
            rows.append(json.loads(path.read_text()))
        except ValueError:
            pass
    return rows


def status(archive: Archive, niche: str | None = None, out: Path | None = None):
    rows = archive.recipes(None, niche)
    if not rows:
        print("the archive holds no recipes yet")
        return
    niche = niche or rows[0]["niche"]
    done = results(out) if out else []
    horizon = max((x["ticks"] for x in done), default=0)
    full = [x for x in done if x["ticks"] == horizon and not x.get("stopped") and x.get("purpose") not in ("probe", "smoke")]
    paid = sum(x.get("cost") or 0 for x in done)
    print(f"niche {niche}; {len(done)} runs left a result ({len(full)} full), costing ${paid:.2f} by their own reports")
    print(f"{'recipe':12} {'status':10} gen {'by':8} {'full runs':>9} {'held-out':>9} {'sd':>5} {'hidden':>7} {'sd':>5}  change")
    for r in sorted(rows, key=lambda r: (r["recipe"]["generation"], r["added"])):
        rid, rec = r["id"], r["recipe"]
        runs = [x for x in full if x["recipe"] == rid]
        m, s, n = mean_sd([x["score"] for x in runs])
        hm, hs, _ = mean_sd([x.get("hidden") for x in runs])
        change = "; ".join(f"{e['key']}={show(e['value'])}" if e.get("op") == "set" else f"{e.get('op')} {e.get('id') or e.get('name') or ''}"
                           for e in rec["origin"].get("edits", [])) or "(the starting rules)"
        print(f"{rid:12} {r['status']:10} {rec['generation']:>3} {str(rec['origin'].get('by', '')):8} {n:>9} {checks(m):>9} "
              f"{'' if s is None else f'{s * HELD:5.1f}':>5} {'' if hm is None else f'{hm:7.0f}':>7} {'' if hs is None else f'{hs:5.0f}':>5}  {change}"
              + (f"  [{r['why']}]" if r.get("why") else ""))
    print("\nfull runs, each as held-out checks / hidden tests:")
    for r in sorted(rows, key=lambda r: (r["recipe"]["generation"], r["added"])):
        runs = [x for x in full if x["recipe"] == r["id"]]
        if runs:
            print(f"  {r['id']}: " + ", ".join(f"{x.get('purpose', '?')} {x['score'] * HELD:.0f}/"
                                              f"{x['hidden'] if x.get('hidden') is not None else '-'}" for x in runs))
    print("\npairs (child minus parent, in held-out checks):")
    for r in rows:
        for purpose in ("screen", "confirm"):
            gains = [p["gain"] for p in archive.pairs(r["id"], (purpose,)) if p.get("gain") is not None]
            if gains:
                print(f"  {r['id']} {purpose:8} {len(gains)} pairs: " + ", ".join(f"{g * HELD:+.0f}" for g in gains)
                      + f"  (mean {statistics.mean(gains) * HELD:+.1f})")
    said = [(r["id"], archive.proposals(r["id"])) for r in rows]
    n_prop = sum(len(p) for _, p in said)
    print(f"\nrule changes proposed by agents: {n_prop}; lessons left: {len(archive.lessons(niche))}")
    for rid, props in said:
        for p in props:
            print(f"  under {rid}: {p['key']} = {show(p['value'])}, in {p['runs']} run(s), {p['backers']} backer(s): {str(p.get('why', ''))[:150]}")
    for les in archive.lessons(niche)[:12]:
        print(f"  lesson [{les.get('status', 'untested')}]: {les['text'][:200]}")
    purposes = {}
    for x in archive.runs(None, niche):
        purposes[x["purpose"]] = purposes.get(x["purpose"], 0) + 1
    print("\nruns recorded:", ", ".join(f"{n} {p}" for p, n in sorted(purposes.items())))


async def smoke(a, out: Path):
    f = code_frame(agents=a.smoke_agents, ticks=a.smoke_ticks, replies=a.replies, fake=a.fake, closing=a.closing)
    runner = make_runner(a, f, out / "smoke", a.cap if a.cap is not None else max(0.5, 4 * est_run(f)))
    start = seed("code", **dict(kv.split("=", 1) for kv in a.set))
    r = await runner._job(start, instance("control", 0), runner._dir("smoke"), "smoke", None, runner._share(1))
    if r is None:
        sys.exit("the smoke run could not be run: see its job.log")
    rules = {op: n for op, n in r.ops.items() if "rules" in op}
    log(f"smoke run under {start.id}: {r.ticks} ticks, held-out {checks(r.score).strip()} of {HELD}, hidden {r.hidden}, "
        f"cost ${r.cost if r.cost is not None else float('nan'):.3f}, stopped: {r.stopped}")
    log(f"  what agents did: {dict(sorted(r.ops.items(), key=lambda kv: -kv[1])[:14])}")
    log(f"  uses of the rules action: {rules or 'none'}; proposals {len(r.proposals)}, lessons {len(r.lessons)}; facts {r.facts}")
    log(f"  run files: {r.dir}")


async def do_report(a, archive: Archive, runner: CodeRunner, names: list[str]):
    niche = runner.niche
    got = {}
    for rid in names:
        recipe = archive.get(rid)
        idx = [archive.next("report") for _ in range(a.report_runs)]
        got[rid] = await runner.score_many(recipe, idx, "report")
        for r in got[rid]:
            archive.record_run(r.recipe, niche, "report", r.seed, r.score, r.progress, r.cost, dir=r.dir, sims=r.sims,
                               **({"hidden": r.hidden} if r.hidden is not None else {}))
    base = names[0]
    for rid in names:
        m, s, n = mean_sd([r.score for r in got[rid]])
        hm, hs, _ = mean_sd([r.hidden for r in got[rid]])
        log(f"report {rid}: {n} runs, held-out {checks(m).strip()} (sd {'-' if s is None else f'{s * HELD:.1f}'}), "
            f"hidden {'-' if hm is None else f'{hm:.0f}'} (sd {'-' if hs is None else f'{hs:.0f}'})")
    for rid in names[1:]:
        d, low = welch([r.hidden for r in got[base]], [r.hidden for r in got[rid]])
        log(f"report {rid} minus {base} on hidden tests: " + ("-" if d is None else f"{d:+.0f}")
            + ("" if low is None else f" (one-sided 90% lower bound {low:+.0f})"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="evolve", choices=["smoke", "evolve", "report", "status"])
    p.add_argument("--out", required=True, help="where the archive and the runs go")
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--replies", type=int, default=3, help="replies an agent a tick, its closing reply included")
    p.add_argument("--fake", action="store_true", help="the in-memory harness and scripted agents: free, machinery only")
    p.add_argument("--backend", default="mock", choices=["mock", "openrouter"])
    p.add_argument("--live", action="store_true", help="needed, with --backend openrouter, for a stage that spends money")
    p.add_argument("--model", default="openai/gpt-6-luna")
    p.add_argument("--effort", default="low", help="reasoning effort")
    p.add_argument("--cap", type=float, help="dollars this stage may spend at most (default: what its plan costs at most)")
    p.add_argument("--jobs", type=int, default=5, help="runs side by side")
    p.add_argument("--stagger", type=float, default=15.0, help="seconds between the starts of runs")
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--per-generation", type=int, default=4, help="candidate revisions a generation")
    p.add_argument("--screen-pairs", type=int, default=2, help="fresh runs of each candidate before the best one goes on")
    p.add_argument("--confirm-pairs", type=int, default=4, help="0: a first pass on single runs, nothing is confirmed")
    p.add_argument("--lead-runs", type=int, default=2, help="single-run pass: runs of its own a revision needs to be taken on")
    p.add_argument("--lead-min", type=float, default=0.08, help="single-run pass: how far ahead of the parent's mark a "
                   "revision's mean must be to be taken on (0.08 is 14 of 179 checks, a little more than one run's spread)")
    p.add_argument("--retests", type=int, default=2, help="single-run pass: set-aside revisions that are ahead of the parent "
                   "so far get another run, at most this many a generation")
    p.add_argument("--lessons-together", type=int, default=0, help="also try the lessons a society left as one body of "
                   "advice: up to this many in one candidate (those citing the most evidence first)")
    p.add_argument("--single-lessons", type=int, default=1, help="lessons tried one at a time, each as a candidate of its own")
    p.add_argument("--parent", help="the recipe this stage's first generation starts from (default: the newest recipe "
                   "taken on in a single-run pass, else one drawn by standing)")
    p.add_argument("--looks", type=int, nargs="*", default=[2, 4])
    p.add_argument("--retest", type=int, default=0, help="fresh runs a recipe has before it seeds a generation")
    p.add_argument("--plant", action="append", default=[], metavar="KEY=VALUE", help="a control: a revision whose effect is "
                   "known from earlier runs is put among the candidates, to see that the ladder keeps what it should")
    p.add_argument("--menu-only", action="store_true", help="a control: every candidate drawn from the menu of edits")
    p.add_argument("--report-runs", type=int, default=3)
    p.add_argument("--recipes", nargs="*", help="report: recipe ids (default: the seed and the best)")
    p.add_argument("--closing", action="store_true", help="after the last tick of a full run each agent may still use the "
                   "rules action (propose, back, lesson); it cannot move the run's score")
    p.add_argument("--propose", metavar="KEY=VALUE", help="scripted agents only: what the first one proposes in the closing round")
    p.add_argument("--smoke-agents", type=int, default=3)
    p.add_argument("--smoke-ticks", type=int, default=3)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="start from the seed recipe with this setting")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    archive = Archive(out / "archive")
    if a.stage == "status":
        return status(archive, out=out)
    if a.stage == "smoke":
        return asyncio.run(smoke(a, out))
    f = code_frame(agents=a.agents, ticks=a.ticks, replies=a.replies, fake=a.fake, closing=a.closing)
    per_gen = 1 + a.retest + a.per_generation * a.screen_pairs + 2 * a.confirm_pairs
    planned = (a.generations * per_gen if a.stage == "evolve" else 2 * a.report_runs) * est_run(f)
    runner = make_runner(a, f, out, planned)
    start = seed("code", **dict(kv.split("=", 1) for kv in a.set))
    runner.standing = lambda rid: archive.runs(rid, runner.niche, FRESH)  # what a screen measures a candidate against
    if a.stage == "report":
        archive.bind(runner.niche, runner.fingerprint())
        names = a.recipes or list(dict.fromkeys([tip_seed(archive, runner.niche, start.id), best(archive, runner.niche, 2) or tip(archive, runner.niche)]))
        try:
            asyncio.run(do_report(a, archive, runner, names))
        except (OverBudget, RunStopped) as exc:
            log(f"stopped: {exc}")
            sys.exit(3)
        log(f"runs so far cost ${runner.spent:.2f} in this stage")
        return
    policy = Policy(mode="paired", screen_pairs=a.screen_pairs, screen_metric="score", confirm_pairs=a.confirm_pairs,
                    looks=tuple(a.looks), retest_runs=a.retest, metric="score", parent="sample" if a.confirm_pairs else "tip",
                    lead_runs=1 if a.confirm_pairs else max(1, a.lead_runs), retests=0 if a.confirm_pairs else max(0, a.retests),
                    lead_min=0.0 if a.confirm_pairs else max(0.0, a.lead_min))
    if a.parent and a.parent not in archive.records:
        sys.exit(f"--parent: no recipe {a.parent} in {out / 'archive'}")
    planted = [edits_of(text, "code") for text in a.plant]
    rng = random.Random(f"{a.seed}:{len(archive.decisions())}")  # a later stage on the same archive draws anew
    log(f"niche {runner.niche}; seed recipe {start.id}; at most {a.generations * per_gen} full runs"
        + ("" if a.fake else f", about ${planned:.2f}") + (f"; planted control(s): {a.plant}" if a.plant else ""))

    async def go():
        try:
            return await evolve(archive, runner, start, a.generations, a.per_generation, policy, rng, log=log,
                                agents=not a.menu_only, planted=planted, start_from=a.parent, advice=a.single_lessons,
                                together=a.lessons_together)
        except (OverBudget, RunStopped) as exc:  # what was judged is on file; the same command carries on from it
            log(f"stopped: {exc}")
            return None

    try:
        res = asyncio.run(go())
    except ArchiveMismatch as exc:
        sys.exit(f"not started: {exc}")
    print()
    status(archive, runner.niche, out)
    log(f"this stage's runs cost ${runner.spent:.2f}" + ("" if not a.fake else " (scripted agents: nothing was spent)"))
    if res is None:
        sys.exit(3)
    (out / "evolution.json").write_text(json.dumps(res, indent=1, default=list))
    log(f"best recipe {res['best']} (lineage {' > '.join(res['lineage'])}); archive in {out / 'archive'}")


def tip_seed(archive: Archive, niche: str, fallback: str) -> str:
    seeds = [r["id"] for r in archive.recipes(("seed",), niche)]
    return seeds[0] if seeds else fallback


if __name__ == "__main__":
    main()
