"""Generations of recipes: the outer loop of the evolution (DESIGN.md section 20).

Each generation:
  1. a parent is drawn from the archive (alphateam/archive.py sample_parent) and re-run until it has enough runs from
     scratch to stand on;
  2. a full society runs under the parent. Its agents may propose rule changes and leave lessons (the `rules` action,
     alphateam/recipe.py Rulebook), and may design actions;
  3. candidate revisions are made from what agents proposed under the parent, in any of its runs so far, most widely
     proposed first; then from a lesson the society left (a revision that adds it to the recipe's advice) and from an
     action its agents designed and at least two adopted. If that is fewer than asked for, the rest are drawn from the
     menu of typed edits, and marked as such. A revision that was already judged is not made again;
  4. the selection ladder (alphateam/selection.py) decides which, if any, is kept;
  5. every lesson goes to the library. A lesson whose revision was confirmed is marked confirmed, in the library and
     in the kept recipe; one whose revision lost its confirmation is marked refuted.
After the last generation, report() scores frozen recipes on instances selection never used.

A first pass on single runs (Policy(confirm_pairs=0, parent="tip")) climbs from the newest recipe taken on: each
generation is one run of it and one run of each candidate, and the best candidate that beat it goes on as a lead. No
lesson is marked by such a pass: nothing in it was confirmed.

Two controls use the same loop:
  Policy(mode="random")  the same generations; one valid revision a generation is kept at random; nothing is scored.
                         Its lineage is a random walk through valid revisions, and no lesson is marked by it.
  agents=False           the same ladder and budget, with every candidate drawn from the menu: what agents' own
                         proposals add over random typed edits.
"""
from __future__ import annotations

import random

from .archive import FRESH, PARENTS, Archive
from .recipe import Recipe, RecipeError, apply_edits, inert, menu_edits, unused, validate
from .selection import Ladder, Policy
from .stats import summary


def ahead(archive: Archive, parent: Recipe, metric: str = "score") -> list[tuple[dict, float, int]]:
    """Revisions that were set aside and stand ahead of the parent so far, best first: (record, mean gain, runs). A
    revision's runs are all its screens, whichever recipe was the parent then: a recipe that was a revision of an
    earlier parent stays in the race. The parent's mark is the mean of its own runs from scratch. Of two that are
    equally ahead, the one with fewer runs comes first: its next run says more."""
    base = [r[metric] for r in archive.runs(parent.id, None, FRESH) if r.get(metric) is not None]
    if not base:
        return []
    mark, out = sum(base) / len(base), []
    niche = archive.records[parent.id]["niche"] if parent.id in archive.records else None
    for rec in archive.recipes(("candidate",), niche):
        if rec["id"] == parent.id:
            continue
        had = [q["b"] for q in archive.pairs(rec["id"], ("screen",)) if q.get("metric") == metric and q.get("b") is not None]
        if had and sum(had) / len(had) > mark:
            out.append((rec, sum(had) / len(had) - mark, len(had)))
    return sorted(out, key=lambda x: (-round(x[1], 6), x[2], x[0]["id"]))


def candidates(parent: Recipe, gen, archive: Archive, rng: random.Random, n: int, locked=(), advice: int = 1,
               log=None, agents: bool = True, planted=(), retests: int = 0, metric: str = "score",
               together: int = 0) -> list[Recipe]:
    """Up to n revisions of the parent that have not been judged yet: what a society kept after trying it on itself,
    then what agents proposed under the parent, a lesson, an action they designed, and last the menu. `planted` is a
    control: lists of edits whose effect is known from earlier runs, taken first and marked as such, so that a
    generation also shows whether the ladder keeps what it should. A menu draw that could change nothing is skipped,
    and so is one that only takes away or tunes something no agent used in the parent's run (recipe.unused)."""
    out: list[Recipe] = []
    seen = {parent.id}

    def take(edits, origin, live_only: bool = False) -> bool:
        if len(out) >= n or not edits:
            return False
        try:
            c = apply_edits(parent, edits, origin, locked)
        except RecipeError as exc:
            if log and not live_only:
                log(f"  a proposal could not be made into a revision: {exc}")
            return False
        if live_only and (inert(parent, c) or unused(parent, c, gen)):  # a menu draw that could not show is not paid for
            return False
        prior = archive.seen(c)
        if live_only and retests and prior is not None:  # the menu then only explores: what was run before comes back
            return False                                 # through `ahead`, or because agents propose it again
        if c.id in seen or (prior is not None and prior["status"] != "candidate"):
            return False
        seen.add(c.id)
        out.append(c)
        return True

    for edits in planted or ():
        take(list(edits), {"by": "control"})
    for rec, gain, runs in ahead(archive, parent, metric)[:max(0, retests)]:  # a first pass on single runs: one more run of
        c = Recipe.from_json(rec["recipe"])  # what is ahead so far: the recipe itself, as filed (it may be a revision of
        if len(out) < n and c.id not in seen:  # an earlier parent, so its edits are not applied to this one)
            seen.add(c.id)
            out.append(c)
    if agents:
        kept = {t["key"]: t for t in getattr(gen, "trials", []) if t.get("kept")}
        take([{"op": "set", "key": t["key"], "value": t["value"]} for t in kept.values()
              if parent.settings.get(t["key"]) != t["value"]],
             {"by": "trial", "trials": [t["n"] for t in kept.values()], "run": gen.dir})
        for p in archive.proposals(parent.id):  # over every run under the parent, this generation's included
            take([{"op": "set", "key": p["key"], "value": p["value"]}],
                 {"by": "agents", "runs": p["runs"], "backers": p["backers"], "why": p["why"], "evidence": p["evidence"]})
        own = [les for les in gen.lessons if les.get("by") != "lab"]  # the lab's trial notes are records, not advice
        if together and len(own) > 1:  # the society's lessons as one body of advice, as a later society would start from
            edits, ids = [], []        # them: those that cite the most evidence first, as many as the size budget holds
            for les in sorted(own, key=lambda x: -len(x.get("evidence") or [])):
                one = {"op": "advise", "text": les["text"], "when": les.get("when", ""), "evidence": les.get("evidence", []),
                       "source": gen.dir}
                try:
                    apply_edits(parent, edits + [one], {}, locked)
                except RecipeError:
                    continue
                edits.append(one)
                ids.append(les["id"])
                if len(edits) >= together:
                    break
            if len(edits) > 1:
                take(edits, {"by": "agents", "lessons": ids, "run": gen.dir})
        for les in own[:advice]:
            take([{"op": "advise", "text": les["text"], "when": les.get("when", ""), "evidence": les.get("evidence", []),
                   "source": gen.dir}], {"by": "agents", "lesson": les["id"], "run": gen.dir})
        used = sorted((a for a in getattr(gen, "actions", []) if a.get("adopters", 0) >= 2), key=lambda a: -a["adopters"])
        for a in used[:1]:
            take([{"op": "adopt", "action": a}], {"by": "agents", "action": a["name"], "adopters": a["adopters"], "run": gen.dir})
    tries = 0
    while len(out) < n and tries < 20 * n:  # the menu fills what the society left open
        tries += 1
        take(menu_edits(parent, rng, locked), {"by": "menu"}, live_only=True)
    return out


def best(archive: Archive, niche: str, min_runs: int = 2) -> str | None:
    """The recipe that stands highest in a niche: by the lower bound of what it scored, among those run enough."""
    rows = []
    for r in archive.recipes(PARENTS, niche):
        v = archive.value(r["id"], niche)
        if v["n"] >= min_runs and v["lcb"] is not None:
            rows.append((v["lcb"], r["recipe"]["generation"], r["id"]))
    return max(rows)[2] if rows else None


def tip(archive: Archive, niche: str) -> str:
    """The newest recipe taken on in a niche, kept or lead (its seed when there is none)."""
    rows = archive.recipes(PARENTS, niche)
    return max(rows, key=lambda r: (r["recipe"]["generation"], r["added"], r["id"]))["id"]


async def evolve(archive: Archive, runner, seed_recipe: Recipe, generations: int = 5, per_generation: int = 4,
                 policy: Policy | None = None, rng: random.Random | None = None, advice: int = 1, log=print,
                 agents: bool = True, planted=(), start_from: str | None = None, together: int = 0) -> dict:
    """Run generations. A run that stops early (RunStopped) or a reached cap (OverBudget) passes through to the
    caller: everything judged so far is on file, and the same call carries on from it. `start_from` names the recipe
    the first of these generations starts from, whatever the policy would have drawn."""
    policy, rng, niche = policy or Policy(), rng or random.Random(0), runner.niche
    archive.bind(niche, runner.fingerprint())
    start = validate(seed_recipe, runner.locked)
    if not archive.recipes(PARENTS, niche):
        archive.add(start, "seed", niche)
    ladder = Ladder(archive, runner, policy, rng, log)
    history = []
    for n in range(generations):
        g = archive.next("generation")
        if start_from and n == 0:
            parent = archive.get(start_from)
        elif policy.mode == "random" or policy.parent == "tip":
            parent = archive.get(tip(archive, niche))
        else:
            parent = archive.sample_parent(niche, rng)
        if policy.mode != "random":
            await ladder.ensure_tested(parent)
        gen = await runner.generation(parent, g)
        # a run that tried changes on itself is shorter, and may have ended under other rules: not a plain run of its recipe
        graded = {"hidden": gen.hidden} if getattr(gen, "hidden", None) is not None else {}
        archive.record_run(parent.id, niche, "generation" if not gen.trials else "trial_run", gen.seed, gen.score,
                           gen.progress, gen.cost, dir=gen.dir, sims=gen.sims, ended_with=gen.ended_with, **graded)
        archive.record_voices(gen, niche)
        cands = candidates(parent, gen, archive, rng, per_generation, runner.locked, advice, log, agents, planted,
                           retests=0 if policy.confirm_pairs else policy.retests, metric=policy.screen_metric,
                           together=together)
        log(f"generation {g}: parent {parent.id} scored {gen.score}; its society made {len(gen.proposals)} proposals and "
            f"left {len(gen.lessons)} lessons; {len(cands)} candidates ({', '.join(c.origin.get('by', '?') for c in cands)})")
        decisions = await ladder.consider(parent, cands)
        if policy.mode == "paired":  # only a confirmation says anything about a lesson
            for c, d in zip(cands, decisions):
                lid = c.origin.get("lesson")
                if lid and d.kept and d.stage == "kept":  # a lead was not confirmed, and neither is its lesson
                    archive.mark_lesson(lid, "confirmed", d.why)
                    archive.set_advice_status(c.id, lid, "confirmed")
                elif lid and d.stage == "confirm":
                    archive.mark_lesson(lid, "refuted", d.why)
        history.append({"generation": g, "parent": parent.id, "score": gen.score, "candidates": [c.id for c in cands],
                        "kept": [d.child for d in decisions if d.kept], "leads": [d.child for d in decisions if d.stage == "lead"],
                        "runs": sum(d.runs for d in decisions) + 1, "cost": round(sum(d.cost for d in decisions) + (gen.cost or 0), 4)})
    by_tip = policy.mode == "random" or not policy.confirm_pairs  # single runs give no bound to rank recipes by
    final = tip(archive, niche) if by_tip else (best(archive, niche, policy.retest_runs) or tip(archive, niche))
    return {"niche": niche, "mode": policy.mode, "agents": agents, "seed": start.id, "best": final, "history": history,
            "lineage": archive.lineage(final)}


async def report(archive: Archive, runner, recipes: dict[str, Recipe], runs: int = 10, conf: float = 0.9, log=print) -> dict:
    """Score frozen recipes on report instances that selection never used: every recipe on the same instances, so
    each comparison with the first one (the seed) is paired. Report runs never count toward a recipe's standing."""
    scores: dict[str, list] = {name: [] for name in recipes}
    for i in range(runs):
        for name, recipe in recipes.items():
            r = await runner.score(recipe, i, "report")
            graded = {"hidden": r.hidden} if getattr(r, "hidden", None) is not None else {}
            archive.record_run(r.recipe, runner.niche, "report", r.seed, r.score, r.progress, r.cost, dir=r.dir, arm=name,
                               sims=r.sims, **graded)
            scores[name].append(r.score)
    names = list(recipes)
    out = {name: summary(scores[name], conf) for name in names}
    for name in names[1:]:
        out[f"{name} - {names[0]}"] = summary([b - a for a, b in zip(scores[names[0]], scores[name])
                                                if a is not None and b is not None], conf)
    for key, v in out.items():
        log(f"report {key}: " + ("no runs" if not v["n"] else f"mean {v['mean']:+.3f} over {v['n']}"
                                 + ("" if v["lcb"] is None else f" (lower bound {v['lcb']:+.3f})")))
    return out
