"""Trials inside one run: a society tries a change to its own rules on a copy of itself, and keeps it only if the
copy does better (DESIGN.md section 20: continual regrouping).

At a beat (every `every` ticks), if the society's agents have proposed a rule change that enough of them back, and
the change can be made mid-run, the run is copied. One copy carries on as it is; the other switches to the proposal.
Both run `window` ticks with the same budget a tick. The copy whose mean score over the window is higher carries on
as the society, and the other is dropped. The comparison is posted on the global board and filed as a note of the lab.

Both copies count toward the run's budget: every trial ends the run `window` ticks sooner, so a society that tries
changes pays for them out of the same total compute as one that does not. A trial is run only if at least `window`
ticks would be left after it to use what it showed.

Nobody outside the society decides what is tried: the proposals are the agents' own (the `rules` action), ranked by
how many agents back them. The harness only copies the run, makes the change and reports the two scores. One window
is one comparison: a change kept this way is a candidate for the next generation, still to be confirmed.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .archive import niche_of
from .paired import Result, RunStopped, _result, fork, start
from .recipe import Recipe, RecipeError, apply_edits, apply_live, lesson, registry, show
from .runner import run, save
from .stats import area


def pick(k, tried: set, backers: int) -> dict | None:
    """The most backed proposal that has enough backers, has not been tried, and can be made in a running society."""
    book = getattr(k, "rulebook", None)
    if book is None:
        return None
    reg = registry(book.recipe.world)
    for p in book.ranked():
        if (len(p["backers"]) >= backers and (p["key"], p["value"]) not in tried and reg[p["key"]].live is not None
                and p["key"] not in book.locked and book.now(p["key"]) != p["value"]):
            return p
    return None


async def run_with_trials(frame_cfg: dict, recipe: Recipe, make_backend, seed: int, out, every: int = 6, window: int = 3,
                          backers: int = 2, margin: float = 0.0, final: bool = False, log=None, meta: dict | None = None,
                          cost_budget: float | None = None) -> Result:
    """One society from scratch, allowed to try its own proposals on copies of itself. Returns the run's result;
    its `trials` say what was tried and `ended_with` the recipe it ended under. cost_budget covers the whole run,
    its dropped copies included."""
    out = Path(out)
    horizon = frame_cfg["ticks"]
    if not 0 < window < every:
        raise ValueError("a trial window must be shorter than the beat between trials")
    k, backend = start(frame_cfg, recipe, seed, cost_budget), make_backend(seed)
    meta = dict(meta or {}, purpose="generation", frame=frame_cfg.get("frame"), recipe=recipe.id, niche=niche_of(frame_cfg),
                seed=seed, trials={"every": every, "window": window, "backers": backers})
    k.rulebook.note = (f"TRIALS: every {every} ticks, the most backed rule proposal with at least {backers} backers that can "
                       f"change mid-run is tried on a copy of this society for {window} ticks. The society carries on as "
                       "whichever copy scored higher over those ticks, and what the other copy did in them is dropped. "
                       f"Each trial ends the run {window} ticks sooner: the copy is paid for out of the same budget.")
    end, tried, n, dropped = horizon, set(), 0, 0.0  # dropped: what the copies that did not carry on cost in their windows

    def cost(x) -> float:
        return (x.cost_fn() if x.cost_fn else None) or 0.0

    while k.tick < end and not meta.get("stopped"):
        beat = min(end, (k.tick // every + 1) * every)
        await run(k, backend, beat, out / "main", meta, seed, quiet=True, start=k.tick, horizon=end, final=False)
        if meta.get("stopped") or k.tick >= end:
            break
        p = pick(k, tried, backers)
        if p is None or end - 2 * window - k.tick < window:  # nothing to try, or too late to use what a trial would show
            continue
        try:
            child = apply_edits(k.recipe, [{"op": "set", "key": p["key"], "value": p["value"]}],
                                {"by": "agents", "proposal": p["id"], "backers": len(p["backers"]), "why": p["why"], "trial": True},
                                frame_cfg.get("locked") or ())
        except RecipeError:
            tried.add((p["key"], p["value"]))
            continue
        n += 1
        tried.add((p["key"], p["value"]))
        end -= window  # the second copy is paid for by ending the run sooner
        t0, before, at_fork = k.tick, k.rulebook.now(p["key"]), cost(k)
        twin, tb = fork(k, backend)
        apply_live(twin, k.recipe, child)
        other = dict(meta)
        await asyncio.gather(
            run(k, backend, t0 + window, out / "main", meta, seed, quiet=True, start=t0, horizon=end, final=False),
            run(twin, tb, t0 + window, out / f"trial-{n}", other, seed, quiet=True, start=t0, horizon=end, final=False))
        if meta.get("stopped") or other.get("stopped"):
            raise RunStopped(f"a trial stopped early ({meta.get('stopped') or other.get('stopped')}); the run has no score ({out})")
        a = area([s["metrics"].get("score") for s in k.snapshots], t0 + 1)
        b = area([s["metrics"].get("score") for s in twin.snapshots], t0 + 1)
        if a is None or b is None:
            raise RunStopped(f"a trial at t{t0} produced no score ({out})")
        kept = b > a + margin
        text = (f"{p['key']} = {show(p['value'])} (before: {show(before)}; proposal {p['id']}, backed by {len(p['backers'])}) was "
                f"tried over t{t0 + 1}-t{t0 + window} on a copy of this society: mean score {b:.3f} with it, {a:.3f} without it. "
                + ("The society carries on with it." if kept else "The society carries on without it."))
        dropped += (cost(k) if kept else cost(twin)) - at_fork
        if kept:  # the copy that switched is the society from here on
            k, backend = twin, tb
        trial = {"n": n, "t": t0, "window": window, "key": p["key"], "value": p["value"], "before": before, "proposal": p["id"],
                 "backers": len(p["backers"]), "with": b, "without": a, "kept": kept, "text": text, "recipe": child.id}
        k.rulebook.trials.append(trial)
        k.post("global", "lab", "result", "Trial: " + text, notify=False)
        les = lesson(f"Setting {p['key']} to {show(p['value'])} mid-run " + ("raised" if kept else "did not raise")
                     + f" the score ({b:.3f} against {a:.3f} over {window} ticks on a copy of the same society).",
                     f"{p['key']} was {show(before)}", world=recipe.world)
        k.rulebook.lessons.append(dict(les, by="lab", t=k.tick, entry=None))  # the lab's note of a measurement, not advice
        if log:
            log("  trial: " + text)
    meta["cost_usd"] = (cost(k) + dropped) if k.cost_fn and k.cost_fn() is not None else None
    if final:
        save(k, out / "main", meta)
    r = _result(k, recipe, seed, "generation", meta, out / "main")
    r.ended_with = k.recipe.id if k.recipe.id != recipe.id else ""
    r.ended_recipe = k.recipe.to_json() if r.ended_with else None
    (out / "main" / "result.json").write_text(json.dumps(asdict(r), default=list))
    return r
