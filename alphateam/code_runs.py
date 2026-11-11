"""Recipes on the code world: societies run in processes of their own, a few at a time (DESIGN.md section 21).

The selection ladder (alphateam/selection.py) was built on the design world, where a society lives in memory, can be
copied mid-run and takes minutes. A code-world society (ProgramBench xsv, alphateam/code_world.py) keeps its state in
Docker containers and git copies, takes about half an hour, and cannot be copied. So here:

  - every run is a process of its own (tools/code_job.py): one that dies takes nothing with it, and is tried once more;
  - runs of one stage start side by side, at most `jobs` at a time, their starts spaced;
  - a screen is fresh full runs. Saved runs say a half-length run tells less per dollar than a full one (the gap
    between two rule sets about doubles, in units of its spread, between tick 12 and tick 24), so nothing is cut short;
  - the parent's side of a screen is what the parent scored in its full runs so far (its generation run, and its
    confirming and retest runs: their mean), not one more run of it: there is one task here, so a pair shares no
    instance luck that a fresh parent run would cancel, and a mean of several runs is a steadier mark than one run;
  - a full run ends with the hidden tests (Result.hidden). Selection reads the held-out checks only (Result.score).

What a run cost comes from its own report. A run whose process died is counted at what a run is expected to cost.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

from .mock import MockBackend
from .paired import CONFIRM, GENERATION, RETEST, AlphaTeamRunner, Result, RunStopped, instance, run_once
from .recipe import Recipe
from .stats import divergence

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "code_job.py"


class CodeRunner(AlphaTeamRunner):
    """What the selection ladder runs its stages on, in a code-world frame (alphateam/paired.py code_frame)."""

    def __init__(self, frame_cfg: dict, backend: dict, out_root, jobs: int = 5, stagger: float = 15.0,
                 max_run_usd: float | None = None, retries: int = 1, reuse_generation: bool = True,
                 python: str | None = None, log=print, **kw):
        super().__init__(frame_cfg, None, out_root, **kw)
        if frame_cfg.get("world") != "code":
            raise ValueError("a CodeRunner runs code-world frames (alphateam/paired.py code_frame)")
        if self.trials:
            raise ValueError("a code-world society cannot be copied, so it cannot try a change on a copy of itself")
        self.backend = dict(backend)  # {"kind": "mock"} or {"kind": "openrouter", "model": ..., "effort": ...}: the worker builds it
        self.jobs, self.stagger, self.max_run_usd, self.retries = max(1, jobs), stagger, max_run_usd, retries
        self.reuse_generation = reuse_generation
        self.python = python or sys.executable
        self.log = log
        self._forkable = False
        self._sem = self._gate = None  # made inside the running loop
        self._started = False
        self._ref: dict[str, Result] = {}  # recipe id -> its latest generation run
        # recipe id -> the frame its runs use instead of this runner's. Only for what cannot change a score: the closing
        # round after the last tick (so that it can be given to some recipes' runs and not paid for on all of them)
        self.frames: dict[str, dict] = {}
        self.standing = None  # recipe id -> the archive's rows of its full fresh runs (set by whoever holds the archive)

    # ------------------------------------------------------------------ one run

    async def _spawn(self, recipe: Recipe | None, seed: int, out: Path, purpose: str, ticks, cost_budget) -> Result | None:
        out.mkdir(parents=True, exist_ok=True)
        frame = self.frames.get(recipe.id, self.frame) if recipe else self.frame
        job = {"frame": frame, "recipe": recipe.to_json() if recipe else None, "seed": seed, "out": str(out),
               "purpose": purpose, "ticks": ticks, "cost_budget": cost_budget, "backend": self.backend, "meta": self.meta}
        (out / "job.json").write_text(json.dumps(job))
        if self._sem is None:
            self._sem, self._gate = asyncio.Semaphore(self.jobs), asyncio.Lock()
        async with self._sem:
            async with self._gate:  # starts are spaced, so that the containers of several runs do not all come up at once
                if self._started and self.stagger:
                    await asyncio.sleep(self.stagger)
                self._started = True
            with open(out / "job.log", "ab") as log:
                proc = await asyncio.create_subprocess_exec(self.python, str(WORKER), str(out / "job.json"), stdout=log,
                                                            stderr=asyncio.subprocess.STDOUT, cwd=str(ROOT))
                code = await proc.wait()
        path = out / "result.json"
        if code != 0 or not path.exists():
            return None
        d = json.loads(path.read_text())
        names = {f.name for f in dataclasses.fields(Result)}
        return Result(**{key: v for key, v in d.items() if key in names})

    async def _job(self, recipe: Recipe | None, seed: int, out: Path, purpose: str, ticks=None, share=None) -> Result | None:
        """One run, tried once more if its process dies. Its cost is counted here. None: it could not be run."""
        part = (ticks or self.frame["ticks"]) / self.frame["ticks"]
        caps = [v for v in (share, None if self.max_run_usd is None else self.max_run_usd * part) if v is not None]
        for attempt in range(self.retries + 1):
            path = out if not attempt else out.with_name(f"{out.name}-again{attempt}")
            r = await self._spawn(recipe, seed, path, purpose, ticks, min(caps) if caps else None)
            if r is not None:
                self.spent += r.cost if r.cost is not None else (self.est_cost or 0.0) * part
                return r
            self.spent += (self.est_cost or 0.0) * part  # it made its calls before it died
            self.log(f"  a {purpose} run died (see {path}/job.log)" + ("; trying it once more" if attempt < self.retries else ""))
        return None

    @staticmethod
    def _usable(runs, what: str):
        for r in runs:
            if r is None:
                raise RunStopped(f"a {what} run could not be run")
            if r.stopped:
                raise RunStopped(f"a {what} run stopped early ({r.stopped}); its score is not used ({r.dir})")
        return list(runs)

    # ------------------------------------------------------------------ the ladder's stages

    async def dry(self, recipe: Recipe) -> tuple[bool, str]:
        """Scripted agents on the in-memory harness for a few ticks, in this process: does the recipe build and run?"""
        fake = dict(self.frame, grade=False, world_opts=dict(self.frame["world_opts"], fake="1"))
        try:
            r = await run_once(fake, recipe, MockBackend(0), instance("dry", 0), self.out / "dry" / recipe.id, "dry", self.dry_ticks)
        except Exception as exc:  # a recipe that cannot run is dropped here, whatever the reason
            return False, f"it breaks a run: {exc.__class__.__name__}: {exc}"
        if r.stopped:
            return False, f"it stops a run: {r.stopped}"
        return True, ""

    async def probe(self, parent: Recipe, child: Recipe, i: int) -> dict:
        seed, out, share = instance("probe", i), self._dir("probe"), self._share(3)
        a, b, again = self._usable(await asyncio.gather(
            self._job(parent, seed, out / "parent", "probe", self.probe_ticks, share),
            self._job(child, seed, out / "child", "probe", self.probe_ticks, share),
            self._job(parent, seed, out / "parent-again", "probe", self.probe_ticks, share)), "probe")
        moved, floor = divergence(Counter(a.ops), Counter(b.ops)), divergence(Counter(a.ops), Counter(again.ops))
        return {"moved": moved, "floor": floor, "cost": sum(x.cost or 0 for x in (a, b, again)), "runs": 3}

    async def screen(self, parent: Recipe, children: list[Recipe], i: int) -> list[tuple[Result, Result]]:
        """Each child in a fresh full run, beside the parent's generation run (or a fresh run of the parent, if it has
        none): one (parent, child) pair of results a child."""
        seed, out = instance("selection", i), self._dir("screen")
        ref = self._ref.get(parent.id) if self.reuse_generation else None
        share = self._share(len(children) + (0 if ref else 1))
        jobs = [self._job(c, seed, out / f"child-{c.id}", "screen", None, share) for c in children]
        if ref is None:
            jobs.insert(0, self._job(parent, seed, out / "parent", "screen", None, share))
        runs = self._usable(await asyncio.gather(*jobs), "screen")
        if ref is None:
            return [(runs[0], b) for b in runs[1:]]
        rows = [x for x in (self.standing(parent.id) if self.standing else []) if x.get("score") is not None]
        score = sum(x["score"] for x in rows) / len(rows) if rows else ref.score
        known = [x["progress"] for x in rows if x.get("progress") is not None]
        a = dataclasses.replace(ref, score=score, progress=sum(known) / len(known) if known else ref.progress, cost=0.0,
                                proposals=[], lessons=[], actions=[])  # those runs were counted and heard already
        return [(a, b) for b in runs]

    async def confirm(self, parent: Recipe, child: Recipe, i: int) -> tuple[Result, Result]:
        return (await self.confirm_many(parent, child, [i]))[0]

    async def confirm_many(self, parent: Recipe, child: Recipe, idx: list[int]) -> list[tuple[Result, Result]]:
        """Fresh pairs side by side. A pair with a run that could not finish is left out; none left stops the stage."""
        dirs = [self._dir("confirm") for _ in idx]
        share = self._share(2 * len(idx))

        async def one(i: int, d: Path):
            seed = instance("selection", CONFIRM + i)
            return await asyncio.gather(self._job(parent, seed, d / "parent", "confirm", None, share),
                                        self._job(child, seed, d / "child", "confirm", None, share))

        pairs = await asyncio.gather(*(one(i, d) for i, d in zip(idx, dirs)))
        good = [(a, b) for a, b in pairs if a is not None and b is not None and not a.stopped and not b.stopped]
        if not good:
            raise RunStopped("no confirming pair finished")
        if len(good) < len(pairs):
            self.log(f"  {len(pairs) - len(good)} confirming pair(s) did not finish and are left out")
        return good

    async def generation(self, recipe: Recipe, i: int) -> Result:
        seed = instance("selection", GENERATION + i)
        (r,) = self._usable([await self._job(recipe, seed, self._dir("generation"), "generation", None, self._share(1))], "generation")
        self._ref[recipe.id] = r
        return r

    async def score(self, recipe: Recipe | None, i: int, purpose: str = "report") -> Result:
        return (await self.score_many(recipe, [i], purpose))[0]

    async def score_many(self, recipe: Recipe | None, idx: list[int], purpose: str = "report") -> list[Result]:
        """Fresh runs of one recipe side by side: retests on selection instances, or report runs on held-out ones."""
        dirs = [self._dir(purpose) for _ in idx]
        share = self._share(len(idx))
        seeds = [instance("report", i) if purpose == "report" else instance("selection", RETEST + i) for i in idx]
        runs = await asyncio.gather(*(self._job(recipe, s, d, purpose, None, share) for s, d in zip(seeds, dirs)))
        good = [r for r in runs if r is not None and not r.stopped]
        if not good:
            raise RunStopped(f"no {purpose} run finished")
        if len(good) < len(runs):
            self.log(f"  {len(runs) - len(good)} {purpose} run(s) did not finish and are left out")
        return good
