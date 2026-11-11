"""The selection ladder: how a revision earns its place in the archive (DESIGN.md section 20).

A society run costs money and its score varies from run to run, so a revision is tested in stages, cheapest first:

  1. typed    it is a valid, typed edit of its parent, it names no agent, and it was not tried before          free
  2. dry      it changes something (a rule the system enforces, or the advice), and scripted agents can run
              it for a few ticks with no model calls                                                           free
  3. probe    a short live run: do agents act differently under it than under its parent, by more than two
              runs of the parent differ? Required of a revision that changes only advice                       cents
  4. screen   a few pairs on selection instances: branches of one run where the change can be made mid-run,
              else fresh runs; the progress score decides which revisions go on                               dollars
  5. confirm  many fresh pairs on new instances; kept when the lower bound of the mean paired gain is above
              the threshold, and stopped early when the gain is clearly not there                              most of the cost
  6. archive  a kept revision is filed with its lineage and credit; a dropped one stays on record as a dead end

A first pass on single runs (confirm_pairs=0) stops after the screen: the best revision that beat its parent there is
taken on as a "lead" and seeds the next generation, marked as not confirmed. One run each says little (runs of one
recipe differ), so a lead is something to confirm later, not a result. In such a pass a revision is judged on every
run it has had so far against what its parent stands at now; it can be asked to have `lead_runs` runs of its own
before it is taken on; and one that is not ahead is set aside, not dropped, until it has had two runs.

mode="random" is the control arm: each generation one valid, runnable revision is drawn at random and kept with a
fixed chance, and nothing is spent on scoring it. Evolution has to beat that on the report instances.

The ladder talks to a runner (alphateam/paired.py AlphaTeamRunner for real societies; SyntheticRunner here for tests and for
sizing the stages by simulation) and writes every step to the archive (alphateam/archive.py).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .archive import FRESH, Archive
from .paired import Result
from .recipe import Recipe, RecipeError, enforced, inert, validate
from .stats import summary


@dataclass
class Policy:
    mode: str = "paired"  # or "random": the control arm
    probe: bool = True
    probe_all: bool = False  # probe every revision; by default only one that changes advice alone must pass it
    probe_factor: float = 1.5  # behaviour must move this many times what two runs of the parent differ by ...
    probe_min: float = 0.02  # ... and at least this much (Jensen-Shannon divergence of the mix of actions)
    screen_pairs: int = 4
    screen_min: float = 0.0  # mean gain a revision needs at the screen to go on
    screen_metric: str = "progress"
    finalists: int = 1
    confirm_pairs: int = 30  # 0: a first pass on single runs; the best screened revision is taken on as a lead, not confirmed
    looks: tuple = (10, 20, 30)  # after how many pairs the gain is looked at
    conf: float = 0.9  # one-sided confidence of the lower bound
    min_gain: float = 0.0  # kept when the lower bound is above this
    futility: float = 0.0  # at a look before the last: stop when the mean gain is at or below this
    metric: str = "score"  # what confirm compares
    retest_runs: int = 3  # a recipe has this many runs from scratch before it seeds a generation
    keep_p: float = 0.5  # random mode: the chance that the drawn revision is kept
    parent: str = "sample"  # who seeds a generation: drawn by standing ("sample"), or the newest recipe taken on ("tip")
    lead_runs: int = 1  # a first pass on single runs: the runs of its own a revision needs before it can be taken on
    lead_min: float = 0.0  # ... and how far its mean must be ahead of the parent's mark (about the spread of one run)
    retests: int = 0  # ... and how many set-aside revisions that are ahead of the parent so far get another run a generation


@dataclass
class Decision:
    parent: str
    child: str
    edits: list = field(default_factory=list)
    by: str = ""
    kept: bool = False
    stage: str = ""  # where it stopped: typed, dry, probe, screen, confirm, or kept
    why: str = ""
    stages: list = field(default_factory=list)
    runs: int = 0
    cost: float = 0.0


class Ladder:
    def __init__(self, archive: Archive, runner, policy: Policy | None = None, rng: random.Random | None = None, log=None):
        self.archive, self.runner, self.policy = archive, runner, policy or Policy()
        if self.policy.mode not in ("paired", "random"):
            raise ValueError("a policy's mode is paired or random")
        if self.policy.mode == "paired" and self.policy.confirm_pairs == 1:
            raise ValueError("a revision needs at least 2 confirming pairs: one pair gives no bound")
        if self.policy.mode == "paired" and not self.policy.confirm_pairs and not self.policy.screen_pairs:
            raise ValueError("with no confirming pairs a revision is judged at its screen: it needs at least 1 screen run")
        if self.policy.parent not in ("sample", "tip"):
            raise ValueError("a policy's parent is sample or tip")
        self.rng = rng or random.Random(0)
        self.log = log or (lambda *_: None)

    def _run(self, r: Result, purpose: str):
        graded = {"hidden": r.hidden} if getattr(r, "hidden", None) is not None else {}
        self.archive.record_run(r.recipe, self.runner.niche, purpose, r.seed, r.score, r.progress, r.cost, dir=r.dir,
                                sims=r.sims, **graded)
        self.archive.record_voices(r, self.runner.niche)

    def _log(self, d: Decision, judged: bool):
        self.archive.decide(parent=d.parent, child=d.child, edits=d.edits, by=d.by, kept=d.kept, judged=judged, stage=d.stage,
                            why=d.why, stages=d.stages, runs=d.runs, cost=round(d.cost, 4), niche=self.runner.niche,
                            mode=self.policy.mode)
        word = ("kept" if d.stage == "kept" else "taken on as a lead") if d.kept else ("dropped at " if judged else "set aside at ") + d.stage
        self.log(f"  {word}: {d.child} "
                 f"({'; '.join(_edit(e) for e in d.edits) or 'no edits'}): {d.why}")

    def _close(self, d: Decision, stage: str, why: str, kept: bool = False, status: str | None = None) -> Decision:
        """A revision judged: kept (or taken on as a lead), or dropped for good (it stays on record and is not tried again)."""
        d.stage, d.why, d.kept = stage, why, kept
        if d.child in self.archive.records and self.archive.records[d.child]["status"] == "candidate":
            self.archive.set_status(d.child, status or ("kept" if kept else "dropped"), why, None if kept else stage)
        self._log(d, True)
        return d

    def _wait(self, d: Decision, stage: str, why: str) -> Decision:
        """A revision not judged on its own merits (another went on in its place): it stays a candidate."""
        d.stage, d.why, d.kept = stage, why, False
        if d.child in self.archive.records:
            self.archive.set_status(d.child, "candidate", why, stage)
        self._log(d, False)
        return d

    async def ensure_tested(self, recipe: Recipe) -> int:
        """Re-run a recipe from scratch until it has enough runs to stand on (before it seeds a generation)."""
        niche, n = self.runner.niche, 0
        many = getattr(self.runner, "score_many", None)
        need = self.policy.retest_runs - len(self.archive.runs(recipe.id, niche, FRESH))
        if many is not None and need > 0:  # a runner that can run them side by side
            for r in await many(recipe, [self.archive.next("retest") for _ in range(need)], "retest"):
                self._run(r, "retest")
                n += 1
            return n
        while len(self.archive.runs(recipe.id, niche, FRESH)) < self.policy.retest_runs:
            self._run(await self.runner.score(recipe, self.archive.next("retest"), "retest"), "retest")
            n += 1
        return n

    async def consider(self, parent: Recipe, candidates: list[Recipe]) -> list[Decision]:
        """Take a generation's candidate revisions of one parent up the ladder. If a run stops early or the spending
        cap is reached, the exception passes through: what was not judged stays a candidate and can be taken up again."""
        p, niche = self.policy, self.runner.niche
        out: list[Decision] = []
        alive: list[tuple[Recipe, Decision]] = []
        for c in candidates:
            d = Decision(parent.id, c.id, list(c.origin.get("edits", [])), str(c.origin.get("by", "")))
            out.append(d)
            prior = self.archive.seen(c)
            twice = any(x.id == c.id for x, _ in alive)
            if twice or (prior is not None and prior["status"] != "candidate"):
                d.stage = "typed"
                d.why = ("the same revision was proposed twice" if twice else
                         f"tried before: {prior['status']}" + (f" ({prior['why']})" if prior.get("why") else ""))
                self._log(d, True)
                continue
            try:
                validate(c, self.runner.locked)
            except RecipeError as exc:
                self.archive.add(c, "candidate", niche)
                self._close(d, "typed", str(exc))
                continue
            self.archive.add(c, "candidate", niche)  # a candidate taken up again keeps its record
            why = inert(parent, c)
            if why:
                self._close(d, "dry", why)
                continue
            ok, why = await self.runner.dry(c)
            d.stages.append({"stage": "dry", "passed": ok, "why": why})
            if not ok:
                self._close(d, "dry", why)
                continue
            alive.append((c, d))
        if p.mode == "random":
            return self._draw(out, alive)
        for c, d in list(alive):  # the probe: required of a revision that changes only what agents are told
            if p.probe and (p.probe_all or not enforced(c)):
                res = await self.runner.probe(parent, c, self.archive.next("probe"))
                moved = res["moved"] > max(p.probe_min, p.probe_factor * res["floor"])
                d.stages.append({"stage": "probe", "passed": moved, "moved": res["moved"], "floor": res["floor"]})
                d.runs += res.get("runs", 0)
                d.cost += res.get("cost") or 0
                if not moved:
                    alive.remove((c, d))
                    self._close(d, "probe", f"agents acted as they did under the parent (their actions moved by "
                                            f"{res['moved']:.3f}; two runs of the parent differ by {res['floor']:.3f})")
        finalists = alive[:p.finalists]
        if alive and p.screen_pairs:
            gains: dict[str, list] = {c.id: [] for c, _ in alive}
            marks: list[float] = []  # what the parent scored in this generation's screens
            for _ in range(p.screen_pairs):
                pairs = await self.runner.screen(parent, [c for c, _ in alive], self.archive.next("screen"))
                shared = Counter(id(a) for a, _ in pairs)  # children that were branches of one run share its parent run
                heard = set()
                for (c, d), (a, b) in zip(alive, pairs):
                    x, y = a.metric(p.screen_metric), b.metric(p.screen_metric)
                    self.archive.record_pair(parent.id, c.id, niche, "screen", a.seed, x, y, p.screen_metric, since=b.since)
                    if id(a) not in heard:  # what agents said under the parent, once a run
                        heard.add(id(a))
                        self.archive.record_voices(a, niche)
                    self.archive.record_voices(b, niche, after=b.since)  # under the child: only what came after the switch
                    if x is not None and y is not None:
                        gains[c.id].append(y - x)
                    if x is not None:
                        marks.append(x)
                    d.runs += 1
                    d.cost += (b.cost or 0) + (a.cost or 0) / shared[id(a)]
            single = not p.confirm_pairs  # a first pass on single runs
            if single and marks:  # every run a revision has had counts, against what the parent stands at now
                mark = sum(marks) / len(marks)
                for c, _ in alive:  # whatever recipe was the parent when a run was made: the run is the revision's own
                    had = [q["b"] for q in self.archive.pairs(c.id, ("screen",))
                           if q.get("metric") == p.screen_metric and q.get("b") is not None]
                    gains[c.id] = [b - mark for b in had]
            mean = {cid: (sum(g) / len(g) if g else -math.inf) for cid, g in gains.items()}
            ranked = sorted(alive, key=lambda cd: -mean[cd[0].id])
            bar = max(p.screen_min, p.lead_min) if single else p.screen_min
            enough = lambda cid: not single or (len(gains[cid]) >= p.lead_runs and mean[cid] > bar)  # noqa: E731
            finalists = [cd for cd in ranked if mean[cd[0].id] > p.screen_min and enough(cd[0].id)][:p.finalists]
            for c, d in ranked:
                n = len(gains[c.id])
                d.stages.append({"stage": "screen", "passed": (c, d) in finalists, "mean_gain": round(mean[c.id], 5),
                                 "pairs": n, "metric": p.screen_metric})
                if (c, d) in finalists:
                    continue
                text = f"mean gain {mean[c.id]:+.3f} over {n} " + (f"run{'s' if n != 1 else ''}" if single else "pairs")
                if mean[c.id] > p.screen_min and not enough(c.id):  # ahead so far: not by enough, or on too few runs
                    need = (f"it needs {p.lead_runs} runs to be taken on" if n < p.lead_runs else
                            f"not by the {p.lead_min:.3f} it takes to be taken on")
                    self._wait(d, "screen", text + f"; ahead of the parent, and {need}")
                elif mean[c.id] > p.screen_min:  # it passed the screen; another revision went on in its place
                    self._wait(d, "screen", text + "; another revision screened better")
                elif single and n < max(2, p.lead_runs):  # one run settles nothing: runs of one recipe differ
                    self._wait(d, "screen", text + "; not ahead of the parent so far")
                else:
                    self._close(d, "screen", text)
        else:
            for c, d in alive[p.finalists:]:
                self._wait(d, "screen", "not among the finalists")
        for c, d in finalists:
            if p.confirm_pairs:
                await self._confirm(parent, c, d)
            else:  # a first pass on single runs: the best screened revision goes on, marked as not confirmed
                n = len(gains[c.id])
                d.stages.append({"stage": "lead", "passed": True, "mean_gain": round(mean[c.id], 5), "pairs": n})
                self._close(d, "lead", f"taken on after {n} run{'s' if n != 1 else ''} of its own, not confirmed: "
                                       f"mean gain {mean[c.id]:+.3f}", True, "lead")
        return out

    async def _confirm(self, parent: Recipe, c: Recipe, d: Decision):
        p, niche = self.policy, self.runner.niche
        looks = sorted({n for n in p.looks if n <= p.confirm_pairs} | {p.confirm_pairs})
        diffs: list[float] = []
        done = 0
        for j in looks:  # the pairs up to a look; a runner that can run pairs side by side gets them together
            async for a, b in self._pairs(parent, c, j - done):
                self._run(a, "confirm")
                self._run(b, "confirm")
                x, y = a.metric(p.metric), b.metric(p.metric)
                self.archive.record_pair(parent.id, c.id, niche, "confirm", a.seed, x, y, p.metric)
                if x is not None and y is not None:
                    diffs.append(y - x)
                d.runs += 2
                d.cost += (a.cost or 0) + (b.cost or 0)
            done = j
            s = summary(diffs, p.conf)
            if j == looks[-1]:
                kept = s["lcb"] is not None and s["lcb"] > p.min_gain
                d.stages.append({"stage": "confirm", "passed": kept, **s, "metric": p.metric})
                bound = "no bound" if s["lcb"] is None else f"lower bound {s['lcb']:+.3f}"
                gain = "no paired scores" if s["mean"] is None else f"paired gain {s['mean']:+.3f}, {bound} over {s['n']} pairs"
                self._close(d, "kept" if kept else "confirm", gain, kept)
                return
            if s["mean"] is not None and s["mean"] <= p.futility:
                d.stages.append({"stage": "confirm", "passed": False, **s, "metric": p.metric, "stopped_early": True})
                self._close(d, "confirm", f"stopped after {s['n']} pairs: mean paired gain {s['mean']:+.3f}")
                return

    async def _pairs(self, parent: Recipe, c: Recipe, n: int):
        """The next n confirming pairs, as they come: one after another, or together where the runner can."""
        many = getattr(self.runner, "confirm_many", None)
        if many is not None:
            for pair in await many(parent, c, [self.archive.next("confirm") for _ in range(n)]):
                yield pair
        else:
            for _ in range(n):
                yield await self.runner.confirm(parent, c, self.archive.next("confirm"))

    def _draw(self, out: list[Decision], alive: list) -> list[Decision]:
        """The control arm: one valid, runnable revision drawn at random and kept with a fixed chance; no scoring."""
        pick = self.rng.choice(alive) if alive else None
        keep = pick is not None and self.rng.random() < self.policy.keep_p
        for c, d in alive:
            if pick is not None and c.id == pick[0].id:
                self._close(d, "kept" if keep else "screen", "kept at random (control)" if keep else "drawn and not kept (control)", keep)
            else:
                self._wait(d, "screen", "not drawn (control)")
        return out


def _edit(e: dict) -> str:
    if e.get("op") == "set":
        return f"{e['key']}={e['value']}"
    return f"{e.get('op')} {e.get('id') or e.get('name') or ''}".strip()


# ---------------------------------------------------------------------------------------------- a stand-in for societies

class SyntheticRunner:
    """A stand-in for societies with a known truth, for testing the ladder and sizing its stages by simulation.

    Each recipe has a true value (base + truth(recipe)). A run returns it plus noise with standard deviation `sd`. A
    share `rho` of the noise variance belongs to the task instance and is common to both runs of a pair, so paired
    differences vary less than single runs; branches of one run share more (`rho_branch`). Nothing here says how real
    societies behave: sd and rho come from saved runs or are swept."""

    def __init__(self, truth, sd: float = 0.45, rho: float = 0.3, rho_branch: float = 0.7, base: float = 0.4, broken=(),
                 still=(), seed: int = 0, niche: str = "synthetic/-/open/n8", cost: float = 0.3, locked: dict | None = None):
        self.truth = truth if callable(truth) else (lambda r, t=dict(truth): t.get(r.id, 0.0))
        self.sd, self.rho, self.rho_branch, self.base = sd, rho, rho_branch, base
        self.broken, self.still = set(broken), set(still)
        self.rng = random.Random(seed)
        self.niche, self.cost, self.locked = niche, cost, dict(locked or {})
        self.count = {"dry": 0, "probe": 0, "screen": 0, "confirm": 0, "generation": 0, "score": 0}

    def fingerprint(self) -> dict:
        return {"world": "synthetic", "sd": self.sd, "rho": self.rho}

    def value(self, recipe: Recipe) -> float:
        return self.base + self.truth(recipe)

    def _one(self, recipe: Recipe, shared: float, rho: float, purpose: str, seed: int, cost: float, since: int = 0) -> Result:
        s = self.value(recipe) + self.sd * (math.sqrt(rho) * shared + math.sqrt(1 - rho) * self.rng.gauss(0, 1))
        return Result(recipe.id, seed, purpose, round(s, 5), round(s, 5), cost=cost, since=since)

    async def dry(self, recipe: Recipe):
        self.count["dry"] += 1
        return (recipe.id not in self.broken, "it breaks a run" if recipe.id in self.broken else "")

    async def probe(self, parent: Recipe, child: Recipe, i: int) -> dict:
        self.count["probe"] += 3
        return {"moved": 0.0 if child.id in self.still else 0.3, "floor": 0.05, "cost": 0.03 * self.cost, "runs": 3}

    async def screen(self, parent: Recipe, children: list[Recipe], i: int):
        shared = self.rng.gauss(0, 1)
        self.count["screen"] += 1 + len(children)
        half = self.cost / 2
        a = self._one(parent, shared, self.rho_branch, "screen", i, half, 1)
        return [(a, self._one(c, shared, self.rho_branch, "screen", i, half, 1)) for c in children]

    async def confirm(self, parent: Recipe, child: Recipe, i: int):
        shared = self.rng.gauss(0, 1)
        self.count["confirm"] += 2
        return (self._one(parent, shared, self.rho, "confirm", i, self.cost), self._one(child, shared, self.rho, "confirm", i, self.cost))

    async def generation(self, recipe: Recipe, i: int) -> Result:
        self.count["generation"] += 1
        return self._one(recipe, self.rng.gauss(0, 1), self.rho, "generation", i, self.cost)

    async def score(self, recipe: Recipe, i: int, purpose: str = "report") -> Result:
        self.count["score"] += 1
        shared = random.Random(f"{purpose}:{i}").gauss(0, 1)  # recipes scored on the same instance share its luck
        return self._one(recipe, shared, self.rho, purpose, i, self.cost)
