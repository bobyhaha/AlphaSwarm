"""Running societies under recipes: single runs, paired runs and branches of one run (DESIGN.md section 20).

A frame is a base run configuration: the world, the budget and everything a recipe may not change. frame() builds the
design world's frames. Arms:
  open       nobody is placed in charge; every institution the recipe offers is there to be used (the default)
  chaired    a control: before the run begins one agent is made the lead of a team that holds every shared knob
  hierarchy  a control: a prescribed org chart that knows the task's modules holds the rights (teams, divisions, a program)
  single     one agent with every block and the same replies and simulations in total
  alone      the same agents with no communication
The budget is per agent (replies and simulations a tick), so the total grows with the team; a run at a given size
costs every arm the same replies and simulations. The arms without rules of their own (single, alone) run under the
starting recipe's memory, review and credit intervals, as every arm of the November runs did.

Task instances are kept apart by purpose (instance()): what selection scores a recipe on is never what a result is
reported on.

A comparison of a revision with its parent is always paired: the same task instance, the same starting state and the
same budget for both. Two kinds:
  pair()      two fresh societies from scratch; what the final decision rests on
  branches()  one society run to a saved state, then copied: one copy carries on, the others switch to a revision
              (recipe.apply_live). The copies share their whole history, so they differ only by the change. Cheaper
              and less noisy, and only possible for settings that can change mid-run in a world that can be copied.

A run that stopped early (its budget ran out, or the provider refused) has no score: the runner raises RunStopped and
nothing is recorded for it.

code_frame() builds the code world's frames (ProgramBench xsv). There a run's score is the held-out checks HEAD passes
(what selection reads), and a full run also ends with the hidden tests (Result.hidden), which selection never reads:
they are what a result is reported on. A code-world society cannot be copied, so its comparisons are all fresh runs
(alphateam/code_runs.py runs each in a process of its own).
"""
from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archive import niche_of
from .mock import MockBackend
from .recipe import Recipe, apply_live, attach, live_ok, options, seed as seed_recipe, to_cfg
from .runner import build, run
from .stats import area, divergence

PURPOSES = {"selection": 1_000_000, "report": 2_000_000, "probe": 3_000_000, "control": 4_000_000, "dry": 5_000_000}
CONFIRM, GENERATION, RETEST = 100_000, 500_000, 900_000  # selection instances by stage: a screen never meets a confirm
ARMS = ("open", "chaired", "hierarchy", "single", "alone")
EVERY_ARM = ("review_every", "credit_every", "continuity", "contest", "economy_share")  # options no arm goes without
# what names the task in a frame, by world: runs are compared only where these are the same
TASK_KEYS = {"design": ("dependency", "blocks", "supers", "sims"), "code": ("steps", "probes", "sandboxes", "held_every", "fake")}


class RunStopped(RuntimeError):
    """A run ended before its last tick (its budget ran out, or the provider refused): it has no score to use."""


class OverBudget(RuntimeError):
    """The spending cap of a runner is reached: nothing more is started."""


def instance(purpose: str, i: int) -> int:
    """The seed of the i-th task instance kept for a purpose. Selection and report never share an instance."""
    return PURPOSES[purpose] + int(i)


def frame(arm: str = "open", agents: int = 8, ticks: int = 24, replies: int = 9, sims: int = 9, blocks: int | None = None,
          supers: int | None = None, dependency: str = "coupled", attention: int = 6000, **world_opts) -> dict:
    """A design-world frame. Defaults are the eight-agent runs of 2026-11-04 (one agent per block, 9 replies and 9
    simulations an agent a tick, 24 ticks). `replies` counts an agent's closing reply, as tools/launch_design.sh does."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm '{arm}' ({', '.join(ARMS)})")
    if replies < 2:
        raise ValueError("an agent needs at least 2 replies a tick: one to act and its closing reply")
    blocks = blocks or agents
    if blocks > agents:
        raise ValueError(f"{blocks} blocks need at least {blocks} agents: an agent simulates one block, and a block nobody "
                         f"simulates cannot be settled")
    supers = supers or max(1, blocks // 4)
    fixed = {"dependency": dependency, "blocks": str(blocks), "supers": str(supers), "sims": str(sims), "economy": "1",
             "substrate": "1", "sync": "1", **{key: str(v) for key, v in world_opts.items()}}
    cfg = {"world": "design", "frame": arm, "condition": "self_org", "agents": agents, "ticks": ticks, "actions": 4,
           "attention": attention, "calls": replies - 1, "shocks": [], "token_budget": None, "cost_budget": None, "seed": 0,
           "structures": ["hierarchical"], "locked": {}, "force_opts": {}, "rules": True, "world_opts": fixed}
    if arm == "chaired":  # one team under one lead holds what is shared: the rights that would cut across it are fixed off
        cfg["locked"] = {"take_lead": "team", "holders": "none", "rights_by_credit": False}
        fixed["chair"] = "1"
    elif arm == "hierarchy":  # the prescribed org chart holds the rights, so nothing else hands them out
        cfg.update(condition="oracle", locked={"holders": "none", "claims": "off", "take_lead": "off", "rights_by_credit": False},
                   force_opts={"authority": "1"})
    elif arm in ("single", "alone"):
        base = options(seed_recipe("design"))
        fixed.update({key: base[key] for key in EVERY_ARM})
        if arm == "single":
            cfg.update(condition="single", calls=replies * agents - 1, rules=False)
            fixed.update(force_budget="1", messages="1")
        else:
            cfg.update(condition="isolated", rules=False)
            del fixed["sync"]
    return cfg


def code_frame(agents: int = 10, ticks: int = 24, replies: int = 3, attention: int = 6000, steps: int = 12,
               sandboxes: int = 3, probes: int = 480, fake: bool = False, closing: bool = False, **world_opts) -> dict:
    """A code-world frame: ProgramBench xsv rebuilt in Python (alphateam/code_world.py). Defaults are the ten-agent runs
    of 2026-11-04 (tools/launch_xsv.sh): 24 ticks of three replies an agent, 12 steps a tick, 480 probes of the
    reference in all, three containers at a time. Only the open arm exists. grade: a full run ends with the hidden
    tests. fake=True swaps in the in-memory harness (no Docker, no grader): dry runs and tests. closing=True: after
    the last tick of a full run, with the program final, each agent may still use the rules action (propose, back,
    lesson); it cannot move the run's score, so frames with and without it compare (it is not in the fingerprint)."""
    if replies < 2:
        raise ValueError("an agent needs at least 2 replies a tick: one to act and its closing reply")
    fixed = {"sync": "1", "board": "1", "final_moves": "1", "steps": str(steps), "sandboxes": str(sandboxes),
             "probes": str(probes), "ways": "0", "held_every": "merge", "heritage": "all",
             **{key: str(v) for key, v in world_opts.items()}}
    if fake:
        fixed["fake"] = "1"
    return {"world": "code", "frame": "open", "condition": "self_org", "agents": agents, "ticks": ticks, "actions": 4,
            "attention": attention, "calls": replies - 1, "shocks": [], "token_budget": None, "cost_budget": None, "seed": 0,
            "structures": ["hierarchical"], "locked": {}, "force_opts": {}, "rules": True, "world_opts": fixed, "grade": True,
            "closing": bool(closing)}


# The modules a code-world run loads (tests/test_code_recipes.py checks the list against a scripted run), and those no
# society ever runs under. A change to a module of the first kind may change what a run does, so it starts a new niche.
CODE_RUN = ("code_world", "directions", "explore", "expr", "game_model", "hidden_laws", "kernel", "llm", "metrics",
            "model_sandbox", "paired", "primitives", "recipe", "runner", "society")
OUTER = ("archive", "code_runs", "generations", "mock", "record", "selection", "stats")
# frame keys that cannot change what a run scores (closing: after the last tick; grade: the hidden tests at the end)
UNSCORED = ("seed", "closing", "grade", "cost_budget")


@functools.lru_cache(maxsize=None)
def code_version(world: str, root: str | None = None) -> str:
    """A short hash of the source a run in this world can load. Two runs with different hashes may not compare: on
    2026-11-03 a control looked 20 checks better only because the code had changed between the two waves of runs.
    For the code world it covers CODE_RUN, the worker (tools/code_job.py) and the harness's own modules; for any other
    world, every module of the package but OUTER. root: another folder of modules to hash (tests)."""
    here = Path(root) if root else Path(__file__).resolve().parent
    listed = world == "code" and root is None
    names = CODE_RUN if listed else sorted(q.stem for q in here.glob("*.py") if q.stem not in OUTER and q.stem != "__init__")
    files = [here / f"{name}.py" for name in sorted(names)]
    if listed:
        files.append(here.parent / "tools" / "code_job.py")
        from .code_world import HARNESS
        files += sorted(Path(HARNESS).glob("codetest/*.py"))
    h = hashlib.sha256()
    for path in files:
        if path.exists():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def fingerprint(frame_cfg: dict, meta: dict | None = None) -> dict:
    """What must stay the same for scores to be compared: the world, the budget, the arm, who plays the agents and at
    what effort, how much an agent may read, and the code a run loads. In the code world every fixed option of the
    frame counts, and so does every other key of the frame that can change a score."""
    wo = frame_cfg.get("world_opts") or {}
    world, meta = frame_cfg.get("world"), meta or {}
    fp = {"world": world, "arm": frame_cfg.get("frame"), "agents": frame_cfg["agents"],
          "ticks": frame_cfg["ticks"], "calls": frame_cfg.get("calls"),
          "task": ({key: wo[key] for key in sorted(wo)} if world == "code"
                   else {key: wo.get(key) for key in TASK_KEYS.get(world, TASK_KEYS["design"])}),
          "backend": meta.get("backend", "mock"), "model": meta.get("model"),
          "effort": meta.get("reasoning_effort"), "max_tokens": meta.get("max_tokens"),
          "attention": frame_cfg.get("attention"), "actions": frame_cfg.get("actions"), "code": code_version(world or "design")}
    if world == "code":
        named = ("world", "frame", "agents", "ticks", "calls", "world_opts", "attention", "actions") + UNSCORED
        fp["rest"] = {key: frame_cfg[key] for key in sorted(frame_cfg) if key not in named}
    return fp


@dataclass
class Result:
    recipe: str
    seed: int
    purpose: str
    score: float | None  # the last tick's score
    progress: float | None  # the mean score over the ticks counted (the area under the progress curve, per tick)
    scores: list = field(default_factory=list)  # by tick, from tick 0
    ticks: int = 0
    since: int = 0  # the tick after which this run differs from its pair (0: from the start)
    cost: float | None = None
    tokens: int = 0
    sims: int = 0  # simulations used (credit can buy an agent more, so they are counted, not assumed equal)
    ops: dict = field(default_factory=dict)  # actions taken after `since`: label -> count
    stopped: str | None = None
    proposals: list = field(default_factory=list)
    lessons: list = field(default_factory=list)
    actions: list = field(default_factory=list)  # actions this society's agents designed: name, procedure, adopters, uses
    trials: list = field(default_factory=list)  # changes the society tried on a branch of itself (alphateam/trial.py)
    suggestions: list = field(default_factory=list)  # rules that are not settings yet, suggested after the last tick
    ended_with: str = ""  # the recipe the run ended under, when a trial changed it
    ended_recipe: dict | None = None  # that recipe, in full
    dir: str = ""
    hidden: int | None = None  # a graded world's end-of-run grade (hidden tests passed); selection never reads it
    facts: dict = field(default_factory=dict)  # a few counts from the last tick, for whoever watches the runs

    def metric(self, name: str) -> float | None:
        return self.progress if name == "progress" else self.score


def behaviour(k, since: int = 0) -> Counter:
    """What a society did after tick `since`: actions by label, with refused ones counted apart."""
    out: Counter = Counter()
    for e in k.events:
        if e["t"] > since and e["op"] != "intent":
            out[e["op"] if e["ok"] else f"refused:{e['op'].split('.')[0]}"] += 1
    return out


def _run_cfg(cfg: dict) -> dict:
    """In the code world the interval between a society's reviews is a setting of the run, not a world option."""
    if cfg.get("world") == "code":
        cfg = dict(cfg, world_opts=dict(cfg.get("world_opts") or {}))
        every = int(float(cfg["world_opts"].pop("reflect", 0) or 0))
        cfg["reflect_window"] = every or None
    return cfg


def start(frame_cfg: dict, recipe: Recipe | None, seed: int, cost_budget: float | None = None):
    """A society at tick 0: the frame's world instance `seed`, under the recipe's rules. cost_budget (dollars) stops
    the run once its own calls have cost that much."""
    if recipe is None or not frame_cfg.get("rules", True):
        k = build(_run_cfg(frame_cfg), seed, frame_cfg["structures"][0])
    else:
        k = build(_run_cfg(to_cfg(recipe, frame_cfg)), seed, frame_cfg["structures"][0])
        attach(k, recipe, frame_cfg.get("locked"), proposals=frame_cfg.get("proposals", True))
        # closing: after the last tick of a full run each agent may still use the rules action (alphateam/runner.py run)
        k.closing_round = bool(frame_cfg.get("closing")) and getattr(k, "rulebook", None) is not None
    if cost_budget is not None:
        k.cost_budget = cost_budget if k.cost_budget is None else min(k.cost_budget, cost_budget)
    return k


def fork(k, backend):
    """A copy of a running society and a backend for it. The copy carries on from the same state: the same design,
    boards, credit, memories and random draws, and nothing ties it to the original."""
    if not getattr(k.env, "forkable", False):
        raise RuntimeError(f"a {type(k.env).__name__} cannot be copied mid-run (its state is not all in memory)")
    held = {name: getattr(k, name, None) for name in ("summarizer", "custom_runner", "cost_fn")}
    llm = getattr(k.env, "llm", None)
    for name in held:  # these point at the backend and the provider: they stay out of the copy and are set again by run()
        setattr(k, name, None)
    if llm is not None:
        k.env.llm = None
    try:
        twin = copy.deepcopy(k)
    finally:
        for name, v in held.items():
            setattr(k, name, v)
        if llm is not None:
            k.env.llm = llm
    return twin, backend.fork(k, twin)


def _result(k, recipe: Recipe | None, seed: int, purpose: str, meta: dict, out: Path, since: int = 0, base_cost: float = 0.0) -> Result:
    scores = [s["metrics"].get("score") for s in k.snapshots]
    book = getattr(k, "rulebook", None)
    exported = book.export() if book is not None else {"proposals": [], "lessons": [], "trials": []}
    cost = meta.get("cost_usd")
    hidden = getattr(k.env, "hidden", None)
    last = k.snapshots[-1]["metrics"] if k.snapshots else {}
    facts = {key: last[key] for key in ("held_out", "merges", "rejections", "law_version", "groups", "claims", "tools",
                                         "payments", "proposals") if last.get(key) is not None}
    designed = [{"name": c["name"], "description": c["description"], "procedure": c["procedure"], "params": c["params"],
                 "adopters": c["adopters"], "uses": c["uses"]} for c in k.export_custom() if not c["author"].startswith("heritage")]
    if isinstance(hidden, dict) and hidden:  # how the hidden grading went, and where the graded program is kept, so that a
        facts["grade"] = {key: hidden[key] for key in ("status", "error", "passed", "total") if hidden.get(key) is not None}
        try:  # run whose grading failed can be graded again later without anyone opening its work folder
            h = k.env.h
            facts["graded_as"] = f"{h.out.parent.name}-{h.repo.version[:8]}"
        except AttributeError:
            pass
    r = Result(recipe=recipe.id if recipe else "-", seed=seed, purpose=purpose, score=scores[-1] if scores else None,
               progress=area(scores, since + 1), scores=scores, ticks=k.tick, since=since,
               cost=None if cost is None else round(cost - base_cost, 5), tokens=k.total_tokens(),
               sims=sum(getattr(k.env, "used", {}).values()) if k.env is not None else 0,
               ops=dict(behaviour(k, since)), stopped=meta.get("stopped"), proposals=exported["proposals"],
               lessons=exported["lessons"], actions=designed, trials=exported["trials"],
               suggestions=exported.get("suggestions", []), dir=str(out),
               hidden=hidden.get("passed") if isinstance(hidden, dict) else None, facts=facts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(asdict(r), default=list))
    return r


async def run_once(frame_cfg: dict, recipe: Recipe | None, backend, seed: int, out: Path, purpose: str = "generation",
                   ticks: int | None = None, final: bool = False, meta: dict | None = None,
                   cost_budget: float | None = None) -> Result:
    """One society from scratch. final=True also writes the full run files (run.json, the viewer, the report)."""
    k = start(frame_cfg, recipe, seed, cost_budget)
    meta = dict(meta or {}, purpose=purpose, frame=frame_cfg.get("frame"), recipe=recipe.id if recipe else None,
                niche=niche_of(frame_cfg), seed=seed, config=frame_cfg)
    horizon = frame_cfg["ticks"]
    graded = bool(frame_cfg.get("grade")) and (ticks or horizon) >= horizon  # a full run of a graded world is graded
    await run(k, backend, ticks or horizon, out, meta, seed, quiet=True, horizon=horizon, final=final or graded)
    return _result(k, recipe, seed, purpose, meta, out)


async def pair(frame_cfg: dict, parent: Recipe, child: Recipe, make_backend, seed: int, out: Path, purpose: str = "confirm",
               ticks: int | None = None, meta: dict | None = None, cost_budget: float | None = None) -> tuple[Result, Result]:
    """Two fresh societies on the same task instance with the same budget: one under each recipe."""
    a, b = await asyncio.gather(
        run_once(frame_cfg, parent, make_backend(seed), seed, out / "parent", purpose, ticks, meta=meta, cost_budget=cost_budget),
        run_once(frame_cfg, child, make_backend(seed), seed, out / "child", purpose, ticks, meta=meta, cost_budget=cost_budget))
    return a, b


async def branches(frame_cfg: dict, parent: Recipe, children: list[Recipe], make_backend, seed: int, out: Path,
                   fork_at: int | None = None, ticks: int | None = None, purpose: str = "screen",
                   meta: dict | None = None, cost_budget: float | None = None) -> tuple[Result, list[Result]]:
    """One society under the parent to tick fork_at, then copied: one copy carries on, each other copy switches to a
    child. Scores are compared over the ticks after the fork, where the copies differ only by the change. The parent's
    result carries the cost of the shared ticks; a child's carries its own ticks only. cost_budget is for each copy."""
    for c in children:
        if not live_ok(parent, c):
            raise ValueError(f"{c.id} changes a setting that only applies from the start of a run; compare it with pair()")
    horizon = frame_cfg["ticks"]
    ticks = ticks or horizon
    fork_at = horizon // 2 if fork_at is None else fork_at
    if not 0 < fork_at < ticks:
        raise ValueError("fork_at must fall inside the run")
    backend = make_backend(seed)
    k = start(frame_cfg, parent, seed, cost_budget)
    meta = dict(meta or {}, purpose=purpose, frame=frame_cfg.get("frame"), recipe=parent.id, niche=niche_of(frame_cfg), seed=seed)
    trunk = dict(meta)
    await run(k, backend, fork_at, out / "trunk", trunk, seed, quiet=True, horizon=horizon, final=False)
    if trunk.get("stopped"):
        raise RunStopped(f"the shared part of a screen stopped early ({trunk['stopped']})")
    spent = (k.cost_fn() if k.cost_fn else None) or 0.0

    async def carry(recipe: Recipe, name: str, base: float) -> Result:
        twin, tb = fork(k, backend)
        if cost_budget is not None:
            twin.cost_budget = spent + cost_budget  # a copy's budget is for its own ticks
        if recipe.id != parent.id:
            apply_live(twin, parent, recipe)
        m = dict(meta, recipe=recipe.id, forked_at=fork_at)
        await run(twin, tb, ticks, out / name, m, seed, quiet=True, start=fork_at, horizon=horizon, final=False)
        return _result(twin, recipe, seed, purpose, m, out / name, since=fork_at, base_cost=base)

    results = await asyncio.gather(carry(parent, "parent", 0.0), *(carry(c, f"child-{c.id}", spent) for c in children))
    return results[0], list(results[1:])


class AlphaTeamRunner:
    """What the selection ladder (alphateam/selection.py) runs its stages on: real AlphaTeam societies in one frame."""

    def __init__(self, frame_cfg: dict, make_backend, out_root, probe_ticks: int = 3, dry_ticks: int = 2,
                 fork_at: int | None = None, trials: dict | None = None, cap_usd: float | None = None,
                 meta: dict | None = None, est_cost: float | None = None):
        self.frame = frame_cfg
        self.trials = dict(trials or {})  # e.g. {"every": 6, "window": 3, "backers": 2}: generation runs may try proposals
        # dollars: no run starts once what the runs reported reaches the cap, and each run is stopped at its share of
        # what is left. est_cost is what a full run is expected to cost, counted for a run whose provider reports none
        self.cap_usd, self.spent, self.est_cost = cap_usd, 0.0, est_cost
        self.meta = dict(meta or {})  # recorded with every run (the model, the preset)
        self.make_backend = make_backend  # seed -> a backend for one society
        self.out = Path(out_root)
        self.probe_ticks, self.dry_ticks, self.fork_at = probe_ticks, dry_ticks, fork_at
        self.niche = niche_of(frame_cfg)
        self._forkable = None

    @property
    def locked(self) -> dict:
        return self.frame.get("locked") or {}

    def fingerprint(self) -> dict:
        return fingerprint(self.frame, self.meta)

    def _dir(self, stage: str) -> Path:
        """A new directory for a stage. Never one that exists: a second session on the same output carries on."""
        if self.cap_usd is not None and self.spent >= self.cap_usd:
            raise OverBudget(f"the runs so far cost ${self.spent:.2f}, at the cap of ${self.cap_usd:.2f}")
        n = 1
        while (self.out / f"{stage}-{n:04d}").exists():
            n += 1
        path = self.out / f"{stage}-{n:04d}"
        path.mkdir(parents=True)
        return path

    def _share(self, runs: int) -> float | None:
        """What each of `runs` runs started together may spend: an equal share of what is left under the cap."""
        return None if self.cap_usd is None else max(0.0, self.cap_usd - self.spent) / max(1, runs)

    def _paid(self, *results: Result):
        """Count what the runs cost, then refuse the lot if any of them stopped early: a stopped run has no score."""
        for r in results:
            self.spent += r.cost if r.cost is not None else (self.est_cost or 0.0)
        for r in results:
            if r.stopped:
                raise RunStopped(f"a {r.purpose} run stopped early ({r.stopped}); its score is not used ({r.dir})")
        return results[0] if len(results) == 1 else results

    async def dry(self, recipe: Recipe) -> tuple[bool, str]:
        """Scripted agents run the recipe for a few ticks with no model calls: does it build and run?"""
        try:
            r = await run_once(self.frame, recipe, MockBackend(0), instance("dry", 0), self.out / "dry" / recipe.id, "dry",
                               self.dry_ticks)
        except Exception as exc:  # a recipe that cannot run is dropped here, whatever the reason
            return False, f"it breaks a run: {exc.__class__.__name__}: {exc}"
        if r.stopped:
            return False, f"it stops a run: {r.stopped}"
        return True, ""

    async def probe(self, parent: Recipe, child: Recipe, i: int) -> dict:
        """A short live run under each recipe on one instance, and a second run of the parent on another draw: did
        what agents do move by more than it moves between two runs of the same rules?"""
        seed, out, each = instance("probe", i), self._dir("probe"), self._share(3)
        a, b = await pair(self.frame, parent, child, self.make_backend, seed, out, "probe", self.probe_ticks, self.meta, each)
        again = await run_once(self.frame, parent, self.make_backend(seed + 1), seed, out / "parent-again", "probe",
                               self.probe_ticks, meta=self.meta, cost_budget=each)
        self._paid(a, b, again)
        moved, floor = divergence(Counter(a.ops), Counter(b.ops)), divergence(Counter(a.ops), Counter(again.ops))
        return {"moved": moved, "floor": floor, "cost": sum(x.cost or 0 for x in (a, b, again)), "runs": 3}

    async def screen(self, parent: Recipe, children: list[Recipe], i: int) -> list[tuple[Result, Result]]:
        """Each child against the parent on one selection instance; one (parent, child) pair of results a child.
        Children whose change can be made mid-run are branches of one run of the parent (they share its history up to
        the fork); the others run fresh beside a fresh run of the parent."""
        seed, out = instance("selection", i), self._dir("screen")
        if self._forkable is None:
            self._forkable = bool(getattr(start(self.frame, parent, seed).env, "forkable", False))
        live = [c for c in children if self._forkable and live_ok(parent, c)]
        rest = [c for c in children if c not in live]
        each = self._share((len(live) + 2 if live else 0) + (len(rest) + 1 if rest else 0))
        pairs: dict[str, tuple[Result, Result]] = {}
        if live:
            a, kids = await branches(self.frame, parent, live, self.make_backend, seed, out / "branches", self.fork_at,
                                     meta=self.meta, cost_budget=each)
            self._paid(a, *kids)
            pairs.update({c.id: (a, b) for c, b in zip(live, kids)})
        if rest:
            a = await run_once(self.frame, parent, self.make_backend(seed), seed, out / "parent", "screen", meta=self.meta,
                               cost_budget=each)
            kids = await asyncio.gather(*(run_once(self.frame, c, self.make_backend(seed), seed, out / f"child-{c.id}", "screen",
                                                   meta=self.meta, cost_budget=each) for c in rest))
            self._paid(a, *kids)
            pairs.update({c.id: (a, b) for c, b in zip(rest, kids)})
        return [pairs[c.id] for c in children]

    async def confirm(self, parent: Recipe, child: Recipe, i: int) -> tuple[Result, Result]:
        return self._paid(*await pair(self.frame, parent, child, self.make_backend, instance("selection", CONFIRM + i),
                                      self._dir("confirm"), meta=self.meta, cost_budget=self._share(2)))

    async def generation(self, recipe: Recipe, i: int) -> Result:
        """A full society on the recipe, with its run files kept: the run whose agents propose revisions and leave
        lessons. With trials set (a dict of alphateam/trial.py options), it may also try its proposals on itself."""
        seed = instance("selection", GENERATION + i)
        if self.trials:
            from .trial import run_with_trials
            return self._paid(await run_with_trials(self.frame, recipe, self.make_backend, seed, self._dir("generation"),
                                                    final=True, meta=self.meta, cost_budget=self._share(2), **self.trials))
        return self._paid(await run_once(self.frame, recipe, self.make_backend(seed), seed, self._dir("generation"),
                                         "generation", final=True, meta=self.meta, cost_budget=self._share(1)))

    async def score(self, recipe: Recipe | None, i: int, purpose: str = "report") -> Result:
        """One fresh run for a recipe's standing: a retest on a selection instance, or a report run on a held-out one."""
        seed = instance("report", i) if purpose == "report" else instance("selection", RETEST + i)
        return self._paid(await run_once(self.frame, recipe, self.make_backend(seed), seed, self._dir(purpose), purpose,
                                         meta=self.meta, cost_budget=self._share(1)))
