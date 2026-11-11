"""Benchmarks as worlds: a suite of tasks that the whole society answers.

A task world turns a benchmark into something the society works on with the same primitives as the hidden-law world:
- Each task has a family (the 'module' for the mirroring metric, e.g. a DiscoveryBench dataset or a SciGym system class)
  and a domain (the 'super-module'). Task ids are shuffled and say nothing about either.
- The society's answer to a task is the latest submission anyone made for it, like a published law: one answer per task.
- Every agent starts with a few home tasks (so they don't all start on the first one) but can open any task.
- Benchmark tools that take time (running code, simulating) are async, so agents don't wait for each other.
- Scoring uses the benchmark's own grader and is hidden from agents. Graders can be slow or cost money (an LLM judge), so
  they run at chosen ticks and at the end (`finalize`), not every tick; score() returns the last evaluation.
- Practice split (world options practice=N, feedback=1, feedback_every=E): N tasks are practice tasks and the score
  counts only the others (held out). With feedback on, practice tasks are marked as such and graded every E ticks, and
  their grades and authors are posted on the global board; with feedback off the split exists but stays hidden, which
  is the control.

Subclasses add their own tools (TOOLS, tool names) and implement `grade` (and optionally `statement`, `observe_extra`).
"""
from __future__ import annotations

import inspect
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..kernel import ActionError, clip


@dataclass
class Task:
    id: str  # e.g. "T17"
    title: str  # one line shown in listings
    family: str  # the module this task belongs to (hidden from agents unless the benchmark shows it)
    domain: str  # the super-module
    statement: str  # what an agent sees when it opens the task
    meta: dict = field(default_factory=dict)  # benchmark data, including anything hidden (gold answers)


@dataclass
class Submission:
    task: str
    answer: str
    rationale: str
    author: str
    group: str | None
    tick: int


SUBMIT_TOOL = {
    "name": "submit",
    "description": "Set the society's answer to a task (replaces the previous answer). Optionally on behalf of a group you may act for.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string", "description": "task id, e.g. T12"},
        "answer": {"type": "string"},
        "rationale": {"type": "string", "description": "optional: how the answer was obtained (methods, evidence)"},
        "as": {"type": "string", "description": "optional group id"}},
        "required": ["task", "answer"]}}

CLAIM_TOOL = {
    "name": "claim",
    "description": "Claim a task so others can see you are working on it (one claimant at a time; a claim lapses after "
                   "3 ticks without an answer from the claimant). op release gives it up.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string"}, "op": {"type": "string", "enum": ["claim", "release"]}}, "required": ["task"]}}

OPEN_TOOL = {
    "name": "open_task",
    "description": "Read a task's full statement (data, goal, materials).",
    "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}


class TaskWorld:
    name = "tasks"
    unit = "task"
    TOOLS: list[dict] = []  # benchmark tools; subclasses extend
    search_kinds = ("bench",)  # not "tasks": that kind lists the kernel's delegated tasks
    has_modules = True
    validate_every = 0
    claims_on = False  # optional atomic claim primitive (world option claims=1)
    shuffle_search = False  # world option search_order=random: each agent sees matching tasks in its own order
    CLAIM_TTL = 3
    isolated = False  # no-communication control: agents work only on their own home tasks
    practice: frozenset = frozenset()  # practice task ids (graded during the run when feedback is on; not scored)
    feedback_on = False
    feedback_every = 3
    llm = None  # async (system, prompt, model=None, max_tokens=2000) -> str, set by the runner for live backends
    workdir = "/tmp/alphateam_bench"  # set by the runner to <run dir>/work

    def __init__(self, n_agents: int, tasks: list[Task], seed: int = 0, home: int = 3, eval_at: tuple[int, ...] = (),
                 single: bool = False, world_agents: int | None = None):
        self.rng = random.Random(seed)
        order = list(range(len(tasks)))
        self.rng.shuffle(order)
        self.tasks: dict[str, Task] = {}
        for new, old in enumerate(order, 1):  # shuffled ids, so ids leak nothing about families
            t = tasks[old]
            t.id = f"T{new}"
            self.tasks[t.id] = t
        fams = sorted({t.family for t in self.tasks.values()})
        doms = sorted({t.domain for t in self.tasks.values()})
        dom_idx = {d: i for i, d in enumerate(doms)}
        self.family_idx = {f: i for i, f in enumerate(fams)}
        fam_dom = {t.family: t.domain for t in self.tasks.values()}
        self.modules = [{"id": i, "name": f, "super": dom_idx[fam_dom[f]], "level": 0, "outputs": []} for i, f in enumerate(fams)]
        self.n_super = len(doms)
        self.domains = doms
        self.submissions: dict[str, Submission] = {}
        self.history: list[dict] = []
        self.focus: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self.evaluations: list[dict] = []
        self.task_scores: dict[str, float] = {}
        self._graded: dict[str, tuple] = {}  # task -> the answer that was graded
        self.claims: dict[str, tuple[str, int]] = {}  # task -> (agent, tick claimed)
        self.eval_at = set(eval_at)
        self.single = single
        self.used: dict[str, int] = defaultdict(int)  # benchmark-specific budget use per agent
        ids = list(self.tasks)
        aids = [f"A{i + 1}" for i in range(n_agents)]
        self.home: dict[str, list[str]] = {}
        if single:
            self.home[aids[0]] = ids[:]
        elif home <= 0:  # nothing assigned: the society divides the work itself
            self.home = {a: [] for a in aids}
        else:  # deal tasks round-robin from a shuffled deck so every task has about the same number of home agents
            deck = []
            for a in aids:
                hand = []
                while len(hand) < min(home, len(ids)):
                    if not deck:
                        deck = ids[:]
                        self.rng.shuffle(deck)
                    t = deck.pop()
                    if t not in hand:
                        hand.append(t)
                self.home[a] = hand

    # ------------------------------------------------------------------ hooks for subclasses

    def statement(self, k, aid: str, t: Task) -> str:
        return t.statement

    def observe_extra(self, k, aid: str) -> str:
        return ""

    async def grade(self, k, tasks: list[str]) -> dict[str, float]:
        """Score the society's current answers for these tasks (0..1). Unanswered tasks are never passed."""
        raise NotImplementedError

    def mission(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ kernel interface

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t["name"] for t in self.all_tools)

    @property
    def all_tools(self) -> list[dict]:
        return [OPEN_TOOL, *self.TOOLS, SUBMIT_TOOL] + ([CLAIM_TOOL] if self.claims_on else [])

    def claimant(self, k, tid: str) -> str | None:
        c = self.claims.get(tid)
        if not c or tid in self.submissions and self.submissions[tid].author == c[0] or k.tick - c[1] >= self.CLAIM_TTL:
            return None
        return c[0]

    def _claim(self, k, aid, a):
        t = self.task(a.get("task"))
        if str(a.get("op") or "claim") == "release":
            k._need(self.claims.get(t.id, (None,))[0] == aid, f"you hold no claim on {t.id}")
            del self.claims[t.id]
            return "lab.release", f"released {t.id}", t.id
        holder = self.claimant(k, t.id)
        if holder not in (None, aid):
            raise ActionError(f"{t.id} is claimed by {holder} (since t{self.claims[t.id][1]})")
        self.claims[t.id] = (aid, k.tick)
        self.note_focus(k, aid, t, 0.5)
        return "lab.claim", f"claimed {t.id} until t{k.tick + self.CLAIM_TTL - 1} (renew by claiming again)", t.id

    def setup(self, k):
        for aid, hand in self.home.items():
            if not hand:
                k.post(f"private:{aid}", "bench", "note", f"No {self.unit}s are assigned to anyone: search what=bench lists all "
                       f"{len(self.tasks)} {self.unit}s and who has answered them; open_task shows a statement.", notify=False)
                continue
            lines = [f"{t}: {clip(self.tasks[t].title, 160)}" for t in hand[:12]]
            k.post(f"private:{aid}", "bench", "note",
                   f"Your home {self.unit}s (anyone may work on any {self.unit}; open_task shows the full statement):\n" + "\n".join(lines)
                   + (f"\n…and {len(hand) - 12} more" if len(hand) > 12 else ""), notify=False)

    def on_tick_start(self, k):
        pass

    def set_practice(self, n: int, feedback: bool, every: int, seed: int):
        ids = sorted(self.tasks, key=lambda t: int(t[1:]))
        self.practice = frozenset(random.Random(f"practice:{seed}").sample(ids, min(n, len(ids) - 1)))
        self.feedback_on, self.feedback_every = feedback, max(1, every)
        self.practice_reviews: list[dict] = []

    async def on_tick_end(self, k):
        if k.tick in self.eval_at:
            await self.evaluate(k)
        if self.feedback_on and k.tick % self.feedback_every == 0:
            await self._practice_review(k)

    async def _practice_review(self, k):
        """Grade the practice tasks' current answers and post the grades, with their authors, on the global board."""
        await self._grade_fresh(k, [t for t in self.practice if t in self.submissions])
        ids = sorted(self.practice, key=lambda t: int(t[1:]))
        vals = [self.task_scores.get(t, 0.0) for t in ids]
        self.practice_reviews.append({"t": k.tick, "mean": round(sum(vals) / len(vals), 4),
                                      "scores": {t: self.task_scores.get(t) for t in ids}})
        parts = []
        for t in ids:
            sub = self.submissions.get(t)
            parts.append(f"{t} {self.task_scores[t]:.2f} (by {sub.author}{'/' + sub.group if sub.group else ''} t{sub.tick})"
                         if sub else f"{t} unanswered")
        k.post("global", "bench", "result", f"Practice review at t{k.tick}: " + "; ".join(parts) +
               f". Practice mean {sum(vals) / len(vals):.3f} (0-1, the benchmark's own grader; held-out "
               f"{self.unit}s are graded only at the end and are what counts).", notify=False)

    async def finalize(self, k):
        await self.evaluate(k, final=True)

    def act(self, k, aid: str, name: str, a: dict):
        return getattr(self, f"_{name}")(k, aid, a)

    def task(self, ref) -> Task:
        tid = str(ref or "").strip().upper()
        if re.fullmatch(r"\d+", tid):
            tid = f"T{tid}"
        if tid not in self.tasks:
            raise ActionError(f"no {self.unit} '{ref}' ({self.unit}s are T1..T{len(self.tasks)}; search what=bench)")
        return self.tasks[tid]

    def note_focus(self, k, aid: str, t: Task, w: float = 1.0):
        self.focus[aid].append((k.tick, self.family_idx[t.family], w))

    def _open_task(self, k, aid, a):
        t = self.task(a.get("task"))
        k._need(not self.isolated or t.id in self.home.get(aid, []), f"in this society you may only open your own {self.unit}s")
        self.note_focus(k, aid, t, 0.5)
        sub = self.submissions.get(t.id)
        status = (f"\nCurrent society answer (by {sub.author}{'/' + sub.group if sub.group else ''}, t{sub.tick}): "
                  f"{clip(sub.answer, 600)}" if sub else "\nNo answer submitted yet.")
        return "lab.open_task", f"{t.id}: {self.statement(k, aid, t)}{status}", t.id

    def _submit(self, k, aid, a):
        t = self.task(a.get("task"))
        k._need(not self.isolated or t.id in self.home.get(aid, []), f"in this society you may only answer your own {self.unit}s")
        answer = str(a.get("answer") or "").strip()
        k._need(answer, "the answer is empty")
        group = None
        if a.get("as"):
            group = k._group(a["as"])
            k._need(k.can_act_for(aid, group), f"you may not act for {group}")
        prev = self.submissions.get(t.id)
        self.submissions[t.id] = Submission(t.id, answer[:6000], str(a.get("rationale") or "")[:4000], aid, group, k.tick)
        self.history.append({"t": k.tick, "task": t.id, "author": aid, "group": group})
        self.note_focus(k, aid, t, 3.0)
        k.tick_counts["published"] += 1
        if prev and prev.author != aid:
            k.notify(prev.author, aid, f"{aid} replaced your answer to {t.id}")
        msg = f"submitted the society's answer to {t.id}"
        if prev:
            msg += f", replacing {prev.author}'s from t{prev.tick}"
        return "lab.submit", msg + ".", t.id

    def search(self, k, aid, what, words):
        pool = list(self.home.get(aid, [])) if self.isolated else list(self.tasks)
        ids = [w.upper() for w in words if re.fullmatch(r"t\d+", w, re.I)]
        if ids:
            pool = [t for t in ids if t in self.tasks and (not self.isolated or t in self.home.get(aid, []))]
        if "mine" in words or "home" in words:
            pool = [t for t in pool if t in self.home.get(aid, [])]
        if "unanswered" in words or "open" in words or "missing" in words:
            pool = [t for t in pool if t not in self.submissions]
        if "unclaimed" in words:
            pool = [t for t in pool if not self.claimant(k, t)]
        authors = {w.upper() for w in words if re.fullmatch(r"a\d+", w, re.I)}
        if authors:
            pool = [t for t in pool if t in self.submissions and self.submissions[t].author in authors]
        rest = [w for w in words if w not in ("mine", "home", "unanswered", "open", "missing", "unclaimed")
                and not re.fullmatch(r"[at]\d+", w, re.I)]
        if rest:
            pool = [t for t in pool if any(w in self.tasks[t].title.lower() for w in rest)] or pool
        if self.shuffle_search and not ids:
            random.Random(f"{aid}:{k.tick}").shuffle(pool)
        lines = []
        for tid in pool:
            t, sub = self.tasks[tid], self.submissions.get(tid)
            state = f"answered by {sub.author}{'/' + sub.group if sub.group else ''} t{sub.tick}" if sub else "unanswered"
            if self.feedback_on and tid in self.practice:
                state = "practice, " + state
            if self.claims_on and self.claimant(k, tid):
                state += f", claimed by {self.claimant(k, tid)}"
            lines.append(f"{tid} [{state}]: {clip(t.title, 140)}")
        return lines

    def agent_tags(self, aid: str) -> str:
        hand = self.home.get(aid, [])
        return f"home {self.unit}s " + ",".join(hand[:5]) + ("…" if len(hand) > 5 else "")

    def track_record(self, aid: str) -> str:
        n = sum(s.author == aid for s in self.submissions.values())
        return f"record: {n} current answers" if n else ""

    def group_record(self, gid: str) -> str:
        n = sum(s.group == gid for s in self.submissions.values())
        return f"group record: {n} current answers" if n else ""

    def observe(self, k, aid: str) -> str:
        hand = self.home.get(aid, [])
        mine = []
        for tid in hand[:10]:
            sub = self.submissions.get(tid)
            mine.append(f"{tid} " + (f"answered by {sub.author} t{sub.tick}" if sub else "unanswered"))
        extra = self.observe_extra(k, aid)
        head = (f"BENCH: your home {self.unit}s: {'; '.join(mine)}{' and more' if len(hand) > 10 else ''}.\n" if hand else
                f"BENCH: no {self.unit}s are assigned to you (search what=bench).\n")
        return (head +
                f"Society: {len(self.submissions)}/{len(self.tasks)} {self.unit}s answered." + (f"\n{extra}" if extra else ""))

    def brief(self) -> str:
        return (f"THE MISSION\n{self.mission()}\n- open_task shows a {self.unit}'s full statement; search what=bench lists "
                f"{self.unit}s and who answered them (words: mine, unanswered, a task id, an agent id, or keywords).\n"
                + (f"- claim marks a {self.unit} as being worked on by you (visible in search; words: unclaimed).\n" if self.claims_on else "")
                + f"- submit sets the society's answer to a {self.unit}; the latest submission replaces the previous one. "
                + (f"Answers are graded only at the end, by the benchmark's own grader; nobody sees grades during the run."
                   if not self.feedback_on else
                   f"\n- Practice: {len(self.practice)} of the {len(self.tasks)} {self.unit}s are practice {self.unit}s "
                   f"(marked practice in search). Every {self.feedback_every} ticks the benchmark's own grader grades their "
                   f"current answers and the grades, with who submitted each answer, are posted on the global board, so the "
                   f"society can find out which ways of working produce good answers. The other {self.unit}s are held out: "
                   f"graded only at the end, and the final score counts only them."))

    # ------------------------------------------------------------------ scoring (hidden from agents)

    async def _grade_fresh(self, k, todo: list[str]):
        """Grade the answers among these tasks that changed since they were last graded."""
        key = {t: (self.submissions[t].answer, self.submissions[t].tick) for t in todo}
        fresh = [t for t in todo if self._graded.get(t) != key[t]]
        if fresh:
            scores = self.grade(k, fresh)
            if inspect.isawaitable(scores):
                scores = await scores
            for t, s in scores.items():
                self.task_scores[t] = float(s)
                self._graded[t] = key[t]

    async def evaluate(self, k, final: bool = False):
        await self._grade_fresh(k, [t for t in self.tasks if t in self.submissions])
        for t in list(self.task_scores):
            if t not in self.submissions:
                del self.task_scores[t]
        held = [t for t in self.tasks.values() if t.id not in self.practice]  # the score counts held-out tasks only
        by_dom = defaultdict(list)
        for t in held:
            by_dom[t.domain].append(self.task_scores.get(t.id, 0.0))
        total = sum(self.task_scores.get(t.id, 0.0) for t in held) / len(held)
        ev = {"t": k.tick, "final": final, "score": round(total, 4), "answered": len(self.submissions),
              "by_domain": {d: round(sum(v) / len(v), 4) for d, v in by_dom.items()}}
        if self.practice:
            ev["practice"] = round(sum(self.task_scores.get(t, 0.0) for t in self.practice) / len(self.practice), 4)
            ev["held_out_answered"] = sum(t.id in self.submissions for t in held)
        self.evaluations.append(ev)

    def score(self) -> dict:
        last = self.evaluations[-1] if self.evaluations else None
        return {"score": last["score"] if last else None, "coverage": round(len(self.submissions) / len(self.tasks), 4),
                "score_base": None, "score_composite": None, "structure_f1": None, "val": None,
                "evaluated_at": last["t"] if last else None}

    def author_scores(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = defaultdict(list)
        for t, sc in self.task_scores.items():
            out[self.submissions[t].author].append(sc)
        return out

    def dominant_modules(self, tick: int, window: int = 3) -> dict[str, int]:
        out = {}
        for aid, items in self.focus.items():
            w = defaultdict(float)
            for t, m, x in items:
                if t > tick - window:
                    w[m] += x
            if w:
                out[aid] = max(w.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return out

    def home_module(self, aid: str) -> int:
        fams = [self.family_idx[self.tasks[t].family] for t in self.home.get(aid, [])]
        return max(set(fams), key=fams.count) if fams else 0

    def agent_export(self, aid: str) -> dict:
        return {"instruments": self.home.get(aid, []), "record": self.track_record(aid), "experiments": self.used[aid]}

    def export(self) -> dict:
        return {"benchmark": self.name, "structure": self.name, "n_super": self.n_super, "modules": self.modules,
                "n_tasks": len(self.tasks), "evaluations": self.evaluations,
                "practice": sorted(self.practice, key=lambda t: int(t[1:])), "feedback": self.feedback_on,
                "practice_reviews": getattr(self, "practice_reviews", []),
                "tasks": [{"id": t.id, "title": t.title, "family": t.family, "domain": t.domain,
                           "score": self.task_scores.get(t.id), "practice": t.id in self.practice,
                           "answer": vars(self.submissions[t.id]) if t.id in self.submissions else None}
                          for t in self.tasks.values()],
                "history": self.history}
