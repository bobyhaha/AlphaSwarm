"""A science world: FrontierScience as a test of whether a society organizes itself on work that divides.

Avalon-M1's organisations gained most on knowledge work that can be divided (FrontierScience 55.9% to 63.5%). This world is the
pilot the user asked for on 2026-11-03: the society answers olympiad and research questions, with the benchmark's own judge prompts.
GOAL.md calls bags of independent tasks a weaker test of a society than one unknown system, so this is a check on the thesis in a
kinder setting, not a replacement for it.

The sheet is the assembled region: one answer per question, and the society's final answer. In GOAL.md's terms:
- Attempts are pieces. None is ever deleted, and all stay visible.
- Independent work stays independent. Until an agent has posted its own attempt on a question, others' attempts there are hidden
  from it; it can peek, and then its attempt no longer counts as independent. This is diversity from information, not assignment.
- Trust comes from evidence, never votes: said; seen (calc receipts behind it); reproduced (another agent, working blind, reached
  an equivalent answer, which the engine checks); contested (someone named a concrete error, until its author revises it).
- Two answers to one question are a conflict the substrate makes visible. The parties settle it directly; whoever comes later
  reconciles, naming what it builds on, and its attempt then replaces those on the sheet.
- The sheet takes the attempt the evidence favours: never one that is contested or superseded, then by trust, then by how many
  agents reached it working blind, then the latest.

Credit without leaks. Paying for correct answers would turn credit into an oracle for the score (Avalon's reward hacks: every
visible channel leaks). So a random sample of questions, unknown to the agents, is audited: when the sheet's answer to one changes,
the judge rules on it, the verdict is posted, and credit flows from it. Any question may be in the sample, so the incentive covers
all of them, and the final score counts only the rest.

The law, wealth goals, roles, trace, pay and the review are shared with the code world (alphateam/society.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .bench.sandbox import run_python
from .kernel import as_list, clip
from .society import GOALS, ROLE_TOOL, RULE_KEYS, TRACE_TOOL, SocietyWorld

GAME = "science"
DATA = Path(__file__).resolve().parents[1] / "data" / "frontierscience"
PASS_POINTS = 7.0  # FrontierScience counts a research answer as correct at seven of ten rubric points

LAW = {"goal": "individual", "value": 1.0, "piece_share": 0.25, "tester_share": 0.25, "said": 0.2, "seen": 0.6, "reproduced": 1.0,
       "steps": 12, "fine_ratio": 2.0}
LAW_BOUNDS = {"value": (0, 4), "piece_share": (0, .6), "tester_share": (0, .6), "said": (0, 1), "seen": (0, 1), "reproduced": (0, 1),
              "steps": (4, 24), "fine_ratio": (0, 4)}
SHOW = ("plans", "findings", "dead_ends", "guide", "interfaces", "paid", "unused", "cycle", "kinds")
RANK = {"reproduced": 3, "seen": 2, "said": 1, "contested": 0}

# The benchmark's own judge prompts (FrontierScience paper, Appendix B): a grader must be the benchmark's, word for word.
OLYMPIAD_JUDGE = """You are grading an attempted answer to a science olympiad problem. You will be given the problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.
The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method.
Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.
***
The problem: {problem}
***
The reference answer: {reference}
***
The attempted answer: {answer}
***
First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT"."""
RESEARCH_JUDGE = """You are grading a science exam.
You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.
Evaluate the attemped answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).
***
The problem: {problem}
***
The rubric: {reference}
***
The attempted answer: {answer}
***
First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8."""
SAME_SYSTEM = "You compare two final answers to one science problem. You do not judge whether either is right."
SAME_PROMPT = """Are these two final answers equivalent: the same expression, the same number within rounding, the same name or formula, accounting for units?

The problem: {problem}

Answer one: {one}

Answer two: {two}

Reply with one word on the last line: SAME or DIFFERENT."""

OPEN_TOOL = {
    "name": "open_task",
    "description": "Read a question. Others' attempts on it stay hidden until you have posted your own. peek=true shows them now, but "
                   "then your own attempt there no longer counts as independent.",
    "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "e.g. T12"},
                                                    "peek": {"type": "boolean"}}, "required": ["task"]}}
ANSWER_TOOL = {
    "name": "answer",
    "description": "Post your attempt at a question. It is never deleted. working: your full working, step by step, with why it holds and "
                   "what you ruled out, for others to build on. final: your answer as the question asks for it (for a research question, "
                   "the whole answer). "
                   "evidence: R-ids of your calc runs behind it. builds_on: E-ids of attempts this one takes in, once you have seen them; "
                   "it replaces them on the sheet, so say what you keep and what you correct. contests: the E-id of someone's attempt "
                   "with a concrete error, named in your working; it leaves the sheet until its author revises it. A new attempt of yours on "
                   "the same question revises your last one.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string"}, "working": {"type": "string"}, "final": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}}, "builds_on": {"type": "array", "items": {"type": "string"}},
        "contests": {"type": "string"}}, "required": ["task", "working", "final"]}}
CALC_TOOL = {
    "name": "calc",
    "description": "Run Python (numpy, sympy, scipy if installed) in your own folder, with no network. Print what you want to see. Uses "
                   "one step; returns a receipt (R-id) to cite as evidence.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}
HEAD_TOOL = {
    "name": "head",
    "description": "See the state of the society in full. what: sheet (every question and what stands on it), audits (every verdict so "
                   "far), credit, roles, law (the rules in force and their history).",
    "parameters": {"type": "object", "properties": {
        "what": {"type": "string", "enum": ["sheet", "audits", "credit", "roles", "law"]}}, "required": ["what"]}}

WORDS = {
    "plan": "Plans and findings on the shared substrate. claim: announce what you will work on: a hypothesis and the result that would "
            "show it (criteria); everyone sees it while it is open, and every claim stays listed under IDEAS with what came of it. "
            "finding: post a finding that others can use, such as a method or a result several questions rest on, with its receipts. "
            "done: close your plan with what changed, the outcome and the receipts (R-ids from calc). drop: abandon it. challenge: "
            "contest someone's finding with a counterexample. test: test a finding before you trust it (entry=<E-id or R-id>): run what "
            "it says, then give result=holds or fails, evidence (your receipts) and finding (what you saw). A claim or finding can name "
            "the pieces it builds on (builds_on: E-ids).",
    "guide": "The society's guide: one durable, revisable text of what it has understood (methods, formulas, how to read these questions, "
             "what was ruled out), shown every turn to everyone, its lines numbered. Never write a question's answer here. view: the full "
             "guide, its versions and who wrote each line. add: append one line, with receipts if any (evidence). merge: combine lines "
             "(lines: their numbers) into one shorter line (line). challenge: flag someone's line with a counterexample (line, evidence, "
             "why). edit: replace it with your revision, written on the current version (base).",
    "lead": "Found and run a team. found: found a team (if you are in a team, it is a sub-team). "
            "join/leave: join an open team or leave yours. assign: give a direct child (a member, or the lead of a sub-team) a role and a "
            "job, shown at the top of its every turn. report: tell your lead what you did and found. tracker: name who keeps your team's "
            "pinned note. pin: (tracker or lead) set the note every team member sees each turn. step_down: hand the team to a member (to) "
            "or dissolve it.",
    "substrate": "Shape the shared substrate. define_kind: add a kind of entry anyone can then write (write kind=<name>), with its purpose "
                 "and, optionally, its fields: an interface every entry of that kind must fill (lines 'field: value'). adopt: declare you "
                 "will use a kind; once enough agents adopt it, it is agreed and shown every turn. revise_kind: change an adopted kind's "
                 "fields or purpose. propose: suggest a change to anything about how this society is run (the credit law, the practices, "
                 "what each turn shows); proposals are read at every review, and only accuracy judges a change.",
}
STRUCTURES = """- Each agent or team takes a subject or a kind of question.
- Several agents solve a question independently, and another reconciles what they found.
- A solver and a critic on each question: the critic looks for the concrete error.
- A lead who works like a CEO or PI: directs a team, delegates jobs and roles to its children, keeps working on details itself; a child who leads a sub-team does the same, so teams nest.
- One agent keeps the guide of methods, and others cite it.
- A market: jobs with prices (delegate with pay), paid when the work holds up."""
CYCLE = ("A working cycle that may pay (yours to use, change or ignore): see what the sheet, the guide and the board already hold, and which "
         "questions are open, unchecked or disputed; say what you will do; work it out; check it with calc where you can; post it with how "
         "you got it and what you ruled out, so others can build on it; when answers differ, find the error.")

FAKE = [{"problem": "What is 2+2?", "answer": "4", "subject": "physics", "track": "olympiad"},
        {"problem": "Name the gas plants release.", "answer": "oxygen", "subject": "biology", "track": "olympiad"},
        {"problem": "Formula of water?", "answer": "H2O", "subject": "chemistry", "track": "olympiad"},
        {"problem": "What is 3*3?", "answer": "9", "subject": "physics", "track": "olympiad"},
        {"problem": "Explain why the sky is blue.", "answer": "Points: 10, Item: mentions scattering", "subject": "physics", "track": "research"},
        {"problem": "Explain osmosis.", "answer": "Points: 10, Item: mentions membrane", "subject": "biology", "track": "research"}]


def headroom(q: dict, screen: dict, most: int = 1) -> bool:
    """Whether one agent leaves room on this question: tools/screen_science.py solved it at most `most` times in its one-shot trials.
    A society can only show a gain where a single agent does not already saturate the task. Research questions it never passed stay."""
    row = screen.get(hashlib.sha1(q["problem"].encode()).hexdigest()[:12])
    if row is None:
        return True
    if q["track"] == "research":
        return not any(m >= PASS_POINTS / 10 for m in row["marks"])
    return sum(row["marks"]) <= most


def load_questions(olympiad: int, research: int, set_seed: int = 0, fake: bool = False, hard: bool = False) -> list[dict]:
    """The same questions for every arm, stratified by subject, in an order that says nothing about them."""
    if fake:
        rows = [dict(q) for q in FAKE]
    else:
        rows = []
        screen = json.loads((DATA / "screen.json").read_text()) if hard and (DATA / "screen.json").exists() else {}
        for track, n in (("olympiad", olympiad), ("research", research)):
            pool = [{**json.loads(line), "track": track} for line in (DATA / f"{track}.jsonl").read_text().splitlines() if line.strip()]
            pool = [q for q in pool if headroom(q, screen)]
            rng = random.Random(f"fs:{track}:{set_seed}")
            by = defaultdict(list)
            for q in pool:
                by[q["subject"]].append(q)
            for group in by.values():
                rng.shuffle(group)
            picked, names = [], sorted(by)
            while len(picked) < min(n, len(pool)):
                for name in names:
                    if by[name] and len(picked) < min(n, len(pool)):
                        picked.append(by[name].pop())
            rows += picked
    random.Random(f"fs:order:{set_seed}").shuffle(rows)
    return [{**q, "id": f"T{i + 1}"} for i, q in enumerate(rows)]


def plain_answer(text: str) -> str:
    return re.sub(r"[\s$`\\{}()\[\]]", "", str(text)).lower().rstrip(".")


def verdict(text: str, research: bool) -> float | None:
    """The last line's verdict, as a share of the marks. None when the judge gave no verdict: no signal, never a zero."""
    found = re.findall(r"VERDICT:\s*\**\s*([A-Za-z]+|[\d.]+)", text or "")
    if not found:
        return None
    last = found[-1].strip().rstrip(".")
    if research:
        try:
            return min(10.0, max(0.0, float(last))) / 10
        except ValueError:
            return None
    return {"CORRECT": 1.0, "INCORRECT": 0.0}.get(last.upper())


class ScienceWorld(SocietyWorld):
    GAME = GAME
    LAW, LAW_BOUNDS, SHOW, WORDS = LAW, LAW_BOUNDS, SHOW, WORDS
    LAW_CHOICES = {"goal": GOALS}
    LAW_WHOLE = ("steps",)
    SHOW_DEFAULT = ("plans", "findings", "dead_ends", "guide", "interfaces", "paid", "unused", "cycle")
    tool_names = ("open_task", "answer", "calc", "head", "trace", "role")
    TOOLS = [OPEN_TOOL, ANSWER_TOOL, CALC_TOOL, HEAD_TOOL, TRACE_TOOL, ROLE_TOOL]
    final_tools = ("answer", "calc", "write", "plan", "guide", "message", "delegate", "pay", "role")
    move_tools = ("answer", "calc")
    search_kinds = ("questions",)
    state_noun = "calculation"
    memory_chars = 4000

    def __init__(self, n_agents: int, seed: int = 0, single: bool = False, world_agents: int | None = None,
                 opts: dict | None = None):
        opts = dict(opts or {})
        on = lambda key, default="1": str(opts.get(key, default)).lower() in ("1", "true", "yes")
        self.fake = on("fake", "0")
        set_seed = int(opts.get("set_seed", 0))
        self.questions = {q["id"]: q for q in load_questions(int(opts.get("olympiad", 40)), int(opts.get("research", 20)), set_seed, self.fake,
                                                             hard=on("hard", "0"))}
        ids = sorted(self.questions, key=lambda t: int(t[1:]))
        rng = random.Random(f"fs:audit:{set_seed}")
        share = float(opts.get("audit", 0.25))
        audited = []
        for track in ("olympiad", "research"):
            pool = [t for t in ids if self.questions[t]["track"] == track]
            audited += rng.sample(pool, max(1, round(share * len(pool)))) if pool else []
        self.audit_set = frozenset(audited)
        self.audit_model = opts.get("audit_model") or None
        self.audit_effort = str(opts.get("audit_effort", "low"))
        self.final_model = opts.get("final_model") or None
        self.final_effort = str(opts.get("final_effort", "high"))
        self.judge_all = on("judge_all")
        self.heritage = str(opts.get("heritage", "org"))
        self._init_society(n_agents, single, world_agents, opts)
        self.add_tools([])
        self.attempts: dict[str, list[dict]] = defaultdict(list)
        self.by_entry: dict[str, dict] = {}
        self.peeked: dict[str, set] = defaultdict(set)
        self.opened: dict[str, dict] = defaultdict(dict)  # question -> agent -> tick last opened
        self.compared = 0
        self.fresh: list[dict] = []  # attempts not compared yet
        self.audits: list[dict] = []
        self.best_audit: Counter = Counter()
        self.audited_as: dict[str, str] = {}  # question -> the attempt last judged
        self.final: dict = {}
        self.judge_gaps = 0
        self.llm = None
        self.folder = Path(tempfile.mkdtemp(prefix="alphateam-science-"))

    def law_words(self) -> str:
        law, c = self.law, self.rules["credit"]
        who = {"individual": "the agent it is paid to", "team": "all members of the payee's group, equally (an agent in no group keeps it)",
               "society": "all agents, equally"}[law["goal"]]
        return (f"an audited answer is worth {law['value']:g} for what it newly gains (a research answer by its share of the rubric). Out of that: "
                f"{law['piece_share']:g} to the pieces it used (named in builds_on, or read in the last three ticks), each weighted by trust "
                f"(reproduced {law['reproduced']:g}, seen {law['seen']:g}, said {law['said']:g}, contested 0); {law['tester_share']:g} to the "
                f"agents who reached it independently; the rest to the agent who posted it" + (f", less {c['lead_share']:g} to its lead" if c.get("lead_share") else "")
                + f". Credit goes to {who}. {law['steps']} steps a tick" + (f", plus {c['steps_per_credit']:g} per credit earned in the last two ticks, at most double" if c.get("steps_per_credit") else "")
                + f". A fine takes {law['fine_ratio']:g} per credit spent.")

    # ------------------------------------------------------------------ lifecycle

    def _start(self, k):
        if self.workdir:
            self.folder = Path(self.workdir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.held_log.append({"t": 0, "built": True, "passed": 0, "total": len(self.audit_set)})
        self.started = True

    def on_tick_start(self, k):
        if not self.started:
            self._start(k)

    async def on_tick_end(self, k):
        await self._compare(k)
        await self._audit(k)
        self.held_log.append({"t": k.tick, "built": True, "passed": round(sum(self.best_audit.values()), 2), "total": len(self.audit_set)})

    def progress(self) -> dict:
        return self.held_log[-1] if self.held_log else {"t": 0, "built": False, "passed": 0, "total": len(self.audit_set)}

    # ------------------------------------------------------------------ actions

    def _question(self, k, ref) -> dict:
        tid = str(ref or "").strip().upper()
        tid = f"T{tid}" if tid.isdigit() else tid
        k._need(tid in self.questions, f"no question {ref} (they are T1..T{len(self.questions)})")
        return self.questions[tid]

    def _has_seen(self, aid: str, tid: str) -> bool:
        return aid in self.peeked[tid] or any(a["agent"] == aid for a in self.attempts[tid])

    def _attempt_lines(self, k, tid: str) -> str:
        rows = []
        for a in self.attempts[tid]:
            state = self.level(a) + (f", superseded by {a['superseded_by']}" if a.get("superseded_by") else "")
            rows.append(f"{a['id']} {a['agent']} t{a['t']} [{state}]{' independent' if a['blind'] else ''}\n  final: {clip(a['final'], 1800)}\n"
                        f"  working: {clip(a['reasoning'], 1800)}")
        return "\n".join(rows)

    def _open_task(self, k, aid, a):
        q = self._question(k, a.get("task"))
        tid = q["id"]
        self.opened[tid][aid] = k.tick
        if a.get("peek") and self.attempts[tid] and aid not in self.peeked[tid]:
            self.peeked[tid].add(aid)
        text = f"{tid} ({q['subject']}, {q['track']}): {q['problem']}"
        n = len(self.attempts[tid])
        if not n:
            return "lab.open_task", text + "\nNo attempts yet.", tid
        if self._has_seen(aid, tid):
            return "lab.open_task", text + f"\n{n} attempts:\n{self._attempt_lines(k, tid)}", tid
        return "lab.open_task", text + (f"\n{n} attempts by {', '.join(sorted({x['agent'] for x in self.attempts[tid]}))} are hidden until you post "
                                        "your own (or open with peek=true, and then yours no longer counts as independent)."), tid

    def _answer(self, k, aid, a):
        q = self._question(k, a.get("task"))
        tid = q["id"]
        final = str(a.get("final") or "").strip()
        k._need(final, "final: your answer")
        reasoning = str(a.get("working") or a.get("reasoning") or "").strip()
        ev = [str(r).strip().upper() for r in as_list(a.get("evidence")) if str(r).strip()]
        bad = [r for r in ev if r not in self.receipts or self.receipts[r]["agent"] != aid]
        k._need(not bad, f"evidence: receipts of your own calc runs ({', '.join(bad)} are not)")
        builds = [str(x).strip().upper() for x in as_list(a.get("builds_on")) if str(x).strip()]
        if builds:
            k._need(self._has_seen(aid, tid), "builds_on: post your own attempt first, or open the question with peek=true, so you have seen them")
            k._need(all(x in self.by_entry and self.by_entry[x]["task"] == tid for x in builds), f"builds_on: attempts on {tid} (E-ids)")
        target = str(a.get("contests") or "").strip().upper()
        if target:
            k._need(target in self.by_entry and self.by_entry[target]["task"] == tid, f"contests: an attempt on {tid} (an E-id)")
            k._need(self.by_entry[target]["agent"] != aid, "that attempt is yours: revise it by posting again")
            k._need(self._has_seen(aid, tid), "contests: you have not seen it; post your own attempt or peek first")
            k._need(reasoning, "working: the concrete error you found")
        mine = [x for x in self.attempts[tid] if x["agent"] == aid and not x.get("superseded_by")]
        blind = not mine and aid not in self.peeked[tid] and not builds
        e = k.post("global", aid, "result", f"ATTEMPT on {tid} ({q['subject']}, {q['track']}) by {aid}" + (" independent" if blind else "")
                   + (f", building on {', '.join(builds)}" if builds else "") + (f", contesting {target}" if target else ""),
                   refs=ev + builds + ([target] if target else []), notify=False)
        row = {"id": e.id, "task": tid, "agent": aid, "t": k.tick, "final": final[:9000], "reasoning": reasoning[:6000], "evidence": ev,
               "builds_on": builds, "blind": blind, "cls": None}
        self.entry_meta[e.id] = {"game": GAME, "level": 1, "type": "attempt"}
        for x in mine:
            x["superseded_by"] = e.id
            x["contested"] = False
        for ref in builds:
            self.by_entry[ref]["superseded_by"] = e.id
            self.parts.setdefault(e.id, []).append(ref)
        if target:
            self.by_entry[target]["contested"] = True
            who = self.by_entry[target]["agent"]
            k.notify(who, aid, f"contests your answer to {tid} ({target}): {clip(reasoning, 300)}. Revise it by posting again on {tid}, or message {aid}")
        self.attempts[tid].append(row)
        self.by_entry[e.id] = row
        self.fresh.append(row)
        if q["track"] == "olympiad" and len(self.attempts[tid]) == 1:
            row["cls"] = 0
        hidden = sum(x["agent"] != aid for x in self.attempts[tid])
        return "lab.answer", (f"posted {e.id} on {tid}" + (" as an independent attempt" if blind else "")
                              + (f". {hidden} other attempts there are open to you now: open_task {tid}" if hidden else "")), e.id

    async def _calc(self, k, aid, a):
        code = str(a.get("code") or "")
        k._need(code.strip(), "code: Python to run")
        self._step(k, aid)
        ok, out = await run_python(code, self.folder / "calc" / aid, timeout=30, limit=4000, python=sys.executable)
        rid = self._log_work(k, aid, "calc", code, out)
        return "lab.calc", f"{rid}: {out}", rid

    def _head(self, k, aid, a):
        what = str(a.get("what") or "sheet")
        if what == "sheet":
            return "lab.head", self.sheet_view(k, aid), None
        rows = {"audits": [{key: x[key] for key in ("t", "task", "agent", "attempt", "score")} for x in self.audits],
                "credit": dict(self.credit.most_common()), "roles": self.titles, "law": {"now": self.law_words(), "history": self.law_log}}
        k._need(what in rows, "what: sheet, audits, credit, roles or law")
        return "lab.head", clip(json.dumps(rows[what], default=str), 12000), None

    # ------------------------------------------------------------------ trust and the sheet

    def independent_agreement(self, a: dict) -> set:
        """The other agents who reached this attempt's answer working blind."""
        if a.get("cls") is None:
            return set()
        return {x["agent"] for x in self.attempts[a["task"]] if x.get("cls") == a["cls"] and x["blind"] and x["agent"] != a["agent"]}

    def level(self, a: dict) -> str:
        if a.get("contested"):
            return "contested"
        if a.get("cls") is not None and len({x["agent"] for x in self.attempts[a["task"]] if x.get("cls") == a["cls"] and x["blind"]}) >= 2:
            return "reproduced"  # two agents reached it blind; a copy of one blind answer reproduces nothing
        return "seen" if a["evidence"] else "said"

    def sheet_attempt(self, tid: str) -> dict | None:
        """The attempt the evidence favours. Votes alone cannot set it: agreement counts only from agents who worked blind."""
        rows = [a for a in self.attempts[tid] if not a.get("superseded_by")]
        rows = [a for a in rows if not a.get("contested")] or rows
        if not rows:
            return None
        return max(rows, key=lambda a: (RANK[self.level(a)], len(self.independent_agreement(a)), a["t"], a["id"]))

    def disputed(self, tid: str) -> list[list[str]]:
        groups = defaultdict(set)
        for a in self.attempts[tid]:
            if a.get("cls") is not None and not a.get("superseded_by"):
                groups[a["cls"]].add(a["agent"])
        return [sorted(v) for v in groups.values()] if len(groups) > 1 else []

    async def _compare(self, k):
        """Olympiad attempts: which give the same final answer. The engine reports this; it does not say who is right."""
        fresh, self.fresh = self.fresh, []

        async def place(a: dict):
            q = self.questions[a["task"]]
            if q["track"] != "olympiad" or a.get("cls") is not None:
                return
            classes: dict[int, dict] = {}
            for x in self.attempts[a["task"]]:
                if x.get("cls") is not None and x is not a:
                    classes.setdefault(x["cls"], x)
            for cls, rep in sorted(classes.items()):
                if await self._same(k, q, a["final"], rep["final"]):
                    a["cls"] = cls
                    return
            a["cls"] = max(classes, default=-1) + 1

        for a in fresh:  # in order, so two new attempts on one question can join one class
            await place(a)
        for tid in sorted({a["task"] for a in fresh}):
            sides = self.disputed(tid)
            if len(sides) > 1:
                parties = [a for side in sides for a in side]
                for who in parties:
                    k.notify(who, "lab", f"answers to {tid} differ: {' vs '.join(', '.join(s) for s in sides)}. open_task {tid} shows them; "
                                         "settle it directly, with the error named")

    async def _same(self, k, q: dict, one: str, two: str) -> bool:
        if plain_answer(one) == plain_answer(two):
            return True
        if self.llm is None:
            return False
        self.compared += 1
        try:
            text = await self.llm(SAME_SYSTEM, SAME_PROMPT.format(problem=q["problem"], one=clip(one, 1500), two=clip(two, 1500)),
                                  max_tokens=3000, who="grader", model=self.audit_model, effort=self.audit_effort)
        except Exception:
            return False
        return "SAME" in text.upper().split("\n")[-1] or (text.upper().rstrip().endswith("SAME"))

    async def judge(self, q: dict, answer: str, model: str | None, effort: str) -> float | None:
        """The benchmark's judge on one answer. No verdict is no signal, never a zero."""
        research = q["track"] == "research"
        if self.llm is None:
            if research:
                return 0.8 if len(answer) > 3 else 0.2
            return 1.0 if plain_answer(answer) == plain_answer(q["answer"]) else 0.0
        prompt = (RESEARCH_JUDGE if research else OLYMPIAD_JUDGE).format(problem=q["problem"], reference=q["answer"], answer=clip(answer, 9000))
        for _ in range(2):
            try:
                text = await self.llm("", prompt, max_tokens=16000, who="grader", model=model, effort=effort)
            except Exception:
                continue
            found = verdict(text, research)
            if found is not None:
                return found
        self.judge_gaps += 1
        return None

    async def _audit(self, k):
        todo = []
        for tid in sorted(self.audit_set, key=lambda t: int(t[1:])):
            a = self.sheet_attempt(tid)
            if a and self.audited_as.get(tid) != a["id"]:
                todo.append((tid, a))
        scores = await asyncio.gather(*(self.judge(self.questions[t], a["final"], self.audit_model, self.audit_effort) for t, a in todo))
        for (tid, a), score in zip(todo, scores):
            if score is None:
                continue
            self.audited_as[tid] = a["id"]
            gain = max(0.0, score - self.best_audit[tid])
            self.best_audit[tid] = max(self.best_audit[tid], score)
            research = self.questions[tid]["track"] == "research"
            self.audits.append({"t": k.tick, "task": tid, "agent": a["agent"], "attempt": a["id"], "score": score, "gain": gain})
            words = f"{score * 10:g} of 10 rubric points" if research else ("correct" if score else "incorrect")
            k.post("global", "lab", "result", f"AUDIT of {tid}: the sheet's answer ({a['id']} by {a['agent']}, {self.level(a)}) is {words}.", notify=False)
            if gain > 0 and self.incentive:
                self._pay_answer(k, a, self.law["value"] * gain)

    def _pay_answer(self, k, a: dict, value: float):
        """An audited answer's value, paid to what made it possible, out of it: its author keeps at least a tenth."""
        aid, law, rows = a["agent"], self.law, []
        pieces = {x: 1.0 for x in a["builds_on"]}
        for eid, t in k.retrieved[aid].items():
            if k.tick - t <= 2:
                pieces.setdefault(eid, 0.5)
        for eid in list(pieces):
            for part in self.parts.get(eid, []):
                pieces.setdefault(part, pieces[eid] / 2)
        used = {}
        for eid, weight in pieces.items():
            e, row = k.entry(eid), self.by_entry.get(eid)
            who = row["agent"] if row else e.author if e else None
            if who in k.agents and who != aid and (e is None or e.kind != "data"):
                used[eid] = (who, weight, self.level(row) if row else self.trust_of(k, eid)["level"])
        total = max(1.0, sum(w for _, w, _ in used.values()))
        for eid, (who, weight, level) in used.items():
            rows.append((who, law["piece_share"] * value * weight * self.trust_weight(level) / total, f"{aid}'s audited answer used your {eid}"))
        testers = sorted(self.independent_agreement(a))
        for who in testers:
            rows.append((who, law["tester_share"] * value / len(testers), f"you reached {aid}'s audited answer independently"))
        rows = [r for r in rows if r[1] > 0]
        paid = sum(r[1] for r in rows)
        scale = min(1.0, 0.9 * value / paid) if paid else 1.0
        self._pay_out(k, [(who, amount * scale, why) for who, amount, why in rows] + [(aid, value - paid * scale, f"audited answer {a['id']}")])

    # ------------------------------------------------------------------ the review

    def window_facts(self, k, start: int) -> dict:
        """What happened since tick start, for reflection: facts, no advice. Audit verdicts only: nothing about the scored questions' truth."""
        fresh = [a for rows in self.attempts.values() for a in rows if a["t"] > start]
        agents = {x: {"attempts": sum(a["agent"] == x for a in fresh), "calc": sum(n for (y, t), n in self.steps.items() if y == x and t > start),
                      "audited_correct": round(sum(u["gain"] for u in self.audits if u["agent"] == x and u["t"] > start), 2),
                      "credit": round(self.credit[x], 2), "attributed": round(self.attributed[x], 2), "role": self.titles.get(x, {}).get("title", "")}
                  for x in k.active_agents()}
        ids = list(self.questions)
        now = self.progress()
        then = next((h for h in reversed(self.held_log) if h["t"] <= start), self.held_log[0] if self.held_log else now)
        verdicts = [u for u in self.audits if u["t"] > start]
        levels = Counter(self.level(a) for t in ids if (a := self.sheet_attempt(t)))
        return {"tick": k.tick, "of": k.total_ticks, "held_out": {"now": now.get("passed"), "total": now.get("total"), "built": True,
                                                                  "gain_this_window": round((now.get("passed") or 0) - (then.get("passed") or 0), 2)},
                "agents": agents, "questions": len(ids), "unanswered": sum(not self.attempts[t] for t in ids),
                "one_attempt_only": sum(len(self.attempts[t]) == 1 for t in ids), "disputed": sum(bool(self.disputed(t)) for t in ids),
                "sheet_by_trust": dict(levels), "attempts_this_window": len(fresh), "independent_attempts": sum(a["blind"] for a in fresh),
                "audits_this_window": [{key: u[key] for key in ("task", "agent", "score")} for u in verdicts],
                "audited_correct_by_trust": self._audit_by_trust(), **self.group_facts(k, lambda x: agents[x]["audited_correct"] if x in agents else 0),
                **self.action_facts(k, start)}

    def _audit_by_trust(self) -> dict:
        out = defaultdict(list)
        for u in self.audits:
            out[self.level(self.by_entry[u["attempt"]])].append(u["score"])
        return {key: {"n": len(v), "mean": round(sum(v) / len(v), 2)} for key, v in out.items()}

    def culture(self) -> dict:
        return {"law": self.law, "rules": {x: self.rules["credit"][x] for x in RULE_KEYS}, "show": self.rules["substrate"]["show"],
                "norms": self.norms, "titles": sorted({v["title"] for v in self.titles.values()}), "law_log": self.law_log,
                "guide": self.guides[GAME][-1]["text"]}

    def inherit_culture(self, k, c: dict):
        goal = self.law["goal"]
        self.law = self.clean_law(c.get("law"), self.law)
        if self.goal_locked:
            self.law["goal"] = goal
        self.steps_per_tick = self.law["steps"] * self.scale
        self.rules["credit"].update({x: v for x, v in (c.get("rules") or {}).items() if x in RULE_KEYS})
        if c.get("show"):
            self.rules["substrate"]["show"] = list(c["show"])
        self.norms = str(c.get("norms") or "")
        text = str(c.get("guide") or "").strip()
        if self.heritage == "all" and text:  # the same questions would make this an answer key, so it is off unless asked for
            lines = {ln.strip(): "inherited" for ln in text.splitlines() if ln.strip()}
            self.guides[GAME] = [{"v": 0, "author": "inherited", "t": 0, "text": text, "why": "inherited from an earlier society", "lines": lines}]

    # ------------------------------------------------------------------ what agents see

    def search(self, k, aid, what, words):
        return [self._row(k, t) for t in self.questions if not words or any(w in self.questions[t]["problem"].lower() or w == t.lower() for w in words)]

    def track_record(self, aid: str) -> str:
        n = sum(a["agent"] == aid for rows in self.attempts.values() for a in rows)
        return f"record: {n} attempts" if n else ""

    def group_record(self, gid: str) -> str:
        return ""

    def deliver_group(self, k, aid: str, e) -> tuple:
        if e.author == "lab":
            return 0, "Audits and the law", 200
        kind = (self.entry_meta.get(e.id) or {}).get("type")
        if kind == "attempt":
            return 4, "Attempts", 120
        if kind in ("dead_end", "challenge"):
            return 1, "Dead ends and challenges", 180
        if kind in ("finding", "test", "guide"):
            return 2, "Findings, tests and guide lines", 200
        if kind == "claim":
            return 3, "Ideas claimed", 180
        return 3, "Posts and notes", 220

    def _row(self, k, tid: str) -> str:
        q, rows = self.questions[tid], self.attempts[tid]
        a = self.sheet_attempt(tid)
        state = "no attempt" if not rows else f"{len(rows)} attempts, sheet {a['agent']} {self.level(a)}"
        sides = self.disputed(tid)
        return f"{tid} {q['subject']} {q['track']}: {state}" + (f"; disputed {' vs '.join(','.join(s) for s in sides)}" if sides else "")

    def sheet_view(self, k, aid: str) -> str:
        return "\n".join(self._row(k, t) for t in self.questions)

    def observe(self, k, aid: str) -> str:
        ids = sorted(self.questions, key=lambda t: random.Random(f"{aid}:{t}").random())  # each agent its own order, so nobody is steered
        L = []
        if self.norms:
            L.append(f"PRACTICES the society adopted at its review at t{self.norms_t} (yours to use, change or ignore; substrate op=propose to "
                     f"change them): {self.norms}")
        untouched = [t for t in ids if not self.attempts[t]]
        single = [t for t in ids if len(self.attempts[t]) == 1 and self.questions[t]["track"] == "olympiad"]
        disputed = [t for t in ids if self.disputed(t)]
        agreed = [t for t in ids if (a := self.sheet_attempt(t)) and self.level(a) == "reproduced"]
        long = [t for t in ids if self.attempts[t] and self.questions[t]["track"] == "research"]
        L.append(f"THE SHEET: {len(ids)} questions; {len(ids) - len(untouched)} with an answer; {len(agreed)} reproduced; {len(disputed)} disputed.")
        if untouched:
            L.append("No attempt yet: " + ", ".join(f"{t} {self.questions[t]['subject'][:4]} {self.questions[t]['track'][:3]}" for t in untouched))
        if single:
            L.append("One attempt, nobody has checked it: " + ", ".join(f"{t} ({self.attempts[t][0]['agent']})" for t in single))
        if disputed:
            L.append("Disputed, for the parties to settle directly: " + "; ".join(f"{t} {' vs '.join(','.join(s) for s in self.disputed(t))}" for t in disputed))
        if agreed:
            L.append("Reproduced: " + ", ".join(f"{t} ({','.join(sorted({self.sheet_attempt(t)['agent']} | self.independent_agreement(self.sheet_attempt(t))))})" for t in agreed))
        if long:
            L.append("Research answers so far: " + ", ".join(f"{t} {len(self.attempts[t])}" for t in long))
        mine = sorted({a["task"] for rows in self.attempts.values() for a in rows if a["agent"] == aid}, key=lambda t: int(t[1:]))
        if mine:
            L.append("You have attempted: " + ", ".join(mine))
        now = {t: [x for x, seen in self.opened[t].items() if seen == k.tick and x != aid] for t in ids}
        now = {t: who for t, who in now.items() if who}
        if now:
            L.append("Open right now elsewhere: " + "; ".join(f"{t} {','.join(w)}" for t, w in now.items()))
        if self.feedback[aid]:
            L.append("Notes: " + " | ".join(self.feedback[aid]))
            self.feedback[aid] = []
        if self.titles:
            L.append("Roles: " + "; ".join(f"{a} {t['title']}" for a, t in sorted(self.titles.items(), key=lambda x: int(x[0][1:]))))
        peers = self._peers(k, aid)
        if peers:
            L.append(f"You share a group with {', '.join(peers)}: tell them what you are finding, by message or on your group's board.")
        L.append(f"Steps left this tick: {self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]}.")
        L += self.wealth_lines(k, aid)
        return "\n".join(L) + "\n" + self._plain(self.explore_observe(k, aid))

    def brief(self) -> str:
        n = len(self.questions)
        credit = ""
        if self.incentive:
            credit = (f"\n\nCREDIT\n{self.goal_text()}\n"
                      "- Credit is paid only for verified progress, never for activity. A random sample of the questions, unknown to you, is "
                      "audited: when the sheet's answer to one changes, the benchmark's grader rules on it, the verdict is posted, and credit "
                      "flows from it. Any question may be in the sample. Only a new best on a question pays. Audited questions do not count in "
                      "the final score; the others are graded at the end.\n"
                      f"- The law in force (v{self.law_v}): {self.law_words()}\n"
                      "- Each turn shows which actions and combinations have paid so far, and which actions nobody has used yet.\n"
                      "- Wealth has consequences: it buys steps, and it pays others.")
        trust = ("\n\nTRUST\nEvery attempt shows what it can be trusted at, worked out from evidence, never votes: said (no engine record), seen "
                 "(calc receipts behind it), reproduced (another agent, working independently, reached an equivalent answer; the engine checks "
                 "this), CONTESTED (someone named a concrete error, until its author revises it). The credit a piece earns from what is built "
                 "on it depends on this level.")
        cycle = f"\n\n{CYCLE}" if "cycle" in self.rules["substrate"]["show"] else ""
        offered = ["role: a title of your own, seen by everyone. trace: read what any agent has been doing.",
                   "groups: a group's memory board is its own, and your private board is yours."]
        offered.append("delegate: hand a job or a decision to an agent or a group, with what it covers and when it ends; it can be refused, handed on "
                       "in part, or revoked." + (" pay is held until the job is accepted." if self.market_on else ""))
        if self.market_on:
            offered.append("pay: give credit, or fine someone at a cost to yourself.")
        offered.append("plan, guide, lead, substrate, message, link, reflect, protocol: described in their tools.")
        ways = (f"\n\nSOME WAYS TO ORGANIZE (any of these, none of them, or your own; nothing here is a recommendation, and nobody assigns any "
                f"of it)\n{STRUCTURES}") if self.ways_on else ""
        return f"""THE MISSION
The society answers {n} hard science questions in physics, chemistry and biology. Olympiad questions have one final answer. Research questions are answered in full and graded against an expert rubric nobody sees, with seven of ten points to pass. The society's sheet holds one answer per question, and it is the society's final answer. Accuracy on it is the only thing that finally counts.

THE PICTURE
A hard problem is a large jigsaw puzzle. Many agents can work on local pieces at once; combining them into one coherent result is just as important, and it is where societies fail. Here the pieces are attempts, calculations, methods and dead ends, and the sheet is the assembled region. A placed piece is never swept off the table, and two pieces for one spot are settled by which fits the evidence. Progress is what stands on the sheet and holds up, not how many attempts lie on the table. The sheet, the guide and the board hold what the society has already established.

HOW THE WORK GOES
- open_task shows a question, and calc runs Python ({self.steps_per_tick} calc steps per tick, more with recent credit).
- answer posts your attempt, with your full working: a result should carry its understanding, why it works and what was ruled out. Nothing is lost: every attempt stays, visible, whatever stands on the sheet.
- Independent work stays independent. Others' attempts on a question are hidden from you until you post your own; peeking shows them at once, and then yours no longer counts as independent.
- The engine compares olympiad answers. Conflicts are visible and settled by the parties: each turn lists disputed questions, and a message reaches another agent during its turn. Evidence decides what is correct, never rank or votes: a concrete error outweighs any number of agreeing attempts. Whoever contributes later reconciles: take in what is there, re-check, then post an attempt that builds on it, rather than pass over it. An attempt that builds on others replaces them on the sheet.
- The sheet takes the attempt the evidence favours: never one that is contested or replaced, then the most trusted, then the one most agents reached independently, then the latest.
- head shows the whole sheet, every audit, credit, roles and the law.{credit}{trust}{cycle}

ACTIONS ARE YOURS TO EXPLORE AND EXTEND
Which actions and combinations of actions are useful is for this society to find out; nothing here is required. You can add to them: design_action adds an action to the society (a procedure over the primitives, which others can adopt, revise or drop); substrate define_kind adds a kind of entry, with fields if pieces must fit together.

OFFERED, NEVER REQUIRED
""" + "\n".join("- " + x for x in offered) + f"""{ways}

THE SOCIETY REVIEWS ITSELF
Every few ticks each agent reflects, and the society can change its credit law, its practices and what each turn shows. A change that does not help accuracy is reverted. A review never says who does what: who works with whom stays yours to decide."""

    # ------------------------------------------------------------------ results

    async def finalize(self, k):
        scored = [t for t in self.questions if t not in self.audit_set]

        async def rule(tid: str):
            a = self.sheet_attempt(tid)
            return tid, (await self.judge(self.questions[tid], a["final"], self.final_model, self.final_effort) if a else 0.0)

        slots = asyncio.Semaphore(8)

        async def held(tid: str):
            async with slots:
                return await rule(tid)

        marks = dict(await asyncio.gather(*(held(t) for t in scored)))
        self.final = self.summary(marks, "scored")
        self.final["marks"] = {t: v for t, v in marks.items()}
        self.final["audited"] = self.summary({t: self.best_audit.get(t, 0.0) for t in self.audit_set}, "audited")
        if self.judge_all:
            await self._judge_attempts(slots)
        print(f"FrontierScience: {self.final['score']} on {self.final['n']} scored questions "
              f"(olympiad {self.final['olympiad']}, research pass {self.final['research_pass']}, points {self.final['research_points']})", flush=True)

    def summary(self, marks: dict, name: str) -> dict:
        oly = [v for t, v in marks.items() if self.questions[t]["track"] == "olympiad" and v is not None]
        res = [v for t, v in marks.items() if self.questions[t]["track"] == "research" and v is not None]
        passed = sum(v * 10 >= PASS_POINTS for v in res)
        total = len(oly) + len(res)
        return {"name": name, "n": total, "no_verdict": sum(v is None for v in marks.values()),
                "score": round((sum(oly) + passed) / total, 4) if total else None, "olympiad": round(sum(oly) / len(oly), 4) if oly else None,
                "research_pass": round(passed / len(res), 4) if res else None, "research_points": round(10 * sum(res) / len(res), 2) if res else None}

    async def _judge_attempts(self, slots):
        """Every attempt, by the audit's judge, after the run: did trust and agreement predict what was right? Nobody sees this."""
        rows = [a for v in self.attempts.values() for a in v]

        async def one(a: dict):
            async with slots:
                a["judged"] = await self.judge(self.questions[a["task"]], a["final"], self.audit_model, self.audit_effort)

        await asyncio.gather(*(one(a) for a in rows))

    def structure_measures(self, k) -> dict:
        acts = self.action_facts(k, 0)
        by = Counter(a["agent"] for v in self.attempts.values() for a in v)
        subjects = defaultdict(Counter)
        for v in self.attempts.values():
            for a in v:
                subjects[a["agent"]][self.questions[a["task"]]["subject"]] += 1
        focus = [c.most_common(1)[0][1] / sum(c.values()) for c in subjects.values()]
        read = Counter()
        for (_, who), n in self.reads.items():
            read[who] += n
        return {"attempters": len(by), "attempts": sum(by.values()), "attempts_by_agent": dict(by), "top_attempter_share": round(max(by.values()) / sum(by.values()), 3) if by else 0.0,
                "subject_focus": round(sum(focus) / len(focus), 3) if focus else 0.0, "independent": sum(a["blind"] for v in self.attempts.values() for a in v),
                "reconciled": sum(bool(a["builds_on"]) for v in self.attempts.values() for a in v), "contests": sum(bool(a.get("contested")) for v in self.attempts.values() for a in v),
                "tasks": self.delegation_measures(k),
                "most_read": read.most_common(3), "reads": sum(read.values()), "roles_taken": len(self.titles), "teams": len(self.teams),
                "suggestions": len(self.suggestions), "action_mix": acts["action_mix"], "working_share": acts["working_share"], "organizing_share": acts["organizing_share"]}

    def score(self) -> dict:
        held = self.progress()
        k = self._k
        answered = sum(bool(self.attempts[t]) for t in self.questions)
        return {"score": self.final.get("score") if self.final else round(held["passed"] / max(1, held["total"]), 4),
                "coverage": round(answered / len(self.questions), 4), "score_base": None, "score_composite": None, "structure_f1": None, "val": None,
                **(self.explore_score(k) if k is not None else {}),
                "attempts": sum(len(v) for v in self.attempts.values()), "audited_gain": held["passed"], "law_version": self.law_v,
                "steps": sum(self.used.values()), "payments": len(self.payments),
                **({key: k.counts[key] for key in ("pushed", "collapsed", "interrupts")} if k is not None and getattr(k, "sync", False) else {})}

    def agent_export(self, aid: str) -> dict:
        return {"instruments": [GAME], "record": self.track_record(aid), "experiments": self.used[aid],
                "role": self.titles.get(aid), "attributed": round(self.attributed[aid], 3)}

    def export(self) -> dict:
        k = self._k
        return {"kind": "science", "structure": "science", "final": self.final, "held_out": self.held_log, "audits": self.audits,
                "questions": {t: {key: q.get(key) for key in ("subject", "track", "task_group_id")} for t, q in self.questions.items()},
                "audit_set": sorted(self.audit_set, key=lambda t: int(t[1:])), "attempts": {t: v for t, v in self.attempts.items()},
                "sheet": {t: (a["id"] if (a := self.sheet_attempt(t)) else None) for t in self.questions}, "compared": self.compared,
                "judge_gaps": self.judge_gaps, "attributed": dict(self.attributed), "law": self.law, "law_log": self.law_log, "norms": self.norms,
                "suggestions": self.suggestions, "titles": self.titles, "title_log": self.title_log, "payments": self.payments,
                "task_log": k.task_log if k is not None else [],
                "reads": [[a, b, n] for (a, b), n in self.reads.most_common()], "traces": dict(self.traces), "steps": dict(self.used),
                "incentive": self.incentive, **({"structure_measures": self.structure_measures(k)} if k is not None else {}),
                **(self.explore_export(k) if k is not None else {})}

    # ------------------------------------------------------------------ scripted agents (mock backend)

    def mock_actions(self, k, aid, t, rng):
        n, tick = int(aid[1:]), k.tick
        ids = sorted(self.questions, key=lambda x: int(x[1:]))
        tid = ids[(n + tick) % len(ids)]
        q = self.questions[tid]
        right = (n + tick) % 3 != 0
        acts = [("open_task", {"task": tid}), ("calc", {"code": "print(2+2)"})]
        rid = [r for r, v in self.receipts.items() if v["agent"] == aid][-1:]
        acts.append(("answer", {"task": tid, "working": "worked it out", "final": q["answer"] if right else "something else", "evidence": rid}))
        if tick == 1:
            acts.insert(0, ("role", {"title": "solver" if n % 2 else "critic"}))
        return acts + [x for x in self.explore_mock(k, aid, t, rng) if x[0] not in ("play", "focus")]
