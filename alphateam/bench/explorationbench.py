"""ExplorationBench (Tencent-Hunyuan, arXiv 2609.30199) as a world: the two PUBLIC DEMO worlds only.

The evaluation worlds are private; the authors say demo-world scores are for development and are not comparable to
the leaderboard. Everything that decides a verdict is the official code (`ext/explorationbench`, Apache-2.0), run
in a subprocess by `explorationbench_worker.py`: the AlienCode interpreter (hardened mode) and the AlienLogic proof
checker, the v2 protocol's limits, worked examples, prompts, answer extraction and grading.

Official protocol (v2, checked against the code): a flawed manual plus fixed worked examples; 4 exploration rounds;
one tool per world. AlienCode: `run_aliencode`, at most 12 calls per round, at most 5 observed values per call (each
top-level EMIT argument is one, inside a loop each counts 3×; a call over the cap is rejected in full and still
counts as a call), at most 40 lines, output cut to 15 lines / 600 characters. AlienLogic: `check_proof`, at most 12
proofs per round (48 in all), opaque diagnostics. After every round (M1..M4, and M0 before exploring) a tool-disabled
copy of the conversation answers each held-out task, graded exactly by the interpreter / checker (an unprovable
theorem scores only when the answer declines with `ANSWER: UNPROVABLE`); the headline counts three answers per task.

How the society version maps onto it (deviations marked *):
- Tasks are the two worlds (family = domain = the world). `open_task` shows the official manual and worked examples.
- Tools keep the official names and per-call rules: `run_aliencode(code)`, `check_proof(proof)`. Results go back to
  the caller and are posted to a board of its choice (default its private board).
- * Budget: by default the SOCIETY shares the official single-system budget, 12 calls per world per round (pooled;
  unused calls do not carry over). `--world-opt probe_budget=per_agent:12` gives every agent its own 12 instead.
- * Rounds are spans of ticks (`rounds`=4, `ticks_per_round` = ticks // rounds by default); a round's budget unlocks
  at its first tick. M1..M3 are graded at the end of rounds 1..3, the last milestone at the end of the run.
- * What carries knowledge to the test is the society's rule report for each world (`submit`, latest one counts),
  not a conversation history: at each milestone a FRESH, tool-free answerer (the agents' model, via `self.llm`)
  gets the official system prompt (manual + worked examples), the society's current report for that world, and the
  official held-out prompt, and answers each held-out task `trials` (3) times. One re-ask, with the official nudge,
  when an answer holds nothing gradable. The answerer never explores, so nothing from the test reaches the society.
- * M0 is the same answerer with an empty report (the manual and worked examples alone).
- * Not computed: the rule-discovery diagnostic (S-expression report scored with an LLM-judge fallback), open-book
  arms, controls, Best@n / Mean@n over trajectories (run several seeds for that).
- Score = mean held-out accuracy (fraction of the trials answered correctly) over both worlds at the last milestone.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from ..kernel import ActionError, clip
from .base import Task, TaskWorld

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().with_name("explorationbench_worker.py")
BENCH_PY = ROOT / ".venv-bench" / "bin" / "python"
CODE, LOGIC = "AlienCode", "AlienLogic"

STUB_ANSWERS = {  # used when no LLM is attached (offline tests): gradable, mostly wrong
    CODE: "预测输出：0\n```aliencode\nCRAFT f(xs):\n    DELIVER 0\n```",
    LOGIC: "ANSWER: UNPROVABLE\n(stub)",
}

REPORT_INTRO = (  # the society's report, handed to the answerer in place of an exploration history
    "以下是一个研究团队在探索这个环境之后提交的规则报告（rule report）。你自己没有做过任何实验；"
    "上面的参考手册、固定示例和这份报告就是你掌握的全部信息。本轮为闭卷作答，没有可用工具。\n"
    "<<<RULE REPORT\n{report}\nRULE REPORT>>>")
NO_REPORT = "研究团队没有提交规则报告。你只有上面的参考手册和固定示例。本轮为闭卷作答，没有可用工具。"


def _python() -> str:
    return str(BENCH_PY) if BENCH_PY.exists() else sys.executable


def _env() -> dict:
    return {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}  # no credentials reach the worker


def run_worker_sync(jobs: list[dict], timeout: float = 120) -> list[dict]:
    p = subprocess.run([_python(), "-I", str(WORKER)], input=json.dumps({"jobs": jobs}), capture_output=True,
                       text=True, env=_env(), timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"ExplorationBench worker failed: {p.stderr[-800:]}")
    return json.loads(p.stdout)


@lru_cache(maxsize=1)
def official_setup() -> tuple[dict, dict]:
    """Manuals, worked examples, system prompts and held-out prompts, as the official harness builds them."""
    code, logic = run_worker_sync([{"op": "code_setup"}, {"op": "logic_setup"}])
    for part in (code, logic):
        if "error" in part:
            raise RuntimeError(f"ExplorationBench setup failed: {part['error']}")
    return code, logic


TOOLS = [
    {"name": "run_aliencode",
     "description": "Run an AlienCode program in the real AlienCode world and get its actual output. Each call uses one "
                    "call of the AlienCode budget (see your observation), even when it is rejected. Per call: at most 40 "
                    "lines and at most 5 observed values (each top-level EMIT argument is one; inside SWEEP/WHILE loops "
                    "each counts 3); a call over 5 is rejected in full and runs nothing. Output is cut to 15 lines / "
                    "600 characters. The program and its output are also posted to the board in 'to' (default your "
                    "private board).",
     "parameters": {"type": "object", "properties": {
         "code": {"type": "string", "description": "AlienCode source; EMIT(...) what you want to observe"},
         "to": {"type": "string", "description": "board to post the result to: 'private', a group id, 'global', ..."}},
         "required": ["code"]}},
    {"name": "check_proof",
     "description": "Submit one complete AlienLogic proof (PROOF ... END, no code fences) to the real verifier and get "
                    "its opaque diagnostic (accepted, reason_class, reason_id, ...). Each call uses one call of the "
                    "AlienLogic budget (see your observation). The proof and the verdict are also posted to the board "
                    "in 'to' (default your private board).",
     "parameters": {"type": "object", "properties": {
         "proof": {"type": "string"},
         "purpose": {"type": "string", "description": "optional: the hypothesis this probe tests"},
         "to": {"type": "string", "description": "board to post the result to"}},
         "required": ["proof"]}},
]


class ExplorationBench(TaskWorld):
    name = "explorationbench"
    unit = "world"
    TOOLS = TOOLS

    def __init__(self, n_agents: int, ticks: int, seed: int = 0, single: bool = False, rounds: int = 4,
                 ticks_per_round: int | None = None, budget: str = "pooled:12", trials: int = 3, m0: bool = True,
                 answer_tokens: int = 4096, concurrency: int = 8):
        code, logic = official_setup()
        self.official = {CODE: code, LOGIC: logic}
        tasks = [
            Task("", "AlienCode: a small calculation language whose operations may not behave as its manual says",
                 CODE, CODE, "", {"world": CODE}),
            Task("", "AlienLogic: a natural-deduction proof checker whose rules may carry hidden side conditions",
                 LOGIC, LOGIC, "", {"world": LOGIC}),
        ]
        super().__init__(n_agents, tasks, seed=seed, home=2, single=single)
        self.world_task = {t.meta["world"]: t.id for t in self.tasks.values()}
        self.rounds = max(1, min(rounds, ticks))
        self.tpr = ticks_per_round or max(1, ticks // self.rounds)
        self.ticks = ticks
        self.milestone_ticks = {r * self.tpr: r for r in range(1, self.rounds)}  # the last one is graded in finalize
        kind, _, n = budget.partition(":")
        if kind not in ("pooled", "per_agent") or not n.isdigit():
            raise ValueError(f"probe_budget must be pooled:N or per_agent:N, not '{budget}'")
        self.budget_kind, self.per_round = kind, int(n)
        self.trials, self.m0, self.answer_tokens = trials, m0, answer_tokens
        self.calls: dict[tuple, int] = {}  # (world, round) or (aid, world, round) -> calls used
        self.probes: list[dict] = []
        self.milestones: list[dict] = []
        self._concurrency = concurrency
        self._sem = None

    # ------------------------------------------------------------------ texts the agents see

    def statement(self, k, aid: str, t: Task) -> str:
        world = t.meta["world"]
        o = self.official[world]
        tool = "run_aliencode" if world == CODE else "check_proof"
        return (f"{world}. {t.title}.\nExplore it with {tool}; budget: {self._budget_line(k, aid, world)}.\n"
                f"What is tested: after each round, a solver who never saw your exploration gets this manual, these worked "
                f"examples and the society's current rule report for {world} (submit to {t.id}; the latest report "
                f"counts), and answers held-out {world} tasks that nobody in the society sees.\n\n"
                f"=== Reference manual (the standard behaviour; the real world may differ) ===\n{o['manual']}\n\n"
                f"=== Worked examples (fixed) ===\n{o['demo_block']}")

    def mission(self) -> str:
        per = ("the society shares" if self.budget_kind == "pooled" else "each agent has") + f" {self.per_round} calls"
        return (f"Two small worlds look familiar but follow hidden rules that differ from their reference manuals: "
                f"AlienCode ({self.world_task[CODE]}), a calculation language, and AlienLogic ({self.world_task[LOGIC]}), "
                f"a natural-deduction proof checker. The goal is to discover how each world really behaves and write it "
                f"down as a rule report (submit to the world's task id).\n"
                f"- Exploration runs in {self.rounds} rounds of {self.tpr} ticks. In each round {per} to each world's "
                f"tool (run_aliencode, check_proof); unused calls do not carry over.\n"
                f"- After each round a solver who has never seen your exploration gets each world's manual, its worked "
                f"examples and the society's current rule report for that world (nothing else), and answers held-out tasks "
                f"in that world that nobody in the society ever sees: writing AlienCode programs, predicting outputs, "
                f"writing proofs, or declaring a theorem unprovable. The society's score is the solver's accuracy, averaged "
                f"over both worlds, after the last round.")

    def round_of(self, tick: int) -> int:
        return min(self.rounds, max(1, (tick - 1) // self.tpr + 1))

    def _left(self, k, aid: str, world: str) -> int:
        r = self.round_of(k.tick)
        key = (world, r) if self.budget_kind == "pooled" else (aid, world, r)
        return self.per_round - self.calls.get(key, 0)

    def _budget_line(self, k, aid: str, world: str) -> str:
        who = "society-wide" if self.budget_kind == "pooled" else "yours"
        return f"{self._left(k, aid, world)}/{self.per_round} calls left this round ({who})"

    def observe_extra(self, k, aid: str) -> str:
        r = self.round_of(k.tick)
        end = r * self.tpr if r < self.rounds else max(self.ticks, r * self.tpr)
        reports = "; ".join(f"{w} {self.world_task[w]}: " + (f"{len(self.submissions[self.world_task[w]].answer)} chars "
                                                              f"by {self.submissions[self.world_task[w]].author}"
                                                              if self.world_task[w] in self.submissions else "none")
                            for w in (CODE, LOGIC))
        return (f"EXPLORATION: round {r}/{self.rounds} (ends after tick {end}, then the solver is tested). "
                f"AlienCode {self._budget_line(k, aid, CODE)}; AlienLogic {self._budget_line(k, aid, LOGIC)}. "
                f"Rule reports: {reports}.")

    # ------------------------------------------------------------------ tools

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem[0] is not loop:
            self._sem = (loop, asyncio.Semaphore(self._concurrency))
        return self._sem[1]

    async def run_worker(self, jobs: list[dict], timeout: float | None = None) -> list[dict]:
        timeout = timeout or 30 + 12 * len(jobs)
        async with self._semaphore():
            proc = await asyncio.create_subprocess_exec(
                _python(), "-I", str(WORKER), env=_env(), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(json.dumps({"jobs": jobs}).encode()), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return [{"error": "TimeoutError: the program ran too long", "output": "[AlienError] TimeoutError",
                         "correct": False, "rejected": False, "units": 0}] * len(jobs)
        if proc.returncode != 0:
            msg = err.decode("utf-8", "replace")[-300:]
            return [{"error": f"worker exited with {proc.returncode}: {msg}", "output": "[AlienError] the program was "
                     "stopped (it used too much time or memory)", "correct": False, "rejected": False, "units": 0}] * len(jobs)
        return json.loads(out)

    def _charge(self, k, aid: str, world: str):
        if self._left(k, aid, world) <= 0:
            r = self.round_of(k.tick)
            nxt = f"; round {r + 1} starts at tick {r * self.tpr + 1}" if r < self.rounds else "; this was the last round"
            who = "the society has" if self.budget_kind == "pooled" else "you have"
            raise ActionError(f"{who} used all {self.per_round} {world} calls of round {r}{nxt}")
        r = self.round_of(k.tick)
        key = (world, r) if self.budget_kind == "pooled" else (aid, world, r)
        self.calls[key] = self.calls.get(key, 0) + 1
        self.used[aid] += 1

    async def _run_aliencode(self, k, aid, a):
        src = str(a.get("code") or "")
        k._need(src.strip(), "the code is empty")
        board = k.board_id(aid, a.get("to") or "private")
        k._need(k.can_write(aid, board), f"you cannot write to {board}")
        self._charge(k, aid, CODE)
        tid = self.world_task[CODE]
        self.note_focus(k, aid, self.tasks[tid], 1.0)
        res = (await self.run_worker([{"op": "code_probe", "code": src}]))[0]
        out = res.get("output") or res.get("error", "")
        r = self.round_of(k.tick)
        self.probes.append({"t": k.tick, "round": r, "world": CODE, "agent": aid, "input": src[:2000],
                            "output": out[:800], "rejected": res.get("rejected", False)})
        k.tick_counts["experiments"] += 1
        k.post(board, aid, "data", f"AlienCode run by {aid} (round {r}):\n```aliencode\n{clip(src, 1500)}\n```\noutput:\n{out}", notify=False)
        return "lab.run_aliencode", f"{out}\n[{self._budget_line(k, aid, CODE)}]", tid

    async def _check_proof(self, k, aid, a):
        proof = str(a.get("proof") or "")
        k._need(proof.strip(), "the proof is empty")
        board = k.board_id(aid, a.get("to") or "private")
        k._need(k.can_write(aid, board), f"you cannot write to {board}")
        self._charge(k, aid, LOGIC)
        tid = self.world_task[LOGIC]
        self.note_focus(k, aid, self.tasks[tid], 1.0)
        res = (await self.run_worker([{"op": "logic_probe", "proof": proof}]))[0]
        out = res.get("output") or res.get("error", "")
        r = self.round_of(k.tick)
        purpose = str(a.get("purpose") or "")
        self.probes.append({"t": k.tick, "round": r, "world": LOGIC, "agent": aid, "input": proof[:2000],
                            "output": out[:800], "accepted": res.get("accepted")})
        k.tick_counts["experiments"] += 1
        k.post(board, aid, "data", f"AlienLogic proof by {aid} (round {r})" + (f", testing: {clip(purpose, 200)}" if purpose else "")
                   + f":\n{clip(proof, 1500)}\nverdict: {out}", notify=False)
        return "lab.check_proof", f"{out}\n[{self._budget_line(k, aid, LOGIC)}]", tid

    def mock_actions(self, k, aid, t, rng):
        world = self.tasks[t].meta["world"]
        if world == CODE:
            probe = ("run_aliencode", {"code": rng.choice(["EMIT(1)", "EMIT(UNDER(2, 2))", "EMIT(GAUGE(STRAND(1, 2)))",
                                                            "EMIT(PLUCK(STRAND(5, 6, 7), 0))"])})
        else:
            probe = ("check_proof", {"proof": rng.choice([
                "PROOF\npremises: AND(p, q)\ngoal: q\n1. AND(p, q) [premise]\n2. q [AND_E2 1]\nEND",
                "PROOF\npremises: p, NOT(p)\ngoal: q\n1. p [premise]\n2. NOT(p) [premise]\n3. BOT [NOT_E 1, 2]\n4. q [BOT_E 3]\nEND"])})
        return [("open_task", {"task": t}), probe,
                ("submit", {"task": t, "answer": f"{world} rule report by {aid}: integer literals look shifted; check later."})]

    # ------------------------------------------------------------------ grading (hidden from agents)

    async def _answer(self, system: str, prompt: str, world: str) -> str:
        if self.llm is None:
            return STUB_ANSWERS[world]
        try:
            return await self.llm(system, prompt, max_tokens=self.answer_tokens)
        except Exception as exc:  # a failed call is a wrong answer, as in the official harness
            return f"(answerer call failed: {type(exc).__name__})"

    async def _grade_world(self, world: str, report: str) -> dict:
        o = self.official[world]
        prefix = REPORT_INTRO.format(report=report) if report.strip() else NO_REPORT
        jobs = [(t, i) for t in o["tests"] for i in range(self.trials)]
        replies = await asyncio.gather(*(self._answer(o["system"], f"{prefix}\n\n---\n\n{t['prompt']}", world) for t, _ in jobs))
        replies = list(replies)
        op = "code" if world == CODE else "logic"
        empty = await self.run_worker([{"op": f"{op}_unanswered", "task": t["id"], "response": r}
                                       for (t, _), r in zip(jobs, replies)])
        redo = [j for j, e in enumerate(empty) if e.get("unanswered")]
        if redo and self.llm is not None:  # one re-ask with the official nudge, as the harness does
            def again(t):
                if world == CODE:
                    return f"{prefix}\n\n---\n\n{t['prompt']}\n\n{o['unsubmitted_retry']}"
                return f"{prefix}\n\n---\n\n{t['retry_prompt']}"
            more = await asyncio.gather(*(self._answer(o["system"], again(jobs[j][0]), world) for j in redo))
            for j, r in zip(redo, more):
                replies[j] = r
        graded = await self.run_worker([{"op": f"{op}_grade", "task": t["id"], "response": r}
                                        for (t, _), r in zip(jobs, replies)])
        per_task: dict[str, list[bool]] = {}
        samples = []
        for (t, i), r, g in zip(jobs, replies, graded):
            per_task.setdefault(t["id"], []).append(bool(g.get("correct")))
            samples.append({"task": t["id"], "trial": i, "correct": bool(g.get("correct")),
                            "verdict": clip(g.get("output") or g.get("error", ""), 200), "reply": clip(r, 400)})
        acc = {tid: sum(v) / len(v) for tid, v in per_task.items()}
        return {"score": sum(acc.values()) / len(acc), "per_task": acc, "samples": samples,
                "report_chars": len(report), "reask": len(redo)}

    async def _milestone(self, k, label: str, t: int, reports: dict[str, str]) -> dict:
        res = await asyncio.gather(*(self._grade_world(w, reports[w]) for w in (CODE, LOGIC)))
        m = {"label": label, "t": t, **{w: r for w, r in zip((CODE, LOGIC), res)},
             "score": round(sum(r["score"] for r in res) / 2, 4)}
        self.milestones.append(m)
        return m

    async def grade(self, k, tasks: list[str]) -> dict[str, float]:
        out = {}
        for tid in tasks:
            world = self.tasks[tid].meta["world"]
            report = self.submissions[tid].answer if tid in self.submissions else ""
            out[tid] = (await self._grade_world(world, report))["score"]
        return out

    async def on_tick_end(self, k):
        if k.tick in self.milestone_ticks:
            await self.evaluate(k, label=f"M{self.milestone_ticks[k.tick]}")

    async def evaluate(self, k, final: bool = False, label: str | None = None):
        if self.m0 and not any(m["label"] == "M0" for m in self.milestones):
            await self._milestone(k, "M0", 0, {CODE: "", LOGIC: ""})
        m = next((x for x in self.milestones if x["t"] == k.tick and x["label"] != "M0"), None)  # already graded
        if m is None:
            reports = {w: (self.submissions[self.world_task[w]].answer if self.world_task[w] in self.submissions else "")
                       for w in (CODE, LOGIC)}
            m = await self._milestone(k, label or f"M{self.round_of(k.tick)}", k.tick, reports)
        self.task_scores = {self.world_task[w]: m[w]["score"] for w in (CODE, LOGIC)}
        self.evaluations.append({"t": k.tick, "final": final, "label": m["label"], "score": m["score"],
                                 "answered": len(self.submissions),
                                 "by_domain": {w: round(m[w]["score"], 4) for w in (CODE, LOGIC)}})

    def author_scores(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for t, sc in self.task_scores.items():
            if t in self.submissions:
                out.setdefault(self.submissions[t].author, []).append(sc)
        return out

    def export(self) -> dict:
        data = super().export()
        data["protocol"] = {"rounds": self.rounds, "ticks_per_round": self.tpr, "budget": f"{self.budget_kind}:{self.per_round}",
                            "trials": self.trials, "worlds": "public demo worlds (not comparable to the leaderboard)"}
        data["milestones"] = self.milestones
        data["probes"] = self.probes
        data["calls"] = {f"{'/'.join(map(str, key))}": v for key, v in self.calls.items()}
        return data


def make(cfg: dict, n_agents: int, seed: int = 0, single: bool = False) -> ExplorationBench:
    opts = cfg.get("world_opts") or {}
    tpr = opts.get("ticks_per_round")
    return ExplorationBench(n_agents, cfg["ticks"], seed=seed, single=single, rounds=int(opts.get("rounds", 4)),
                            ticks_per_round=int(tpr) if tpr else None, budget=opts.get("probe_budget", "pooled:12"),
                            trials=int(opts.get("trials", 3)), m0=opts.get("m0", "1") not in ("0", "false", "no"),
                            answer_tokens=int(opts.get("answer_tokens", 4096)))
