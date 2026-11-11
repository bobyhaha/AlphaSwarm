"""SciGym (Duan et al., NeurIPS 2025) as a world: a systems-biology dry lab.

Each task is a BioModels system from SciGym's small split (137 systems; Hugging Face h4duan/scigym-sbml, saved under
ext/scigym_data/small). The agents see the incomplete SBML model (all species, every reaction removed; ids scrambled by
SciGym) and must submit a complete model. As in SciGym's default configuration ('fully observable', perturbation
actions 'observe' and 'change_initial_concentration'):
- experiment runs the TRUE system, with default settings or new initial concentrations, and stores the time series;
- run_code runs Python (numpy, pandas, scipy, sklearn, libsbml) with SciGym's globals: input_sbml_string,
  experiment_history (here: the experiments on this task posted to boards the agent can read), simulate(sbml_string)
  and shared_variables; defining final_sbml saves it to the workspace;
- submit sets the society's model for a system (a file from the workspace or inline SBML); it must load and simulate.

Grading (no LLM judge) follows SciGym's evaluator: exact recovery of the missing reactions (precision/recall/F1 of the
added reactions against the missing ones, lenient = reactants+products, strict = +modifiers), typed species-interaction
F1, and the SMAPE of simulated time courses under the original and perturbed initial conditions. The world's score for a
system is one of them (world option metric=rp_f1 by default); the rest are kept in the export.

Deviations from the official harness: many agents share a system (experiments are shared through boards); experiment
previews show 12 sampled rows (SciGym shows 50); perturbed SMAPE uses 10 rounds at 20% noise and compares perturbed
truth with perturbed prediction (SciGym's code compares with the unperturbed truth); a per-agent experiment budget
replaces SciGym's iteration limit.
"""
from __future__ import annotations

import asyncio
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from ..kernel import ActionError, clip
from .base import Submission, Task, TaskWorld
from .sandbox import run_python

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ext" / "scigym_data" / "small"
INDEX = ROOT / "ext" / "scigym_data" / "small_index.json"
BENCH_PY = ROOT / ".venv-bench" / "bin" / "python"
WORKER = Path(__file__).with_name("scigym_worker.py")
METRICS = ("rp_f1", "rpm_f1", "smape", "perturb_smape")

_slots: asyncio.Semaphore | None = None


async def call_worker(req: dict, timeout: float = 600) -> dict:
    """One request to scigym_worker.py in .venv-bench (simulation needs libsbml/roadrunner)."""
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(8)
    async with _slots:
        proc = await asyncio.create_subprocess_exec(str(BENCH_PY), "-I", str(WORKER), stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(json.dumps(req).encode()), timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"timed out after {timeout:.0f}s"}
        finally:
            if proc.returncode is None:  # timed out or cancelled
                proc.kill()
                await proc.wait()
    text = out.decode("utf-8", "replace")
    if "@@RESULT@@" not in text:
        return {"ok": False, "error": "the simulator failed"}
    try:  # native libraries may print after the result, so decode only the first JSON value
        return json.JSONDecoder().raw_decode(text.rsplit("@@RESULT@@", 1)[1].lstrip())[0]
    except ValueError:
        return {"ok": False, "error": "the simulator returned an unreadable result"}


def load_index() -> dict:
    """Species, sizes and the canonical incomplete model of every system (built once with the worker, then cached)."""
    if not INDEX.exists():
        dirs = sorted(str(p) for p in DATA.iterdir() if (p / "truth.xml").exists())
        res = subprocess.run([str(BENCH_PY), "-I", str(WORKER)], input=json.dumps({"op": "describe", "dirs": dirs}),
                             capture_output=True, text=True, timeout=1800)
        raw = json.JSONDecoder().raw_decode(res.stdout.rsplit("@@RESULT@@", 1)[1].lstrip())[0]
        INDEX.write_text(json.dumps({Path(d).name: v for d, v in raw.items()}))
    return json.loads(INDEX.read_text())


def size_class(n_missing: int) -> str:
    return "2-3 reactions" if n_missing <= 3 else "4-5 reactions" if n_missing <= 5 else "6-7 reactions" if n_missing <= 7 else "8+ reactions"


EXPERIMENT_TOOL = {
    "name": "experiment",
    "description": "Run an experiment on a system's real biology. Without concentrations it observes the system with default "
                   "settings; with them it first sets the initial concentrations of the listed species (constant and "
                   "boundary species cannot be changed). The full time series is stored as experiment_history[id] for "
                   "run_code; a short preview is posted to the board in 'to' (default your private board). Uses one unit "
                   "of your experiment budget.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string"},
        "concentrations": {"type": "object", "description": "species id -> initial concentration, e.g. {\"id_bya5\": 250}"},
        "to": {"type": "string", "description": "board to post the preview to: 'private', a group id, 'global', ..."}},
        "required": ["task"]}}

CODE_TOOL = {
    "name": "run_code",
    "description": "Run Python for a system (numpy, pandas, scipy, sklearn, libsbml). Globals: input_sbml_string (the "
                   "incomplete model), experiment_history (dict of DataFrames: this system's experiments on boards you can "
                   "read), simulate(sbml_string) -> DataFrame with 'Time' and species columns, shared_variables.add(name, "
                   "value) / .access(name). Print what you want to see. Defining final_sbml saves it to final_sbml.xml in "
                   "your workspace for submit. Files you write persist in your workspace for this system.",
    "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "code": {"type": "string"}},
                   "required": ["task", "code"]}}

SUBMIT_MODEL = {
    "name": "submit",
    "description": "Set the society's model for a system (replaces the previous one). Give a file from your workspace for "
                   "this system (e.g. final_sbml.xml) or the SBML inline. The model must load and simulate. Optionally on "
                   "behalf of a group you may act for.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string"},
        "file": {"type": "string", "description": "file in your workspace for this system, e.g. final_sbml.xml"},
        "sbml": {"type": "string", "description": "the complete SBML model, if not given as a file"},
        "rationale": {"type": "string", "description": "optional: which reactions and why"},
        "as": {"type": "string", "description": "optional group id"}},
        "required": ["task"]}}

SETUP = """
import json as _json, pickle as _pickle, sys as _sys
_sys.path.insert(0, {bench!r})
import scigym_worker as _sg
_sg._SEDML = {sedml!r}
simulate = _sg.simulate
import pandas as pd
input_sbml_string = open({partial!r}).read()
experiment_history = {{_k: pd.read_csv(_v) for _k, _v in _json.load(open({manifest!r})).items()}}
class _SharedVariables:
    def add(self, name, val):
        _pickle.dump(val, open(f".shared_{{name}}.pkl", "wb"))
    def access(self, name):
        return _pickle.load(open(f".shared_{{name}}.pkl", "rb"))
shared_variables = _SharedVariables()
"""

SAVE_FINAL = """

try:
    final_sbml
except NameError:
    pass
else:
    open("final_sbml.xml", "w").write(str(final_sbml))
    print("[final_sbml saved to final_sbml.xml]")
"""


class SciGymWorld(TaskWorld):
    name = "scigym"
    unit = "system"
    TOOLS = [EXPERIMENT_TOOL, CODE_TOOL]

    def __init__(self, n_agents: int, tasks: list[Task], *, experiments: int = 12, metric: str = "rp_f1",
                 rounds: int = 10, noise: float = 0.2, world_agents: int | None = None, **kw):
        super().__init__(n_agents, tasks, **kw)
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.metric, self.rounds, self.noise = metric, rounds, noise
        per = experiments * (world_agents or n_agents) if self.single else experiments
        self.budget = {a: per for a in self.home}
        self.experiments: dict[str, list[dict]] = defaultdict(list)  # task -> experiments (id, author, board, path)
        self.details: dict[str, dict] = {}

    @property
    def all_tools(self) -> list[dict]:
        from .base import OPEN_TOOL
        return [OPEN_TOOL, *self.TOOLS, SUBMIT_MODEL]

    # ------------------------------------------------------------------ paths

    def _dir(self, t: Task) -> Path:
        return Path(t.meta["dir"])

    def _task_root(self, t: Task) -> Path:
        root = Path(self.workdir) / t.id
        (root / "data").mkdir(parents=True, exist_ok=True)
        partial = root / "input.xml"
        if not partial.exists():
            partial.write_text(t.meta["index"]["partial"])
        return root

    def _workspace(self, t: Task, aid: str) -> Path:
        ws = self._task_root(t) / aid
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    # ------------------------------------------------------------------ views

    def mission(self) -> str:
        return (f"The society works on {len(self.tasks)} biological systems (from a curated model database). For each "
                "system you get an incomplete SBML model: every species is listed and observable, but all reactions are "
                "missing. The goal is a complete SBML model per system that captures the true mechanisms; you do not need "
                "to add species.\n"
                "- experiment runs the real system (default settings, or new initial concentrations) and stores its time "
                "series; run_code analyzes data and builds models (libsbml: createReaction, createReactant, createProduct, "
                "createKineticLaw; then libsbml.writeSBMLToString). simulate() runs a candidate model.\n"
                "- At the end each system's submitted model is compared with the true system: whether the missing "
                "reactions were recovered (reactants and products) and how well its simulations match the true time "
                "courses, also under perturbed initial conditions. Mechanisms matter more than fitting one experiment.\n"
                f"- Each agent has a budget of {next(iter(self.budget.values()), 0)} experiments; experiments posted on a "
                "board can be analyzed by everyone who can read that board.")

    def statement(self, k, aid: str, t: Task) -> str:
        ix = t.meta["index"]
        rows = [f"{s['id']}: initial {s['init'] if s['init'] is None else format(s['init'], 'g')}" + (" (constant)" if s["constant"] else "") + (" (boundary)" if s["boundary"] else "")
                for s in ix["species"]]
        visible = [e for e in self.experiments.get(t.id, []) if k.can_read(aid, e["board"])]
        exps = "; ".join(f"{e['id']} by {e['author']} ({e['desc']})" for e in visible[-12:]) or "none yet"
        return (f"a biological system where all species are observable. Discover the missing reactions; you do not need to "
                f"add new species.\nSpecies: {'; '.join(rows)}.\nTime course: 0 to {ix['end_time']:g} "
                f"({ix['steps']} steps).\nExperiments you can analyze: {exps}.\n"
                f"Incomplete SBML model (input_sbml_string in run_code):\n{ix['partial']}")

    def observe_extra(self, k, aid: str) -> str:
        return f"Experiments left: {self.budget.get(aid, 0) - self.used[aid]}/{self.budget.get(aid, 0)}."

    # ------------------------------------------------------------------ tools

    async def _experiment(self, k, aid, a):
        t = self.task(a.get("task"))
        k._need(self.used[aid] < self.budget.get(aid, 0), "you have used your whole experiment budget")
        conc = a.get("concentrations") or {}
        k._need(isinstance(conc, dict), "concentrations must map species ids to numbers")
        try:
            conc = {str(s): float(v) for s, v in conc.items()}
        except (TypeError, ValueError):
            raise ActionError("concentrations must map species ids to numbers") from None
        board = k.board_id(aid, a.get("to", "private"))
        k._need(k.can_write(aid, board), f"you cannot write to {board}")
        root = self._task_root(t)
        eid = f"exp{len(self.experiments[t.id]) + 1}"
        path = root / "data" / f"{eid}.csv"
        res = await call_worker({"op": "experiment", "dir": str(self._dir(t)), "concentrations": conc, "out": str(path)}, 300)
        k._need(res.get("ok"), res.get("error", "the experiment failed"))
        self.used[aid] += 1
        self.note_focus(k, aid, t, 1.0)
        k.tick_counts["experiments"] += 1
        desc = "observe" if not conc else "initial " + ", ".join(f"{s}={v:g}" for s, v in conc.items())
        preview = _preview(path)
        content = (f"Experiment {eid} on {t.id} by {aid} ({desc}): {res['rows']} time points. Full data: "
                   f"experiment_history['{eid}'] in run_code for {t.id}.\n{preview}")
        e = k.post(board, aid, "data", content, notify=False)
        self.experiments[t.id].append({"id": eid, "author": aid, "board": board, "path": str(path), "desc": desc,
                                       "tick": k.tick, "entry": e.id})
        return "lab.experiment", content, t.id

    async def _run_code(self, k, aid, a):
        t = self.task(a.get("task"))
        code = str(a.get("code") or "")
        k._need(code.strip(), "the code is empty")
        root, ws = self._task_root(t), self._workspace(t, aid)
        visible = {e["id"]: e["path"] for e in self.experiments.get(t.id, []) if k.can_read(aid, e["board"])}
        (ws / ".experiments.json").write_text(json.dumps(visible))
        setup = SETUP.format(bench=str(WORKER.parent), sedml=str(self._dir(t) / "truth.sedml"), partial=str(root / "input.xml"),
                             manifest=str(ws / ".experiments.json"))
        ok, out = await run_python(code + SAVE_FINAL, ws, setup=setup, timeout=90, limit=3000, python=str(BENCH_PY))
        self.note_focus(k, aid, t, 1.0)
        return "lab.run_code", (out if ok else f"error:\n{out}"), t.id

    async def _submit(self, k, aid, a):
        t = self.task(a.get("task"))
        if a.get("file"):
            ws = self._workspace(t, aid)
            p = (ws / str(a["file"])).resolve()
            k._need(p.is_relative_to(ws.resolve()) and p.is_file(), f"no file '{a['file']}' in your workspace for {t.id}")
            sbml = p.read_text()
        else:
            sbml = str(a.get("sbml") or a.get("answer") or "")
        k._need(sbml.strip(), "give the model as a file from your workspace or inline as sbml")
        group = None
        if a.get("as"):
            group = k._group(a["as"])
            k._need(k.can_act_for(aid, group), f"you may not act for {group}")
        check = await call_worker({"op": "check", "sbml": sbml, "sedml": str(self._dir(t) / "truth.sedml")}, 120)
        k._need(check.get("ok"), f"the model does not load or simulate: {check.get('error', '')}")
        prev = self.submissions.get(t.id)
        self.submissions[t.id] = Submission(t.id, sbml, str(a.get("rationale") or "")[:2000], aid, group, k.tick)
        self.history.append({"t": k.tick, "task": t.id, "author": aid, "group": group, "reactions": check.get("reactions")})
        self.note_focus(k, aid, t, 3.0)
        k.tick_counts["published"] += 1
        if prev and prev.author != aid:
            k.notify(prev.author, aid, f"{aid} replaced your model for {t.id}")
        msg = f"submitted the society's model for {t.id} ({check.get('reactions')} reactions)"
        if prev:
            msg += f", replacing {prev.author}'s from t{prev.tick}"
        return "lab.submit", msg + ".", t.id

    # ------------------------------------------------------------------ grading (hidden from agents)

    async def finalize(self, k):
        """SciGym's controller asks for final_sbml when an agent's iterations run out, so every official episode ends
        with a model. Here a system nobody submitted gets the newest final_sbml.xml any agent saved for it, if it loads."""
        self.auto_submitted = []
        for t in self.tasks.values():
            if t.id in self.submissions or not (Path(self.workdir) / t.id).exists():
                continue
            for p in sorted((Path(self.workdir) / t.id).glob("*/final_sbml.xml"), key=lambda p: -p.stat().st_mtime):
                sbml = p.read_text()
                check = await call_worker({"op": "check", "sbml": sbml, "sedml": str(self._dir(t) / "truth.sedml")}, 120)
                if check.get("ok"):
                    self.submissions[t.id] = Submission(t.id, sbml, "auto: newest final_sbml.xml at the end", p.parent.name,
                                                        None, k.tick)
                    self.auto_submitted.append(t.id)
                    break
        await super().finalize(k)

    async def grade(self, k, tasks: list[str]) -> dict[str, float]:
        async def one(tid):
            t = self.tasks[tid]
            m = await call_worker({"op": "grade", "dir": str(self._dir(t)), "sbml": self.submissions[tid].answer,
                                   "rounds": self.rounds, "noise": self.noise, "seed": 0}, 900)
            self.details[tid] = m
            return tid, self.headline(m)
        return dict(await asyncio.gather(*(one(t) for t in tasks)))

    def headline(self, m: dict) -> float:
        if not m.get("valid"):
            return 0.0
        if self.metric in ("smape", "perturb_smape"):
            v = m.get("observe_smape" if self.metric == "smape" else "perturb_smape")
            return 0.0 if v is None else max(0.0, 1.0 - float(v))
        return float(m.get(self.metric) or 0.0)

    def export(self) -> dict:
        out = super().export()
        keys = ("rp_f1", "rpm_f1", "rp_precision", "rp_recall", "reactant_product_f1", "observe_smape", "perturb_smape")
        avg = {}
        for key in keys:
            vals = [self.details[t].get(key) for t in self.details if self.details[t].get(key) is not None]
            avg[key] = round(sum(vals) / len(vals), 4) if vals else None
        out |= {"metric": self.metric, "details": self.details, "mean_over_answered": avg,
                "auto_submitted": getattr(self, "auto_submitted", []),
                "experiments": {t: [{x: e[x] for x in ("id", "author", "board", "desc", "tick")} for e in v]
                                for t, v in self.experiments.items()}}
        for row in out["tasks"]:
            if row["answer"]:
                row["answer"] = dict(row["answer"], answer=clip(row["answer"]["answer"], 200))
        return out

    # ------------------------------------------------------------------ scripted agents (tests)

    def mock_actions(self, k, aid, t, rng):
        return [("open_task", {"task": t}), ("experiment", {"task": t}),
                ("submit", {"task": t, "sbml": self.tasks[t].meta["index"]["partial"]})]


def _preview(path: Path, rows: int = 12) -> str:
    """A compact table: evenly spaced rows, 4 significant digits."""
    lines = path.read_text().splitlines()
    head, body = lines[0].split(","), lines[1:]
    if not body:
        return "(no data)"
    step = max(1, (len(body) - 1) // (rows - 1))
    picked = body[::step][: rows - 1] + ([body[-1]] if (len(body) - 1) % step else [])
    order = [head.index("Time")] + [i for i, h in enumerate(head) if h != "Time"]
    fmt = lambda x: f"{float(x):.4g}"
    out = ["  ".join(head[i] for i in order)]
    for line in picked:
        cells = line.split(",")
        out.append("  ".join(fmt(cells[i]) for i in order))
    return "\n".join(out)


def make(cfg: dict, n_agents: int, seed: int = 0, single: bool = False) -> SciGymWorld:
    opts = cfg.get("world_opts") or {}
    idx = load_index()
    names = sorted(n for n, v in idx.items() if "error" not in v)
    if cfg.get("tasks"):
        names = sorted(random.Random(seed).sample(names, min(int(cfg["tasks"]), len(names))))
    tasks = []
    for name in names:
        ix = idx[name]
        ids = [s["id"] for s in ix["species"]]
        title = f"system with {len(ids)} species ({', '.join(ids[:6])}{'…' if len(ids) > 6 else ''})"
        tasks.append(Task("", title, family=name, domain=size_class(ix["n_missing"]), statement="",
                          meta={"dir": str(DATA / name), "index": ix}))
    return SciGymWorld(n_agents, tasks, seed=seed, home=cfg.get("home", 3), eval_at=tuple(cfg.get("eval_at") or ()),
                       single=single, world_agents=cfg.get("agents"), experiments=int(opts.get("experiments", 12)),
                       metric=opts.get("metric", "rp_f1"), rounds=int(opts.get("rounds", 10)),
                       noise=float(opts.get("noise", 0.2)))
