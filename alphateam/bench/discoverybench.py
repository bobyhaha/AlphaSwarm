"""DiscoveryBench (Majumder et al., ICLR 2025; github.com/allenai/discoverybench) as a world.

Tasks are the 239 queries of the REAL TEST split. Each asks for a data-driven scientific hypothesis about one or more
datasets; the answer is a natural-language hypothesis (`answer`) with an optional workflow summary (`rationale`).

- Family = dataset folder (e.g. nls_ses), domain = the metadata's domain (e.g. sociology); for the mirroring metric.
- Agents see what the official agents see (`utils/autonomous_single_agent.py:get_dv_query_for_real`): dataset names,
  descriptions and column descriptions, and the query. Domain knowledge and workflow tags are off by default, as in the
  paper's main setting. The metadata's `hypotheses` (empty for every test task) and the answer key are never shown.
- run_code runs Python in the agent's own working directory for a task, which holds copies of the task's dataset files.
- Grading follows the official evaluation (`eval/new_eval.py:run_eval_gold_vs_gen_NL_hypo_workflow`, prompts verbatim):
  an LLM judge splits gold and predicted hypotheses into sub-hypotheses, matches them by context, then scores variables
  (F1) and relation (1 / 0.5 / 0); the Hypothesis Match Score is recall_context × mean accuracy. Default judge: gpt-4o.
  Deviations (all to keep the score in [0, 1] and the bill down): unparseable judge output scores 0 where the official
  code returns -1; the var/rel calls get 512 output tokens (the official helper overrides its own 512 with 250); gold
  sub-hypotheses are extracted once per task and reused across evaluations; temperature and JSON mode are passed only if
  the runner's completion helper accepts them. The gold workflow is empty, as in the answer key.
"""
from __future__ import annotations

import asyncio
import csv
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from ..kernel import ActionError, clip
from .base import Task, TaskWorld
from .sandbox import run_python

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ext" / "discoverybench"
BENCH_PYTHON = ROOT / ".venv-bench" / "bin" / "python"

RUN_CODE = {
    "name": "run_code",
    "description": "Run Python code in your working directory for a task, which holds copies of the task's dataset files "
                   "(pandas, numpy, scipy, statsmodels and scikit-learn are installed). Files you write there persist "
                   "between calls. What the code prints is returned; long output is cut.",
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string", "description": "task id, e.g. T12"},
        "code": {"type": "string"}},
        "required": ["task", "code"]}}

ANSWER_FORMAT = ("In the final answer, please write down a scientific hypothesis in natural language, derived from the "
                 "provided dataset, clearly stating the context of hypothesis (if any), variables chosen (if any) and "
                 "relationship between those variables (if any) including any statistical significance. Also generate a "
                 "summary of the full workflow starting from data loading that led to the final answer.\n"
                 "Submit the hypothesis as `answer` and the workflow summary as `rationale`.")


def _flag(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def dataset_meta_json(metadata: dict, use_column_metadata: bool = True) -> list[dict]:
    """eval/new_eval.py:prepare_dataset_metadata_json for the real split."""
    return [{"dataset_description": d["description"],
             "columns": [{"name": c["name"], "description": c["description"]} for c in d["columns"]["raw"]]
             if use_column_metadata else []}
            for d in metadata["datasets"]]


def statement_for(metadata: dict, query: str, domain_knowledge: bool = False, workflow_tags: bool = False) -> str:
    """utils/autonomous_single_agent.py:get_dv_query_for_real, without metadata['hypotheses'] (empty in the test split)."""
    meta = ""
    for d in metadata["datasets"]:
        meta += "Dataset name: " + d["name"]
        meta += "Dataset description: " + d["description"]
        meta += "\nBrief description of columns: "
        for col in d["columns"]["raw"]:
            meta += col["name"] + ": " + col["description"] + ", "
        meta += "\n"
    text = meta + f"\nQuery: {query}"
    if domain_knowledge and metadata.get("domain_knowledge"):
        text += "\nAdditionally, we provide some hints that might be useful to solve the task. Domain Knowledge: \n" + \
                metadata["domain_knowledge"] + ".\n"
    if workflow_tags and metadata.get("workflow_tags"):
        text += "\nThe meta tags are: " + metadata["workflow_tags"] + ".\n"
    files = ", ".join(d["name"] for d in metadata["datasets"])
    return (f"{text}\n\nThe dataset files ({files}) are in your run_code working directory for this task.\n{ANSWER_FORMAT}")


def load_tasks(data: Path = DATA, domain_knowledge: bool = False, workflow_tags: bool = False) -> list[Task]:
    test = data / "discoverybench" / "real" / "test"
    with open(data / "eval" / "answer_key_real.csv") as f:
        key = {(r["dataset"], int(r["metadataid"]), int(r["query_id"])): r["gold_hypo"] for r in csv.DictReader(f)}
    tasks = []
    for path in sorted(test.glob("*/metadata_*.json"), key=lambda p: (p.parent.name, int(p.stem.split("_")[1]))):
        ds, mid = path.parent.name, int(path.stem.split("_")[1])
        metadata = json.loads(path.read_text())
        for group in metadata["queries"]:
            for q in group:
                gold = key[(ds, mid, int(q["qid"]))]
                tasks.append(Task(
                    id="", title=q["question"], family=ds, domain=metadata["domain"],
                    statement=statement_for(metadata, q["question"], domain_knowledge, workflow_tags),
                    meta={"dataset": ds, "metadataid": mid, "qid": int(q["qid"]), "query": q["question"],
                          "gold": gold, "gold_workflow": "", "files": [str(path.parent / d["name"]) for d in metadata["datasets"]],
                          "datasets_json": dataset_meta_json(metadata)}))
    return tasks


def stratified_subset(tasks: list[Task], n: int, seed: int) -> list[Task]:
    """n tasks, dealt round-robin across families in a seeded random order (so every family is represented)."""
    rng = random.Random(seed)
    by_fam = defaultdict(list)
    for t in tasks:
        by_fam[t.family].append(t)
    fams = sorted(by_fam)
    rng.shuffle(fams)
    for f in fams:
        rng.shuffle(by_fam[f])
    out = []
    while len(out) < min(n, len(tasks)):
        for f in fams:
            if by_fam[f] and len(out) < n:
                out.append(by_fam[f].pop())
    return out


# ---------------------------------------------------------------------------------------------------------------- judge
# Prompts copied from eval/new_eval.py and utils/openai_helpers.py (allenai/discoverybench, Apache-2.0).

SYSTEM_PLAIN = ("You are a helpful assistant who is not talkative. You only respond with the exact answer to a query "
                "without additional conversation.")
SYSTEM_DIMENSION = ("You are an AI assistant that helps evaluate a data-driven hypothesis. You are a helpful assistant who "
                    "is not talkative. You only respond with the exact answer to a query without additional conversation.")


def extraction_prompt(datasets_json, hypo, workflow) -> str:
    extraction = f"""\
        Given a set of dataset columns, a ground-truth hypothesis, and the analysis workflow used, your task is to extract three dimensions that define the hypothesis: Context, Variables, and Relations. \
        Here are the definitions for these dimensions:
        - Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., “for men over \
        the age of 30”, “in Asia and Europe”. If the context applies to the full dataset, then extract the context from the dataset_descrption.
        - Variables: Known concepts that interact in a meaningful way under a given context to \
        produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
        - Relations: Interactions between a given set of variables under a given context to produce \
        the hypothesis. E.g., “quadratic relationship”, “inversely proportional”, piecewise conditionals, \
        or "None" if there is no interacting relationship.
        Make sure to only use the information present in the hypothesis and the workflow. Do not add any new information. \
        For each dimension, be specific, and do not omit any important details.

        Here is the metadata for the task:
        ```json
        {{
        "datasets": %s,
        "hypothesis": "%s",
        "workflow": "%s"
        }}
        ```

        Return your answer as a JSON object in the following format:
        ```json
        {{
        "sub_hypo": [
            {{
                "text": the hypothesis in natural language,
                "context": a short text description of the context of the hypothesis,
                "variables": a list of columns involved in the hypothesis,
                "relations": a short text description of the relationship between the variables of the hypothesis
            }},
            ...
        ]
        }}```
        """
    return extraction % (datasets_json, hypo, workflow)


def context_prompt(gold_hyp, gold_context, pred_hyp, pred_context) -> str:
    return f"""\
        Given a gold hypothesis, a gold context, a predicted hypothesis, and a predicted context, your task is \
        to determine if the predicted context semantically matches the ground-truth context. \
        Here is the definition for Context: Boundary conditions that limit the scope of a sub-hypothesis. E.g., “for men over the age of 30”, “in Asia and Europe”. If the context applies to the full dataset, then the context is derived from the dataset_descrption. \
        Here is the definition for Context: Boundary conditions that limit the scope of a sub-hypothesis. E.g., “for men over the age of 30”, “in Asia and Europe”. If the context applies to the full dataset, then the context is derived from the dataset_descrption. \
        If the predicted context matches the gold context, return true, otherwise return false.
        If both gold and predicted hypotheses are defined over the context of the full dataset, then also return true.
        If both gold and predicted hypotheses are defined over the context of the full dataset, then also return true.

        Here is the metadata for the task:
        ```json
        {{
            "gold_hypothesis": "{gold_hyp}",
            "gold_context": "{gold_context}",
            "predicted_hypothesis": "{pred_hyp}",
            "predicted_context": "{pred_context}"
        }}
        ```

        Return your answer as a JSON object in the following format:
        ```json
        {{
            "match": true or false
        }}
        ```"""


VAR_QUESTION = """\
        Question: For both HypoA and HypoB, what are the different variables found in the hypotheses? \
        Return your answer as a JSON object in the following format:
        ```json
        {{
        "sizeA": num of variables used in HypoA
        "sizeB": num of variables used in HypoB
        "intersection": num of variables common in HypoA and HypoB. Use *fuzzy matching* to determine intersection, accounting for paraphrases or slightly different surface forms
        "explanation": a short text explanation about the variables
        }}```
        Answer:"""

REL_QUESTION = """\
        Question: Does HypoB exhibit the same relation as HypoA?
        Compare using following example hierarchy of relationships (based on specificity): \
        "there exists a relationship" > "positive relationship" > "positive AND (linear OR quadratic)" > "positive AND linear".
        Options: A) very similar B) similar but general than HypoA C) different
        Return your answer as a JSON object in the following format:
        ```json
        {{
        "answer": one of the options from A) very similar B) similar but general than HypoA C) different
        "explanation": a short text explanation about the relationship comparison
        }}```
        Answer:"""


def dimension_prompt(datasets_json, query, gold_hypo, gold_workflow, gen_hypo, gen_workflow, dimension_question) -> str:
    return f"""\
        You are going to compare two natural-language hypotheses HypoA and HypoB accompanied with optional workflows: WorkflowA for HypoA and WorkflowB for HypoB. \
        Both the hypotheses answer the natural language query "QUERY" over the dataset(s) described by dataset description(s) and column description(s) below. \
        Compare HypoA and HypoB in terms of three aspects: Contexts, Variables, and Relations. \
        E.g., for the hypothesis "From 1995 to 2009, the number of sandhill cranes around the tundra (Indigilka River) surged by an astounding ~10X":
        * Contexts refer to stratification of the data under which the given hypothesis is True. E.g., "For all women", "From 1995 to 2009".
        * Variables refer to the set of variables (either dependent or independent) that are mentioned in the hypothesis. E.g., number of sandhill cranes, location.
        * Relations refer to the form of relation between the variables. E.g., "surged by ~10x".

        Answer following questions for a given pair of hypotheses, HypoA and HypoB, along with an explanation grounded on the QUERY and the DATASET(S).

        Here is the metadata for the task:
        ```json
        {{
        "datasets": {datasets_json},
        "query": {query},
        "HypoA": {gold_hypo},
        "WorkflowA": {gold_workflow},
        "HypoB": {gen_hypo},
        "WorkflowB": {gen_workflow}
        }}
        ```

        {dimension_question}"""


def parse_json(text: str | None):
    """Like utils/openai_helpers.get_response's parsing, but tolerant of code fences and surrounding prose."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    for candidate in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        try:
            out = json.loads(candidate)
            return out if isinstance(out, dict) else None
        except (ValueError, TypeError):
            continue
    return None


def var_score(answer: str | None) -> dict:
    """eval/new_eval.py:get_score_from_answer('var'); failures score 0 (official: -1)."""
    j = parse_json(answer)
    try:
        inter, a, b = float(j["intersection"]), float(j["sizeA"]), float(j["sizeB"])
    except (TypeError, KeyError, ValueError):
        return {"p": 0.0, "r": 0.0, "f1": 0.0, "parse_error": True}
    p = inter / b if b else 0.0
    r = inter / a if a else 0.0
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.0
    return {"p": p, "r": r, "f1": f1, "sizeA": a, "sizeB": b, "intersection": inter, "explanation": j.get("explanation")}


def rel_score(answer: str | None) -> float:
    """eval/new_eval.py:get_score_from_answer('rel'); failures score 0 (official: -1 or an exception)."""
    j = parse_json(answer)
    s = str((j or {}).get("answer", "")).strip()
    if s.startswith("A") or "very similar" in s:
        return 1.0
    if s.startswith("B") or "similar but general than HypoA" in s:
        return 0.5
    return 0.0


def stub_score(gold: str, answer: str) -> float:
    """OFFLINE STUB used only when no judge is attached (tests, mock runs): exact match after normalizing."""
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    return float(norm(gold) == norm(answer))


class DiscoveryBenchWorld(TaskWorld):
    name = "discoverybench"
    unit = "task"
    TOOLS = [RUN_CODE]

    def __init__(self, n_agents: int, tasks: list[Task], seed: int = 0, home: int = 3, eval_at: tuple[int, ...] = (),
                 single: bool = False, judge: str = "openai/gpt-4o", timeout: float = 60, python: str | None = None,
                 judge_concurrency: int = 16, judge_temperature: float = 1.0):
        super().__init__(n_agents, tasks, seed=seed, home=home, eval_at=eval_at, single=single)
        self.judge = judge
        self.timeout = timeout
        self.python = python or (str(BENCH_PYTHON) if BENCH_PYTHON.exists() else sys.executable)
        self.judge_concurrency = judge_concurrency
        self.judge_temperature = judge_temperature  # official: 1.0 for segmentation and context matching (noisy)
        self._gold_sub: dict[str, list[dict]] = {}
        self._llm_extras = True  # pass temperature / JSON mode while the completion helper accepts them
        self.grading: dict[str, dict] = {}  # task -> the judge's record for its graded answer

    def mission(self) -> str:
        return (f"This is DiscoveryBench: {len(self.tasks)} data-driven discovery tasks over real datasets from "
                f"{self.n_super} domains. Each task gives one or more datasets with descriptions and a query; the answer "
                "is a scientific hypothesis in natural language that the data supports, stating its context, variables "
                "and relationship (with statistical significance where relevant), plus a summary of the workflow that "
                "produced it.\n- run_code runs Python in your own working directory for a task; the task's dataset files "
                "are there.\n- Each answer is graded at the end by an LLM judge against a hidden gold hypothesis, on "
                "whether its contexts, variables and relationships match. A task without an answer scores 0; the "
                "society's score is the average over all tasks.")

    def observe_extra(self, k, aid: str) -> str:
        return f"Code runs so far: {self.used[aid]}." if self.used[aid] else ""

    # ------------------------------------------------------------------ tools

    def _stage(self, wd: Path, t: Task):
        wd.mkdir(parents=True, exist_ok=True)
        for src in t.meta["files"]:
            dst = wd / Path(src).name
            if not dst.exists():  # copies, so agents can't alter the benchmark's own files
                shutil.copyfile(src, dst)

    async def _run_code(self, k, aid, a):
        t = self.task(a.get("task"))
        code = str(a.get("code") or "")
        if not code.strip():
            raise ActionError("the code is empty")
        wd = Path(self.workdir) / t.id / aid
        self._stage(wd, t)
        self.used[aid] += 1
        self.note_focus(k, aid, t, 1.0)
        k.tick_counts["experiments"] += 1
        ok, out = await run_python(code, wd, timeout=self.timeout, python=self.python)
        return "lab.run_code", out if ok else f"(error)\n{out}", t.id

    def mock_actions(self, k, aid, t, rng):
        first = Path(self.tasks[t].meta["files"][0]).name
        code = f"import pandas as pd\ndf = pd.read_csv({first!r})\nprint(df.shape)\nprint(df.head(3).to_string()[:600])"
        return [("open_task", {"task": t}), ("run_code", {"task": t, "code": code}),
                ("submit", {"task": t, "answer": f"Placeholder hypothesis for {t} from {aid}.", "rationale": "mock"})]

    # ------------------------------------------------------------------ grading (hidden from agents)

    async def _ask(self, system: str, prompt: str, max_tokens: int, temperature: float, json_mode: bool) -> str:
        if self._llm_extras:
            try:
                return await self.llm(system, prompt, model=self.judge, max_tokens=max_tokens,
                                      temperature=temperature, json_mode=json_mode)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                self._llm_extras = False
        return await self.llm(system, prompt, model=self.judge, max_tokens=max_tokens)

    async def _sub_hypotheses(self, t: Task, hypo: str, workflow: str) -> list[dict]:
        """eval/new_eval.py:get_sub_hypotheses (get_response: temperature 1, one try)."""
        out = parse_json(await self._ask(SYSTEM_PLAIN, extraction_prompt(t.meta["datasets_json"], hypo, workflow),
                                         2048, self.judge_temperature, False))
        subs = (out or {}).get("sub_hypo") or []
        subs = [s for s in subs if isinstance(s, dict)]
        return subs or [{"text": hypo, "context": "None", "variables": [], "relations": "", "explanation": "unable to segment"}]

    async def _context_match(self, gold: dict, gen: dict) -> bool:
        """eval/new_eval.py:is_matching_context."""
        gc, pc = str(gold.get("context", "")), str(gen.get("context", ""))
        if gc == pc:
            return True
        if "None" in (gc, pc):
            return False
        out = parse_json(await self._ask(SYSTEM_PLAIN, context_prompt(gold.get("text", ""), gc, gen.get("text", ""), pc),
                                         256, self.judge_temperature, False))
        m = (out or {}).get("match", False)
        return m is True or str(m).strip().lower() == "true"

    async def _dimension(self, t: Task, gold: str, gen: str, gen_workflow: str, question: str) -> str:
        prompt = dimension_prompt(t.meta["datasets_json"], t.meta["query"], gold, t.meta["gold_workflow"], gen,
                                  gen_workflow, question)
        return await self._ask(SYSTEM_DIMENSION, prompt, 512, 0.0, True)

    async def hypothesis_match(self, t: Task, gen_hypo: str, gen_workflow: str) -> dict:
        """eval/new_eval.py:run_eval_gold_vs_gen_NL_hypo_workflow; returns the record with final_score."""
        gold = t.meta["gold"]
        if t.id not in self._gold_sub:
            self._gold_sub[t.id] = await self._sub_hypotheses(t, gold, t.meta["gold_workflow"])
        gold_sub = self._gold_sub[t.id]
        gen_sub = await self._sub_hypotheses(t, gen_hypo, gen_workflow)
        covered, gen_to_gold = [], {}
        for p_id, gsub in enumerate(gen_sub):  # greedy, in order, one gold sub-hypothesis per prediction (as official)
            gen_to_gold[p_id] = -1
            for g_id, osub in enumerate(gold_sub):
                if g_id in covered:
                    continue
                if await self._context_match(osub, gsub):
                    gen_to_gold[p_id] = g_id
                    covered.append(g_id)
                    break
        matched = [p for p, g in gen_to_gold.items() if g >= 0]

        async def pair(p_id):  # the official code compares the FULL hypotheses for every matched pair
            var_ans, rel_ans = await asyncio.gather(self._dimension(t, gold, gen_hypo, gen_workflow, VAR_QUESTION),
                                                    self._dimension(t, gold, gen_hypo, gen_workflow, REL_QUESTION))
            v, r = var_score(var_ans), rel_score(rel_ans)
            return {"pair": f"P{p_id}||G{gen_to_gold[p_id]}", "var": v, "rel": r, "accuracy_score": 1.0 * v["f1"] * r}

        pairs = await asyncio.gather(*(pair(p) for p in matched))
        recall_context = len(covered) / len(gold_sub) if gold_sub else 0.0
        mean_accuracy = sum(x["accuracy_score"] for x in pairs) / len(gen_to_gold) if gen_to_gold else 0.0
        return {"gold_sub_hypo": gold_sub, "gen_sub_hypo": gen_sub, "gen_subh_to_gold_subh": gen_to_gold,
                "pairs": pairs, "recall_context": recall_context, "mean_accuracy_score": mean_accuracy,
                "final_score": recall_context * mean_accuracy, "judge": self.judge}

    async def grade(self, k, tasks: list[str]) -> dict[str, float]:
        if self.llm is None:
            out = {}
            for tid in tasks:
                s = stub_score(self.tasks[tid].meta["gold"], self.submissions[tid].answer)
                self.grading[tid] = {"final_score": s, "stub": True}
                out[tid] = s
            return out
        sem = asyncio.Semaphore(self.judge_concurrency)

        async def one(tid):
            sub = self.submissions[tid]
            async with sem:
                try:
                    rec = await self.hypothesis_match(self.tasks[tid], sub.answer, sub.rationale)
                except Exception as exc:  # not cached: retried at the next evaluation; counts 0 until then
                    self.grading[tid] = {"error": f"{exc.__class__.__name__}: {clip(exc, 200)}"}
                    return tid, None
            rec["answer_tick"] = sub.tick
            self.grading[tid] = rec
            return tid, rec["final_score"]

        results = await asyncio.gather(*(one(t) for t in tasks))
        return {t: s for t, s in results if s is not None}

    def export(self) -> dict:
        out = super().export()
        out["judge"] = self.judge
        out["grading"] = self.grading
        for t in out["tasks"]:
            t["gold"] = self.tasks[t["id"]].meta["gold"]  # for analysis after the run; never shown to agents
        return out


def make(cfg: dict, n_agents: int, seed: int = 0, single: bool = False) -> DiscoveryBenchWorld:
    opts = cfg.get("world_opts") or {}
    tasks = load_tasks(Path(opts.get("data", DATA)), domain_knowledge=_flag(opts.get("domain_knowledge")),
                       workflow_tags=_flag(opts.get("workflow_tags")))
    if cfg.get("tasks"):
        tasks = stratified_subset(tasks, int(cfg["tasks"]), seed)
    return DiscoveryBenchWorld(n_agents, tasks, seed=seed, home=int(cfg["home"]) if cfg.get("home") is not None else 3,
                               eval_at=tuple(cfg.get("eval_at") or ()), single=single,
                               judge=opts.get("judge", "openai/gpt-4o"), timeout=float(opts.get("timeout", 60)),
                               python=opts.get("python"), judge_concurrency=int(opts.get("judge_concurrency", 16)),
                               judge_temperature=float(opts.get("judge_temperature", 1.0)))
