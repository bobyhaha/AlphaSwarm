"""DiscoveryBench world: tasks, no gold leakage, code execution, the HMS judge (with a scripted judge) and a mock run."""
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.bench.discoverybench import (dimension_prompt, extraction_prompt, load_tasks, make, rel_score,  # noqa: E402
                                         var_score)
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.metrics import measure  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import run  # noqa: E402

CFG = {"world_opts": {}, "home": 3, "eval_at": []}


def world(n_agents=6, tasks=None, workdir="/tmp/alphateam_db_test", **opts):
    w = make(dict(CFG, tasks=tasks, world_opts=opts), n_agents, seed=0)
    w.workdir = workdir
    return w


def test_loads_real_test_split():
    tasks = load_tasks()
    assert len(tasks) == 239
    fams = {t.family for t in tasks}
    doms = {t.domain for t in tasks}
    assert len(fams) == 10 and {"humanities", "sociology", "meta science", "biology", "engineering", "economics"} == doms
    w = world()
    assert len(w.tasks) == 239 and len(w.modules) == 10 and w.n_super == 6
    assert all(m["super"] < w.n_super for m in w.modules)


def test_no_gold_leakage():
    w = world(n_agents=4)
    k = Kernel(4, max_actions=4)
    k.env = w
    w.setup(k)
    golds = [t.meta["gold"] for t in w.tasks.values()]
    seen = [w.statement(k, "A1", t) for t in w.tasks.values()]
    seen += [w.mission(), w.brief(), w.observe(k, "A1"), "\n".join(w.search(k, "A1", "tasks", []))]
    seen += [e.content for b in k.boards.values() for e in b.entries]
    for t in list(w.tasks.values())[:40]:
        seen.append(asyncio.run(k.act("A1", "open_task", {"task": t.id}))[1])
    blob = "\n".join(seen)
    leaks = [g for g in golds if len(g) > 25 and g in blob]
    assert not leaks, leaks[:3]
    assert "hypotheses" not in blob and "intermediate" not in blob.lower()
    # domain knowledge is off by default and can be turned on
    t0 = next(t for t in w.tasks.values() if t.family == "archaeology")
    assert "Domain Knowledge" not in t0.statement
    w2 = world(domain_knowledge="1")
    assert any("Domain Knowledge" in t.statement for t in w2.tasks.values())


def test_run_code_reads_dataset(tmp=Path("/tmp/alphateam_db_code")):
    shutil.rmtree(tmp, ignore_errors=True)
    w = world(n_agents=2, workdir=str(tmp))
    k = Kernel(2, max_actions=4)
    k.env = w
    t = next(t for t in w.tasks.values() if t.family == "archaeology")
    name = Path(t.meta["files"][0]).name
    ok, msg = asyncio.run(k.act("A1", "run_code", {"task": t.id, "code": f"import pandas as pd\nprint(pd.read_csv({name!r}).shape)"}))
    assert ok and msg.strip().startswith("("), msg
    # state persists in the workspace; an error is reported, not raised
    asyncio.run(k.act("A1", "run_code", {"task": t.id, "code": "open('note.txt','w').write('hi')"}))
    ok, msg = asyncio.run(k.act("A1", "run_code", {"task": t.id, "code": "print(open('note.txt').read())"}))
    assert ok and "hi" in msg
    ok, msg = asyncio.run(k.act("A1", "run_code", {"task": t.id, "code": "1/0"}))
    assert ok and "ZeroDivisionError" in msg
    assert w.used["A1"] == 4 and (tmp / t.id / "A1" / name).exists()
    # the benchmark's own file is a copy, not a link
    assert not (tmp / t.id / "A1" / name).is_symlink()


def test_stub_grading_offline():
    w = world(n_agents=2)
    k = Kernel(2, max_actions=4)
    k.env = w
    t1, t2 = list(w.tasks)[:2]
    asyncio.run(k.act("A1", "submit", {"task": t1, "answer": w.tasks[t1].meta["gold"]}))
    asyncio.run(k.act("A2", "submit", {"task": t2, "answer": "something else"}))
    asyncio.run(w.finalize(k))
    m = measure(k)
    assert abs(m["score"] - 1 / 239) < 1e-3 and w.grading[t1]["stub"]


def test_judge_protocol_with_scripted_llm():
    """The HMS pipeline with a scripted judge: 2 gold sub-hypotheses, 1 predicted that matches the first by context."""
    w = world(n_agents=2)
    k = Kernel(2, max_actions=4)
    k.env = w
    tid = list(w.tasks)[0]
    calls = []

    async def judge(system, prompt, model=None, max_tokens=2000, temperature=None, json_mode=False):
        calls.append((model, temperature, json_mode, prompt))
        if "extract three dimensions" in prompt:
            if '"hypothesis": "' + w.tasks[tid].meta["gold"] in prompt:
                return json.dumps({"sub_hypo": [{"text": "g1", "context": "women", "variables": ["a"], "relations": "up"},
                                                {"text": "g2", "context": "men", "variables": ["b"], "relations": "down"}]})
            return "```json\n" + json.dumps({"sub_hypo": [{"text": "p1", "context": "female", "variables": ["a"], "relations": "up"}]}) + "\n```"
        if '"match": true or false' in prompt:
            return json.dumps({"match": '"gold_context": "women"' in prompt})
        if "what are the different variables" in prompt:
            return json.dumps({"sizeA": 2, "sizeB": 1, "intersection": 1, "explanation": "x"})
        if "exhibit the same relation" in prompt:
            return json.dumps({"answer": "B) similar but general than HypoA", "explanation": "x"})
        raise AssertionError(prompt[:200])

    w.llm = judge
    asyncio.run(k.act("A1", "submit", {"task": tid, "answer": "a pred", "rationale": "wf"}))
    asyncio.run(w.finalize(k))
    rec = w.grading[tid]
    # recall_context = 1/2; var f1: p=1, r=0.5 -> 2/3; rel 0.5; mean accuracy over 1 prediction = 1/3
    assert abs(rec["recall_context"] - 0.5) < 1e-9 and abs(rec["final_score"] - 0.5 * (2 / 3) * 0.5) < 1e-9, rec
    assert all(c[0] == "openai/gpt-4o" for c in calls)
    dims = [c for c in calls if "HypoA" in c[3]]
    assert dims and all(c[1] == 0.0 and c[2] for c in dims)
    # gold sub-hypotheses are cached: regrading a changed answer extracts only the prediction
    n = len(calls)
    asyncio.run(k.act("A1", "submit", {"task": tid, "answer": "another pred"}))
    asyncio.run(w.evaluate(k))
    extractions = [c for c in calls[n:] if "extract three dimensions" in c[3]]
    assert len(extractions) == 1
    # an unchanged answer is not regraded
    n = len(calls)
    asyncio.run(w.evaluate(k))
    assert len(calls) == n


def test_judge_without_extra_kwargs_and_failures():
    """A completion helper without temperature/json_mode still works; judge errors leave the task ungraded (score 0)."""
    w = world(n_agents=2)
    k = Kernel(2, max_actions=4)
    k.env = w
    t1, t2 = list(w.tasks)[:2]

    async def judge(system, prompt, model=None, max_tokens=2000):
        if '"hypothesis": "y"' in prompt:  # the extraction of t2's predicted hypothesis fails
            raise TimeoutError("judge down")
        if "extract three dimensions" in prompt:
            return json.dumps({"sub_hypo": [{"text": "h", "context": "all", "variables": ["a"], "relations": "r"}]})
        if "what are the different variables" in prompt:
            return "not json"
        return json.dumps({"answer": "A) very similar"})

    w.llm = judge
    asyncio.run(k.act("A1", "submit", {"task": t1, "answer": "x"}))
    asyncio.run(k.act("A1", "submit", {"task": t2, "answer": "y"}))
    asyncio.run(w.finalize(k))
    assert w.grading[t1]["final_score"] == 0.0 and w.grading[t1]["pairs"][0]["var"]["parse_error"]
    assert "error" in w.grading[t2] and t2 not in w.task_scores and not w._llm_extras


def test_score_parsers():
    assert var_score('{"sizeA": 2, "sizeB": 2, "intersection": 2}')["f1"] == 1.0
    assert var_score("garbage")["f1"] == 0.0
    assert rel_score('{"answer": "A) very similar"}') == 1.0 and rel_score('{"answer": "C) different"}') == 0.0
    assert rel_score("{bad") == 0.0
    # the prompts render as the official f-strings do
    p = extraction_prompt([{"dataset_description": "d", "columns": []}], "H", "W")
    assert '"hypothesis": "H"' in p and "{{" not in p
    q = dimension_prompt([], "Q", "G", "", "P", "", "Question?")
    assert '"query": Q' in q and '"HypoB": P' in q


def test_stratified_subset():
    w = world(tasks=30)
    assert len(w.tasks) == 30 and len({t.family for t in w.tasks.values()}) == 10


def test_mock_run(tmp=Path("/tmp/alphateam_db_run")):
    shutil.rmtree(tmp, ignore_errors=True)
    k = Kernel(10, max_actions=4)
    k.env = make(dict(CFG, tasks=12), 10, seed=1)
    k.env.setup(k)
    data = asyncio.run(run(k, MockBackend(0), 2, tmp, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"]
    assert m["score"] == 0.0 and m["coverage"] > 0 and m["groups"] > 0
    report = (tmp / "report.md").read_text()
    assert "benchmark discoverybench" in report and "## Answers" in report
    assert any(p.name == "A1" or p.is_dir() for p in (tmp / "work").iterdir())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
