"""The ExplorationBench world (public demo worlds) with the official engines; offline (stub answerer, mock agents)."""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.bench.explorationbench import CODE, LOGIC, TOOLS, make, official_setup, run_worker_sync  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import run  # noqa: E402


def world(n=4, ticks=8, **opts):
    k = Kernel(n, max_actions=4)
    k.env = make({"ticks": ticks, "world_opts": opts}, n, seed=0)
    k.env.setup(k)
    k.tick = 1
    return k


def act(k, aid, name, **args):
    return asyncio.run(k.act(aid, name, args))


def heldout_lines() -> list[str]:
    """Distinctive lines of every held-out prompt (programs, function specs, premises and goals), except lines the
    official materials already show (the demo world's refused example R01 is U01's naive proof, by the authors' design)."""
    code, logic = official_setup()
    official = "\n".join(o[x] for o in (code, logic) for x in ("manual", "demo_block"))
    out = []
    for t in code["tests"] + logic["tests"]:
        for line in t["prompt"].splitlines():
            line = line.strip()
            if (("(" in line and len(line) >= 12 and not line.startswith(("【", "如果", "否则", "再附")))
                    or line.startswith("goal:")) and line not in official:
                out.append(line)
    return out


def test_worlds_load():
    k = world()
    env = k.env
    assert sorted(t.meta["world"] for t in env.tasks.values()) == [CODE, LOGIC]
    code, logic = official_setup()
    assert len(code["tests"]) == 5 and len(logic["tests"]) == 5 and sum(not t["provable"] for t in logic["tests"]) == 1
    assert code["limits"]["calls_per_round"] == 12 and code["limits"]["units_per_call"] == 5
    assert logic["limits"]["calls_per_round"] == 12 and logic["limits"]["total"] == 48
    ok, msg = act(k, "A1", "open_task", task=env.world_task[CODE])
    assert ok and "SHATTER" in msg and "EMIT(32)" in msg and "39" in msg  # manual + worked example with its real output
    ok, msg = act(k, "A1", "open_task", task=env.world_task[LOGIC])
    assert ok and "AND_E2" in msg and "verifier" in msg
    assert env.milestone_ticks == {2: 1, 4: 2, 6: 3} and env.round_of(1) == 1 and env.round_of(8) == 4


def test_probe_caps_and_feedback():
    k = world()
    ok, msg = act(k, "A1", "run_aliencode", code="EMIT(32)")
    assert ok and msg.startswith("39")
    ok, msg = act(k, "A1", "run_aliencode", code="EMIT(1, 2, 3, 4, 5, 6)")  # 6 observed values > 5: rejected in full
    assert ok and "[ToolError]" in msg and "11/12" not in msg and "10/12" in msg  # ...and still uses a call
    ok, msg = act(k, "A1", "run_aliencode", code="SWEEP i IN EXTENT(3):\n    EMIT(i, i)")  # 2 values x3 in a loop
    assert "[ToolError]" in msg
    ok, msg = act(k, "A1", "run_aliencode", code="EMIT(open)")
    assert ok and "AlienError" in msg  # hardened interpreter
    ok, msg = act(k, "A2", "check_proof", proof="PROOF\npremises: AND(p, q)\ngoal: q\n1. AND(p, q) [premise]\n2. q [AND_E2 1]\nEND",
                  to="global")
    assert ok and '"reason_id": "X01"' in msg and '"accepted": false' in msg
    assert any("AlienLogic proof by A2" in e.content for e in k.boards["global"].entries)
    ok, msg = act(k, "A1", "run_aliencode", code="")
    assert not ok and "empty" in msg


def test_pooled_budget_per_round():
    k = world(n=3, ticks=8)
    for i in range(12):
        ok, msg = act(k, f"A{i % 3 + 1}", "run_aliencode", code="EMIT(1)")
        assert ok, msg
    ok, msg = act(k, "A2", "run_aliencode", code="EMIT(1)")
    assert not ok and "used all 12 AlienCode calls of round 1" in msg and "tick 3" in msg
    assert act(k, "A2", "check_proof", proof="PROOF\npremises: p\ngoal: p\n1. p [premise]\nEND")[0]  # the other world
    k.tick = 3  # round 2 unlocks a fresh budget; unused calls never carry over
    assert act(k, "A2", "run_aliencode", code="EMIT(1)")[0]
    assert k.env.calls[(CODE, 1)] == 12 and k.env.calls[(CODE, 2)] == 1
    k2 = world(n=2, ticks=8, probe_budget="per_agent:2")
    for aid in ("A1", "A2"):
        assert act(k2, aid, "run_aliencode", code="EMIT(1)")[0] and act(k2, aid, "run_aliencode", code="EMIT(2)")[0]
    ok, msg = act(k2, "A1", "run_aliencode", code="EMIT(3)")
    assert not ok and "you have used all 2" in msg


def test_heldout_never_visible():
    k = world(n=4, ticks=8)
    data = asyncio.run(run(k, MockBackend(1), 8, Path("/tmp/alphateam_eb_leak"), {"condition": "self_org", "backend": "mock"},
                           quiet=True))
    visible = [k.env.mission(), k.env.brief(), str(TOOLS)]
    visible += [k.env.statement(k, "A1", t) for t in k.env.tasks.values()]
    visible += [k.observe(a, consume=False) for a in k.agents]
    visible += [e.content for b in k.boards.values() for e in b.entries]
    text = "\n".join(visible)
    lines = heldout_lines()
    assert len(lines) >= 8
    leaked = [line for line in lines if line in text]
    assert not leaked, leaked
    assert data["world"]["milestones"]  # grading did run on those held-out items


def _reference_llm():
    refs = {**run_worker_sync([{"op": "code_references"}])[0], **run_worker_sync([{"op": "logic_references"}])[0]}

    async def llm(system, prompt, max_tokens=0):
        tid = re.search(r"【(?:测试|Held-out) (\w+)】", prompt).group(1)
        return refs[tid]
    return llm


def test_grading_runs_the_real_checker():
    k = world(n=2, ticks=4)
    asyncio.run(k.env.evaluate(k, final=True))  # stub answerer: gradable but wrong except the unprovable theorem
    env = k.env
    m0, last = env.milestones[0], env.milestones[-1]
    assert m0["label"] == "M0" and last["label"] == "M1"
    assert last[CODE]["score"] == 0.0 and abs(last[LOGIC]["score"] - 0.2) < 1e-9  # U01 declined correctly
    assert env.score()["score"] == 0.1
    assert len(last[CODE]["samples"]) == 15  # 5 tasks x 3 trials
    k.env.llm = _reference_llm()  # the demo worlds' reference solutions grade 100%
    k.tick = 4
    asyncio.run(k.env.evaluate(k, final=True))
    best = k.env.milestones[-1]
    assert best[CODE]["score"] == 1.0 and best[LOGIC]["score"] == 1.0, (best[CODE]["samples"], best[LOGIC]["samples"])


def test_report_reaches_the_answerer():
    k = world(n=2, ticks=4)
    seen = []

    async def llm(system, prompt, max_tokens=0):
        seen.append((system, prompt))
        return "ANSWER: UNPROVABLE"
    k.env.llm = llm
    tid = k.env.world_task[LOGIC]
    assert act(k, "A1", "submit", task=tid, answer="X01: AND_E2 is refused; use AND_E1 only.")[0]
    asyncio.run(k.env.evaluate(k, final=True))
    logic_prompts = [p for s, p in seen if "AlienLogic" in s or "alienlogic" in s.lower()]
    assert any("AND_E2 is refused" in p for p in logic_prompts)
    assert all("PROOF" in s for s, p in seen if "Held-out" in p)  # the official system prompt with the manual
    assert any("没有提交规则报告" in p for s, p in seen if "【测试" in p)  # no AlienCode report yet


def test_mock_run_reports_milestones(tmp=Path("/tmp/alphateam_eb_run")):
    k = world(n=8, ticks=8)
    data = asyncio.run(run(k, MockBackend(0), 8, tmp, {"condition": "self_org", "backend": "mock"}, quiet=True))
    ms = data["world"]["milestones"]
    assert [m["label"] for m in ms] == ["M0", "M1", "M2", "M3", "M4"] and [m["t"] for m in ms] == [0, 2, 4, 6, 8]
    m = data["snapshots"][-1]["metrics"]
    assert m["score"] is not None and 0 <= m["score"] <= 1
    assert data["world"]["probes"] and data["world"]["calls"]
    assert all(v <= 12 for v in data["world"]["calls"].values())
    assert (tmp / "report.md").read_text().startswith("# AlphaTeam run")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
