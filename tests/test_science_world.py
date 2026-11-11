"""The science world on built-in questions: blind attempts, trust, the sheet, audits, credit, the review. No API."""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphateam.kernel import PRIMITIVES, Kernel  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.primitives import system_prompt, tool_schemas  # noqa: E402
from alphateam.reflection import ReflectionLoop  # noqa: E402
from alphateam.runner import build, run  # noqa: E402
from alphateam.science_world import DATA, ScienceWorld, load_questions, plain_answer, verdict  # noqa: E402

OPTS = {"fake": "1", "sync": "1", "board": "1", "private": "1", "teaming": "1", "messages": "1", "audit": "0.5"}


def world(n=4, **opts):
    k = Kernel(n, max_actions=30, attention=6000, allowed={p: None for p in PRIMITIVES})
    k.sync, k.full_board, k.private_boards, k.teaming, k.messages_on = True, True, True, True, True
    k.total_ticks = 12
    env = ScienceWorld(n, opts={**OPTS, **opts})
    k.env = env
    env.isolated = False
    env.local = False
    env.setup(k)
    env.workdir = tempfile.mkdtemp(prefix="alphateam-science-test-")
    k.tick = 1
    env.on_tick_start(k)
    for aid in k.agents:
        k.begin_turn(aid)
    return k, env


def act(k, aid, tool, /, **args):
    return asyncio.run(k.act(aid, tool, args))


def tick_end(k, env):
    asyncio.run(env.on_tick_end(k))
    k.tick += 1


def task(env, gold=None, track="olympiad", audited=None):
    for t, q in env.questions.items():
        if q["track"] == track and (gold is None or q["answer"] == gold) and (audited is None or (t in env.audit_set) == audited):
            return t
    raise AssertionError("no such question")


def post(k, aid, tid, final, **more):
    ok, msg = act(k, aid, "answer", task=tid, final=final, working=more.pop("reasoning", "because"), **more)
    assert ok, msg
    return re.search(r"E\d+", msg).group()


def test_real_questions_load_stratified_and_the_same_for_every_arm():
    if not DATA.exists():
        return
    one, two = load_questions(40, 20, 0), load_questions(40, 20, 0)
    assert [q["problem"] for q in one] == [q["problem"] for q in two] and len(one) == 60
    assert sum(q["track"] == "research" for q in one) == 20
    assert {q["subject"] for q in one} == {"physics", "chemistry", "biology"}
    assert [q["problem"] for q in load_questions(40, 20, 1)] != [q["problem"] for q in one]


def test_judge_verdicts_are_read_and_no_verdict_is_no_signal():
    assert verdict("reasoning...\nVERDICT: CORRECT", False) == 1.0
    assert verdict("VERDICT: INCORRECT", False) == 0.0
    assert verdict("x\nVERDICT: 7.5", True) == 0.75 and verdict("VERDICT: 12", True) == 1.0
    assert verdict("I cannot decide", False) is None and verdict("VERDICT: maybe", False) is None
    assert plain_answer("\\( 4 \\)") == plain_answer("4")


def test_tools_and_brief_follow_goal():
    k, env = world()
    names = {t["name"] for t in tool_schemas(k, "A1")}
    assert {"open_task", "answer", "calc", "head", "trace", "role", "plan", "guide", "lead", "delegate", "pay", "substrate"} <= names
    assert "focus" not in names and set(env.final_tools) <= names
    text = system_prompt(k)
    for line in ("large jigsaw puzzle", "A placed piece is never swept off the table", "Nothing is lost", "never rank or votes",
                 "Whoever contributes later reconciles", "Independent work stays independent", "unknown to you", "never for activity",
                 "Which actions and combinations", "A review never says who does what",
                 "a result should carry its understanding"):
        assert line in text, line
    assert "SOME WAYS TO ORGANIZE" not in text and "CEO" not in text
    seen = env.observe(k, "A1")
    assert "THE SHEET" in seen and "WHAT HAS PAID" not in seen and "ACTIONS NOBODY HAS USED YET" in seen and "xsv" not in seen


def test_attempts_stay_hidden_until_you_post_and_a_peek_costs_independence():
    k, env = world()
    tid = task(env, "4")
    post(k, "A1", tid, "4", reasoning="two and two")
    ok, msg = act(k, "A2", "open_task", task=tid)
    assert ok and "hidden" in msg and "two and two" not in msg
    entry = [e for e in k.boards["global"].entries if "ATTEMPT" in e.content][-1]
    assert "two and two" not in entry.content and "4" not in entry.content.replace(tid, "")  # the board does not leak answers
    ok, msg = act(k, "A3", "open_task", task=tid, peek=True)
    assert "two and two" in msg
    a3 = post(k, "A3", tid, "4")
    a2 = post(k, "A2", tid, "4")
    assert env.by_entry[a2]["blind"] and not env.by_entry[a3]["blind"]
    ok, msg = act(k, "A2", "open_task", task=tid)
    assert "two and two" in msg


def test_independent_agreement_makes_an_answer_reproduced_and_it_takes_the_sheet():
    k, env = world()
    tid = task(env, "4")
    first = post(k, "A1", tid, "4")
    later = post(k, "A2", tid, "5")
    third = post(k, "A3", tid, "4")
    tick_end(k, env)
    assert env.level(env.by_entry[first]) == "reproduced" and env.level(env.by_entry[third]) == "reproduced"
    assert env.level(env.by_entry[later]) == "said"
    assert env.sheet_attempt(tid)["id"] in (first, third)
    assert env.disputed(tid) == [["A1", "A3"], ["A2"]]
    assert "differ" in list(k.agents["A2"].inbox)[-1]["text"]
    assert "Disputed, for the parties to settle directly" in env.observe(k, "A4")


def test_copying_is_not_reproduction():
    k, env = world()
    tid = task(env, "4")
    post(k, "A1", tid, "4")
    act(k, "A2", "open_task", task=tid, peek=True)
    second = post(k, "A2", tid, "4")
    tick_end(k, env)
    assert env.level(env.by_entry[second]) != "reproduced" and env.sheet_attempt(tid)["agent"] in ("A1", "A2")
    assert not env.independent_agreement(env.by_entry[second]) - {"A1"}
    assert env.level(env.attempts[tid][0]) == "said"  # nobody reached A1's answer blind


def test_calc_receipts_make_an_attempt_seen():
    k, env = world()
    tid = task(env, "9")
    ok, msg = act(k, "A1", "calc", code="print(3*3)")
    assert ok and "9" in msg, msg
    rid = re.search(r"R\d+", msg).group()
    a = post(k, "A1", tid, "9", evidence=[rid])
    assert env.level(env.by_entry[a]) == "seen"
    ok, msg = act(k, "A2", "answer", task=tid, final="9", evidence=[rid])
    assert not ok and "your own" in msg


def test_a_concrete_error_outweighs_agreement_until_revised():
    k, env = world()
    tid = task(env, "4")
    a = post(k, "A1", tid, "5")
    b = post(k, "A2", tid, "5")
    tick_end(k, env)
    assert env.level(env.by_entry[a]) == "reproduced"
    c = post(k, "A3", tid, "4")
    ok, msg = act(k, "A3", "answer", task=tid, final="4", working="they dropped a sign", contests=a)
    assert ok, msg
    d = re.search(r"E\d+", msg).group()
    assert env.by_entry[a]["contested"] and env.by_entry[c]["superseded_by"] == d
    assert "contests your answer" in list(k.agents["A1"].inbox)[-1]["text"]
    assert env.sheet_attempt(tid)["agent"] in ("A2", "A3")
    fix = post(k, "A1", tid, "4", reasoning="revised")
    assert env.by_entry[a]["superseded_by"] == fix and not env.by_entry[a]["contested"] and b


def test_a_later_attempt_that_builds_on_others_replaces_them_and_keeps_them_visible():
    k, env = world()
    tid = task(env, track="research")
    a = post(k, "A1", tid, "scattering")
    ok, msg = act(k, "A2", "answer", task=tid, final="x", builds_on=[a])
    assert not ok and "seen" in msg
    b = post(k, "A2", tid, "mie")
    c = post(k, "A2", tid, "rayleigh scattering, with the part A1 had", builds_on=[a, b])
    assert env.sheet_attempt(tid)["id"] == c
    assert env.by_entry[a]["superseded_by"] == c and len(env.attempts[tid]) == 3
    ok, msg = act(k, "A3", "open_task", task=tid, peek=True)
    assert "scattering" in msg and "mie" in msg


def test_audit_pays_only_verified_gain_and_only_on_the_sample():
    k, env = world()
    tid, other = task(env, audited=True), task(env, audited=False)
    gold = env.questions[tid]["answer"]
    post(k, "A1", tid, "wrong")
    post(k, "A2", other, env.questions[other]["answer"])
    tick_end(k, env)
    assert sum(env.credit.values()) == 0
    assert any("AUDIT" in e.content and "incorrect" in e.content for e in k.boards["global"].entries)
    assert not any(other in e.content and "AUDIT" in e.content for e in k.boards["global"].entries)
    post(k, "A3", tid, gold)
    tick_end(k, env)
    assert env.audits[-1]["task"] == tid
    assert env.progress()["passed"] == (1.0 if env.sheet_attempt(tid)["final"] == gold else 0)


def test_shares_never_mint_and_independent_agents_share_the_credit():
    k, env = world()
    tid = task(env, audited=True)
    gold = env.questions[tid]["answer"]
    post(k, "A1", tid, gold)
    post(k, "A2", tid, gold)
    tick_end(k, env)
    assert abs(sum(env.credit.values()) - 1.0) < 1e-6 and env.credit["A1"] > 0 and env.credit["A2"] > 0


def test_goals_share_wealth_and_keep_the_attribution():
    k, env = world(goal="society")
    tid = task(env, audited=True)
    post(k, "A1", tid, env.questions[tid]["answer"])
    tick_end(k, env)
    assert len({round(env.credit[a], 6) for a in k.agents}) == 1 and env.attributed["A1"] == 1.0
    assert "What each contributed" in env.observe(k, "A2")


def test_activity_earns_nothing():
    k, env = world()
    tid = task(env, audited=False)
    post(k, "A1", tid, env.questions[tid]["answer"])
    act(k, "A1", "plan", op="claim", hypothesis="method")
    tick_end(k, env)
    assert sum(env.credit.values()) == 0


def test_review_reads_audits_and_changes_rules_only():
    class Org(MockBackend):
        async def complete(self, k, system, prompt, **kw):
            if kw.get("who") != "organizer":
                return "we did not check each other"
            return json.dumps({"law": {"tester_share": 0.4}, "practices": "A1 checks T3; say what you ruled out.", "teams": [{"members": ["A1"]}],
                               "suggestions": ["More independent attempts."], "why": "w", "expect": "e"})

    k, env = world()
    tid = task(env, audited=True)
    post(k, "A1", tid, env.questions[tid]["answer"])
    tick_end(k, env)
    k.tick = 3
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    asyncio.run(loop.on_tick_end(k, Org(0)))
    assert env.law["tester_share"] == 0.4 and "A1" not in env.norms and "T3" not in env.norms and "say what you ruled out" in env.norms
    assert not env.teams and loop.history[-1]["held_out"] == 1.0
    assert "REVIEW at t3" in k.boards["global"].entries[-1].content


def test_window_facts_never_touch_the_scored_questions_truth():
    k, env = world()
    scored = next(t for t, q in env.questions.items() if t not in env.audit_set and len(q["answer"]) > 1)
    post(k, "A1", scored, env.questions[scored]["answer"])
    tick_end(k, env)
    f = env.window_facts(k, 0)
    assert f["attempts_this_window"] == 1 and f["unanswered"] == len(env.questions) - 1
    assert all(u["task"] in env.audit_set for u in f["audits_this_window"])
    assert env.questions[scored]["answer"] not in json.dumps({key: f[key] for key in ("agents", "held_out", "audits_this_window")})


def test_rehearsal_through_the_runner():
    cfg = {"condition": "self_org", "agents": 4, "ticks": 5, "actions": 12, "attention": 6000, "world": "science", "calls": None,
           "world_opts": dict(OPTS), "reflect_window": 2, "reflect_floor": 0.5, "instruments": 3, "experiments": 30, "seed_runs": 3, "noise": .3,
           "validate_every": 3}
    k = build(cfg, 1, "science")
    out = Path(tempfile.mkdtemp(prefix="alphateam-science-run-"))
    data = asyncio.run(run(k, MockBackend(1), 5, out, {"condition": "self_org", "backend": "mock"}, 1, quiet=True))
    w = data["world"]
    assert w["kind"] == "science" and w["final"]["n"] > 0 and data["meta"]["reflection"] and w["attempts"]
    assert all(a.get("judged") is not None for v in w["attempts"].values() for a in v)
    assert "FrontierScience" in (out / "report.md").read_text()


def test_single_agent_has_no_society_tools():
    env = ScienceWorld(1, single=True, world_agents=10, opts=dict(OPTS))
    names = {t["name"] for t in env.TOOLS}
    assert not names & {"pay", "lead"} and env.steps_per_tick == 120


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
