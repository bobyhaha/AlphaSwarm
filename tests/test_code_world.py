"""The code world on its fake harness: the gate, the credit engine, trust, goals, the law and the review. No Docker, no API."""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphateam.code_world import LAW, CodeWorld, clean_law, law_text  # noqa: E402
from alphateam.kernel import PRIMITIVES, Kernel  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.primitives import system_prompt, tool_schemas  # noqa: E402
from alphateam.reflection import ReflectionLoop  # noqa: E402
from alphateam.runner import build, culture, inherit, run  # noqa: E402

OPTS = {"fake": "1", "sync": "1", "board": "1", "private": "1", "teaming": "1", "messages": "1"}


def world(n=4, **opts):
    k = Kernel(n, max_actions=30, attention=6000, allowed={p: None for p in PRIMITIVES})
    k.sync, k.full_board, k.private_boards, k.teaming, k.messages_on = True, True, True, True, True
    k.total_ticks = 12
    env = CodeWorld(n, opts={**OPTS, **opts})
    k.env = env
    env.isolated = False
    env.local = False
    env.setup(k)
    env.workdir = tempfile.mkdtemp(prefix="alphateam-test-")
    k.tick = 1
    env.on_tick_start(k)
    for aid in k.agents:
        k.begin_turn(aid)
    return k, env


def act(k, aid, tool, /, **args):
    return asyncio.run(k.act(aid, tool, args))


def land(k, env, aid, *names, note="change", cites=()):
    for name in names:
        ok, msg = act(k, aid, "shell", command=f"write {name} v1")
        assert ok, msg
    ok, msg = act(k, aid, "publish", note=note, cites=list(cites))
    assert ok, msg


def tick_end(k, env):
    env.on_tick_end(k)
    k.tick += 1
    env.on_tick_start(k)


def test_tools_are_the_worlds_and_no_game_words_remain():
    k, env = world()
    names = [t["name"] for t in env.TOOLS]
    for need in ("shell", "probe", "publish", "head", "trace", "role", "own", "toolbox", "pay", "plan", "guide", "lead", "substrate"):
        assert need in names, need
    assert "focus" not in names and "contract" not in names
    assert not [t for t in env.TOOLS if t["name"] in ("plan", "guide") and "game" in t["parameters"]["properties"]]
    assert "unsolved level" not in json.dumps([t for t in env.TOOLS if t["name"] == "plan"])
    text = env.observe(k, "A1")
    assert "OPEN PROBLEMS" not in text and "xsv level" not in text
    offered = {t["name"] for t in tool_schemas(k, "A1")}
    assert set(env.final_tools) <= offered, set(env.final_tools) - offered


def test_brief_states_goal_law_and_possibilities_without_recommending():
    k, env = world()
    text = system_prompt(k)
    assert "YOUR GOAL: maximize your own wealth" in text
    assert "SOME WAYS TO ORGANIZE" not in text and "CEO" not in text
    k2, env2 = world(ways="1")
    assert "nothing here is a recommendation, and nobody assigns any of it" in system_prompt(k2)
    for line in ("large jigsaw puzzle", "Progress is what is assembled and coherent at HEAD", "Nothing is lost when someone else's work lands",
                 "Evidence decides what is correct, never rank or votes", "Whoever contributes later reconciles", "one agent at a time",
                 "A result should carry its understanding", "Work that unblocks others is taken first", "which actions and combinations have paid",
                 "A working cycle that may pay", "A review never says who does what"):
        assert line in text, line
    assert "There are no private boards" not in text  # private=1 keeps them
    k2, env2 = world(goal="society")
    assert "shared by all agents equally" in system_prompt(k2)


def test_a_merge_pays_verified_value_to_the_agent_who_landed_it():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2")
    tick_end(k, env)
    m = env.merges[0]
    # two logged checks and two held-out checks: 0.3*2 + 1.0*2
    assert m["gains"] == 2 and m["verified"] == 2 and abs(m["credit"] - 2.6) < 1e-6
    assert abs(env.credit["A1"] - 2.6) < 1e-6
    assert env.held_log[-1]["passed"] == 2 and env.best_held == 2


def test_only_a_new_best_on_held_out_counts():
    k, env = world()
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    land(k, env, "A2", "plain_file")
    tick_end(k, env)
    assert env.merges[1]["verified"] == 0 and env.merges[1]["credit"] == 0
    assert env.credit["A2"] == 0


def test_activity_earns_nothing():
    k, env = world()
    ok, msg = act(k, "A1", "shell", command="write notes x")
    assert ok
    rid = re.search(r"R\d+", msg).group()
    ok, msg = act(k, "A1", "plan", op="claim", hypothesis="the join command needs a hash table")
    assert ok, msg
    ok, msg = act(k, "A1", "plan", op="done", finding="it does", outcome="works", evidence=[rid])
    assert ok, msg
    ok, msg = act(k, "A1", "guide", op="add", line="join is hashed", evidence=[rid])
    assert ok, msg
    assert "join is hashed" in env.observe(k, "A2")
    assert sum(env.credit.values()) == 0


def test_probe_author_shares_a_merge_that_passes_its_check():
    k, env = world()
    ok, msg = act(k, "A2", "probe", args=["count", "a.csv"])
    assert ok, msg
    oid = "ocount_a.csv"
    assert env.h.author_of(oid) == "A2"
    # the fake gate names gains after the file; this merge passes A2's check
    env.h._land = (lambda land: lambda label, files: {**land(label, files), "gains": [oid]})(env.h._land)
    land(k, env, "A1", "plain")
    tick_end(k, env)
    assert env.attributed["A2"] > 0
    assert env.attributed["A1"] > env.attributed["A2"]


def test_pieces_are_paid_by_trust_and_testers_share():
    k, env = world()
    ok, msg = act(k, "A2", "shell", command="write x y")
    rid = re.search(r"R\d+", msg).group()
    act(k, "A2", "plan", op="claim", hypothesis="csv quoting doubles quotes")
    ok, msg = act(k, "A2", "plan", op="done", finding="quotes are doubled", outcome="works", evidence=[rid])
    assert ok, msg
    eid = re.search(r"E\d+", msg).group()
    assert env.trust_of(k, eid)["level"] == "seen"
    ok, msg = act(k, "A3", "shell", command="write y z")
    mine = re.search(r"R\d+", msg).group()
    ok, msg = act(k, "A3", "plan", op="test", entry=eid, result="holds", evidence=[mine], finding="saw it too")
    assert ok, msg
    assert env.trust_of(k, eid)["level"] == "reproduced"
    assert "credit" not in msg  # a test pays through what is built on the piece, not when it is made
    land(k, env, "A1", "feature_1", cites=[eid])
    tick_end(k, env)
    assert env.attributed["A2"] > env.attributed["A3"] > 0
    paid = env.merges[0]["pieces"][eid]
    assert paid["level"] == "reproduced" and paid["author"] == "A2"
    # what a piece earned scales with trust
    v = env.merges[0]["credit"]
    assert abs(paid["paid"] - LAW["piece_share"] * v) < 1e-3


def test_a_contested_piece_earns_nothing():
    k, env = world()
    ok, msg = act(k, "A2", "shell", command="write x y")
    rid = re.search(r"R\d+", msg).group()
    act(k, "A2", "plan", op="claim", hypothesis="h")
    _, msg = act(k, "A2", "plan", op="done", finding="f", outcome="works", evidence=[rid])
    eid = re.search(r"E\d+", msg).group()
    _, msg = act(k, "A3", "shell", command="write z y")
    mine = re.search(r"R\d+", msg).group()
    ok, msg = act(k, "A3", "plan", op="challenge", entry=eid, finding="it fails on empty input", evidence=[mine])
    assert ok, msg
    land(k, env, "A1", "feature_1", cites=[eid])
    tick_end(k, env)
    assert env.merges[0]["pieces"][eid]["paid"] == 0
    assert env.attributed["A2"] == 0


def test_shares_never_mint():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2", "feature_3")
    tick_end(k, env)
    assert abs(sum(env.credit.values()) - env.merges[0]["credit"]) < 1e-6


def test_goals_decide_whose_wealth_a_merge_is():
    k, env = world(goal="society")
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    assert len({round(env.credit[a], 6) for a in k.agents}) == 1
    assert abs(env.attributed["A1"] - 1.3) < 1e-6  # the engine still knows who earned it

    k, env = world(goal="team")
    act(k, "A1", "lead", op="found", name="core")
    gid = next(iter(env.teams))
    act(k, "A2", "lead", op="join", team=gid)
    land(k, env, "A1", "feature_1")
    land(k, env, "A3", "feature_2")
    tick_end(k, env)
    assert round(env.credit["A1"], 6) == round(env.credit["A2"], 6) and env.credit["A1"] > 0
    assert env.credit["A3"] > 0 and env.credit["A4"] == 0  # an agent in no team keeps its own


def test_an_umbrella_group_changes_nobodys_income_unless_they_choose_it():
    k, env = world(goal="team")
    act(k, "A1", "lead", op="found", name="one")
    inner = next(iter(env.teams))
    act(k, "A2", "lead", op="join", team=inner)
    act(k, "A3", "connect", op="form_group", name="umbrella", purpose="p")
    outer = next(g.id for g in k.active_groups() if g.id != inner)
    act(k, "A1", "connect", op="join", target=outer)
    act(k, "A4", "connect", op="join", target=outer)
    assert env.team_of(k, "A1") == ["A1", "A2"]
    ok, msg = act(k, "A1", "role", title="bridge", team=outer)
    assert ok, msg
    assert env.team_of(k, "A1") == ["A1", "A3", "A4"]
    ok, msg = act(k, "A2", "role", title="x", team="G99")
    assert not ok


def test_with_private_boards_a_closing_note_stays_private():
    k, env = world()
    k.end_turn("A1", "my working note")
    assert not any("my working note" in e.content for e in k.boards["global"].entries)
    assert any("my working note" in e.content for e in k.boards["private:A1"].entries)
    assert "my working note" in k.recent_work("A1")


def test_lead_royalty_is_paid_out_of_the_members_credit():
    k, env = world()
    act(k, "A1", "lead", op="found", name="core")
    gid = next(iter(env.teams))
    act(k, "A2", "lead", op="join", team=gid)
    land(k, env, "A2", "feature_1")
    tick_end(k, env)
    assert env.credit["A1"] == 0  # a position pays nothing by default: fees are for agents to negotiate
    k, env = world(lead_share="0.2")
    act(k, "A1", "lead", op="found", name="core")
    gid = next(iter(env.teams))
    act(k, "A2", "lead", op="join", team=gid)
    land(k, env, "A2", "feature_1")
    tick_end(k, env)
    assert env.credit["A1"] > 0
    assert abs(env.credit["A1"] + env.credit["A2"] - env.merges[0]["credit"]) < 1e-6


def test_credit_buys_steps_and_goes_first_at_the_gate():
    k, env = world(steps="4")
    land(k, env, "A1", "feature_1", "feature_2", "feature_3", "feature_4")
    tick_end(k, env)
    assert env.steps_allowed("A1", k.tick) > env.steps_allowed("A2", k.tick)
    land(k, env, "A2", "plain")
    land(k, env, "A1", "plain2")
    order = sorted(env.queue, key=lambda a: (-env.credit[a], int(a[1:])))
    assert order[0] == "A1"
    assert env._gate_order()[0] == "A1"  # no code depends on either: credit breaks the tie


def test_stream_landing_takes_an_offer_at_once_and_catches_other_copies_up():
    k, env = world(stream="1")
    ok, _ = act(k, "A1", "shell", command="write feature_1 v1")
    assert ok
    ok, msg = act(k, "A1", "publish", note="now")
    assert ok and "merged feature_1" in msg, msg
    assert env.merges and env.held_log[-1]["passed"] == 1 and env.credit["A1"] > 0
    assert "A2" in env.stale and "A1" not in env.stale
    act(k, "A2", "shell", command="write other v1")
    assert "A2" not in env.stale and "feature_1" in env.h.copies["A2"]
    ok, msg = act(k, "A3", "publish")
    assert "nothing to offer" in msg


def test_a_failed_sandbox_costs_one_command_not_the_run():
    k, env = world()
    def broken(*a, **kw):
        raise FileNotFoundError("gone")
    env.h.shell = broken
    ok, msg = act(k, "A1", "shell", command="anything")
    assert not ok and "sandbox failed" in msg


def test_the_last_tick_offers_what_is_left_in_each_copy():
    k, env = world()
    act(k, "A1", "shell", command="write feature_1 v1")
    k.total_ticks = 1
    tick_end(k, env)
    assert env.merges and env.merges[0]["agent"] == "A1" and env.merges[0]["note"] == "offered at the end of the run"


def test_work_that_unblocks_others_goes_first():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2", "feature_3")
    tick_end(k, env)
    land(k, env, "A1", "plain")
    land(k, env, "A2", "core.py")
    env.h.depends = {"core.py": 5}
    assert env._gate_order()[0] == "A2"
    env.set_law(k, {"offer_order": "credit"})
    assert env._gate_order()[0] == "A1"
    assert "unblocks" not in law_text(env.law)
    assert "work that unblocks others comes first" in law_text(LAW)


def test_nothing_is_lost_what_did_not_land_stays_visible():
    k, env = world()
    land(k, env, "A1", "broken_change")
    tick_end(k, env)
    assert env.rejections and env.unmerged[0]["files"] == ["broken_change"]
    where = env.unmerged[0]["at"]
    assert (env.unmerged_dir / where.split("/unmerged/")[1] / "broken_change").exists()
    assert "stays in your copy" in env.observe(k, "A1")
    assert "broken_change" in env.h.copies["A1"]
    assert "Work that did not land" in env.observe(k, "A3")
    ok, msg = act(k, "A3", "head", what="unmerged")
    assert ok and "broken_change" in msg
    assert "unmerged" in env._mounts(k, "A3")


def test_a_conflict_is_visible_and_the_later_party_reconciles():
    k, env = world()
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    act(k, "A2", "shell", command="write feature_1 mine")
    env._conflict(k, "A2", ["feature_1"])
    a2 = list(k.agents["A2"].inbox)[-1]["text"]
    a1 = list(k.agents["A1"].inbox)[-1]["text"]
    assert "landed by A1" in a2 and "You came later, so you reconcile" in a2 and "message A1" in a2
    assert "A2's work conflicts" in a1


def test_overlap_in_copies_is_shown_to_both_parties():
    k, env = world()
    act(k, "A1", "shell", command="write shared.py a")
    act(k, "A2", "shell", command="write shared.py b")
    act(k, "A3", "shell", command="write other.py b")
    env._wip_t = -1
    seen = env.observe(k, "A1")
    assert "In other copies, not at HEAD yet: A2 shared.py; A3 other.py" in seen
    assert "shared.py also in A2" in seen


def test_an_overlap_is_told_to_both_parties_during_their_turns_once():
    k, env = world()
    act(k, "A1", "shell", command="write shared.py a")
    ok, msg = act(k, "A2", "shell", command="write shared.py b")
    assert "Overlap: shared.py also changed in A1's copy" in msg and "Message A1" in msg
    assert "A2 has also changed shared.py" in list(k.agents["A1"].inbox)[-1]["text"]
    ok, msg = act(k, "A2", "shell", command="write shared.py c")
    assert "Overlap" not in msg


def test_a_decision_can_be_handed_over_and_comes_back_when_the_task_ends():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5")
    tick_end(k, env)
    credit = env.credit["A1"]
    ok, msg = act(k, "A1", "delegate", op="post", title="the csv interface", spec="settle it", to="A5" if False else "A2", scope=["core*"],
                  accept="both groups build on it", until=k.tick + 3, pay=1.5)
    assert ok, msg
    tid = next(iter(k.tasks))
    assert abs(env.credit["A1"] - (credit - 1.5)) < 1e-6 and "pay 1.5" in k.observe("A2", consume=False)
    assert k.tasks[tid].to == "A2" and k.tasks[tid].scope == ["core*"]
    ok, msg = act(k, "A3", "delegate", op="take", task=tid)
    assert not ok and "addressed to" in msg
    ok, msg = act(k, "A2", "delegate", op="take", task=tid)
    assert ok, msg
    assert env._claim_on("core_x")["holder"] == "A2"
    land(k, env, "A3", "core_x")
    tick_end(k, env)
    assert len(env.proposals) == 1 and not any("core_x" in m["files"] for m in env.merges)  # the taker decides
    ok, msg = act(k, "A2", "delegate", op="deliver", task=tid, deliverable="settled", evidence=["R1"])
    assert ok, msg
    ok, msg = act(k, "A2", "delegate", op="review", task=tid, verdict="accept")
    assert not ok and "someone other than the taker" in msg
    ok, msg = act(k, "A1", "delegate", op="review", task=tid, verdict="accept")
    assert ok, msg
    assert env.credit["A2"] >= 1.5 and env._claim_on("core_x") is None
    m = env.delegation_measures(k)
    assert m["offered"] == 1 and m["taken"] == 1 and m["delivered"] == 1 and m["accepted"] == 1 and m["with_scope"] == 1 and m["with_pay"] == 1


def test_a_holder_alone_can_hand_over_what_it_holds_and_gets_it_back():
    k, env = world()
    act(k, "A1", "own", op="claim", paths=["core*"], why="mine")
    ok, msg = act(k, "A2", "delegate", op="post", title="grab it", to="A3", scope=["core*"])
    assert ok, msg
    tid = next(iter(k.tasks))
    ok, msg = act(k, "A3", "delegate", op="take", task=tid)
    assert not ok and "only its holder" in msg
    ok, msg = act(k, "A1", "delegate", op="post", title="hold it", to="A3", scope=["core*"])
    second = sorted(k.tasks)[-1]
    ok, msg = act(k, "A3", "delegate", op="take", task=second)
    assert ok, msg
    assert next(iter(env.claims.values()))["holder"] == "A3"
    ok, msg = act(k, "A1", "delegate", op="drop", task=second, notes="done with this")
    assert ok, msg
    assert next(iter(env.claims.values()))["holder"] == "A1"


def test_a_task_can_be_handed_on_within_its_scope_and_ends_with_its_parent():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2", "feature_3")
    tick_end(k, env)
    act(k, "A1", "delegate", op="post", title="the whole layer", to="A2", scope=["core*", "help*"], pay=1.0)
    top = sorted(k.tasks)[-1]
    act(k, "A2", "delegate", op="take", task=top)
    ok, msg = act(k, "A2", "delegate", op="post", title="part", to="A3", parent=top, scope=["other*"])
    assert not ok and "covers" in msg
    ok, msg = act(k, "A2", "delegate", op="post", title="part", to="A3", parent=top, scope=["help*"], pay=0.4)
    assert ok, msg
    child = sorted(k.tasks, key=lambda t: int(t[1:]))[-1]
    assert k.tasks[top].held == 0.6 and k.tasks[child].held == 0.4
    act(k, "A3", "delegate", op="take", task=child)
    assert env.delegation_measures(k)["deepest"] == 1
    credit = env.credit["A1"]
    ok, msg = act(k, "A1", "delegate", op="drop", task=top)
    assert ok, msg
    assert k.tasks[child].status == "dropped" and abs(env.credit["A1"] - (credit + 1.0)) < 1e-6
    assert env._claim_on("help_x") is None
    assert "ended with" in list(k.agents["A3"].inbox)[-1]["text"]


def test_a_task_can_go_to_a_group_and_any_member_takes_it():
    k, env = world()
    act(k, "A1", "lead", op="found", name="core")
    gid = next(iter(env.teams))
    act(k, "A2", "lead", op="join", team=gid)
    act(k, "A3", "delegate", op="post", title="review my change", to=gid)
    tid = next(iter(k.tasks))
    assert "review my change" in list(k.agents["A1"].inbox)[-1]["text"]  # a team hears through whoever speaks for it
    ok, msg = act(k, "A4", "delegate", op="take", task=tid)
    assert not ok
    ok, msg = act(k, "A2", "delegate", op="take", task=tid)
    assert ok, msg
    assert env.delegation_measures(k)["to_a_group"] == 1


def test_a_task_past_its_tick_lapses_and_returns_what_it_held():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2")
    tick_end(k, env)
    credit = env.credit["A1"]
    act(k, "A1", "delegate", op="post", title="x", to="A2", until=k.tick, pay=1.0, scope=["core*"])
    tid = next(iter(k.tasks))
    act(k, "A2", "delegate", op="take", task=tid)
    assert env._claim_on("core_x")
    k.tick += 1
    k.expire_tasks()
    assert k.tasks[tid].status == "lapsed" and abs(env.credit["A1"] - credit) < 1e-6 and env._claim_on("core_x") is None
    act(k, "A1", "delegate", op="post", title="y", to="A2", until=k.tick, pay=1.0)
    two = sorted(k.tasks, key=lambda t: int(t[1:]))[-1]
    act(k, "A2", "delegate", op="take", task=two)
    act(k, "A2", "delegate", op="deliver", task=two, deliverable="done")
    k.tick += 1
    k.expire_tasks()
    assert k.tasks[two].status == "accepted" and env.credit["A2"] >= 1.0  # delivered work is not lost to a silent requester


def test_a_refused_task_tells_the_requester_and_open_tasks_are_shown_to_everyone():
    k, env = world()
    act(k, "A1", "delegate", op="post", title="anyone?", spec="a job")
    assert "TASKS OPEN TO ANYONE" in k.observe("A2", consume=False) and "anyone?" in k.observe("A2", consume=False)
    act(k, "A1", "delegate", op="post", title="you", to="A2")
    tid = sorted(k.tasks, key=lambda t: int(t[1:]))[-1]
    ok, msg = act(k, "A2", "delegate", op="refuse", task=tid, notes="not now")
    assert ok, msg
    assert k.tasks[tid].status == "refused" and "refused task" in list(k.agents["A1"].inbox)[-1]["text"]
    assert env.delegation_measures(k)["seen_by_someone"] >= 1


def test_turning_down_a_proposal_takes_evidence_not_rank():
    k, env = world()
    act(k, "A1", "own", op="claim", paths=["feature_*"], why="mine")
    land(k, env, "A2", "feature_9")
    tick_end(k, env)
    pid = next(iter(env.proposals))
    ok, msg = act(k, "A1", "own", op="reject", id=pid, why="I do not like it")
    assert not ok and "evidence" in msg
    _, msg = act(k, "A1", "shell", command="write t x")
    rid = re.search(r"R\d+", msg).group()
    ok, msg = act(k, "A1", "own", op="reject", id=pid, why="it fails on empty input", evidence=[rid])
    assert ok, msg
    assert env.proposals[pid]["rejection"] == [rid]
    assert "Evidence" in env.observe(k, "A2")


def test_each_turn_shows_what_paid_what_is_unused_and_what_is_untried():
    k, env = world()
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    seen = env.observe(k, "A2")
    assert "WHAT HAS PAID" in seen and "shell" in seen.split("WHAT HAS PAID")[1]
    assert "ACTIONS NOBODY HAS USED YET" in seen and "toolbox" in seen.split("ACTIONS NOBODY HAS USED YET")[1]
    assert "NOT TRIED YET" in seen and "--human-readable" in seen
    used = env.unused_actions(k)
    assert "shell" not in used
    act(k, "A1", "own", op="claim", paths=["x*"], why="w")
    assert "own" not in env.unused_actions(k)
    env.set_law(k, rules={"show": ["plans"]})
    assert "WHAT HAS PAID" not in env.observe(k, "A2")


def test_the_views_cut_nothing():
    k, env = world()
    env.h.view = lambda: {"coverage": "c", "head": "h", "checks": 1, "probes_left": 1, "implemented": 1, "commands": 1, "checks_by_feature": [],
                          "failing": [{"id": f"cmd:x{i}", "fail": 1, "checks": 2, "critical": i == 29} for i in range(30)],
                          "mismatches": [{"oid": f"o{i}", "args": ["x"], "diff": "d" * 300} for i in range(40)]}
    seen = env.observe(k, "A1")
    assert "cmd:x29" in seen and "cmd:x0 " in seen and "*cmd:x29" in seen
    assert all(f"o{i} " in seen for i in range(40))  # every row, shortened past the budget
    for i in range(12):
        env._tell("A1", f"gate note {i}")
    seen = env.observe(k, "A1")
    assert "gate note 0" in seen and "gate note 11" in seen
    assert "gate note 0" not in env.observe(k, "A1")  # each reaches its agent once


def test_under_a_shared_goal_the_ledger_still_follows_the_earner():
    k, env = world(goal="society")
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    assert any("shell" in c and "publish" in c for c in env.combo_credit)
    assert "What each contributed" in env.observe(k, "A2") and "A1 1.3" in env.observe(k, "A2")
    assert "shared" not in json.dumps(env.window_facts(k, 0)["what_paid"])


def test_ownership_turns_changes_into_proposals_and_pays_the_owner():
    k, env = world()
    ok, msg = act(k, "A1", "own", op="claim", paths=["feature_*"], why="mine")
    assert ok, msg
    land(k, env, "A2", "feature_9")
    tick_end(k, env)
    assert not env.merges and len(env.proposals) == 1
    pid = next(iter(env.proposals))
    ok, msg = act(k, "A1", "own", op="accept", id=pid)
    assert ok, msg
    tick_end(k, env)
    assert env.merges[0]["owner"] == "A1" and env.attributed["A1"] > 0 and env.merges[0]["agent"] == "A2"


def test_a_lapsed_claim_loses_no_work():
    k, env = world(own_ticks="2")
    act(k, "A1", "own", op="claim", paths=["feature_*"], why="mine")
    land(k, env, "A2", "feature_9")
    tick_end(k, env)
    tick_end(k, env)
    tick_end(k, env)
    assert not env.claims and env.merges and env.merges[0]["agent"] == "A2"


def test_a_group_can_hold_code_and_share_copies():
    k, env = world()
    act(k, "A1", "lead", op="found", name="core")
    gid = next(iter(env.teams))
    act(k, "A2", "lead", op="join", team=gid)
    ok, msg = act(k, "A2", "own", **{"op": "claim", "paths": ["feature_*"], "for": gid})
    assert ok, msg
    land(k, env, "A1", "feature_5")  # inside the group: lands
    land(k, env, "A3", "feature_6")  # outside: a proposal
    tick_end(k, env)
    assert [m["agent"] for m in env.merges] == ["A1"] and len(env.proposals) == 1
    assert env._peers(k, "A1") == ["A2"]


def test_toolbox_author_shares_credit_and_uses_are_counted():
    k, env = world()
    ok, msg = act(k, "A2", "toolbox", op="add", name="fails", description="lists failures", script="#!/bin/sh\necho hi\n")
    assert ok, msg
    ok, _ = act(k, "A1", "shell", command="/public/tools/fails; write feature_1 v")
    assert ok
    act(k, "A1", "publish", note="used it")
    tick_end(k, env)
    assert env.tools["fails"]["uses"] == 1 and env.attributed["A2"] > 0


def test_trace_is_open_and_counted_and_role_is_shown():
    k, env = world()
    act(k, "A1", "shell", command="write a b")
    ok, msg = act(k, "A2", "trace", agent="A1")
    assert ok and "write a b" in msg
    assert env.reads[("A2", "A1")] == 1
    act(k, "A1", "role", title="lead builder", does="send me failing cases")
    assert "A1 lead builder" in env.observe(k, "A3")


def test_fine_costs_the_one_who_fines_and_destroys_what_it_takes():
    k, env = world()
    land(k, env, "A1", "feature_1", "feature_2")
    tick_end(k, env)
    land(k, env, "A2", "feature_3", "feature_4")
    tick_end(k, env)
    a1, a2 = env.credit["A1"], env.credit["A2"]
    ok, msg = act(k, "A2", "pay", op="fine", to="A1", amount=1, why="broke the build")
    assert ok, msg
    assert abs(env.credit["A2"] - (a2 - 1)) < 1e-6 and abs(env.credit["A1"] - (a1 - 2)) < 1e-6


def test_law_is_bounded_and_goal_can_be_locked():
    law = clean_law({"gain": 99, "steps": 1, "goal": "nonsense", "offer_order": "open", "unknown": 3})
    assert law["gain"] == 2 and law["steps"] == 4 and law["goal"] == "individual" and law["offer_order"] == "open"
    assert "unknown" not in law
    assert "Credit goes to" in law_text(law)
    k, env = world()
    out = env.set_law(k, {"goal": "society", "piece_share": 0.4}, {"lead_share": 0.9}, "Test before you build.", "because")
    assert env.law["goal"] == "society" and env.law["piece_share"] == 0.4 and env.rules["credit"]["lead_share"] == 0.6
    assert "goal" in out["changed"] and env.norms and "PRACTICES" in env.observe(k, "A1") and env.law_v == 2
    k, env = world(goal_locked="1")
    env.set_law(k, {"goal": "society"})
    assert env.law["goal"] == "individual"


def test_a_review_changes_rules_and_never_says_who_does_what():
    class Org(MockBackend):
        async def complete(self, k, system, prompt, **kw):
            if kw.get("who") != "organizer":
                return "fine"
            assert "never say who does what" in system
            return json.dumps({"law": {"check_share": 0.3}, "practices": "A1 holds the skeleton and G2 tests it; say which piece you used.",
                               "teams": [{"name": "core", "members": ["A1", "A2"], "lead": "A1"}], "roles": {"A2": "verifier"},
                               "suggestions": ["Two agents wrote csvio twice; a team around it may pay."], "why": "overlap", "expect": "gain rises"})

    k, env = world()
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 3
    env.held_log.append({"t": 3, "built": True, "passed": 5, "total": 179})
    asyncio.run(loop.on_tick_end(k, Org(0)))
    assert env.law["check_share"] == 0.3
    assert not k.active_groups() and not env.teams and not env.titles  # teams and roles stay the agents' own
    assert "A1" not in env.norms and "G2" not in env.norms and "say which piece you used" in env.norms
    assert env.suggestions and "Suggested (nothing requires it)" in k.boards["global"].entries[-1].content
    assert "PRACTICES" in env.observe(k, "A1") and "yours to use, change or ignore" in env.observe(k, "A1")


def test_window_facts_are_counts_and_never_name_hidden_tests():
    k, env = world()
    land(k, env, "A1", "feature_1")
    land(k, env, "A2", "feature_2")
    tick_end(k, env)
    f = env.window_facts(k, 0)
    assert f["merges"] == 2 and f["verified_gains"] == 2 and f["held_out"]["gain_this_window"] == 2
    assert f["agents"]["A1"]["merges"] == 1 and "shell" in f["action_mix"]
    assert 0 < f["working_share"] <= 1
    assert "hidden" not in json.dumps(f)


def run_review(k, env, loop, window_gains):
    """Drive the loop with chosen held-out gains."""
    backend = MockBackend(0)
    out = []
    for gain in window_gains:
        k.tick += loop.window
        now = env.held_log[-1]["passed"] + gain
        env.held_log.append({"t": k.tick, "built": True, "passed": now, "total": 179})
        asyncio.run(loop.on_tick_end(k, backend))
        out.append(loop.history[-1])
    return out


def test_review_changes_the_law_and_reverts_when_gain_collapses():
    k, env = world()
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 0
    env.held_log = [{"t": 0, "built": True, "passed": 0, "total": 179}]
    start = env.law["piece_share"]
    first, second = run_review(k, env, loop, [30, 2])
    assert first["changed"] and env.law_v >= 2
    assert second["verdict_on_last"].startswith("the change at") and "reverted" in second["verdict_on_last"]
    assert first["verdict"] == "reverted" and first["gain_after"] == 2
    assert k.boards["global"].entries and any("REFLECTION" in e.content for e in k.boards["global"].entries)
    assert env.law["piece_share"] != start  # the mock proposes again after the revert
    assert k.agents["A1"].notes and any(e.kind == "lesson" for e in k.boards["private:A1"].entries)


def test_a_reverted_review_restores_the_views_and_practices():
    class Org(MockBackend):
        async def complete(self, k, system, prompt, **kw):
            if kw.get("who") != "organizer":
                return "fine"
            return json.dumps({"rules": {"show": ["plans"]}, "practices": "Say what you ruled out.", "why": "w"}) if k.tick == 3 else json.dumps({"why": "nothing"})

    k, env = world()
    show = list(env.rules["substrate"]["show"])
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 3
    env.held_log.append({"t": 3, "built": True, "passed": 30, "total": 179})
    asyncio.run(loop.on_tick_end(k, Org(0)))
    assert env.rules["substrate"]["show"] == ["plans"] and env.norms
    k.tick = 6
    env.held_log.append({"t": 6, "built": True, "passed": 31, "total": 179})
    asyncio.run(loop.on_tick_end(k, Org(0)))
    assert env.rules["substrate"]["show"] == show and not env.norms and "reverted" in loop.history[-1]["verdict_on_last"]


def test_a_change_that_keeps_gaining_stands():
    k, env = world()
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 0
    env.held_log = [{"t": 0, "built": True, "passed": 0, "total": 179}]
    first, second = run_review(k, env, loop, [30, 25])
    assert first["verdict"] == "kept"


def test_an_unbuilt_head_is_no_signal():
    k, env = world()
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 3
    env.held_log = [{"t": 0, "built": True, "passed": 0, "total": 179}, {"t": 3, "built": True, "passed": 30, "total": 179}]
    asyncio.run(loop.on_tick_end(k, MockBackend(0)))
    change = loop.open_change
    assert change
    k.tick = 6
    env.held_log.append({"t": 6, "built": False, "passed": 0, "total": 179})
    asyncio.run(loop.on_tick_end(k, MockBackend(0)))
    assert "no signal" in loop.history[-1]["verdict_on_last"]
    assert "verdict" not in change  # still waiting for a real reading


def test_review_failure_keeps_the_law():
    class Bad(MockBackend):
        async def complete(self, *a, **kw):
            if kw.get("who") == "organizer":
                return "not json"
            return "a reflection"

    k, env = world()
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 3
    before = dict(env.law)
    env.held_log.append({"t": 3, "built": True, "passed": 5, "total": 179})
    asyncio.run(loop.on_tick_end(k, Bad(0)))
    assert env.law == before and loop.history[-1]["why"].startswith("(review failed")


def test_proposals_reach_the_organizer_and_the_answer_is_enforced():
    seen = {}

    class Org(MockBackend):
        async def complete(self, k, system, prompt, **kw):
            if kw.get("who") != "organizer":
                seen.setdefault("reflect", prompt)
                return "we overlap too much"
            seen["prompt"] = prompt
            return json.dumps({"law": {"goal": "team", "check_share": 0.3, "offer_order": "open"}, "practices": "Hold what you build.",
                               "why": "overlap", "expect": "gain rises"})

    k, env = world()
    ok, msg = act(k, "A3", "substrate", op="propose", text="pay testers more")
    assert ok, msg
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    loop = ReflectionLoop(3)
    k.total_ticks = 100
    k.tick = 3
    asyncio.run(loop.on_tick_end(k, Org(0)))
    assert "pay testers more" in seen["prompt"] and "we overlap too much" in seen["prompt"]
    assert "what_paid" in seen["prompt"] and "actions_never_used" in seen["prompt"]
    assert "which actions, combinations of actions" in seen["reflect"]
    assert env.law["goal"] == "team" and env.law["check_share"] == 0.3 and env.norms == "Hold what you build."
    assert "hidden_tests" not in seen["prompt"]


def test_culture_carries_the_law_and_tools_but_no_code():
    k, env = world()
    act(k, "A2", "toolbox", op="add", name="fails", description="d", script="#!/bin/sh\necho hi\n")
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    env.set_law(k, {"piece_share": 0.4}, norms="Cite what you used.")
    c = env.culture()
    k2, env2 = world()
    env2.started = False
    env2.inherit_culture(k2, json.loads(json.dumps(c)))
    env2.on_tick_start(k2)
    assert env2.law["piece_share"] == 0.4 and env2.norms == "Cite what you used."
    assert "fails" in env2.tools and (env2.tools_dir / "fails").exists()
    assert "feature_1" not in json.dumps(c) and not any("feature_1" in f for f in env2.h.head)


def test_heritage_carries_what_was_established_or_only_how_it_ran():
    k, env = world()
    _, msg = act(k, "A1", "shell", command="write x y")
    rid = re.search(r"R\d+", msg).group()
    act(k, "A1", "guide", op="add", line="sort is stable", evidence=[rid])
    act(k, "A1", "plan", op="claim", hypothesis="h")
    act(k, "A1", "plan", op="done", finding="no sampling seed exists", outcome="dead_end", evidence=[rid])
    c = json.loads(json.dumps(env.culture()))
    k2, env2 = world()
    env2.inherit_culture(k2, c)
    assert "sort is stable" in env2.observe(k2, "A1")
    assert any("no sampling seed" in e.content for e in k2.boards["global"].entries)
    k3, env3 = world(heritage="org")
    env3.inherit_culture(k3, c)
    assert "sort is stable" not in env3.observe(k3, "A1")
    assert not any("no sampling seed" in e.content for e in k3.boards["global"].entries)


def test_rehearsal_through_the_runner_with_reflection_and_lineage():
    cfg = {"condition": "self_org", "agents": 6, "ticks": 7, "actions": 12, "attention": 6000, "world": "code", "calls": None,
           "world_opts": {**OPTS, "continuity": "0"}, "reflect_window": 3, "reflect_floor": 0.5, "instruments": 3, "experiments": 30,
           "seed_runs": 3, "noise": .3, "validate_every": 3}
    k = build(cfg, 1, "code")
    out = Path(tempfile.mkdtemp(prefix="alphateam-run-"))
    meta = {"condition": "self_org", "backend": "mock"}
    data = asyncio.run(run(k, MockBackend(1), 7, out, meta, 1, quiet=True))
    world_data = data["world"]
    assert data["meta"]["reflection"] and world_data["kind"] == "code"
    assert world_data["hidden"]["status"] == "fake" and world_data["merges"]
    assert (out / "report.md").read_text().startswith("# AlphaTeam run")
    assert "Hidden tests" in (out / "report.md").read_text()
    assert data["culture"]["world"]["law"]
    k2 = build(cfg, 2, "code")
    inherit(k2, data["culture"])
    assert k2.env.law == data["culture"]["world"]["law"]
    json.dumps(culture(k, "lessons"), default=list)


def test_single_agent_has_no_society_tools():
    env = CodeWorld(1, single=True, world_agents=10, opts=dict(OPTS))
    names = {t["name"] for t in env.TOOLS}
    assert not names & {"own", "toolbox", "pay", "lead"} and env.steps_per_tick == 120


def test_credit_off_is_a_plain_baseline():
    k, env = world(credit="0")
    names = {t["name"] for t in env.TOOLS}
    assert "pay" not in names
    land(k, env, "A1", "feature_1")
    tick_end(k, env)
    assert sum(env.credit.values()) == 0 and env.merges
    assert "YOUR GOAL" not in env.brief()


# ---------------------------------------------------------------------------------------------------------------
# Three rules proposed on 2026-11-10 by readers of the record of ten runs (DESIGN.md section 22). Each is a setting
# whose default is what the world always did.

def test_take_head_after_a_conflict_the_copy_holds_heads_file_and_the_agents_version_is_kept():
    """Proposed by all three readers: a file that once conflicted stayed in its copy, went to the gate again with every
    later offer (108 of 129 conflict refusals were repeats), and what the agent ran was no longer what the gate built."""
    k, env = world(on_conflict="take_head")
    act(k, "A1", "shell", command="write feature_1 theirs")
    act(k, "A1", "publish", note="first")
    act(k, "A2", "shell", command="write feature_1 mine")
    act(k, "A2", "shell", command="write feature_2 other")
    act(k, "A2", "publish", note="second")
    tick_end(k, env)
    assert [(m["agent"], m["files"]) for m in env.merges] == [("A1", ["feature_1"]), ("A2", ["feature_2"])]
    assert [(x["agent"], x["file"], x["reason"]) for x in env.rejections] == [("A2", "feature_1", "conflict")]
    assert env.h.copies["A2"]["feature_1"] == "theirs" and env.h.changed("A2") == []  # HEAD's version; nothing left to offer
    (row,) = [u for u in env.unmerged if u["agent"] == "A2"]  # kept once, where everyone can see it
    assert (env.unmerged_dir / row["at"].split("/unmerged/")[1] / "feature_1").read_text() == "mine"
    note = list(k.agents["A2"].inbox)[-1]["text"]
    assert note == ("conflict on rebase: your copy now holds HEAD's version of feature_1; your version was moved to "
                    "/public/unmerged/t1-A2/. Landed by A1. Your other changed files are untouched.")
    assert "A2's work conflicts" in list(k.agents["A1"].inbox)[-1]["text"]
    seen = env.observe(k, "A2")
    assert "rejected: conflict with what landed this tick" in seen and "Conflicts with HEAD" not in seen
    assert "Moved aside by a conflict (HEAD's version is in your copy): feature_1 at t1, your version at /public/unmerged/t1-A2/" in seen
    act(k, "A2", "shell", command="write feature_3 new")
    act(k, "A2", "publish", note="third")
    tick_end(k, env)
    assert len(env.rejections) == 1 and env.merges[-1]["files"] == ["feature_3"]  # the old file is not turned away again
    act(k, "A2", "shell", command="write feature_1 again")  # the line stays until the agent next changes that path
    assert "Moved aside" not in env.observe(k, "A2")
    assert "the copy then holds HEAD's version of it" in env.brief() and not any(w in note.lower() for w in ("you should", "you must", "please"))
    k0, env0 = world()  # the default: as it always was
    assert not env0.take_head and "what did not land stays in its copy and is visible to everyone at /public/unmerged." in env0.brief()


def test_take_head_on_the_real_merge():
    """The same rule on te-next's three-way merge (skipped where that harness is not installed)."""
    from alphateam.code_world import HARNESS, Copy, Harness
    if not Path(HARNESS, "codetest").is_dir():
        return
    for path in (str(Path(HARNESS) / "vendor"), HARNESS):
        if path not in sys.path:
            sys.path.insert(0, path)
    from codetest import evidence, sandbox

    def tree(root: Path, files: dict):
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)

    for take in (False, True):
        root = Path(tempfile.mkdtemp(prefix="alphateam-rebase-"))
        h = Harness.__new__(Harness)
        h.ev, h.sb, h.take_head, h.moved = evidence, sandbox, take, {}
        h.repo = SimpleNamespace(path=root / "head")
        c = h.copies = {"A1": Copy(root / "A1")}
        c = c["A1"]
        tree(h.repo.path, {"prog/a.py": "x = 2\n", "prog/keep.py": "k = 1\n"})          # someone landed a.py
        tree(c.base, {"prog/keep.py": "k = 1\n"})                                         # the copy started without it
        tree(c.source, {"prog/a.py": "x = 3\n", "prog/keep.py": "k = 1\n", "prog/b.py": "y = 1\n"})  # and wrote its own, and b.py
        new = h.rebase_one("A1")
        assert new == ["prog/a.py"], new
        if take:
            assert (c.source / "prog/a.py").read_text() == "x = 2\n" and h.changed("A1") == ["prog/b.py"] and not c.conflicts
            assert h.moved["A1"]["prog/a.py"][0] == b"x = 3\n"
            assert h.rebase_one("A1") == []  # nothing stands: the next rebase reports nothing
        else:
            assert (c.source / "prog/a.py").read_text() == "x = 3\n" and h.changed("A1") == ["prog/a.py", "prog/b.py"]
            assert set(c.conflicts) == {"prog/a.py"} and not h.moved


def test_offer_named_only_the_files_a_publish_names_reach_the_gate():
    """Proposed from the record: in 57 of 426 publishes the note said not to offer something, and the world offered
    every changed file regardless."""
    from alphateam.code_world import PUBLISH_TOOL, publish_tool
    k, env = world(offer="named")
    tool = next(t for t in env.TOOLS if t["name"] == "publish")
    assert tool["parameters"]["required"] == ["paths"] and "only those are offered, and the note does not select files" in tool["description"]
    assert publish_tool("asked") is PUBLISH_TOOL and "paths" not in PUBLISH_TOOL["parameters"]["properties"]
    act(k, "A1", "shell", command="write feature_1 v1")
    act(k, "A1", "shell", command="write broken_draft v1")
    ok, msg = act(k, "A1", "publish", note="everything is fine")
    assert not ok and "publish offers the files you name in paths. Your copy changes: broken_draft, feature_1" in msg
    ok, msg = act(k, "A1", "publish", note="one", paths=["feature_1"])
    assert ok and "queued for the gate at the end of this tick: feature_1. Not offered, only in your copy: broken_draft." in msg
    tick_end(k, env)
    assert env.merges[-1]["files"] == ["feature_1"] and not env.rejections and not env.unmerged  # neither offered nor turned away
    assert env.h.changed("A1") == ["broken_draft"]
    ok, msg = act(k, "A1", "publish", note="all of it", paths=["*"])
    assert ok and "queued for the gate at the end of this tick: broken_draft." in msg
    tick_end(k, env)
    assert [x["file"] for x in env.rejections] == ["broken_draft"]
    assert "publish offers the changed files you name to HEAD at the end of the tick; the rest stay in your copy." in env.brief()
    k.tick = k.total_ticks  # the end of the run still offers whatever is left in a copy
    act(k, "A2", "shell", command="write feature_7 v1")
    env.on_tick_end(k)
    assert env.merges[-1]["files"] == ["feature_7"] and env.merges[-1]["note"] == "offered at the end of the run"


def test_offer_every_tick_the_gate_takes_each_copy_without_a_publish():
    """Proposed from the record: an agent has three actions a round and publish takes one; 249 of 357 offers came in a
    later round than the change they carried."""
    k, env = world(offer="every_tick")
    act(k, "A1", "shell", command="write feature_1 v1")  # no publish
    tick_end(k, env)
    assert env.merges[-1]["files"] == ["feature_1"] and env.merges[-1]["note"] == "offered at the end of the tick"
    act(k, "A2", "shell", command="write broken_x v1")
    tick_end(k, env)
    assert len(env.rejections) == 1
    tick_end(k, env)
    assert len(env.rejections) == 1  # the version the gate turned away is not offered again by the world
    act(k, "A2", "shell", command="write broken_x v2")
    tick_end(k, env)
    assert len(env.rejections) == 2  # a changed one is
    ok, msg = act(k, "A2", "publish", note="again, as it is")
    assert ok and msg.startswith("noted: the gate takes your copy at the end of this tick")
    tick_end(k, env)
    assert len(env.rejections) == 3  # and a publish offers it again
    act(k, "A3", "own", op="claim", paths=["feature_9"], why="mine for now")
    act(k, "A4", "shell", command="write feature_9 v1")
    tick_end(k, env)
    assert not env.proposals and env.h.changed("A4") == ["feature_9"]  # a held file is left out; nothing is proposed for it
    ok, _ = act(k, "A4", "publish", note="for its holder")
    tick_end(k, env)
    assert len(env.proposals) == 1  # a publish still proposes it
    text = env.brief()
    assert "At the end of every tick the gate takes whatever your copy has changed since it last judged it." in text
    assert "offered once more" not in text and "they are offered at the end of every tick with or without it" in \
        next(t for t in env.TOOLS if t["name"] == "publish")["description"]


def test_show_open_a_turn_says_which_missing_commands_no_copy_has_changed():
    """Proposed from the record: every run ended with commands nobody had started (4.8 of 20 on average), and agents
    spent actions asking who was on what."""
    k, env = world(show_open="1")
    env.h.listed = ["1", "2", "3"]
    act(k, "A2", "shell", command="write feature_2 v1")
    seen = env.observe(k, "A1")
    assert "Not at HEAD and changed in no copy: 1, 3. Not at HEAD, changed in a copy: 2 (A2)." in seen
    land(k, env, "A3", "feature_1", "feature_3")
    tick_end(k, env)
    assert "Not at HEAD and changed in no copy: none. Not at HEAD, changed in a copy: 2 (A2)." in env.observe(k, "A1")
    k0, env0 = world()
    env0.h.listed = ["1", "2", "3"]
    assert "changed in no copy" not in env0.observe(k0, "A1")  # the default: the line is not there


# Two rules the societies themselves suggested, in the closing rounds of the five baseline runs of 2026-11-10.

def test_hold_back_a_conflicted_file_stays_out_of_offers_until_its_agent_changes_it():
    """Six suggestions from three societies, in their words: "reject unchanged stale snapshots, rather than repeatedly
    attempting conflicting offers"; "require the contributor to rebase/reconcile and explicitly resubmit"."""
    k, env = world(on_conflict="hold_back")
    act(k, "A1", "shell", command="write feature_1 theirs")
    act(k, "A1", "publish", note="first")
    act(k, "A2", "shell", command="write feature_1 mine")
    act(k, "A2", "shell", command="write feature_2 other")
    act(k, "A2", "publish", note="second")
    tick_end(k, env)
    assert [(m["agent"], m["files"]) for m in env.merges] == [("A1", ["feature_1"]), ("A2", ["feature_2"])]
    assert [(x["agent"], x["file"], x["reason"]) for x in env.rejections] == [("A2", "feature_1", "conflict")]
    assert env.h.copies["A2"]["feature_1"] == "mine" and env.h.changed("A2") == ["feature_1"]  # the copy keeps its version
    seen = env.observe(k, "A2")
    assert "Conflicts with HEAD, left out of your offers until you change them: feature_1." in seen
    assert "it is left out of your offers until you change it" in seen and "Moved aside" not in seen
    assert list(k.agents["A2"].inbox)[-1]["text"] == (
        "conflict on rebase in feature_1: your version stays in your copy, and out of your offers until you change it; it is "
        "also at /public/unmerged/t1-A2/. Landed by A1. HEAD's is in /inputs/head.")
    act(k, "A2", "shell", command="write feature_3 new")
    act(k, "A2", "publish", note="third")
    tick_end(k, env)
    assert len(env.rejections) == 1 and env.merges[-1]["files"] == ["feature_3"]  # the stale file did not go to the gate again
    assert env.h.changed("A2") == ["feature_1"] and env.h.head["feature_1"] == "theirs"
    act(k, "A2", "shell", command="write feature_1 reconciled")  # once its agent changes it, it is offered, against the HEAD of now
    act(k, "A2", "publish", note="reconciled with what landed")
    tick_end(k, env)
    assert env.merges[-1]["files"] == ["feature_1"] and env.h.head["feature_1"] == "reconciled" and len(env.rejections) == 1
    assert "Conflicts with HEAD" not in env.observe(k, "A2")
    assert "is left out of its copy's offers until its agent changes it" in env.brief()


def test_publish_replays_a_publish_returns_the_logged_checks_on_the_copy():
    """About seven suggestions from four societies, three of them backed by a second agent: "automatically run a
    syntax/import smoke test ... then replay all relevant logged checks; report concise traceback to the author"."""
    k, env = world(publish_replays="1")
    act(k, "A1", "shell", command="write feature_1 v1")
    steps = env.steps[("A1", k.tick)]
    ok, msg = act(k, "A1", "publish", note="first")
    assert ok and msg.startswith("queued: the gate takes offers at the end of this tick\nR")
    assert "the logged checks replayed on your copy as it is now:" in msg and "ran python3 /public/replay.py" in msg
    assert env.steps[("A1", k.tick)] == steps  # the world runs it: it takes none of the agent's steps
    assert next(t for t in env.TOOLS if t["name"] == "publish")["description"].endswith("replayed on your copy as it is at that moment.")
    assert "A publish also returns the logged checks replayed on your copy" in env.brief()
    k0, env0 = world()
    act(k0, "A1", "shell", command="write feature_1 v1")
    assert act(k0, "A1", "publish", note="first") == (True, "queued: the gate takes offers at the end of this tick")


# Two rules proposed in round 2 by three readers of the record of 25 runs (2026-11-10).

def test_first_copy_a_new_file_is_written_in_one_copy_at_a_time():
    """All three readers, independently: the commands missing at the end were never started, while several copies wrote the
    same new file (cmd_count.py in five copies in one round) and settled it by messages."""
    k, env = world(new_file="first_copy")
    ok, msg = act(k, "A1", "shell", command="write feature_1 first")
    assert ok and "Not kept" not in msg
    ok, msg = act(k, "A2", "shell", command="write feature_1 second write feature_2 other")
    assert ok and "Not kept: feature_1. It is not at HEAD and A1's copy has had it since t1" in msg and "until it lands or t4 ends" in msg
    assert "Your version is at /public/unmerged/t1-A2/feature_1" in msg and "Overlap" not in msg
    assert env.h.changed("A2") == ["feature_2"] and "feature_1" not in env.h.copies["A2"]  # only that file is taken out
    assert (env.unmerged_dir / "t1-A2" / "feature_1").read_text() == "second"  # nothing is lost
    assert "feature_1 (held until t4)" in env.observe(k, "A3") and "Held for you (not at HEAD, first written in your copy): feature_1 until t4." in env.observe(k, "A1")
    assert "written in one copy at a time" in env.brief()
    act(k, "A1", "publish", note="first")
    tick_end(k, env)
    assert [(m["agent"], m["files"]) for m in env.merges] == [("A1", ["feature_1"])]
    ok, msg = act(k, "A2", "shell", command="write feature_1 revised")  # once it is at HEAD the rule no longer applies
    assert ok and "Not kept" not in msg and "feature_1" in env.h.changed("A2")
    # a hold ends by time and is not replaced: the file is then open, as under the starting rules
    act(k, "A3", "shell", command="write feature_9 a")
    for _ in range(4):
        tick_end(k, env)
    ok, msg = act(k, "A4", "shell", command="write feature_9 b")
    assert ok and "Not kept" not in msg and "feature_9" in env.h.changed("A4") and "feature_9" in env.h.changed("A3")
    k0, env0 = world()  # the default: any copy may write it, and the texts are the old ones
    act(k0, "A1", "shell", command="write feature_1 first")
    ok, msg = act(k0, "A2", "shell", command="write feature_1 second")
    assert "Not kept" not in msg and "Message A1 to settle it directly." in msg and "one copy at a time" not in env0.brief()


def test_instructions_off_the_texts_keep_their_facts_and_drop_what_to_do():
    """A reader who followed single agents turn by turn: 185 of 720 actions are the ones the standing texts name."""
    from alphateam.code_world import CYCLE
    k, env = world(instructions="0")
    assert CYCLE not in env.brief()
    act(k, "A1", "shell", command="write feature_1 first")
    ok, msg = act(k, "A2", "shell", command="write feature_1 second")
    assert msg.rstrip().endswith("Overlap: feature_1 also changed in A1's copy.")
    assert list(k.agents["A1"].inbox)[-1]["text"] == "A2 has also changed feature_1 in its copy"
    seen = env.observe(k, "A2")
    assert "Overlap with your changes: feature_1 also in A1" in seen and "settle" not in seen and "Test one nobody has" not in seen
    k0, env0 = world()
    assert CYCLE in env0.brief()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
