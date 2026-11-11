"""Directions (directions=1) on the DiG-bench fake server (fake=1): no network, no model calls."""
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import build, run  # noqa: E402

OPTS = {"fake": "1", "games": "T-1", "explore": "1", "sync": "1", "carry": "1", "guide": "1", "hypotheses": "1",
        "substrate": "1", "board": "1", "publish_solutions": "1", "trust": "1", "directions": "1"}
CFG = {"condition": "self_org", "agents": 4, "ticks": 4, "actions": 4, "attention": 6000, "world": "dig",
       "world_opts": OPTS, "seed": 1, "structures": ["dig"], "episodes": 1, "population": 1}
rid = lambda out: out.split("receipt ")[1].split("]")[0]


def act(k, aid, name, args):
    return asyncio.run(k.act(aid, name, args))


def test_directions_are_offered_shown_and_file_their_pieces():
    k = build(CFG, 0, "dig")
    w = k.env
    a, b, c, d = sorted(k.agents, key=lambda x: int(x[1:]))
    assert "direction" in w.tool_names and "Directions:" in w.brief()
    k.tick = 1
    # c opens a direction first, then a opens another; b joins a's
    ok, out = act(k, c, "direction", {"op": "open", "text": "the counter on the right is the exit"})
    assert ok and "you are on it" in out, out
    d2 = out.split("opened ")[1].split()[0]
    ok, out = act(k, a, "direction", {"op": "open", "text": "the goal is to stamp on the star"})
    d1 = out.split("opened ")[1].split()[0]
    assert f"{d2} 1 agent" in out and f"{d1} 1 agent" in out, out
    k.tick = 2
    ok, out = act(k, b, "direction", {"op": "join", "id": d1})
    assert ok and f"you are on {d1} (2 agents" in out, out
    assert any(f"joined your direction {d1}" in m["text"] for m in k.agents[a].inbox)
    assert not act(k, d, "direction", {"op": "join", "id": "D999"})[0]
    # a claim and a finding are filed under the author's direction, and the IDEAS list says which
    ok, out = act(k, a, "plan", {"op": "claim", "hypothesis": "stamping (3) on the star clears the level"})
    pid = out.split("claimed ")[1].split()[0]
    assert w.plans[pid]["direction"] == d1 and w.plans[pid]["entry"] in w.directions[d1]["pieces"]
    ra = rid(act(k, a, "play", {"actions": ["1"]})[1])
    ok, out = act(k, a, "plan", {"op": "done", "finding": "1 moves the cursor left", "evidence": [ra], "outcome": "works"})
    fa = out.split("posted finding ")[1].split()[0]
    assert w.piece_dir[fa] == d1 and f"[{d1}]" in w.ideas_view(k, "t1", 1)
    # the view: one's own direction on top (YOUR DIRECTION), then every direction, fewest agents first; herding and
    # agents on none are counted
    view = w.observe(k, b)
    assert f"YOUR DIRECTION {d1}" in view and "2 of 4 agents here" in view, view
    rows = view.split("DIRECTIONS on t1 level 1")[1]
    assert rows.index(f"  {d2} 1 agent") < rows.index(f"  {d1} 2 agents") and "1 agent on none" in rows, rows
    # c, on d2, builds on a's finding: the claim combines d1 and d2, and d1's members are told
    ok, out = act(k, c, "plan", {"op": "claim", "hypothesis": "move left with 1 until the counter stops, then 3",
                                 "builds_on": [fa]})
    assert "combines" in out and d1 in out and d2 in out, out
    assert any("combines pieces of" in m["text"] for m in k.agents[b].inbox)
    # leaving keeps the ticks one spent on it
    ok, out = act(k, d, "direction", {"op": "join", "id": d2})
    ok, out = act(k, d, "direction", {"op": "leave"})
    assert ok and "ticks on it still count" in out and w.directions[d2]["tenure"][d] == 1, out


def test_deliveries_put_smaller_directions_first():
    k = build(CFG, 0, "dig")
    w = k.env
    a, b, c, d = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    out_c = act(k, c, "direction", {"op": "open", "text": "the counter on the right is the exit"})[1]
    act(k, c, "plan", {"op": "claim", "hypothesis": "the counter reaching 9 opens the exit"})
    out_a = act(k, a, "direction", {"op": "open", "text": "the goal is to stamp on the star"})[1]
    act(k, a, "plan", {"op": "claim", "hypothesis": "stamping on the star clears the level"})
    d1 = out_a.split("opened ")[1].split()[0]
    e_c, e_a = w.directions[out_c.split("opened ")[1].split()[0]]["entry"], w.directions[d1]["entry"]
    act(k, b, "direction", {"op": "join", "id": d1})
    act(k, d, "direction", {"op": "join", "id": d1})
    before = k.counts["pushed"]
    pushed = k._push(b)
    # c's direction holds 1 agent and b's own holds 3: the smaller direction's posts reach b first, though b is on d1
    assert pushed.index(e_c) < pushed.index(e_a), pushed
    assert k.counts["pushed"] - before == 4 and "Ideas claimed on your game (4)" in pushed, pushed  # nothing dropped
    # among directions of equal size, one's own comes first
    act(k, d, "direction", {"op": "leave"})
    act(k, b, "direction", {"op": "leave"})
    assert w.deliver_order(k, a, k.entry(e_a)) < w.deliver_order(k, a, k.entry(e_c))
    assert w.deliver_order(k, c, k.entry(e_c)) < w.deliver_order(k, c, k.entry(e_a))


def test_a_level_that_falls_pays_the_directions_it_cites_by_ticks_and_tests():
    k = build(CFG, 0, "dig")
    w = k.env
    a, b, c, d = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    d1 = act(k, a, "direction", {"op": "open", "text": "the goal is to stamp on the star"})[1].split("opened ")[1].split()[0]
    d2 = act(k, c, "direction", {"op": "open", "text": "walk to the star first"})[1].split("opened ")[1].split()[0]
    d3 = act(k, d, "direction", {"op": "open", "text": "the counter is the exit"})[1].split("opened ")[1].split()[0]
    out = act(k, c, "plan", {"op": "claim", "hypothesis": "the star is where 3 works"})[1]
    piece = w.plans[out.split("claimed ")[1].split()[0]]["entry"]
    k.tick = 2
    act(k, b, "direction", {"op": "join", "id": d1})
    pa = act(k, a, "plan", {"op": "claim", "hypothesis": "3 on the star clears the level"})[1].split("claimed ")[1].split()[0]
    ra = rid(act(k, a, "play", {"actions": ["1"]})[1])
    act(k, a, "plan", {"op": "done", "finding": "1 moves left; nothing cleared yet", "evidence": [ra], "outcome": "dead_end"})
    assert w.plans[pa]["status"] == "done" and w.plans[pa]["direction"] == d1
    k.tick = 3
    before = dict(w.credit)
    g = w.server.sessions[w.sessions[(a, "t1")]["sid"]][0]
    walk = ["2" if g.target > g.pos else "1"] * abs(g.target - g.pos) + ["3"]
    ok, out = act(k, a, "play", {"actions": walk, "cites": [piece]})
    assert "LEVEL 1 COMPLETED" in out, out
    # d1 is the solver's: 3 credit shared by ticks on it plus 3 shares per claim tested: a (t1-t3, one claim tested)
    # 3 + 3 = 6 shares, b (t2-t3) 2 shares; d2 was cited: c alone gets 3; d3 nothing
    paid = {x["agent"]: x["credit"] for x in w.directions[d1]["paid"]["to"]}
    assert paid == {a: 2.25, b: 0.75} and w.directions[d1]["paid"]["why"] == "the solver's", w.directions[d1]
    assert [x["credit"] for x in w.directions[d2]["paid"]["to"]] == [3.0] and w.directions[d2]["paid"]["why"] == "cited"
    assert "paid" not in w.directions[d3] and any("without citing your direction" in m["text"] for m in k.agents[d].inbox)
    assert round(w.credit[b] - before.get(b, 0), 3) >= 0.75
    assert any("your share of 3 (2 ticks on it, 0 claims tested)" in m["text"] for m in k.agents[b].inbox)
    # every direction on the fallen level is closed; nobody is on one, and the next level starts with none
    assert all(x["status"] == "closed" for x in w.directions.values()) and not w.direction_of(b)
    assert not act(k, b, "direction", {"op": "join", "id": d1})[0]
    assert "YOUR DIRECTION" not in w.observe(k, b)
    s = w.score()
    assert s["directions"] == 3 and s["directions_paid"] == 2 and s["direction_peak_share"] >= 0.25
    assert s["direction_credit"] == 6.0  # reported apart from unique-idea and understanding credit


def test_without_the_option_nothing_changes():
    k = build(dict(CFG, world_opts={x: v for x, v in OPTS.items() if x != "directions"}), 0, "dig")
    w = k.env
    a = sorted(k.agents, key=lambda x: int(x[1:]))[0]
    k.tick = 1
    assert "direction" not in w.tool_names and "Directions:" not in w.brief() and "DIRECTIONS" not in w.observe(k, a)
    out = act(k, a, "plan", {"op": "claim", "hypothesis": "stamping on the star clears the level"})[1]
    assert "direction" not in w.plans[out.split("claimed ")[1].split()[0]] and "directions" not in w.score()


def test_a_mock_run_with_directions():
    out = Path(__file__).resolve().parent / "_tmp_directions"
    shutil.rmtree(out, ignore_errors=True)
    k = build(dict(CFG, agents=6, world_opts=dict(OPTS, games="T-1,T-2")), 0, "dig")
    data = asyncio.run(run(k, MockBackend(0), 4, out, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"] if "snapshots" in data else json.loads((next(out.rglob("metrics.json"))).read_text())[-1]
    assert m["directions"] >= 2 and m["directions_joined"] >= 1, m
    ex = json.loads(next(out.rglob("run.json")).read_text())["world"]
    assert ex["directions"] and ex["direction_log"]
    shutil.rmtree(out, ignore_errors=True)


def test_on_arc_level_5_directions_follow_islands():
    """vc33 level 5 as in I8 (evidence, islands, teaming): a direction is a post like any other, so another island sees
    it, and can join it, once it opens."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_incentives import CFG as ARC  # noqa: E402
    start = str(Path(__file__).resolve().parents[1] / "tools" / "vc33_start_l5x.json")
    opts = {"games": "vc33", "explore": "1", "sync": "1", "carry": "1", "guide": "1", "hypotheses": "1", "substrate": "1",
            "board": "1", "publish_solutions": "1", "evidence": "1", "start": start, "islands": "1", "embargo": "3",
            "publish": "1/3", "group_cap": "5", "teaming": "1", "directions": "1"}
    k = build(dict(ARC, world="arc", agents=4, world_opts=opts), 0, "arc")
    k.total_ticks = 20
    w = k.env
    a, b, c, d = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    game = w.assigned[a]
    assert w._problem(game) == 5
    ok, out = act(k, a, "direction", {"op": "open", "text": "each boat must end beside the marker of its own colour"})
    assert ok, out
    did = out.split("opened ")[1].split()[0]
    assert f"YOUR DIRECTION {did} on {game.split('-')[0]} level {w._problem(game)}" in w.observe(k, a)
    ok, out = act(k, b, "direction", {"op": "join", "id": did})
    assert not ok and "not open to you yet" in out, out
    hidden = w.directions_view(k, b)  # b sees who is on it, not what it says
    assert f"{did} 1 agent ({a}), opened t1 by {a}: not open to you yet (opens by t4)" in hidden, hidden
    assert "marker of its own colour" not in hidden and "marker of its own colour" not in w.observe(k, b)
    k.tick = 4  # the embargo has passed: b sees the direction and may join it
    assert "marker of its own colour" in w.directions_view(k, b)
    ok, out = act(k, b, "direction", {"op": "join", "id": did})
    assert ok and "2 agents" in out, out
    # a group's purpose is called its purpose here, since "direction" names these
    from alphateam.primitives import system_prompt
    act(k, c, "connect", {"op": "form_group", "name": "pumps", "purpose": "map what each pump moves"})
    assert ", purpose: map what each pump moves" in k.membership(d) and ", direction:" not in k.membership(d)
    assert "Everyone sees a group's purpose" in system_prompt(k) and "purpose is its direction" not in system_prompt(k)


def test_change_credit_pays_tests_and_counterexamples_by_use_and_ideas_by_direction():
    k = build(dict(CFG, world_opts=dict(OPTS, change_credit="1")), 0, "dig")
    w = k.env
    a, b, c, d = sorted(k.agents, key=lambda x: int(x[1:]))
    assert "times one plus the number of other agents who used the piece" in w.brief()
    k.tick = 1
    # b posts a backed finding; c and d use it (they build on it); a then holds a direction alone
    rb = rid(act(k, b, "play", {"actions": ["1", "1"]})[1])
    fb = act(k, b, "plan", {"op": "finding", "finding": "1 moves the cursor left", "evidence": [rb]})[1].split(
        "posted finding ")[1].split()[0]
    for x in (c, d):
        act(k, x, "plan", {"op": "claim", "hypothesis": f"{x}: moving left then stamping clears it", "builds_on": [fb]})
    assert sorted(w.piece_users(k, fb)) == [c, d]
    # a's test that moves fb from seen to reproduced pays 0.3 x (1 + 2 users)
    ra = rid(act(k, a, "play", {"actions": ["1"]})[1])
    ok, out = act(k, a, "plan", {"op": "test", "entry": fb, "result": "holds", "evidence": [ra], "finding": "left again"})
    assert ok and "+0.9 credit for a test" in out, out
    # a counterexample to it pays the same weight, once per challenger and piece
    ok, out = act(k, a, "plan", {"op": "challenge", "entry": fb, "finding": "on my copy 1 did nothing", "evidence": [ra]})
    assert ok and "+0.9 credit: a counterexample to a piece 2 other agents used" in out, out
    ok, out = act(k, a, "plan", {"op": "challenge", "entry": fb, "finding": "again", "evidence": [ra]})
    assert "credit" not in out, out
    # a guide line that others used: a counterexample with receipts pays by its use too
    ok, out = act(k, b, "guide", {"op": "add", "line": "the star is at the right end", "evidence": [rb]})
    gid = w.guide_ids[w.assigned[b]]["the star is at the right end"]
    act(k, c, "plan", {"op": "claim", "hypothesis": f"walk right as {gid} says, then stamp"})
    ok, out = act(k, a, "guide", {"op": "challenge", "line": "the star is at the right end", "evidence": [ra],
                                  "why": "my star is on the left"})
    assert ok and "contested" in out and "+0.6 credit" in out, out
    # the first test of a new idea on a direction a holds alone pays 0.5 x (2 - 1/4)
    act(k, a, "direction", {"op": "open", "text": "the counter is a budget, not an exit"})
    act(k, a, "plan", {"op": "claim", "hypothesis": "the counter falls by one with every move"})
    ra2 = rid(act(k, a, "play", {"actions": ["2"]})[1])
    ok, out = act(k, a, "plan", {"op": "done", "finding": "it fell by one", "evidence": [ra2], "outcome": "works"})
    assert "+0.875 credit" in out, out
    s = w.score()
    assert s["change_payments"] == 3 and s["challenges_paid"] == 2 and s["change_credit"] == 2.4, s


def test_new_states_leave_out_the_counters_the_engine_found():
    """vc33's row 0 is the level's step bar: a frame new only there is not a new state (in every level-5 run, 56-65% of
    the states paid as new were new only in row 0)."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_incentives import CFG as ARC  # noqa: E402
    start = str(Path(__file__).resolve().parents[1] / "tools" / "vc33_start_l5x.json")
    opts = {"games": "vc33", "explore": "1", "sync": "1", "substrate": "1", "board": "1", "evidence": "1", "start": start}
    k = build(dict(ARC, world="arc", agents=2, world_opts=opts), 0, "arc")
    w = k.env
    a = sorted(k.agents, key=lambda x: int(x[1:]))[0]
    game = w.assigned[a]
    k.tick = 1
    assert (0, 0) in w.counters.lines[game]  # the start file's moves already showed row 0 changing alone
    w.counters.lines[game] = set()  # as if it had not: then a frame new only in row 0 counts as new
    w.mask[game] = np.zeros((64, 64), bool)  # past the learning steps, nothing volatile found that way
    base = np.zeros((64, 64), int)
    bar = base.copy()
    bar[0, 60:] = 4
    moved = bar.copy()
    moved[20:26, 30:36] = 4
    level = w.best[game]
    w._novel(k, a, game, base, base, level)
    n = w.novel_count[game]
    w._novel(k, a, game, bar, base, level)
    assert w.novel_count[game] == n + 1  # the counter is not known yet: new
    w.counters.lines[game].add((0, 0))  # the engine learns row 0 is a counter: states are keyed without it from now on
    bar2 = base.copy()
    bar2[0, 50:] = 4
    w._novel(k, a, game, bar2, bar, level)
    assert w.novel_count[game] == n + 2  # the first state keyed the new way
    bar3 = base.copy()
    bar3[0, 40:] = 4
    w._novel(k, a, game, bar3, bar2, level)
    assert w.novel_count[game] == n + 2  # new only in row 0: not a new state
    w._novel(k, a, game, moved, bar3, level)
    assert w.novel_count[game] == n + 3  # something else changed: new


def test_a_new_claim_says_which_open_plan_it_dropped_untested():
    """Half of all level-5 claims were dropped, untested, by their author's next claim, which said nothing about it."""
    k = build(CFG, 0, "dig")
    w = k.env
    a = sorted(k.agents, key=lambda x: int(x[1:]))[0]
    k.tick = 1
    first = act(k, a, "plan", {"op": "claim", "hypothesis": "the star is where 3 works"})[1].split("claimed ")[1].split()[0]
    ok, out = act(k, a, "plan", {"op": "claim", "hypothesis": "the counter is a budget"})
    assert ok and f"This dropped your open {first} (never tested)" in out and w.plans[first]["status"] == "dropped", out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
