"""The design world: optimum, score, tools through the kernel, thrash, org charts, and a mock run."""
import asyncio
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.design_world import DesignWorld  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.metrics import measure  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import build, run  # noqa: E402

TINY = [{"blocks": 2, "supers": 1, "private": 1, "interface": 1, "top": 1, "top_per_block": 1},
        {"blocks": 2, "supers": 2, "private": 1, "interface": 1, "top": 1, "top_per_block": 1},
        {"blocks": 3, "supers": 1, "private": 1, "interface": 2, "top": 1, "top_per_block": 1}]


def brute_force(w: DesignWorld) -> float:
    grid = np.array(list(itertools.product(range(11), repeat=w.D)), dtype=float)
    costs = np.einsum("ij,jk,ik->i", grid, w.H, grid) - 2 * grid @ w.g + w.c0
    return float(costs.min())


def world(n=8, seed=0, **opts):
    k = Kernel(n, max_actions=6)
    k.env = DesignWorld(n, seed=seed, opts=opts)
    k.env.setup(k)
    k.tick = 1
    return k


def test_optimum_matches_brute_force():
    for opts in TINY:
        for seed in range(4):
            w = DesignWorld(4, seed=seed, opts=opts)
            assert w.D <= 6
            assert abs(w.c_opt - brute_force(w)) < 1e-6, (opts, seed, w.c_opt, brute_force(w))
            assert abs(w.cost(w.x_opt) - sum(w.block_cost(b, w.x_opt) for b in range(len(w.blocks)))) < 1e-6


def test_score_scale_and_level_gaps():
    w = DesignWorld(50, seed=1)
    assert w.normalized(w.x_start) == 0.0 and w.normalized(w.x_opt) == 1.0
    worse = w.x_start.copy()
    worse[:] = np.where(w.x_opt > 5, 0, 10)  # far from the optimum
    assert w.cost(worse) > w.c_start and w.normalized(worse) < 0 and w.normalized(worse, clamp=True) == 0.0
    assert all(v == 0 for v in w.level_gaps(w.x_opt).values())
    assert all(v > 0 for v in w.level_gaps(w.x_start).values())
    per_block = [sum(bs[0] == b for bs in w.instruments.values()) for b in range(len(w.blocks))]
    assert len(w.blocks) == 14 and w.n_super == 4 and min(per_block) == 3 and max(per_block) == 4
    names = [m["name"] for m in w.modules]
    assert sorted(int(x[1:]) for x in names) == list(range(1, 15)) and names != sorted(names, key=lambda x: int(x[1:]))


def test_simulate_set_and_budget_through_kernel():
    k = world()
    w = k.env
    aid = "A1"
    b = w.instruments[aid][0]
    mine = [w.name[i] for i in w.blocks[b]["idx"]]
    other = next(nm for nm in w.name if nm not in mine)
    before = w.x.copy()
    ok, msg = asyncio.run(k.act(aid, "simulate", {"changes": {mine[0]: 10, mine[1]: 0}}))
    assert ok and "cost now" in msg and "with" in msg, msg
    assert np.array_equal(before, w.x)  # hypothetical only
    assert any("Simulation of" in e.content for e in k.boards[f"private:{aid}"].entries)
    assert asyncio.run(k.act(aid, "simulate", {}))[0] and asyncio.run(k.act(aid, "simulate", {}))[0]
    ok, msg = asyncio.run(k.act(aid, "simulate", {}))
    assert not ok and "simulations this tick" in msg
    k.tick = 2
    assert asyncio.run(k.act(aid, "simulate", {"changes": {"k999": 1}}))[1].startswith("refused: no knob")
    assert not asyncio.run(k.act(aid, "simulate", {"block": "B999"}))[0]
    assert not asyncio.run(k.act(aid, "simulate", {"changes": {mine[0]: 11}}))[0]
    # set_knob on any knob, reversal counting, notification of the previous setter
    i = w.index[other]
    start = int(w.x[i])
    up, down = (start + 1, start) if start < 10 else (start - 1, start)
    assert asyncio.run(k.act(aid, "set_knob", {"knob": other, "value": up}))[0] and w.x[i] == up
    assert asyncio.run(k.act("A2", "set_knob", {"knob": other, "value": down}))[0]
    assert w.sets[-1]["reversal"] and not w.sets[0]["reversal"]
    assert any("changed" in str(item) for item in k.agents[aid].inbox)  # the previous setter hears about it
    ok, msg = asyncio.run(k.act("A3", "view_design", {"knobs": [other]}))
    assert ok and f"{other} = {down}" in msg and "A2" in msg
    s = measure(k)
    assert s["knob_sets"] == 2 and s["score"] is not None and "gap_interface" in s


def test_shared_knob_metrics_and_shift():
    k = world(n=12, seed=3, shocks=["2:shift"])
    w = k.env
    shared = next(i for i in range(w.D) if w.level[i] == "interface")
    users = w.users[shared]
    insider = next(a for a, bs in w.instruments.items() if bs[0] in users)
    outsider = next((a for a, bs in w.instruments.items() if bs[0] not in users), None)
    assert asyncio.run(k.act(insider, "set_knob", {"knob": w.name[shared], "value": 0 if w.x[shared] else 1}))[0]
    if outsider:
        assert asyncio.run(k.act(outsider, "set_knob", {"knob": w.name[shared], "value": 5 if w.x[shared] != 5 else 6}))[0]
    sc = w.score()
    assert sc["shared_sets"] == (2 if outsider else 1) and sc["shared_informed"] == (0.5 if outsider else 1.0)
    old_opt = w.c_opt
    k.tick = 2
    w.on_tick_start(k)
    assert w.shocks and w.shocks[0]["kind"] == "shift" and w.c_opt != old_opt
    assert w.normalized(w.x_start) == 0.0


def test_oracle_prescription_and_single():
    cfg = {"condition": "oracle", "agents": 50, "actions": 4, "attention": 3000, "world": "design", "world_opts": {}}
    k = build(cfg, 0, "design")
    w = k.env
    teams = [g for g in k.active_groups() if g.name.startswith("Module team")]
    assert len(teams) == len(w.blocks)
    for g in teams:
        assert len({w.instruments[a][0] for a in g.members}) == 1 and "Settle the knobs of block" in g.purpose
    divisions = [g for g in k.active_groups() if g.name.startswith("Division")]
    assert len(divisions) == w.n_super and any(g.name == "Program" for g in k.active_groups())
    assert max(k.depth(g.id) for g in k.active_groups()) == 3
    single = build(dict(cfg, condition="single"), 0, "design")
    assert len(single.env.instruments["A1"]) == len(single.env.blocks) == len(w.blocks)
    assert single.env.sim_budget() == 3 * 50
    ok, msg = asyncio.run(single.act("A1", "simulate", {}))
    assert not ok and "which block" in msg
    single.tick = 1
    assert asyncio.run(single.act("A1", "simulate", {"block": single.env.block_label[3]}))[0]


def test_mock_run_writes_report(tmp=Path("/tmp/alphateam_design_test")):
    cfg = {"condition": "self_org", "agents": 20, "actions": 4, "attention": 3000, "world": "design", "world_opts": {}}
    k = build(cfg, 0, "design")
    data = asyncio.run(run(k, MockBackend(0), 4, tmp, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"]
    assert m["knob_sets"] > 0 and m["shared_reversals"] >= 0 and data["world"]["kind"] == "design"
    assert any(e.content.startswith("Design review at t3") for e in k.boards["global"].entries)
    text = (tmp / "report.md").read_text()
    assert "design world" in text and "## Shared knobs" in text


def test_decision_rights():
    from alphateam.runner import build
    cfg = {"condition": "oracle", "agents": 50, "actions": 4, "attention": 3000, "world": "design",
           "world_opts": {"authority": "1"}}
    k = build(cfg, 0, "design")
    w = k.env
    assert len(w.rights) == w.D and all(h in k.groups for h in w.rights.values())
    i = next(j for j in range(w.D) if str(w.level[j]) == "interface")
    outsider = next(a for a in k.agents if not k.can_act_for(a, w.rights[i]))
    ok, msg = asyncio.run(k.act(outsider, "set_knob", {"knob": w.name[i], "value": 3}))
    assert not ok and "held by" in msg
    insider = next(a for a in k.agents if k.can_act_for(a, w.rights[i]))
    assert asyncio.run(k.act(insider, "set_knob", {"knob": w.name[i], "value": 3}))[0]
    cfg = {"condition": "self_org", "agents": 50, "actions": 4, "attention": 3000, "world": "design",
           "world_opts": {"ownership": "1"}}
    k = build(cfg, 0, "design")
    w = k.env
    assert "own_knob" in w.tool_names and not w.rights
    assert asyncio.run(k.act("A1", "own_knob", {"knob": "k5"}))[0]
    ok, msg = asyncio.run(k.act("A2", "set_knob", {"knob": "k5", "value": 2}))
    assert not ok and "held by A1" in msg
    assert asyncio.run(k.act("A1", "set_knob", {"knob": "k5", "value": 2}))[0]
    assert asyncio.run(k.act("A1", "own_knob", {"knob": "k5", "op": "release"}))[0]
    assert asyncio.run(k.act("A2", "set_knob", {"knob": "k5", "value": 4}))[0]
    cfg["world_opts"] = {"ownership": "group"}
    k = build(cfg, 0, "design")
    ok, msg = asyncio.run(k.act("A1", "own_knob", {"knob": "k5"}))
    assert not ok and "only groups" in msg
    assert asyncio.run(k.act("A1", "connect", {"op": "form_group", "name": "k5 owners", "purpose": "hold k5"}))[0]
    gid = next(iter(k.groups))
    assert asyncio.run(k.act("A1", "own_knob", {"knob": "k5", "as": gid}))[0]
    assert k.env.rights[k.env.index["k5"]] == gid


def test_deeper_world_optimum():
    for seed in range(3):
        w = DesignWorld(4, seed=seed, opts={"blocks": 4, "supers": 1, "subs": 2, "private": 0, "interface": 1, "mid": 1,
                                            "top": 1, "top_per_block": 1})
        assert "mid" in set(w.level.tolist()) and abs(w.c_opt - brute_force(w)) < 1e-6
    w = DesignWorld(50, seed=0, opts={"blocks": 16, "supers": 4, "subs": 2})
    assert w.D == 107 and all(len(w.users[i]) == 2 for i in range(w.D) if w.level[i] == "mid")


def test_the_three_worlds_differ_only_in_who_shares_what():
    coupled, _ = None, None
    a = DesignWorld(24, seed=1, opts={"blocks": 8, "supers": 2})
    b = DesignWorld(24, seed=1, opts={"blocks": 8, "supers": 2, "dependency": "independent"})
    c = DesignWorld(24, seed=1, opts={"blocks": 8, "supers": 2, "dependency": "moving"})
    assert a.D == b.D == c.D and len(a.blocks) == len(b.blocks)
    assert all(len(v) == 1 for v in b.users.values()) and any(len(v) > 1 for v in a.users.values())
    assert set(c.shock_plan) == {12, 13}


def test_rewire_changes_who_shares_a_knob_and_the_best_design():
    k = Kernel(24, max_actions=6)
    w = DesignWorld(24, seed=2, opts={"blocks": 8, "supers": 2, "dependency": "moving", "rewire_at": 2, "leave_for": 2})
    k.env = w
    w.setup(k)
    before = {i: sorted(v) for i, v in w.users.items()}
    opt = w.c_opt
    k.tick = 2
    w.on_tick_start(k)
    assert {i: sorted(v) for i, v in w.users.items()} != before and w.c_opt != opt
    assert all(len(set(blk["idx"].tolist())) == len(blk["idx"]) for blk in w.blocks)
    assert w.shocks[0]["kind"] == "rewire" and w.normalized(w.x_start) == 0
    w.rights[0] = "A3"
    k.tick = 3
    w.on_tick_start(k)
    assert not k.agents["A3"].active and 0 not in w.rights and w.shocks[1]["agent"] == "A3"
    k.tick = 5
    w.on_tick_start(k)
    assert k.agents["A3"].active


def test_a_knob_can_be_handed_over_and_comes_back():
    k = Kernel(8, max_actions=6)
    w = DesignWorld(8, seed=0, opts={"blocks": 2, "supers": 1, "ownership": "1"})
    k.env = w
    w.setup(k)
    k.tick = 1
    k.begin_turn("A1")
    act = lambda aid, tool, **a: asyncio.run(k.act(aid, tool, a))
    knob = w.name[0]
    assert act("A1", "own_knob", knob=knob)[0]
    assert act("A2", "delegate", op="post", title="grab", to="A3", scope=[knob])[0]
    t1 = sorted(k.tasks)[-1]
    ok, msg = act("A3", "delegate", op="take", task=t1)
    assert not ok and "only its holder" in msg
    assert act("A1", "delegate", op="post", title="settle it", to="A3", scope=[knob], until=5)[0]
    t2 = sorted(k.tasks, key=lambda x: int(x[1:]))[-1]
    assert act("A3", "delegate", op="take", task=t2)[0]
    assert w.holder(0) == "A3"
    ok, msg = act("A1", "set_knob", knob=knob, value=5)
    assert not ok and "held by A3" in msg
    assert act("A3", "set_knob", knob=knob, value=5)[0]
    assert act("A1", "delegate", op="drop", task=t2)[0]
    assert w.holder(0) == "A1"
    assert k.delegation_measures()["taken"] == 1 and k.delegation_measures()["with_scope"] == 2


def test_posts_can_be_searched_once_agents_have_credit():
    k = world()
    k.substrate = True
    k.post("global", "A2", "data", "Simulation of B1: lowering k4 helps", notify=False)
    k.env.credit["A2"][0] += 0.2  # this world keeps [credit, changes]; a search used to be refused on it
    k.env.credit["A2"][1] += 1
    k.agents["A1"].attention = 2000
    ok, msg = asyncio.run(k.act("A1", "search", {"what": "posts", "query": "lowering k4", "limit": "10"}))
    assert ok and "lowering k4" in msg, msg


def test_a_review_can_say_which_shared_knobs_keep_moving():
    k = world(contest="1", ownership="1")
    w = k.env
    shared = next(i for i in range(w.D) if str(w.level[i]) != "private")
    start = int(w.x[shared])
    for t, (who, step) in enumerate([("A1", 3), ("A2", 6), ("A3", 1)], start=1):
        k.tick = t
        k.begin_turn(who)
        assert asyncio.run(k.act(who, "set_knob", {"knob": w.name[shared], "value": (start + step) % 11}))[0]
    line = w._contest_line(k)
    assert w.name[shared] in line and "moved 3 times" in line and "A1, A2, A3" in line and "used by" in line, line
    w.contest_facts = False
    assert w._contest_line(k) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
