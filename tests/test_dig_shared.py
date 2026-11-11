"""Shared instances on DiG-bench (copies=shared), on the local fake server (fake=1): no network, no model calls."""
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import build, run  # noqa: E402

OPTS = {"fake": "1", "games": "T-1", "explore": "1", "sync": "1", "carry": "1", "guide": "1", "hypotheses": "1",
        "substrate": "1", "board": "1", "publish_solutions": "1", "trust": "1", "copies": "shared"}
CFG = {"condition": "self_org", "agents": 3, "ticks": 4, "actions": 4, "attention": 6000, "world": "dig",
       "world_opts": OPTS, "seed": 1, "structures": ["dig"], "episodes": 1, "population": 1}
rid = lambda out: out.split("receipt ")[1].split("]")[0]


def test_everyone_on_a_game_plays_one_instance_and_reconciles_before_acting():
    k = build(CFG, 0, "dig")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    assert w.shared and "one shared instance" in w.brief() and w.instances == 1
    assert w.state[(a, "t1")] is w.state[("shared", "t1")] is w.state[(c, "t1")]
    play = next(t for t in w.TOOLS if t["name"] == "play")
    assert "shared instance" in play["description"] and "own instance" not in play["description"]
    k.tick = 1
    for x in (a, b, c):
        assert "the shared instance, with" in w.observe(k, x)
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["2"]}))
    assert ok and "the shared instance" in out and "receipt R" in out, out
    assert w.receipts[rid(out)]["instance"] == "shared#1"
    # b last saw the instance before a moved: its play is not sent, no step is spent, and it is told who did what
    idx = w.sessions[("shared", "t1")]["idx"]
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["1"]}))
    assert "NOT SENT" in out and f"{a} t1: 2" in out, out
    assert w.sessions[("shared", "t1")]["idx"] == idx and w.steps[(b, 1)] == 0 and w.stale_refusals == 1
    # having seen the change, its next play goes through, from where a left the instance
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["1"]}))
    assert "NOT SENT" not in out and w.sessions[("shared", "t1")]["idx"] == idx + 1, out
    assert w.receipts[rid(out)]["start"] == ["2"]
    ok, out = asyncio.run(k.act(c, "look", {}))
    assert f"{a}: 2" in out and f"{b}: 1" in out and "by anyone" in out, out
    # c clears level 1 for everyone on the game
    g = w.server.sessions[w.sessions[("shared", "t1")]["sid"]][0]
    walk = ["2" if g.target > g.pos else "1"] * abs(g.target - g.pos) + ["3"]
    ok, out = asyncio.run(k.act(c, "play", {"actions": walk}))
    assert "LEVEL 1 COMPLETED" in out and w.best["t1"] == 1 and w.state[(a, "t1")]["levels"] == 1, out
    assert any(f["agent"] == c and f["level"] == 1 for f in w.firsts)
    sol = [e for e in k.boards["global"].entries if "level 1 was solved by" in e.content][0]
    assert "on the shared instance" in sol.content
    # a RESET would lose the level for everyone, so it is refused unless asked for
    w.observe(k, a)
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["RESET"]}))
    assert "refused" in out and "everyone on t1" in out and w.state[("shared", "t1")]["levels"] == 1, out
    # plays sent at the same time are applied one at a time, each checked against what its agent last saw
    w.observe(k, a)
    w.observe(k, b)

    async def both():
        return await asyncio.gather(k.act(a, "play", {"actions": ["1"]}), k.act(b, "play", {"actions": ["2"]}))
    (_, oa), (_, ob) = asyncio.run(both())
    assert sum("NOT SENT" in o for o in (oa, ob)) == 1, (oa, ob)
    s = w.score()
    assert s["instances"] == 1 and s["stale_refusals"] == 2
    ex = w.export()
    assert ex["copies_mode"] == "shared" and [m["agent"] for m in ex["moves_on"]["t1"][:2]] == [a, b]


def test_own_instances_are_unchanged_without_the_option():
    k = build(dict(CFG, world_opts={x: v for x, v in OPTS.items() if x != "copies"}), 0, "dig")
    w = k.env
    assert not w.shared and w.instances == 3 and "own instance" in w.brief() and "NOT SENT" not in w.brief()
    a, b = sorted(k.agents, key=lambda x: int(x[1:]))[:2]
    k.tick = 1
    w.observe(k, a)
    w.observe(k, b)
    asyncio.run(k.act(a, "play", {"actions": ["2"]}))
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["1"]}))
    assert "NOT SENT" not in out and "your instance" in out and w.export()["copies_mode"] == "own"


def test_holding_the_shared_instance_is_offered_and_passes_in_turn():
    k = build(dict(CFG, world_opts=dict(OPTS, lease="1")), 0, "dig")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    assert "hold" in w.tool_names and any(t["name"] == "hold" for t in w.TOOLS) and "never required" in w.brief()
    k.tick = 1
    assert "nobody holds" in w.observe(k, b)
    ok, out = asyncio.run(k.act(a, "hold", {"op": "take", "ticks": 2, "why": "map the keys"}))
    assert ok and "you hold the shared instance of t1 until the end of t2" in out, out
    # while a holds it, b's play is not sent, and b is told who holds it, until when and why
    w.observe(k, b)
    idx = w.sessions[("shared", "t1")]["idx"]
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["1"]}))
    assert "NOT SENT" in out and f"{a} holds" in out and "map the keys" in out and w.lease_refusals == 1, out
    assert w.sessions[("shared", "t1")]["idx"] == idx and w.steps[(b, 1)] == 0
    w.observe(k, a)
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["2"]}))
    assert "NOT SENT" not in out, out
    # b and c queue; a cannot extend its hold while others wait; c cannot take it
    assert "position 1" in asyncio.run(k.act(b, "hold", {"op": "request", "why": "test 3"}))[1]
    assert "position 2" in asyncio.run(k.act(c, "hold", {"op": "request", "ticks": 1}))[1]
    assert not asyncio.run(k.act(a, "hold", {"op": "take", "ticks": 2}))[0]
    assert not asyncio.run(k.act(c, "hold", {"op": "take"}))[0]
    assert any("asks for the shared instance" in m["text"] for m in k.agents[a].inbox)
    # a releases: b, first in the queue, holds it now and is told
    ok, out = asyncio.run(k.act(a, "hold", {"op": "release"}))
    assert ok and w.leases["t1"]["holder"] == b and any("you now hold" in m["text"] for m in k.agents[b].inbox), out
    # b passes it to c directly
    ok, out = asyncio.run(k.act(b, "hold", {"op": "pass", "to": c}))
    assert ok and w.leases["t1"]["holder"] == c and not w.lease_queue["t1"], out
    # a hold runs out at the end of its last tick
    until = w.leases["t1"]["until"]
    k.tick = until + 1
    w.on_tick_start(k)
    assert "t1" not in w.leases and "nobody holds" in w.observe(k, a)
    # a hold for a group lets its members move the instance and no one else
    ok, msg = asyncio.run(k.act(c, "connect", {"op": "form_group", "name": "t1 movers"}))
    gid = msg.split()[1]
    assert not asyncio.run(k.act(a, "hold", {"op": "take", "for": gid}))[0]
    ok, out = asyncio.run(k.act(c, "hold", {"op": "take", "for": gid}))
    assert ok and f"for its group {gid}" in out, out
    w.observe(k, c)
    assert "NOT SENT" not in asyncio.run(k.act(c, "play", {"actions": ["1"]}))[1]
    w.observe(k, a)
    assert "NOT SENT" in asyncio.run(k.act(a, "play", {"actions": ["1"]}))[1]
    s, ex = w.score(), w.export()
    assert s["lease_refusals"] == 2 and s["holds"] >= 4 and [x["how"] for x in ex["leases"]][:2] == ["took", "released"]


def test_a_mock_run_with_shared_instances():
    out = Path(__file__).resolve().parent / "_tmp_dig_shared"
    shutil.rmtree(out, ignore_errors=True)
    k = build(dict(CFG, agents=4, world_opts=dict(OPTS, games="T-1,T-2", lease="1")), 0, "dig")
    data = asyncio.run(run(k, MockBackend(0), 4, out, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"] if "snapshots" in data else json.loads((next(out.rglob("metrics.json"))).read_text())[-1]
    assert m["steps"] > 0 and m["instances"] >= 2 and "lease_refusals" in m
    assert "one shared instance per game" in next(out.rglob("report.md")).read_text()
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
