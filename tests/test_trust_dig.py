"""Trust levels (trust=1) and the DiG-bench world, on the local fake server (fake=1): no network, no model calls."""
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import build, run  # noqa: E402

OPTS = {"fake": "1", "games": "T-1", "explore": "1", "sync": "1", "carry": "1", "guide": "1", "hypotheses": "1",
        "substrate": "1", "board": "1", "publish_solutions": "1", "trust": "1"}
CFG = {"condition": "self_org", "agents": 3, "ticks": 4, "actions": 4, "attention": 6000, "world": "dig",
       "world_opts": OPTS, "seed": 1, "structures": ["dig"], "episodes": 1, "population": 1}
rid = lambda out: out.split("receipt ")[1].split("]")[0]


def posted(msg: str) -> str:
    return msg.split("posted finding ")[1].split()[0]


def test_dig_world_plays_text_games_on_separate_instances():
    k = build(CFG, 0, "dig")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    assert w.games == ["t1"] and "own instance" in w.brief() and "Test before you trust" in w.brief()
    assert {w.sessions[(x, "t1")]["instance"] for x in (a, b, c)} == {f"{x}#{i}" for i, x in enumerate((a, b, c), 1)} \
        or len({w.sessions[(x, "t1")]["instance"] for x in (a, b, c)}) == 3
    k.tick = 1
    ok, out = asyncio.run(k.act(a, "look", {}))
    assert ok and "screen: '" in out and "legal actions: 1, 2, 3" in out, out
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["2", "9", "1"]}))
    assert ok and "9: not a legal action now" in out and "receipt R" in out, out
    r = w.receipts[rid(out)]
    assert r["instance"].startswith(a) and r["start"] == [] and [x["move"] for x in r["results"]] == ["2", "1"]
    # the next play starts where this one left off, and its receipt says so
    r2 = w.receipts[rid(asyncio.run(k.act(a, "play", {"actions": ["2"]}))[1])]
    assert r2["start"] == ["2", "1"] and "from the start of level 1 + 2 1" in w.receipt_line(r2["id"])
    # solve level 1 on b's instance by walking to the target and stamping it: the solution is published with its moves
    g = w.server.sessions[w.sessions[(b, "t1")]["sid"]][0]
    walk = ["2" if g.target > g.pos else "1"] * abs(g.target - g.pos) + ["3"]
    ok, out = asyncio.run(k.act(b, "play", {"actions": walk}))
    assert "LEVEL 1 COMPLETED" in out and w.best["t1"] == 1, out
    sol = [e for e in k.boards["global"].entries if "level 1 was solved by" in e.content][0]
    assert "moves: " + " ".join(walk) in sol.content and w.entry_moves(k, c, sol.id, {}) == walk
    # a stamp off the target costs a life: the engine posts it as a dead end, and the level starts over
    g = w.server.sessions[w.sessions[(c, "t1")]["sid"]][0]
    off = "1" if g.pos != 0 and g.pos - 1 != g.target else "2"
    ok, out = asyncio.run(k.act(c, "play", {"actions": [off, "3"] if g.pos + (1 if off == "2" else -1) != g.target else ["3"]}))
    assert "failed" in out and any(f["agent"] == "lab" and f["text"].startswith("LIFE LOST") for f in w.findings), out
    assert w.trace[(c, "t1")] == []
    # RESET starts a new instance, so it is refused while levels would be lost
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["RESET"]}))
    assert "refused" in out and w.state[(b, "t1")]["levels"] == 1, out
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["RESET"], "restart_game": True}))
    assert "a new instance" in out and w.state[(b, "t1")]["levels"] == 0 and w.instances == 4, out
    s = w.score()
    assert s["score"] == round(1 / 3, 4) and s["instances"] == 4 and s["tests"] == 0


def test_trust_levels_tests_and_what_a_conclusion_rests_on():
    k = build(dict(CFG, world_opts=dict(OPTS, games="T-2")), 0, "dig")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    # b posts a backed finding; c tests it on c's own instance and it holds: reproduced, and the test earns credit
    rb = rid(asyncio.run(k.act(b, "play", {"actions": ["1", "1"]}))[1])
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "finding", "finding": "1 counts up by one", "evidence": [rb]}))
    f1 = posted(msg)
    assert w.trust_of(k, f1)["level"] == "seen" and "[seen in 1 game instance]" in w.trust_tag(k, f1)
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "test", "entry": f1}))
    assert not ok and "cannot rerun recorded runs" in msg  # each instance has its own seed
    rc = rid(asyncio.run(k.act(c, "play", {"actions": ["1"]}))[1])
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "test", "entry": f1, "result": "holds", "evidence": [rc],
                                            "finding": "my count went up by one"}))
    assert ok and f"{f1} is now reproduced" in msg and "credit" in msg, msg
    test_entry = k.entry(msg.split("(")[1].split(")")[0])
    assert "not a repeat of its runs" in test_entry.content and f1 in test_entry.refs
    assert any("tested your" in n["text"] for n in k.agents[b].inbox)
    # a failure from another instance is counted, not ruled on; only one from the same starting state contests
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "test", "entry": f1, "result": "fails",
                                            "evidence": [rid(asyncio.run(k.act(a, "play", {"actions": ["2"]}))[1])]}))
    assert ok and w.trust_of(k, f1)["level"] == "reproduced" and "failed in 1 test under other conditions" in \
        w.trust_tag(k, f1), msg
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "test", "entry": f1, "result": "holds", "evidence": [rb]}))
    assert not ok and "it is yours" in msg
    # a note with no run behind it is said; a finding that builds on it shows that it rests on an unbacked piece
    note = k.post("global", a, "note", "the gate opens only when the count shows 3")
    assert w.trust_of(k, note.id)["level"] == "said"
    rb2 = rid(asyncio.run(k.act(b, "play", {"actions": ["2"]}))[1])
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "finding", "finding": f"so 2 is the gate's switch (after {note.id})",
                                            "evidence": [rb2]}))
    f2 = posted(msg)
    assert note.id in w.deps(k, f2) and f"rests on unbacked {note.id}" in w.trust_tag(k, f2), w.trust_tag(k, f2)
    # a standing counterexample contests f1, and everything resting on it within three steps is flagged
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "challenge", "entry": f1, "finding": "it went down",
                                            "evidence": [rid(asyncio.run(k.act(a, "play", {"actions": ["1"]}))[1])]}))
    assert ok and w.trust_of(k, f1)["level"] == "contested"
    rc2 = rid(asyncio.run(k.act(c, "play", {"actions": ["1"]}))[1])
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "finding", "finding": "counting to 3 first", "evidence": [rc2],
                                            "builds_on": [f1]}))
    f3 = posted(msg)
    assert f"rests on contested {f1}" in w.trust_tag(k, f3)
    # the tags reach everyone where entries are delivered and found, and the trace is exported
    k.tick = 2
    view = k.observe(a)
    assert f"{f3}" in view and "[" in view.split(f3)[1][:300] and "rests on contested" in view, view
    k.begin_turn(a)
    ok, hits = asyncio.run(k.act(a, "search", {"what": "posts", "query": "count"}))
    assert ok and "[said: no run behind it]" in hits and "rests on contested" in hits, hits
    ex = w.export()
    assert ex["trust"][f1]["level"] == "contested" and note.id in ex["trust"][f2]["rests_on"]
    assert len(ex["tests"][f1]) == 2 and ex["receipt_starts"][rb]["instance"].startswith(b)
    s = w.score()
    assert s["tests"] == 2 and s["tests_other_conditions"] == 2 and s["pieces_contested"] >= 1


def test_on_arc_the_engine_reruns_the_gate_and_a_click_from_another_state_does_not_refute_it():
    """vc33 level 4 (copies start there from the old run's published solutions): A20's three moves fire the gate."""
    root = Path(__file__).resolve().parents[1]
    k = build(dict(CFG, world="arc", agents=3, world_opts={
        "games": "vc33", "explore": "1", "sync": "1", "carry": "1", "hypotheses": "1", "board": "1", "trust": "1",
        "substrate": "1", "start": str(root / "tools" / "vc33_start_l4.json")}), 0, "arc")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    assert w.supports_replay and w.state[(a, w.games[0])]["levels"] == 3
    k.tick = 1
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["click(16,62)", "click(16,62)", "click(13,44)"]}))
    assert ok and "click(13,44): 57 cells changed" in out, out
    ra = rid(out)
    assert w.receipts[ra]["start"] == [] and [x["changed"] for x in w.receipts[ra]["results"]] == [99, 150, 57]
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "finding", "finding": "the bar at (13,44) moves the key across when the "
                                            "tanks on both sides are level", "evidence": [ra]}))
    f = posted(msg)
    # the engine reruns a's moves on a spare copy from where they started: reproduced, and the tester earns credit
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "test", "entry": f}))
    assert ok and f"{f} is now reproduced" in msg and "gave the recorded result again" in msg, msg
    # c clicks the same cell from the level's start: nothing happens, and the failure is from another state
    rc = rid(asyncio.run(k.act(c, "play", {"actions": ["click(13,44)"]}))[1])
    assert w.receipts[rc]["results"][0]["changed"] <= 2
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "test", "entry": f, "result": "fails", "evidence": [rc],
                                            "finding": "clicked (13,44), nothing moved"}))
    assert ok and w.trust_of(k, f)["level"] == "reproduced", msg
    assert "not a repeat of its runs" in k.entry(msg.split("(")[1].split(")")[0]).content
    assert "failed in 1 test under other conditions" in w.trust_tag(k, f)
    # from the same state the moves give the same result, and the receipt line says which state that is
    asyncio.run(k.act(c, "play", {"actions": ["RESET"]}))
    rc2 = rid(asyncio.run(k.act(c, "play", {"actions": ["click(16,62)", "click(16,62)", "click(13,44)"]}))[1])
    assert w.start_key(rc2) == w.start_key(ra) and "57" in str(w.receipts[rc2]["results"][-1]["changed"])
    assert "from the start of level 4" in w.receipt_line(rc2)
    # a record the game does not reproduce is caught by the rerun and contests what rests on it
    w.receipts[rc2]["results"][-1]["changed"] = 3
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "test", "entry": rc2}))
    assert ok and f"{rc2} is now contested" in msg and "the record says 3 cells changed" in msg, msg


def test_without_trust_nothing_changes():
    k = build(dict(CFG, world_opts={x: v for x, v in OPTS.items() if x != "trust"}), 0, "dig")
    w = k.env
    a = sorted(k.agents)[0]
    k.tick = 1
    r = rid(asyncio.run(k.act(a, "play", {"actions": ["1"]}))[1])
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "finding", "finding": "1 moves left", "evidence": [r]}))
    assert ok and w.trust_tag(k, posted(msg)) == "" and "Test before you trust" not in w.brief()
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "test", "entry": posted(msg)}))
    assert not ok
    assert "test" not in [t for t in w.TOOLS if t["name"] == "plan"][0]["parameters"]["properties"]["op"]["enum"]


def test_a_mock_society_runs_on_the_fake_server(tmp_path=None):
    out = Path(tmp_path or Path(__file__).resolve().parent / "_tmp_dig_run")
    shutil.rmtree(out, ignore_errors=True)
    k = build(dict(CFG, agents=4, world_opts=dict(OPTS, games="T-1,T-2")), 0, "dig")
    data = asyncio.run(run(k, MockBackend(0), 4, out, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"] if "snapshots" in data else json.loads((next(out.rglob("metrics.json"))).read_text())[-1]
    assert m["steps"] > 0 and "tests" in m
    report = next(out.rglob("report.md")).read_text()
    assert "DiG-bench games, one instance per agent" in report
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
