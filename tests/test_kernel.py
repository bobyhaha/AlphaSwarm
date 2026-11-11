"""Kernel tests: permissions, structure, governance, blocks, custom actions, and the Claude tool loop (stubbed)."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from society.actions import tool_schemas  # noqa: E402
from society.claude_backend import ClaudeBackend  # noqa: E402
from society.runner import build_world  # noqa: E402
from society.world import World  # noqa: E402

PEOPLE = [{"name": n, "persona": ""} for n in ["Avery", "Blake", "Casey", "Devon", "Emery"]]


def world() -> World:
    w = World("Test mission", PEOPLE)

    async def summarizer(world, aid, text, focus):
        return "S:" + text[-40:]

    w.summarizer = summarizer
    return w


def act(w, _aid, _action, **args):
    return asyncio.run(w.act(_aid, _action, args))


def test_boards_and_sharing():
    w = world()
    assert act(w, "A1", "write", board="private", content="secret")[0]
    assert not act(w, "A2", "read", board="A1")[0]
    assert act(w, "A1", "share", **{"with": ["A2"], "mode": "read"})[0]
    assert act(w, "A2", "read", board="A1")[0]
    assert not act(w, "A2", "write", board="A1", content="x")[0]
    assert any("shared private:A1" in n["text"] for n in w.agents["A2"].inbox)


def test_teams_subteams_and_superagent_access():
    w = world()
    assert act(w, "A1", "connect", op="form_team", name="Top", charter="c")[0]
    assert act(w, "A2", "connect", op="join", target="T1")[0]
    ok, msg = act(w, "A2", "connect", op="form_team", name="Sub", charter="c", parent="T1")
    assert ok, msg
    assert w.teams["T2"].parent == "T1" and w.agents["A2"].team == "T2"
    # a sub-team member is inside the enclosing superagent and can use its board
    assert act(w, "A2", "write", board="T1", content="up")[0]
    # but the top team's member cannot read the sub-team board unless shared
    assert not act(w, "A1", "read", board="T2")[0]
    assert w.subtree_agents("T1") == ["A1", "A2"]
    # cannot nest a team under its own sub-team
    assert not act(w, "A1", "connect", op="nest", target="T1", parent="T2")[0]


def test_empty_teams_dissolve_and_blocks_release():
    w = world()
    act(w, "A1", "connect", op="form_team", name="Solo", charter="c")
    act(w, "A1", "plan", op="decompose", block="M0", blocks=[{"title": "B", "spec": "s", "owner": "T1"}])
    assert w.blocks["B1"].owner == "T1"
    assert act(w, "A1", "connect", op="leave")[0]
    assert w.teams["T1"].dissolved == 0
    assert w.blocks["B1"].owner is None and w.blocks["B1"].status == "open"


def test_majority_rule_opens_motion_and_vote_executes():
    w = world()
    act(w, "A1", "connect", op="form_team", name="T", charter="c")
    for a in ("A2", "A3"):
        act(w, a, "connect", op="join", target="T1")
    assert act(w, "A1", "connect", op="govern", target="T1", rule="majority")[0]
    ok, msg = act(w, "A2", "connect", op="recharter", target="T1", name="Renamed")
    assert ok and "motion V1" in msg and w.teams["T1"].name == "T"
    ok, msg = act(w, "A3", "connect", op="vote", motion="V1", vote="yes")
    assert ok and "passed" in msg, msg
    assert w.teams["T1"].name == "Renamed"


def test_rep_rule_and_closed_admission():
    w = world()
    act(w, "A1", "connect", op="form_team", name="T", charter="c")
    act(w, "A2", "connect", op="join", target="T1")
    assert act(w, "A1", "connect", op="govern", target="T1", rule="rep", reps=["A1"], open=False)[0]
    ok, msg = act(w, "A3", "connect", op="join", target="T1")
    assert ok and "requested admission" in msg and w.agents["A3"].team is None
    # a non-rep's yes does not pass it; the rep's does
    assert "recorded" in act(w, "A2", "connect", op="vote", motion="V1", vote="yes")[1]
    ok, msg = act(w, "A1", "connect", op="vote", motion="V1", vote="yes")
    assert ok and w.agents["A3"].team == "T1", msg
    # outsiders cannot vote
    act(w, "A4", "connect", op="join", target="T1")
    assert not act(w, "A5", "connect", op="vote", motion="V2", vote="yes")[0]


def test_merge_and_split():
    w = world()
    act(w, "A1", "connect", op="form_team", name="X", charter="c")
    act(w, "A2", "connect", op="form_team", name="Y", charter="c")
    act(w, "A3", "connect", op="join", target="T2")
    assert act(w, "A2", "connect", op="merge", target="T2", into="T1")[0]
    assert w.teams["T2"].dissolved is not None and set(w.direct_members("T1")) == {"A1", "A2", "A3"}
    ok, msg = act(w, "A1", "connect", op="split", target="T1", members=["A2", "A3"], name="Z", charter="c")
    assert ok, msg
    assert w.teams["T3"].parent == "T1" and w.agents["A2"].team == "T3"


def test_blocks_verify_needs_someone_else():
    w = world()
    act(w, "A1", "plan", op="decompose", blocks=[{"title": "Theory", "spec": "s"}])
    assert act(w, "A1", "plan", op="claim", block="B1")[0]
    assert not act(w, "A2", "plan", op="claim", block="B1")[0]
    assert act(w, "A1", "plan", op="submit", block="B1", deliverable="d")[0]
    assert not act(w, "A1", "plan", op="verify", block="B1", verdict="accept")[0]
    assert act(w, "A2", "plan", op="verify", block="B1", verdict="accept")[0]
    assert w.blocks["B1"].status == "done"


def test_team_submission_follows_team_rule():
    w = world()
    act(w, "A1", "connect", op="form_team", name="T", charter="c")
    act(w, "A2", "connect", op="join", target="T1")
    act(w, "A1", "connect", op="govern", target="T1", rule="consensus")
    act(w, "A1", "plan", op="decompose", blocks=[{"title": "B", "spec": "s", "owner": "T1"}])
    ok, msg = act(w, "A1", "plan", op="submit", block="B1", deliverable="d")
    assert ok and "motion" in msg and w.blocks["B1"].status == "claimed"
    act(w, "A2", "connect", op="vote", motion="V1", vote="yes")
    assert w.blocks["B1"].status == "submitted"


def test_custom_actions_and_conditions():
    w = world()
    calls = []

    async def runner(world, aid, ca, args):
        calls.append((aid, ca.name, args))
        return "ran"

    w.custom_runner = runner
    assert act(w, "A1", "design_action", op="define", name="sync_up", description="d", procedure="p", params=["focus"])[0]
    assert not act(w, "A2", "x_sync_up", focus="f")[0]  # must adopt first
    assert act(w, "A2", "design_action", op="adopt", name="sync_up")[0]
    assert act(w, "A2", "x_sync_up", focus="f")[0] and calls == [("A2", "sync_up", {"focus": "f"})]
    assert "x_sync_up" in [t["name"] for t in tool_schemas(w, "A2")]
    flat = build_world({"mission": "m", "people": PEOPLE}, "flat", 4, False)
    assert not asyncio.run(flat.act("A1", "connect", {"op": "form_team", "name": "n", "charter": "c"}))[0]
    connect = next(t for t in tool_schemas(flat, "A1") if t["name"] == "connect")
    assert set(connect["input_schema"]["properties"]["op"]["enum"]) == {"link", "unlink", "vote"}
    fixed = build_world({"mission": "m", "people": PEOPLE}, "fixed", 4, False, team_size=2)
    assert fixed.metrics()["depth"] == 2 and fixed.metrics()["free"] == 0
    nodesign = build_world({"mission": "m", "people": PEOPLE}, "no_design", 4, False)
    assert "design_action" not in [t["name"] for t in tool_schemas(nodesign, "A1")]


class FakeMessages:
    """Scripted stand-in for client.beta.messages: replays responses and records requests."""

    def __init__(self, script):
        self.script, self.requests = list(script), []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def response(*blocks, stop="tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                                 cache_read_input_tokens=0, cache_creation_input_tokens=0))


def tool_use(i, _tool, **inp):
    return SimpleNamespace(type="tool_use", id=f"tu{i}", name=_tool, input=inp)


def text(t):
    return SimpleNamespace(type="text", text=t)


def test_claude_turn_loop_with_stub():
    w = world()
    w.max_actions = 2
    msgs = FakeMessages([
        response(tool_use(1, "connect", op="form_team", name="Interfaces", charter="c"),
                 tool_use(2, "write", board="team", content="hello")),
        response(text("Next I will invite Blake."), stop="end_turn"),
    ])
    backend = ClaudeBackend(client=SimpleNamespace(beta=SimpleNamespace(messages=msgs)))
    asyncio.run(backend.turn(w, "A1"))
    assert w.agents["A1"].team == "T1"
    first, closing = msgs.requests
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert "tool_choice" not in first and closing["tool_choice"] == {"type": "none"}
    results = closing["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"] and not any(r.get("is_error") for r in results)
    assert w.events[-1]["op"] == "intent" and "invite Blake" in w.events[-1]["msg"]
    assert w.ledger["A1"]["deliberation"] == 240 and w.ledger["A1"]["communication"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
