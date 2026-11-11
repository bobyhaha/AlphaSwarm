"""AlphaTeam tests: kernel primitives, the hidden-law world, metrics, conditions, culture, and both LLM adapters (stubbed)."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphateam.hidden_laws import HiddenLaws  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.llm import AnthropicProvider, LLMBackend, OpenAICompatible  # noqa: E402
from alphateam.metrics import measure, mirroring_index  # noqa: E402
from alphateam.primitives import tool_schemas  # noqa: E402
from alphateam.runner import build, culture, inherit  # noqa: E402

CFG = {"condition": "self_org", "agents": 24, "actions": 4, "attention": 3000, "instruments": 3, "redundancy": None,
       "experiments": 30, "seed_runs": 3, "noise": 0.3, "validate_every": 3, "shocks": []}


def kernel(n=6, **kw) -> Kernel:
    k = Kernel(n, **kw)
    k.tick = 1
    for a in k.agents:
        k.begin_turn(a)
    return k


def act(k, _aid, _action, **args):
    return asyncio.run(k.act(_aid, _action, args))


def test_private_boards_and_sharing():
    k = kernel()
    assert act(k, "A1", "write", board="private", content="secret")[0]
    assert not act(k, "A2", "read", board="A1")[0]
    assert act(k, "A1", "share", **{"with": ["A2"], "mode": "read"})[0]
    assert act(k, "A2", "read", board="A1")[0]
    assert not act(k, "A2", "write", board="A1", content="x")[0]


def test_groups_nest_and_insiders_reach_outer_memory():
    k = kernel()
    assert act(k, "A1", "connect", op="form_group", name="Outer", purpose="p")[0]
    assert act(k, "A2", "connect", op="form_group", name="Inner", purpose="p")[0]
    ok, msg = act(k, "A2", "connect", op="join", target="G1", **{"as": "G2"})
    assert ok, msg
    assert k.inside("A2", "G1") and k.depth("G1") == 2
    assert act(k, "A2", "write", board="G1", content="hello outer")[0]
    assert not act(k, "A1", "read", board="G2")[0]  # outer members do not see inner memory unless shared
    assert not act(k, "A1", "connect", op="join", target="G2", **{"as": "G1"})[0]  # would create a cycle
    assert act(k, "A3", "read", board="pub:G2")[0] and act(k, "A3", "write", board="pub:G2", content="request")[0]
    assert any("public board" in n["text"] for n in k.agents["A2"].inbox)


def test_overlapping_membership_is_allowed():
    k = kernel()
    act(k, "A1", "connect", op="form_group", name="X", purpose="p")
    act(k, "A2", "connect", op="form_group", name="Y", purpose="p")
    assert act(k, "A3", "connect", op="join", target="G1")[0] and act(k, "A3", "connect", op="join", target="G2")[0]
    assert k.member_of["A3"] == {"G1", "G2"}


def test_delegates_closed_groups_and_admission():
    k = kernel()
    act(k, "A1", "connect", op="form_group", name="T", purpose="p")
    act(k, "A2", "connect", op="join", target="G1")
    assert act(k, "A1", "protocol", group="G1", delegates=["A1"], open=False, text="A1 admits")[0]
    assert not act(k, "A2", "protocol", group="G1", text="coup")[0]
    ok, msg = act(k, "A3", "connect", op="join", target="G1")
    assert ok and "requested" in msg and "A3" not in k.groups["G1"].members
    assert not act(k, "A2", "connect", op="admit", group="G1", target="A3")[0]
    assert act(k, "A1", "connect", op="admit", group="G1", target="A3")[0] and "A3" in k.groups["G1"].members


def test_empty_groups_dissolve_and_leave_parents():
    k = kernel()
    act(k, "A1", "connect", op="form_group", name="Outer", purpose="p")
    act(k, "A2", "connect", op="form_group", name="Inner", purpose="p")
    act(k, "A2", "connect", op="join", target="G1", **{"as": "G2"})
    assert act(k, "A2", "connect", op="leave", target="G2")[0]
    assert k.groups["G2"].dissolved == 1 and "G2" not in k.groups["G1"].members


def test_attention_budget_limits_reading():
    k = kernel(attention=60)
    for i in range(10):
        act(k, "A1", "write", board="global", content=f"post {i} " + "x" * 120)
    ok, msg = act(k, "A2", "read", board="global", last=10)
    assert ok and "attention exhausted" in msg and k.agents["A2"].attention < 60


def test_links_need_both_agents_to_agree():
    k = kernel()
    assert not act(k, "A1", "link", op="send", target="A2", content="hi")[0]
    ok, msg = act(k, "A1", "link", op="open", target="A2", note="we both measure y3")
    assert ok and "asked" in msg and any("asks to open a link" in n["text"] for n in k.agents["A2"].inbox)
    assert not act(k, "A1", "link", op="send", target="A2", content="hi")[0]  # still waiting for A2
    assert not act(k, "A1", "link", op="open", target="A2")[0]  # asking twice does not open it
    assert act(k, "A2", "link", op="open", target="A1")[0]
    assert act(k, "A1", "link", op="send", target="A2", content="my data is on G1")[0]
    assert any(n["text"].startswith("[link]") for n in k.agents["A2"].inbox)
    assert act(k, "A2", "read", board="link:A1")[0] and not act(k, "A3", "read", board="link:A1")[0]
    assert k.structure()["links"] == [["A1", "A2"]]
    assert act(k, "A2", "link", op="close", target="A1")[0] and not act(k, "A1", "link", op="send", target="A2", content="x")[0]


def test_same_level_agents_talk_without_asking():
    k = kernel()
    act(k, "A1", "connect", op="form_group", name="Top", purpose="p")
    act(k, "A2", "connect", op="join", target="G1")
    assert act(k, "A1", "link", op="send", target="A2", content="direct")[0]  # same group, same level
    act(k, "A3", "connect", op="form_group", name="Sub", purpose="p")
    act(k, "A4", "connect", op="join", target="G2")
    act(k, "A3", "protocol", group="G2", delegates=["A3"])
    act(k, "A3", "connect", op="join", target="G1", **{"as": "G2"})
    assert k.same_level("A3", "A1") == "G1"  # A3 speaks for G2, a member of G1
    assert k.same_level("A4", "A1") is None  # A4 is inside G1 but below its level
    assert act(k, "A3", "link", op="send", target="A1", content="from G2's delegate")[0]
    assert not act(k, "A4", "link", op="send", target="A1", content="skipping the delegate")[0]
    assert act(k, "A4", "link", op="send", target="A3", content="to my delegate")[0]


def test_tasks_need_review_by_someone_else():
    k = kernel()
    act(k, "A1", "delegate", op="post", title="Screen y1", spec="s")
    assert act(k, "A2", "delegate", op="take", task="K1")[0]
    assert act(k, "A2", "delegate", op="deliver", task="K1", deliverable="done")[0]
    assert not act(k, "A2", "delegate", op="review", task="K1", verdict="accept")[0]
    assert act(k, "A1", "delegate", op="review", task="K1", verdict="accept")[0] and k.tasks["K1"].status == "accepted"
    ok, msg = act(k, "A1", "delegate", op="claim")
    assert not ok and "its ops are post, take" in msg, msg
    ok, msg = act(k, "A1", "delegate", op="drop", task="P45")
    assert not ok and "K ids" in msg, msg
    ok, msg = act(k, "A1", "delegate", op="post", title="anyone can", spec="s", to="anyone")
    assert ok and "open to anyone" in msg, msg


def test_custom_actions():
    k = kernel()
    calls = []

    async def runner(kk, aid, ca, args):
        calls.append((aid, ca.name, args))
        return "ran"

    k.custom_runner = runner
    assert act(k, "A1", "design_action", op="define", name="pool_data", description="d",
               procedure="experiment to my group, then screen my output", params=["output"])[0]
    assert not act(k, "A2", "x_pool_data", output="y1")[0]
    assert act(k, "A2", "design_action", op="adopt", name="pool_data")[0]
    assert act(k, "A2", "x_pool_data", output="y1")[0] and calls == [("A2", "pool_data", {"output": "y1"})]


def world(n=12, **kw):
    k = kernel(n)
    env = HiddenLaws(n, seed=3, **kw)
    k.env = env
    env.setup(k)
    return k, env


def test_world_experiments_data_visibility_and_scoring():
    k, env = world()
    aid, o = next((a, o) for a, outs in env.instruments.items() for o in outs
                  if env.level[o] == 0 and sum(o in v for v in env.instruments.values()) >= 2)
    other = next(a for a in env.instruments if a != aid and o in env.instruments[a])
    if aid != "A1":  # keep the rest of the test readable: A1 is the agent that measures y{o}
        env.instruments[aid], env.instruments["A1"] = env.instruments["A1"], env.instruments[aid]
        other = "A1" if other == "A1" else other
        aid, other = "A1", (aid if other == "A1" else other)
    for _ in range(4):
        assert act(k, "A1", "experiment", design="random", runs=5)[0]
    assert env.used["A1"] == 20
    assert not act(k, "A1", "experiment", design="random", runs=5, vary=["x1"])[0] or env.used["A1"] == 25
    terms = sorted(env.truth[o].parents)
    before, _ = env._data(k, other, o)
    act(k, "A1", "share", **{"with": [other]})
    after, _ = env._data(k, other, o)
    assert len(after) > len(before)  # sharing a private board shares its data
    ok, msg = act(k, other, "analyze", op="fit", output=f"y{o}", terms=terms)
    assert ok and "cross-validated" in msg, msg
    assert not act(k, "A1", "publish", output=f"y{o}", formula="__import__('os').system('ls')")[0]
    assert not act(k, "A1", "publish", output=f"y{o}", formula="x1.real")[0]
    assert act(k, "A1", "publish", output=f"y{o}", formula=env.truth[o].formula)[0]
    s = env.score()
    assert abs(s["score"] - 1 / env.n_out) < 1e-3 and s["coverage"] == round(1 / env.n_out, 4)
    k.tick = 3
    env.on_tick_end(k)
    assert "Lab validation at t3" in k.boards["global"].entries[-1].content
    assert "validated" in env.track_record("A1")


def test_composite_outputs_build_on_other_laws():
    k, env = world()
    comp = next(o for o, lv in env.level.items() if lv == 1)
    parts = sorted(int(p[1:]) for p in env.truth[comp].parents)
    assert env.truth[comp].parents == {f"y{p}" for p in parts}
    assert act(k, "A1", "publish", output=f"y{comp}", formula=env.truth[comp].formula)[0]
    assert env.score()["score_composite"] == 0.0  # its parts have no laws yet, so it predicts nothing
    for p in parts:
        assert act(k, "A2", "publish", output=f"y{p}", formula=env.truth[p].formula)[0]
    assert env.score()["score_composite"] > 0 and env.laws[comp].test > 0.99  # now exact, through its parts
    assert not act(k, "A3", "publish", output=f"y{parts[0]}", formula=f"1 + y{comp}")[0]  # no circles
    assert not act(k, "A3", "analyze", op="fit", output=f"y{comp}", terms=[f"y{comp}"])[0]
    assert "group record" not in k.group_card(asyncio.run(k.act("A1", "connect", {"op": "form_group", "name": "n"}))[1].split()[1])


def test_world_structure_and_shocks():
    k, env = world(60, shocks=["2:shift", "2:churn:0.5"])
    assert len({m["super"] for m in env.modules}) == env.n_super >= 2
    before = dict(env.truth)
    k.tick = 2
    env.on_tick_start(k)
    assert any(env.truth[o] is not before[o] for o in before)
    assert len(k.active_agents()) == 30
    flat = HiddenLaws(60, seed=3, structure="flat")
    assert not flat.has_modules


def test_mirroring_index():
    assert mirroring_index(["a", "a", "b", "b"], [0, 0, 1, 1]) == 1.0
    assert mirroring_index(["a", "a", "a", "a"], [0, 0, 1, 1]) == 0.0
    assert mirroring_index(["a", "b", "a", "b"], [0, 0, 1, 1]) == -0.5


def test_conditions():
    flat = build(dict(CFG, condition="flat"), 1, "hierarchical")
    assert not asyncio.run(flat.act("A1", "connect", {"op": "form_group", "name": "n"}))[0]
    ops = next(t for t in tool_schemas(flat, "A1") if t["name"] == "connect")["parameters"]["properties"]["op"]["enum"]
    assert set(ops) == {"follow", "unfollow"}
    oracle = build(dict(CFG, condition="oracle"), 1, "hierarchical")
    assert measure(oracle)["depth"] >= 2
    single = build(dict(CFG, condition="single"), 1, "hierarchical")
    assert len(single.agents) == 1 and len(single.env.instruments["A1"]) == single.env.n_out
    assert "design_action" not in [t["name"] for t in tool_schemas(build(dict(CFG, condition="no_design"), 1, "modular"), "A1")]


def test_culture_carries_over():
    k = build(CFG, 1, "hierarchical")
    asyncio.run(k.act("A1", "design_action", {"op": "define", "name": "sync_up", "description": "d",
                                              "procedure": "summarize my group board to its public board"}))
    asyncio.run(k.act("A1", "write", {"board": "library", "content": "Lesson: share data early.", "kind": "lesson"}))
    c = culture(k, "lessons text")
    k2 = build(CFG, 2, "hierarchical")
    inherit(k2, c)
    assert "sync_up" in k2.custom and k2.custom["sync_up"].author == "heritage:A1"
    assert any("lessons text" in e.content for e in k2.boards["library"].entries)


class FakeCompletions:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


def oa_reply(content="", calls=(), finish="tool_calls"):
    tcs = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None,
                          model_extra={"reasoning_details": [{"type": "reasoning.text", "text": "hmm"}]})
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=60),
                            model_extra={"cost": 0.0002})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage)


def test_openrouter_turn_loop_with_stub():
    k = build(dict(CFG, actions=1), 1, "hierarchical")
    completions = FakeCompletions([
        oa_reply(calls=[("connect", '{"op": "form_group", "name": "y1 team", "purpose": "p"}'), ("write", "{not json")]),
        oa_reply(content="Next I will screen y1.", finish="stop"),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = LLMBackend(OpenAICompatible("openrouter", client=client, run_id="test"))
    asyncio.run(backend.turn(k, "A1"))
    assert k.member_of["A1"] == {"G1"}
    first, closing = completions.requests
    assert first["model"] == "deepseek/deepseek-v4.1-flash" and first["tool_choice"] == "auto"
    assert first["extra_body"]["reasoning"] == {"enabled": False} and first["extra_body"]["session_id"] == "test-A1"
    assert first["messages"][0]["role"] == "system" and first["tools"][0]["type"] == "function"
    assert closing["tool_choice"] == "none"
    assistant, r1, r2 = closing["messages"][-3:]
    assert assistant["reasoning_details"] and [c["id"] for c in assistant["tool_calls"]] == ["c0", "c1"]
    assert r1["tool_call_id"] == "c0" and "formed" in r1["content"]
    assert r2["tool_call_id"] == "c1" and "not valid JSON" in r2["content"]
    assert k.usage["cached"] == 120 and k.usage["cost_microusd"] == 400
    assert k.events[-1]["op"] == "intent" and "screen y1" in k.events[-1]["msg"]


def test_claude_turn_loop_with_stub():
    k = build(dict(CFG, actions=1), 1, "hierarchical")
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    script = [SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="t1", name="reflect", input={"content": "plan"})],
                              stop_reason="tool_use", usage=usage),
              SimpleNamespace(content=[SimpleNamespace(type="text", text="Done.")], stop_reason="end_turn", usage=usage)]
    msgs = FakeCompletions(script)
    backend = LLMBackend(AnthropicProvider(client=SimpleNamespace(beta=SimpleNamespace(messages=msgs))))
    asyncio.run(backend.turn(k, "A1"))
    first, closing = msgs.requests
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default" and "tool_choice" not in first
    assert closing["tool_choice"] == {"type": "none"} and closing["messages"][-1]["content"][0]["tool_use_id"] == "t1"
    assert k.agents["A1"].notes == ["plan"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
