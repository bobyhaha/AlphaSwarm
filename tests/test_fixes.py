"""Fixes for three bugs the fork found in the DiG pilot traces (2026-11-02)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_incentives import CFG  # noqa: E402
from alphateam.runner import build  # noqa: E402


def test_a_reply_does_not_run_the_same_refused_call_again():
    """D_shared: A1 sent the same refused plan claim 97 times in one reply."""
    from alphateam.llm import Call, LLMBackend, Reply
    k = build(dict(CFG, world="arc", agents=1, calls=1, world_opts={"games": "lp85", "explore": "1"}), 0, "arc")
    k.max_calls = 1
    aid = next(iter(k.agents))
    k.tick = 1
    tokens = {"input": 1, "cached": 0, "output": 1}
    seen = []

    class Fake:
        transient = ()

        def tools(self, schemas):
            return [{"type": "function", "function": s} for s in schemas]

        def first(self, text):
            return [{"role": "user", "content": text}]

        def results(self, rs):
            seen.extend(rs)
            return [{"role": "tool", "tool_call_id": cid, "content": text} for cid, _, text in rs]

        async def create(self, system, messages, tools, final, who):
            if final:
                return Reply("done", [], "ok", tokens, [{"role": "assistant", "content": "done"}])
            calls = [Call(f"c{i}", "plan", {"op": "claim"}) for i in range(5)]  # no hypothesis: refused
            return Reply("", calls, "ok", tokens, [{"role": "assistant", "content": "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": "plan", "arguments": "{}"}} for c in calls]}])

    asyncio.run(LLMBackend(Fake()).turn(k, aid))
    ran = [e for e in k.events if e["op"] == "lab.claim"]
    assert len(ran) == 1 and not ran[0]["ok"], ran
    assert sum("the same call was refused earlier in this reply" in text for _, _, text in seen) == 4, seen


def test_link_with_content_and_no_op_is_a_send():
    k = build(dict(CFG, agents=3), 0, "design")
    a, b, _ = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    ok, msg = asyncio.run(k.act(a, "connect", {"op": "form_group", "name": "pair"}))
    gid = msg.split("formed ")[1].split()[0]
    assert asyncio.run(k.act(b, "connect", {"op": "join", "target": gid}))[0]
    ok, out = asyncio.run(k.act(a, "link", {"target": b, "content": "can you try the other knob?"}))
    assert ok, out
    assert any("can you try the other knob?" in n["text"] for n in k.agents[b].inbox)


def test_a_plan_with_a_result_and_no_op_is_a_test():
    start = str(Path(__file__).resolve().parents[1] / "tools" / "vc33_start_l4.json")
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "vc33", "explore": "1", "sync": "1", "board": "1",
                                                           "trust": "1", "start": start}), 0, "arc")
    a = sorted(k.agents, key=lambda x: int(x[1:]))[0]
    k.tick = 1
    ok, out = asyncio.run(k.act(a, "plan", {"result": "holds", "entry": "E999"}))
    assert not ok and "say what you will test" not in out, out  # routed to op=test, which wants a real piece
    ok, out = asyncio.run(k.act(a, "plan", {"op": "claim", "result": "holds"}))
    assert not ok and "belongs to op=test" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
