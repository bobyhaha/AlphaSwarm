"""The task-world layer with a toy benchmark: answers are numbers, graded exactly; one async tool."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.bench.base import Task, TaskWorld  # noqa: E402
from alphateam.bench.sandbox import run_python  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.metrics import measure  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.runner import _prescribe, run  # noqa: E402


class Toy(TaskWorld):
    name = "toy"
    TOOLS = [{"name": "compute", "description": "run python", "parameters": {"type": "object", "properties": {
        "task": {"type": "string"}, "code": {"type": "string"}}, "required": ["task", "code"]}}]

    def mission(self):
        return "Answer each task with a number."

    async def _compute(self, k, aid, a):
        t = self.task(a.get("task"))
        ok, out = await run_python(a["code"], Path(self.workdir) / t.id / aid, timeout=10)
        self.used[aid] += 1
        return "lab.compute", out, t.id

    async def grade(self, k, tasks):
        return {t: float(self.submissions[t].answer.strip() == self.tasks[t].meta["gold"]) for t in tasks}

    def mock_actions(self, k, aid, t, rng):
        a, b = self.tasks[t].meta["ab"]
        return [("open_task", {"task": t}), ("compute", {"task": t, "code": f"print({a}*{b})"}),
                ("submit", {"task": t, "answer": str(a * b) if rng.random() < 0.8 else "0"})]


def toy(n, tmp, single=False):
    tasks = [Task("", f"multiply {a} by {b}", f"fam{a % 3}", f"dom{a % 2}", f"What is {a}*{b}?",
                  {"gold": str(a * b), "ab": (a, b)}) for a, b in [(i, i + 7) for i in range(1, 13)]]
    w = Toy(n, tasks, seed=1, home=2, eval_at=(2,), single=single)
    w.workdir = str(tmp)
    return w


def test_toy_world(tmp=Path("/tmp/alphateam_toy")):
    k = Kernel(8, max_actions=4)
    k.env = toy(8, tmp)
    k.env.setup(k)
    assert len(k.env.tasks) == 12 and all(len(h) == 2 for h in k.env.home.values())
    ok, msg = asyncio.run(k.act("A1", "compute", {"task": "T1", "code": "print(6*7)"}))
    assert ok and msg.strip() == "42", msg
    ok, msg = asyncio.run(k.act("A1", "compute", {"task": "T1", "code": "while True: pass"}))
    assert ok and "stopped" in msg or "code" in msg
    ok, msg = asyncio.run(k.act("A1", "submit", {"task": "T99", "answer": "1"}))
    assert not ok and "no task" in msg
    gold = k.env.tasks["T3"].meta["gold"]
    assert asyncio.run(k.act("A2", "submit", {"task": "T3", "answer": gold}))[0]
    assert measure(k)["score"] is None  # nothing graded yet
    asyncio.run(k.env.finalize(k))
    m = measure(k)
    assert abs(m["score"] - 1 / 12) < 1e-3 and m["coverage"] == round(1 / 12, 4)
    assert "T3" in "\n".join(k.env.search(k, "A1", "bench", ["a2"]))
    k.begin_turn("A1")
    ok, msg = asyncio.run(k.act("A1", "search", {"what": "bench", "limit": 20}))  # through the kernel, as agents call it
    assert ok and "12 bench match" in msg and "T12" in msg, msg
    ok, msg = asyncio.run(k.act("A1", "search", {"what": "bench", "query": "unanswered"}))
    assert ok and "11 bench match" in msg, msg


def test_toy_run_with_mock(tmp=Path("/tmp/alphateam_toy_run")):
    k = Kernel(10, max_actions=4)
    k.env = toy(10, tmp)
    k.env.setup(k)
    data = asyncio.run(run(k, MockBackend(0), 3, tmp, {"condition": "self_org", "backend": "mock"}, quiet=True))
    m = data["snapshots"][-1]["metrics"]
    assert m["score"] is not None and m["score"] > 0 and m["groups"] > 0
    assert data["world"]["evaluations"][0]["t"] == 2 and data["world"]["evaluations"][-1]["final"]
    assert (tmp / "report.md").read_text().startswith("# AlphaTeam run")


def test_toy_oracle_prescription(tmp=Path("/tmp/alphateam_toy_oracle")):
    k = Kernel(9, max_actions=4, allowed=None)
    k.env = toy(9, tmp)
    _prescribe(k, by_module=True, rng=__import__("random").Random(0))
    assert k.active_groups()


def test_claims(tmp=Path("/tmp/alphateam_toy_claims")):
    k = Kernel(4, max_actions=4)
    k.env = toy(4, tmp)
    k.env.claims_on = True
    k.tick = 1
    assert "claim" in k.env.tool_names
    assert asyncio.run(k.act("A1", "claim", {"task": "T2"}))[0]
    ok, msg = asyncio.run(k.act("A2", "claim", {"task": "T2"}))
    assert not ok and "claimed by A1" in msg
    k.begin_turn("A2")
    ok, msg = asyncio.run(k.act("A2", "search", {"what": "bench", "query": "unclaimed", "limit": 20}))
    assert "11 bench match" in msg and "T2 " not in msg
    k.tick = 4  # lapsed after 3 ticks without an answer
    assert asyncio.run(k.act("A2", "claim", {"task": "T2"}))[0]
    assert asyncio.run(k.act("A2", "claim", {"task": "T2", "op": "release"}))[0]
    assert k.env.claimant(k, "T2") is None


def test_isolated_condition(tmp=Path("/tmp/alphateam_toy_iso")):
    from alphateam.runner import build
    import alphateam.bench as bench
    cfg = {"condition": "isolated", "agents": 6, "actions": 4, "attention": 3000, "world": "toy"}
    orig = bench.make_world
    bench.make_world = lambda name, cfg, n, seed=0, single=False: toy(n, tmp)
    try:
        k = build(cfg, 0, "flat")
    finally:
        bench.make_world = orig
    assert k.isolated and k.env.isolated and "connect" not in k.allowed and "link" not in k.allowed
    mine, other = k.env.home["A1"][0], next(t for t in k.env.tasks if t not in k.env.home["A1"])
    assert asyncio.run(k.act("A1", "submit", {"task": mine, "answer": "1"}))[0]
    ok, msg = asyncio.run(k.act("A1", "submit", {"task": other, "answer": "1"}))
    assert not ok and "own" in msg
    assert not asyncio.run(k.act("A1", "write", {"board": "global", "content": "hi"}))[0]
    assert asyncio.run(k.act("A1", "write", {"board": "private", "content": "note"}))[0]
    assert not asyncio.run(k.act("A2", "read", {"board": "private:A1"}))[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
