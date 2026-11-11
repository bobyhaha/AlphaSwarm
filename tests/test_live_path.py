"""The path a paid run takes, with a stubbed provider: recipes, pairs, branches, trials and the cap through the LLM
backend. The stub bills a fixed price a reply, so what a runner counts can be checked against what was billed."""
import asyncio
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.archive import Archive  # noqa: E402
from alphateam.llm import LLMBackend, OpenAICompatible  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.paired import AlphaTeamRunner, OverBudget, RunStopped, branches, fork, frame, pair, start  # noqa: E402
from alphateam.recipe import apply_edits, seed  # noqa: E402
from alphateam.runner import run  # noqa: E402
from alphateam.selection import Ladder, Policy  # noqa: E402
from alphateam.trial import run_with_trials  # noqa: E402

HELD = apply_edits(seed("design"), [{"op": "set", "key": "holders", "value": "shared"}])
META = {"model": "stub", "preset": "openrouter"}
PRICE = 0.001  # what the stub bills for a reply


class Stub:
    """An endless scripted provider: every working reply looks at the design; the closing reply is a note. `first`
    (a tool name and its arguments) is what the very first reply does instead."""

    def __init__(self, first=None):
        self.requests = 0
        self.systems = set()
        self.first = first

    async def create(self, **kw):
        self.requests += 1
        self.systems.add(kw["messages"][0]["content"] if isinstance(kw["messages"][0]["content"], str) else "")
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=60),
                                model_extra={"cost": PRICE})
        if kw.get("tool_choice") == "none":
            msg = SimpleNamespace(content=f"note {self.requests}", tool_calls=None, model_extra={})
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)
        name, args = self.first if self.first and self.requests == 1 else ("view_design", "{}")
        call = SimpleNamespace(id="c0", function=SimpleNamespace(name=name, arguments=args))
        msg = SimpleNamespace(content="", tool_calls=[call], model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=usage)


class ClosingStub(Stub):
    """As Stub. When only the rules action is on offer (the closing round), the first agent asked proposes a rule
    change and the others answer with a note."""

    def __init__(self):
        super().__init__()
        self.closing, self.told = 0, []

    async def create(self, **kw):
        names = [t["function"]["name"] for t in kw.get("tools") or []]
        if names != ["rules"]:
            return await super().create(**kw)
        self.requests += 1
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=60),
                                model_extra={"cost": PRICE})
        users = [m["content"] for m in kw["messages"] if m.get("role") == "user" and isinstance(m.get("content"), str)]
        if kw.get("tool_choice") != "none" and not any(m.get("role") == "tool" for m in kw["messages"][-2:]):
            self.closing += 1
            self.told.append(users[-1])
            if self.closing == 1:
                args = json.dumps({"op": "propose", "key": "holders", "value": "shared",
                                   "why": "two of us kept overwriting one shared knob; one holder a knob would settle it"})
                call = SimpleNamespace(id="c9", function=SimpleNamespace(name="rules", arguments=args))
                msg = SimpleNamespace(content="", tool_calls=[call], model_extra={})
                return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=usage)
        msg = SimpleNamespace(content="nothing more to add", tool_calls=None, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


class LookBackStub(Stub):
    """As Stub. In the closing round the first agent asked looks at the run's record, then suggests a rule that is not
    a setting; the others answer with a note."""

    def __init__(self):
        super().__init__()
        self.lead, self.offered, self.read = False, [], ""

    async def create(self, **kw):
        names = [t["function"]["name"] for t in kw.get("tools") or []]
        if names != ["rules"]:
            return await super().create(**kw)
        self.requests += 1
        self.offered.append(kw["tools"][0]["function"])
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=SimpleNamespace(cached_tokens=60),
                                model_extra={"cost": PRICE})
        done = {m.get("tool_call_id"): m.get("content") for m in kw["messages"] if m.get("role") == "tool"}

        def call(cid, args):
            c = SimpleNamespace(id=cid, function=SimpleNamespace(name="rules", arguments=json.dumps(args)))
            msg = SimpleNamespace(content="", tool_calls=[c], model_extra={})
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=usage)

        if kw.get("tool_choice") != "none":
            if not self.lead and "lb1" not in done:  # an agent's kept conversation holds earlier tool results too
                self.lead = True
                return call("lb1", {"op": "record", "what": "timeline"})
            if "lb1" in done and "lb2" not in done:
                self.read = str(done["lb1"])
                return call("lb2", {"op": "suggest", "evidence": ["t1"], "when": "two agents changed one knob in tick 1",
                                    "text": "Show each agent, before it sets a knob, who else set that knob in this tick."})
        msg = SimpleNamespace(content="nothing more to add", tool_calls=None, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


def backends(first=None, stub=None):
    stub = stub or Stub(first)
    base = LLMBackend(OpenAICompatible("openrouter", client=SimpleNamespace(chat=SimpleNamespace(completions=stub)), run_id="t"))

    def make(seed_):
        b = copy.copy(base)  # one request limit for all runs; each society's backend keeps its own agents' conversations
        b.memory = {}
        return b
    return stub, make


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="alphateam-live-"))


def billed(stub) -> float:
    return round(stub.requests * PRICE, 5)


def small(ticks=2):
    return frame("open", agents=4, ticks=ticks, replies=3, sims=3)  # 4 agents x 3 replies a tick: $0.012 a tick


def test_a_pair_runs_through_the_llm_backend_and_reports_its_cost():
    async def go():
        stub, make = backends()
        a, b = await pair(small(), seed("design"), HELD, make, 5, tmp())
        assert stub.requests == 2 * 4 * 2 * 3 and a.cost == b.cost == 0.024 and a.ops == {"lab.view_design": 16}
        assert any("only that agent can set it" in s for s in stub.systems)  # the child's agents were told its rule
        assert any("Anyone may set a knob nobody holds" in s for s in stub.systems)
    asyncio.run(go())


def test_after_the_last_tick_agents_may_still_propose_rules_and_the_score_does_not_move():
    async def go():
        from alphateam.paired import run_once
        plain, (stub, make) = small(), backends(stub=ClosingStub())
        base = await run_once(plain, seed("design"), backends()[1](5), 5, tmp(), "generation", final=True)
        f = dict(plain, closing=True)
        r = await run_once(f, seed("design"), make(5), 5, tmp(), "generation", final=True)
        assert stub.closing == 4 and all("The run is over" in x and "THE RULES THIS SOCIETY RUNS UNDER" in x for x in stub.told)
        assert not any(w in x.lower() for x in stub.told for w in ("you should", "you must", "please"))  # it asks for nothing
        assert [(p["key"], p["value"], len(p["backers"])) for p in r.proposals] == [("holders", "shared", 1)]
        assert r.ops.get("rules.propose") == 1 and r.ops.get("closing.note") == 4 and not base.proposals
        assert r.scores == base.scores and r.score == base.score  # the design was final: nothing said here moves the score
        assert r.cost == round(base.cost + 5 * PRICE, 5)  # four agents' replies, and one more after the proposal
        part = await run_once(f, seed("design"), backends(stub=ClosingStub())[1](5), 5, tmp(), "probe", 1)
        assert not part.proposals and "closing.note" not in part.ops  # a part of a run has no closing round
        k = start(plain, seed("design"), 5)
        assert not k.closing_round and start(f, seed("design"), 5).closing_round and not start(f, None, 5).__dict__.get("closing_round")
    asyncio.run(go())


def test_after_the_last_tick_an_agent_can_look_at_the_runs_record_and_suggest_a_rule():
    async def go():
        from alphateam.paired import run_once
        from alphateam.recipe import RULES_TOOL
        plain, (stub, make) = small(), backends(stub=LookBackStub())
        base = await run_once(plain, seed("design"), backends()[1](5), 5, tmp(), "generation", final=True)
        r = await run_once(dict(plain, closing=True), seed("design"), make(5), 5, tmp(), "generation", final=True)
        tool = stub.offered[0]  # the same action, with the two ops that look back
        assert tool["description"].startswith(RULES_TOOL) and "record:" in tool["description"] and "suggest:" in tool["description"]
        assert tool["parameters"]["properties"]["op"]["enum"] == ["view", "record", "propose", "suggest", "back", "lesson"]
        assert "round  1 |" in stub.read and "\nround  2 |" in stub.read  # what it was shown: the run, a line a tick
        (s,) = r.suggestions
        assert s["text"].startswith("Show each agent") and s["evidence"] == ["t1"] and s["backers"] == [s["by"]]
        assert r.ops.get("rules.record") == 1 and r.ops.get("rules.suggest") == 1 and not base.suggestions
        assert r.scores == base.scores and r.score == base.score  # looking back moves nothing
        # the run's full trace: what each agent was shown, what the model replied, what each action got back, uncut
        rows = [json.loads(x) for x in (Path(r.dir) / "trace.jsonl").read_text().splitlines()]
        kinds = {}
        for x in rows:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        assert kinds["system"] == 1 and kinds["turn"] == 4 * 2 and kinds["closing"] == 4 and kinds["reply"] == stub.requests - 1  # all but the run's one retrospective call
        look = next(x for x in rows if x["kind"] == "action" and x["op"] == "rules.record")
        assert look["result"] == stub.read.rsplit("\n[uses left", 1)[0] and len(look["result"]) > 60  # the whole result
        turn = next(x for x in rows if x["kind"] == "turn")
        assert turn["shown"].startswith("TICK 1 of 2") and "The run is over" in next(x for x in rows if x["kind"] == "closing")["shown"]
    asyncio.run(go())


def test_a_branch_remembers_the_shared_past_and_the_shared_ticks_are_charged_once():
    async def go():
        stub, make = backends()
        f = small(4)
        k, backend = start(f, seed("design"), 5), make(5)
        await run(k, backend, 2, tmp(), {}, 5, quiet=True, horizon=4, final=False)
        assert k.continuity == 2 and set(backend.memory) == set(k.agents)  # each agent keeps its last two ticks
        twin, tb = fork(k, backend)
        assert tb.memory == backend.memory and tb.memory is not backend.memory and tb.sem is backend.sem and tb.p is backend.p
        tb.memory["A1"].append("only the copy")
        assert "only the copy" not in backend.memory["A1"]
        stub2, make2 = backends()
        same, kids = await branches(f, seed("design"), [HELD], make2, 5, tmp(), fork_at=2)
        assert stub2.requests == 4 * 3 * (2 + 2 + 2)  # the trunk's two ticks once, then two ticks for each branch
        assert same.cost == 0.048 and kids[0].cost == 0.024 and same.since == kids[0].since == 2  # the parent's copy carries the trunk
        assert round(same.cost + kids[0].cost, 5) == billed(stub2)
        stub3, make3 = backends()
        runner = AlphaTeamRunner(f, make3, tmp(), cap_usd=1.0, fork_at=2, meta=META)
        (a, b), = await runner.screen(seed("design"), [HELD], 1)  # the same through a runner: what it counts is what was billed
        assert round(runner.spent, 5) == billed(stub3) == 0.072 and a.since == b.since == 2
    asyncio.run(go())


def test_the_cap_stops_a_live_runner_and_counts_what_was_billed():
    async def go():
        stub, make = backends()
        runner = AlphaTeamRunner(small(), make, tmp(), cap_usd=0.03, meta=META)
        r = await runner.score(seed("design"), 1, "retest")
        assert r.cost == 0.024 and runner.spent == 0.024
        try:  # under the cap when it starts: each of the two runs gets half of what is left, and stops there
            await runner.confirm(seed("design"), HELD, 1)
        except RunStopped as exc:
            assert "stopped early" in str(exc) and "budget exhausted" in str(exc)
        else:
            raise AssertionError("runs that ran out of budget were scored")
        assert round(runner.spent, 5) == billed(stub) == 0.03  # each stopped at its share: what is counted is what was billed
        try:
            await runner.score(seed("design"), 2, "retest")
        except OverBudget as exc:
            assert "at the cap" in str(exc) and stub.requests == 30  # nothing more was asked of the provider
            return
        raise AssertionError("a run started past the cap")
    asyncio.run(go())


def test_a_run_stopped_at_its_budget_leaves_no_score():
    async def go():
        stub, make = backends()
        runner = AlphaTeamRunner(small(), make, tmp(), cap_usd=0.01, meta=META)
        A = Archive(tmp())
        A.add(seed("design"), "seed", runner.niche)
        try:
            await Ladder(A, runner, Policy(retest_runs=2)).ensure_tested(seed("design"))
        except RunStopped:
            assert A.runs() == [] and round(runner.spent, 5) == billed(stub) == 0.012  # its cost is counted; no score is on file
            return
        raise AssertionError("a run that stopped at its budget was scored")
    asyncio.run(go())


def test_a_trial_is_billed_in_full():
    async def go():
        proposal = {"op": "propose", "key": "holders", "value": "shared", "why": "the knobs we share kept being overwritten"}
        stub, make = backends(first=("rules", json.dumps(proposal)))
        r = await run_with_trials(small(6), seed("design"), make, 5, tmp(), every=2, window=1, backers=1, meta=META)
        assert len(r.trials) == 1 and r.trials[0]["t"] == 2 and r.ticks == 5  # one trial, paid for by ending a tick sooner
        assert stub.requests == 4 * 3 * (5 + 1) and r.cost == billed(stub) == 0.072  # the dropped copy's tick is in the cost
    asyncio.run(go())


def test_a_provider_that_reports_no_cost_is_counted_at_the_estimate():
    async def go():
        runner = AlphaTeamRunner(small(), lambda s: MockBackend(s), tmp(), cap_usd=1.0, est_cost=0.5)
        for i in (1, 2):
            assert (await runner.score(seed("design"), i, "retest")).cost is None
        assert runner.spent == 1.0
        try:
            await runner.score(seed("design"), 3, "retest")
        except OverBudget:
            return
        raise AssertionError("a run started past the cap")
    asyncio.run(go())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
