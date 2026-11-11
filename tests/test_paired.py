"""Paired runs: frames and arms, copying a running society, branches, the chaired control arm, and trials inside a run."""
import asyncio
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.mock import MockBackend  # noqa: E402
from alphateam.paired import (AlphaTeamRunner, OverBudget, behaviour, branches, fingerprint, fork, frame, instance, pair, run_once,  # noqa: E402
                           start)
from alphateam.recipe import REGISTRY, Setting, apply_edits, seed  # noqa: E402
from alphateam.runner import run  # noqa: E402
from alphateam.trial import pick, run_with_trials  # noqa: E402

mock = lambda s: MockBackend(s)  # noqa: E731
HELD = apply_edits(seed("design"), [{"op": "set", "key": "holders", "value": "shared"}])


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="alphateam-paired-"))


def test_frames_match_the_runs_they_stand_for():
    f = frame("open", agents=8)
    assert (f["calls"], f["world_opts"]["sims"], f["world_opts"]["blocks"], f["world_opts"]["supers"], f["ticks"]) == (8, "9", "8", "2", 24)
    wide = frame("open", agents=24, replies=3, sims=3, blocks=8)  # the 24-agent runs: the same 72 replies a tick
    assert wide["calls"] == 2 and wide["agents"] * (wide["calls"] + 1) == f["agents"] * (f["calls"] + 1) == 72
    one = frame("single", agents=8)
    assert one["condition"] == "single" and one["calls"] == 71 and one["world_opts"]["force_budget"] == "1" and not one["rules"]
    alone = frame("alone", agents=8)
    assert alone["condition"] == "isolated" and "sync" not in alone["world_opts"] and not alone["rules"]
    chaired = frame("chaired", agents=8)
    assert chaired["locked"] == {"take_lead": "team", "holders": "none", "rights_by_credit": False}
    assert chaired["world_opts"]["chair"] == "1"
    ranked = frame("hierarchy", agents=8)
    assert ranked["condition"] == "oracle" and ranked["force_opts"] == {"authority": "1"} and ranked["locked"]["take_lead"] == "off"
    grown = frame("open", agents=64)  # the task grows with the team: one block an agent
    assert grown["world_opts"]["blocks"] == "64" and grown["world_opts"]["supers"] == "16"
    assert fingerprint(f) != fingerprint(frame("open", agents=8, ticks=12)) and fingerprint(f) == fingerprint(frame("open", agents=8))
    for bad in (lambda: frame("ceo"), lambda: frame("open", replies=1), lambda: frame("open", agents=4, blocks=8)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("not refused")


def test_every_arm_has_the_same_memory_and_reviews():
    """What differs between arms is who is in charge and who can talk, not how much an agent remembers or how often
    the score is posted (a single agent with no memory and rarer reviews would not be a fair bar)."""
    seen = {}
    for arm in ("open", "chaired", "hierarchy", "single", "alone"):
        k = start(frame(arm, agents=8), seed("design"), 3)
        seen[arm] = (k.continuity, k.env.validate_every, k.env.credit_every, k.env.contest_facts, k.env.economy_share, k.env.economy)
    assert set(seen.values()) == {(2, 1, 3, False, 0.0, True)}, seen
    ranked = start(frame("hierarchy", agents=8), seed("design"), 3)
    assert len(ranked.env.rights) == ranked.env.D and all(h in ranked.groups for h in ranked.env.rights.values())
    assert not ranked.env.ownership and not ranked.env.leadership and "rules" in ranked.allowed


def test_selection_and_report_never_share_an_instance():
    kinds = {p: {instance(p, i) for i in range(5000)} for p in ("selection", "report", "probe", "dry")}
    assert not kinds["selection"] & kinds["report"] and not kinds["probe"] & kinds["selection"] and not kinds["dry"] & kinds["report"]


def test_every_arm_runs_with_scripted_agents():
    async def go():
        out = tmp()
        for arm in ("open", "chaired", "single", "alone"):
            r = await run_once(frame(arm, agents=8, ticks=4), seed("design"), mock(1), 7, out / arm)
            assert r.ticks == 4 and len(r.scores) == 5 and r.score is not None and r.stopped is None, arm
            assert (out / arm / "result.json").exists()
        full = await run_once(frame("open", agents=8, ticks=3), seed("design"), mock(1), 7, out / "full", final=True)
        assert (out / "full" / "run.json").exists() and full.recipe == seed("design").id
    asyncio.run(go())


def test_a_copy_of_a_running_society_carries_on_exactly():
    """The property branches rest on: a copy that keeps its parent's rules reproduces the unbroken run."""
    async def go():
        out, f = tmp(), frame("open", agents=8, ticks=10)
        straight = await run_once(f, seed("design"), mock(7), 7, out / "straight")
        k, backend = start(f, seed("design"), 7), mock(7)
        await run(k, backend, 5, out / "trunk", {}, 7, quiet=True, horizon=10, final=False)
        twin, tb = fork(k, backend)
        assert twin is not k and twin.env is not k.env and twin.env._k is twin and twin.rulebook is not k.rulebook
        await run(twin, tb, 10, out / "twin", {}, 7, quiet=True, start=5, horizon=10, final=False)
        assert [s["metrics"]["score"] for s in twin.snapshots] == straight.scores
        assert k.tick == 5 and len(k.snapshots) == 6  # the original stands where it was
        await run(k, backend, 10, out / "trunk", {}, 7, quiet=True, start=5, horizon=10, final=False)
        assert [s["metrics"]["score"] for s in k.snapshots] == straight.scores and behaviour(k) == behaviour(twin)
    asyncio.run(go())


def test_branches_differ_only_after_the_change():
    async def go():
        out, f = tmp(), frame("open", agents=8, ticks=10)
        straight = await run_once(f, seed("design"), mock(7), 7, out / "straight")
        same, kids = await branches(f, seed("design"), [HELD], mock, 7, out / "b", fork_at=5)
        held = kids[0]
        assert same.scores == straight.scores and held.scores[:6] == straight.scores[:6] and held.scores != straight.scores
        assert same.since == held.since == 5 and held.recipe == HELD.id and same.recipe == seed("design").id
        assert "refused:lab" in held.ops and "refused:lab" not in same.ops  # non-holders are turned away after the change
        assert same.progress == round(sum(straight.scores[6:]) / 5, 5)  # the progress score counts the ticks after the fork
        team = apply_edits(seed("design"), [{"op": "set", "key": "take_lead", "value": "team"}])
        try:
            await branches(f, seed("design"), [team], mock, 7, out / "c", fork_at=5)
        except ValueError as exc:
            assert "only applies from the start" in str(exc)
        else:
            raise AssertionError("a change that needs a fresh start was branched")
        a, b = await pair(f, seed("design"), team, mock, 7, out / "p")  # it is compared from scratch instead
        assert a.scores == straight.scores and b.recipe == team.id and a.since == b.since == 0
    asyncio.run(go())


def test_the_chaired_arm_puts_one_agent_in_charge():
    async def go():
        k = start(frame("chaired", agents=8, ticks=4), seed("design"), 3)
        env = k.env
        team = next(iter(env.teams))
        assert env.chairman and env.teams[team]["lead"] == env.chairman and len(k.agents_inside(team)) == 8
        shared = {i for i in range(env.D) if len(env.users[i]) > 1}
        assert set(env.team_rights) == shared and set(env.rights) == shared and set(env.rights.values()) == {team}
        assert "Before the run began" in env.brief() and env.chairman in env.brief()
        assert "every knob that several of its members' blocks share" in env.brief()
        from alphateam.primitives import system_prompt
        said = system_prompt(k)  # the prompt stays true: it says what was set before the run
        assert "One team and its lead were set before this run began" in said and "Nobody assigns roles" not in said
        assert "Nobody assigns roles, teams, leaders" in system_prompt(start(frame("open", agents=8), seed("design"), 3))
        k.tick = 1
        member = next(a for a in sorted(k.agents) if a != env.chairman)
        i = min(shared)
        knob, value = env.name[i], 5 if env.x[i] != 5 else 6
        ok, msg = await k.act(member, "set_knob", {"knob": knob, "value": value})
        assert not ok and "held by" in msg  # a shared knob waits for the chair
        own = next(j for j in range(env.D) if env.users[j] == env.instruments[member])
        ok, msg = await k.act(member, "set_knob", {"knob": env.name[own], "value": 5 if env.x[own] != 5 else 6})
        assert ok, msg  # its own block's knob is its own
        other = next(a for a in sorted(k.agents) if a not in (env.chairman, member))
        ok, msg = await k.act(member, "delegate", {"op": "post", "title": "settle it", "spec": "set it", "to": other, "scope": [knob]})
        tid = next(iter(k.tasks))
        ok, msg = await k.act(other, "delegate", {"op": "take", "task": tid})
        assert not ok and "only its holder can hand it over" in msg  # a member cannot hand out what the team holds
        ok, msg = await k.act(env.chairman, "delegate", {"op": "post", "title": "settle it", "spec": "set it", "to": other, "scope": [knob]})
        ok, msg = await k.act(other, "delegate", {"op": "take", "task": sorted(k.tasks, key=lambda t: int(t[1:]))[-1]})
        assert ok and env.holder(i) == other, msg  # the chair can
        ok, msg = await k.act(env.chairman, "set_knob", {"knob": env.name[max(shared)], "value": 5 if env.x[max(shared)] != 5 else 6})
        assert ok, msg
        assert env._leave(k, 2) == env.chairman  # the leave shock takes the agent with the most decisions: the chair
        chairs = {start(frame("chaired", agents=4), seed("design"), s).env.chairman for s in range(12)}
        assert len(chairs) > 1  # who chairs is drawn with the task instance
        ok, msg = await k.act(member, "rules", {"op": "propose", "key": "take_lead", "value": "off", "why": "nobody should be in charge here"})
        assert not ok and "fixed in this run's frame" in msg
        opened = start(frame("open", agents=8), seed("design"), 3)
        assert not opened.env.teams and opened.env.chairman is None and not opened.env.rights  # nobody is placed in charge
        r = await run_once(frame("chaired", agents=8, ticks=6), seed("design"), mock(1), 3, tmp())
        assert r.ops.get("refused:lab", 0) > 0  # scripted members are turned away: every decision waits for the chair
    asyncio.run(go())


def test_a_runner_drops_a_recipe_that_cannot_run_and_keeps_to_its_cap():
    async def go():
        runner = AlphaTeamRunner(frame("open", agents=8, ticks=6), mock, tmp())
        assert await runner.dry(seed("design")) == (True, "")
        REGISTRY["design"]["broken"] = Setting("broken", "procedures", "bool", False, "test only", opt="chair")
        try:  # chair=1 without team mode: the world refuses to start
            ok, why = await runner.dry(apply_edits(seed("design"), [{"op": "set", "key": "broken", "value": True}]))
        finally:
            del REGISTRY["design"]["broken"]
        assert not ok and "it breaks a run" in why and "chair=1 needs leadership=team" in why
        (a, b), = await runner.screen(seed("design"), [HELD], 1)
        assert a.since == b.since == 3 and a.recipe == seed("design").id  # screened as branches of one run, forked at the half
        team = apply_edits(seed("design"), [{"op": "set", "key": "take_lead", "value": "team"}])
        (a1, b1), (a2, b2) = await runner.screen(seed("design"), [team, HELD], 2)
        assert (a1.since, b1.since, b1.recipe) == (0, 0, team.id)  # a change that needs a fresh start runs from scratch,
        assert (a2.since, b2.since, b2.recipe) == (3, 3, HELD.id)  # beside one that is still a branch
        p = await runner.probe(seed("design"), HELD, 1)
        assert p["runs"] == 3 and p["moved"] > 0.03 and p["moved"] > 5 * p["floor"]  # holders change what agents do, far
        # more than two runs under the same rules differ
        again = AlphaTeamRunner(frame("open", agents=8, ticks=6), mock, runner.out)  # a second session on the same output
        r = await again.score(seed("design"), 1, "retest")
        first = await runner.score(seed("design"), 1, "retest")
        assert r.dir != first.dir and Path(r.dir).name == "retest-0001" and Path(first.dir).name == "retest-0002"
        capped = AlphaTeamRunner(frame("open", agents=8, ticks=4), mock, tmp(), cap_usd=1.0)
        capped.spent = 1.0  # (the cap under real costs is tested in test_live_path.py)
        try:
            await capped.score(seed("design"), 1)
        except OverBudget:
            return
        raise AssertionError("a run started past the cap")
    asyncio.run(go())


class Proposing(MockBackend):
    """Scripted agents, two of whom propose a rule change at tick 2 (scripted agents never do so themselves)."""

    def __init__(self, seed, key="holders", value="shared"):
        super().__init__(seed)
        self.key, self.value = key, value

    async def turn(self, k, aid):
        await super().turn(k, aid)
        if k.tick == 2 and aid in ("A1", "A2") and getattr(k, "rulebook", None):
            await k.act(aid, "rules", {"op": "propose", "key": self.key, "value": self.value,
                                       "why": "the knobs we share kept being overwritten"})


def test_a_society_can_try_its_own_proposal_on_a_copy_of_itself():
    async def go():
        out, f = tmp(), frame("open", agents=8, ticks=12)
        plain = await run_with_trials(f, seed("design"), mock, 7, out / "plain", every=4, window=2)
        assert plain.trials == [] and plain.ticks == 12 and not plain.ended_with  # nothing proposed: nothing tried, nothing paid
        r = await run_with_trials(f, seed("design"), lambda s: Proposing(s), 7, out / "tried", every=4, window=2, backers=2)
        assert len(r.trials) == 1 and r.ticks == 10  # one trial, paid for by ending two ticks sooner
        t = r.trials[0]
        assert (t["t"], t["window"], t["key"], t["value"], t["before"], t["backers"]) == (4, 2, "holders", "shared", "none", 2)
        assert t["kept"] == (t["with"] > t["without"]) and (r.ended_with == HELD.id) == t["kept"]
        assert round(sum(r.scores[5:7]) / 2, 5) == (t["with"] if t["kept"] else t["without"])  # the better copy carried on
        assert "tried over t5-t6 on a copy of this society" in t["text"] and any("mid-run" in x["text"] for x in r.lessons)
        from alphateam.primitives import system_prompt
        k = start(f, seed("design"), 7)
        assert "TRIALS" not in system_prompt(k)  # said only where proposals can be tried inside the run
        k.rulebook.note = "TRIALS: every 4 ticks"
        assert "TRIALS: every 4 ticks" in system_prompt(k) and "TRIALS: every 4 ticks" in k.rulebook.view()
        alone = await run_with_trials(f, seed("design"), lambda s: Proposing(s), 7, out / "few", every=4, window=2, backers=3)
        assert alone.trials == [] and len(alone.proposals) == 1  # two backers are not three
        late = await run_with_trials(f, seed("design"), lambda s: Proposing(s, "take_lead", "team"), 7, out / "late", every=4, window=2)
        assert late.trials == []  # a change that needs a fresh start is left for the next generation
        short = await run_with_trials(frame("open", agents=8, ticks=6), seed("design"), lambda s: Proposing(s), 7, out / "short",
                                      every=4, window=2)
        assert short.trials == [] and short.ticks == 6  # too late to use what a trial would show: none is run
        assert all(x.get("by") == "lab" for x in r.lessons)  # a trial's note is the lab's record, not an agent's advice
        k = start(f, seed("design"), 7)
        assert pick(k, set(), 1) is None
    asyncio.run(go())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
