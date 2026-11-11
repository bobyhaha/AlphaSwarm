"""Recipes on the code world: its rule settings and frames, runs in processes of their own, and the ladder on fresh
runs. Everything here is on the in-memory harness with scripted agents: no Docker, no API."""
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from alphateam.archive import FRESH, Archive  # noqa: E402
from alphateam.code_runs import CodeRunner  # noqa: E402
from alphateam.generations import candidates, evolve  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.paired import CODE_RUN, OUTER, Result, RunStopped, code_frame, code_version, fingerprint, run_once, start  # noqa: E402
from alphateam.recipe import apply_edits, inert, options, seed  # noqa: E402
from alphateam.selection import Policy  # noqa: E402

SEED = seed("code")
STREAM = apply_edits(SEED, [{"op": "set", "key": "stream", "value": "on"}], {"by": "control"})
quiet = lambda *_: None  # noqa: E731


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="alphateam-code-recipes-"))


def small() -> dict:
    return code_frame(agents=3, ticks=3, fake=True)


def runner(out: Path, **kw) -> CodeRunner:
    return CodeRunner(small(), {"kind": "mock"}, out, stagger=0.0, log=quiet, **kw)


def test_the_seed_recipe_is_the_plain_arm_of_the_saved_runs():
    o = options(SEED)
    assert (o["credit"], o["stream"], o["reflect"], o["own"], o["continuity"], o["goal"]) == ("0", "0", "0", "1", "2", "individual")
    f = code_frame()
    assert (f["agents"], f["ticks"], f["calls"], f["attention"], f["grade"]) == (10, 24, 2, 6000, True)
    wo = f["world_opts"]
    assert (wo["steps"], wo["probes"], wo["sandboxes"], wo["held_every"], wo["ways"]) == ("12", "480", "3", "merge", "0")
    assert fingerprint(f)["world"] == "code" and fingerprint(f)["task"]["probes"] == "480" and "fake" not in wo


def test_runs_that_may_not_compare_have_different_fingerprints():
    """Found by a review on 2026-11-10: a halved reading budget or a higher reasoning effort left the fingerprint as it
    was, so an archive would have pooled runs that do not compare. The code a run loads counts too."""
    import subprocess
    f, live = code_frame(), {"backend": "openrouter", "model": "m", "reasoning_effort": "low", "max_tokens": 16384}
    base = fingerprint(f, live)
    assert fingerprint(code_frame(attention=3000), live) != base  # how much an agent may read
    assert fingerprint(f, dict(live, reasoning_effort="high")) != base  # how hard the model thinks
    assert fingerprint(f, dict(live, max_tokens=4096)) != base
    assert fingerprint(code_frame(heritage="org"), live) != base  # any fixed option of the code world
    assert fingerprint(dict(f, rules=False), live) != base  # any other key of the frame that can change a score
    assert fingerprint(code_frame(closing=True), live) == base and fingerprint(dict(f, seed=7, grade=False), live) == base
    # the code: a changed module is a new version, an untouched one is not
    root = tmp()
    (root / "world.py").write_text("x = 1\n")
    (root / "selection.py").write_text("y = 1\n")
    v1 = code_version("toy", str(root))
    (root / "selection.py").write_text("y = 2\n")  # no society runs under the selection ladder
    code_version.cache_clear()
    assert code_version("toy", str(root)) == v1
    (root / "world.py").write_text("x = 2\n")
    code_version.cache_clear()
    assert code_version("toy", str(root)) != v1 and len(v1) == 12 and base["code"] == code_version("code")
    # the list of what a code-world run loads is checked against a scripted run in a process of its own
    code = ("import asyncio, sys, tempfile\nfrom pathlib import Path\nfrom alphateam.paired import code_frame, run_once, instance\n"
            "from alphateam.recipe import seed\nfrom alphateam.mock import MockBackend\nimport alphateam.llm\n"
            "f = dict(code_frame(agents=3, ticks=2, fake=True), grade=False)\n"
            "asyncio.run(run_once(f, seed('code'), MockBackend(0), instance('dry', 0), Path(tempfile.mkdtemp()), 'dry', 2))\n"
            "print(' '.join(sorted(m.split('.')[1] for m in sys.modules if m.startswith('alphateam.') and m.count('.') == 1)))\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert done.returncode == 0, done.stderr[-400:]
    loaded = set(done.stdout.strip().splitlines()[-1].split())
    assert loaded <= set(CODE_RUN) | set(OUTER), f"a code run loads modules the code version does not cover: {loaded - set(CODE_RUN) - set(OUTER)}"


def test_a_setting_that_cannot_act_is_not_a_revision():
    assert "credit is off" in inert(SEED, apply_edits(SEED, [{"op": "set", "key": "goal", "value": "team"}]))
    paid = apply_edits(SEED, [{"op": "set", "key": "credit", "value": True}])
    assert inert(paid, apply_edits(paid, [{"op": "set", "key": "goal", "value": "team"}])) is None
    assert inert(SEED, STREAM) is None
    assert "stream is on" in inert(STREAM, apply_edits(STREAM, [{"op": "set", "key": "offer_order", "value": "open"}]))


def test_a_recipe_sets_the_world_and_how_often_the_society_reviews_itself():
    r = apply_edits(SEED, [{"op": "set", "key": "stream", "value": True}, {"op": "set", "key": "review_every", "value": 6},
                           {"op": "set", "key": "credit", "value": True}, {"op": "set", "key": "goal", "value": "society"},
                           {"op": "set", "key": "own", "value": False}, {"op": "set", "key": "memory_ticks", "value": 1},
                           {"op": "set", "key": "offer_order", "value": "credit"}, {"op": "set", "key": "own_ticks", "value": 4}])
    k = start(small(), r, 1)
    assert k.env.stream and k.env.incentive and not k.env.own_on and k.continuity == 1
    assert (k.env.law["goal"], k.env.law["offer_order"], k.env.law["own_ticks"]) == ("society", "credit", 4)
    assert k.reflection.window == 6
    plain = start(small(), SEED, 1)
    assert not plain.env.stream and not plain.env.incentive and plain.env.own_on and getattr(plain, "reflection", None) is None
    assert "reflect" not in plain.env.__dict__  # the interval is a setting of the run, never a world option


def test_the_rules_action_is_offered_also_in_a_ticks_last_reply():
    k = start(small(), SEED, 1)
    assert "rules" in k.allowed and "rules" in k.env.final_tools and "rules" not in type(k.env).final_tools
    text = k.rulebook.view()
    assert "stream = off" in text and "credit = off" in text and "review_every = 0" in text


def test_a_full_run_is_graded_and_a_short_one_is_not():
    async def go():
        full = await run_once(small(), SEED, MockBackend(0), 1, tmp(), "retest")
        assert full.hidden is not None and full.score is not None and full.ticks == 3 and "merges" in full.facts
        short = await run_once(small(), SEED, MockBackend(0), 1, tmp(), "probe", ticks=2)
        assert short.hidden is None and short.ticks == 2

    asyncio.run(go())


def test_runs_are_processes_of_their_own_side_by_side():
    async def go():
        r = runner(tmp(), jobs=3)
        runs = await r.score_many(SEED, [0, 1, 2], "retest")
        assert len(runs) == 3 and all(x.hidden is not None and x.recipe == SEED.id and x.ticks == 3 for x in runs)
        assert len({x.dir for x in runs}) == 3 and all((Path(x.dir) / "job.json").exists() for x in runs)
        pairs = await r.confirm_many(SEED, STREAM, [0, 1])
        assert [(a.recipe, b.recipe) for a, b in pairs] == [(SEED.id, STREAM.id)] * 2
        ok, why = await r.dry(STREAM)
        assert ok, why

    asyncio.run(go())


def test_a_screen_measures_a_candidate_against_what_the_parent_scored_so_far():
    async def go():
        r = runner(tmp())
        await r.generation(SEED, 0)
        r.standing = lambda rid: [{"score": 0.5, "progress": 0.25}, {"score": 0.7, "progress": 0.35}]
        ((a, b),) = await r.screen(SEED, [STREAM], 0)
        assert abs(a.score - 0.6) < 1e-9 and abs(a.progress - 0.3) < 1e-9 and a.cost == 0.0 and a.proposals == []
        assert b.recipe == STREAM.id and b.ticks == 3 and b.hidden is not None
        fresh = runner(tmp())  # no generation run to stand on: the parent runs beside the child
        ((a2, b2),) = await fresh.screen(SEED, [STREAM], 0)
        assert a2.recipe == SEED.id and Path(a2.dir).name == "parent" and a2.dir != b2.dir

    asyncio.run(go())


def test_a_run_that_dies_is_tried_again_and_counted():
    async def go():
        r = runner(tmp(), est_cost=0.5)
        real, calls = r._spawn, []

        async def flaky(recipe, seed, out, purpose, ticks, cost_budget):
            calls.append(out.name)
            return None if len(calls) == 1 else await real(recipe, seed, out, purpose, ticks, cost_budget)

        r._spawn = flaky
        x = await r.score(SEED, 0, "retest")
        assert x.score is not None and calls[1].endswith("-again1") and abs(r.spent - 1.0) < 1e-9  # the dead run and the live one

        async def dead(*_):
            return None

        r._spawn = dead
        try:
            await r.generation(SEED, 1)
            raise AssertionError("a generation run that cannot be run must stop the stage")
        except RunStopped:
            pass

    asyncio.run(go())


def test_a_planted_control_comes_first_and_the_menu_skips_what_cannot_act():
    got = candidates(SEED, Result(SEED.id, 0, "generation", 0.3, 0.2), Archive(tmp()), random.Random(1), 6,
                     planted=[[{"op": "set", "key": "stream", "value": "on"}]])
    assert got[0].origin["by"] == "control" and got[0].settings["stream"] is True and len({c.id for c in got}) == 6
    assert all(c.origin["by"] == "menu" and inert(SEED, c) is None for c in got[1:])
    assert not any(e["key"] in ("goal", "piece_share") for c in got[1:] for e in c.origin["edits"])  # idle while credit is off


def test_the_ladder_runs_on_fresh_runs_and_selection_never_reads_the_hidden_grade():
    async def go():
        out = tmp()
        archive, r = Archive(out / "archive"), runner(out / "runs")
        r.standing = lambda rid: archive.runs(rid, r.niche, FRESH)
        policy = Policy(screen_pairs=1, screen_metric="score", confirm_pairs=2, looks=(2,), retest_runs=0)
        res = await evolve(archive, r, SEED, 1, 2, policy, random.Random(0), log=quiet,
                           planted=[[{"op": "set", "key": "stream", "value": "on"}]])
        recs = archive.recipes(None, r.niche)
        assert {x["recipe"]["origin"].get("by") for x in recs} == {"seed", "control", "menu"} and len(recs) == 3
        assert all(d["stage"] in ("screen", "confirm", "kept") for d in archive.decisions()) and len(archive.decisions()) == 2
        graded = [x for x in archive.runs(None, r.niche) if x["purpose"] in ("generation", "confirm")]
        assert graded and all(x.get("hidden") is not None for x in graded)
        assert all(p["metric"] == "score" for p in archive.pairs())  # the held-out checks; hidden is only recorded
        assert res["niche"] == "code/-/open/n3" and res["seed"] == SEED.id

    asyncio.run(go())


def test_what_agents_propose_after_the_last_tick_becomes_a_candidate():
    async def go():
        out = tmp()
        f = code_frame(agents=3, ticks=3, fake=True, closing=True)
        assert fingerprint(f) == fingerprint(small())  # with or without the closing round, scores compare
        r = CodeRunner(f, {"kind": "mock", "proposes": {"key": "own_ticks", "value": "3"}}, out / "runs", stagger=0.0, log=quiet)
        archive = Archive(out / "archive")
        r.standing = lambda rid: archive.runs(rid, r.niche, FRESH)
        res = await evolve(archive, r, SEED, 1, 2, Policy(screen_pairs=1, screen_metric="score", confirm_pairs=0, retest_runs=0,
                                                          parent="tip"), random.Random(0), log=quiet)
        by = {x["recipe"]["origin"].get("by"): x for x in archive.recipes(None, r.niche)}
        assert set(by) == {"seed", "agents", "menu"}  # what the society proposed comes before the menu
        mine = by["agents"]["recipe"]
        assert [(e["op"], e["key"], e["value"]) for e in mine["origin"]["edits"]] == [("set", "own_ticks", 3)]
        assert mine["origin"]["backers"] == 2
        gen = [x for x in archive.runs(SEED.id, r.niche) if x["purpose"] == "generation"]
        plain = await run_once(small(), SEED, MockBackend(gen[0]["seed"]), gen[0]["seed"], out / "plain", "generation")
        assert gen[0]["score"] == plain.score and res["history"][0]["parent"] == SEED.id  # the same run without the round

    asyncio.run(go())


def test_the_menu_does_not_pay_to_change_a_rule_nobody_used():
    from types import SimpleNamespace
    from alphateam.recipe import unused
    quiet_run = SimpleNamespace(facts={"claims": 0, "tools": 0, "merges": 30}, ops={"lab.shell": 300, "message": 16, "lab.publish": 50},
                                lessons=[], proposals=[], actions=[], trials=[], dir="x")
    edit = lambda key, value: apply_edits(SEED, [{"op": "set", "key": key, "value": value}], {"by": "menu"})
    assert "own_ticks was not used" in unused(SEED, edit("own_ticks", 3), quiet_run)
    assert unused(SEED, edit("own", False), quiet_run) and unused(SEED, edit("toolbox", False), quiet_run)
    assert unused(SEED, edit("delegation", False), quiet_run)
    assert unused(SEED, edit("messages", False), quiet_run) is None  # agents did send messages
    assert unused(SEED, edit("credit", True), quiet_run) is None and unused(SEED, edit("stream", True), quiet_run) is None
    assert unused(SEED, edit("memory_ticks", 1), quiet_run) is None  # it names no counts: never judged this way
    busy = SimpleNamespace(facts={"claims": 2}, ops={"lab.claim": 2, "delegate.post": 1})
    assert unused(SEED, edit("own_ticks", 3), busy) is None and unused(SEED, edit("delegation", False), busy) is None
    assert unused(SEED, edit("own_ticks", 3), SimpleNamespace(facts={}, ops={})) is None  # nothing known about the run
    out = tmp()
    archive = Archive(out / "archive")
    archive.add(SEED, "seed", "code/-/open/n3")
    drawn = set()
    for i in range(12):
        for c in candidates(SEED, quiet_run, archive, random.Random(i), 4, agents=False):
            drawn |= {e["key"] for e in c.origin["edits"]}
    assert drawn and not drawn & {"own", "own_ticks", "toolbox", "delegation"} and {"stream", "credit"} <= drawn


def test_a_lesson_may_cite_the_worlds_receipts_and_lessons_can_be_tried_together():
    async def go():
        from types import SimpleNamespace
        from alphateam.recipe import heritage, lesson
        k = start(dict(small(), closing=True), SEED, 1)
        k.tick = 1
        k.begin_turn("A1")
        k.env.receipts["R7"] = {"agent": "A1"}
        text = "Before publishing, replay the logged checks and compare the bytes, not the rendered output."
        ok, msg = await k.act("A1", "rules", {"op": "lesson", "text": text, "evidence": ["r7"]})
        assert ok and len(k.rulebook.lessons) == 1 and k.rulebook.lessons[0]["evidence"] == ["r7"]  # a receipt is evidence
        ok, msg = await k.act("A1", "rules", {"op": "lesson", "text": text + " Also rebase first.", "evidence": ["R99"]})
        assert not ok and "no entry, receipt or tick R99" in msg
        # a society's lessons as one body of advice: most evidence first, as many as the size budget holds
        left = [dict(lesson(f"Lesson {i}: " + "keep shared interfaces stable and check them before every change. " * 5, f"case {i}",
                            ["E1"] * n, world="code"), by=f"A{i}", t=3, entry=f"E{i}")
                for i, n in ((1, 0), (2, 3), (3, 1), (4, 2), (5, 0), (6, 0), (7, 0), (8, 0))]
        gen = SimpleNamespace(lessons=left, proposals=[], actions=[], trials=[], ops={"lab.shell": 9}, facts={}, dir="run-x")
        archive = Archive(tmp() / "archive")
        archive.add(SEED, "seed", "code/-/open/n3")
        cands = candidates(SEED, gen, archive, random.Random(0), 3, advice=0, together=4)
        body = cands[0]
        assert body.origin["by"] == "agents" and body.origin["lessons"][:3] == [left[1]["id"], left[3]["id"], left[2]["id"]]
        assert 2 <= len(body.advice) <= 4 and len(heritage(body)) <= 2400 and "LESSONS FROM EARLIER SOCIETIES" in heritage(body)
        assert all(c.origin["by"] == "menu" for c in cands[1:])  # no lesson on its own was asked for
        one = candidates(SEED, gen, archive, random.Random(0), 2, advice=1, together=0)
        assert len(one[0].advice) == 1 and one[0].origin["lesson"] == left[0]["id"]  # as before: the first lesson alone

    asyncio.run(go())


def test_the_record_of_runs_is_an_index_into_what_happened_not_a_replacement():
    """The proposer of the next change reads the whole record (DESIGN.md section 22): each run's files are copied as
    they are, and the index and the timelines only say where to look."""
    from alphateam import record

    async def go():
        out = tmp()
        r = runner(out / "X" / "runs")
        runs = await r.score_many(SEED, [0], "eval") + await r.score_many(STREAM, [1], "eval")
        assert len(runs) == 2
        to = out / "record"
        rows = record.build([out / "X"], to, log=quiet)
        assert [x["change"] for x in rows] == ["starting rules", "stream=on"] and all(x["full"] for x in rows)
        row = rows[0]
        assert row["rounds"] == 3 and row["agents"] == 3 and row["held_out"] == round(runs[0].score * 179)
        assert row["landings"] == runs[0].facts["merges"] and row["hidden"] == runs[0].hidden
        folder = record.find(to, "eval-0001")
        for name in ("events.jsonl", "run.json", "result.json", "job.json", "timeline.md"):  # the run's own files, whole
            assert (folder / name).exists(), name
        assert (folder / "events.jsonl").read_bytes() == (Path(runs[0].dir) / "events.jsonl").read_bytes()
        assert not (folder / "work").exists()  # a run's work folder is never copied
        text = (folder / "timeline.md").read_text()
        assert "round  1 |" in text and "landed:" in text and "starting rules" in text
        lines = record.events(folder, round_=1, op="shell")
        assert lines and all(" r 1 " in x and "lab.shell" in x for x in lines)
        n = int(lines[0].split()[0])  # an event is cited by its line in events.jsonl
        assert '"op": "lab.shell"' in (folder / "events.jsonl").read_text().splitlines()[n - 1]
        assert "starting rules" in (to / "INDEX.md").read_text() and "stream=on" in (to / "INDEX.md").read_text()

    asyncio.run(go())


def test_after_the_last_tick_an_agent_can_read_its_runs_record_and_suggest_a_rule():
    """The societies' own side of proposing (DESIGN.md section 22): looking back is offered only once the shared work
    is final, so it cannot move a score, and a suggested rule reaches whoever maintains the rules."""
    from alphateam.primitives import tool_schemas
    from alphateam.recipe import CLOSING_TOOL, RULES_TOOL

    async def go():
        out = tmp()
        text = "A task's scope should not hold files that do not exist yet: offers on them waited four ticks here."
        f = code_frame(agents=3, ticks=3, fake=True, closing=True)
        r = CodeRunner(f, {"kind": "mock", "proposes": {"suggest": text}}, out / "runs", stagger=0.0, log=quiet)
        (res,) = await r.score_many(SEED, [0], "eval")
        (plain,) = await runner(out / "plain").score_many(SEED, [0], "eval")
        assert res.scores == plain.scores and res.hidden == plain.hidden  # nothing said after the last tick moves a score
        assert [(x["text"], x["evidence"], x["backers"]) for x in res.suggestions] == [(text, ["t1"], ["A1"])]
        assert res.ops.get("rules.record") == 1 and res.ops.get("rules.suggest") == 1 and not plain.suggestions
        archive = Archive(out / "archive")
        archive.record_voices(res, r.niche)
        (row,) = [json.loads(x) for x in (out / "archive" / "suggestions.jsonl").read_text().splitlines()]
        assert row["text"] == text and row["by"] == "A1" and row["recipe"] == SEED.id
        # during a run the action is what it was: its description and its ops are unchanged, and looking back is refused
        k = start(dict(f, grade=False), SEED, 0)
        (rules,) = [t for t in tool_schemas(k, "A1") if t["name"] == "rules"]
        assert rules["description"] == RULES_TOOL and rules["parameters"]["properties"]["op"]["enum"] == ["view", "propose", "back", "lesson"]
        ok, msg = await k.act("A1", "rules", {"op": "record", "what": "timeline"})
        assert not ok and "after the last tick" in msg
        ok, msg = await k.act("A1", "rules", {"op": "suggest", "text": text})
        assert not ok and "after the last tick" in msg
        # what an agent cites after looking at the record: a tick of the run, as well as entries and receipts
        k.tick, k.closing_now = 3, True
        ok, msg = await k.act("A1", "rules", {"op": "lesson", "text": "Offer narrow, checked changes early in a run.", "evidence": ["t2"]})
        assert ok, msg
        ok, msg = await k.act("A1", "rules", {"op": "lesson", "text": "Reconcile overlapping files before the gate.", "evidence": ["t9", "tick two"]})
        assert not ok and "t9" in msg and "evidence is a list of ids" in msg  # a tick that did not happen, and prose, are not evidence
        assert "record:" in CLOSING_TOOL and "suggest:" in CLOSING_TOOL and CLOSING_TOOL.startswith(RULES_TOOL)
        assert not any(w in CLOSING_TOOL.lower() for w in ("you should", "you must", "please"))  # it says what is on offer, no more

    asyncio.run(go())


def test_the_judge_uses_later_runs_only_counts_collapses_and_needs_both_baselines():
    import judge
    base = "Rbase"

    def row(rid, hidden, held=80):
        return {"run": "", "recipe": rid, "rules": rid, "by": "", "hidden": hidden, "held": held, "cost": 0.6}

    rows = [row(base, 560), row("Ra", 700), row("Rb", 650), row("Rc", 640, 90), row("Rd", 640, 70),
            row(base, 580), row(base, 540), row("Ra", 600), row("Ra", 620), row("Rb", 50), row("Rb", 640), row(base, 20)]
    pool = [row(base, 500), row(base, 620), row(base, 10)]
    got = {c["recipe"]: c for c in judge.judge(rows, pool, base)}
    a = got["Ra"]
    assert a["first"] == 700 and a["later"] == [600, 620] and a["mean"] == 610  # the run that picked it is not in its estimate
    assert a["verdict"] == "taken on" and all(low > 0 for _, low in a["gaps"].values())
    assert round(a["gaps"]["own"][0]) == 50 and round(a["gaps"]["pooled"][0]) == 50  # the baselines' collapsed runs are left out
    b = got["Rb"]
    assert b["collapsed"] == 1 and b["mean"] == 640 and b["verdict"].startswith("not shown") and "flagged" in b["verdict"]
    assert got["Rc"]["verdict"] == "one run: not judged"
    assert judge.pick(rows, base, 3) == ["Ra", "Rb", "Rc"]  # a tie on hidden tests goes to the one with more held-out checks
    # a candidate above its own round's baseline but not above the pooled one is not taken on
    rows2 = [row(base, 500), row(base, 510), row("Re", 700), row("Re", 560), row("Re", 570)]
    pool2 = [row(base, 640), row(base, 650), row(base, 660)]
    e = judge.judge(rows2, pool2, base)[0]
    assert e["gaps"]["own"][1] > 0 and not e["gaps"]["pooled"][1] > 0 and e["verdict"] == "not shown"
    assert "taken on" not in judge.report(rows2, pool2, base) and "starting rules, pooled" in judge.report(rows2, pool2, base)


def test_the_keyed_runner_starts_only_this_projects_tools():
    import keyed_runner as kr
    py = str(kr.ROOT / ".venv/bin/python")
    assert kr.allowed({"argv": [py, "tools/code_evolve.py", "--stage", "status"], "log": "runs_recipes/x.log"}) is None
    assert kr.allowed({"argv": ["/bin/sh", "-c", "echo hi"], "log": "runs_recipes/x.log"})
    assert kr.allowed({"argv": [py, "run_alphateam.py"], "log": "runs_recipes/x.log"})
    assert kr.allowed({"argv": [py, "tools/../run_alphateam.py"], "log": "runs_recipes/x.log"})
    assert kr.allowed({"argv": [py, "tools/code_evolve.py"], "log": "/tmp/x.log"})
    assert kr.allowed({"argv": [py], "log": "runs_recipes/x.log"})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
