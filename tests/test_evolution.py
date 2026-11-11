"""Evolving recipes: the statistics, the archive, the selection ladder, generations, and the report."""
import asyncio
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.archive import Archive, ArchiveMismatch, niche_of  # noqa: E402
from alphateam.generations import best, candidates, evolve, report, tip  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.paired import AlphaTeamRunner, Result, fingerprint, frame  # noqa: E402
from alphateam.recipe import apply_edits, lesson, seed  # noqa: E402
from alphateam.selection import Ladder, Policy, SyntheticRunner  # noqa: E402
from alphateam.stats import area, divergence, pairs_needed, summary, t_quantile  # noqa: E402

N = "design/coupled/open/n8"
SEED = seed("design")
quiet = lambda *a: None  # noqa: E731


def edit(key, value, parent=SEED, by="test"):
    return apply_edits(parent, [{"op": "set", "key": key, "value": value}], {"by": by})


def truth(r) -> float:
    """A known landscape on the design registry: one holder a shared knob is worth 0.4; shared pay up to 0.2."""
    s = r.settings
    return ((0.4 if s["holders"] == "shared" else 0.1 if s["holders"] == "all" else 0) + 0.2 * s["shared_pay"]
            + (0.1 if s["moving_facts"] else 0) - (0.2 if not s["messages"] else 0))


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="alphateam-evolution-"))


def test_statistics():
    for (p, df), want in {(.95, 1): 6.314, (.95, 2): 2.920, (.95, 3): 2.353, (.95, 9): 1.833, (.9, 3): 1.638, (.9, 29): 1.311,
                          (.975, 5): 2.571, (.99, 20): 2.528}.items():
        assert abs(t_quantile(p, df) - want) < 0.004, (p, df, t_quantile(p, df))
    s = summary([0.1, 0.2, 0.3, 0.4])
    assert (s["n"], s["mean"]) == (4, 0.25) and abs(s["lcb"] - (0.25 - 1.638 * 0.1291 / 2)) < 0.001 and s["lcb"] < s["mean"] < s["ucb"]
    assert summary([]) ["n"] == 0 and summary([0.3])["lcb"] is None and summary([0.3, None, 0.5])["n"] == 2
    assert pairs_needed(0.2, 0.45) == 24 and pairs_needed(0.4, 0.45) < 10 < pairs_needed(0.1, 0.45)
    assert divergence(Counter(a=5, b=5), Counter(a=50, b=50)) == 0 and divergence(Counter(a=3), Counter(b=3)) == 1
    assert 0 < divergence(Counter(a=5, b=5), Counter(a=8, b=2)) < 0.1
    assert area([0.0, 0.2, 0.4, 0.6], 1) == 0.4 and area([0.0, 0.2, 0.4, 0.6], 3) == 0.6 and area([None], 0) is None


def test_the_archive_keeps_everything_and_reloads():
    root = tmp()
    A = Archive(root)
    held, quiet_one = edit("holders", "shared"), edit("messages", False)
    assert A.add(SEED, "seed", N)["status"] == "seed" and A.add(held, "candidate", N)["status"] == "candidate"
    assert A.add(SEED, "candidate", N)["status"] == "seed"  # a recipe on file keeps its record
    A.add(quiet_one, "candidate", N)
    A.set_status(held.id, "kept", "paired gain +0.300")
    A.set_status(quiet_one.id, "dropped", "mean gain -0.2 over 4 pairs", "screen")
    for i, (a, b) in enumerate([(.3, .6), (.4, .7), (.2, .5)]):
        A.record_run(SEED.id, N, "confirm", i, a, a, 0.3)
        A.record_run(held.id, N, "confirm", i, b, b, 0.3)
        A.record_pair(SEED.id, held.id, N, "confirm", i, a, b)
    A.record_pair(SEED.id, held.id, N, "screen", 9, .1, .9, "progress")  # a screen is not part of a recipe's credit
    A.record_run(held.id, N, "screen", 9, .9, .9, 0.1)
    A.decide(parent=SEED.id, child=held.id, kept=True, stage="kept", why="paired gain +0.300")
    assert A.value(held.id, N)["mean"] == 0.6 and A.value(held.id, N)["n"] == 3 and A.gain(held.id)["mean"] == 0.3
    assert A.seen(edit("holders", "shared", by="another proposer"))["status"] == "kept"  # the same content, tried before
    assert A.seen(edit("holders", "all")) is None and A.children(SEED.id) == [held.id] and A.lineage(held.id) == [SEED.id, held.id]
    assert round(A.spent(), 2) == 1.9 and [r["id"] for r in A.recipes("dropped")] == [quiet_one.id]
    B = Archive(root)  # everything is on disk
    assert set(B.records) == set(A.records) and len(B.runs()) == 7 and len(B.pairs()) == 4 and len(B.decisions()) == 1
    assert B.get(held.id).id == held.id and B.records[quiet_one.id]["stage"] == "screen" and B.gain(held.id) == A.gain(held.id)
    assert held.id in B.table() and "holders=shared" in B.table() and "+0.300" in B.table()
    assert niche_of(frame("chaired", agents=16)) == "design/coupled/chaired/n16"


def test_parents_are_drawn_by_standing_and_spread_over_lineages():
    A = Archive(tmp())
    held, paid = edit("holders", "shared"), edit("shared_pay", 0.5)
    for r, status, scores in ((SEED, "seed", [.3, .35, .25]), (held, "kept", [.7, .75, .65]), (paid, "dropped", [.9, .9, .9])):
        A.add(r, status, N)
        for i, s in enumerate(scores):
            A.record_run(r.id, N, "confirm", i, s, s)
    rng = random.Random(0)
    draws = Counter(A.sample_parent(N, rng).id for _ in range(400))
    assert draws[held.id] > 2.5 * draws[SEED.id] > 0 and paid.id not in draws  # the better stands higher; a dead end never seeds
    assert [x["id"] for x in A.leaders(N)] == [held.id, SEED.id]
    for i in range(6):  # a recipe that already has many children is drawn less
        A.add(edit("delivered", 9 + i, held), "kept", N)
    busy = Counter(A.sample_parent(N, rng, explore=0).id for _ in range(400))
    assert busy[held.id] < draws[held.id]
    B, lucky = Archive(tmp()), edit("moving_facts", True)
    for r, status, scores in ((SEED, "seed", [.3, .35, .25]), (held, "kept", [.7, .75, .65]), (lucky, "kept", [.99])):
        B.add(r, status, N)
        for i, s in enumerate(scores):
            B.record_run(r.id, N, "generation", i, s, s)
    B.record_run(lucky.id, N, "report", 1, 0.99, 0.99)  # a report run is no part of a recipe's standing
    assert B.needs_retest(lucky.id, N, 3) and not B.needs_retest(held.id, N, 3)
    assert B.value(lucky.id, N)["n"] == 1 and B.value(lucky.id, N)["lcb"] is None
    one = Counter(B.sample_parent(N, rng, explore=0).id for _ in range(600))
    assert one[held.id] > one[lucky.id] > one[SEED.id]  # one high run does not make a leader
    assert [x["id"] for x in B.leaders(N)] == [held.id, SEED.id, lucky.id]
    try:
        A.sample_parent("design/coupled/open/n64", rng)
    except LookupError:
        return
    raise AssertionError("a parent was drawn from an empty niche")


def test_an_archive_belongs_to_one_niche_and_frame():
    root = tmp()
    A, f = Archive(root), frame("open", agents=8)
    assert niche_of(f) == N
    A.bind(N, fingerprint(f))
    A.bind(N, fingerprint(f))  # the same niche and frame again: it carries on
    for niche, other in ((niche_of(frame("open", agents=16)), frame("open", agents=16)), (N, frame("open", agents=8, ticks=12))):
        try:
            Archive(root).bind(niche, fingerprint(other))
        except ArchiveMismatch as exc:
            assert "scores would not compare" in str(exc)
        else:
            raise AssertionError("an archive took runs from another frame")
    assert [A.next("retest"), A.next("retest"), A.next("generation")] == [1, 2, 1]
    assert Archive(root).next("retest") == 3  # a later session never redraws an instance
    A.add(SEED, "seed", N)
    for i, s in enumerate([.3, .35, .25]):
        A.record_run(SEED.id, N, "retest", i, s, s)
    standing = A.value(SEED.id, N)
    for purpose in ("report", "screen", "probe", "trial_run"):  # none of these is a plain run of the recipe from scratch
        A.record_run(SEED.id, N, purpose, 7, 0.99, 0.99)
    assert A.value(SEED.id, N) == standing and standing["n"] == 3


def test_the_lesson_library_merges_and_ranks():
    A = Archive(tmp())
    a = apply_edits(SEED, [{"op": "advise", "text": "Agree who settles a shared knob before anyone sets it", "when": "blocks share knobs"}]).advice[0]
    b = apply_edits(SEED, [{"op": "advise", "text": "Post each simulation with the values it was run at"}]).advice[0]
    A.add_lesson(a, N, "run-1")
    A.add_lesson(b, N, "run-1")
    assert A.add_lesson(a, "design/coupled/open/n16", "run-2")["count"] == 2  # the same lesson left again counts
    A.mark_lesson(b["id"], "confirmed", "paired gain +0.2")
    assert [x["id"] for x in A.lessons()] == [b["id"], a["id"]]  # confirmed first
    assert [x["id"] for x in A.lessons(query="who settles a shared knob")] == [a["id"], b["id"]]  # then relevance
    assert len(A.lessons(k=1)) == 1 and len(A.lessons()) == 2  # nothing is cut unless asked
    again = Archive(A.root).lessons()
    assert again[0]["status"] == "confirmed" and again[1]["count"] == 2 and again[1]["niches"] == [N, "design/coupled/open/n16"]


def ladder(truth_fn=truth, policy=None, seed_=1, **kw):
    A = Archive(tmp())
    R = SyntheticRunner(truth_fn, seed=seed_, niche=N, **kw)
    A.add(SEED, "seed", N)
    return A, R, Ladder(A, R, policy or Policy(), random.Random(seed_), quiet)


def test_a_real_gain_is_kept_and_a_useless_change_is_dropped():
    async def go():
        A, R, L = ladder(sd=0.3)
        held, noop, worse = edit("holders", "shared"), edit("delivered", 12), edit("messages", False)
        d = {x.child: x for x in await L.consider(SEED, [held, noop, worse])}
        assert d[held.id].kept and d[held.id].stage == "kept" and "lower bound" in d[held.id].why
        assert not d[worse.id].kept and d[worse.id].stage == "screen" and not d[noop.id].kept
        assert A.records[held.id]["status"] == "kept" and A.records[worse.id]["status"] == "dropped"
        g = A.gain(held.id)
        assert g["n"] == 30 and g["lcb"] > 0 and abs(g["mean"] - 0.4) < 0.15
        assert R.count["confirm"] == 60 and R.count["screen"] == 4 * 4 and R.count["probe"] == 0  # one finalist confirmed
        assert len(A.decisions()) == 3 and A.decisions()[-1]["kept"] and len(A.pairs(held.id, "screen")) == 4
        again = await L.consider(SEED, [edit("holders", "shared", by="someone else"), edit("messages", False)])
        assert [x.stage for x in again] == ["typed", "typed"] and "tried before" in again[1].why  # a dead end is not tried twice
        assert R.count["confirm"] == 60
    asyncio.run(go())


def test_a_revision_that_was_not_judged_stays_a_candidate():
    async def go():
        A, R, L = ladder(sd=0.05)
        held, paid = edit("holders", "shared"), edit("shared_pay", 0.5)
        d = {x.child: x for x in await L.consider(SEED, [held, paid])}
        assert d[held.id].kept and not d[paid.id].kept and d[paid.id].stage == "screen"
        assert "another revision screened better" in d[paid.id].why and A.records[paid.id]["status"] == "candidate"
        assert {x["child"]: x["judged"] for x in A.decisions()} == {held.id: True, paid.id: False}
        again = await L.consider(SEED, [paid])  # it only lost its place: it is taken up again and judged on its own
        assert again[0].kept and A.records[paid.id]["status"] == "kept" and A.gain(paid.id)["lcb"] > 0
        for bad in (Policy(confirm_pairs=1), Policy(mode="vote")):
            try:
                Ladder(A, R, bad)
            except ValueError:
                continue
            raise AssertionError("a ladder took a policy that cannot judge")
    asyncio.run(go())


def test_a_first_pass_on_single_runs_takes_the_best_on_as_a_lead():
    async def go():
        single = Policy(screen_pairs=1, confirm_pairs=0, retest_runs=0, parent="tip")
        A, R, L = ladder(sd=0.02, policy=single)
        held, worse, paid = edit("holders", "shared"), edit("messages", False), edit("shared_pay", 0.5)
        d = {x.child: x for x in await L.consider(SEED, [held, worse, paid])}
        assert d[held.id].kept and d[held.id].stage == "lead" and "not confirmed" in d[held.id].why
        assert A.records[held.id]["status"] == "lead" and R.count["confirm"] == 0 and not A.pairs(purpose="confirm")
        assert d[worse.id].stage == "screen" and A.records[worse.id]["status"] == "candidate"  # one run settles nothing
        assert "not ahead of the parent so far" in d[worse.id].why
        assert A.records[paid.id]["status"] == "candidate"  # it beat the parent too; another went on in its place
        again = {x.child: x for x in await L.consider(SEED, [worse])}  # its second run: now it is judged on both
        assert A.records[worse.id]["status"] == "dropped" and "over 2 runs" in again[worse.id].why
        assert tip(A, N) == held.id  # a lead seeds the next generation
        assert {x["child"]: (x["stage"], x["kept"]) for x in A.decisions()}[held.id] == ("lead", True)
        for bad in (Policy(screen_pairs=0, confirm_pairs=0), Policy(parent="oldest")):
            try:
                Ladder(A, R, bad)
            except ValueError:
                continue
            raise AssertionError("a ladder took a policy that cannot judge")
        B = Archive(tmp())  # generations of single runs climb from the newest lead, and mark no lesson
        out = await evolve(B, SyntheticRunner(truth, sd=0.02, seed=2, niche=N), SEED, generations=3, per_generation=3,
                           policy=single, rng=random.Random(4), log=quiet, planted=[[{"op": "set", "key": "holders", "value": "shared"}]])
        gens = out["history"]
        assert gens[0]["parent"] == SEED.id and gens[0]["leads"] and gens[1]["parent"] == gens[0]["leads"][0]
        assert out["best"] == tip(B, N) and out["lineage"][0] == SEED.id and len(out["lineage"]) >= 2
        assert not B.runs(purpose="confirm") and all(x["status"] != "kept" for x in B.recipes(None, N))
        assert all(les.get("status", "untested") == "untested" for les in B.lessons(N))
        C = Archive(tmp())  # a generation can be started from a named recipe
        await evolve(C, SyntheticRunner(truth, sd=0.02, seed=3, niche=N), SEED, generations=1, per_generation=2, policy=single,
                     rng=random.Random(5), log=quiet, planted=[[{"op": "set", "key": "holders", "value": "shared"}]])
        lead = tip(C, N)
        res = await evolve(C, SyntheticRunner(truth, sd=0.02, seed=4, niche=N), SEED, generations=1, per_generation=2,
                           policy=single, rng=random.Random(6), log=quiet, start_from=SEED.id)
        assert lead != SEED.id and res["history"][0]["parent"] == SEED.id
    asyncio.run(go())


def test_in_a_single_run_pass_what_is_ahead_gets_a_second_run_before_it_is_taken_on():
    async def go():
        from alphateam.generations import ahead
        racing = Policy(screen_pairs=1, confirm_pairs=0, retest_runs=0, parent="tip", lead_runs=2, retests=2)
        A, R, L = ladder(sd=0.02, policy=racing)
        A.record_run(SEED.id, N, "generation", 1, 0.4, 0.4, 0.3)  # the parent's own run: what revisions are measured against
        held, worse, paid = edit("holders", "shared"), edit("messages", False), edit("shared_pay", 0.5)
        d = {x.child: x for x in await L.consider(SEED, [held, worse, paid])}
        assert not any(x.kept for x in d.values()) and "needs 2 runs to be taken on" in d[held.id].why  # nobody leads on one run
        assert {A.records[c.id]["status"] for c in (held, worse, paid)} == {"candidate"}
        front = ahead(A, SEED, "progress")  # the metric this policy screens on
        assert [rec["id"] for rec, _, _ in front] == [held.id, paid.id] and front[0][1] > front[1][1] > 0 and front[0][2] == 1
        gen = type("Run", (), {"lessons": [], "proposals": [], "actions": [], "trials": [], "ops": {}, "facts": {}, "dir": "x"})()
        nxt = candidates(SEED, gen, A, random.Random(1), 4, retests=2, metric="progress")
        assert [c.id for c in nxt[:2]] == [held.id, paid.id]  # what is ahead comes back first, the best first
        assert all(A.seen(c) is None for c in nxt[2:]) and len(nxt) == 4  # the menu then only explores what was never run
        assert worse.id not in {c.id for c in nxt}
        d2 = {x.child: x for x in await L.consider(SEED, nxt)}
        assert d2[held.id].kept and d2[held.id].stage == "lead" and "after 2 runs of its own" in d2[held.id].why
        assert A.records[held.id]["status"] == "lead" and len(A.pairs(held.id, "screen")) == 2
        assert A.records[paid.id]["status"] == "candidate" and "another revision screened better" in d2[paid.id].why
        assert not A.pairs(purpose="confirm") and R.count["confirm"] == 0
        B = Archive(tmp())  # the whole loop: nothing is taken on in the first generation, the best of it in the second
        out = await evolve(B, SyntheticRunner(truth, sd=0.02, seed=2, niche=N), SEED, generations=3, per_generation=3,
                           policy=racing, rng=random.Random(4), log=quiet, planted=[[{"op": "set", "key": "holders", "value": "shared"}]])
        h = out["history"]
        assert not h[0]["leads"] and h[1]["leads"] and h[1]["parent"] == SEED.id and h[2]["parent"] == h[1]["leads"][0]
        assert B.get(h[1]["leads"][0]).settings["holders"] == "shared"
    asyncio.run(go())


def test_a_lead_needs_a_margin_and_an_earlier_parents_revisions_stay_in_the_race():
    async def go():
        from alphateam.generations import ahead
        strict = Policy(screen_pairs=1, confirm_pairs=0, retest_runs=0, parent="tip", lead_runs=2, retests=2, lead_min=0.2)
        A, R, L = ladder(sd=0.01, policy=strict)
        A.record_run(SEED.id, N, "generation", 1, 0.4, 0.4, 0.3)
        held, paid = edit("holders", "shared"), edit("shared_pay", 0.5)  # worth +0.4 and +0.1 in this landscape
        await L.consider(SEED, [held, paid])
        d = {x.child: x for x in await L.consider(SEED, [held, paid])}
        assert d[held.id].kept and d[held.id].stage == "lead"  # two runs, ahead by more than the margin
        assert not d[paid.id].kept and "not by the 0.200 it takes" in d[paid.id].why  # two runs, ahead by less: set aside
        assert A.records[paid.id]["status"] == "candidate"
        lead = A.get(held.id)
        A.record_run(lead.id, N, "generation", 2, 0.45, 0.45, 0.3)  # the lead's own run came in low: paid is ahead of that mark
        assert [rec["id"] for rec, _, _ in ahead(A, lead, "progress")] == [paid.id]  # a revision of the seed is still in the race
        gen = type("Run", (), {"lessons": [], "proposals": [], "actions": [], "trials": [], "ops": {}, "facts": {}, "dir": "x"})()
        nxt = candidates(lead, gen, A, random.Random(2), 2, retests=2, metric="progress")
        assert nxt[0].id == paid.id and nxt[0].settings["holders"] == SEED.settings["holders"]  # the recipe as filed, not
        d3 = {x.child: x for x in await L.consider(lead, nxt[:1])}                               # its edit on top of the lead
        assert len(A.pairs(paid.id, "screen")) == 3 and "over 3 runs" in d3[paid.id].why  # all its runs count, under either parent
    asyncio.run(go())


def test_selection_holds_its_error_rates_under_noise():
    """Over many ladders at the spread of saved runs: a useless revision is rarely kept, a real one usually is."""
    async def rate(effect, n=60):
        kept = runs = 0
        for s in range(n):
            A, R, L = ladder(lambda r, e=effect: e if r.settings["holders"] == "shared" else 0.0,
                             Policy(screen_pairs=0), seed_=100 + s, sd=0.45, rho=0.3)
            d = (await L.consider(SEED, [edit("holders", "shared")]))[0]
            kept += d.kept
            runs += d.runs
        return kept / n, runs / n
    null, null_runs = asyncio.run(rate(0.0))
    real, real_runs = asyncio.run(rate(0.4))
    assert null <= 0.15 and real >= 0.85, (null, real)
    assert null_runs < 45 < 55 < real_runs <= 60, (null_runs, real_runs)  # a gain that is not there is given up early


def test_the_cheap_stages_drop_before_anything_is_paid():
    async def go():
        broken, still = edit("contracts", False), apply_edits(SEED, [{"op": "advise", "text": "Be thoughtful and work well together always"}])
        idle = edit("claim_price", 0.1, edit("claims", "off"))
        A, R, L = ladder(broken=[broken.id], still=[still.id])
        A.add(edit("claims", "off"), "kept", N)
        d = await L.consider(SEED, [broken, still])
        assert [(x.stage, x.runs) for x in d] == [("dry", 0), ("probe", 3)] and "acted as they did" in d[1].why
        d2 = await L.consider(edit("claims", "off"), [idle])
        assert d2[0].stage == "dry" and "no effect while claims is off" in d2[0].why
        assert R.count["confirm"] == R.count["screen"] == 0
        moved = apply_edits(SEED, [{"op": "advise", "text": "Agree who settles a shared knob before anyone sets it"}])
        d3 = await L.consider(SEED, [moved])  # advice that moves behaviour goes on to be scored like any revision
        assert d3[0].stages[1]["stage"] == "probe" and d3[0].stages[1]["passed"] and d3[0].stage in ("screen", "confirm", "kept")
    asyncio.run(go())


def test_the_control_arm_keeps_at_random_and_scores_nothing():
    async def go():
        A, R, L = ladder(policy=Policy(mode="random", keep_p=1.0))
        d = await L.consider(SEED, [edit("messages", False), edit("holders", "shared"), edit("delivered", 12)])
        assert sum(x.kept for x in d) == 1 and all("control" in x.why for x in d)
        assert R.count["screen"] == R.count["confirm"] == R.count["probe"] == 0 and R.count["dry"] == 3
        A2, R2, L2 = ladder(policy=Policy(mode="random", keep_p=0.0))
        assert not any(x.kept for x in await L2.consider(SEED, [edit("holders", "shared")]))
    asyncio.run(go())


def test_a_leader_is_rerun_before_it_seeds_a_generation():
    async def go():
        A, R, L = ladder()
        assert await L.ensure_tested(SEED) == 3 and await L.ensure_tested(SEED) == 0
        assert [r["purpose"] for r in A.runs(SEED.id)] == ["retest"] * 3
    asyncio.run(go())


def test_candidates_come_from_the_society_first():
    gen = Result(SEED.id, 1, "generation", 0.5, 0.4, dir="run-7",
                 proposals=[{"id": "P1", "t": 3, "key": "holders", "value": "shared", "why": "knobs kept flipping", "evidence": ["E3"],
                             "backers": ["A1", "A2"]},
                            {"id": "P2", "t": 4, "key": "take_lead", "value": "off", "why": "nobody used it", "evidence": [], "backers": ["A4"]},
                            {"id": "P3", "t": 5, "key": "delivered", "value": 12, "why": "too much to read", "evidence": [], "backers": ["A5"]}],
                 lessons=[lesson("Agree who settles a shared knob before anyone sets it", evidence=["E9"], world="design")],
                 actions=[{"name": "settle_shared", "description": "ask the holder", "params": [], "adopters": 3, "uses": 5,
                           "procedure": "View the design, then ask the holder of each shared knob to set it"},
                          {"name": "lonely_tool", "description": "used by its author only", "params": [], "adopters": 1, "uses": 9,
                           "procedure": "Simulate the block twice and post both results to the team"}])
    A, locked = Archive(tmp()), {"take_lead": "team"}
    A.record_voices(gen, N)
    c = candidates(SEED, gen, A, random.Random(1), 6, locked=locked)
    assert [x.origin["by"] for x in c] == ["agents", "agents", "agents", "agents", "menu", "menu"]
    assert c[0].settings["holders"] == "shared" and c[0].origin["backers"] == 2 and c[0].origin["runs"] == 1
    assert c[1].settings["delivered"] == 12  # the proposal on a setting this arm locks was left out
    assert c[2].advice and c[2].origin["lesson"] == c[2].advice[0]["id"] and c[2].origin["run"] == "run-7"  # a lesson is a revision
    assert [x["name"] for x in c[3].actions] == ["settle_shared"] and c[3].origin["adopters"] == 3  # so is an action several adopted
    assert len({x.id for x in c}) == 6 and all(x.parent == SEED.id for x in c)
    assert [x.origin["by"] for x in candidates(SEED, gen, A, random.Random(1), 3, locked=locked, agents=False)] == ["menu"] * 3
    again = Result(SEED.id, 2, "retest", 0.4, 0.3, dir="run-8",
                   proposals=[{"id": "P1", "t": 2, "key": "delivered", "value": 12, "why": "the same trouble", "evidence": ["E1"],
                               "backers": ["A2"]},
                              {"id": "P2", "t": 0, "key": "messages", "value": False, "why": "said before the rules changed",
                               "evidence": [], "backers": ["A3"]}])
    A.record_voices(again, N)  # proposals are pooled over every run under the recipe
    assert [(x["key"], x["runs"], x["backers"]) for x in A.proposals(SEED.id)] == [("delivered", 2, 2), ("holders", 1, 2),
                                                                                  ("take_lead", 1, 1)]
    first = candidates(SEED, gen, A, random.Random(1), 1, locked=locked)[0]
    assert first.settings["delivered"] == 12 and first.origin["runs"] == 2  # what came up in two runs goes first
    A.add(c[0], "candidate", N)  # still waiting to be judged: it can be taken up again
    A.add(c[1], "dropped", N)  # judged: it is not made again
    later = {x.id for x in candidates(SEED, gen, A, random.Random(1), 6, locked=locked)}
    assert c[0].id in later and c[1].id not in later
    tried = Result(SEED.id, 1, "generation", 0.5, 0.4, trials=[{"n": 1, "key": "shared_pay", "value": 0.5, "kept": True},
                                                                {"n": 2, "key": "messages", "value": False, "kept": False}])
    first = candidates(SEED, tried, A, random.Random(1), 2)[0]
    assert first.origin["by"] == "trial" and first.settings["shared_pay"] == 0.5 and first.settings["messages"] is True


def test_evolution_climbs_a_known_landscape_and_the_control_does_not():
    async def go():
        gains = {}
        for mode in ("paired", "random"):
            A = Archive(tmp())
            R = SyntheticRunner(truth, sd=0.45, rho=0.3, seed=1, niche=N)
            out = await evolve(A, R, SEED, generations=8, per_generation=4, policy=Policy(mode=mode), rng=random.Random(3), log=quiet)
            top = A.get(out["best"])
            gains[mode] = truth(top)
            assert out["lineage"][0] == SEED.id and out["lineage"][-1] == top.id and len(out["history"]) == 8
            if mode == "paired":
                assert top.settings["holders"] == "shared" and best(A, N, 3) == top.id
                assert all(A.gain(rid)["lcb"] > 0 for rid in out["lineage"][1:])  # every step of the lineage was confirmed
                rep = await report(A, SyntheticRunner(truth, sd=0.45, rho=0.3, seed=9, niche=N), {"seed": SEED, "evolved": top},
                                   runs=20, log=quiet)
                assert rep["evolved - seed"]["n"] == 20 and rep["evolved - seed"]["lcb"] > 0
                assert len(A.runs(purpose="report")) == 40
            else:
                assert R.count["confirm"] == 0 and tip(A, N) == top.id
        assert gains["paired"] >= 0.4 > gains["random"]
    asyncio.run(go())


def test_the_whole_loop_runs_on_scripted_societies():
    """Mechanics only: scripted agents do not propose, so candidates come from the menu, and nothing here says how
    societies of models would score."""
    async def go():
        root = tmp()
        f = frame("open", agents=8, ticks=8)
        A = Archive(root / "archive")
        R = AlphaTeamRunner(f, lambda s: MockBackend(s), root / "runs")
        out = await evolve(A, R, SEED, generations=2, per_generation=3, rng=random.Random(5), log=quiet,
                           policy=Policy(screen_pairs=2, confirm_pairs=4, looks=(2, 4), retest_runs=2))
        assert out["niche"] == N and len(A.decisions()) == 6 and len(A.runs(SEED.id, N, "generation")) == 2
        assert len(A.runs(SEED.id, N, "retest")) == 2 and all(r["score"] is not None for r in A.runs())
        assert (root / "runs" / "generation-0001" / "run.json").exists() and any((root / "runs" / "dry").glob("*/result.json"))
        reg = __import__("alphateam.recipe", fromlist=["registry"]).registry("design")
        for p in A.pairs(purpose="screen"):  # a change that can be made mid-run was screened as a branch, forked at the half
            key = A.records[p["child"]]["recipe"]["origin"]["edits"][0]["key"]
            assert p["since"] == (4 if reg[key].live is not None else 0), (key, p["since"])
        standing = A.value(SEED.id, N)
        rep = await report(A, R, {"seed": SEED, "evolved": A.get(out["best"])}, runs=2, log=quiet)
        assert rep["seed"]["n"] == rep["evolved"]["n"] == 2 and "evolved - seed" in rep
        assert A.value(SEED.id, N) == standing and len(A.runs(purpose="report")) == 4  # report runs move no recipe's standing
        reloaded = Archive(root / "archive")
        assert len(reloaded.decisions()) == 6 and set(reloaded.records) == set(A.records)
        try:  # the archive belongs to this frame: runs of another length would not compare
            await evolve(reloaded, AlphaTeamRunner(frame("open", agents=8, ticks=6), lambda s: MockBackend(s), root / "other"), SEED,
                         generations=1, log=quiet)
        except ArchiveMismatch:
            pass
        else:
            raise AssertionError("an archive took runs from another frame")
        menu = Archive(root / "menu")  # the menu-only control: the same loop with no candidate from the society
        out2 = await evolve(menu, AlphaTeamRunner(f, lambda s: MockBackend(s), root / "menu-runs"), SEED, generations=1, per_generation=2,
                            rng=random.Random(5), log=quiet, agents=False,
                            policy=Policy(screen_pairs=1, confirm_pairs=2, looks=(2,), retest_runs=2))
        assert out2["agents"] is False and {x["by"] for x in menu.decisions()} == {"menu"}
    asyncio.run(go())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
