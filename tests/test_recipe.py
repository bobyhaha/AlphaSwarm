"""Recipes: typed rules, edits, names kept out, compiling into a run, the rules action, and changes mid-run."""
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.kernel import Kernel  # noqa: E402
from alphateam.paired import frame, start  # noqa: E402
from alphateam.primitives import system_prompt, tool_schemas  # noqa: E402
from alphateam.recipe import (ADVICE_CHARS, MAX_ADVICE, REGISTRY, RecipeError, Recipe, Setting, anonymize, apply_edits, apply_live,  # noqa: E402
                           changed, enforced, heritage, inert, live_ok, menu_edits, options, registry, seed, to_cfg, validate)
from alphateam.runner import build  # noqa: E402


def refused(fn, *words):
    try:
        fn()
    except RecipeError as exc:
        assert all(w in str(exc) for w in words), (str(exc), words)
        return str(exc)
    raise AssertionError("not refused")


def test_the_seed_recipe_is_the_free_arm():
    """The registry's defaults compile to the options tools/launch_design.sh gives its free arm."""
    o = options(seed("design"))
    assert (o["authority"], o["ownership"], o["leadership"], o["contracts"], o["project_mult"]) == ("", "1", "1", "1", "2")
    assert (o["messages"], o["continuity"], o["review_every"], o["credit_every"], o["contest"]) == ("1", "2", "1", "3", "0")
    assert o["economy_share"] == "0" and o["board"] == "" and o["sync_items"] == "8"
    k = build(to_cfg(seed("design"), frame("open", agents=8)), 1, "hierarchical")
    assert k.env.ownership and k.env.leadership and k.env.contracts_on and not k.env.dictators and k.messages_on
    assert {"own_knob", "take_lead", "contract"} <= set(k.env.tool_names)


def test_a_revision_is_typed_and_bounded():
    r = seed("design")
    c = apply_edits(r, [{"op": "set", "key": "holders", "value": "shared"}, {"op": "set", "key": "shared_pay", "value": "0.5"}],
                    {"by": "test"})
    assert c.parent == r.id and c.generation == 1 and c.settings["holders"] == "shared" and c.settings["shared_pay"] == 0.5
    assert [e["part"] for e in c.origin["edits"]] == ["rights", "incentives"] and c.origin["edits"][0]["from"] == "none"
    assert enforced(c) and changed(r, c) == ["holders", "shared_pay"] and r.settings["holders"] == "none"
    refused(lambda: apply_edits(r, [{"op": "set", "key": "blocks", "value": 3}]), "no setting 'blocks'")
    refused(lambda: apply_edits(r, [{"op": "set", "key": "shared_pay", "value": 3}]), "0 to 1")
    refused(lambda: apply_edits(r, [{"op": "set", "key": "delivered", "value": 7.5}]), "whole numbers")
    for bad in ("inf", "-inf", "nan", "1e400", float("inf"), None, [], "seven"):  # nothing an agent types can crash a run
        refused(lambda bad=bad: apply_edits(r, [{"op": "set", "key": "delivered", "value": bad}]), "delivered takes")
        refused(lambda bad=bad: apply_edits(r, [{"op": "set", "key": "shared_pay", "value": bad}]), "shared_pay takes")
    refused(lambda: apply_edits(r, [{"op": "set", "key": "holders", "value": "none"}]), "already none")
    refused(lambda: apply_edits(r, [{"op": "set", "key": "take_lead", "value": "off"}], locked={"take_lead": "team"}), "fixed")
    refused(lambda: apply_edits(r, [{"op": "hire", "who": "A3"}]), "unknown edit")
    refused(lambda: apply_edits(r, []), "at least one edit")
    refused(lambda: Recipe("design", {"nope": 1}), "no setting 'nope'")
    refused(lambda: registry("chess"), "no rules are registered")


def test_a_recipe_is_named_by_its_content():
    a = apply_edits(seed("design"), [{"op": "set", "key": "messages", "value": False}])
    b = apply_edits(seed("design"), [{"op": "set", "key": "messages", "value": "off"}], {"by": "someone else"})
    assert a.id == b.id != seed("design").id
    back = apply_edits(a, [{"op": "set", "key": "messages", "value": True}])
    assert back.id == seed("design").id and back.generation == 2  # the same rules are the same recipe, whatever the path
    assert Recipe.from_json(a.to_json()).id == a.id and Recipe.from_json(a.to_json()).origin == a.origin
    paid = apply_edits(seed("design"), [{"op": "set", "key": "shared_pay", "value": 0.5}])
    zero = apply_edits(paid, [{"op": "set", "key": "shared_pay", "value": "-0"}])
    assert zero.id == seed("design").id and str(zero.settings["shared_pay"]) == "0.0"  # one zero: equal rules, one recipe
    before = seed("design").id, a.id
    REGISTRY["design"]["later"] = Setting("later", "procedures", "bool", False, "a setting added to the registry later")
    try:  # a recipe is named by what differs from the defaults, so a new setting does not rename the recipes on file
        assert (seed("design").id, Recipe.from_json(a.to_json()).id) == before
    finally:
        del REGISTRY["design"]["later"]


def test_a_recipe_never_names_an_agent_or_an_instance():
    assert anonymize("A12 told G3 that A1 holds it") == "an agent told a group that an agent holds it"
    assert anonymize("B3 and B11 share k7", "design") == "a block and a block share a knob"
    assert anonymize("agent 7, a3 and Group #2 read E12 after P3", "design") == "an agent, an agent and a group read an entry after a proposal"
    assert anonymize("knob 12, K4, block 3 and b9", "design") == "a knob, a knob, a block and a block"
    c = apply_edits(seed("design"), [{"op": "advise", "text": "Let A7 hold k3 and add up what B1 and B2 report before setting it",
                                      "when": "G2 shares a knob", "evidence": ["E4"]}])
    les = c.advice[0]
    assert les["text"] == "Let an agent hold a knob and add up what a block and a block report before setting it"
    assert les["when"] == "a group shares a knob" and les["status"] == "untested" and les["evidence"] == ["E4"]
    assert not enforced(c) and inert(seed("design"), c) is None
    named = Recipe("design", advice=[{"id": "L1", "text": "ask A3 first", "when": ""}])
    refused(lambda: validate(named), "may not name")
    refused(lambda: apply_edits(seed("design"), [{"op": "advise", "text": "too short"}]), "at least 20")
    refused(lambda: apply_edits(c, [{"op": "advise", "text": "Let A9 hold k1 and add up what B4 and B5 report before setting it",
                                     "when": "G7 shares a knob"}]), "already in the recipe")
    full = seed("design")
    for i in range(MAX_ADVICE):
        full = apply_edits(full, [{"op": "advise", "text": f"lesson number {i} about settling shared decisions"}])
    refused(lambda: apply_edits(full, [{"op": "advise", "text": "one lesson more than a recipe may carry"}]), "at most")
    assert len(apply_edits(full, [{"op": "unadvise", "id": full.advice[0]["id"]}]).advice) == MAX_ADVICE - 1


def test_a_designed_action_can_be_carried():
    act = {"name": "settle_knob", "description": "Collect costs from A2 and settle", "procedure": "1) message the users of a "
           "shared knob for their costs at each value. 2) add them up. 3) set the knob to the lowest total.", "params": ["knob"]}
    c = apply_edits(seed("design"), [{"op": "adopt", "action": act}])
    assert c.actions[0]["name"] == "settle_knob" and "an agent" in c.actions[0]["description"] and enforced(c)
    refused(lambda: apply_edits(c, [{"op": "adopt", "action": act}]), "already carries")
    refused(lambda: apply_edits(seed("design"), [{"op": "adopt", "action": {"name": "X", "procedure": "short"}}]), "name")
    assert apply_edits(c, [{"op": "unadopt", "name": "settle_knob"}]).actions == []
    k = start(frame("open", agents=4), c, 1)
    assert k.custom["settle_knob"].author == "heritage"


def test_a_change_that_cannot_matter_is_caught_for_free():
    r = seed("design")
    off = apply_edits(r, [{"op": "set", "key": "claims", "value": "off"}])
    assert inert(r, off) is None
    idle = apply_edits(off, [{"op": "set", "key": "claim_price", "value": 0.1}])
    assert "no effect while claims is off" in inert(off, idle)
    assert inert(r, apply_edits(r, [{"op": "set", "key": "claim_price", "value": 0.1}])) is None
    no_lead = apply_edits(r, [{"op": "set", "key": "take_lead", "value": "off"}])
    assert "lead_pay has no effect" in inert(no_lead, apply_edits(no_lead, [{"op": "set", "key": "lead_pay", "value": 3}]))
    assert inert(r, Recipe("design", parent=r.id)) == "it changes nothing"


def test_a_frame_holds_what_a_recipe_may_not_change():
    r = apply_edits(seed("design"), [{"op": "set", "key": "take_lead", "value": "off"}])
    chaired = to_cfg(r, frame("chaired", agents=8))
    assert chaired["world_opts"]["leadership"] == "team" and chaired["world_opts"]["chair"] == "1"  # the frame's lock wins
    opened = to_cfg(r, frame("open", agents=8, blocks=8))
    assert opened["world_opts"]["leadership"] == "" and opened["world_opts"]["blocks"] == "8" and opened["recipe"] == r.id
    assert opened["world_opts"]["economy"] == "1" and opened["world_opts"]["sync"] == "1"
    refused(lambda: to_cfg(r, dict(frame("open"), world="code")), "cannot run")


def test_every_agent_is_shown_all_of_a_recipes_advice():
    """A recipe's advice has a size budget, so nothing it carries goes unseen."""
    r, n = seed("design"), 0
    while True:
        try:
            r = apply_edits(r, [{"op": "advise", "text": f"lesson {n}: " + "settle a shared decision before acting on it. " * 6,
                                 "status": "confirmed" if n == 3 else "untested"}])
        except RecipeError as exc:
            assert "characters of advice" in str(exc)
            break
        n += 1
    assert 4 < n < MAX_ADVICE and len(heritage(r)) <= ADVICE_CHARS
    text = heritage(r)
    assert text.startswith("LESSONS FROM EARLIER SOCIETIES") and "[confirmed] lesson 3" in text.splitlines()[1]
    assert all(f"lesson {i}:" in text for i in range(n))
    k = start(frame("open", agents=4), r, 1)
    said = system_prompt(k)
    assert all(f"lesson {i}:" in said for i in range(n)) and not k.boards["library"].entries
    assert heritage(seed("design")) == "" and "LESSONS FROM" not in system_prompt(start(frame("open", agents=4), seed("design"), 1))


def test_the_rules_action_is_offered_only_under_a_recipe():
    plain = build(frame("open", agents=4), 1, "hierarchical")
    assert "rules" not in [t["name"] for t in tool_schemas(plain, "A1")] and "rules" not in plain.allowed
    plain.tick = 1
    ok, msg = asyncio.run(plain.act("A1", "rules", {"op": "view"}))
    assert not ok and "unknown action" in msg
    bare = Kernel(3)  # a kernel that lists it without a rulebook still refuses
    bare.allowed["rules"] = None
    assert "not open to proposals" in asyncio.run(bare.act("A1", "rules", {"op": "view"}))[1]
    assert "rules" in [t["name"] for t in tool_schemas(start(frame("open", agents=4), seed("design"), 1), "A1")]
    assert "rules" not in [t["name"] for t in tool_schemas(start(frame("single", agents=4), seed("design"), 1), "A1")]


def test_agents_propose_back_and_leave_lessons():
    async def go():
        k = start(frame("open", agents=8, ticks=6), seed("design"), 3)
        k.tick = 2
        k.begin_turn("A1")
        ok, view = await k.act("A1", "rules", {"op": "view"})
        assert ok and "holders = none [none | shared | all]" in view and "incentives:" in view
        e = k.post("global", "A2", "result", "B1 costs 40 at k3=2 and 31 at k3=5")
        ok, msg = await k.act("A1", "rules", {"op": "propose", "key": "holders", "value": "shared", "evidence": [e.id],
                                              "why": "three of us kept overwriting the same knob all run"})
        assert ok and "P1 proposed" in msg
        assert "your support was added" in (await k.act("A2", "rules", {"op": "propose", "key": "holders", "value": "shared",
                                                                         "why": "the same knob flipped every tick for us"}))[1]
        assert "3 backers" in (await k.act("A3", "rules", {"op": "back", "id": "p1"}))[1]
        await k.act("A4", "rules", {"op": "propose", "key": "shared_pay", "value": 0.5, "why": "my block gained while the total fell"})
        for bad, word in (({"op": "propose", "key": "holders", "value": "sideways", "why": "x" * 30}, "none | shared | all"),
                          ({"op": "propose", "key": "delivered", "value": "inf", "why": "x" * 30}, "delivered takes"),
                          ({"op": "propose", "key": "shared_pay", "value": 1e400, "why": "x" * 30}, "shared_pay takes"),
                          ({"op": "propose", "key": "blocks", "value": 3, "why": "x" * 30}, "no setting"),
                          ({"op": "propose", "key": "holders", "value": "none", "why": "x" * 30}, "already none"),
                          ({"op": "propose", "key": "contracts", "value": False, "why": "short"}, "say why"),
                          ({"op": "propose", "key": "contracts", "value": False, "why": "x" * 30, "evidence": ["E999"]}, "no entry"),
                          ({"op": "back", "id": "P9"}, "no proposal"), ({"op": "vote"}, "op is one of")):
            ok, msg = await k.act("A5", "rules", bad)
            assert not ok and word in msg, (bad, msg)
        ok, msg = await k.act("A4", "rules", {"op": "lesson", "text": "A7 held k3 and added up what B1 and B2 reported",
                                              "when": "a knob is shared", "evidence": [e.id]})
        assert ok
        x = k.rulebook.export()
        assert [(p["id"], p["key"], p["backers"]) for p in x["proposals"]] == [("P1", "holders", ["A1", "A2", "A3"]),
                                                                                ("P2", "shared_pay", ["A4"])]
        assert x["lessons"][0]["text"] == "an agent held a knob and added up what a block and a block reported"
        posts = [en.content for en in k.boards["global"].entries if en.kind in ("proposal", "lesson")]
        assert posts[0].startswith("Rule proposal P1: holders = shared (now none)") and "Lesson for later societies" in posts[-1]
        assert "PROPOSALS SO FAR" in k.rulebook.view() and [ev["op"] for ev in k.events if ev["ok"]][:2] == ["rules.view", "rules.propose"]
    asyncio.run(go())


def test_rules_can_change_in_a_running_society():
    r = seed("design")
    k = start(frame("open", agents=8), r, 2)
    env = k.env
    claimed = next(i for i in range(env.D) if str(env.level[i]) != "private")
    env.rights[claimed] = "A1"  # a claim made before the change
    held = apply_edits(r, [{"op": "set", "key": "holders", "value": "shared"}, {"op": "set", "key": "shared_pay", "value": 0.4},
                           {"op": "set", "key": "messages", "value": False}, {"op": "set", "key": "review_every", "value": 2}])
    assert live_ok(r, held) and apply_live(k, r, held) == ["holders", "messages", "shared_pay", "review_every"]
    shared = [i for i in range(env.D) if str(env.level[i]) != "private"]
    assert all(i in env.rights for i in shared) and env.rights[claimed] == "A1" and claimed not in env.dictated
    assert not any(str(env.level[i]) == "private" for i in env.dictated)
    assert env.economy_share == 0.4 and not k.messages_on and env.validate_every == 2 and k.recipe.id == held.id
    assert "held by one agent" in env.brief() and k.rulebook.recipe.id == held.id
    fresh = build(to_cfg(held, frame("open", agents=8)), 2, "hierarchical")  # the same draw as a society started under the rule
    assert all(fresh.env.rights[i] == h for i, h in env.dictated.items())
    freed = apply_edits(held, [{"op": "set", "key": "holders", "value": "none"}])
    apply_live(k, held, freed)
    assert env.rights == {claimed: "A1"} and not env.dictators
    late = apply_edits(r, [{"op": "set", "key": "take_lead", "value": "team"}])
    assert not live_ok(r, late)
    refused(lambda: apply_live(k, freed, late), "only change at the start")
    # authority by credit takes knobs over from their holders; switching it off gives them back
    k2 = start(frame("open", agents=8), held, 2)
    env2 = k2.env
    by_rule = dict(env2.dictated)
    credit = apply_edits(held, [{"op": "set", "key": "rights_by_credit", "value": True}])
    apply_live(k2, held, credit)
    taken = sorted(by_rule)[:3]
    for i in taken:
        env2.rights[i] = env2.credit_held[i] = "A8"
    apply_live(k2, credit, held)
    assert not env2.rights_by_credit and env2.credit_held == {} and {i: env2.rights[i] for i in by_rule} == by_rule


def test_the_rules_shown_are_the_rules_in_force():
    chaired = start(frame("chaired", agents=8), seed("design"), 2)
    view = chaired.rulebook.view()
    assert "take_lead = team (fixed in this run's frame)" in view and "holders = none (fixed in this run's frame)" in view
    assert chaired.rulebook.now("take_lead") == "team" and seed("design").settings["take_lead"] == "scope"
    priced = apply_edits(seed("design"), [{"op": "set", "key": "claim_price", "value": 0.05}, {"op": "set", "key": "claim_rent", "value": 0.02},
                                          {"op": "set", "key": "group_pay", "value": True}, {"op": "set", "key": "contracts", "value": False}])
    brief = start(frame("open", agents=8), priced, 2).env.brief()
    assert "Claiming a knob with own_knob costs 0.05" in brief and "Holding a knob costs 0.02" in brief
    assert "Groups also earn from their area" in brief and "pay each other through contracts" not in brief
    plain = start(frame("open", agents=8), seed("design"), 2).env.brief()
    assert "pay each other through contracts" in plain and "Claiming a knob" not in plain
    old = build(frame("single", agents=8), 2, "hierarchical").env.brief()  # without a recipe the brief is as it always was
    assert "Agents can pay each other through contracts" in old


def test_the_menu_draws_valid_edits():
    rng, r, reg = random.Random(4), seed("design"), registry("design")
    seen = set()
    for _ in range(300):
        e = menu_edits(r, rng, locked={"take_lead"})
        c = apply_edits(r, e, {"by": "menu"})  # every draw is a valid, typed edit
        assert e[0]["key"] != "take_lead" and reg[e[0]["key"]].coerce(e[0]["value"]) == c.settings[e[0]["key"]]
        seen.add(e[0]["key"])
    assert seen == set(reg) - {"take_lead"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
