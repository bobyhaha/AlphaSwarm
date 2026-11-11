"""Incentive mechanisms: credit in the design world, a practice split with feedback, and selection with experiments."""
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.bench.base import Task  # noqa: E402
from alphateam.evolve import lineage  # noqa: E402
from alphateam.kernel import Kernel  # noqa: E402
from alphateam.mock import MockBackend  # noqa: E402
from alphateam.primitives import system_prompt  # noqa: E402
from alphateam.runner import build  # noqa: E402
from test_bench_base import Toy  # noqa: E402

CFG = {"condition": "self_org", "agents": 8, "ticks": 6, "actions": 4, "attention": 3000, "world": "design",
       "world_opts": {"credit": "1"}, "seed": 3, "structures": ["design"], "episodes": 1, "population": 1}


def test_credit_adds_up_and_is_shown_only_at_reviews():
    k = build(CFG, 0, "design")
    w = k.env
    assert "Credit:" in w.brief()
    k.tick = 1
    gid = None
    assert asyncio.run(k.act("A1", "connect", {"op": "form_group", "name": "pair", "purpose": "settle knobs"}))[0]
    gid = next(iter(k.groups))
    for n, (aid, ref, v, as_) in enumerate([("A1", 0, 3, None), ("A2", 1, 9, None), ("A1", 0, 8, gid), ("A3", 2, 0, None)]):
        args = {"knob": w.name[ref], "value": v} | ({"as": as_} if as_ else {})
        ok, msg = asyncio.run(k.act(aid, "set_knob", args))
        assert ok, msg
    total = sum(c for who, (c, _) in w.credit.items() if who.startswith("A"))
    assert abs(total - w.normalized(w.x)) < 1e-9  # agents' credits add up to the score
    assert abs(w.credit[gid][0] - w.sets[2]["gain"]) < 1e-4 and w.credit[gid][1] == 1
    assert "Your credit" not in w.observe(k, "A1")  # nothing shown before a review
    k.tick = 3
    w.on_tick_end(k)
    assert "Your credit" in w.observe(k, "A1")
    ledger = [e for e in k.boards["global"].entries if e.content.startswith("Credit ledger")]
    assert ledger and gid in ledger[-1].content and "A1" in ledger[-1].content
    assert any("earned" in n for n in map(str, k.agents["A1"].inbox))
    off = build(dict(CFG, world_opts={}), 0, "design")
    assert "Credit:" not in off.env.brief()
    off.tick = 3
    off.env.on_tick_end(off)
    assert not [e for e in off.boards["global"].entries if e.content.startswith("Credit ledger")]


def toy(feedback: bool, tmp: Path):
    tasks = [Task("", f"multiply {a} by {b}", f"fam{a % 3}", f"dom{a % 2}", f"What is {a}*{b}?",
                  {"gold": str(a * b), "ab": (a, b)}) for a, b in [(i, i + 7) for i in range(1, 11)]]
    w = Toy(4, tasks, seed=1, home=2)
    w.workdir = str(tmp)
    w.set_practice(4, feedback, 2, seed=5)
    k = Kernel(4, max_actions=4)
    k.env = w
    w.setup(k)
    return k, w


def test_practice_split_scores_held_out_only(tmp=Path("/tmp/alphateam_practice")):
    k, w = toy(True, tmp)
    assert len(w.practice) == 4 and "Practice:" in w.brief() and "nobody sees grades" not in w.brief()
    p = sorted(w.practice)[0]
    h = next(t for t in w.tasks if t not in w.practice)
    k.tick = 2
    assert asyncio.run(k.act("A1", "submit", {"task": p, "answer": w.tasks[p].meta["gold"]}))[0]
    assert asyncio.run(k.act("A2", "submit", {"task": h, "answer": w.tasks[h].meta["gold"]}))[0]
    asyncio.run(w.on_tick_end(k))
    post = [e for e in k.boards["global"].entries if e.content.startswith("Practice review")][-1].content
    assert f"{p} 1.00 (by A1" in post and not re.search(rf"\b{h}\b", post)
    assert any("practice" in line for line in w.search(k, "A1", "bench", [p.lower()]))
    asyncio.run(w.finalize(k))
    ev = w.evaluations[-1]
    assert abs(ev["score"] - 1 / 6) < 1e-3 and abs(ev["practice"] - 1 / 4) < 1e-3  # 6 held-out tasks, 4 practice
    k2, w2 = toy(False, tmp)  # the control: same split, nothing shown
    assert w2.practice == w.practice and "nobody sees grades" in w2.brief()
    k2.tick = 2
    asyncio.run(w2.on_tick_end(k2))
    assert not [e for e in k2.boards["global"].entries if e.content.startswith("Practice review")]
    assert not any("practice" in line for line in w2.search(k2, "A1", "bench", []))


def test_selection_lineage_assigns_experiments(out=Path("/tmp/alphateam_selection")):
    shutil.rmtree(out, ignore_errors=True)
    cfg = dict(CFG, episodes=2, population=3, ticks=3)
    seen = []

    class Spy(MockBackend):
        async def turn(self, k, aid):
            if aid == "A1":
                seen.append((k.env.x_start.tolist(), getattr(k, "stakes", ""), system_prompt(k)))
            await super().turn(k, aid)

    summary = asyncio.run(lineage(cfg, Spy(0), out, {"condition": "self_org"}))
    assert len(summary) == 6 and sum(s["selected"] for s in summary) == 2
    gen1 = [s for s in seen if "EXPERIMENT" not in s[1] and "carries on" not in s[1]]
    assert len({json.dumps(s[0]) for s in gen1}) == 1  # one world instance per generation
    assert all("SELECTION: 3 societies" in s[2] for s in seen)
    gen2 = summary[3:]
    assert gen2[0]["experiment"] is None and all(s["experiment"] for s in gen2[1:])
    assert len({s["experiment"] for s in gen2[1:]}) >= 1
    data = json.loads((out / "lineage.json").read_text())
    assert len(data["reviews"]) == 1 and len(data["reviews"][0]["experiments"]) == 2
    assert "Experiments for the next generation" in (out / "lineage.md").read_text()
    lib = json.loads((out / "ep02-soc02" / "run.json").read_text())
    assert lib["meta"]["experiment"] and lib["meta"]["world_seed"] == 1003


def test_teamwork_strategy_is_enforced():
    from alphateam.org_learning import OrgLearner, clean, clusters
    cfg = dict(CFG, agents=14, world_opts={"blocks": "14", "supers": "4", "ownership": "1"})
    k = build(cfg, 0, "design")
    w = k.env
    comps = clusters(w)
    assert sorted(sorted(c) for c in comps) == sorted(sorted(b for b in range(14) if w.blocks[b]["super"] == s) for s in range(4))
    s = clean({"teams": "clusters", "council": True, "rights": "team_shared", "explore": 1, "results_to": "team",
               "norms": "settle k12 first with A3", "delegate_role": "collect member results"})
    assert "k12" not in s["norms"] and "A3" not in s["norms"]
    L = OrgLearner(3)
    k.tick, k.total_ticks = 3, 20
    L.apply(k, s)
    teams = [g for g in k.active_groups() if g.name.startswith("Team")]
    council = [g for g in k.active_groups() if g.name == "Council"]
    assert len(teams) == 4 and len(council) == 1 and "TEAMWORK STRATEGY" in system_prompt(k)
    inter = [i for i in range(w.D) if str(w.level[i]) == "interface"]
    tops = [i for i in range(w.D) if str(w.level[i]) == "top"]
    assert all(w.rights[i] in {g.id for g in teams} for i in inter) and all(w.rights[i] == council[0].id for i in tops)
    assert not any(str(w.level[i]) == "private" and i in w.rights for i in range(w.D))
    k.tick = 4
    ok, msg = asyncio.run(k.act("A1", "set_knob", {"knob": w.name[inter[0]], "value": 2}))
    assert not ok and "exploration phase" in msg
    k.tick = 5
    holder = w.rights[inter[0]]
    outsider = next(a for a in k.agents if not k.can_act_for(a, holder))
    insider = next(a for a in k.agents if k.can_act_for(a, holder))
    assert not asyncio.run(k.act(outsider, "set_knob", {"knob": w.name[inter[0]], "value": 2}))[0]
    assert asyncio.run(k.act(insider, "set_knob", {"knob": w.name[inter[0]], "value": 2}))[0]
    k.begin_turn(insider)
    assert asyncio.run(k.act(insider, "simulate", {}))[0]
    assert any(e.content.startswith("Simulation") for e in k.boards[f"mem:{w.default_to[insider]}"].entries)
    L.apply(k, clean({"teams": "none", "rights": "team_shared"}))  # no teams: shared knobs go to single users
    assert not [g for g in k.active_groups() if g.name.startswith("Team")]
    assert all(w.rights[i].startswith("A") for i in inter + tops)
    L.apply(k, clean({}))
    assert not w.rights and not k.active_groups()


def test_continual_learning_in_a_mock_run(out=Path("/tmp/alphateam_continual")):
    shutil.rmtree(out, ignore_errors=True)
    cfg = dict(CFG, agents=8, ticks=9, org_window=3, world_opts={"ownership": "1"})
    summary = asyncio.run(lineage(cfg, MockBackend(0), out, {"condition": "self_org"}))
    data = json.loads((out / "run.json").read_text())
    hist = data["meta"]["org_learning"]
    assert [h["t"] for h in hist] == [3, 6] and hist[0]["decision"] == "baseline"
    assert hist[0]["next"]["teams"] == "clusters" and len(hist[0]["reflections"]) == 8
    entries = [json.loads(line) for line in open(out / "entries.jsonl")]
    assert any(e["author"] == "organizer" and e["kind"] == "decision" for e in entries)
    assert any(e["kind"] == "lesson" and e["content"].startswith("Reflection at t3") for e in entries)


def test_calls_limit_allows_many_actions_per_reply():
    from types import SimpleNamespace
    from alphateam.llm import LLMBackend, OpenAICompatible
    from test_alphateam import FakeCompletions, oa_reply
    k = build(dict(CFG, calls=2), 0, "design")
    k.tick, k.total_ticks = 1, 5
    writes = [("write", json.dumps({"board": "private", "content": f"note {i}"})) for i in range(6)]
    completions = FakeCompletions([oa_reply(calls=writes), oa_reply(calls=writes[:2]), oa_reply(content="done", finish="stop")])
    backend = LLMBackend(OpenAICompatible("openrouter", client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
                                          run_id="test"))
    asyncio.run(backend.turn(k, "A1"))
    first, second, closing = completions.requests
    assert first["tool_choice"] == "auto" and second["tool_choice"] == "auto" and closing["tool_choice"] == "none"
    assert sum(e.content.startswith("note") for e in k.boards["private:A1"].entries) == 8  # nothing refused
    assert "replies left this tick: 0" in closing["messages"][-1]["content"]
    assert "replies this tick" in first["messages"][1]["content"] and "what is limited is your replies" in first["messages"][0]["content"]


def test_durable_credit_rewards_what_lasts_and_shares_with_groups():
    k = build(dict(CFG, world_opts={"credit": "durable", "credit_every": "3", "ownership": "1"}), 0, "design")
    w = k.env
    assert "Credit and resources" in w.brief()
    k.tick = 1
    i, j, m, n = 0, 1, 2, 3
    v = lambda x: int(w.x[x] + 3) % 11
    assert asyncio.run(k.act("A5", "connect", {"op": "form_group", "name": "pair", "purpose": "p"}))[0]
    gid = next(iter(k.groups))
    assert asyncio.run(k.act("A6", "connect", {"op": "join", "target": gid}))[0]
    start = w.x.copy()
    for aid, knob, val, as_ in [("A1", i, v(i), None), ("A2", j, v(j), None), ("A3", j, (v(j) + 2) % 11, None),
                                ("A4", m, v(m), None), ("A4", m, int(start[m]), None), ("A5", n, v(n), gid)]:
        ok, msg = asyncio.run(k.act(aid, "set_knob", {"knob": w.name[knob], "value": val} | ({"as": as_} if as_ else {})))
        assert ok, msg
    k.tick = 3
    w.on_tick_end(k)

    def lasting(knob):
        y = w.x.copy()
        y[knob] = start[knob]
        return (w.cost(y) - w.cost(w.x)) / (w.c_start - w.c_opt)
    assert abs(w.dcredit["A1"][0] - lasting(i)) < 1e-9 and abs(w.dcredit["A3"][0] - lasting(j)) < 1e-9
    assert "A2" not in w.dcredit and w.dcredit["A4"] == [0.0, 1]  # overwritten earns nothing; reverted earns 0
    assert abs(w.dcredit[gid][0] - lasting(n)) < 1e-9 and "A5" not in w.dcredit
    assert abs(w.effective["A5"] - lasting(n) / 2) < 1e-9 and abs(w.effective["A6"] - w.effective["A5"]) < 1e-12
    for a in ("A1", "A3", "A5", "A7"):
        assert w.sim_budget(a) == max(1, min(9, 3 + round(w.effective[a] / 0.05)))
    assert any("overwritten" in str(x) for x in k.agents["A2"].inbox)
    assert any(e.content.startswith("Durable credit at t3") for e in k.boards["global"].entries)
    assert "simulations per tick" in w.observe(k, "A1")


def test_evidence_share_and_claim_rent():
    k = build(dict(CFG, world_opts={"credit": "durable", "credit_every": "3", "ownership": "1", "evidence_share": "0.5",
                                    "claim_rent": "0.01"}), 0, "design")
    w = k.env
    assert "Evidence earns credit" in w.brief() and "costs 0.01" in w.brief()
    k.tick = 1
    gain = {}
    for v in range(11):  # find a value of knob 0 that improves the design
        y = w.x.copy(); y[0] = v
        gain[v] = w.cost(w.x) - w.cost(y)
    best = max(gain, key=gain.get)
    assert gain[best] > 0
    user = next(a for a, bs in w.instruments.items() if 0 in [int(i) for b in bs for i in w.blocks[b]["idx"]])
    other = next(a for a in w.instruments if a != user)
    k.begin_turn(user)
    assert asyncio.run(k.act(user, "simulate", {"changes": {w.name[0]: best}, "to": "global"}))[0]
    assert asyncio.run(k.act(other, "own_knob", {"knob": w.name[5]}))[0]
    assert asyncio.run(k.act(other, "set_knob", {"knob": w.name[0], "value": best}))[0]
    k.tick = 3
    w.on_tick_end(k)
    total = gain[best] / (w.c_start - w.c_opt)
    assert abs(w.dcredit[user][0] - total / 2) < 1e-9
    assert abs(w.dcredit[other][0] - (total / 2 - 0.01)) < 1e-9  # half the credit, minus rent for k5
    assert any("informed" in str(x) for x in k.agents[user].inbox)


def test_group_area_credit():
    k = build(dict(CFG, world_opts={"credit": "durable", "credit_every": "3", "group_area": "1"}), 0, "design")
    w = k.env
    assert "earn from their area" in w.brief()
    k.tick = 1
    assert asyncio.run(k.act("A1", "connect", {"op": "form_group", "name": "pair", "purpose": "p"}))[0]
    gid = next(iter(k.groups))
    assert asyncio.run(k.act("A2", "connect", {"op": "join", "target": gid}))[0]
    area = sorted({b for a in ("A1", "A2") for b in w.instruments[a]})
    knob = int(w.blocks[area[0]]["idx"][0])
    best = min(range(11), key=lambda v: sum(w.block_cost(b, __import__("numpy").where(__import__("numpy").arange(w.D) == knob, v, w.x)) for b in area))
    start = w.x.copy()
    assert asyncio.run(k.act("A3", "set_knob", {"knob": w.name[knob], "value": best}))[0]
    k.tick = 3
    w.on_tick_end(k)
    gain = sum(w.block_cost(b, start) - w.block_cost(b, w.x) for b in area) / (w.c_start - w.c_opt)
    assert abs(w.dcredit[gid][0] - gain) < 1e-9 and abs(w.effective["A1"] - gain / 2) < 1e-9
    assert "A1" not in w.dcredit  # members earn through the group, not by moving knobs


def test_claim_price_spends_credit():
    k = build(dict(CFG, world_opts={"credit": "durable", "credit_every": "3", "ownership": "1", "claim_price": "0.02"}), 0, "design")
    w = k.env
    assert "costs 0.02 credit" in w.brief()
    k.tick = 1
    ok, msg = asyncio.run(k.act("A1", "own_knob", {"knob": w.name[0]}))
    assert not ok and "costs 0.02 credit" in msg
    w.dcredit["A1"] = [0.05, 1]
    assert asyncio.run(k.act("A1", "own_knob", {"knob": w.name[0]}))[0]
    assert abs(w.dcredit["A1"][0] - 0.03) < 1e-12 and w.rights[0] == "A1"


def test_top_credit_rights():
    from alphateam.org_learning import OrgLearner, clean
    k = build(dict(CFG, world_opts={"credit": "durable", "ownership": "1"}), 0, "design")
    w = k.env
    k.tick, k.total_ticks = 3, 20
    w.effective = {a: 0.0 for a in w.instruments}
    w.effective["A5"] = 0.4
    L = OrgLearner(3)
    L.apply(k, clean({"rights": "top_credit"}))
    shared = [i for i in range(w.D) if len(w.users[i]) > 1]
    a5 = [i for i in shared if set(w.users[i]) & set(w.instruments["A5"])]
    assert a5 and all(w.rights[i] == "A5" for i in a5)
    assert all(w.rights[i] in w.instruments for i in shared)


def test_stalled_window_forces_a_change(out=Path("/tmp/alphateam_stall")):
    shutil.rmtree(out, ignore_errors=True)
    cfg = dict(CFG, agents=8, ticks=10, org_window=3, org_stall=True, world_opts={"ownership": "1"})
    asyncio.run(lineage(cfg, MockBackend(0), out, {"condition": "self_org"}))
    hist = json.loads((out / "run.json").read_text())["meta"]["org_learning"]
    assert hist[0]["decision"] == "baseline" and all(h["decision"].split(":")[0] in ("kept", "stalled", "reverted") for h in hist[1:])


def test_delta_judge(out=Path("/tmp/alphateam_delta")):
    shutil.rmtree(out, ignore_errors=True)
    cfg = dict(CFG, agents=8, ticks=10, org_window=3, org_stall=True, org_judge="delta", world_opts={"ownership": "1"})
    asyncio.run(lineage(cfg, MockBackend(0), out, {"condition": "self_org"}))
    hist = json.loads((out / "run.json").read_text())["meta"]["org_learning"]
    assert all("delta" in h for h in hist) and hist[0]["decision"] == "baseline"


def test_rights_follow_credit():
    k = build(dict(CFG, world_opts={"credit": "durable", "credit_every": "3", "rights_by_credit": "1"}), 0, "design")
    w = k.env
    assert "Authority follows credit" in w.brief()
    k.tick = 3
    w.dcredit["A5"] = [0.3, 1]
    w.on_tick_end(k)
    shared = [i for i in range(w.D) if len(w.users[i]) > 1]
    mine = [i for i in shared if set(w.users[i]) & set(w.instruments["A5"])]
    assert mine and all(w.rights[i] == "A5" for i in mine)
    assert not any(i in w.rights for i in shared if i not in mine)  # no other user has positive credit
    w.dcredit["A5"] = [-0.1, 1]
    k.tick = 6
    w.on_tick_end(k)
    assert not any(i in w.rights for i in shared)


def test_take_lead_nests_into_a_hierarchy_and_contracts_bind():
    cfg = dict(CFG, agents=14, world_opts={"blocks": "14", "supers": "4", "credit": "durable", "leadership": "1",
                                           "contracts": "1", "local": "1"})
    k = build(cfg, 0, "design")
    w = k.env
    assert "take_lead" in w.tool_names and "contract" in w.tool_names and "Leadership:" in w.brief() and "Local" in w.brief()
    k.tick = 1
    a = next(x for x in w.instruments if w.blocks[w.instruments[x][0]]["super"] == 0)
    b0 = w.instruments[a][0]
    sup = sorted(b for b in range(14) if w.blocks[b]["super"] == 0)
    other = next(x for x in w.instruments if w.instruments[x][0] in sup and w.instruments[x][0] != b0)
    ok, msg = asyncio.run(k.act(a, "take_lead", {}))
    assert ok, msg
    private = [i for i in range(w.D) if set(w.users[i]) == {b0}]
    assert private and all(w.rights[i] == a for i in private)
    ok, msg = asyncio.run(k.act(other, "take_lead", {"scope": [w.block_label[b] for b in sup]}))
    assert ok, msg
    inter = [i for i in range(w.D) if str(w.level[i]) == "interface" and set(w.users[i]) <= set(sup)]
    assert inter and all(w.rights[i] == other for i in inter) and all(w.rights[i] == a for i in private)
    assert w.lead_depth() == 2 and w.superior(a) == other
    k.begin_turn(other)
    ok, msg = asyncio.run(k.act(other, "simulate", {"area": True}))
    assert ok and "Area simulation" in msg, msg
    third = next(x for x in w.instruments if w.instruments[x][0] == b0 and x != a) if \
        any(w.instruments[x][0] == b0 and x != a for x in w.instruments) else None
    challenger = next(x for x in w.instruments if w.instruments[x][0] in sup and x not in (a, other))
    ok, msg = asyncio.run(k.act(challenger, "take_lead", {"scope": [w.block_label[b] for b in sup]}))
    assert not ok and "more credit" in msg
    w.effective[challenger] = 0.5
    assert asyncio.run(k.act(challenger, "take_lead", {"scope": [w.block_label[b] for b in sup]}))[0]
    assert all(w.rights[i] == challenger for i in inter)
    b2 = w.instruments[other][0]
    assert asyncio.run(k.act(challenger, "take_lead", {"op": "appoint", "to": other, "scope": [w.block_label[b2]]}))[0]
    assert w.leads[frozenset({b2})] == other
    # contracts: pay another agent to set and lock a knob
    w.dcredit[a] = [0.2, 1]
    knob = private[0]
    target = (int(w.x[knob]) + 3) % 11
    ok, msg = asyncio.run(k.act(other, "contract", {"op": "offer", "to": a, "knob": w.name[knob], "value": target, "pay": 0.1}))
    assert not ok and "credit" in msg  # other has no credit to pay
    w.dcredit[other] = [0.3, 1]
    ok, msg = asyncio.run(k.act(other, "contract", {"op": "offer", "to": a, "knob": w.name[knob], "value": target,
                                                     "ticks": 3, "pay": 0.1}))
    assert ok, msg
    cid = next(iter(w.contracts))
    assert asyncio.run(k.act(a, "contract", {"op": "accept", "id": cid}))[0]
    assert w.x[knob] == target and abs(w.dcredit[a][0] - 0.3) < 1e-9 and abs(w.dcredit[other][0] - 0.2) < 1e-9
    ok, msg = asyncio.run(k.act(a, "set_knob", {"knob": w.name[knob], "value": (target + 1) % 11}))
    assert not ok and "locked by contract" in msg
    ok, msg = asyncio.run(k.act(a, "write", {"board": "global", "content": "hello"}))
    assert not ok  # local communication


def test_economy_pays_individuals_and_project_leads():
    import numpy as np
    cfg = dict(CFG, agents=14, world_opts={"blocks": "14", "supers": "4", "economy": "1", "credit_every": "3",
                                           "leadership": "1", "contracts": "1"})
    k = build(cfg, 0, "design")
    w = k.env
    assert "YOUR REWARD is your own credit" in w.brief() and "nobody is rewarded for the total" in w.brief()
    k.tick = 1
    sup = sorted(b for b in range(14) if w.blocks[b]["super"] == 0)
    lead = next(x for x in w.instruments if w.instruments[x][0] in sup)
    assert asyncio.run(k.act(lead, "take_lead", {"scope": [w.block_label[b] for b in sup]}))[0]
    start = w.x.copy()
    i = next(i for i in range(w.D) if str(w.level[i]) == "interface" and set(w.users[i]) <= set(sup))
    best = min(range(11), key=lambda v: sum(w.block_cost(b, np.where(np.arange(w.D) == i, v, w.x)) for b in sup))
    assert asyncio.run(k.act(lead, "set_knob", {"knob": w.name[i], "value": best}))[0]
    k.tick = 3
    w.on_tick_end(k)
    gap = w.c_start - w.c_opt
    mine = sum(w.block_cost(b, start) - w.block_cost(b, w.x) for b in w.instruments[lead]) / gap
    proj = 2 * sum(w.block_cost(b, start) - w.block_cost(b, w.x) for b in sup) / gap
    assert abs(w.dcredit[lead][0] - (mine + proj)) < 1e-9
    member = next(x for x in w.instruments if w.instruments[x][0] in sup and x != lead)
    assert abs(w.dcredit[member][0] - sum(w.block_cost(b, start) - w.block_cost(b, w.x) for b in w.instruments[member]) / gap) < 1e-9


def test_team_leadership_nests_recursively():
    cfg = dict(CFG, agents=14, world_opts={"blocks": "14", "supers": "4", "economy": "1", "credit_every": "3",
                                           "leadership": "team", "contracts": "1"})
    k = build(cfg, 0, "design")
    w = k.env
    assert "Teams: take_lead founds a team" in w.brief()
    k.tick = 1
    sup = sorted(b for b in range(14) if w.blocks[b]["super"] == 0)
    agents = [x for x in sorted(w.instruments, key=lambda a: int(a[1:])) if w.instruments[x][0] in sup]
    ceo, b, c = agents[0], agents[1], agents[2]
    ok, msg = asyncio.run(k.act(ceo, "take_lead", {"name": "Lab"}))
    assert ok, msg
    team = next(g for g in w.teams)
    own = [i for i in range(w.D) if set(w.users[i]) <= set(w.instruments[ceo])]
    assert own and all(w.rights[i] == team for i in own)  # the team holds its members' private knobs
    w.dcredit[ceo] = [0.5, 1]
    for x in (b, c):
        assert asyncio.run(k.act(ceo, "contract", {"op": "hire", "to": x, "pay": 0.01, "ticks": 9}))[0]
        cid = list(w.contracts)[-1]
        ok, msg = asyncio.run(k.act(x, "contract", {"op": "accept", "id": cid}))
        assert ok, msg
    assert k.inside(b, team) and k.inside(c, team)
    ok, msg = asyncio.run(k.act(b, "take_lead", {"name": "Sub-lab"}))  # a member founds a sub-team inside the team
    assert ok and "sub-team" in msg, msg
    sub = next(g for g in w.teams if g != team)
    assert sub in k.groups[team].members and k.inside(b, team) and w.team_depth(k) >= 2
    b_only = [i for i in range(w.D) if set(w.users[i]) <= set(w.instruments[b])]
    assert all(w.rights[i] == sub for i in b_only)
    shared = [i for i in range(w.D) if set(w.users[i]) <= set(w.instruments[ceo]) | set(w.instruments[b]) | set(w.instruments[c])
              and len(set(w.users[i])) > 1]
    assert all(w.rights[i] == team for i in shared)
    ok, msg = asyncio.run(k.act(c, "set_knob", {"knob": w.name[own[0]], "value": 3}))
    assert not ok  # only the lead (or its delegates) decides the team's knobs
    assert asyncio.run(k.act(ceo, "set_knob", {"knob": w.name[own[0]], "value": 3}))[0]  # the lead still does details
    k.tick = 3
    w.on_tick_end(k)
    assert any("salary" in str(n) for n in k.agents[c].inbox)
    ok, msg = asyncio.run(k.act(c, "contract", {"op": "quit"}))
    assert ok and not k.inside(c, team)


def test_substrate_search_ranks_by_relevance_and_citations():
    k = build(dict(CFG, world_opts={"substrate": "1"}), 0, "design")
    k.tick = 1
    assert k.substrate and "shared substrate of linked posts" in system_prompt(k)
    for a in ("A1", "A2", "A3", "A4", "A5"):
        k.begin_turn(a)
    asyncio.run(k.act("A1", "write", {"board": "global", "content": "k12 should be low for B3 evidence"}))
    e1 = k.boards["global"].entries[-1].id
    asyncio.run(k.act("A2", "write", {"board": "global", "content": "agree k12 low", "refs": [e1]}))
    asyncio.run(k.act("A3", "write", {"board": "global", "content": "k12 low confirmed for B9", "refs": [e1]}))
    asyncio.run(k.act("A4", "write", {"board": "global", "content": "unrelated k40 note"}))
    assert k.cited[e1] == 2
    ok, msg = asyncio.run(k.act("A5", "search", {"what": "posts", "query": "k12 low evidence", "limit": 3}))
    assert ok and msg.splitlines()[1].startswith(f"- {e1}") and "cited 2x" in msg and "k40" not in msg, msg


def test_mixed_society_routes_strong_agents():
    from types import SimpleNamespace
    from alphateam.llm import LLMBackend, OpenAICompatible
    from test_alphateam import FakeCompletions, oa_reply
    weak = FakeCompletions([oa_reply(content="w", finish="stop")])
    strong = FakeCompletions([oa_reply(content="s", finish="stop")])
    mk = lambda c, r: OpenAICompatible("openrouter", reasoning=r, client=SimpleNamespace(chat=SimpleNamespace(completions=c)), run_id="t")
    backend = LLMBackend(mk(weak, False), 4, mk(strong, True), {"A2"})
    k = build(dict(CFG), 0, "design")
    k.tick, k.total_ticks = 1, 3
    asyncio.run(backend.turn(k, "A1"))
    asyncio.run(backend.turn(k, "A2"))
    assert len(weak.requests) == 1 and len(strong.requests) == 1
    assert strong.requests[0]["extra_body"]["reasoning"] != {"enabled": False}


def test_scope_teams_nest():
    cfg = dict(CFG, agents=14, world_opts={"blocks": "14", "supers": "4", "economy": "1", "leadership": "1", "scope_teams": "1"})
    k = build(cfg, 0, "design")
    w = k.env
    k.tick = 1
    sup = sorted(b for b in range(14) if w.blocks[b]["super"] == 0)
    agents = [x for x in sorted(w.instruments, key=lambda a: int(a[1:])) if w.instruments[x][0] in sup]
    big, small = agents[0], next(x for x in agents if w.instruments[x][0] != w.instruments[agents[0]][0])
    assert asyncio.run(k.act(big, "take_lead", {"scope": [w.block_label[b] for b in sup]}))[0]
    g_big = w.scope_groups[frozenset(sup)]
    assert all(k.inside(x, g_big) for x in agents) and k.groups[g_big].delegates == {big}
    assert asyncio.run(k.act(small, "take_lead", {}))[0]
    g_small = w.scope_groups[frozenset(w.instruments[small])]
    assert g_small in k.groups[g_big].members and small not in k.groups[g_big].members and k.depth(g_big) == 2
    assert asyncio.run(k.act(small, "take_lead", {"op": "step_down"}))[0]
    assert k.groups[g_small].dissolved is not None and small in k.groups[g_big].members


def test_profit_sharing():
    cfg = dict(CFG, agents=14, world_opts={"blocks": "14", "supers": "4", "economy": "1", "economy_share": "0.5", "credit_every": "3"})
    k = build(cfg, 0, "design")
    w = k.env
    assert "Profit sharing" in w.brief()
    k.tick = 1
    start = w.x.copy()
    i = 0
    best = min(range(11), key=lambda v: w.cost(__import__("numpy").where(__import__("numpy").arange(w.D) == i, v, w.x)))
    assert asyncio.run(k.act("A1", "set_knob", {"knob": w.name[i], "value": best}))[0]
    k.tick = 3
    w.on_tick_end(k)
    gap = w.c_start - w.c_opt
    social = 0.5 * (w.cost(start) - w.cost(w.x)) / gap
    a = "A2"
    own = sum(w.block_cost(b, start) - w.block_cost(b, w.x) for b in w.instruments[a]) / gap
    assert abs(w.dcredit[a][0] - (own + social)) < 1e-9


def test_arc_world_parallel_copies():
    k = build(dict(CFG, world="arc", agents=4, world_opts={"games": "vc33,ls20", "steps": "5"}), 0, "arc")
    w = k.env
    assert len(w.games) == 2 and set(w.assigned.values()) == set(w.games) and "unfamiliar games" in w.brief()
    k.tick = 1
    a = next(x for x, g in w.assigned.items() if g.startswith("ls20"))
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["ACTION1", "ACTION2", 3]}))
    assert ok and "cells changed" in msg, msg
    ok, msg = asyncio.run(k.act(a, "look", {"region": [0, 0, 7, 3]}))
    assert ok and "rows 0-3" in msg and len(msg.splitlines()) >= 5, msg
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["ACTION9"]}))
    assert not ok and "unknown action" in msg
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["ACTION1"] * 4}))
    assert ok and "steps left this tick: 0" in msg
    b = next(x for x, g in w.assigned.items() if g == w.assigned[a] and x != a) if \
        sum(g == w.assigned[a] for g in w.assigned.values()) > 1 else None
    if b:  # parallel copies: one agent's moves leave another's copy untouched
        assert w.used[b] == 0 and (b, w.assigned[a]) in w.copies
    assert 0 <= w.score()["score"] <= 1


def test_arc_click_strings_and_publish_brief():
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "steps": "5", "publish_solutions": "1"}), 0, "arc")
    w = k.env
    assert "Solved levels are published" in w.brief()
    action, data, label = w._parse("click(4, 32)")
    assert data == {"x": 4, "y": 32} and label == "click(4,32)"


def test_arc_replay_reproduces_published_solution():
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "steps": "15", "publish_solutions": "1"}), 0, "arc")
    w = k.env
    k.tick = 1
    a, b = sorted(w.assigned, key=lambda x: int(x[1:]))
    sol = ("click(12,20) click(9,20) click(12,20) click(22,20) click(26,20) click(30,20) click(12,44) click(20,44) "
           "click(9,31) click(5,31) click(12,25) click(50,25) click(12,37) click(50,38) click(16,19) click(20,19) "
           "click(5,29) click(5,32) click(6,32) click(4,32)").split()
    asyncio.run(k.act(a, "play", {"actions": sol[:15]}))
    k.tick = 2
    asyncio.run(k.act(a, "play", {"actions": sol[15:]}))
    assert (w.games[0], 1) in w.solutions and any("was solved by" in e.content for e in k.boards["global"].entries)
    k.tick = 3
    asyncio.run(k.act(b, "play", {"actions": ["click(1,1)", "click(40,40)"]}))
    ok, msg = asyncio.run(k.act(b, "replay", {}))
    assert ok and "call replay again" in msg, msg
    k.tick = 4
    ok, msg = asyncio.run(k.act(b, "replay", {}))
    assert ok and "Reached the next level" in msg and w.state[(b, w.games[0])]["levels"] == 1, msg


def test_arc_explore_substrate_credit_and_delegation():
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85,vc33", "steps": "15", "publish_solutions": "1",
                                                           "explore": "1", "delegation": "1"}), 0, "arc")
    w = k.env
    assert "Problems, plans and findings" in w.brief() and "like a CEO or PI" in w.brief()
    lp = next(g for g in w.games if g.startswith("lp85"))
    a = next(x for x, g in sorted(w.assigned.items()) if g == lp)
    b, c = [x for x in sorted(w.assigned, key=lambda x: int(x[1:])) if x != a]
    if w.assigned[b] != lp:
        ok, msg = asyncio.run(k.act(b, "focus", {"game": "lp85"}))
        assert ok and "frontier there is level 1" in msg, msg
    ok, msg = asyncio.run(k.act(b, "focus", {"game": "lp85"}))
    assert not ok and "already play it" in msg
    k.tick = 1
    sol = ("click(12,20) click(9,20) click(12,20) click(22,20) click(26,20) click(30,20) click(12,44) click(20,44) "
           "click(9,31) click(5,31) click(12,25) click(50,25) click(12,37) click(50,38) click(16,19) click(20,19) "
           "click(5,29) click(5,32) click(6,32) click(4,32)").split()
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "claim", "hypothesis": "clicks on the left column swap tiles"}))
    assert ok and "Nobody else" in msg, msg
    plan_entry = msg.split("(")[1].split(")")[0]
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "pairs of clicks move a block"}))
    assert ok and "clicks on the left column" in msg, msg
    assert "OPEN PLANS on lp85 level 1" in k.observe(a, consume=False)
    ok, msg = asyncio.run(k.act(a, "play", {"actions": sol[:15]}))
    assert ok and "receipt R1" in msg, msg
    # a team: a leads, b joins, gets a role and a job, reports up, and pins the team's note as its tracker
    ok, team = asyncio.run(k.act(a, "lead", {"op": "found"}))
    gid = next(iter(w.teams))
    assert ok and asyncio.run(k.act(b, "lead", {"op": "join", "team": gid}))[0]
    ok, msg = asyncio.run(k.act(a, "lead", {"op": "assign", "to": b, "role": "tracker of hypotheses", "job": "test clicks"}))
    assert ok, msg
    assert "YOUR ROLE (from " + a in k.observe(b, consume=False)
    assert asyncio.run(k.act(b, "lead", {"op": "report", "text": "the left column does nothing"}))[0]
    assert "the left column does nothing" in k.observe(a, consume=False)
    assert asyncio.run(k.act(a, "lead", {"op": "tracker", "to": b}))[0]
    ok, msg = asyncio.run(k.act(b, "lead", {"op": "pin", "text": "known: pairs of clicks move blocks"}))
    assert ok and "PINNED in " + gid in k.observe(a, consume=False), msg
    # c joins, then founds a sub-team inside a's team: a's direct children are b and c
    assert asyncio.run(k.act(c, "lead", {"op": "join", "team": gid}))[0]
    ok, msg = asyncio.run(k.act(c, "lead", {"op": "found"}))
    assert ok and "sub-team of " + gid in msg and set(w.children(k, gid)) == {b, c} and w.team_depth(k) == 2, msg
    ok, msg = asyncio.run(k.act(a, "lead", {"op": "assign", "to": "A99", "role": "x"}))
    assert not ok
    # a reads b's plan, then completes the level first: priority credit to a, reuse credit to b, a share of b's to a
    k.tick = 2
    k.begin_turn(a)
    assert asyncio.run(k.act(a, "read", {"board": "global", "entry": plan_entry}))[0]
    ok, msg = asyncio.run(k.act(a, "play", {"actions": sol[15:]}))
    assert ok and "LEVEL 1 COMPLETED" in msg, msg
    assert w.credit[a] >= 5.3 and abs(w.credit[b] - 0.7) < 1e-9, dict(w.credit)  # b pays a 0.3 royalty on its 1
    assert any(x["reason"].startswith("share of " + b) for x in w.credit_log)
    k.tick = 3
    assert w.steps_allowed(a, 3) == 15 + min(15, round(2 * w.earned[(a, 2)])) >= 25
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "done", "finding": "the solution is pairs of clicks", "evidence": ["R1", "R2"]}))
    assert ok and "backed by engine records" in msg, msg
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "done", "finding": "x", "evidence": ["R77"]}))
    assert not ok and "no such receipts" in msg
    view = k.observe(a, consume=False)
    assert "WHAT HAS PAID" in view and "OPEN PROBLEMS" in view and "lp85 level 2" in view, view
    sc = w.score()
    assert sc["teams"] == 2 and sc["roles"] == 1 and sc["plans"] == 2 and sc["findings"] == 1, sc
    assert "plans" in w.export() and w.export()["combo_credit"]


def test_arc_rules_kinds_contracts_and_rule_learning():
    from alphateam.explore import clean_rules
    from alphateam.rule_learning import RuleLearner
    r = clean_rules({"credit": {"novel": 9, "priority": -3}, "substrate": {"show": ["plans", "bogus"], "plan_ttl": 99}})
    assert r["credit"]["novel"] == 1 and r["credit"]["priority"] == 0 and r["substrate"]["show"] == ["plans"]
    assert r["substrate"]["plan_ttl"] == 8 and r["credit"]["reuse"] == 1.0
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "steps": "15", "explore": "1",
                                                           "rules": '{"credit": {"priority": 7}}'}), 0, "arc")
    w = k.env
    assert w.rules["credit"]["priority"] == 7 and "+7" in w.brief() and "contract:" in w.brief()
    a, b, c = sorted(w.assigned, key=lambda x: int(x[1:]))
    k.tick = 1
    # an agent-defined kind of entry, written with write and found by search
    ok, msg = asyncio.run(k.act(a, "substrate", {"op": "define_kind", "name": "mechanic", "purpose": "what an action does"}))
    assert ok and "mechanic" in k.kinds_allowed(), msg
    assert not asyncio.run(k.act(b, "substrate", {"op": "define_kind", "name": "mechanic", "purpose": "x"}))[0]
    ok, msg = asyncio.run(k.act(b, "write", {"board": "global", "kind": "mechanic", "content": "clicks swap tiles"}))
    assert ok and k.boards["global"].entries[-1].kind == "mechanic"
    assert asyncio.run(k.act(c, "substrate", {"op": "propose", "text": "show dead ends first"}))[0]
    # a dead end backed by a receipt earns the dead-end weight and is shown on the problem
    asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "the top row is a button"}))
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["click(5,1)"]}))
    rid = msg.split("receipt ")[1].split("]")[0]
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "done", "finding": "the top row does nothing", "evidence": [rid],
                                            "outcome": "dead_end"}))
    assert ok and w.credit[a] == w.rules["credit"]["dead_end"], (msg, dict(w.credit))
    assert "DEAD ENDS on lp85 level 1" in k.observe(b, consume=False)
    # a contract: the pay is held on acceptance and released when the offerer marks it done
    ok, msg = asyncio.run(k.act(a, "contract", {"op": "offer", "to": b, "job": "map the board", "pay": 0.2}))
    cid = msg.split()[1]
    assert ok and not asyncio.run(k.act(a, "contract", {"op": "offer", "to": b, "job": "x", "pay": 50}))[0]
    assert asyncio.run(k.act(b, "contract", {"op": "accept", "id": cid}))[0] and abs(w.credit[a] - 0.1) < 1e-9
    assert asyncio.run(k.act(a, "contract", {"op": "done", "id": cid}))[0] and abs(w.credit[b] - 0.2) < 1e-9
    # the rule learner revises the rules (scripted reviser), posts them, and they take effect
    k.total_ticks, k.tick = 10, 4
    learner = RuleLearner(4)
    ev = learner.evidence(k)
    assert ev["proposals_from_agents"] and "mechanic" in ev["kinds_of_entry"] and ev["rules_now"] == w.rules
    asyncio.run(learner.on_tick_end(k, object()))
    assert w.rules_version == 2 and w.rules["credit"]["novel"] == 0.3 and learner.history[0]["changed"]
    assert any(e.content.startswith("RULES v2") for e in k.boards["global"].entries) and "Rules v2" in k.observe(a, consume=False)
    k.rank["recency"] = 1.0
    assert "unseen frontier state +0.3" in w.brief()
    assert w.score()["rules_version"] == 2 and w.export()["kind_stats"]


def test_arc_inventor_of_a_custom_action_earns_a_royalty():
    from alphateam.kernel import CustomAction
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "steps": "15", "explore": "1"}), 0, "arc")
    w = k.env
    a, b = sorted(w.assigned, key=lambda x: int(x[1:]))
    k.tick = 1
    k.custom["probe"] = CustomAction("probe", "click every tile once", "play clicks", [], a, 0)
    k.running_custom[b] = "probe"  # b is running a's action when it earns credit
    w._award(k, b, 5.0, "first to complete lp85 level 1")
    k.running_custom.pop(b)
    share = w.rules["credit"]["action_share"]
    assert abs(w.credit[b] - 5.0 * (1 - share)) < 1e-9 and abs(w.credit[a] - 5.0 * share) < 1e-9  # a royalty, not minted
    assert abs(w.credit[a] + w.credit[b] - 5.0) < 1e-9
    assert any("x_probe" in c for c in w.combo_credit) and "inventor share" in w.combo_credit
    w._award(k, a, 1.0, "own use")  # no royalty to yourself
    assert abs(w.credit[a] - (1.0 + 5.0 * share)) < 1e-9


def test_arc_plan_lifecycle_claims_findings_and_untried():
    from alphateam.primitives import system_prompt
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "steps": "15", "explore": "1",
                                                           "publish_solutions": "1"}), 0, "arc")
    w = k.env
    assert "from claim to result" in w.brief() and "workflows or stages" in system_prompt(k)
    assert "A working cycle" not in w.brief()
    k2 = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "explore": "1", "cycle": "1"}), 0, "arc")
    assert "A working cycle" in k2.env.brief() and "cycle" in k2.env.rules["substrate"]["show"]
    k2.env.rules["substrate"]["show"].remove("cycle")  # the rule loop can hide it again
    assert "A working cycle" not in k2.env.brief()
    a, b, c = sorted(w.assigned, key=lambda x: int(x[1:]))
    k.tick = 1
    assert asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "pairs of clicks move a block",
                                         "criteria": "a block moves"}))[0]
    assert asyncio.run(k.act(b, "plan", {"op": "claim", "hypothesis": "clicks on the left move blocks"}))[0]
    assert any("message @" + b in n["text"] for n in k.agents[a].inbox)  # a hears of b's overlapping claim
    view = k.observe(c, consume=False)
    assert "UNTRIED by any copy at lp85 level 1" in view and "64 of 64" in view, view
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["click(12,20)", "click(9,20)"]}))
    rid = msg.split("receipt ")[1].split("]")[0]
    assert "63 of 64" in k.observe(c, consume=False)  # both clicks fall in region r2 c1
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "finding", "finding": "the block moved left", "evidence": [rid]}))
    assert ok and any(p["status"] == "open" for p in w.plans.values() if p["agent"] == a)
    assert "criteria: a block moves" in k.boards["global"].entries[-1].content
    sol = ("click(12,20) click(9,20) click(12,20) click(22,20) click(26,20) click(30,20) click(12,44) click(20,44) "
           "click(9,31) click(5,31) click(12,25) click(50,25) click(12,37) click(50,38) click(16,19) click(20,19) "
           "click(5,29) click(5,32) click(6,32) click(4,32)").split()
    asyncio.run(k.act(c, "play", {"actions": sol[:15]}))
    k.tick = 2
    asyncio.run(k.act(c, "play", {"actions": sol[15:]}))
    assert any("completed lp85 level 1" in n["text"] for n in k.agents[b].inbox)  # b's problem was solved by c


def test_arc_interfaces_runnable_entries_and_challenges():
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "steps": "15", "explore": "1"}), 0, "arc")
    w = k.env
    a, b, c = sorted(w.assigned, key=lambda x: int(x[1:]))
    k.tick = 1
    # an interface: a kind with fields; entries must fill them; adopted by enough agents it is agreed and shown
    assert asyncio.run(k.act(a, "substrate", {"op": "define_kind", "name": "mechanic", "purpose": "what an action does",
                                              "fields": ["action", "effect"]}))[0]
    ok, msg = asyncio.run(k.act(b, "write", {"board": "global", "kind": "mechanic", "content": "action: click(12,20)"}))
    assert not ok and "missing effect" in msg, msg
    assert asyncio.run(k.act(b, "write", {"board": "global", "kind": "mechanic",
                                          "content": "action: click(12,20)\neffect: selects the block"}))[0]
    assert "AGREED KIND" not in k.observe(c, consume=False)
    ok, msg = asyncio.run(k.act(b, "substrate", {"op": "adopt", "name": "mechanic"}))
    assert ok and "now agreed" in msg and "AGREED KIND 'mechanic'" in k.observe(c, consume=False), msg
    assert asyncio.run(k.act(b, "substrate", {"op": "revise_kind", "name": "mechanic", "fields": ["action", "effect", "level"]}))[0]
    assert w.kinds["mechanic"]["version"] == 2 and any("revised kind" in n["text"] for n in k.agents[a].inbox)
    # a runnable entry: moves written in an entry, with placeholders, run by the engine; its author earns a royalty
    e = k.post("global", a, "note", "swap: click({x},20) click(9,20)")
    ok, msg = asyncio.run(k.act(b, "play", {"entry": e.id, "args": {"x": 12}}))
    assert ok and msg.startswith(f"ran {e.id}") and "click(12,20)" in w.receipts[msg.split("receipt ")[1].split("]")[0]]["moves"], msg
    assert not asyncio.run(k.act(b, "play", {"entry": e.id}))[0]  # placeholders need args
    w._running_entry[b] = e.id
    w._award(k, b, 1.0, "test")
    w._running_entry.pop(b)
    assert abs(w.credit[a] - w.rules["credit"]["action_share"]) < 1e-9 and "ran 1x" in k.header(e)
    ok, msg = asyncio.run(k.act(c, "play", {"entry": "R1"}))  # a receipt runs too
    assert ok, msg
    # a challenge: a counterexample with receipts contests a finding until its author revises it
    asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "the left column is inert", "ttl": 1}))
    rid = next(r for r, v in w.receipts.items() if v["agent"] == b)
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "done", "finding": "clicks on the left do nothing", "evidence": [rid]}))
    fid = msg.split()[2]
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "challenge", "entry": fid, "finding": "click(12,20) moved a block",
                                            "evidence": [rid]}))
    assert ok and "CONTESTED by " + c in k.observe(b, consume=False), msg
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "finding", "finding": "only the top-left cell is inert", "evidence": [rid],
                                            "entry": fid}))
    assert ok and "revised as" in k.observe(b, consume=False) and w.score()["challenges"] == 1 and w.score()["revisions"] == 1
    # an expired plan is shown as free to take over; records show findings and credit
    asyncio.run(k.act(b, "plan", {"op": "claim", "hypothesis": "the bottom row is a goal", "ttl": 1}))
    k.tick = 4
    assert "EXPIRED WITHOUT A RESULT" in k.observe(c, consume=False)
    assert "findings" in w.track_record(a) and "credit" in w.track_record(a)


def test_arc_game_over_frame_does_not_change_level_count():
    from types import SimpleNamespace
    k = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85"}), 0, "arc")
    w = k.env
    a, g = next(iter(w.assigned.items()))
    w.best[g] = 1
    w.state[(a, g)]["levels"] = 1
    over = SimpleNamespace(frame=[], levels_completed=0, win_levels=0, state=SimpleNamespace(name="GAME_OVER"), available_actions=[0])
    w._record((a, g), over)  # a GAME_OVER frame reports "level 0/0"
    assert w.win[g] == 8 and w.state[(a, g)]["levels"] == 1 and w.score()["score"] == round(1 / 8, 4)



def test_arc_sync_public_substrate_push_once_and_interrupts():
    k = build(dict(CFG, world="arc", agents=6, world_opts={"games": "lp85,vc33", "explore": "1", "sync": "1",
                                                           "publish_solutions": "1"}), 0, "arc")
    w = k.env
    prompt = system_prompt(k)
    assert k.sync and "There are no private boards" in prompt and "private board is yours" not in prompt
    assert "Sync: everything is public" in w.brief() and "paid" not in w.rules["substrate"]["show"]
    lp = sorted((x for x, g in w.assigned.items() if g.startswith("lp85")), key=lambda x: int(x[1:]))
    vc = sorted((x for x, g in w.assigned.items() if g.startswith("vc33")), key=lambda x: int(x[1:]))
    a, b, c = lp[0], lp[1], vc[0]
    k.tick = 1
    for x in k.agents:
        k.observe(x)
    # no private boards: a 'private' write is public, and nobody has a board of their own to read
    ok, msg = asyncio.run(k.act(a, "write", {"board": "private", "content": "lp85: clicks on blocks move them"}))
    assert ok and "global" in msg, msg
    assert not asyncio.run(k.act(b, "read", {"board": a}))[0]
    # plays stay in the player's recent work, with their receipts, instead of on a board
    n = len(k.boards["global"].entries)
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["click(12,20)", "click(9,20)"]}))
    rid = msg.split("receipt ")[1].split("]")[0]
    assert len(k.boards["global"].entries) == n and rid in k.recent_work(a)
    k.end_turn(a, "gist: block clicks move blocks\nthe longer reasoning")
    assert k.boards["global"].entries[-1].kind == "note" and "YOUR RECENT WORK" in k.observe(a, consume=False)
    # a dead end on b's level reaches b during its turn, once; c plays another game and does not get it
    asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "clicking the right edge moves blocks"}))
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "done", "finding": "right-edge clicks do nothing",
                                            "outcome": "dead_end", "evidence": [rid]}))
    dead = msg.split("finding ")[1].split()[0]
    note = k.interrupts(b)
    assert "NEW WHILE YOU WORK" in note and dead in note and dead not in k.interrupts(b)
    assert dead not in k.interrupts(c)
    # a message reaches its addressee during its turn, and leaves the inbox
    asyncio.run(k.act(c, "write", {"board": "global", "content": f"@{b} which clicks did you try?"}))
    assert "which clicks" in k.interrupts(b) and not k.agents[b].inbox
    # the turn-start delivery: b's game first, duplicates collapsed, each entry once, the overflow summarized
    for x in vc:
        asyncio.run(k.act(x, "write", {"board": "global", "content": "vc33: ACTION6 on a red cell turns it blue"}))
    push = k.observe(b).split("NEW SINCE YOUR LAST TURN")[1]
    assert dead not in push and "which clicks" not in push  # both already reached b during its turn
    assert push.index("lp85: clicks on blocks") < push.index("vc33: ACTION6"), push
    assert "(x3: also" in push, push
    assert "NEW SINCE YOUR LAST TURN: nothing." in k.observe(b)
    k.push_items = 1
    for i in range(3):
        asyncio.run(k.act(c, "write", {"board": "global", "content": f"vc33 test {i}: ACTION6 at ({i},{i})"}))
    assert "+2 more" in k.observe(b)
    assert w.score()["pushed"] > 0 and w.score()["interrupts"] >= 2
    try:
        build(dict(CFG, world="arc", agents=2, condition="isolated", world_opts={"games": "lp85", "sync": "1"}), 0, "arc")
        assert False, "sync cannot run isolated"
    except ValueError:
        pass


def test_sync_notices_reach_an_llm_agent_during_its_turn():
    from alphateam.llm import Call, LLMBackend, Reply
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "sync": "1"}), 0, "arc")
    a, b = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    seen = []
    tokens = {"input": 1, "cached": 0, "output": 1}

    class Fake:  # a provider whose agent takes one action and stops; b writes to it while it waits
        transient = ()

        def tools(self, schemas):
            return schemas

        def first(self, text):
            return [{"role": "user", "content": text}]

        def results(self, rs):
            seen.extend(text for _, _, text in rs)
            return [{"role": "tool", "content": text} for _, _, text in rs]

        async def create(self, system, messages, tools, final, who):
            if len(messages) == 1:
                await k.act(b, "write", {"board": "global", "content": f"@{a} stop: ACTION1 resets the level"})
                return Reply("", [Call("c1", "look", {})], "ok", tokens, [{"role": "assistant", "content": ""}])
            return Reply("looked once; next: test ACTION2", [], "ok", tokens, [])

    asyncio.run(LLMBackend(Fake()).turn(k, a))
    assert any("NEW WHILE YOU WORK" in t and "ACTION1 resets" in t for t in seen), seen
    last = k.boards["global"].entries[-1]
    assert last.author == a and last.kind == "note" and "next: test ACTION2" in last.content


LP85_L1 = ("click(12,20) click(9,20) click(12,20) click(22,20) click(26,20) click(30,20) click(12,44) click(20,44) "
           "click(9,31) click(5,31) click(12,25) click(50,25) click(12,37) click(50,38) click(16,19) click(20,19) "
           "click(5,29) click(5,32) click(6,32) click(4,32)").split()


def test_arc_carry_forward_explanations_and_contested_dependents():
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "steps": "15", "explore": "1",
                                                           "publish_solutions": "1", "carry": "1"}), 0, "arc")
    w = k.env
    a, b, c = sorted(w.assigned, key=lambda x: int(x[1:]))
    assert "Nothing learned is lost when a level falls" in w.brief() and "carried" in w.rules["substrate"]["show"]
    k.tick = 1
    rid = lambda msg: msg.split("receipt ")[1].split("]")[0]
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["click(12,20)", "click(9,20)"]}))
    ra = rid(msg)
    asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "the gems are the only interactive zones"}))
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "finding", "finding": "two interactive zones: left and right gem",
                                            "evidence": [ra]}))
    fa = msg.split("finding ")[1].split()[0]
    assert asyncio.run(k.act(b, "plan", {"op": "claim", "hypothesis": "clicks on the bottom row draw frames"}))[0]
    ok, msg = asyncio.run(k.act(c, "play", {"actions": LP85_L1[:15]}))
    rc = rid(msg)
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "finding", "finding": "pairs of clicks swap blocks", "evidence": [rc]}))
    fc = msg.split("finding ")[1].split()[0]
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "finding", "finding": "frames follow the swapped blocks",
                                            "builds_on": [fc]}))
    fb = msg.split("finding ")[1].split()[0]
    k.tick = 2
    ok, msg = asyncio.run(k.act(c, "play", {"actions": LP85_L1[15:]}))
    assert "LEVEL 1 COMPLETED" in msg, msg
    # b's open plan moved to level 2 instead of going stale; the solver was asked to explain
    pb = next(p for p in w.plans.values() if p["agent"] == b)
    assert pb["level"] == 2 and pb["carried_from"] == 1 and pb["status"] == "open"
    assert any("carried to level 2" in n["text"] for n in k.agents[b].inbox)
    sid = w.solution_of[(w.games[0], 1)]
    assert any(f"plan op=explain entry={sid}" in n["text"] for n in k.agents[c].inbox)
    assert "The solver's findings on this game" in k.entry(sid).content and fc in k.entry(sid).content
    # level-1 findings stay in view at level 2
    view = k.observe(a, consume=False)
    assert "CARRIED FORWARD from earlier levels of lp85" in view and fa in view, view
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "explain", "entry": sid, "finding": "each pair swaps two blocks so "
                                            "the rows match the target", "evidence": [rc]}))
    assert ok and w.entry_meta[w.explanations[sid][0]]["type"] == "explanation", msg
    # a replay carries the understanding with the moves
    ok, msg = asyncio.run(k.act(a, "replay", {}))
    assert ok and msg.startswith(f"WHY THESE MOVES WORK ({sid})") and "rows match the target" in msg, msg[:300]
    k.tick = 3
    ok, msg = asyncio.run(k.act(a, "replay", {}))
    assert "Reached the next level" in msg and "Carried forward on this game" in msg, msg[-300:]
    # a receipt from level 2 retires a level-1 finding there, without calling it wrong
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["click(12,20)"]}))
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "challenge", "entry": fc, "finding": "at level 2 the pair does not swap",
                                            "evidence": [rid(msg)]}))
    f = next(x for x in w.findings if x["id"] == fc)
    assert ok and f["superseded_at"] == 2 and not f.get("contested") and "no longer carried" in msg, msg
    assert fc not in k.observe(a, consume=False).split("CARRIED FORWARD")[1].split("\n")[0]
    # a counterexample at the finding's own level contests it, and flags what was built on it
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "challenge", "entry": fa, "finding": "a third zone reacts", "evidence": [ra]}))
    assert ok and next(x for x in w.findings if x["id"] == fa)["contested"]
    k.tick = 4
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "finding", "finding": "the zones explain the swaps", "builds_on": [fa]}))
    assert next(x for x in w.findings if x["id"] == msg.split("finding ")[1].split()[0])["built_on_contested"] == fa
    assert w.score()["plans_carried"] == 2 and w.score()["explanations"] == 1  # a's and b's plans


def test_arc_shared_model_merges_only_what_breaks_nothing():
    import numpy as np
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "explore": "1", "model": "1"}), 0, "arc")
    w = k.env
    a, b = sorted(w.assigned, key=lambda x: int(x[1:]))
    g, m = w.games[0], w.models
    assert "The shared model" in w.brief() and "model" in w.rules["substrate"]["show"]
    zero = np.zeros((64, 64), int)
    lit = zero.copy()
    lit[0, 0] = 5
    m.record(g, 1, "ACTION1", zero, lit, "", 1)        # T0: ACTION1 lights the corner
    m.record(g, 1, "ACTION2", zero, zero, "", 1)       # T1: ACTION2 does nothing
    m.record(g, 1, "ACTION3", zero, zero, "level_up", 1)  # T2: ACTION3 completes the level
    m.record(g, 1, "ACTION2", zero, zero, "", 1)       # the same move again: counted, kept once
    assert len(m.transitions[g]) == 3 and m.transitions[g][1]["n"] == 2
    k.tick = 1
    ok, msg = asyncio.run(k.act(a, "model", {"op": "view"}))
    assert ok and "v0" in msg and "Explains 1 of 3" in msg and "T0 level 1 ACTION1" in msg, msg
    good = ("def step(grid, action, x, y):\n    if action == 3:\n        return grid, 'level_up'\n"
            "    if action == 1:\n        grid[0, 0] = 5\n    return grid, ''\n")
    assert not asyncio.run(k.act(a, "model", {"op": "propose", "code": good}))[0]  # base is required
    ok, msg = asyncio.run(k.act(a, "model", {"op": "test", "code": good}))
    assert ok and msg.startswith("TEST") and "adds 2" in msg, msg
    ok, msg = asyncio.run(k.act(a, "model", {"op": "propose", "base": 0, "code": good, "why": "corner light and level"}))
    assert ok and "merged as v1" in msg and m.current(g)["author"] == a, msg
    ok, msg = asyncio.run(k.act(b, "model", {"op": "propose", "base": 0, "code": good + "# mine\n"}))
    assert not ok and "changed underneath" in msg, msg  # written on an old version: take in v1 and try again
    ident = "def step(grid, action, x, y):\n    return grid, ''\n"
    ok, msg = asyncio.run(k.act(b, "model", {"op": "propose", "base": 1, "code": ident}))
    assert not ok and "breaks 2" in msg, msg  # nothing that works may break
    ok, msg = asyncio.run(k.act(b, "model", {"op": "propose", "base": 1, "code": good + "# same\n"}))
    assert not ok and "at least one more" in msg, msg
    for bad, why in (("import os\ndef step(grid, action, x, y):\n    return grid\n", "Import"),
                     ("def step(grid, action, x, y):\n    open('f', 'w')\n    return grid\n", "'open'"),
                     ("def step(grid, action, x, y):\n    return grid.__class__\n", "'__class__'"),
                     ("def step(grid, action, x, y):\n    np.save('f', grid)\n    return grid\n", "'save'")):
        ok, msg = asyncio.run(k.act(b, "model", {"op": "test", "code": bad}))
        assert not ok and why in msg, msg
    ok, msg = asyncio.run(k.act(b, "model", {"op": "test", "code": "def step(grid, action, x, y):\n    while True:\n        pass\n"}))
    assert ok and "explains 0 of 3" in msg, msg  # a call that runs too long explains nothing
    ok, msg = asyncio.run(k.act(b, "model", {"op": "case", "id": "T0"}))
    assert ok and "BEFORE" in msg and "AFTER" in msg and "0->5" in msg, msg
    # credit is prequential: moves recorded after the merge that v1 predicts and v0 did not
    other = np.ones((64, 64), int)
    lit2 = other.copy()
    lit2[0, 0] = 5
    k.tick = 2
    m.record(g, 1, "ACTION1", other, lit2, "", 2)
    before = w.credit[a]
    asyncio.run(w.model_tick_end(k))
    assert abs(w.credit[a] - before - w.rules["credit"]["model"]) < 1e-9 and w.model_credit_log[-1]["moves"] == 1
    asyncio.run(w.model_tick_end(k))
    assert len(w.model_credit_log) == 1  # paid once
    assert w.score()["model_merges"] == 1 and w.score()["model_coverage"] == 1.0
    # real play is recorded; the identity explains moves that change nothing
    k2 = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85", "explore": "1", "model": "1"}), 0, "arc")
    k2.tick = 1
    a2 = next(iter(k2.env.assigned))
    asyncio.run(k2.act(a2, "play", {"actions": ["click(1,1)", "click(12,20)"]}))
    asyncio.run(k2.env.on_tick_end(k2))
    got, total = k2.env.models.coverage(k2.env.games[0])
    assert total == 2 and got >= 1 and "SHARED MODEL of lp85 v0" in k2.observe(a2, consume=False)
    # a later society starts from the models this one assembled
    import tempfile
    path = Path(tempfile.mkdtemp()) / "run.json"
    path.write_text(json.dumps({"world": w.explore_export(k)}))
    k3 = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85", "explore": "1", "model": "1",
                                                            "models": str(path)}), 0, "arc")
    first = k3.env.models.current(k3.env.games[0])
    assert first["v"] == 0 and first["code"] == good and "inherited" in first["why"]


def test_arc_guide_keeps_others_lines_unless_evidence_says_otherwise():
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "explore": "1", "guide": "1"}), 0, "arc")
    w = k.env
    a, b = sorted(w.assigned, key=lambda x: int(x[1:]))
    assert "The guide: one short, durable text per game" in w.brief() and "guide" in w.rules["substrate"]["show"]
    k.tick = 1
    ok, msg = asyncio.run(k.act(a, "guide", {"op": "edit", "base": 0, "text": "gems at left and right react\nclicks elsewhere do nothing",
                                             "why": "first notes"}))
    assert ok and "v1" in msg, msg
    ok, msg = asyncio.run(k.act(b, "guide", {"op": "edit", "base": 0, "text": "my version"}))
    assert not ok and "changed underneath" in msg  # written on an old version
    ok, msg = asyncio.run(k.act(b, "guide", {"op": "edit", "base": 1, "text": "gems at left and right react\npairs swap blocks"}))
    assert not ok and "removed 1 line(s) others wrote" in msg, msg  # a's line needs evidence to go
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["click(30,40)"]}))
    rid = out.split("receipt ")[1].split("]")[0]
    ok, msg = asyncio.run(k.act(b, "guide", {"op": "edit", "base": 1, "text": "gems at left and right react\npairs swap blocks",
                                             "why": "clicks off the gems do something", "evidence": [rid]}))
    assert ok and any("removed your line" in n["text"] for n in k.agents[a].inbox), msg
    ok, msg = asyncio.run(k.act(b, "guide", {"op": "view"}))
    assert f"[{a}] gems at left and right react" in msg and f"[{b}] pairs swap blocks" in msg, msg
    ok, msg = asyncio.run(k.act(b, "guide", {"op": "edit", "base": 2, "text": "gems at left and right react"}))
    assert ok, msg  # removing your own line needs nothing
    assert "GUIDE to lp85 v3" in k.observe(a, consume=False) and w.score()["guide_edits"] == 3


def test_continuity_keeps_an_agents_recent_ticks():
    from alphateam.llm import Call, LLMBackend, Reply
    k = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85", "continuity": "2"}), 0, "arc")
    a = next(iter(k.agents))
    from alphateam.primitives import system_prompt
    assert "conversation from your last 2 ticks" in system_prompt(k)
    sent = []
    tokens = {"input": 1, "cached": 0, "output": 1}

    class Fake:
        transient = ()

        def tools(self, schemas):
            return schemas

        def first(self, text):
            return [{"role": "user", "content": text}]

        def results(self, rs):
            return [{"role": "tool", "tool_call_id": cid, "content": text + "x" * 2000} for cid, _, text in rs]

        async def create(self, system, messages, tools, final, who):
            sent.append([dict(m) for m in messages])
            if messages[-1]["role"] == "user":
                return Reply("", [Call(f"c{k.tick}", "look", {})], "ok", tokens,
                             [{"role": "assistant", "content": "", "reasoning": "long thoughts",
                               "tool_calls": [{"id": f"c{k.tick}", "type": "function",
                                               "function": {"name": "look", "arguments": "{}"}}]}])
            return Reply(f"tick {k.tick}: looked", [], "ok", tokens, [])

    backend = LLMBackend(Fake())
    for t in (1, 2, 3, 4):
        k.tick = t
        asyncio.run(backend.turn(k, a))
    last = sent[-2]  # tick 4's first call: ticks 2 and 3 are remembered, tick 1 is not
    notes = [m["content"] for m in last if m["role"] == "assistant" and m.get("content", "").startswith("tick")]
    assert notes == ["tick 2: looked", "tick 3: looked"], notes
    assert all("reasoning" not in m for m in last[:-1]) and all(len(m["content"]) <= 600 for m in last if m["role"] == "tool")
    assert last[-1]["role"] == "user" and "TICK 4" in last[-1]["content"]


def test_trace_fixes_counter_mask_guide_add_last_reply_moves_and_levels():
    import numpy as np
    from alphateam.llm import Call, LLMBackend, Reply
    # a counter on the frame's edge no longer breaks every prediction
    k = build(dict(CFG, world="arc", agents=1, world_opts={"games": "vc33", "explore": "1", "model": "1", "sync": "1"}), 0, "arc")
    w = k.env
    g, m = w.games[0], w.models
    for i in range(25):
        before = np.zeros((64, 64), int)
        before[0, :i] = 4
        after = before.copy()
        after[0, i] = 4  # each move fills one more cell of row 0
        before[30, i] = i % 7
        after[30, i] = i % 7
        m.record(g, 1, f"click({i},30)", before, after, "", 1)
    assert m.masks[g][0].all() and not m.masks[g][30].any() and "row 0" in m.mask_note[g]
    got = asyncio.run(m.explained(g, 0))
    assert sum(got.values()) == 25  # version 0 explains them once the counter is ignored
    assert "on level 1 of 7, 0 completed" in k.observe(next(iter(k.agents)), consume=False)
    assert "'level K/N' counts levels completed" in w.brief()
    # appends to the guide are combined without a base
    k2 = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "explore": "1", "guide": "1"}), 0, "arc")
    a, b = sorted(k2.agents, key=lambda x: int(x[1:]))
    k2.tick = 1
    assert asyncio.run(k2.act(a, "guide", {"op": "add", "line": "only the gems react to clicks"}))[0]
    assert asyncio.run(k2.act(b, "guide", {"op": "add", "line": "row 1 shows the target"}))[0]
    assert not asyncio.run(k2.act(b, "guide", {"op": "add", "line": "row 1 shows the target"}))[0]
    cur = k2.env.guides[k2.env.games[0]][-1]
    assert cur["v"] == 2 and cur["lines"] == {"only the gems react to clicks": a, "row 1 shows the target": b}
    # the last reply of a tick may still make game moves
    k3 = build(dict(CFG, world="arc", agents=1, calls=1, world_opts={"games": "lp85", "final_moves": "1"}), 0, "arc")
    k3.max_calls = 1
    a3 = next(iter(k3.agents))
    k3.tick = 1
    offered = []
    tokens = {"input": 1, "cached": 0, "output": 1}

    class Fake:
        transient = ()

        def tools(self, schemas):
            return [{"type": "function", "function": s} for s in schemas]

        def first(self, text):
            return [{"role": "user", "content": text}]

        def results(self, rs):
            return [{"role": "tool", "tool_call_id": cid, "content": text} for cid, _, text in rs]

        async def create(self, system, messages, tools, final, who):
            offered.append(sorted(t["function"]["name"] for t in tools) if not final else [])
            if len(offered) == 1:
                return Reply("", [Call("c1", "look", {})], "ok", tokens, [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}]}])
            return Reply("note: tried two clicks", [Call("c2", "play", {"actions": ["click(5,32)"]}), Call("c3", "write", {"content": "x"})],
                         "ok", tokens, [{"role": "assistant", "content": "", "tool_calls": [
                             {"id": "c2", "type": "function", "function": {"name": "play", "arguments": "{}"}},
                             {"id": "c3", "type": "function", "function": {"name": "write", "arguments": "{}"}}]}])

    from alphateam.primitives import system_prompt
    assert "Your last reply of a tick may still make game moves" in system_prompt(k3)
    asyncio.run(LLMBackend(Fake()).turn(k3, a3))
    assert offered[1] == ["play"], offered  # only game moves are offered in the last reply
    plays = [e for e in k3.events if e["op"] == "lab.play" and e["ok"]]
    assert len(plays) == 1 and not any(e["op"].startswith("write") for e in k3.events)


def test_last_reply_can_record_and_long_guide_lines_are_kept():
    from alphateam.llm import Call, LLMBackend, Reply
    k = build(dict(CFG, world="arc", agents=1, calls=1, world_opts={"games": "lp85", "explore": "1", "guide": "1",
                                                                    "final_moves": "1"}), 0, "arc")
    k.max_calls = 1
    aid = next(iter(k.agents))
    k.tick = 1
    line = "the 8-blobs are pivots: each click rotates its ring one block-step " * 6  # about 400 characters
    offered, tokens = [], {"input": 1, "cached": 0, "output": 1}

    class Fake:
        transient = ()

        def tools(self, schemas):
            return [{"type": "function", "function": s} for s in schemas]

        def first(self, text):
            return [{"role": "user", "content": text}]

        def results(self, rs):
            return [{"role": "tool", "tool_call_id": cid, "content": text} for cid, _, text in rs]

        async def create(self, system, messages, tools, final, who):
            offered.append(sorted(t["function"]["name"] for t in tools))
            if len(offered) == 1:
                return Reply("", [Call("c1", "look", {})], "ok", tokens, [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}]}])
            calls = [Call("c2", "play", {"actions": ["click(5,32)"], "note": "testing whether the left blob is a pivot"}),
                     Call("c3", "guide", {"op": "add", "line": line}), Call("c4", "look", {})]
            return Reply("", calls, "ok", tokens, [{"role": "assistant", "content": "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": "{}"}} for c in calls]}])

    asyncio.run(LLMBackend(Fake()).turn(k, aid))
    assert set(offered[1]) >= {"play", "guide", "plan"} and "look" not in offered[1], offered
    assert k.env.guides[k.env.games[0]][-1]["text"] == " ".join(line.split())  # a 400-character line is kept
    assert sum(e["op"] == "lab.look" for e in k.events) == 1  # look is refused in the last reply
    note = [e for e in k.events if e["op"] == "intent"][-1]["msg"]
    assert note.startswith("testing whether the left blob is a pivot (last moves:"), note
    assert not asyncio.run(k.act(aid, "guide", {"op": "add", "line": "x" * 700}))[0]


def test_engine_posts_game_over_as_a_dead_end():
    from types import SimpleNamespace
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "vc33", "explore": "1", "carry": "1", "sync": "1"}), 0, "arc")
    w = k.env
    a, b = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    for x in k.agents:
        k.observe(x)
    real = w._record

    def over_after_two(key, obs):  # the third move ends the game on a's copy
        real(key, obs)
        if key[0] == a and len(w.trace[key]) >= 2:
            w.state[key]["state"] = "GAME_OVER"
    w._record = over_after_two
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["click(5,5)", "click(6,6)", "click(7,7)"]}))
    dead = [f for f in w.findings if f["agent"] == "lab"]
    assert ok and len(dead) == 1 and dead[0]["outcome"] == "dead_end" and dead[0]["level"] == 1, (msg, dead)
    assert "GAME_OVER on vc33 level 1 after 3 moves" in k.entry(dead[0]["id"]).content
    assert dead[0]["id"] in k.interrupts(b)  # urgent for the other copy on that level


def test_engine_warns_when_reset_restarts_the_game():
    k = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85", "explore": "1", "carry": "1", "sync": "1",
                                                           "publish_solutions": "1"}), 0, "arc")
    w = k.env
    a = next(iter(k.agents))
    k.tick = 1
    asyncio.run(k.act(a, "play", {"actions": LP85_L1[:15]}))
    k.tick = 2
    assert "LEVEL 1 COMPLETED" in asyncio.run(k.act(a, "play", {"actions": LP85_L1[15:]}))[1]
    k.tick = 3
    ok, msg = asyncio.run(k.act(a, "play", {"actions": ["RESET"]}))  # at the start of level 2 it would restart the game
    assert "refused" in msg and w.state[(a, w.games[0])]["levels"] == 1, msg  # so it is refused unless asked for
    asyncio.run(k.act(a, "play", {"actions": ["RESET"], "restart_game": True}))
    dead = [f for f in w.findings if f["agent"] == "lab"]
    assert len(dead) == 1 and "restarted the whole game" in dead[0]["text"] and dead[0]["level"] == 2
    asyncio.run(k.act(a, "play", {"actions": LP85_L1[:15]}))
    asyncio.run(k.act(a, "play", {"actions": LP85_L1[15:]}))
    asyncio.run(k.act(a, "play", {"actions": ["RESET"], "restart_game": True}))
    dead = [f for f in w.findings if f["agent"] == "lab"]
    assert len(dead) == 1 and dead[0]["repeats"] == 2  # the same fact is counted, not posted again


def test_guide_and_delivered_findings_earn_reuse_credit_at_a_first_solve():
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "explore": "1", "sync": "1", "carry": "1",
                                                           "guide": "1", "publish_solutions": "1"}), 0, "arc")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    for x in k.agents:
        k.observe(x)
    assert asyncio.run(k.act(b, "guide", {"op": "add", "line": "pairs of clicks swap blocks"}))[0]
    ok, out = asyncio.run(k.act(c, "play", {"actions": ["click(1,1)"]}))
    rid = out.split("receipt ")[1].split("]")[0]
    asyncio.run(k.act(c, "plan", {"op": "claim", "hypothesis": "the far corner is inert"}))
    asyncio.run(k.act(c, "plan", {"op": "done", "finding": "corner clicks do nothing", "outcome": "dead_end", "evidence": [rid]}))
    k.tick = 2
    k.observe(a)  # c's dead end is delivered to a
    before = {x: w.credit[x] for x in (b, c)}
    asyncio.run(k.act(a, "play", {"actions": LP85_L1[:15]}))
    k.tick = 3
    asyncio.run(k.act(a, "play", {"actions": LP85_L1[15:]}))
    reuse = w.rules["credit"]["reuse"]
    assert w.credit[b] - before[b] >= reuse - 1e-9  # the guide line was in a's view every turn
    assert w.credit[c] - before[c] >= reuse - 1e-9  # c's dead end was delivered to a
    assert any("read:guide" in x["combo"] for x in w.credit_log if x["agent"] == a)


def test_a_society_inherits_guides_and_models():
    import tempfile
    k = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85", "explore": "1", "guide": "1", "model": "1"}), 0, "arc")
    a = next(iter(k.agents))
    k.tick = 1
    asyncio.run(k.act(a, "guide", {"op": "add", "line": "only the gems react"}))
    path = Path(tempfile.mkdtemp()) / "run.json"
    path.write_text(json.dumps({"world": k.env.explore_export(k)}))
    k2 = build(dict(CFG, world="arc", agents=1, world_opts={"games": "lp85", "explore": "1", "guide": "1", "model": "1",
                                                            "inherit": str(path)}), 0, "arc")
    g = k2.env.guides[k2.env.games[0]][0]
    assert g["text"] == "only the gems react" and g["lines"] == {"only the gems react": "inherited"}
    assert "inherited" in k2.env.models.current(k2.env.games[0])["why"] or k2.env.models.current(k2.env.games[0])["v"] == 0


def test_guide_lines_carry_receipts_and_counterexamples_mark_them():
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "explore": "1", "guide": "1"}), 0, "arc")
    w = k.env
    a, b = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["click(58,32)"]}))
    ra = out.split("receipt ")[1].split("]")[0]
    assert asyncio.run(k.act(a, "guide", {"op": "add", "line": "the right gem reacts", "evidence": [ra]}))[0]
    assert asyncio.run(k.act(a, "guide", {"op": "add", "line": "nothing else reacts"}))[0]
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["click(5,32)"]}))
    rb = out.split("receipt ")[1].split("]")[0]
    assert not asyncio.run(k.act(b, "guide", {"op": "challenge", "line": "2"}))[0]  # a challenge needs receipts
    ok, msg = asyncio.run(k.act(b, "guide", {"op": "challenge", "line": "2", "evidence": [rb], "why": "the left gem reacts too"}))
    assert ok and any("challenged your lp85 guide line" in n["text"] for n in k.agents[a].inbox), msg
    view = k.observe(b, consume=False)
    assert f"the right gem reacts [{ra}]" in view and f"nothing else reacts [CONTESTED by {b}: {rb}" in view, view
    assert not asyncio.run(k.act(a, "guide", {"op": "challenge", "line": "1", "evidence": [ra]}))[0]  # own line: edit it


def test_a_long_guide_keeps_every_line_and_merging_combines_them():
    from alphateam.explore import GUIDE_MAX
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "explore": "1", "guide": "1", "sync": "1",
                                                           "carry": "1", "publish_solutions": "1"}), 0, "arc")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    game = w.games[0]
    k.tick = 1
    # past GUIDE_MAX nothing is refused: everyone sees the oldest lines shortened and the newest whole
    lines = [f"line {i}: " + ("the left gem turns the ring " * 20)[:480].strip() for i in range(12)]
    for i, line in enumerate(lines):
        ok, msg = asyncio.run(k.act((a, b)[i % 2], "guide", {"op": "add", "line": line}))
        assert ok, msg
    assert "over 4000 characters" in msg and len(w.guides[game][-1]["text"]) > GUIDE_MAX
    view = w.guide_text(game, fit=GUIDE_MAX)
    assert len(view) <= GUIDE_MAX + 200 and "older lines shortened" in view and f"12. {lines[11]}" in view, view[:300]
    ok, full = asyncio.run(k.act(c, "guide", {"op": "view"}))
    assert all(x in full for x in lines)
    # merge: c combines a's line 1 and b's line 2 into one shorter line without claiming either was wrong
    assert not asyncio.run(k.act(c, "guide", {"op": "merge", "lines": [1], "line": "x"}))[0]  # two or more lines
    assert not asyncio.run(k.act(c, "guide", {"op": "merge", "lines": [1, 2], "line": "y" * 601}))[0]  # one line's limit
    ok, msg = asyncio.run(k.act(c, "guide", {"op": "merge", "lines": [1, 2], "line": "lines 0-1: the left gem turns the ring"}))
    assert ok, msg
    cur = w.guides[game][-1]
    first = cur["text"].splitlines()[0]
    assert first == "lines 0-1: the left gem turns the ring" and lines[0] not in cur["text"] and lines[1] not in cur["text"]
    assert cur["lines"][first] == f"{a},{b},{c}" and cur["merged"] == lines[:2]
    assert lines[0] in w.guides[game][-2]["text"]  # the old lines stay in the history
    assert any("merged 2 lp85 guide lines" in n["text"] for n in k.agents[a].inbox)
    assert any("merged 2 lp85 guide lines" in n["text"] for n in k.agents[b].inbox)
    assert w.score()["guide_merges"] == 1
    # the merged line's co-authors are paid when it was in a solver's view at a first solve
    k.tick = 2
    before = {x: w.credit[x] for x in (a, b)}
    asyncio.run(k.act(c, "play", {"actions": LP85_L1[:15]}))
    k.tick = 3
    asyncio.run(k.act(c, "play", {"actions": LP85_L1[15:]}))
    assert all(w.credit[x] - before[x] >= w.rules["credit"]["reuse"] - 1e-9 for x in (a, b))


def test_ideas_are_listed_new_ones_pay_and_cited_pieces_earn_understanding():
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "explore": "1", "sync": "1", "carry": "1",
                                                           "guide": "1", "lessons": "1", "hypotheses": "1",
                                                           "publish_solutions": "1"}), 0, "arc")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    rid = lambda out: out.split("receipt ")[1].split("]")[0]
    idea = w.rules["credit"]["idea"]
    k.tick = 1
    # b tests an idea first: credit for a new idea
    asyncio.run(k.act(b, "plan", {"op": "claim", "hypothesis": "the left gem rotates the outer ring"}))
    rb = rid(asyncio.run(k.act(b, "play", {"actions": ["click(5,32)"]}))[1])
    before = w.credit[b]
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "done", "finding": "the ring did not turn", "outcome": "dead_end",
                                            "evidence": [rb]}))
    assert ok and "first test of this idea" in msg, msg
    assert w.credit[b] - before >= idea - 1e-9
    fb = msg.split("posted finding ")[1].split()[0]
    # c claims the same idea: it is pointed out, never refused, and testing it again earns no credit for a new idea
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "claim", "hypothesis": "The left gem rotates the outer ring!"}))
    assert ok and "Similar ideas already claimed" in msg and "REFUTED" in msg, msg
    rc = rid(asyncio.run(k.act(c, "play", {"actions": ["click(5,32)"]}))[1])
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "done", "finding": "no turn", "outcome": "dead_end", "evidence": [rc]}))
    assert ok and "tested before" in msg and "first test" not in msg, msg
    # c combines b's piece into a new claim; b is told
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "claim", "hypothesis": "pairs of clicks swap blocks, so the gem only "
                                                                            "matters after a swap", "builds_on": [fb]}))
    assert ok and any("building on your" in n["text"] for n in k.agents[b].inbox), msg
    pc = next(p for p in w.plans.values() if p["agent"] == c and p["status"] == "open")
    assert pc["builds_on"] == [fb] and w.parts[pc["entry"]] == [fb]
    # lessons are shared by all games
    assert asyncio.run(k.act(b, "guide", {"op": "add", "game": "lessons", "line": "edge counters are move budgets"}))[0]
    view = k.observe(a, consume=False)
    assert "IDEAS on lp85 level 1" in view and "REFUTED" in view and "testing until" in view, view
    assert "LESSONS shared by all games" in view and "edge counters are move budgets" in view
    # a completes the level citing c's claim: c earns credit for understanding, b half as much for the piece it built on
    k.tick = 2
    asyncio.run(k.act(a, "play", {"actions": LP85_L1[:15], "cites": [pc["entry"]]}))
    k.tick = 3
    got = {x: w.credit[x] for x in (b, c)}
    ok, out = asyncio.run(k.act(a, "play", {"actions": LP85_L1[15:], "note": f"done, thanks to {pc['entry']}"}))
    assert "LEVEL 1 COMPLETED" in out, out
    u = w.rules["credit"]["understanding"]
    paid = {x["to"]: x["credit"] for x in w.understanding_log[-1]["paid"]}
    assert paid == {c: u, b: u / 2}, w.understanding_log
    assert w.credit[c] - got[c] >= u - 1e-9 and w.credit[b] - got[b] >= u / 2 - 1e-9
    s = w.ideas_score(k)
    assert s["ideas_new"] == 1 and s["ideas_retested"] == 1 and s["ideas_retested_refuted"] == 1, s
    assert s["understanding_solves"] == 1 and w.explore_score(k)["lesson_lines"] == 1


def test_a_play_that_repeats_a_refuted_test_is_told_so():
    k = build(dict(CFG, world="arc", agents=2, world_opts={"games": "lp85", "explore": "1", "sync": "1", "carry": "1",
                                                           "hypotheses": "1", "publish_solutions": "1"}), 0, "arc")
    w = k.env
    a, b = sorted(k.agents, key=lambda x: int(x[1:]))
    k.tick = 1
    asyncio.run(k.act(a, "plan", {"op": "claim", "hypothesis": "clicking the top-left corner six times opens a door"}))
    out = asyncio.run(k.act(a, "play", {"actions": ["click(1,1)"] * 6}))[1]
    ra = out.split("receipt ")[1].split("]")[0]
    ok, msg = asyncio.run(k.act(a, "plan", {"op": "done", "finding": "nothing completed", "outcome": "dead_end",
                                            "evidence": [ra]}))
    dead = msg.split("posted finding ")[1].split()[0]
    k.tick = 2
    out = asyncio.run(k.act(b, "play", {"actions": ["click(2,62)"] + ["click(1,1)"] * 6}))[1]
    assert f"repeat part of a refuted test: {ra} behind dead end {dead}" in out, out
    out = asyncio.run(k.act(b, "play", {"actions": ["click(2,62)"] * 6}))[1]
    assert "refuted test" not in out
    assert w.ideas_score(k)["refuted_repeats"] == 1


def test_board_delivers_everything_once_ranks_search_by_use_and_lists_every_piece():
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "lp85", "explore": "1", "sync": "1", "carry": "1",
                                                           "guide": "1", "lessons": "1", "hypotheses": "1", "substrate": "1",
                                                           "board": "1", "publish_solutions": "1"}), 0, "arc")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    assert k.full_board and w.board_on
    rid = lambda out: out.split("receipt ")[1].split("]")[0]
    k.tick = 1
    asyncio.run(k.act(a, "look", {}))
    k.observe(a)  # a's first turn: nothing new yet
    # b claims ten ideas, more than the old eight-row view held, and records a dead end and a backed finding
    words = "amber basalt cobalt dahlia ember fennel garnet hazel indigo jasper".split()
    for x in words:
        assert asyncio.run(k.act(b, "plan", {"op": "claim", "hypothesis": f"the {x} tile completes the level"}))[0]
    r1 = rid(asyncio.run(k.act(b, "play", {"actions": ["click(5,32)"]}))[1])
    ok, msg = asyncio.run(k.act(b, "plan", {"op": "done", "finding": "the jasper tile does nothing", "outcome": "dead_end",
                                            "evidence": [r1]}))
    assert ok, msg
    dead = msg.split("posted finding ")[1].split()[0]
    asyncio.run(k.act(c, "plan", {"op": "claim", "hypothesis": "the ring rotates by one cell per click"}))
    r2 = rid(asyncio.run(k.act(c, "play", {"actions": ["click(5,32)"]}))[1])
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "finding", "finding": "the ring rotates one cell per click and wraps",
                                            "evidence": [r2]}))
    assert ok, msg
    fid = msg.split("posted finding ")[1].split()[0]
    # everything others posted reaches a once, grouped by type; nothing is cut
    k.tick = 2
    view = k.observe(a)
    fresh = [e for e in k.boards["global"].entries if e.author != a and e.tick == 1]
    assert f"NEW SINCE YOUR LAST TURN: all {len(fresh)} posts by others" in view, view
    assert "Ideas claimed on your game (11)" in view and "Dead ends and challenges on your game" in view, view
    assert all(e.id in view for e in fresh if w.entry_meta.get(e.id, {}).get("type") == "claim")
    assert "NEW SINCE YOUR LAST TURN: nothing." in k.observe(a)
    # every idea on the level is listed, and the backed finding nobody else has used is pointed out
    assert all(p["id"] in view for p in w.plans.values()), view
    assert "BACKED FINDINGS NOBODY ELSE HAS USED YET" in view and fid in view.split("BACKED FINDINGS")[1], view
    assert dead in view.split("DEAD ENDS on lp85 level 1")[1]
    # a uses c's finding in a play: it is no longer unused, the finding shows who used it, and search ranks it first
    asyncio.run(k.act(a, "play", {"actions": ["click(5,32)"], "cites": [fid]}))
    view = k.observe(a, consume=False)
    assert "BACKED FINDINGS NOBODY ELSE HAS USED YET" not in view and f"[used by {a}]" in view, view
    k.begin_turn(b)
    ok, out = asyncio.run(k.act(b, "search", {"what": "posts", "query": "ring rotates cell"}))
    first = out.splitlines()[1]
    assert ok and fid in first and f"cited by {a} play t2" in first, out
    w0 = k.cite_weight(fid)
    assert 0.29 < w0 < 0.31
    k.tick = 12
    assert k.cite_weight(fid) < 0.85 * w0  # use fades when it stops
    # guide lines are entries: never refused for space, each shown with its id, the least used shortened, none hidden
    lines = [f"line {i}: " + " ".join([words[i % 10]] * 95) for i in range(30)]
    for ln in lines:
        ok, msg = asyncio.run(k.act(b, "guide", {"op": "add", "line": ln[:590]}))
        assert ok, msg
    game = w.assigned[b]
    assert len(w.guides[game][-1]["text"]) > 16000
    view = k.observe(c, consume=False)
    g = view.split("GUIDE to lp85")[1].split("\n\n")[0]
    assert "30. (E" in g and "least used lines are shortened" in g, g
    ok, msg = asyncio.run(k.act(c, "guide", {"op": "merge", "lines": [1, 2], "line": "lines 0 and 1: amber and basalt"}))
    assert ok, msg
    merged = k.entry(msg.split("(")[-1].rstrip(")"))
    assert set(merged.refs[:2]) == {w.guide_ids[game][lines[0][:590]], w.guide_ids[game][lines[1][:590]]}


def test_evidence_reports_objects_keeps_frames_and_flags_moves_whose_effect_depends_on_the_state():
    """vc33 level 4 as the H_vc33_20_s0 run met it: the gate that works only when the tanks beside it are level."""
    start = str(Path(__file__).resolve().parents[1] / "tools" / "vc33_start_l4.json")
    k = build(dict(CFG, world="arc", agents=3, world_opts={"games": "vc33", "explore": "1", "sync": "1", "carry": "1",
                                                           "guide": "1", "hypotheses": "1", "substrate": "1", "board": "1",
                                                           "publish_solutions": "1", "evidence": "1", "start": start}), 0, "arc")
    w = k.env
    a, b, c = sorted(k.agents, key=lambda x: int(x[1:]))
    rid = lambda out: out.split("receipt ")[1].split("]")[0]
    k.tick = 1
    assert all(w.state[(x, w.assigned[x])]["levels"] == 3 for x in (a, b, c))  # every copy starts on level 4
    assert w.counters.lines[w.assigned[a]] == {(0, 0)}  # the step bar (row 0) was learned from inert clicks
    # two pours level the tanks beside the gate (it lights up); the gate click moves the key into the next tank
    ok, out = asyncio.run(k.act(a, "play", {"actions": ["click(16,62)", "click(16,62)", "click(13,44)"],
                                            "expect": "the key sinks twice, then maybe nothing"}))
    assert ok and out.startswith("You expected: the key sinks twice"), out
    assert "moved down 3: a 6x6 object (colours 4,b)" in out and "recoloured 1 -> c: 36 cells at r43-54 c12-14" in out, out
    assert "moved right 15: a 6x6 object (colours 4,b) from r49-54 c3-8 to r49-54 c18-23" in out, out
    r = rid(out)
    # anyone can open the receipt's frames, in-between ones too, without playing
    ok, seen = asyncio.run(k.act(b, "look", {"receipt": r, "step": 3, "region": [0, 49, 25, 54]}))
    assert ok and "what it changed: moved right 15" in seen and "bbbbbb" in seen.splitlines()[-1][3 + 18:3 + 24 + 1], seen
    n = len(w.receipts[r]["frames"][2])
    assert n > 1 and asyncio.run(k.act(b, "look", {"receipt": r, "step": 3, "frame": 1}))[1].count(f"frame 1 of {n}") == 1
    assert "the frame before its first move" in asyncio.run(k.act(b, "look", {"receipt": r, "step": 0}))[1]
    # from another state the same click does nothing, and the engine says where it did something
    ok, out = asyncio.run(k.act(b, "play", {"actions": ["click(13,48)"]}))
    assert "nothing changed but the counter" in out, out
    assert f"NOTE: click(13,48) changed nothing here, but click(13,44) (the object at r43-54 c12-14) changed" in out, out
    assert f"in {r} step 3 ({a} t1" in out and "after click(16,62), click(16,62)" in out and "depends on the state" in out, out
    # a guide line or dead end that names the click carries the record
    ok, msg = asyncio.run(k.act(c, "guide", {"op": "add", "line": "(13,44) is NOT a gate: inert from pristine (settled)"}))
    assert ok and "Engine records on the moves it names: a click at (13,44) changed" in msg and r in msg, msg
    view = k.observe(c, consume=False)
    assert "[engine records: a click at (13,44) changed" in view.split("GUIDE to vc33")[1], view
    ok, msg = asyncio.run(k.act(c, "plan", {"op": "finding", "finding": "click(13,44) is inert", "outcome": "dead_end"}))
    assert ok and "Engine records on the moves it names" in msg, msg
    assert "Engine records: a click at (13,44)" in k.entry(msg.split("posted finding ")[1].split()[0]).content
    # moves already played from the same states are named, and look outcome answers without using steps
    ok, out = asyncio.run(k.act(c, "play", {"actions": ["click(16,62)", "click(16,62)"]}))
    assert "all 2 moves repeat recorded runs from the same states" in out and r in out, out
    used = w.steps[(c, k.tick)]
    ok, out = asyncio.run(k.act(c, "look", {"outcome": ["click(13,44)", "click(40,62)"]}))
    assert ok and "click(13,44): " in out and "moved right 15" in out and f"[{r} step 3, {a} t1]" in out, out
    assert "click(40,62): not recorded from this state" in out and w.steps[(c, k.tick)] == used, out
    s = w.score()
    assert s["ev_state_notes"] >= 1 and s["ev_exact_repeats"] == 1 and s["ev_outcome_hits"] == 1 and s["ev_expects"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
