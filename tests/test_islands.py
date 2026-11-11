"""islands=1 (kernel.py): posts reach the poster's island at once and everyone else when they open; nothing is lost."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_incentives import CFG  # noqa: E402
from alphateam.runner import build  # noqa: E402

START = str(Path(__file__).resolve().parents[1] / "tools" / "vc33_start_l4.json")
OPTS = {"games": "vc33", "explore": "1", "sync": "1", "carry": "1", "guide": "1", "hypotheses": "1", "substrate": "1",
        "board": "1", "publish_solutions": "1", "evidence": "1", "start": START}


def world(islands: bool, **extra):
    opts = dict(OPTS, **({"islands": "1", "embargo": "3", "publish": "1/3", "group_cap": "2"} if islands else {}), **extra)
    k = build(dict(CFG, world="arc", agents=4, world_opts=opts), 0, "arc")
    k.total_ticks = 20
    return k, k.env, sorted(k.agents, key=lambda x: int(x[1:]))


def act(k, aid, name, args):
    ok, out = asyncio.run(k.act(aid, name, args))
    return ok, out


def test_islands_hide_until_open_publish_cap_and_lose_nothing():
    k, w, (a, b, c, d) = world(True)
    k.tick = 1
    for x in (a, b, c, d):
        k.observe(x)  # clear the start-of-run delivery
    ok, msg = act(k, a, "plan", {"op": "claim", "hypothesis": "the colour-1 wall segments are gates that open when tanks are level"})
    assert ok, msg
    claim = msg.split("(")[1].split(")")[0]
    ok, msg = act(k, a, "guide", {"op": "add", "line": "pumps move 3 rows from the tank across the wall"})
    assert ok, msg
    line = msg.split("(")[1].split(")")[0]
    ok, msg = act(k, a, "guide", {"op": "add", "line": "the b mark on wall c42-44 is the exit"})
    exit_line = msg.split("(")[1].split(")")[0]
    # another island sees none of it, and is told how many posts are held and when the next opens
    view = k.observe(b)
    assert "gates that open when tanks are level" not in view and "pumps move 3 rows" not in view, view
    assert "posts by other islands are not open to you yet" in view and "by t4" in view, view
    assert "YOUR ISLAND: just you" in view and "share entry=E.., 1 per 3 ticks each" in view, view
    ok, out = act(k, b, "search", {"what": "posts", "query": "gates level"})
    assert claim not in out, out
    ok, out = act(k, b, "read", {"board": "global", "entry": claim})
    assert not ok and "opens to you by t4" in out, out
    ok, out = act(k, b, "plan", {"op": "claim", "hypothesis": "the colour-1 wall segments are gates opened by level tanks"})
    assert ok and a not in out.split("Others' open plans")[-1][:60], out  # a's open plan is not named to b
    # forming a group makes an island: b now sees a's posts at once; the cap holds the group at 2
    ok, msg = act(k, a, "connect", {"op": "form_group", "name": "pumps", "purpose": "map the tanks"})
    gid = msg.split("formed ")[1].split()[0]
    assert act(k, b, "connect", {"op": "join", "target": gid})[0]
    ok, out = act(k, d, "connect", {"op": "join", "target": gid})
    assert not ok and "more than 2 agents" in out, out
    view = k.observe(b)
    assert claim in view and "YOUR ISLAND: you and " + a in view, view
    assert "pumps move 3 rows" in view.split("GUIDE to vc33")[1], view
    # c, outside, edits the guide before the embargo ends: lines it cannot see stay as they are
    cur = w.guides[w.assigned[c]][-1]["v"]
    ok, msg = act(k, c, "guide", {"op": "edit", "base": cur, "text": "c's own line", "why": "rewrite"})
    assert ok, msg
    text = w.guides[w.assigned[c]][-1]["text"]
    assert "pumps move 3 rows" in text and "the b mark on wall c42-44 is the exit" in text and "c's own line" in text, text
    # a publishes one line now; a second publish in the same window is refused
    ok, out = act(k, a, "share", {"entry": exit_line})
    assert ok and "published" in out, out
    ok, out = act(k, a, "share", {"entry": line})
    assert not ok and "1 entries per 3 ticks" in out, out
    view = k.observe(c)
    assert exit_line in view and "the b mark on wall c42-44 is the exit" in view.split("GUIDE to vc33")[1], view
    assert "pumps move 3 rows" not in view.split("GUIDE to vc33")[1], view
    # evidence is open at once: c can open a's receipt
    ok, out = act(k, a, "play", {"actions": ["click(16,62)"]})
    r = out.split("receipt ")[1].split("]")[0]
    assert act(k, c, "look", {"receipt": r, "step": 1})[0]
    # by the embargo tick everything a posted at t1 reaches c, each entry once
    k.tick = 4
    view = k.observe(c)
    assert claim in view and line in view and "pumps move 3 rows" in view.split("GUIDE to vc33")[1], view
    again = k.observe(c)
    assert claim not in again.split("NEW SINCE YOUR LAST TURN")[1], again
    assert all(k.visible(x, e) for x in (a, b, c, d) for e in k.index.values() if e.tick <= 1)
    # a post about a level opens when that level falls, before its embargo
    k.tick = 5
    ok, msg = act(k, d, "plan", {"op": "finding", "finding": "pumping C to r46 lights the second gate", "outcome": "unclear"})
    f = msg.split("posted finding ")[1].split()[0]
    assert not k.visible(a, k.entry(f))
    w.best[w.assigned[d]] = w.entry_meta[f]["level"]
    assert k.visible(a, k.entry(f))
    ex = w.explore_export(k)
    assert ex["islands"]["published"][exit_line][1] == a and f in ex["islands"]["opened_by_level"]


def test_the_group_cap_holds_for_teams_and_nested_groups():
    k, w, (a, b, c, d) = world(True, delegation="1")
    k.tick = 1
    ok, msg = act(k, a, "lead", {"op": "found", "name": "tanks"})
    team = msg.split("you lead team ")[1].split()[0].rstrip(";,")
    assert act(k, b, "lead", {"op": "join", "team": team})[0]
    ok, out = act(k, c, "lead", {"op": "join", "team": team})
    assert not ok and "more than 2 agents" in out, out
    ok, msg = act(k, c, "connect", {"op": "form_group", "name": "outer"})
    outer = msg.split("formed ")[1].split()[0]
    ok, out = act(k, a, "connect", {"op": "join", "target": outer, "as": team})  # the team of 2 joining c's group: 3
    assert not ok and "more than 2 agents" in out, out
    assert k.island(c) == {c} and k.island(a) == {a, b}


def test_teaming_says_what_working_together_does_and_messages_cross_islands():
    from alphateam.primitives import system_prompt
    k, w, (a, b, c, d) = world(True, teaming="1")
    sp = system_prompt(k)
    assert "WORKING WITH OTHERS" in sp and "Islands: your posts reach your island" in sp and "at most 2 agents" in sp
    # the start file's levels are records like published solutions, open to everyone
    recs = [e for e in k.index.values() if "was solved before this run" in e.content]
    assert [w.entry_meta[e.id]["level"] for e in recs] == [1, 2, 3]
    assert sum(len(e.content.split(": ", 1)[1].split(", ")) for e in recs) == 61
    k.tick = 1
    for x in (a, b, c, d):
        k.observe(x)
    ok, msg = act(k, a, "plan", {"op": "claim", "hypothesis": "the colour-1 segments are gates"})
    ok, msg = act(k, a, "guide", {"op": "add", "line": "pumps move 3 rows"})
    view = k.observe(b)
    m = view.split("MEMBERSHIP")[1]
    assert f"Not yet open to you: {a} 2 (claim on level 4, guide line on level 4; opens by t4)" in m, m
    assert "Groups: none yet" in m and "connect op=join target=G.." in m, m
    act(k, a, "connect", {"op": "form_group", "name": "gates"})
    assert "Groups: G1 'gates' (" + a + "; 1 places)" in k.observe(c).split("MEMBERSHIP")[1]
    # a post addressed to someone reaches it at once, across islands; others still wait
    ok, msg = act(k, a, "write", {"board": "global", "content": f"@{c} could you pump C to r46 and click the band?"})
    assert ok, msg
    e = max(k.index.values(), key=lambda e: int(e.id[1:]))
    assert k.visible(c, e) and not k.visible(d, e)
    assert any(f"mentioned you in {e.id}" in n["text"] for n in k.agents[c].inbox)


def test_messages_founders_directions_island_guides_and_unique_credit():
    from alphateam.primitives import system_prompt, tool_schemas
    k, w, (a, b, c, d) = world(True, teaming="1", messages="1", island_guides="1", unique_credit="1")
    sp = system_prompt(k)
    assert "message to=A12 text=.." in sp and "Each island keeps its own guide" in sp and "0.5 each, up to 3" in sp
    assert "message" in [t["name"] for t in tool_schemas(k, a)]
    k.tick = 1
    for x in (a, b, c, d):
        k.observe(x)
    # a direct message reaches its addressee during its turn, across islands, and is kept
    ok, out = act(k, a, "message", {"to": c, "text": "can you pump C to r46 and click the band?"})
    assert ok and out.startswith("sent M1"), out
    assert any("MESSAGE M1: can you pump C" in n["text"] for n in k.agents[c].inbox)
    assert k.messages_log[0]["to"] == c and "MESSAGE M1" in k.interrupts(c)
    # a group's purpose is its direction; a full group's refusal names the others turned away and the way out
    ok, msg = act(k, a, "connect", {"op": "form_group", "name": "doors", "purpose": "door rules: which tanks light which band"})
    gid = msg.split("formed ")[1].split()[0]
    assert act(k, b, "connect", {"op": "join", "target": gid})[0]
    ok, out = act(k, c, "connect", {"op": "join", "target": gid})
    assert not ok and "connect op=form_group" in out, out
    ok, out = act(k, d, "connect", {"op": "join", "target": gid})
    assert not ok and f"also turned away lately and still in no group: {c}" in out, out
    m = k.observe(c).split("MEMBERSHIP")[1]
    assert "direction: door rules" in m and "; full)" in m and f"turned away by a full group lately: {d}" in m, m
    ok, out = act(k, c, "plan", {"op": "claim", "hypothesis": "door rules: the tanks beside a band light it"})
    assert f"close to {gid}'s direction" in out, out
    # each island keeps its own guide; other islands' open lines are listed, and where they meet is shown
    assert act(k, a, "guide", {"op": "add", "line": "the band lights when both tanks beside it are level"})[0]
    assert act(k, c, "guide", {"op": "add", "line": "the band lights when the boat sits beside it in the tank"})[0]
    k.tick = 5  # past the embargo
    va, vc = k.observe(a), k.observe(c)
    ga = va.split("GUIDE to vc33")[1].split("OTHER ISLANDS")[0]
    assert "both tanks beside it are level" in ga and "boat sits beside it" not in ga, ga
    assert f"OTHER ISLANDS' GUIDES (open lines; guide op=view island=G1 or A5 shows one): {c} 1 lines" in va, va
    assert "WHERE ISLANDS' GUIDES MEET" in va and "both tanks beside it" in va.split("MEET")[1], va
    ok, out = act(k, a, "guide", {"op": "view", "island": c})
    assert ok and "boat sits beside it" in out, out
    cur = w.guides[w.assigned[c]][-1]["v"]  # c rewrites its own guide: a's island's line stays
    assert act(k, c, "guide", {"op": "edit", "base": cur, "text": "the band lights when the boat is level with it"})[0]
    assert "both tanks beside it are level" in w.guides[w.assigned[c]][-1]["text"]
    # when the level falls, ideas tested by one island only earn their testers credit
    for x, hyp in ((a, "door rules: pump both tanks level then click the band"), (d, "the exit is the marker of the boat's colour")):
        ok, out = act(k, x, "plan", {"op": "claim", "hypothesis": hyp})
        assert act(k, x, "plan", {"op": "finding", "finding": "tried it", "outcome": "unclear"})[0]
    before = dict(w.credit)
    w._first_solve(k, b, w.assigned[b], w._problem(w.assigned[b]))
    paid = {e["agent"] for e in w.unique_log}
    assert d in paid and w.credit[d] > before.get(d, 0), w.unique_log


def test_a_start_file_can_carry_an_earlier_societys_explanations():
    start = str(Path(__file__).resolve().parents[1] / "tools" / "vc33_start_l5x.json")
    k, w, (a, b, c, d) = world(False, start=start)
    sol = w.solution_of[(w.assigned[a], 4)]
    exps = [k.entry(x) for x in w.explanations[sol]]
    assert len(exps) == 2 and all(w.entry_meta[x.id]["type"] == "explanation" for x in exps)
    assert "level with the b marker" in exps[1].content and "earlier society" in exps[0].content


def test_without_islands_everything_reaches_everyone_at_once():
    k, w, (a, b, c, d) = world(False)
    k.tick = 1
    for x in (a, b, c, d):
        k.observe(x)
    ok, msg = act(k, a, "plan", {"op": "claim", "hypothesis": "the colour-1 wall segments are gates"})
    claim = msg.split("(")[1].split(")")[0]
    view = k.observe(b)
    assert claim in view and "not open to you yet" not in view and "YOUR ISLAND" not in view, view
    ok, out = act(k, a, "share", {"entry": claim})
    assert not ok and "there are no islands" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
