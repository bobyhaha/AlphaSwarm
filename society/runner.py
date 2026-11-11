"""Experiment conditions, the tick loop, and run outputs (run.json, viewer.html, report.md)."""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

from .world import PRIMITIVES, CustomAction, Team, Board, World, clip

VIEWER_TEMPLATE = Path(__file__).resolve().parent.parent / "viewer" / "template.html"

# Each condition differs only in which primitives/ops exist and the starting structure.
CONDITIONS = {
    "self_org": "Full action space; no starting structure (default).",
    "no_design": "Self-organizing, but design_action is removed (isolates the value of an evolving action space).",
    "flat": "No teams: connect only links/unlinks; boards, plans and custom actions remain.",
    "fixed": "A prescribed two-level hierarchy with representatives; agents cannot restructure it.",
    "single": "One generalist agent with a larger per-tick action budget (pair with --token-budget).",
}


def build_world(scenario: dict, condition: str, max_actions: int, blank_personas: bool, team_size: int = 4) -> World:
    people = [{"name": p["name"], "persona": "" if blank_personas else p.get("persona", "")} for p in scenario["people"]]
    allowed: dict = {p: None for p in PRIMITIVES}
    if condition == "no_design":
        allowed.pop("design_action")
    elif condition in ("flat", "fixed"):
        allowed["connect"] = {"link", "unlink", "vote"}
    elif condition == "single":
        people = [{"name": "Solo", "persona": "" if blank_personas else "A generalist researcher covering every specialty the mission needs."}]
        allowed["connect"] = {"link", "unlink"}
        max_actions *= 3
    elif condition != "self_org":
        raise ValueError(f"unknown condition '{condition}' (choose from {', '.join(CONDITIONS)})")
    world = World(scenario["mission"], people, max_actions=max_actions, allowed=allowed)
    if condition == "fixed":
        _prescribe_hierarchy(world, team_size)
    return world


def _prescribe_hierarchy(world: World, team_size: int):
    """A conventional org chart: a director in the top team, groups of `team_size` underneath, each with a rep."""
    ids = list(world.agents)

    def make(name, charter, parent, founder, reps):
        tid = world.new_id("T")
        world.teams[tid] = Team(tid, name, charter, founder, parent, 0, rule="rep", reps=set(reps), open=False)
        world.boards[f"team:{tid}"] = Board(f"team:{tid}", tid)
        return tid

    director, rest = ids[0], ids[1:]
    groups = [rest[i:i + team_size] for i in range(0, len(rest), team_size)]
    top = make("Program", "Run the mission end to end.", None, director, [director] + [g[0] for g in groups])
    world.agents[director].team = top
    for i, g in enumerate(groups, 1):
        tid = make(f"Group {i}", "Deliver the blocks the program assigns.", top, g[0], [g[0]])
        for a in g:
            world.agents[a].team = tid


def inherit(world: World, previous_run: Path):
    """Level-3 hook: a new society starts with a previous society's actions and organizational lessons."""
    prev = json.loads(previous_run.read_text())
    for c in prev.get("custom", []):
        world.custom[c["name"]] = CustomAction(c["name"], c["description"], c["procedure"], c["params"],
                                               f"heritage:{c['author']}", 0, c.get("version", 1))
    lessons = prev.get("meta", {}).get("lessons")
    if lessons:
        world.post("global", "society", "summary", "Organizational lessons from a previous society:\n" + lessons)


async def run(world: World, backend, ticks: int, out_dir: Path, meta: dict, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    world.total_ticks = ticks
    world.summarizer = backend.summarize
    world.custom_runner = backend.run_custom
    world.snapshots.append(world.snapshot())
    rng = random.Random(seed)
    started = time.time()
    for t in range(1, ticks + 1):
        left = world.budget_left()
        if left is not None and left <= 0:
            print(f"token budget exhausted before tick {t}")
            break
        world.tick = t
        order = list(world.agents)
        rng.shuffle(order)
        await asyncio.gather(*(backend.turn(world, aid) for aid in order))
        world.snapshots.append(world.snapshot())
        m = world.snapshots[-1]["metrics"]
        print(f"t{t:>3}  teams {m['teams']:>2}  depth {m['depth']}  free {m['free']:>2}  links {m['links']:>3}  "
              f"custom {m['custom']:>2}  blocks {m['done']}/{m['blocks']} done  reorg {m['reorganizations']:>3}  "
              f"tokens {m['tokens']:>10,}  ({time.time() - started:.0f}s)", flush=True)
        save(world, out_dir, meta)
    meta["lessons"] = await backend.retrospective(world)
    data = save(world, out_dir, meta)
    (out_dir / "report.md").write_text(report(data))
    return data


def save(world: World, out_dir: Path, meta: dict) -> dict:
    data = world.export(meta)
    payload = json.dumps(data, default=list)
    (out_dir / "run.json").write_text(payload)
    if VIEWER_TEMPLATE.exists():
        html = VIEWER_TEMPLATE.read_text().replace("/*__RUN_DATA__*/null", payload.replace("</", "<\\/"))
        (out_dir / "viewer.html").write_text(html)
    return data


def report(data: dict) -> str:
    meta, snap = data["meta"], data["snapshots"][-1]
    teams = {t["id"]: t for t in snap["teams"]}
    names = {a["id"]: a["name"] for a in data["agents"]}
    L = [f"# Society run: {meta.get('condition', '')} ({meta.get('backend', '')})", "",
         f"**Mission.** {meta['mission']}", "",
         f"Ticks {meta['ticks']} · agents {len(data['agents'])} · tokens {snap['metrics']['tokens']:,} · "
         f"reorganizations {snap['metrics']['reorganizations']} · blocks done {snap['metrics']['done']}/{snap['metrics']['blocks']}", "",
         "## Final organization", ""]

    def walk(parent, depth):
        for t in sorted((t for t in teams.values() if t["parent"] == parent), key=lambda t: int(t["id"][1:])):
            members = ", ".join(f"{m} {names[m]}" for m in t["members"]) or "—"
            L.append(f"{'  ' * depth}- **{t['id']} {t['name']}** (rule {t['rule']}, lead {t['lead'] or '—'}): {members}  \n"
                     f"{'  ' * depth}  _{clip(t['charter'], 200)}_")
            walk(t["id"], depth + 1)

    walk(None, 0)
    free = [a["id"] for a in snap["agents"] if a["team"] is None]
    if free:
        L.append(f"- Free agents: {', '.join(free)}")
    L += ["", "## Block tree and deliverables", ""]
    blocks = {b["id"]: b for b in data["blocks"]}

    def walk_blocks(bid, depth):
        b = blocks[bid]
        L.append(f"{'#' * min(depth + 3, 6)} {b['id']} {b['title']} [{b['status']}, owner {b['owner'] or '—'}]")
        if bid != "M0" and b["spec"]:
            L.append(f"_Spec:_ {b['spec']}")
        if b["deliverable"]:
            L.extend(["", b["deliverable"]])
        for r in b["reviews"]:
            L.append(f"- review by {r['by']} (t{r['t']}): **{r['verdict']}** {r['notes']}")
        L.append("")
        for c in [x for x in blocks.values() if x["parent"] == bid]:
            walk_blocks(c["id"], depth + 1)

    walk_blocks("M0", 0)
    L += ["## Action space at the end", "", "| action | author | version | adopters | uses | description |", "|---|---|---|---|---|---|"]
    adopters = {c["name"]: c["adopters"] for c in snap["custom"]}
    for c in data["custom"]:
        L.append(f"| x_{c['name']} | {c['author']} | {c['version']} | {adopters.get(c['name'], 0)} | {c['uses']} | {clip(c['description'], 120)} |")
    L += ["", "## Organizational retrospective", "", meta.get("lessons") or "(none)", "",
          "## Structure over time", "", "| t | teams | depth | free | links | custom | done | reorg | tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for s in data["snapshots"]:
        m = s["metrics"]
        L.append(f"| {s['t']} | {m['teams']} | {m['depth']} | {m['free']} | {m['links']} | {m['custom']} | {m['done']} | {m['reorganizations']} | {m['tokens']:,} |")
    return "\n".join(L) + "\n"
