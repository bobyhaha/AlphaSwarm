"""Experimental conditions, the tick loop, and run outputs."""
from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from collections import Counter
from pathlib import Path

from .hidden_laws import HiddenLaws
from .kernel import PRIMITIVES, Board, Group, Kernel, clip
from .llm import Fatal, cost_usd
from .metrics import snapshot

VIEWER_TEMPLATE = Path(__file__).resolve().parent / "viewer.html"
KEEP_OPS = ("connect.", "protocol", "design.", "delegate.", "lab.publish", "critique", "summarize")

# Conditions differ only in which primitives exist and the starting structure.
CONDITIONS = {
    "self_org": "All primitives; no starting structure (default).",
    "no_design": "Self-organizing, but design_action is removed (what does an evolving action space add?).",
    "flat": "No groups: connect can only follow and unfollow (is structure worth its cost?).",
    "fixed": "A prescribed, frozen org chart: random groups of 8 with a delegate, supergroups of up to 6 groups, one program group.",
    "oracle": "Like fixed, but the org chart follows the hidden modules and super-modules (a designer who knows the answer).",
    "isolated": "Same agents and home tasks, but no communication: each agent reads and writes only its own board and works only on its own tasks.",
    "single": "One agent with every instrument and the society's total measurement budget; 3x actions and reading per tick.",
}


def build(cfg: dict, seed: int, structure: str) -> Kernel:
    condition = cfg["condition"]
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition '{condition}' (choose from {', '.join(CONDITIONS)})")
    single = condition == "single"
    allowed = {p: None for p in PRIMITIVES}
    if condition == "no_design":
        allowed.pop("design_action")
    elif condition in ("flat", "single"):
        allowed["connect"] = {"follow", "unfollow"}
        allowed.pop("protocol")
        if condition == "single":
            allowed.pop("link")
    elif condition in ("fixed", "oracle"):
        allowed["connect"] = {"follow", "unfollow"}
    elif condition == "isolated":
        allowed = {p: None for p in ("read", "search", "write", "reflect")}
    n = 1 if single else cfg["agents"]
    k = Kernel(n, max_actions=cfg["actions"] * (3 if single else 1), attention=cfg["attention"] * (3 if single else 1),
               allowed=allowed)
    k.isolated = condition == "isolated"
    k.max_calls = cfg.get("calls")  # per-call limit: this many replies per tick, any number of actions in each
    k.local = str((cfg.get("world_opts") or {}).get("local", "")).lower() in ("1", "true", "yes")
    k.substrate = str((cfg.get("world_opts") or {}).get("substrate", "")).lower() in ("1", "true", "yes")
    k.sync = str((cfg.get("world_opts") or {}).get("sync", "")).lower() in ("1", "true", "yes")
    k.private_boards = str((cfg.get("world_opts") or {}).get("private", "")).lower() in ("1", "true", "yes")
    if k.sync and k.isolated:
        raise ValueError("sync shares everything, so it cannot run in the isolated condition")
    k.push_items = int((cfg.get("world_opts") or {}).get("sync_items", 8))
    # board=1: every entry reaches each agent once, and search ranks by relevance and citation weight (needs sync)
    k.full_board = k.sync and str((cfg.get("world_opts") or {}).get("board", "")).lower() in ("1", "true", "yes")
    wo = cfg.get("world_opts") or {}
    if str(wo.get("islands", "")).lower() in ("1", "true", "yes"):  # islands (kernel.py): embargo, publishing, group cap
        if not k.full_board:
            raise ValueError("islands needs sync=1 and board=1 (posts open into the shared board)")
        quota, _, window = str(wo.get("publish", "1/3")).partition("/")
        k.islands = {"embargo": int(wo.get("embargo", 3)), "quota": int(quota), "window": int(window or 1),
                     "cap": int(wo.get("group_cap", 5))}
    k.teaming = str(wo.get("teaming", "")).lower() in ("1", "true", "yes")  # kernel.py: what working with others does
    k.messages_on = str(wo.get("messages", "")).lower() in ("1", "true", "yes")  # kernel.py: the message action
    k.continuity = int((cfg.get("world_opts") or {}).get("continuity", 0) or 0)
    k.final_moves = str((cfg.get("world_opts") or {}).get("final_moves", "")).lower() in ("1", "true", "yes")
    k.token_budget = cfg.get("token_budget")
    k.cost_budget = cfg.get("cost_budget")
    if cfg.get("world", "hidden") == "hidden":
        env = HiddenLaws(n, seed=seed, structure=structure, instruments=cfg["instruments"], redundancy=cfg.get("redundancy"),
                         experiments=cfg["experiments"], seed_runs=cfg["seed_runs"], noise=cfg["noise"],
                         validate_every=cfg["validate_every"], shocks=cfg.get("shocks"), single=single, world_agents=cfg["agents"])
    elif cfg["world"] == "arc":  # ARC-AGI-3 games played in parallel copies
        from .arc_world import ArcWorld
        env = ArcWorld(n, seed=seed, single=single, world_agents=cfg["agents"], opts=dict(cfg.get("world_opts") or {}))
    elif cfg["world"] == "dig":  # DiG-bench text games, each agent on its own instance (alphateam/dig_world.py)
        from .dig_world import DigWorld
        env = DigWorld(n, seed=seed, single=single, world_agents=cfg["agents"], opts=dict(cfg.get("world_opts") or {}))
    elif cfg["world"] == "code":  # rebuild a real program, with te-next's evidence layer (alphateam/code_world.py)
        from .code_world import CodeWorld
        env = CodeWorld(n, seed=seed, single=single, world_agents=cfg["agents"], opts=dict(cfg.get("world_opts") or {}))
    elif cfg["world"] == "science":  # FrontierScience: a sheet of answers, audited at random (alphateam/science_world.py)
        from .science_world import ScienceWorld
        env = ScienceWorld(n, seed=seed, single=single, world_agents=cfg["agents"], opts=dict(cfg.get("world_opts") or {}))
    elif cfg["world"] == "design":  # one shared design with conflicting block objectives
        from .design_world import DesignWorld
        opts = dict(cfg.get("world_opts") or {}, shocks=cfg.get("shocks"))
        env = DesignWorld(n, seed=seed, single=single, world_agents=cfg["agents"], opts=opts)
    else:  # a real benchmark (alphateam/bench)
        from .bench import make_world
        env = make_world(cfg["world"], cfg, n, seed=seed, single=single)
    k.env = env
    env.isolated = k.isolated
    env.local = getattr(k, "local", False)
    if condition in ("fixed", "oracle"):
        _prescribe(k, by_module=condition == "oracle", rng=random.Random(seed))
    env.setup(k)
    if cfg.get("org_window"):  # continual organizational learning (design world)
        from .org_learning import OrgLearner
        k.org_learner = OrgLearner(cfg["org_window"], stall=bool(cfg.get("org_stall")), judge=cfg.get("org_judge") or "level")
    if cfg.get("rule_window"):  # evolving the credit rules and the substrate (ARC world, explore=1)
        from .rule_learning import RuleLearner
        k.rule_learner = RuleLearner(cfg["rule_window"])
    if cfg.get("reflect_window"):  # the society reviews itself; any rule may change, accuracy judges (code world)
        from .reflection import ReflectionLoop
        k.reflection = ReflectionLoop(cfg["reflect_window"], floor=cfg.get("reflect_floor", 0.5))
    return k


def _group(k: Kernel, name: str, purpose: str, members: list[str], delegates: list[str]) -> str:
    gid = k.new_id("G")
    k.groups[gid] = Group(gid, name, purpose, members[0], 0, open=False, delegates=set(delegates),
                          protocol="Prescribed by the designers; membership is fixed.")
    k.boards[f"mem:{gid}"] = Board(f"mem:{gid}", gid)
    k.boards[f"pub:{gid}"] = Board(f"pub:{gid}", gid)
    for m in members:
        k._add_member(gid, m)
    return gid


def _prescribe(k: Kernel, by_module: bool, rng: random.Random):
    """A conventional org chart: teams with a delegate, divisions of teams, one program at the top."""
    env = k.env
    agents = list(k.agents)
    if hasattr(env, "tasks") and not any(env.home.values()):
        _prescribe_bench(k, agents, rng, aligned=by_module)
        return
    purpose = getattr(env, "team_purpose", None)  # worlds may word team purposes themselves
    if by_module:
        home = {}
        for a in agents:
            if hasattr(env, "home_module"):
                home[a] = env.home_module(a)
                continue
            mods = Counter(env.module_of_out[o] for o in env.instruments[a])
            home[a] = mods.most_common(1)[0][0]
        teams = {}
        for m in env.modules:
            members = [a for a in agents if home[a] == m["id"]]
            if members:
                teams[m["id"]] = _group(k, f"Module team {m['id']}", purpose(m) if purpose else
                                        "Discover the laws of the outputs assigned to this team.", members, [members[0]])
        divisions = []
        for s in range(env.n_super):
            ts = [teams[m["id"]] for m in env.modules if m["super"] == s and m["id"] in teams]
            if ts:
                divisions.append(_group(k, f"Division {s}", "Coordinate the teams of this division.", ts, ts))
    else:
        rng.shuffle(agents)
        teams = [_group(k, f"Team {i + 1}", purpose(None) if purpose else "Discover laws for the outputs your members measure.",
                        agents[j:j + 8], [agents[j]])
                 for i, j in enumerate(range(0, len(agents), 8))]
        divisions = [_group(k, f"Division {i + 1}", "Coordinate the teams of this division.", teams[j:j + 6], teams[j:j + 6])
                     for i, j in enumerate(range(0, len(teams), 6))]
    if len(divisions) > 1:
        _group(k, "Program", "Run the whole mission.", divisions, divisions)


def _prescribe_bench(k: Kernel, agents: list[str], rng: random.Random, aligned: bool = True):
    """Org chart for a benchmark with no individual assignments: one team per task family (agents split in
    proportion to its tasks, at least one each), told its task ids; divisions per domain; a program on top. Inside a
    team, members still divide the work themselves. With aligned=False (condition fixed) the same team sizes and
    division layout get random task lists instead, so structure is kept but its match to the benchmark is not."""
    env = k.env
    fam_tasks = {m["id"]: [t.id for t in env.tasks.values() if env.family_idx[t.family] == m["id"]] for m in env.modules}
    if not aligned:
        deck = [t for ids in fam_tasks.values() for t in ids]
        rng.shuffle(deck)
        fam_tasks = {m: [deck.pop() for _ in ids] for m, ids in fam_tasks.items()}
    order = sorted(fam_tasks, key=lambda m: -len(fam_tasks[m]))
    share = {m: 1 for m in order[:len(agents)]}
    for _ in range(len(agents) - len(share)):  # largest remaining tasks-per-agent gets the next agent
        m = max(share, key=lambda m: (len(fam_tasks[m]) / share[m], -m))
        share[m] += 1
    rng.shuffle(agents)
    teams, i = {}, 0
    for m in order:
        if m not in share:
            continue
        members = agents[i:i + share[m]]
        i += share[m]
        ids = sorted(fam_tasks[m], key=lambda t: int(t[1:]))
        teams[m] = _group(k, f"Team {len(teams) + 1}", f"Answer this team's {len(ids)} tasks: {', '.join(ids)}.", members, [members[0]])
    divisions = []
    for s in range(env.n_super):
        ts = [teams[m["id"]] for m in env.modules if m["super"] == s and m["id"] in teams]
        if ts:
            divisions.append(_group(k, f"Division {s + 1}", "Coordinate the teams of this division.", ts, ts))
    if len(divisions) > 1:
        _group(k, "Program", "Run the whole mission.", divisions, divisions)


async def run(k: Kernel, backend, ticks: int, out_dir: Path, meta: dict, seed: int = 0, quiet: bool = False,
              start: int = 0, horizon: int | None = None, final: bool = True) -> dict:
    """Run a society to tick `ticks`. A run can be taken in parts (alphateam/paired.py): `start` is the tick already
    reached, `horizon` the tick the whole run ends at (what agents are told), and final=False leaves out the end of a
    run (final grading, the retrospective, the saved files) so that a later call can carry on from where this stops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k.total_ticks = horizon or ticks
    k.summarizer = backend.summarize
    k.custom_runner = backend.run_custom
    if not start:
        k.snapshots.append(snapshot(k))
    k.cost_fn = lambda: cost_usd(k, meta.get("preset", ""), meta.get("model", ""))
    if hasattr(backend, "complete"):  # lets benchmark worlds call a judge or answerer, billed to this run
        k.env.llm = lambda *a, **kw: backend.complete(k, *a, **kw)
    if hasattr(k.env, "workdir"):  # where benchmark worlds keep agents' code workspaces
        k.env.workdir = str((out_dir / "work").resolve())
    rng = getattr(k, "order_rng", None) or random.Random(seed)  # kept on the kernel, so a copied run draws the same order
    k.order_rng = rng
    started = time.time()
    flushed = traced = 0

    def write_trace(mode: str = "a"):  # the whole of what was shown, said, done and returned (kernel.trace)
        nonlocal traced
        rows = getattr(k, "trace", [])
        with open(out_dir / "trace.jsonl", mode) as f:
            for row in rows[traced:]:
                f.write(json.dumps(row, default=str) + "\n")
        traced = len(rows)

    write_trace("w")
    with open(out_dir / "events.jsonl", "w") as events:
        try:
            for t in range(start + 1, ticks + 1):
                if k.exhausted():
                    meta["stopped"] = f"budget exhausted before tick {t}"
                    print(meta["stopped"])
                    break
                k.tick = t
                k.tick_counts.clear()
                k.expire_tasks()
                k.env.on_tick_start(k)
                order = k.active_agents()
                rng.shuffle(order)
                try:
                    async with asyncio.TaskGroup() as tg:
                        for aid in order:
                            tg.create_task(backend.turn(k, aid))
                except* Fatal as eg:
                    raise eg.exceptions[0] from None
                res = k.env.on_tick_end(k)
                if inspect.isawaitable(res):  # benchmark graders may call an LLM judge or a simulator
                    await res
                snap = snapshot(k)
                k.snapshots.append(snap)
                rules = getattr(k, "rule_learner", None)
                if rules:  # revise the credit rules and the substrate from proposals and what has paid
                    n = len(rules.history)
                    await rules.on_tick_end(k, backend)
                    if len(rules.history) > n and not quiet:
                        h = rules.history[-1]
                        print(f"     rules at t{t}: {json.dumps(h['changed'])}"
                              + (f" {h['why'][:200]}" if h["why"].startswith("(revision failed") else ""), flush=True)
                learner = getattr(k, "org_learner", None)
                if learner:  # reflect, decide, and enforce the next teamwork strategy
                    n = len(learner.history)
                    await learner.on_tick_end(k, backend)
                    if len(learner.history) > n and not quiet:
                        h = learner.history[-1]
                        print(f"     organization at t{t}: window score {h['score']:.3f}, {h['decision']}; next: "
                              f"{json.dumps({x: h['next'][x] for x in ('teams', 'council', 'rights', 'explore', 'results_to')})}", flush=True)
                loop = getattr(k, "reflection", None)
                if loop:
                    n = len(loop.history)
                    await loop.on_tick_end(k, backend)
                    if len(loop.history) > n and not quiet:
                        h = loop.history[-1]
                        print(f"     review at t{t}: held-out {h['held_out']} ({h['verdict_on_last']}); changed "
                              f"{json.dumps(sorted(h['changed']))}"
                              + (f"; {h['why'][:200]}" if h["why"].startswith("(review failed") else ""), flush=True)
                for e in k.events[flushed:]:
                    events.write(json.dumps(e) + "\n")
                flushed = len(k.events)
                write_trace()
                (out_dir / "metrics.json").write_text(json.dumps([s["metrics"] | {"t": s["t"]} for s in k.snapshots]))
                if not quiet:
                    m = snap["metrics"]
                    fmt = lambda v: "  -  " if v is None else f"{v:+.2f}"
                    spent = k.cost_fn()
                    comp = f"/{m['score_composite']:.2f}" if m.get("score_composite") is not None else ""
                    sc = "  -  " if m["score"] is None else f"{m['score']:.3f}"
                    print(f"t{t:>3} score {sc}{comp} cov {m['coverage']:.2f} | groups {m['groups']:>4} depth {m['depth']} "
                          f"super {m['supergroups']:>3} in-group {m['in_group']:.2f} links {m['links']:>4} | mirror "
                          f"{fmt(m['mirror_l1'])}/{fmt(m['mirror_l2'])} hier {fmt(m.get('hier_align'))} "
                          f"| reorg {m['reorganizations']:>4} | tokens {m['tokens']:>12,}" + (f" ${spent:.2f}" if spent is not None else "")
                          + f" ({time.time() - started:.0f}s)", flush=True)
        except Fatal as exc:
            print(f"stopped: {exc}")
            meta["stopped"] = str(exc)
        for e in k.events[flushed:]:
            events.write(json.dumps(e) + "\n")
        flushed = len(k.events)
    write_trace()
    if final and getattr(k, "closing_round", False) and "stopped" not in meta and hasattr(backend, "closing"):
        # after the last tick, with the shared work final: each agent may still propose rule changes and leave lessons
        # for later societies (alphateam/recipe.py Rulebook). Nothing here can move this run's score.
        try:
            async with asyncio.TaskGroup() as tg:
                for aid in k.active_agents():
                    tg.create_task(backend.closing(k, aid))
        except* Exception as eg:  # the run's result stands whatever happens here
            meta["closing_cut"] = f"{eg.exceptions[0].__class__.__name__}: {eg.exceptions[0]}"
        with open(out_dir / "events.jsonl", "a") as events:
            for e in k.events[flushed:]:
                events.write(json.dumps(e) + "\n")
        flushed = len(k.events)
        write_trace()
    if not final:  # a part of a run: the caller carries on (or closes it with a final call)
        meta["cost_usd"] = cost_usd(k, meta.get("preset", ""), meta.get("model", ""))
        return {"meta": meta, "snapshots": k.snapshots}
    if hasattr(k.env, "finalize"):  # final grading by the benchmark's own grader
        try:
            await k.env.finalize(k)
        except Exception as exc:
            meta["grading_error"] = f"{exc.__class__.__name__}: {exc}"
            print(f"final grading failed: {meta['grading_error']}")
        k.snapshots[-1] = snapshot(k)
        (out_dir / "metrics.json").write_text(json.dumps([s["metrics"] | {"t": s["t"]} for s in k.snapshots]))
        m = k.snapshots[-1]["metrics"]
        print(f"final score {m['score']} coverage {m['coverage']}", flush=True)
    if "stopped" not in meta:
        try:
            meta["lessons"] = await backend.retrospective(k, [_brief(s["metrics"]) | {"t": s["t"]} for s in k.snapshots])
        except Exception as exc:  # the retrospective is optional; keep the run's results
            meta["lessons"] = f"(retrospective failed: {exc})"
    if getattr(k, "org_learner", None):
        meta["org_learning"] = k.org_learner.export()
    if getattr(k, "rule_learner", None):
        meta["rule_learning"] = k.rule_learner.export()
    if getattr(k, "reflection", None):
        meta["reflection"] = k.reflection.export()
    meta["cost_usd"] = cost_usd(k, meta.get("preset", ""), meta.get("model", ""))
    return save(k, out_dir, meta)


def _brief(m: dict) -> dict:
    keys = ("score", "coverage", "groups", "depth", "supergroups", "in_group", "reorganizations", "protocol_changes",
            "custom", "mirror_l1", "mirror_l2", "tokens")
    return {x: m.get(x) for x in keys}


def culture(k: Kernel, lessons: str | None) -> dict:
    """What a society hands to the next one: its actions, its library, its group protocols and its lessons."""
    groups = sorted(k.active_groups(), key=lambda g: -len(k.agents_inside(g.id)))[:10]
    world = getattr(k.env, "culture", None)  # what a world adds, e.g. the code world's law, norms and tools
    return {"custom": k.export_custom(),
            "library": [{"author": e.author, "kind": e.kind, "content": e.content} for e in k.boards["library"].entries[-30:]],
            "protocols": [{"name": g.name, "purpose": g.purpose, "protocol": g.protocol, "agents": len(k.agents_inside(g.id)),
                           "depth": k.depth(g.id), "delegates": len(g.delegates)} for g in groups if g.protocol or g.purpose],
            "lessons": lessons or "", **({"world": world()} if world else {})}


def inherit(k: Kernel, c: dict):
    from .kernel import CustomAction
    for x in c.get("custom", []):
        k.custom[x["name"]] = CustomAction(x["name"], x["description"], x["procedure"], x["params"], f"heritage:{x['author']}", 0, x["version"])
    for e in c.get("library", []):
        k.post("library", "heritage", e["kind"], e["content"], notify=False)
    if c.get("protocols"):
        k.post("library", "heritage", "lesson", "Group protocols used by the previous society (largest groups first):\n" + "\n".join(
            f"- '{p['name']}' ({p['agents']} agents, depth {p['depth']}): {clip(p['purpose'], 120)} | {clip(p['protocol'], 200)}"
            for p in c["protocols"]), notify=False)
    if c.get("lessons"):
        k.post("library", "heritage", "lesson", "Organizational lessons from the previous society:\n" + c["lessons"], notify=False)
    world = getattr(k.env, "inherit_culture", None)
    if world and c.get("world"):
        world(k, c["world"])


def save(k: Kernel, out_dir: Path, meta: dict) -> dict:
    env = k.env
    world = env.export()
    agents = {a: (env.agent_export(a) if hasattr(env, "agent_export") else
                  {"instruments": env.instruments.get(a, []), "record": env.track_record(a), "experiments": env.used[a]})
              | {"active": x.active, "notes": x.notes[-2:], "follows": len(x.follows), "tokens": sum(k.ledger[a].values()),
                 "actions": sorted(x.actions)} for a, x in k.agents.items()}
    final_groups = {g.id: {"mem": [k.header(e) for e in k.boards[f"mem:{g.id}"].entries[-6:]],
                           "pub": [k.header(e) for e in k.boards[f"pub:{g.id}"].entries[-4:]]} for g in k.active_groups()}
    data = {"meta": meta, "world": world, "snapshots": k.snapshots, "agents": agents, "group_posts": final_groups,
            "custom": k.export_custom(), "tasks": [vars(t) for t in k.tasks.values()],
            "usage": dict(k.usage), "counts": dict(k.counts), "culture": culture(k, meta.get("lessons"))}
    (out_dir / "run.json").write_text(json.dumps(data, default=list))
    with open(out_dir / "entries.jsonl", "w") as f:
        for b in k.boards.values():
            for e in b.entries:
                f.write(json.dumps(vars(e)) + "\n")
    write_viewer(out_dir, data, k.events)
    (out_dir / "report.md").write_text(report(data))
    return data


def write_viewer(out_dir: Path, data: dict, events: list[dict]):
    if VIEWER_TEMPLATE.exists():
        viewer = dict(data, events=_sample_events(events), culture=None)
        html = VIEWER_TEMPLATE.read_text().replace("/*__RUN_DATA__*/null", json.dumps(viewer, default=list).replace("</", "<\\/"))
        (out_dir / "viewer.html").write_text(html)


def rebuild_viewer(out_dir: Path):
    """Re-render viewer.html for a saved run (for example after the viewer template changed)."""
    data = json.loads((out_dir / "run.json").read_text())
    with open(out_dir / "events.jsonl") as f:
        events = [json.loads(line) for line in f]
    write_viewer(out_dir, data, events)


def _sample_events(events: list[dict], per_tick: int = 120) -> list[dict]:
    """The viewer gets every structural event (up to a cap) plus a sample of the rest, per tick."""
    rng = random.Random(0)
    by_tick: dict[int, list[int]] = {}
    for i, e in enumerate(events):
        by_tick.setdefault(e["t"], []).append(i)
    kept = []
    for _, idx in sorted(by_tick.items()):
        keep = [i for i in idx if events[i]["op"].startswith(KEEP_OPS)]
        rest = [i for i in idx if not events[i]["op"].startswith(KEEP_OPS)]
        if len(keep) > per_tick * 2:
            keep = rng.sample(keep, per_tick * 2)
        kept += keep + rng.sample(rest, min(len(rest), per_tick))
    return [events[i] for i in sorted(kept)]


def report(data: dict) -> str:
    meta, world, snaps = data["meta"], data["world"], data["snapshots"]
    last = snaps[-1]
    m = last["metrics"]
    groups = {g["id"]: g for g in last["structure"]["groups"]}
    contained = {x for g in groups.values() for x in g["members"] if x in groups}
    f = lambda v: "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    L = [f"# AlphaTeam run: {meta.get('condition')} · {meta.get('backend')} {meta.get('model', '')}", ""]
    if world.get("kind") in ("arc", "dig"):
        what = "ARC-AGI-3 games in parallel copies" if world["kind"] == "arc" else (
            "DiG-bench games, one shared instance per game" if world.get("copies_mode") == "shared"
            else "DiG-bench games, one instance per agent")
        L += [f"{len(data['agents'])} agents · {len(snaps) - 1} ticks · {what}", "",
              f"**Score {f(m['score'])}** (mean over games of the most levels any copy completed / levels to win) · "
              f"{m.get('steps')} game steps", "", "| game | best levels | levels to win | players |", "|---|---|---|---|"]
        for g in world["games"]:
            players = sorted((a for a, gg in world["assigned"].items() if gg == g), key=lambda a: int(a[1:]))
            L.append(f"| {g} | {world['best'][g]} | {world['win'].get(g, '?')} | {', '.join(players)} |")
        L += ["", "First copy to reach each level: " + ("; ".join(f"{x['game'].split('-')[0]} L{x['level']} by {x['agent']} t{x['t']}"
                                                               for x in world["firsts"]) or "none"),
              "", "## Organizational retrospective", "", meta.get("lessons") or "(none)", ""]
        return "\n".join(L)
    if world.get("kind") == "code":
        held = world["held_out"][-1] if world["held_out"] else {}
        s = world.get("structure_measures", {})
        L += [f"{len(data['agents'])} agents · {len(snaps) - 1} ticks · rebuild xsv from its behaviour", "",
              f"**Hidden tests {world['hidden'].get('passed')} of {world['hidden'].get('total')}** · held-out {held.get('passed')} of "
              f"{held.get('total')} · {len(world['merges'])} merges, {len(world['rejections'])} rejected files · law v{m.get('law_version')} "
              f"· goal {world['law']['goal']}" + (f" · cost ${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else ""), "",
              "Held-out by tick: " + ", ".join(f"t{h['t']} {h['passed']}" for h in world["held_out"]), "",
              f"Shape: {s.get('builders')} agents landed work, the top builder made {s.get('top_builder_share')} of verified gains; working "
              f"share {s.get('working_share')}, organizing share {s.get('organizing_share')}; {s.get('roles_taken')} roles taken; "
              f"{s.get('teams')} teams; depth {m['depth']}; most read {s.get('most_read')}; {s.get('work_not_at_head')} files never assembled", "",
              f"Mechanisms: {m.get('claims')} claims on code, {m.get('proposals')} proposals ({m.get('proposals_landed')} landed), "
              f"{m.get('tools')} tools ({m.get('tool_uses_by_others')} uses by others), {m.get('contracts')} contracts "
              f"({m.get('contracts_paid')} paid), {m.get('payments')} payments, {m.get('tests')} tests, {m.get('findings')} findings", "",
              "## Organizational retrospective", "", meta.get("lessons") or "(none)", ""]
        return "\n".join(L)
    if world.get("kind") == "science":
        fin, s = world.get("final") or {}, world.get("structure_measures", {})
        L += [f"{len(data['agents'])} agents · {len(snaps) - 1} ticks · FrontierScience", "",
              f"**Score {fin.get('score')} on {fin.get('n')} scored questions** · olympiad {fin.get('olympiad')} · research pass {fin.get('research_pass')} "
              f"· research points {fin.get('research_points')} of 10 · goal {world['law']['goal']} · law v{m.get('law_version')}"
              + (f" · cost ${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else ""), "",
              f"Audited questions, by the audit's judge: {fin.get('audited')}", "",
              "Audit gains by tick: " + ", ".join(f"t{h['t']} {h['passed']}" for h in world["held_out"]), "",
              f"Shape: {s.get('attempts')} attempts by {s.get('attempters')} agents, {s.get('independent')} independent, {s.get('reconciled')} reconciling, "
              f"{s.get('contests')} contested; subject focus {s.get('subject_focus')}; working share {s.get('working_share')}, organizing share "
              f"{s.get('organizing_share')}; {s.get('roles_taken')} roles taken; {s.get('teams')} teams; depth {m['depth']}", "",
              "## Organizational retrospective", "", meta.get("lessons") or "(none)", ""]
        return "\n".join(L)
    if world.get("kind") == "design":
        L += [f"{len(data['agents'])} agents · {len(snaps) - 1} ticks · design world ({world['D']} knobs, {world['n_blocks']} "
              f"blocks in {world['n_super']} super-blocks)", "",
              f"**Score {f(m['score'])}** (share of the achievable improvement over the starting design) · knobs at optimum "
              f"{f(m.get('at_optimum'))} · shared-knob reversals {m.get('shared_reversals')} · tokens {m['tokens']:,}"
              + (f" · cost ${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else ""), "",
              "Remaining gap by level (improvement still available by fixing that level's knobs to the optimum): "
              + ", ".join(f"{lv} {v:.2f}" for lv, v in world["level_gaps"].items()), ""]
    elif "benchmark" in world:
        L += [f"{len(data['agents'])} agents · {len(snaps) - 1} ticks · benchmark {world['benchmark']} ({world['n_tasks']} tasks, "
              f"{len(world['modules'])} families in {world['n_super']} domains)", "",
              f"**Score {f(m['score'])}** (benchmark grader, hidden) · coverage {m['coverage']:.2f} · tokens {m['tokens']:,}"
              + (f" · cost ${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else ""), ""]
        ev = [e for e in world.get("evaluations", []) if e.get("final")]
        if ev:
            L += ["By domain: " + ", ".join(f"{d} {v:.3f}" for d, v in sorted(ev[-1]["by_domain"].items())), ""]
    else:
        L += [f"{len(data['agents'])} agents · {len(snaps) - 1} ticks · world {world['structure']} ({world['n_out']} outputs, "
              f"{world['n_in']} inputs, {len(world['modules'])} modules in {world['n_super']} super-modules)", "",
              f"**Score {m['score']:.3f}** (test R², hidden) · coverage {m['coverage']:.2f} · structure F1 {m['structure_f1']:.2f} · "
              f"tokens {m['tokens']:,}" + (f" · cost ${meta['cost_usd']:.2f}" if meta.get("cost_usd") is not None else ""), ""]
    if meta.get("stopped"):
        L += [f"> Stopped early: {meta['stopped']}", ""]
    L += ["## Over time", "", "| t | score | composite | cov | groups | depth | supergroups | in group | links | mirror L1 | mirror L2 | hier align | expertise | reorg | protocol | tokens |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in snaps:
        x = s["metrics"]
        L.append(f"| {s['t']} | {f(x['score'])} | {f(x.get('score_composite'))} | {f(x['coverage'])} | {x['groups']} | {x['depth']} | "
                 f"{x['supergroups']} | {f(x['in_group'])} | {x.get('links', 0)} | {f(x.get('mirror_l1'))} | {f(x.get('mirror_l2'))} | "
                 f"{f(x.get('hier_align'))} | {f(x.get('expertise'))} | {x['reorganizations']} | {x['protocol_changes']} | {x['tokens']:,} |")
    if world.get("shocks"):
        L += ["", "Shocks: " + "; ".join(f"t{s['t']} {s['kind']} (" + ", ".join(f"{key} {v}" for key, v in s.items() if key not in ("t", "kind")) + ")" for s in world["shocks"])]
    L += ["", "## Final organization (largest top-level groups)", ""]
    budget = [60]

    def walk(gid, depth):
        if budget[0] <= 0 or depth > 3:
            return
        budget[0] -= 1
        g = groups[gid]
        agents = [x for x in g["members"] if x not in groups]
        L.append(f"{'  ' * depth}- **{gid} {g['name']}**: {g['inside']} agents inside, depth {g['depth']}, delegates "
                 f"{', '.join(g['delegates']) or 'anyone'}{', closed' if not g['open'] else ''}. {clip(g['purpose'], 120)}"
                 + (f" _Protocol: {clip(g['protocol'], 160)}_" if g["protocol"] else "") + (f" Agents: {', '.join(agents[:12])}{'…' if len(agents) > 12 else ''}" if agents else ""))
        for sub in sorted((x for x in g["members"] if x in groups), key=lambda x: -groups[x]["inside"]):
            walk(sub, depth + 1)

    for gid in sorted((g for g in groups if g not in contained), key=lambda g: -groups[g]["inside"])[:20]:
        walk(gid, 0)
    if not groups:
        L.append("No groups.")
    L += ["", "## Action space at the end", ""]
    if data["custom"]:
        L += ["| action | author | version | adopters | uses | description |", "|---|---|---|---|---|---|"]
        L += [f"| x_{c['name']} | {c['author']} | {c['version']} | {c['adopters']} | {c['uses']} | {clip(c['description'], 120)} |" for c in data["custom"]]
    else:
        L.append("No custom actions.")
    if world.get("kind") == "design":
        shared = [s for s in world["sets"] if s["level"] != "private"]
        L += ["", "## Shared knobs", "", f"{len(world['sets'])} knob changes, {len(shared)} on shared knobs "
              f"({sum(s['reversal'] for s in shared)} reversals; {sum(s['informed'] for s in shared)} by an agent whose block uses "
              f"the knob; {sum(s['spanning'] for s in shared)} on behalf of a group spanning several of its blocks).", "",
              "| knob | level | blocks | value | optimum | start | last set by |", "|---|---|---|---|---|---|---|"]
        for kn in sorted((x for x in world["knobs"] if x["level"] != "private"), key=lambda x: (x["level"], int(x["knob"][1:]))):
            by = kn["last_set"]
            L.append(f"| {kn['knob']} | {kn['level']} | {len(kn['blocks'])} | {kn['value']:g} | {kn['optimum']:g} | {kn['start']:g} | "
                     f"{(by[0] + ('/' + by[1] if by[1] else '') + f' t{by[2]}') if by else '–'} |")
        L += ["", "## Organizational retrospective", "", meta.get("lessons") or "(none)", ""]
        return "\n".join(L)
    if "benchmark" in world:
        tasks = world["tasks"]
        answered = [t for t in tasks if t["answer"]]
        L += ["", f"## Answers ({len(answered)} of {len(tasks)} tasks)", "", "| task | family | score | author | answer |", "|---|---|---|---|---|"]
        for t in sorted(answered, key=lambda t: -(t["score"] or 0))[:40]:
            a = t["answer"]
            L.append(f"| {t['id']} | {t['family']} | {f(t['score'])} | {a['author']}{'/' + a['group'] if a['group'] else ''} | "
                     f"{clip(a['answer'], 90).replace('|', '/')} |")
        L += ["", "## Organizational retrospective", "", meta.get("lessons") or "(none)", ""]
        return "\n".join(L)
    laws = sorted(world["laws"], key=lambda x: -(x["test"] or 0))
    L += ["", f"## Laws ({len(laws)} published of {world['n_out']})", "", "| output | module | test R² | val R² | author | formula | truth |", "|---|---|---|---|---|---|---|"]
    for law in laws[:15] + (laws[-10:] if len(laws) > 25 else laws[15:]):
        L.append(f"| y{law['output']} | {law['module']} | {f(law['test'])} | {f(law['val'])} | {law['author']}{'/' + law['group'] if law['group'] else ''} | "
                 f"`{clip(law['formula'], 70)}` | `{clip(world['truth'][law['output']], 70)}` |")
    L += ["", "## Organizational retrospective", "", meta.get("lessons") or "(none)", ""]
    return "\n".join(L)
