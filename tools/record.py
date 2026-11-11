"""Query the record of code-world runs (alphateam/record.py; DESIGN.md section 22).

  .venv/bin/python tools/record.py build --from runs_recipes/XSV1 runs_recipes/XSV2   # copy runs in, write index and timelines
  .venv/bin/python tools/record.py runs                      # the index: one row a run
  .venv/bin/python tools/record.py show RUN                  # a run round by round
  .venv/bin/python tools/record.py events RUN --round 6      # every action of round 6, with what came back
  .venv/bin/python tools/record.py events RUN --agent A5 --op publish --grep csvio --refused
  .venv/bin/python tools/record.py trace RUN --round 6 --agent A5 --kind reply   # the full trace: what was shown, said, done
  .venv/bin/python tools/record.py diff RUN_A RUN_B          # two runs side by side
  .venv/bin/python tools/record.py checks XSV2b              # the counts the proposals made predictions about, a row a run
  .venv/bin/python tools/record.py rules                     # the settings a recipe can set, and what a frame fixes
  .venv/bin/python tools/record.py shown                     # what an agent is shown at the start of a run

RUN is a run's id in the index, or any part of it that names one run. The record lives in runs_recipes/record unless
--record says otherwise. Events are numbered by their line in the run's events.jsonl, so a finding can cite them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphateam import record  # noqa: E402


def shown() -> str:
    """The texts every agent gets under the starting rules: the system prompt, the actions with their descriptions, and
    the first turn. Built on the in-memory harness, so the file listing of the first turn is a stand-in."""
    from alphateam.paired import code_frame, start
    from alphateam.primitives import system_prompt, tool_schemas
    from alphateam.recipe import seed
    k = start(dict(code_frame(fake=True), grade=False), seed("code"), 0)
    aid = sorted(k.agents)[0]
    tools = tool_schemas(k, aid)
    out = ["# What an agent is shown (starting rules)", "", "## System prompt", "", system_prompt(k), "", "## Actions on offer", ""]
    for t in tools:
        fn = t.get("function") or t
        params = (fn.get("parameters") or fn.get("input_schema") or {}).get("properties") or {}
        out.append(f"### {fn.get('name')}\n\n{fn.get('description', '')}\n\nparameters: {json.dumps(params, ensure_ascii=False)}\n")
    try:
        import tempfile
        k.env.workdir = tempfile.mkdtemp()
        k.tick = 1
        k.env.on_tick_start(k)
        k.begin_turn(aid)
        out += ["## The first turn (in-memory harness)", "", k.observe(aid)]
    except Exception as exc:  # the texts above are what matters; the first turn needs a started world
        out += ["## The first turn", "", f"(not built here: {exc.__class__.__name__}: {exc})"]
    return "\n".join(out) + "\n"


def rules() -> str:
    from alphateam.paired import code_frame
    from alphateam.recipe import registry
    out = ["# The rules a recipe can set (code world)", "",
           "A recipe is a value for each setting below, plus lessons carried from earlier societies. It never names an agent.", "",
           "| setting | part | values | starting rules | what it sets |", "|---|---|---|---|---|"]
    for key, s in registry("code").items():
        needs = "; acts only while " + ", ".join(f"{k} is {'/'.join(str(v) for v in vs)}" for k, vs in s.needs) if s.needs else ""
        out.append(f"| {key} | {s.part} | {s.allowed()} | {'on' if s.default is True else 'off' if s.default is False else s.default} | {s.doc}{needs} |")
    f = code_frame()
    out += ["", "## What a frame fixes (never a setting)", "",
            f"{f['agents']} agents, {f['ticks']} rounds, {f['calls'] + 1} replies an agent a round, {f['world_opts']['steps']} shell steps "
            f"an agent a round, {f['world_opts']['probes']} probes of the reference program in all, a reading budget of {f['attention']} "
            "a turn, the model and its effort, the task, and both scores."]
    return "\n".join(out) + "\n"


def diff(to: Path, a: str, b: str) -> str:
    ra, rb = record.find(to, a), record.find(to, b)
    da, db = record.digest(ra, ra.name), record.digest(rb, rb.name)
    out = [f"{'':28} {ra.name[-34:]:>36} {rb.name[-34:]:>36}"]
    for key in da:
        if key in ("id", "dir", "held_out_by_round"):
            continue
        if da[key] != db[key]:
            out.append(f"{key:28} {str(da[key])[:36]:>36} {str(db[key])[:36]:>36}")
    out.append("")
    out.append("held-out by round:")
    out.append("  " + " ".join(f"{v:>3}" for v in da["held_out_by_round"]))
    out.append("  " + " ".join(f"{v:>3}" for v in db["held_out_by_round"]))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default="runs_recipes/record")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--from", dest="roots", nargs="+", required=True)
    sub.add_parser("runs")
    s = sub.add_parser("show")
    s.add_argument("run")
    e = sub.add_parser("events")
    e.add_argument("run")
    e.add_argument("--round", type=int)
    e.add_argument("--agent")
    e.add_argument("--op")
    e.add_argument("--grep")
    e.add_argument("--refused", action="store_true")
    e.add_argument("--chars", type=int, default=400)
    e.add_argument("--limit", type=int, default=200)
    t = sub.add_parser("trace")
    t.add_argument("run")
    t.add_argument("--round", type=int)
    t.add_argument("--agent")
    t.add_argument("--kind", choices=["system", "turn", "reply", "action", "closing"])
    t.add_argument("--grep")
    t.add_argument("--chars", type=int, default=4000)
    t.add_argument("--limit", type=int, default=60)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    c = sub.add_parser("checks")
    c.add_argument("experiments", nargs="*", help="experiment names as in the index (default: all)")
    sub.add_parser("rules")
    sub.add_parser("shown")
    a = p.parse_args()
    to = (ROOT / a.record) if not Path(a.record).is_absolute() else Path(a.record)
    if a.cmd == "build":
        roots = [(ROOT / r) if not Path(r).is_absolute() else Path(r) for r in a.roots]
        record.build([r for r in roots if r.exists()], to)
        (to / "RULES.md").write_text(rules())
        (to / "SHOWN.md").write_text(shown())
    elif a.cmd == "runs":
        print((to / "INDEX.md").read_text())
    elif a.cmd == "show":
        print((record.find(to, a.run) / "timeline.md").read_text())
    elif a.cmd == "events":
        lines = record.events(record.find(to, a.run), a.round, a.agent, a.op, a.grep, a.refused, a.chars)
        print("\n".join(lines[:a.limit]))
        if len(lines) > a.limit:
            print(f"... {len(lines) - a.limit} more (raise --limit, or narrow the query)")
    elif a.cmd == "trace":
        folder = record.find(to, a.run)
        if not (folder / "trace.jsonl").exists():
            sys.exit(f"{folder.name} has no full trace (runs before 2026-11-10 11:00 kept only events.jsonl, with arguments "
                     "cut at 160 characters and results at 240)")
        lines = record.trace_lines(folder, a.round, a.agent, a.kind, a.grep, a.chars)
        print("\n".join(lines[:a.limit]))
        if len(lines) > a.limit:
            print(f"... {len(lines) - a.limit} more (raise --limit, or narrow the query)")
    elif a.cmd == "diff":
        print(diff(to, a.a, a.b))
    elif a.cmd == "checks":
        rows = [r for r in json.loads((to / "index.json").read_text()) if r.get("full") and r.get("agents") == 10
                and (not a.experiments or r["experiment"] in a.experiments)]
        keys = None
        groups = {}
        for r in rows:
            c = record.checks(to / r["dir"])
            keys = keys or list(c)
            groups.setdefault(r["change"], []).append((r, c))
        short = {"publish_calls": "publish", "landings": "land", "landings_no_gain": "land+0", "landed_by_the_world": "by world",
                 "conflict_refusals": "confl", "repeat_conflict_pairs": "rep.pairs", "repeat_conflict_refusals": "rep.refus",
                 "unseen_per_agent_with_a_conflict": "unseen/agent hit", "unseen_per_agent_without": "unseen/agent not",
                 "agents_with_a_conflict": "agents hit", "commands_missing_at_end": "cmds missing", "wrong_action_refusals": "wrong action",
                 "moved_aside": "aside", "regression_refusals": "regr", "repeat_regression_pairs": "rep.regr", "files_landed": "files",
                 "shell_calls": "shell", "messages": "msgs", "claims": "claims", "looks_at_head": "head"}
        print(f"{'rules':28} {'run':30} hidden held " + " ".join(f"{short[k]:>{max(6, len(short[k]))}}" for k in keys))
        for name, items in groups.items():
            for r, c in items:
                print(f"{name[:28]:28} {r['id'][-30:]:30} {str(r['hidden']):>6} {str(r['held_out']):>4} "
                      + " ".join(f"{('-' if c[k] is None else c[k])!s:>{max(6, len(short[k]))}}" for k in keys))
    elif a.cmd == "rules":
        print(rules())
    elif a.cmd == "shown":
        print(shown())


if __name__ == "__main__":
    main()
