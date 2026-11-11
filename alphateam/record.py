"""The record of code-world runs, laid out for whoever proposes the next change (DESIGN.md section 22).

A run leaves events.jsonl (every action, with what came back), entries.jsonl (the boards), run.json (what the world
kept: landings, rejections, holds, tasks, plans, each agent's shell trace), metrics.json and result.json. This module
copies those into one folder, a run a subfolder, and derives two things from them:

  INDEX.md / index.json   one row a run: the rules it ran under, both scores, and counts of what happened
  runs/<id>/timeline.md   one line a round: what landed, what was turned away and why, holds, tasks, plans

The index and the timelines are summaries, and summaries are for finding where to look. What happened is in the
copied files. Meta-Harness (Lee et al., arXiv 2603.28052, Table 3) measured the difference: a proposer given scores
and summaries did no better than one given scores alone, and one given the raw traces did far better. Our own
example is in RESEARCH_LOG.md (2026-11-10): every counter of the worst run looked ordinary, and its events showed
that one agent had held every command file for four rounds.

Nothing here reads a run's work folder, the grader, or anything of the hidden tests: a run's files hold counts only.
No society runs under this module (alphateam/paired.py OUTER).
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .recipe import registry

FILES = ("job.json", "result.json", "run.json", "events.jsonl", "trace.jsonl", "entries.jsonl", "metrics.json", "report.md")
HELD, HIDDEN = 179, 1182
EARLY = 8  # rounds counted as early: by then the spread between runs is as large as it gets


def _rows(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def short(path: str) -> str:
    return path.replace("prog/", "")


def change(recipe: dict | None) -> str:
    """A recipe as its difference from the starting rules of its world."""
    if not recipe:
        return "(no recipe)"
    reg = registry(recipe.get("world", "code"))
    settings = recipe.get("settings") or {}
    diff = [f"{key}={'on' if v is True else 'off' if v is False else v}" for key, v in sorted(settings.items())
            if key in reg and reg[key].default != v]
    advice = len(recipe.get("advice") or [])
    if advice:
        diff.append(f"+{advice} lesson{'s' if advice > 1 else ''}")
    return ", ".join(diff) or "starting rules"


def run_dirs(root: Path) -> list[Path]:
    """Every folder under an experiment that holds a finished run (scripted dry runs and probes left out)."""
    found = []
    for res in sorted(root.glob("**/result.json")):
        parts = res.relative_to(root).parts
        if "work" in parts or "dry" in parts or any(p.startswith("probe-") for p in parts):
            continue
        if (res.parent / "run.json").exists() and (res.parent / "events.jsonl").exists():
            found.append(res.parent)
    return found


def run_id(root: Path, run: Path) -> str:
    parts = [p for p in run.relative_to(root).parts if p != "runs"]
    return ".".join([root.name] + parts)


def digest(run: Path, rid: str = "") -> dict:
    """One run as a row of the index. Counts only; every count can be checked against the run's files."""
    res = json.loads((run / "result.json").read_text())
    job = json.loads((run / "job.json").read_text()) if (run / "job.json").exists() else {}
    full = json.loads((run / "run.json").read_text())
    w, meta = full.get("world") or {}, full.get("meta") or {}
    events = _rows(run / "events.jsonl")
    acts = [e for e in events if e.get("op") not in ("intent", "closing.note")]
    scores = [round((s or 0) * HELD) for s in res.get("scores") or []]
    first = next((i for i, v in enumerate(scores) if v > 0), None)
    gains = [b - a for a, b in zip(scores, scores[1:])]
    merges, rej = w.get("merges") or [], w.get("rejections") or []
    conflicts = [x for x in rej if x.get("reason") == "conflict"]
    holds = [c for c in w.get("claims") or [] if c.get("paths")]
    by_agent = Counter(m["agent"] for m in merges)
    frame = job.get("frame") or meta.get("config") or {}
    return {
        "id": rid or run.name, "recipe": res.get("recipe"), "change": change(job.get("recipe")), "purpose": res.get("purpose"),
        "agents": frame.get("agents"), "rounds": res.get("ticks"), "full": bool(frame) and res.get("ticks") == frame.get("ticks"),
        "model": meta.get("model"), "effort": meta.get("reasoning_effort"), "closing_round": bool(frame.get("closing")),
        "hidden": res.get("hidden"), "held_out": scores[-1] if scores else None, "held_out_by_round": scores,
        "first_landing_round": first,
        "rounds_without_gain": sum(1 for i, g in enumerate(gains) if first is not None and i + 1 > first and g <= 0),
        "landings": len(merges), "landings_without_unseen_gain": sum(1 for m in merges if not m.get("verified")),
        "files_landed": len({f for m in merges for f in m.get("files") or []}),
        "files_landed_by_several": sum(1 for f, who in _who(merges).items() if len(who) > 1),
        "top_builder_share": round(max(by_agent.values()) / len(merges), 2) if merges else None,
        "offers": sum(1 for e in acts if e.get("op") == "lab.publish"),
        "rejected_conflict": len(conflicts), "rejected_conflict_early": sum(1 for x in conflicts if (x.get("t") or 0) <= EARLY),
        "rejected_regression": sum(1 for x in rej if x.get("reason") == "regression"),
        "unmerged_at_end": len(w.get("unmerged") or []),
        "holds": len(holds), "widest_hold": max((len(c["paths"]) + 20 * sum("*" in p for p in c["paths"]) for c in holds), default=0),
        "tasks": len(w.get("delegations") or []), "plans": len(w.get("plans") or []),
        "messages": sum(1 for e in acts if e.get("op") == "message" and e.get("ok")),
        "refused_calls": sum(1 for e in acts if not e.get("ok")),
        "lessons_left": len(res.get("lessons") or []), "rule_proposals": len(res.get("proposals") or []),
        "cost_usd": res.get("cost"), "dir": str(run),
    }


def _who(merges: list[dict]) -> dict[str, set]:
    out = defaultdict(set)
    for m in merges:
        for f in m.get("files") or []:
            out[f].add(m["agent"])
    return out


def timeline_lines(scores: list[int], w: dict, events: list[dict], ticks: int, unseen: str = "held-out") -> list[str]:
    """A run tick by tick, from what the world kept (merges, rejections, holds, tasks) and from its actions."""
    at = defaultdict(lambda: defaultdict(list))
    for m in w.get("merges") or []:
        at[m["t"]]["landed"].append(f"{m['agent']} {','.join(short(f) for f in m['files'])} (+{m.get('gains', 0)} logged, "
                                     f"+{m.get('verified', 0):g} unseen)")
    turned = defaultdict(lambda: defaultdict(list))
    for x in w.get("rejections") or []:
        turned[x["t"]][(x["agent"], x["reason"])].append(short(x["file"]))
    for t, groups in turned.items():
        for (agent, reason), files in groups.items():
            at[t]["turned away"].append(f"{agent} {','.join(files)} ({reason})")
    for c in w.get("claims") or []:
        what = f"{c.get('holder')} {c.get('op')}" + (f" {','.join(short(p) for p in c['paths'])}" if c.get("paths") else "") \
               + (f" [task {c['task']}]" if c.get("task") else "") + f" ({c.get('id')})"
        at[c["t"]]["holds"].append(what)
    for x in w.get("task_log") or []:
        at[x["t"]]["tasks"].append(f"{x.get('task')} {x.get('op')} by {x.get('by')}" + (f" to {x['to']}" if x.get("to") else "")
                                   + (f" scope {','.join(short(s) for s in x.get('scope') or [])}" if x.get("op") == "post" else ""))
    for p in w.get("proposals") or []:
        at[p["t"]]["waiting on a holder"].append(f"{p['by']} {','.join(short(f) for f in p.get('paths') or [])} ({p.get('id')}, {p.get('status')})")
    counts = defaultdict(Counter)
    for e in events:
        t = e.get("t") or 0
        if e.get("op") in ("intent", "closing.note") or str(e.get("op")).startswith("rules."):
            continue
        if not e.get("ok"):
            counts[t]["refused calls"] += 1
        if e.get("op") == "lab.publish" and e.get("ok"):
            counts[t]["offers"] += 1
        if e.get("op") == "message" and e.get("ok"):
            counts[t]["messages"] += 1
        if e.get("op") == "lab.plan" and e.get("ok"):
            counts[t]["plans " + str((e.get("args") or {}).get("op") or "?")] += 1
    lines = []
    for t in range(1, ticks + 1):
        gain = scores[t] - scores[t - 1] if t < len(scores) else 0
        head = f"round {t:>2} | {unseen} {scores[t] if t < len(scores) else '?':>3} ({gain:+d})"
        tail = "; ".join(f"{k} {v}" for k, v in sorted(counts[t].items()))
        lines.append(head + (f" | {tail}" if tail else ""))
        for label in ("landed", "turned away", "waiting on a holder", "holds", "tasks"):
            for item in at[t].get(label, []):
                lines.append(f"    {label}: {item}")
    left = w.get("unmerged") or []
    if left:
        why = Counter(str(u.get("why")) for u in left)
        lines += ["", f"left unmerged at the end: {len(left)} pieces ({', '.join(f'{n} {k}' for k, n in why.most_common())})"]
    return lines


def timeline(run: Path) -> str:
    """A run round by round. Each line can be opened up with `tools/record.py events RUN --round N`."""
    res = json.loads((run / "result.json").read_text())
    w = (json.loads((run / "run.json").read_text()).get("world") or {})
    scores = [round((s or 0) * HELD) for s in res.get("scores") or []]
    head = [f"# {run.name}: {change(json.loads((run / 'job.json').read_text()).get('recipe')) if (run / 'job.json').exists() else ''}",
            "", f"held-out checks by round: {scores}", f"hidden tests at the end: {res.get('hidden')}", ""]
    return "\n".join(head + timeline_lines(scores, w, _rows(run / "events.jsonl"), res.get("ticks") or 0)) + "\n"


def live_timeline(k) -> str:
    """The same view of a run that is still in memory: what an agent may look at after the last tick (the rules
    action's op=record). Counts of checks nobody can see are shown as the world tells them; nothing of the hidden tests."""
    scores = [round((s["metrics"].get("score") or 0) * HELD) for s in k.snapshots]
    w = k.env.export() if hasattr(k.env, "export") else {}
    return "\n".join(timeline_lines(scores, w, k.events, k.tick, unseen="unseen checks passing"))


def agent_text(w: dict, events: list[dict], aid: str) -> str:
    """What one agent offered and what became of it."""
    lines = [f"{aid}:"]
    for m in w.get("merges") or []:
        if m["agent"] == aid:
            lines.append(f"  round {m['t']}: landed {','.join(short(f) for f in m['files'])} (+{m.get('gains', 0)} logged, +{m.get('verified', 0):g} unseen)")
    for x in w.get("rejections") or []:
        if x["agent"] == aid:
            lines.append(f"  round {x['t']}: turned away {short(x['file'])} ({x['reason']})")
    for u in w.get("unmerged") or []:
        if u["agent"] == aid:
            lines.append(f"  round {u['t']}: kept unmerged {','.join(short(f) for f in u['files'])} ({u.get('why')})")
    for c in w.get("claims") or []:
        if c.get("holder") == aid or c.get("by") == aid:
            lines.append(f"  round {c['t']}: hold {c.get('op')} {','.join(short(p) for p in c.get('paths') or [])} ({c.get('id')})")
    body = sorted(lines[1:], key=lambda x: int(re.search(r"round (\d+)", x).group(1)))
    ops = Counter(str(e.get("op")) for e in events if e.get("agent") == aid and e.get("op") != "intent")
    return "\n".join([lines[0]] + (body or ["  nothing offered"])
                     + ["  actions: " + ", ".join(f"{op} {n}" for op, n in ops.most_common(12))])


def table(rows: list[dict]) -> str:
    cols = [("id", "run"), ("change", "rules (difference from the starting rules)"), ("hidden", "hidden"), ("held_out", "held-out"),
            ("first_landing_round", "first landing"), ("rounds_without_gain", "rounds w/o gain"), ("landings", "landings"),
            ("rejected_conflict", "conflicts"), ("rejected_conflict_early", f"conflicts by round {EARLY}"),
            ("rejected_regression", "regressions"), ("unmerged_at_end", "unmerged"), ("files_landed", "files"),
            ("files_landed_by_several", "files by several"), ("agents", "agents"), ("rounds", "rounds"), ("holds", "holds"), ("widest_hold", "widest hold (paths, +20 a wildcard)"),
            ("tasks", "tasks"), ("messages", "messages"), ("refused_calls", "refused calls"), ("cost_usd", "cost $")]
    out = ["| " + " | ".join(name for _, name in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            cells.append("-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def summary(rows: list[dict]) -> str:
    """Runs grouped by the rules they ran under: how many, and the mean and spread of both scores."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("full"):
            size = "" if (r.get("agents"), r.get("rounds")) == (10, 24) else f" ({r.get('agents')} agents, {r.get('rounds')} rounds)"
            groups[r["change"] + size].append(r)
    out = ["| rules | runs | hidden mean | hidden sd | held-out mean | held-out sd | each run (hidden/held-out) |", "|---|---|---|---|---|---|---|"]
    def rank(rs):  # by the judge where a run has it, else by the held-out checks
        hid = [x["hidden"] for x in rs if x.get("hidden") is not None]
        return (0, -statistics.mean(hid)) if hid else (1, -statistics.mean([x["held_out"] or 0 for x in rs]))

    for name, rs in sorted(groups.items(), key=lambda kv: rank(kv[1])):
        hid = [x["hidden"] for x in rs if x.get("hidden") is not None]
        held = [x["held_out"] for x in rs if x.get("held_out") is not None]
        f = lambda xs, fn: "-" if not xs or (fn is statistics.stdev and len(xs) < 2) else f"{fn(xs):.0f}"
        out.append(f"| {name} | {len(rs)} | {f(hid, statistics.mean)} | {f(hid, statistics.stdev)} | {f(held, statistics.mean)} | "
                   f"{f(held, statistics.stdev)} | {', '.join(str(x.get('hidden')) + '/' + str(x.get('held_out')) for x in rs)} |")
    return "\n".join(out)


def build(roots: list[Path], to: Path, log=print) -> list[dict]:
    """Copy every finished run under the experiment folders into `to`, and write the index and the timelines."""
    (to / "runs").mkdir(parents=True, exist_ok=True)
    rows = []
    for root in roots:
        for run in run_dirs(root):
            rid = run_id(root, run)
            dest = to / "runs" / rid
            dest.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                src = run / name
                if src.exists() and (not (dest / name).exists() or src.stat().st_mtime > (dest / name).stat().st_mtime):
                    shutil.copy2(src, dest / name)
            row = digest(dest, rid)
            row["dir"] = f"runs/{rid}"
            row["experiment"] = root.name
            (dest / "timeline.md").write_text(timeline(dest))
            rows.append(row)
    rows.sort(key=lambda r: (r["experiment"], r["id"]))
    (to / "index.json").write_text(json.dumps(rows, indent=1))
    full = [r for r in rows if r.get("full")]
    (to / "INDEX.md").write_text(
        "# Index of runs\n\nA summary for finding where to look. What happened is in each run's files.\n\n"
        f"Scores: hidden tests passed at the end (of {HIDDEN}; nobody in a run sees them or is told about them) and "
        f"held-out checks passed (of {HELD}; agents never see them, but an agent whose work lands is told how many of them "
        "newly pass). Hidden tests are the judge.\n\n## By rules (full runs)\n\n" + summary(rows)
        + "\n\n## Every full run\n\n" + table(full)
        + ("\n\n## Shorter runs\n\n" + table([r for r in rows if not r.get("full")]) if len(full) < len(rows) else "") + "\n")
    log(f"record: {len(rows)} runs ({len(full)} full) in {to}")
    return rows


def find(to: Path, name: str) -> Path:
    """A run's folder in the record, by its id or any unambiguous part of it."""
    hits = [p for p in sorted((to / "runs").iterdir()) if p.is_dir() and name in p.name]
    exact = [p for p in hits if p.name == name]
    if exact:
        return exact[0]
    if len(hits) != 1:
        raise SystemExit(f"'{name}' names {len(hits)} runs" + (": " + ", ".join(p.name for p in hits[:8]) if hits else ""))
    return hits[0]


def event_lines(rows: list[dict], round_: int | None = None, agent: str | None = None, op: str | None = None,
                grep: str | None = None, refused: bool = False, chars: int = 400) -> list[str]:
    """Actions that match, one line each, numbered by their place in the list (line N of events.jsonl)."""
    out = []
    for n, e in enumerate(rows, 1):
        if round_ is not None and e.get("t") != round_:
            continue
        if agent and e.get("agent") != agent:
            continue
        if op and op not in str(e.get("op")):
            continue
        if refused and e.get("ok"):
            continue
        text = json.dumps(e.get("args") or {}, ensure_ascii=False) + " -> " + str(e.get("msg"))
        if grep and grep.lower() not in text.lower():
            continue
        text = " ".join(text.split())
        out.append(f"{n:>5} r{e.get('t'):>2} {e.get('agent'):>3} {str(e.get('op')):16} {'ok ' if e.get('ok') else 'REFUSED'} "
                   + (text if len(text) <= chars else text[:chars] + "...[+%d chars]" % (len(text) - chars)))
    return out


def events(run: Path, round_: int | None = None, agent: str | None = None, op: str | None = None, grep: str | None = None,
           refused: bool = False, chars: int = 400) -> list[str]:
    """The run's actions that match, one line each, numbered as in events.jsonl (line N of the file)."""
    return event_lines(_rows(run / "events.jsonl"), round_, agent, op, grep, refused, chars)


def trace_lines(run: Path, round_: int | None = None, agent: str | None = None, kind: str | None = None,
                grep: str | None = None, chars: int = 4000) -> list[str]:
    """The run's full trace (trace.jsonl, runs from 2026-11-10 on): what an agent was shown at a turn (kind turn), what
    the model replied (reply), each action with its whole arguments and whole result (action), the system prompt
    (system) and the closing round (closing). Nothing is cut in the file; `chars` only bounds what is printed here."""
    out = []
    for n, e in enumerate(_rows(run / "trace.jsonl"), 1):
        if round_ is not None and e.get("t") != round_:
            continue
        if agent and e.get("agent") != agent:
            continue
        if kind and e.get("kind") != kind:
            continue
        body = {k: v for k, v in e.items() if k not in ("t", "agent", "kind")}
        text = json.dumps(body, ensure_ascii=False)
        if grep and grep.lower() not in text.lower():
            continue
        out.append(f"{n:>6} r{e.get('t'):>2} {str(e.get('agent')):>3} {str(e.get('kind')):7} "
                   + (text if len(text) <= chars else text[:chars] + "...[+%d chars]" % (len(text) - chars)))
    return out


MISSING = re.compile(r"(\d+) of (\d+) commands not implemented at HEAD")


def checks(run: Path) -> dict:
    """The counts the proposals of round 1 made predictions about, for one run (DESIGN.md section 22). Each is read
    from what the run kept; none is a judgement."""
    w = (json.loads((run / "run.json").read_text()).get("world") or {})
    events = _rows(run / "events.jsonl")
    acts = [e for e in events if e.get("op") not in ("intent", "closing.note") and not str(e.get("op")).startswith("rules.")]
    merges, rej = w.get("merges") or [], w.get("rejections") or []
    rounds = defaultdict(set)
    for x in rej:
        if x.get("reason") == "conflict":
            rounds[(x["agent"], x["file"])].add(x["t"])
    repeat_pairs = sum(1 for ts in rounds.values() if len(ts) > 1)
    repeat_refusals = sum(len(ts) - 1 for ts in rounds.values())
    broke = defaultdict(set)  # the same, for offers turned away because a check that passed at HEAD would stop passing
    for x in rej:
        if x.get("reason") == "regression":
            broke[(x["agent"], x["file"])].add(x["t"])
    ops = Counter(e.get("op") for e in acts)
    hit = {a for a, _ in rounds}
    unseen = Counter()
    for m in merges:
        unseen[m["agent"]] += m.get("verified") or 0
    agents = sorted({e.get("agent") for e in acts if e.get("agent")}, key=lambda a: int(a[1:]))
    mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
    missing = None
    for e in reversed(_rows(run / "trace.jsonl")):  # the last turn any agent was shown says what HEAD still lacks
        if e.get("kind") == "turn":
            m = MISSING.search(str(e.get("shown")))
            missing = int(m.group(1)) if m else (0 if "commands answer at HEAD" in str(e.get("shown")) else None)
            break
    return {
        "publish_calls": sum(1 for e in acts if e.get("op") == "lab.publish" and e.get("ok")),
        "landings": len(merges),
        "landings_no_gain": sum(1 for m in merges if not m.get("gains") and not m.get("verified")),
        "landed_by_the_world": sum(1 for m in merges if str(m.get("note", "")).startswith("offered at the end of the")),
        "conflict_refusals": sum(len(ts) for ts in rounds.values()),
        "repeat_conflict_pairs": repeat_pairs, "repeat_conflict_refusals": repeat_refusals,
        "unseen_per_agent_with_a_conflict": mean([unseen[a] for a in agents if a in hit]),
        "unseen_per_agent_without": mean([unseen[a] for a in agents if a not in hit]),
        "agents_with_a_conflict": len(hit),
        "commands_missing_at_end": missing,
        "wrong_action_refusals": sum(1 for e in acts if not e.get("ok") and "give a shell command" in str(e.get("msg"))),
        "moved_aside": sum(1 for u in w.get("unmerged") or [] if u.get("why") == "conflict with HEAD"),
        "regression_refusals": sum(len(ts) for ts in broke.values()),
        "repeat_regression_pairs": sum(1 for ts in broke.values() if len(ts) > 1),
        "files_landed": len({f for m in merges for f in m.get("files") or []}),
        "shell_calls": ops["lab.shell"],
        "messages": ops["message"] + ops["connect.message"],
        "claims": ops["lab.claim"],
        "looks_at_head": ops["lab.head"],
    }
