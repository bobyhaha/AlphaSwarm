"""How diverse a run's search was, level by level, next to how far it got.

    python3 tools/diversity.py RUN [RUN ...]      (a run folder, or a folder holding one run)

For each level the copies worked on:
- ideas: claims (plans) on the level, grouped into distinct ideas by word overlap (explore.py's idea_words and
  same_idea, at IDEA_SAME = 0.33: on E_vc33_20_s0 level 5, claims stating the same door rule scored 0.36-0.40 and
  claims about different mechanisms at most 0.27; the world's own SAME_IDEA of 0.5 merges almost no long claims), and how
  many distinct ideas were tested (a claim with an interim or final finding), and how many claims never were. Word
  overlap is a crude proxy for meaning;
- similarity: the mean word overlap of pairs of claims by different agents (lower is more diverse), and, when the run
  had groups, of pairs from different islands and from the same island (membership rebuilt from the connect events);
- attention: the share of citations of others' posts that went to the three most-cited authors;
- states: how many distinct move histories (since the level's start) the plays on the level started from;
- directions (directions=1): how many were opened and how many had a claim tested; the most agents on one direction at
  once, as a share of the agents; the largest direction's share of the agent-ticks spent on directions; the share of
  claims filed under a direction; claims combining two or more directions; the directions paid when the level fell;
- progress: the tick the level first fell, and how many copies got past it.
Run-wide: the best and the mean copy's final level, groups formed, entries published, cost.
"""
from __future__ import annotations

import collections
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.explore import idea_words, same_idea  # noqa: E402

IDEA_SAME = 0.33


def run_dir(p: str) -> Path:
    p = Path(p)
    if (p / "run.json").exists():
        return p
    subs = sorted(x for x in p.iterdir() if (x / "run.json").exists())
    if not subs:
        raise SystemExit(f"no run.json under {p}")
    return subs[-1]


def clusters(plans: list[dict]) -> list[list[dict]]:
    """Greedy: each claim joins the first idea it is the same as (by the world's rule), else starts a new one."""
    out: list[tuple[frozenset, list]] = []
    for p in sorted(plans, key=lambda p: p["t"]):
        w = idea_words(p["hypothesis"])
        hit = next((c for c in out if same_idea(w, c[0]) >= IDEA_SAME), None)
        if hit:
            hit[1].append(p)
        else:
            out.append((w, [p]))
    return [c[1] for c in out]


def mean_sim(pairs) -> float | None:
    sims = [same_idea(idea_words(a["hypothesis"]), idea_words(b["hypothesis"])) for a, b in pairs]
    return round(sum(sims) / len(sims), 3) if sims else None


def memberships(events: list[dict]) -> list[tuple[int, str, str, str]]:
    """(tick, op, group, agent) from the connect events, in order."""
    out = []
    for e in events:
        op = e.get("op", "")
        if not (op.startswith("connect.") or op == "lab.lead") or not e.get("ok"):
            continue
        msg = e.get("msg", "")
        if op == "connect.form_group":
            m = re.match(r"formed (G\d+) .* with member (A\d+)", msg)
            if m:
                out.append((e["t"], "in", m.group(1), m.group(2)))
        elif op in ("connect.join", "connect.admit"):
            m = re.search(r"(A\d+) joined (G\d+)", msg) or re.search(r"admitted (A\d+) to (G\d+)", msg)
            if m:
                out.append((e["t"], "in", m.group(2), m.group(1)))
        elif op == "lab.lead":  # teams (delegation=1) are groups too
            m = re.match(r"you lead team (G\d+)", msg) or re.match(r"you joined (G\d+)", msg)
            if m:
                out.append((e["t"], "in", m.group(1), e["agent"]))
            m = re.match(r"you left (G\d+)", msg)
            if m:
                out.append((e["t"], "out", m.group(1), e["agent"]))
        elif op in ("connect.leave", "connect.remove"):
            m = re.search(r"(A\d+) left (G\d+)", msg) or re.search(r"removed (A\d+) from (G\d+)", msg)
            if m:
                out.append((e["t"], "out", m.group(2), m.group(1)))
    return out


def islands_at(changes, agents, t: int) -> dict[str, frozenset]:
    """Each agent's island at tick t: itself and everyone sharing a group with it (nesting is not rebuilt here)."""
    members = collections.defaultdict(set)
    for tt, op, g, a in changes:
        if tt > t:
            break
        (members[g].add if op == "in" else members[g].discard)(a)
    isl = {a: {a} for a in agents}
    for g, ms in members.items():
        for a in ms:
            isl[a] |= ms
    return {a: frozenset(s) for a, s in isl.items()}


def analyse(path: str) -> dict:
    d = run_dir(path)
    r = json.loads((d / "run.json").read_text())
    w = r["world"]
    events = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]
    entries = {}
    if (d / "entries.jsonl").exists():
        for x in (d / "entries.jsonl").read_text().splitlines():
            if x.strip():
                e = json.loads(x)
                entries[e["id"]] = e
    agents = sorted({e["agent"] for e in events if re.fullmatch(r"A\d+", str(e.get("agent")))}, key=lambda a: int(a[1:]))
    plans = w.get("plans", [])
    levels = sorted({p["level"] for p in plans})
    changes = memberships(events)
    out = {"run": str(d), "cost": r["meta"].get("cost_usd"), "ticks": max((e["t"] for e in events), default=0),
           "levels": {}}
    copies = {k.split("|")[0]: v["levels"] for k, v in w.get("copies", {}).items()}
    start = min(copies.values()) if copies else 0
    out["best_final"] = max(copies.values()) if copies else None
    out["mean_final"] = round(sum(copies.values()) / len(copies), 2) if copies else None
    out["first_falls"] = {f["level"]: (f["t"], f["agent"]) for f in w.get("firsts", [])}
    out["groups_formed"] = len({g for _, op, g, _ in changes})
    out["published"] = len((w.get("islands") or {}).get("published", {}))
    # citations of others' posts, and the top three authors' share
    cites = collections.Counter()
    for e in events:
        a = e.get("args") or {}
        ids = set()
        for key in ("builds_on", "cites", "cite", "evidence"):
            ids |= set(re.findall(r"E\d+", str(a.get(key, ""))))
        for i in ids:
            au = entries.get(i, {}).get("author")
            if au and au != e["agent"] and re.fullmatch(r"A\d+", au):
                cites[(levels_of(entries, w, i), au)] += 1
    receipts = w.get("receipts_full", [])
    for lv in levels:
        here = [p for p in plans if p["level"] == lv]
        cs = clusters(here)
        tested = [c for c in cs if any(p.get("finding") or p["status"] == "done" for p in c)]
        pairs = list(itertools.combinations(here, 2))
        row = {"claims": len(here), "ideas": len(cs), "ideas_tested": len(tested),
               "claims_never_tested": sum(not (p.get("finding") or p["status"] == "done") for p in here),
               "agents_claiming": len({p["agent"] for p in here}),
               "sim_across_agents": mean_sim([(a, b) for a, b in pairs if a["agent"] != b["agent"]])}
        if changes:  # each claim's island at the tick it was made
            isl = {p["id"]: islands_at(changes, agents, p["t"]).get(p["agent"], frozenset({p["agent"]})) for p in here}
            if any(len(s) > 1 for s in isl.values()):
                row["sim_across_islands"] = mean_sim([(a, b) for a, b in pairs if a["agent"] not in isl[b["id"]]])
                row["sim_within_islands"] = mean_sim([(a, b) for a, b in pairs if a["agent"] != b["agent"]
                                                      and a["agent"] in isl[b["id"]]])
        c_here = collections.Counter({au: v for (l2, au), v in cites.items() if l2 == lv})
        tot = sum(c_here.values())
        row["top3_cite_share"] = round(sum(v for _, v in c_here.most_common(3)) / tot, 2) if tot else None
        rs = [x for x in receipts if x.get("level_start") == lv]
        row["plays"] = len(rs) if receipts else None
        row["distinct_start_histories"] = len({tuple(x.get("start") or []) for x in rs}) if receipts else None
        if w.get("directions") is not None:
            row.update(direction_measures(w, here, lv, len(agents), out["ticks"]))
        row["fell_at"] = out["first_falls"].get(lv)
        row["copies_past"] = sum(v >= lv for v in copies.values())
        out["levels"][lv] = row
    out["start_level"] = (w["log"][0]["levels"] + 1) if w.get("log") else start + 1
    return out


def direction_measures(w: dict, plans: list[dict], lv: int, n_agents: int, last_tick: int) -> dict:
    """directions=1, one level: spread of agents over directions (membership rebuilt from the direction log) and use."""
    ds = [d for d in w["directions"] if d["level"] == lv]
    ids = {d["id"] for d in ds}
    on, peak = collections.Counter(), 0
    for x in sorted((x for x in w.get("direction_log", []) if x["direction"] in ids), key=lambda x: x["t"]):
        on[x["direction"]] += -1 if x["op"] == "leave" else 1
        peak = max(peak, on[x["direction"]])
    ticks = {d["id"]: sum(d["tenure"].values()) + sum(last_tick - since + 1 for since in d["members"].values())
             for d in ds}  # members still on it at the end count up to the last tick
    status = {p["id"]: p for p in w.get("plans", [])}
    tested = [d for d in ds if any(status.get(p, {}).get("finding") or status.get(p, {}).get("status") == "done"
                                   for p in d.get("plans", []))]
    return {"directions": len(ds), "directions_tested": len(tested),
            "direction_peak_share": round(peak / n_agents, 2) if n_agents and ds else None,
            "top_direction_tick_share": round(max(ticks.values()) / sum(ticks.values()), 2) if sum(ticks.values()) else None,
            "claims_on_directions": round(sum(bool(p.get("direction")) for p in plans) / len(plans), 2) if plans else None,
            "claims_combining": sum(bool(p.get("combines")) for p in plans),
            "directions_paid": [d["id"] for d in ds if d.get("paid")]}


def levels_of(entries: dict, w: dict, eid: str) -> int | None:
    """The level an entry was about, from the world's records where it has one (findings, plans), else unknown."""
    for f in w.get("findings", []):
        if f["id"] == eid:
            return f["level"]
    for p in w.get("plans", []):
        if p.get("entry") == eid:
            return p["level"]
    return None


def main(argv):
    rows = [analyse(p) for p in argv]
    for x in rows:
        print(f"\n{x['run']}\n  cost ${x['cost']}, ticks {x['ticks']}, first level played {x['start_level']}, levels "
              f"completed: best copy {x['best_final']}, mean copy {x['mean_final']}; groups formed {x['groups_formed']}, "
              f"published {x['published']}, first falls {x['first_falls']}")
        for lv, row in sorted(x["levels"].items()):
            print(f"  level {lv}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
