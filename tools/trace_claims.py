#!/usr/bin/env python3
"""Trace a conclusion back to the assumptions under it, with what each piece could be trusted at.

    python3 tools/trace_claims.py RUN_DIR            # summary: load-bearing weak pieces, weak guide lines, tests
    python3 tools/trace_claims.py RUN_DIR E1921      # the tree of what E1921 rests on

RUN_DIR is a run's output folder (the one with run.json and entries.jsonl, or a parent of it). Runs made with trust=1 carry
the engine's trust levels (said / seen in k states / reproduced / contested) and every test; older runs are traced from
their citations and the receipts they name, which is marked as approximate.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

E_ID, R_ID = re.compile(r"\bE\d+\b"), re.compile(r"\bR\d+\b")
RANK = {"contested": 0, "said": 1, "seen": 2, "reproduced": 3, "engine": 3}


def load(run: Path):
    if not (run / "run.json").exists():
        found = sorted(run.rglob("run.json"))
        if not found:
            sys.exit(f"no run.json under {run}")
        run = found[-1].parent
    data = json.loads((run / "run.json").read_text())
    entries = {e["id"]: e for e in map(json.loads, (run / "entries.jsonl").open())}
    return run, data["world"], entries


def build(world: dict, entries: dict):
    """Each piece's trust level and what it rests on: the engine's when the run had trust on, else from citations."""
    claims = {e["id"] for e in entries.values() if e["content"].startswith("PLAN ")}
    if world.get("trust"):
        trust = {x: dict(v) for x, v in world["trust"].items()}
        approx = False
    else:
        trust, approx = {}, True
        for x, e in entries.items():
            if e["author"] == "lab" or e["kind"] == "data":
                continue
            rs = sorted(set(R_ID.findall(e["content"])) | {r for r in e.get("refs") or [] if r.startswith("R")})
            trust[x] = {"level": "seen" if rs else "said", "states": None, "other": 0, "receipts": rs[:6]}
    deps = {}
    for x, e in entries.items():
        if x in trust and "rests_on" in trust[x]:
            deps[x] = trust[x]["rests_on"]
            continue
        ids = list(dict.fromkeys(list(e.get("refs") or []) + E_ID.findall(e["content"])))
        deps[x] = [y for y in ids if y.startswith("E") and y != x and y in entries and y not in claims]
    return trust, deps, approx


def level_of(trust: dict, entries: dict, x: str) -> str:
    if x in trust:
        return trust[x]["level"]
    e = entries.get(x)
    return "engine" if e and (e["author"] == "lab" or e["kind"] == "data") else "said"  # play records are engine output


def label(trust, entries, x) -> str:
    t, e = trust.get(x, {}), entries.get(x, {})
    lv = level_of(trust, entries, x)
    extra = f" in {t['states']} state{'s' if t['states'] != 1 else ''}" if lv == "seen" and t.get("states") else ""
    runs = f" {','.join(t['receipts'][:3])}" if t.get("receipts") else ""
    other = f"; failed in {t['other']} test(s) from other states" if t.get("other") else ""
    text = " ".join(str(e.get("content", "")).split())[:140]
    return f"{x} {e.get('author', '?')} t{e.get('tick', '?')} [{lv}{extra}{runs}{other}] {text}"


def tree(trust, deps, entries, x, depth=4, pre="", seen=None):
    seen = set() if seen is None else seen
    print(pre + label(trust, entries, x))
    if x in seen:
        if deps.get(x):
            print(pre + "    (shown above)")
        return
    seen.add(x)
    if depth == 0:
        return
    for y in deps.get(x, []):
        tree(trust, deps, entries, y, depth - 1, pre + "    ", seen)


def under(deps, x, depth=4) -> set:
    out, frontier = set(), [x]
    for _ in range(depth):
        nxt = []
        for y in frontier:
            for z in deps.get(y, []):
                if z not in out:
                    out.add(z)
                    nxt.append(z)
        frontier = nxt
    return out


def summary(world, trust, deps, entries, approx):
    pieces = [x for x in trust if entries.get(x, {}).get("author") != "lab"]
    print(("APPROXIMATE (this run had trust off: levels come from the receipts each entry names)\n" if approx else "")
          + f"{len(pieces)} pieces: " + ", ".join(f"{k} {v}" for k, v in Counter(level_of(trust, entries, x)
                                                                               for x in pieces).most_common()))
    resting = defaultdict(set)  # weak piece -> the pieces that rest on it within four steps
    for x in pieces:
        for y in under(deps, x):
            if level_of(trust, entries, y) in ("said", "contested") or (trust.get(y, {}).get("states") == 1):
                resting[y].add(x)
    print("\nLOAD-BEARING WEAK PIECES (said, contested or seen in one state, by how many pieces rest on them):")
    for y, xs in sorted(resting.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {len(xs):3d} rest on {label(trust, entries, y)}")
    guide = [x for x, e in entries.items() if e["content"].startswith("GUIDE ") and " added: " in e["content"]]
    weak = []
    for x in guide:
        lows = [y for y in under(deps, x, 2) if RANK.get(level_of(trust, entries, y), 1) < 2]
        if level_of(trust, entries, x) in ("said", "contested") or lows:
            weak.append((x, lows))
    print(f"\nGUIDE LINES THAT ARE UNBACKED, CONTESTED OR REST ON SUCH PIECES ({len(weak)} of {len(guide)}):")
    for x, lows in weak[-15:]:
        print(f"  {label(trust, entries, x)}" + (f"\n      rests on: {', '.join(sorted(lows)[:6])}" if lows else ""))
    tests = world.get("tests") or {}
    if tests:
        print(f"\nTESTS ({sum(map(len, tests.values()))}):")
        for x, ts in tests.items():
            for t in ts:
                where = "engine rerun" if t["how"] == "replay" else "same starting state" if t["same_state"] else "another state"
                print(f"  {x} t{t['t']} by {t['by']}: {t['result']} ({where}) {str(t.get('why', ''))[:100]}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run, world, entries = load(Path(sys.argv[1]))
    trust, deps, approx = build(world, entries)
    if len(sys.argv) > 2:
        x = sys.argv[2].strip().upper()
        if x not in entries:
            sys.exit(f"no entry {x} in {run}")
        if approx:
            print("APPROXIMATE (this run had trust off: levels come from the receipts each entry names)")
        tree(trust, deps, entries, x)
    else:
        summary(world, trust, deps, entries, approx)


if __name__ == "__main__":
    main()
