"""The recipe archive: every recipe with its parent, its runs and its credit (DESIGN.md section 20).

On disk, under one directory:
  recipes/<id>.json   one record per recipe: the recipe, its status, its niche and why it stands where it does
  runs.jsonl          every evaluation run: recipe, niche, purpose, seed, final score, progress score, cost
  pairs.jsonl         every paired comparison: parent, child, purpose, seed, and the two scores
  decisions.jsonl     every decision of the selection ladder, in order: what was tried, where it stopped and why
  lessons.jsonl       the lesson library: what runs left for later societies, and how each lesson has fared
  proposals.jsonl     every rule proposal agents made, in whichever run they made it, by the recipe they made it under
  archive.json        the niche and frame this archive belongs to
  counters.json       how many instances each stage has drawn, so a later session draws new ones

Records are written whole and atomically (a temporary file, then a rename); the logs only grow. Nothing is deleted: a
dropped revision stays on record as a dead end that later proposals are checked against, and it never becomes a parent.
One process writes an archive at a time.

A niche names where a recipe was tried: the world, its dependency, the arm and the team size. An archive belongs to
one niche and one frame (bind): scores from another team size, run length, task size or model are refused, because
they do not compare. Other niches get archives of their own.

Statuses: seed and kept recipes may seed a generation; a dropped one was judged and lost; a candidate is not judged
yet (its test was cut short, or another revision went on in its place), and can be taken up again.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from statistics import median

from .recipe import Recipe
from .stats import summary

# runs from scratch on selection instances: what a recipe's standing rests on. Report runs are kept apart, so that
# what a result is reported on never moves a selection; a run that tried changes on itself (trial_run) is not a
# plain run of its recipe
FRESH = ("generation", "confirm", "retest")
PARENTS = ("seed", "kept", "lead")  # statuses that may seed a generation. A lead was taken on after single runs: not confirmed
WORDS = re.compile(r"[a-z0-9_]+")


def niche_of(cfg: dict) -> str:
    wo = cfg.get("world_opts") or {}
    dep = wo.get("dependency", "coupled") if cfg.get("world") == "design" else "-"
    return f"{cfg.get('world', 'hidden')}/{dep}/{cfg.get('frame', 'open')}/n{cfg['agents']}"


def _write(path: Path, data):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(data, indent=1, default=list))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=list) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ArchiveMismatch(RuntimeError):
    """An archive was opened for a niche or frame other than the one it belongs to."""


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "recipes").mkdir(parents=True, exist_ok=True)
        self._counters = json.loads((self.root / "counters.json").read_text()) if (self.root / "counters.json").exists() else {}
        self._proposals = _read(self.root / "proposals.jsonl")
        self.records: dict[str, dict] = {}
        for p in sorted((self.root / "recipes").glob("*.json")):
            rec = json.loads(p.read_text())
            self.records[rec["id"]] = rec
        self._runs = _read(self.root / "runs.jsonl")
        self._pairs = _read(self.root / "pairs.jsonl")
        self._decisions = _read(self.root / "decisions.jsonl")
        self._lessons: dict[str, dict] = {}
        for row in _read(self.root / "lessons.jsonl"):
            self._lesson_event(row)

    def bind(self, niche: str, frame: dict):
        """Tie the archive to one niche and frame, or check that it is tied to this one."""
        path = self.root / "archive.json"
        if not path.exists():
            _write(path, {"niche": niche, "frame": frame})
            return
        have = json.loads(path.read_text())
        diff = [f"{key}: {have['frame'].get(key)!r} here, {frame.get(key)!r} asked" for key in sorted(set(have["frame"]) | set(frame))
                if have["frame"].get(key) != frame.get(key)]
        if have["niche"] != niche or diff:
            raise ArchiveMismatch(f"{self.root} belongs to niche {have['niche']}" + (f" ({'; '.join(diff)})" if diff else "")
                                  + f", not to {niche}: scores would not compare. Use another output folder.")

    def next(self, name: str) -> int:
        """The next unused index of a stage's instances (kept on disk: a later session never redraws an instance)."""
        self._counters[name] = self._counters.get(name, 0) + 1
        _write(self.root / "counters.json", self._counters)
        return self._counters[name]

    # ------------------------------------------------------------------ recipes

    def add(self, recipe: Recipe, status: str = "candidate", niche: str = "", why: str = "") -> dict:
        """File a recipe. One that is already on file keeps its record (and its status): see seen()."""
        rid = recipe.id
        if rid in self.records:
            return self.records[rid]
        rec = {"id": rid, "recipe": recipe.to_json(), "status": status, "niche": niche, "why": why, "stage": None,
               "added": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.records[rid] = rec
        _write(self.root / "recipes" / f"{rid}.json", rec)
        return rec

    def seen(self, recipe: Recipe) -> dict | None:
        """The record of a recipe with this same content, if it was tried before (a dead end is not tried twice)."""
        return self.records.get(recipe.id)

    def get(self, rid: str) -> Recipe:
        return Recipe.from_json(self.records[rid]["recipe"])

    def set_status(self, rid: str, status: str, why: str = "", stage: str | None = None):
        rec = self.records[rid]
        rec.update(status=status, why=why, stage=stage)
        _write(self.root / "recipes" / f"{rid}.json", rec)

    def recipes(self, status=None, niche: str | None = None) -> list[dict]:
        want = (status,) if isinstance(status, str) else status
        return [r for r in self.records.values() if (want is None or r["status"] in want)
                and (niche is None or r["niche"] == niche)]

    def children(self, rid: str, status=PARENTS) -> list[str]:
        return [r["id"] for r in self.records.values() if r["recipe"].get("parent") == rid and r["status"] in status]

    def lineage(self, rid: str) -> list[str]:
        """A recipe's ancestors, oldest first, ending with itself."""
        out = []
        while rid and rid in self.records and rid not in out:
            out.append(rid)
            rid = self.records[rid]["recipe"].get("parent")
        return out[::-1]

    # ------------------------------------------------------------------ runs, pairs, decisions

    def record_run(self, rid: str, niche: str, purpose: str, seed: int, score, progress=None, cost=None, **extra) -> dict:
        row = {"recipe": rid, "niche": niche, "purpose": purpose, "seed": seed, "score": score, "progress": progress,
               "cost": cost, **extra}
        self._runs.append(row)
        _append(self.root / "runs.jsonl", row)
        return row

    def record_pair(self, parent: str, child: str, niche: str, purpose: str, seed: int, a: float, b: float,
                    metric: str = "score", **extra) -> dict:
        row = {"parent": parent, "child": child, "niche": niche, "purpose": purpose, "seed": seed, "a": a, "b": b,
               "gain": None if a is None or b is None else round(b - a, 5), "metric": metric, **extra}
        self._pairs.append(row)
        _append(self.root / "pairs.jsonl", row)
        return row

    def decide(self, **row) -> dict:
        row = {"n": len(self._decisions) + 1, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        self._decisions.append(row)
        _append(self.root / "decisions.jsonl", row)
        return row

    def runs(self, rid: str | None = None, niche: str | None = None, purpose=None) -> list[dict]:
        want = (purpose,) if isinstance(purpose, str) else purpose
        return [r for r in self._runs if (rid is None or r["recipe"] == rid) and (niche is None or r["niche"] == niche)
                and (want is None or r["purpose"] in want)]

    def pairs(self, child: str | None = None, purpose=None) -> list[dict]:
        want = (purpose,) if isinstance(purpose, str) else purpose
        return [p for p in self._pairs if (child is None or p["child"] == child) and (want is None or p["purpose"] in want)]

    def decisions(self) -> list[dict]:
        return list(self._decisions)

    def spent(self) -> float:
        """Dollars the archive's runs have cost so far (runs that reported a cost)."""
        return round(sum(r.get("cost") or 0 for r in self._runs), 4)

    def value(self, rid: str, niche: str, metric: str = "score", conf: float = 0.9) -> dict:
        """What a recipe has scored in a niche, over its runs from scratch."""
        return summary([r.get(metric) for r in self.runs(rid, niche, FRESH)], conf)

    def gain(self, child: str, purpose=("confirm", "retest"), conf: float = 0.9) -> dict:
        """A revision's credit: its paired gain over its parent."""
        return summary([p["gain"] for p in self.pairs(child, purpose)], conf)

    # ------------------------------------------------------------------ parents and leaders

    def _standing(self, rid: str, niche: str) -> float | None:
        """The lower bound of what a recipe scored. A recipe with fewer than two runs has no standing yet (it is
        treated as middling), so one lucky run does not make a leader."""
        return self.value(rid, niche)["lcb"]

    def sample_parent(self, niche: str, rng: random.Random, explore: float = 0.2, lam: float = 2.0) -> Recipe:
        """Choose who seeds the next generation. Mostly by standing (the lower bound of what a recipe has scored, so
        one lucky run does not make a leader), less for a recipe that already has children, and sometimes at random."""
        pool = self.recipes(PARENTS, niche)
        if not pool:
            raise LookupError(f"no recipe to start from in niche {niche}")
        if len(pool) == 1 or rng.random() < explore:
            return Recipe.from_json(rng.choice(pool)["recipe"])
        stand = {r["id"]: self._standing(r["id"], niche) for r in pool}
        known = [v for v in stand.values() if v is not None]
        mid = median(known) if known else 0.0
        scale = max(median([abs(v - mid) for v in known]) if known else 0.0, 1e-6)
        weights = []
        for r in pool:
            v = stand[r["id"]]
            s = 0.5 if v is None else 1 / (1 + math.exp(-max(-30.0, min(30.0, lam * (v - mid) / scale))))
            weights.append(s / (1 + len(self.children(r["id"]))))
        return Recipe.from_json(rng.choices(pool, weights=weights)[0]["recipe"])

    def leaders(self, niche: str, k: int = 3) -> list[dict]:
        """The recipes that stand highest in a niche: by the lower bound of what they scored. A recipe with fewer than
        two runs has no bound and comes last, however high its one run."""
        rows = [dict(id=r["id"], status=r["status"], **self.value(r["id"], niche)) for r in self.recipes(PARENTS, niche)]
        return sorted(rows, key=lambda r: (r["lcb"] is None, -(r["lcb"] or 0.0), r["id"]))[:k]

    def needs_retest(self, rid: str, niche: str, min_runs: int) -> bool:
        """A leader is re-run before it seeds a generation: the best of several noisy scores is biased upward."""
        return len(self.runs(rid, niche, FRESH)) < min_runs

    # ------------------------------------------------------------------ what agents said

    def record_voices(self, r, niche: str, after: int = 0):
        """File what a run's agents proposed and the lessons they left, under the recipe they ran under. `after`: for
        a copy that switched rules mid-run, only what was said after the switch belongs to its recipe."""
        for p in r.proposals:
            if p.get("t", 0) > after:
                row = {"recipe": r.recipe, "niche": niche, "key": p["key"], "value": p["value"], "why": p.get("why", ""),
                       "evidence": p.get("evidence", []), "backers": len(p.get("backers", [])), "run": r.dir, "seed": r.seed}
                self._proposals.append(row)
                _append(self.root / "proposals.jsonl", row)
        for les in r.lessons:
            if les.get("t", 0) > after:
                self.add_lesson(les, niche, source=r.dir)
        for x in getattr(r, "suggestions", None) or []:  # rules that are not settings yet: read by whoever maintains the rules
            _append(self.root / "suggestions.jsonl", {
                "recipe": r.recipe, "niche": niche, "id": x.get("id"), "by": x.get("by"), "text": x.get("text"),
                "when": x.get("when", ""), "evidence": x.get("evidence", []), "backers": len(x.get("backers", [])),
                "run": r.dir, "seed": r.seed})

    def proposals(self, rid: str) -> list[dict]:
        """What agents proposed under a recipe, over all of its runs: one row a (setting, value), with the runs it came
        up in and the backers it had in all, most widely proposed first."""
        out: dict[tuple, dict] = {}
        for p in self._proposals:
            if p["recipe"] != rid:
                continue
            x = out.setdefault((p["key"], json.dumps(p["value"])), {"key": p["key"], "value": p["value"], "runs": 0, "backers": 0,
                                                                     "why": p["why"], "evidence": [], "sources": []})
            src = [p["run"], p["seed"]]
            if src not in x["sources"]:  # the same change proposed twice in one run is one run
                x["sources"].append(src)
                x["runs"] += 1
            x["backers"] += p["backers"]
            x["evidence"] = (x["evidence"] + [e for e in p["evidence"] if e not in x["evidence"]])[:12]
        return sorted(out.values(), key=lambda x: (-x["runs"], -x["backers"], x["key"]))

    def set_advice_status(self, rid: str, lesson_id: str, status: str):
        """Mark a lesson inside a recipe's record (its status is not part of the recipe's name, so the id stays)."""
        rec = self.records[rid]
        for x in rec["recipe"].get("advice", []):
            if x["id"] == lesson_id:
                x["status"] = status
        _write(self.root / "recipes" / f"{rid}.json", rec)

    # ------------------------------------------------------------------ lessons

    def _lesson_event(self, row: dict):
        if row.get("op") == "mark":
            if row["id"] in self._lessons:
                self._lessons[row["id"]].update(status=row["status"], why=row.get("why", ""))
            return
        x = self._lessons.get(row["id"])
        if x is None:
            self._lessons[row["id"]] = {"id": row["id"], "text": row["text"], "when": row.get("when", ""),
                                        "status": row.get("status", "untested"), "count": 1, "niches": [row.get("niche", "")],
                                        "sources": [row.get("source", "")], "evidence": list(row.get("evidence") or []), "why": ""}
        else:  # the same lesson left again: it counts, and its evidence adds up
            x["count"] += 1
            for key, v in (("niches", row.get("niche", "")), ("sources", row.get("source", ""))):
                if v and v not in x[key]:
                    x[key].append(v)

    def add_lesson(self, les: dict, niche: str = "", source: str = "") -> dict:
        row = {"op": "add", "id": les["id"], "text": les["text"], "when": les.get("when", ""),
               "status": les.get("status", "untested"), "evidence": les.get("evidence", []), "niche": niche, "source": source}
        self._lesson_event(row)
        _append(self.root / "lessons.jsonl", row)
        return self._lessons[les["id"]]

    def mark_lesson(self, lid: str, status: str, why: str = ""):
        row = {"op": "mark", "id": lid, "status": status, "why": why}
        self._lesson_event(row)
        _append(self.root / "lessons.jsonl", row)

    def lessons(self, niche: str | None = None, query: str = "", k: int | None = None) -> list[dict]:
        """The library, most useful first: relevance to the query, then standing (confirmed, untested, refuted),
        then how often the lesson was left. With k=None every lesson is returned: nothing is cut, only ordered."""
        want = set(WORDS.findall(query.lower()))
        rank = {"confirmed": 2, "untested": 1, "refuted": 0}

        def key(x):
            words = set(WORDS.findall((x["text"] + " " + x["when"]).lower()))
            here = 1 if niche and niche in x["niches"] else 0
            return (-len(want & words), -rank.get(x["status"], 1), -here, -x["count"], x["id"])
        out = sorted(self._lessons.values(), key=key)
        return out if k is None else out[:k]

    # ------------------------------------------------------------------ reading it

    def table(self, niche: str | None = None) -> str:
        """The archive as a small table: each recipe, its parent, where it stands and what it scored."""
        lines = ["| recipe | parent | gen | status | edits | runs | mean | gain over parent | why |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(self.recipes(niche=niche), key=lambda r: (r["recipe"]["generation"], r["added"], r["id"])):
            rec = r["recipe"]
            edits = "; ".join(f"{e['key']}={e['value']}" if e["op"] == "set" else f"{e['op']} {e.get('id') or e.get('name')}"
                              for e in rec.get("origin", {}).get("edits", [])) or "-"
            v = self.value(r["id"], r["niche"])
            g = self.gain(r["id"])
            gain = "-" if not g["n"] else f"{g['mean']:+.3f} (n={g['n']}" + (f", lower {g['lcb']:+.3f})" if g["lcb"] is not None else ")")
            lines.append(f"| {r['id']} | {rec.get('parent') or '-'} | {rec['generation']} | {r['status']} | {edits} | {v['n']} | "
                         f"{'-' if v['mean'] is None else format(v['mean'], '.3f')} | {gain} | {r.get('why') or ''} |")
        return "\n".join(lines)
