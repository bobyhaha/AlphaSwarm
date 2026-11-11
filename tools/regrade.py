"""Grade saved programs again on the hidden tests. Counts only.

  .venv/bin/python tools/regrade.py --names regrade-check-old-1110        # folders under the grades directory
  .venv/bin/python tools/regrade.py --out runs_recipes/XSV1 --write       # every run of an experiment that kept a program

A full run packages its final program under the grades directory before it grades (result.json facts.graded_as names
the folder). This runs the grader's script on that package again and, with --write, puts the counts into the run's
result.json (hidden, facts.grade). Nothing in the grades folder is read here: only the grader's own summary (status,
passed, total) comes back, as in alphateam/code_world.py.

Why it exists: on 2026-11-08 the benchmark's test data was lost from this machine and every run of 2026-11-10 ended
with "scoring failed". The data was restored on 2026-11-10; the programs those runs wrote were kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphateam.code_world import GRADER, GRADES, HARNESS, score_native  # noqa: E402


def targets(out: Path | None, names: list[str]) -> list[tuple[str, Path | None]]:
    """(grades folder, run folder or None), in a stable order, each folder once."""
    rows = [(n, None) for n in names]
    if out is not None:
        for res in sorted(list((out / "runs").glob("*/result.json")) + list((out / "runs").glob("*/*/result.json"))
                          + list(out.glob("smoke/runs/*/result.json"))):
            facts = (json.loads(res.read_text()).get("facts") or {})
            if facts.get("graded_as"):
                rows.append((facts["graded_as"], res.parent))
    seen, uniq = set(), []
    for name, run in rows:
        if name not in seen:
            seen.add(name)
            uniq.append((name, run))
    return uniq


def grade(name: str) -> dict:
    submission = GRADES / name / "submission"
    if not submission.is_dir():
        return {"status": "no saved program"}
    tids = [p.name for p in submission.iterdir() if p.is_dir()]
    if len(tids) != 1:
        return {"status": f"expected one task folder, found {len(tids)}"}
    native = GRADES / name / "native"
    if native.exists():  # what an earlier, failed attempt left: set aside, never deleted
        native.rename(GRADES / name / f"native.before-{int(time.time())}")
    began = time.time()
    got = score_native(GRADER, Path(HARNESS), submission, tids[0])
    return {**got, "seconds": round(time.time() - began)}


def write(run: Path, got: dict):
    path = run / "result.json"
    res = json.loads(path.read_text())
    ok = got.get("passed") is not None and got.get("total")
    res["hidden"] = got.get("passed") if ok else None
    res.setdefault("facts", {})["grade"] = {**{k: got.get(k) for k in ("status", "passed", "total", "resolved")},
                                             "regraded": time.strftime("%Y-%m-%d %H:%M")}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, help="an experiment folder: every run in it that kept a program")
    p.add_argument("--names", nargs="*", default=[], help="folders under the grades directory")
    p.add_argument("--write", action="store_true", help="put the counts into each run's result.json")
    p.add_argument("--jobs", type=int, default=1, help="programs graded side by side")
    p.add_argument("--only-missing", action="store_true", help="skip runs whose result.json already has a hidden count")
    a = p.parse_args()
    todo = targets(a.out, a.names)
    if a.only_missing:
        todo = [(n, r) for n, r in todo if r is None or json.loads((r / "result.json").read_text()).get("hidden") is None]
    print(f"{len(todo)} programs to grade, {a.jobs} at a time", flush=True)

    def one(row):
        name, run = row
        got = grade(name)
        if a.write and run is not None:
            write(run, got)
        held = None
        if run is not None:
            held = (json.loads((run / "result.json").read_text()).get("facts") or {}).get("held_out")
        print(f"{name}: {got.get('status')}, hidden {got.get('passed')} of {got.get('total')}"
              + (f", held-out {held}" if held is not None else "") + f", {got.get('seconds', 0)} s", flush=True)
        return got

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        done = list(pool.map(one, todo))
    bad = sum(1 for g in done if g.get("passed") is None)
    print(f"graded {len(done) - bad} of {len(done)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
