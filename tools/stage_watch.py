#!/usr/bin/env python3
"""Wait for news from a queued paid stage, then say what happened (for whoever watches an experiment).

  .venv/bin/python tools/stage_watch.py --out runs_recipes/XSV1 --job XSV1_g1 --seen 0 --minutes 60

Exits when the stage's job has ended, when more runs than --seen have left a result (and none followed for --settle
seconds), when the runner that holds the key is gone, or after --minutes. It prints every run's result so far and the
end of the stage's log. It reads result.json, job.json and log files only: never a run's work folder.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs_recipes" / "queue"
HELD = 179


def results(out: Path) -> list[dict]:
    rows = []
    for path in sorted((out / "runs").glob("*/**/result.json"), key=lambda p: p.stat().st_mtime):
        if "dry" in path.relative_to(out).parts or "work" in path.relative_to(out).parts:
            continue
        try:
            d = json.loads(path.read_text())
        except ValueError:
            continue
        job = path.with_name("job.json")
        d["_minutes"] = (path.stat().st_mtime - job.stat().st_mtime) / 60 if job.exists() else None
        d["_where"] = str(path.parent.relative_to(out))
        rows.append(d)
    return rows


def running(out: Path) -> list[str]:
    return [str(p.parent.relative_to(out)) for p in (out / "runs").glob("*/**/job.json")
            if "work" not in p.relative_to(out).parts and "dry" not in p.relative_to(out).parts
            and not p.with_name("result.json").exists() and not p.with_name("error.json").exists()]


def runner_alive() -> bool:
    try:
        os.kill(int((QUEUE / "runner.pid").read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def line(d: dict) -> str:
    f = d.get("facts") or {}
    took = "" if d.get("_minutes") is None else f", {d['_minutes']:.0f} min"
    cost = "" if d.get("cost") is None else f", ${d['cost']:.2f}"
    said = f", proposals {len(d.get('proposals') or [])}, lessons {len(d.get('lessons') or [])}"
    return (f"  {d['_where']}: {d.get('purpose')} of {d.get('recipe')}: held-out {d['score'] * HELD:.0f}/{HELD}, hidden {d.get('hidden')}"
            f"{cost}{took}, {d.get('ticks')} ticks{said}, merges {f.get('merges')}, rejections {f.get('rejections')}"
            + (f", STOPPED: {d['stopped']}" if d.get("stopped") else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--job", required=True, help="the stage's name in the queue")
    p.add_argument("--seen", type=int, default=0, help="results already looked at")
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--settle", type=float, default=150, help="seconds to wait for runs that end together")
    a = p.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    done, start, why = QUEUE / f"{a.job}.done", time.time(), ""
    while not why:
        n = len(results(out))
        if done.exists():
            why = f"the stage's job ended: {done.read_text().strip()}"
        elif n > a.seen:
            last = time.time()
            while time.time() - last < a.settle and not done.exists():  # runs started together end together
                time.sleep(5)
                if len(results(out)) > n:
                    n, last = len(results(out)), time.time()
            why = f"{n - a.seen} more run(s) left a result"
        elif not runner_alive():
            why = "the runner that holds the key is not running: nothing queued will start, and running jobs were stopped"
        elif time.time() - start > a.minutes * 60:
            why = f"no news for {a.minutes:.0f} minutes"
        else:
            time.sleep(5)
    rows = results(out)
    print(time.strftime("%H:%M:%S"), why)
    print(f"{len(rows)} run(s) with a result (reported cost ${sum(d.get('cost') or 0 for d in rows):.2f}); still running: {running(out) or 'none'}")
    for d in rows:
        print(line(d))
    errors = [str(e.parent.relative_to(out)) for e in (out / "runs").glob("*/**/error.json") if "work" not in e.relative_to(out).parts]
    if errors:
        print("runs that ended in an error:", errors)
    log = ROOT / "runs_recipes" / f"{a.job}.log"
    if log.exists():
        print("--- end of the stage log")
        print("\n".join(log.read_text().splitlines()[-14:]))


if __name__ == "__main__":
    main()
