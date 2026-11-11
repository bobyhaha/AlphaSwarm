#!/usr/bin/env python3
"""Reopen revisions that a single-run pass closed after one run (free; changes statuses in an archive, nothing else).

  .venv/bin/python tools/reopen_single_runs.py --out runs_recipes/XSV1

The first single-run generation of 2026-11-10 ran under the rule "dropped when behind the parent after one run, taken
on when ahead after one run". Runs of one recipe differ by about nine held-out checks, so one run settles neither.
Under the rule that replaced it (alphateam/selection.py: lead_runs, retests) such a revision is set aside until it has
two runs. This puts the archive in the state the new rule would have left: a revision dropped at the screen, or taken
on as a lead, with fewer than two runs of its own becomes a candidate again. Every change is written to the
archive's decision log.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphateam.archive import FRESH, Archive  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--lead-min", type=float, default=0.0, help="also reopen a lead whose mean is ahead of its parent's "
                   "mark by less than this (the margin the pass now asks for)")
    a = p.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    archive = Archive(out / "archive")
    n = 0
    for rec in list(archive.recipes(("dropped", "lead"))):
        runs = [q for q in archive.pairs(rec["id"], ("screen",)) if q.get("b") is not None]
        if rec["status"] == "dropped" and rec.get("stage") != "screen":
            continue  # dropped before any run (typed, dry, probe): that stands
        if not runs:
            continue
        if len(runs) >= 2:  # judged on two runs: that stands, unless it is a lead that is ahead by less than the margin
            parent = rec["recipe"].get("parent")
            base = [r["score"] for r in archive.runs(parent, None, FRESH) if r.get("score") is not None]
            if rec["status"] != "lead" or not base or not a.lead_min:
                continue
            gain = sum(q["b"] for q in runs) / len(runs) - sum(base) / len(base)
            if gain > a.lead_min:
                continue
            why = (f"mean gain {gain:+.3f} over {len(runs)} runs; reopened on 2026-11-10: ahead of its parent by less than "
                   f"the {a.lead_min:.3f} it now takes to be taken on (it was 'lead')")
        else:
            gain = sum(q["gain"] for q in runs if q.get("gain") is not None) / len(runs)
            why = (f"gain {gain:+.3f} on its one run; reopened on 2026-11-10: one run settles nothing, it needs two "
                   f"(it was '{rec['status']}')")
        archive.set_status(rec["id"], "candidate", why, "screen")
        archive.decide(parent=rec["recipe"].get("parent"), child=rec["id"], edits=rec["recipe"]["origin"].get("edits", []),
                       by="reopen", kept=False, judged=False, stage="screen", why=why, stages=[], runs=0, cost=0.0,
                       niche=rec.get("niche"), mode="paired")
        print(f"reopened {rec['id']}: {why}")
        n += 1
    print(f"{n} revision(s) reopened")


if __name__ == "__main__":
    main()
