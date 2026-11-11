#!/usr/bin/env python3
"""FrontierScience arms side by side: score on the scored questions, how the sheet was reached, and what trust predicted.

  python3 tools/science_compare.py [runs_science]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs_science")
    for path in sorted(base.glob("F*_s*/*/run.json")):
        d = json.loads(path.read_text())
        w, meta = d["world"], d["meta"]
        fin, s = w.get("final") or {}, w.get("structure_measures", {})
        name = path.parent.parent.name
        rows = [a for v in w["attempts"].values() for a in v]
        judged = [a for a in rows if a.get("judged") is not None and w["questions"][a["task"]]["track"] == "olympiad"]
        sheet = {t: e for t, e in w["sheet"].items() if e}
        by = {a["id"]: a for a in rows}
        best, on_sheet = 0, 0
        for t, group in w["attempts"].items():
            if w["questions"][t]["track"] != "olympiad":
                continue
            marks = [a.get("judged") or 0 for a in group]
            best += bool(marks and max(marks) > 0)
            on_sheet += bool(sheet.get(t) and (by[sheet[t]].get("judged") or 0) > 0)
        level = defaultdict(list)
        for a in judged:
            n = len({x["agent"] for x in w["attempts"][a["task"]] if x.get("cls") == a.get("cls") and x["blind"]})
            level["reproduced" if a.get("cls") is not None and n >= 2 else "alone"].append(a["judged"])
        print(f"{name}: score {fin.get('score')} on {fin.get('n')} (olympiad {fin.get('olympiad')}, research pass {fin.get('research_pass')}, points {fin.get('research_points')}); "
              f"audited {(fin.get('audited') or {}).get('score')}; ${meta.get('cost_usd')}; {d['usage'].get('calls')} calls")
        print(f"   {s.get('attempts')} attempts on {sum(bool(v) for v in w['attempts'].values())} of {len(w['questions'])} questions by {s.get('attempters')} agents; independent {s.get('independent')}, "
              f"reconciled {s.get('reconciled')}, contested {s.get('contests')}; subject focus {s.get('subject_focus')}; teams {s.get('teams')}; tasks {s.get('tasks')}")
        print(f"   olympiad: a right answer existed on {best} questions, the sheet held it on {on_sheet}; attempt accuracy by agreement: "
              + ", ".join(f"{key} {sum(v) / len(v):.2f} (n={len(v)})" for key, v in level.items()))


if __name__ == "__main__":
    main()
