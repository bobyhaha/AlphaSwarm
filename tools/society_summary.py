#!/usr/bin/env python3
"""What structure a society built for itself with take_lead, contracts and teams (design world).

For each run: score, the leads (or teams) at the end and over time, whether each led scope is exactly a hidden block, a
hidden super-block, or neither (mirroring), nesting depth, takeovers, delegations, and contracts.
  .venv/bin/python tools/society_summary.py runs_society
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphateam.design_world import DesignWorld  # noqa: E402


def hidden_structure(cfg: dict, seed: int):
    opts = cfg.get("world_opts") or {}
    w = DesignWorld(cfg["agents"], seed=seed, opts={k: v for k, v in opts.items() if k in ("blocks", "supers", "subs", "mid", "private", "interface", "top", "top_per_block", "agents_per_block")})
    supers = {frozenset(w.block_label[b] for b in range(len(w.blocks)) if w.blocks[b]["super"] == s) for s in range(w.n_super)}
    subs = set()
    for key in set(w.block_sub.values()):
        chunk = frozenset(w.block_label[b] for b, v in w.block_sub.items() if v == key)
        if len(chunk) >= 2:
            subs.add(chunk)
    return supers | {("sub", x) for x in subs}, len(w.blocks), w.block_label


def kind(scope: frozenset, supers: set, n_blocks: int) -> str:
    if len(scope) == 1:
        return "block"
    if ("sub", scope) in supers:
        return "PAIR (sub-group)"
    supers = {s for s in supers if not isinstance(s, tuple)}
    if scope in supers:
        return "SUPER-BLOCK"
    if len(scope) == n_blocks:
        return "everything"
    if any(scope < s for s in supers):
        return "part of a super-block"
    return "crosses super-blocks"


def summarize(d: Path):
    data = json.loads((d / "run.json").read_text())
    meta, w, snaps = data["meta"], data["world"], data["snapshots"]
    cfg = meta["config"]
    supers, n_blocks, label = hidden_structure(cfg, meta.get("world_seed", meta.get("seed", 0)))
    sc = [s["metrics"]["score"] for s in snaps[1:]]
    half = sc[len(sc) // 2:]
    print(f"\n== {d.parent.name}/{d.name}  agents {len(data['agents'])}  final {sc[-1]:+.3f}  second half "
          f"{sum(half) / len(half):+.3f}  reversals {snaps[-1]['metrics'].get('shared_reversals')}  cost ${meta.get('cost_usd', 0):.2f}")
    log = w.get("lead_log", [])
    if w.get("teams"):
        print(f"  teams founded {sum(e['op'] == 'found' for e in log)}; at the end:")
        for g, t in w["teams"].items():
            print(f"    {g} lead {t['lead']} founded t{t['t']} parent {t['parent']}")
        print(f"  jobs {len(w.get('jobs', {}))}; metrics: " + ", ".join(f"{k} {snaps[-1]['metrics'].get(k)}" for k in
                                                                          ("teams", "team_depth", "in_teams", "team_knobs")))
    takes = [e for e in log if e["op"] == "take"]
    if takes or w.get("leads"):
        print(f"  take_lead events {len(takes)} (takeovers {sum(bool(e.get('from')) for e in takes)}), appoint "
              f"{sum(e['op'] == 'appoint' for e in log)}, step_down {sum(e['op'] == 'step_down' for e in log)}")
        for e in takes[:14]:
            scope = frozenset(label[b] for b in e["scope"])
            print(f"    t{e['t']} {e['agent']} took {{{','.join(sorted(scope, key=lambda x: int(x[1:])))}}} "
                  f"({kind(scope, supers, n_blocks)})" + (f" from {e['from']}" if e.get("from") else ""))
        end = [(frozenset(x["scope"]), x["lead"]) for x in w.get("leads", [])]
        kinds = Counter(kind(s, supers, n_blocks) for s, _ in end)
        print(f"  leads at the end: {len(end)} ({dict(kinds)}); depth {snaps[-1]['metrics'].get('lead_depth')}; "
              f"knobs held by leads {snaps[-1]['metrics'].get('lead_knobs')}")
        for s, h in sorted(end, key=lambda x: -len(x[0])):
            print(f"    {h} leads {{{','.join(sorted(s, key=lambda x: int(x[1:])))}}}: {kind(s, supers, n_blocks)}")
    cs = w.get("contracts", [])
    if cs:
        acc = [c for c in cs if c["status"] == "accepted"]
        print(f"  contracts offered {len(cs)}, accepted {len(acc)}: " + "; ".join(
            f"{c['id']} {c['from']}->{c['to']} " + (f"job in {c.get('team')} salary {c.get('salary')}" if c.get("kind") == "hire"
                                                    else f"{c['knob']}={c['value']} for {c['pay']}") for c in acc[:6]))


for root in sys.argv[1:]:
    for f in sorted(Path(root).glob("**/run.json")):
        summarize(f.parent)
