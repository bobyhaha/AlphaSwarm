"""Measurements of the society (hidden from the agents).

- Science: test score, coverage, structure F1, last validation.
- Organization: groups, nesting depth, supergroups, membership overlap, formation/dissolution, protocol changes.
- Mirroring: does the organization line up with the hidden structure of the task? For each agent we take the
  module it has recently worked on. Index = P(share a group | same module) - P(share a group | different module):
  1 means groups follow modules exactly, 0 means groups ignore them. Level 1 uses each agent's smallest group and
  modules; level 2 uses its outermost superagent and super-modules.
- Expertise: do agents with better validated track records get more followers and delegate roles?
- Communication: tokens by category this tick.
"""
from __future__ import annotations

import math
from collections import Counter

from .kernel import Kernel


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def mirroring_index(labels: list, cats: list) -> float | None:
    n = len(labels)
    same_cat = sum(_pairs(c) for c in Counter(cats).values())
    both = sum(_pairs(c) for c in Counter(zip(labels, cats)).values())
    same_lab = sum(_pairs(c) for c in Counter(labels).values())
    diff_cat = _pairs(n) - same_cat
    if not same_cat or not diff_cat:
        return None
    return round(both / same_cat - (same_lab - both) / diff_cat, 4)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 4)


def measure(k: Kernel) -> dict:
    env = k.env
    active = k.active_agents()
    groups = k.active_groups()
    size = {g.id: len(k.agents_inside(g.id)) for g in groups}
    depth = {g.id: k.depth(g.id) for g in groups}
    m = {"agents": len(active), "groups": len(groups),
         "groups_2plus": sum(s >= 2 for s in size.values()),
         "supergroups": sum(any(x in k.groups for x in g.members) for g in groups),
         "depth": max(depth.values(), default=0),
         "largest": max(size.values(), default=0),
         "in_group": round(sum(bool(k.member_of.get(a)) for a in active) / max(1, len(active)), 4),
         "memberships": round(sum(len(k.member_of.get(a, ())) for a in active) / max(1, len(active)), 3),
         "links": sum(ln.status == "open" for ln in k.links.values()),
         "link_requests": k.tick_counts["link_requests"], "links_opened": k.tick_counts["links_opened"],
         "link_messages": k.tick_counts["link_messages"],
         "formed": k.tick_counts["formed"], "dissolved": k.tick_counts["dissolved"],
         "reorganizations": k.tick_counts["reorganizations"], "protocol_changes": k.tick_counts["protocol_changes"],
         "custom": len(k.custom), "custom_uses": sum(c.uses for c in k.custom.values()),
         "tasks_open": sum(t.status == "open" for t in k.tasks.values()),
         "tasks_done": sum(t.status == "accepted" for t in k.tasks.values()),
         "experiments": k.tick_counts["experiments"], "published": k.tick_counts["published"],
         "tokens": k.total_tokens()}
    for cat in ("deliberation", "read", "communication", "summarize", "critique", "custom"):
        m[f"tok_{cat}"] = k.tick_counts[f"tok_{cat}"]
    if env is not None:
        m.update(env.score())
        m.update(mirroring(k, size))
        m["expertise"] = expertise(k)
    return m


def mirroring(k: Kernel, size: dict[str, int], window: int = 3) -> dict:
    env = k.env
    if not getattr(env, "has_modules", False):
        return {"mirror_l1": None, "mirror_l2": None, "focused": 0}
    mods = {a: m for a, m in env.dominant_modules(k.tick, window).items() if k.agents[a].active}
    l1, l2, c1, c2 = [], [], [], []
    for a, mod in mods.items():
        direct = k.member_of.get(a, ())
        small = min(direct, key=lambda g: (size.get(g, 0), g)) if direct else f"solo:{a}"
        roots = [g for g in k.containers(a) if not k.member_of.get(g)]
        top = max(roots, key=lambda g: (size.get(g, 0), g)) if roots else small
        l1.append(small)
        c1.append(mod)
        l2.append(top)
        c2.append(env.modules[mod]["super"])
    return {"mirror_l1": mirroring_index(l1, c1) if len(l1) >= 4 else None,
            "mirror_l2": mirroring_index(l2, c2) if len(l2) >= 4 and env.n_super > 1 else None,
            "hier_align": hierarchy_alignment(k, mods), "focused": len(mods)}


def hierarchy_alignment(k: Kernel, mods: dict[str, int]) -> float | None:
    """For agents working on composite outputs: P(shares a group with someone working on the composite's parts)
    - P(shares a group with someone working on unrelated modules). Positive means integration is organized
    around what it integrates, i.e. the organization's hierarchy follows the task's."""
    env = k.env
    level = {m["id"]: m.get("level", 0) for m in env.modules}
    if not any(level.values()):
        return None
    parts = {}
    for m in env.modules:
        if m.get("level", 0) == 1:
            parts[m["id"]] = {c["id"] for c in env.modules if c.get("level", 0) == 0 and c["super"] == m["super"]}
        elif m.get("level", 0) == 2:
            parts[m["id"]] = {c["id"] for c in env.modules if c.get("level", 0) == 1}
    workers: dict[int, list[str]] = {}
    for a, m in mods.items():
        workers.setdefault(m, []).append(a)
    hit = [0, 0]
    tot = [0, 0]
    for comp, kids in parts.items():
        for a in workers.get(comp, []):
            ca = k.containers(a)
            for m, agents in workers.items():
                if m == comp:
                    continue
                j = 0 if m in kids else 1
                for b in agents:
                    tot[j] += 1
                    hit[j] += bool(ca & k.containers(b))
    if not tot[0] or not tot[1]:
        return None
    return round(hit[0] / tot[0] - hit[1] / tot[1], 4)


def expertise(k: Kernel) -> float | None:
    """Correlation between an agent's validated record and the standing others give it (followers + delegate roles)."""
    env = k.env
    record: dict[str, list[float]] = {}
    if hasattr(env, "author_scores"):  # benchmark worlds: graded answers (known only after grading)
        record = env.author_scores()
    else:
        for law in env.laws.values():
            if law.val is not None:
                record.setdefault(law.author, []).append(law.val)
    if len(record) < 5:
        return None
    followers = Counter(f for a in k.agents.values() for f in a.follows)
    delegate = Counter(d for g in k.active_groups() for d in g.delegates)
    xs, ys = [], []
    for aid, vals in record.items():
        xs.append(sum(vals) / len(vals))
        ys.append(followers[aid] + delegate[aid])
    return _pearson(xs, ys)


def snapshot(k: Kernel) -> dict:
    env = k.env
    focus = env.dominant_modules(k.tick) if env is not None else {}
    return {"t": k.tick, "metrics": measure(k), "structure": k.structure(),
            "focus": {a: m for a, m in focus.items() if k.agents[a].active}}
