"""Offline backend for exercising the kernel and the viewer without API calls.

Its choices are random heuristics, so any structure it produces is scripted,
not emergent. Use it to test plumbing; use the Claude backend for experiments.
"""
from __future__ import annotations

import random

from .world import CustomAction, World, clip, est_tokens

TOPICS = ["hypotheses", "screening pipeline", "synthesis route", "characterization", "transport theory",
          "experimental design", "risk analysis", "integration", "data analysis", "literature map"]
NOTES = ["Draft result: {t} looks promising under condition {n}.", "Open question on {t}: what is the acceptance threshold?",
         "Data point: {t} gave {n}% improvement in the toy check.", "Decision: we prioritize {t} over alternatives for now.",
         "Interface for {t}: inputs = candidate list, outputs = ranked table with uncertainty."]
INTENTS = ["Keep pushing my block forward.", "Sync my team's findings upward next.", "Look for someone to verify my work.",
           "Consider whether our team is too big.", "Read what other teams found before acting."]


class MockBackend:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def _topic(self, aid: str) -> str:
        return TOPICS[int(aid[1:]) % len(TOPICS)]

    async def turn(self, world: World, aid: str):
        left = world.budget_left()
        if left is not None and left <= 0:
            return
        invites = [n["invite"] for n in world.agents[aid].inbox if n.get("invite")]
        world.charge(aid, "deliberation", est_tokens(world.observe(aid)) + 400)
        for _ in range(self.rng.randint(1, world.max_actions)):
            name, args = self._choose(world, aid, invites)
            await world.act(aid, name, args)
            invites = []
        world.log_intent(aid, self.rng.choice(INTENTS))

    def _choose(self, w: World, aid: str, invites: list[str]):
        r, me = self.rng, w.agents[aid]
        opts: list[tuple[float, str, dict]] = []

        def add(weight, name, args):
            prim = "design_action" if name.startswith("x_") else name
            ops = w.allowed.get(prim, "missing")
            if ops != "missing" and (ops is None or name.startswith("x_") or args.get("op") in ops):
                opts.append((weight, name, args))

        teams = w.active_teams()
        others = [x for x in w.agents if x != aid]
        topic = self._topic(aid)
        note = r.choice(NOTES).format(t=topic, n=r.randint(2, 40))

        for m in w.motions.values():
            if m.status == "open" and aid in w.subtree_agents(m.team) and aid not in m.yes | m.no:
                add(4, "connect", {"op": "vote", "motion": m.id, "vote": "yes" if r.random() < 0.8 else "no"})
        for tid in invites:
            if tid in w.teams and w.teams[tid].dissolved is None and tid != me.team:
                add(5, "connect", {"op": "join", "target": tid})

        if me.team is None:
            if teams:
                add(4, "connect", {"op": "join", "target": r.choice(teams).id})
            add(2 if len(teams) < 3 else 0.4, "connect", {"op": "form_team", "name": topic.title(),
                "charter": f"Own the {topic} part of the mission.", "invite": r.sample(others, min(2, len(others)))})
            if others:
                add(1, "connect", {"op": "link", "target": r.choice(others)})
            add(1, "write", {"board": "global", "kind": "proposal", "content": f"I propose we organize around {topic}."})
        else:
            tid, t = me.team, w.teams[me.team]
            members = w.direct_members(tid)
            add(3, "write", {"board": "team", "kind": r.choice(["note", "data", "result", "question"]), "content": note})
            add(0.6, "write", {"board": "global", "content": f"@{tid} update: {note}"})
            mates = [m for m in members if m != aid]
            if len(members) >= 4:
                add(1.5, "connect", {"op": "form_team", "name": f"{topic.title()} cell", "parent": tid,
                                     "charter": f"Narrower focus inside {t.name}.", "invite": r.sample(mates, 2)})
            if len(members) >= 5:
                add(0.6, "connect", {"op": "split", "target": tid, "members": r.sample(members, 2),
                                     "name": f"{topic.title()} squad", "charter": "Split to cut coordination overhead."})
            tops = [x for x in teams if x.parent is None and x.id != tid]
            if t.parent is None and tops:
                biggest = max(tops, key=lambda x: len(w.subtree_agents(x.id)))
                add(0.5, "connect", {"op": "nest", "target": tid, "parent": biggest.id})
                if len(members) <= 2:
                    add(0.3, "connect", {"op": "merge", "target": tid, "into": r.choice(tops).id})
            if t.rule == "any":
                add(0.15, "connect", {"op": "govern", "target": tid, "rule": r.choice(["rep", "majority"]), "reps": [aid]})
            add(1, "summarize", {"source": "team", "dest": t.parent or "global", "include_subteams": True,
                                 "focus": "what the enclosing superagent needs"})
            add(0.6, "reflect", {"target": r.choice(["self", tid, "society", *others]),
                                 "content": "Too many small updates; coordination is eating the budget.",
                                 "proposal": "Batch updates and summarize once per tick."})
            add(0.3, "share", {"with": [tid], "mode": "read"})
            if others:
                add(0.4, "connect", {"op": "link", "target": r.choice(others)})

        blocks = w.blocks

        def kids(bid):
            return [x for x in blocks.values() if x.parent == bid]

        if not kids("M0"):
            add(3, "plan", {"op": "decompose", "block": "M0", "blocks": [
                {"title": s.title(), "spec": f"Deliver the {s} with explicit acceptance criteria."} for s in r.sample(TOPICS, 4)]})
        unowned = [b for b in blocks.values() if b.status == "open" and b.owner is None and b.id != "M0"]
        if unowned:
            b = r.choice(unowned)
            add(2, "plan", {"op": "claim", "block": b.id})
            if me.team:
                add(1, "plan", {"op": "assign", "block": b.id, "to": me.team})
        for b in blocks.values():
            if b.id != "M0" and b.status == "claimed" and (b.owner == aid or b.owner in w.agent_teams(aid)):
                ch = kids(b.id)
                if not ch and r.random() < 0.3:
                    add(1, "plan", {"op": "decompose", "block": b.id, "blocks": [
                        {"title": f"{b.title} part {i}", "spec": "Sub-spec with its own interface."} for i in (1, 2)]})
                if all(c.status == "done" for c in ch):
                    add(2, "plan", {"op": "submit", "block": b.id, "deliverable": f"Integrated result for {b.title}. {note}"})
        submitted = [b for b in blocks.values() if b.status == "submitted" and b.submitted_by != aid]
        if submitted:
            add(2, "plan", {"op": "verify", "block": r.choice(submitted).id,
                            "verdict": "accept" if r.random() < 0.75 else "reject", "notes": "Checked against the spec."})
        top = kids("M0")
        if top and all(c.status == "done" for c in top) and blocks["M0"].status in ("open", "claimed"):
            add(3, "plan", {"op": "submit", "block": "M0", "deliverable": "Integrated mission deliverable from all top-level blocks."})

        if len(w.custom) < 4:
            add(0.25, "design_action", {"op": "define", "name": f"{r.choice(['sync_up', 'digest', 'review_round'])}_{r.randint(1, 99)}",
                                        "description": "Roll my team's board up into the enclosing team, then post a status line.",
                                        "procedure": "summarize team -> parent team board (or global); write one-line status to global.",
                                        "params": ["focus"]})
        unadopted = [c for c in w.custom if c not in me.actions]
        if unadopted:
            add(0.6, "design_action", {"op": "adopt", "name": r.choice(unadopted)})
        if me.actions:
            add(1, "x_" + r.choice(sorted(me.actions)), {"focus": "status"})

        total = sum(o[0] for o in opts)
        x = r.random() * total
        for weight, name, args in opts:
            x -= weight
            if x <= 0:
                return name, args
        return opts[-1][1], opts[-1][2]

    async def run_custom(self, world: World, aid: str, ca: CustomAction, args: dict) -> str:
        me = world.agents[aid]
        if me.team:
            dest = world.teams[me.team].parent or "global"
            ok, msg = await world.act(aid, "summarize", {"source": "team", "dest": dest, "focus": args.get("focus", "")}, via=ca.name)
        else:
            ok, msg = await world.act(aid, "write", {"board": "global", "content": "Status: working solo."}, via=ca.name)
        return f"x_{ca.name}: {clip(msg, 160)}"

    async def summarize(self, world: World, aid: str, text: str, focus: str) -> str:
        lines = [ln.split("] ", 1)[-1] for ln in text.splitlines() if ln.startswith("[")]
        world.charge(aid, "summarize", est_tokens(text) // 4)
        return "Summary: " + " | ".join(clip(ln, 80) for ln in lines[-4:])

    async def retrospective(self, world: World) -> str:
        return "(mock run: no retrospective; use the Claude backend for organizational lessons)"
