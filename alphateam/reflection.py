"""The society reviews itself (code world; flag --reflect W).

GOAL.md: the society learns to organize itself, with continual learning: every agent reflects, and the society tries, measures,
keeps or reverts its practices. The user's frame for this loop is that the only goal is accuracy and any level of reflection and
change is allowed, credit assignment included. So every W ticks:

1. Judge. Accuracy is read from the held-out checks. If the last review changed something and the gain since is below floor
   times the gain before it, the change is reverted. An unbuilt HEAD is no signal, never a zero: nothing is judged on it.
2. Reflect. Each agent gets the facts of the window and says, in a few lines, which actions and ways of working paid or did not,
   and one concrete change. It goes to its strategy notes, its private board and the public board.
3. Propose. An organizer, the same model acting for the society, reads every reflection, the agents' own proposals (substrate
   op=propose), the facts, and the record of earlier changes, and returns any change to the rules: the law (credit weights and
   shares, the goal, steps, ownership terms, the gate's order), what each turn shows, and the practices.
4. Enforce. The harness applies it at once, within bounds, and posts the review.

GOAL.md's limit on all this: design the actions, institutions and learning loop, not the organization. So a review changes rules,
never who does what. It cannot place an agent in a team or a role, a practice names no agent or group (ids are stripped), and what
it thinks about teams it can only post as a suggestion. Teams, roles and leads remain the agents' own choices, which credit can
make worth choosing. Institutions stay offered: a review adds rules, and it never takes an action away.

Facts, not advice, go to everyone: they are measured by the world (alphateam/code_world.py window_facts) and never touch the hidden
tests. What is never revised is the score. A window's gain is noisy (in the ten-agent baseline the 12-tick gains ran from 0 to 49
of 179), so the revert is a rough guard and the organizer sees the record of what followed each change.
"""
from __future__ import annotations

import asyncio
import json

from .kernel import clip
from .society import RULE_KEYS

REFLECT_SYSTEM = ("You are one agent in a society of AI agents rebuilding a program. Reflect briefly and concretely on how the society "
                  "works: which actions and combinations of actions paid, what is assembled and what still lies unassembled, how work is "
                  "checked and handed over, who works with whom, and how credit is assigned.")
PROPOSE_SYSTEM = ("You revise the rules of a society of AI agents, acting for its members. Accuracy on hidden tests is the only goal, and the "
                  "held-out checks are your only reading of it. Credit assignment is the central means: it tells agents which actions and "
                  "combinations paid off, and it should have consequences. Progress is what is assembled and coherent at HEAD, not how many "
                  "pieces lie in working copies, and work that unblocks others should come first. You may change any rule offered below: credit weights and shares, "
                  "whose wealth credit is, steps, ownership terms, the gate's order, what each turn shows, and the practices. You design the "
                  "rules, not the organization: you never say who does what. Structure comes from the agents' own choices; practices are "
                  "offered, never required, and name no agent or group. Change a thing only for a reason in the evidence, and say what result "
                  "would show it helped. A window's gain is noisy and depends on what the last window left. Reply with JSON only.")


class ReflectionLoop:
    def __init__(self, window: int, floor: float = 0.5):
        self.window = max(2, int(window))
        self.floor = float(floor)
        self.history: list[dict] = []
        self.last_t, self.last_passed = 0, None
        self.gain_before = None
        self.open_change = None  # the last review's change, until a later window judges it
        self._seen_proposals = 0

    def _judge(self, k, w, gain: int | None) -> str:
        c = self.open_change
        if c is None or gain is None:
            return "no change to judge" if c is None else "no signal: HEAD did not build, so nothing is judged"
        self.open_change = None
        c["gain_after"] = gain
        before = c["gain_before"] or 0
        if before > 0 and gain < self.floor * before:
            c["verdict"] = "reverted"
            undo = {**c["undo"]["rules"], "show": c["undo"]["show"]}
            w.set_law(k, c["undo"]["law"], undo, c["undo"]["norms"], why=f"gain fell from {before} to {gain} after the change at t{c['t']}")
            return f"the change at t{c['t']} was reverted: gain {before} before it, {gain} after"
        c["verdict"] = "kept"
        return f"the change at t{c['t']} stands: gain {before} before it, {gain} after"

    async def _reflect_one(self, k, backend, aid: str, facts: dict) -> str:
        w = k.env
        row = facts["agents"].get(aid, {})
        prompt = (f"You are {aid}, role: {row.get('role') or 'none declared'}. The last window (ticks {facts['tick'] - self.window + 1}"
                  f"-{facts['tick']} of {facts['of']}):\n{json.dumps(facts, default=str)}\n\nThe law now: {w.law_words()}\n"
                  f"Practices now: {w.norms or 'none'}\n\nIn at most 100 words: which actions, combinations of actions and ways of working "
                  "helped or hurt accuracy in this window, and one concrete change: to the credit law or a practice, or a move of your "
                  "own (a team to form or join, a role to take, a piece to hold or hand over).")
        try:
            return (await backend.complete(k, REFLECT_SYSTEM, prompt, max_tokens=1500, who=aid)).strip()
        except Exception as exc:  # a lost reflection loses one voice, not the review
            return f"(no reflection: {exc.__class__.__name__})"

    async def _reflect(self, k, backend, facts: dict) -> dict:
        agents = list(k.active_agents())
        if not hasattr(backend, "complete"):  # scripted agents (tests)
            texts = [f"{a}: too many of us touched the same files; hold them." for a in agents]
        else:
            texts = await asyncio.gather(*(self._reflect_one(k, backend, a, facts) for a in agents))
        out = {}
        for aid, text in zip(agents, texts):
            if not text:
                continue
            out[aid] = clip(text, 700)
            k.agents[aid].notes.append(f"t{k.tick} reflection: {clip(text, 300)}")
            k.post(f"private:{aid}", aid, "lesson", f"Reflection at t{k.tick}: {clip(text, 700)}", notify=False)
            k.post("global", aid, "critique", f"REFLECTION at t{k.tick}: {clip(text, 700)}", notify=False)
        return out

    async def _propose(self, k, backend, facts: dict, reflections: dict, verdict: str) -> dict:
        w = k.env
        posts = [e for e in k.boards["global"].entries if e.kind == "proposal"]
        new, self._seen_proposals = posts[self._seen_proposals:], len(posts)
        if not hasattr(backend, "complete"):  # scripted agents (tests): a small, legal change
            return {"law": {"piece_share": round(min(w.LAW_BOUNDS["piece_share"][1], w.law["piece_share"] + 0.05), 3)},
                    "norms": "Say which piece your change rests on.", "why": "scripted", "expect": "more citations"}
        evidence = {"facts": facts, "verdict_on_last_change": verdict, "law_now": w.law, "law_bounds": w.LAW_BOUNDS, "law_choices": w.LAW_CHOICES,
                    "goal_can_change": not w.goal_locked,
                    "rules_now": {x: w.rules["credit"][x] for x in RULE_KEYS}, "rules_bounds": {"lead_share": [0, .6], "action_share": [0, .6],
                                                                                                "steps_per_credit": [0, 4]},
                    "views_shown": w.rules["substrate"]["show"], "views": list(w.SHOW), "practices_now": w.norms,
                    "reflections": reflections, "proposals_from_agents": [f"{e.author} t{e.tick}: {clip(e.content, 300)}" for e in new[-12:]],
                    "earlier_changes": [{"t": h["t"], "changed": h["changed"], "why": clip(h["why"], 200), "gain_before": h.get("gain_before"),
                                         "gain_after": h.get("gain_after"), "verdict": h.get("verdict")} for h in self.history[-5:]]}
        prompt = ("Evidence about the society so far:\n" + json.dumps(evidence, default=str) + "\n\nReturn JSON {\"law\": {changed keys only}, "
                  "\"rules\": {changed keys only, and \"show\": [views] if that changes}, \"practices\": \"text shown to every agent every "
                  "turn as offered practice, naming no agent or group, or leave it out to keep the current ones\", \"suggestions\": [short "
                  "posts for the agents about what the evidence shows on teams, roles or handovers; nothing makes anyone follow them], "
                  "\"why\": \"the evidence for each change\", \"expect\": \"what change in held-out gain would show it helped\"}. Leave "
                  "anything out that the evidence does not argue for changing.")
        text = await backend.complete(k, PROPOSE_SYSTEM, prompt, max_tokens=8000, who="organizer", json_mode=True)
        if not text.strip():
            raise ValueError("the organizer returned no content")
        return json.loads(text[text.find("{"): text.rfind("}") + 1])

    async def on_tick_end(self, k, backend):
        if k.tick % self.window or k.tick >= k.total_ticks or not hasattr(k.env, "window_facts"):
            return
        w = k.env
        now = w.progress()
        healthy = bool(now.get("built", True))
        first = w.held_log[0]["passed"] if w.held_log else 0
        gain = now["passed"] - (self.last_passed if self.last_passed is not None else first) if healthy else None
        verdict = self._judge(k, w, gain)
        facts = w.window_facts(k, self.last_t)
        reflections = await self._reflect(k, backend, facts)
        row = {"t": k.tick, "held_out": now.get("passed"), "gain": gain, "verdict_on_last": verdict, "reflections": reflections,
               "changed": {}, "suggestions": [], "why": "", "expect": "", "gain_before": gain}
        try:
            out = await self._propose(k, backend, facts, reflections, verdict)
        except Exception as exc:  # a failed review keeps the law
            row["why"] = f"(review failed: {exc.__class__.__name__}: {clip(exc, 200)})"
            out = None
        if out is not None:
            done = w.set_law(k, out.get("law"), out.get("rules"), out.get("practices", out.get("norms")), str(out.get("why") or ""))
            row.update(changed=done["changed"], suggestions=w.suggest(k, out.get("suggestions")), why=clip(str(out.get("why") or ""), 1500),
                       expect=clip(str(out.get("expect") or ""), 600))
            if done["changed"]:
                row["undo"] = done["before"]
                self.open_change = row
        self.history.append(row)
        k.post("global", "lab", "decision", f"REVIEW at t{k.tick}: held-out {now.get('passed')} of {now.get('total')}"
               + (f", gain {gain} in this window" if gain is not None else ", no reading: HEAD did not build") + f". {verdict}."
               + (f" Changed: {', '.join(sorted(row['changed']))}. Expected: {row['expect']}" if row["changed"] else " No rule changed.")
               + "".join(f"\nSuggested (nothing requires it): {x}" for x in row["suggestions"]), notify=False)
        self.last_t = k.tick
        if healthy:
            self.last_passed = now["passed"]

    def export(self) -> list[dict]:
        return [{key: v for key, v in h.items() if key != "undo"} for h in self.history]
