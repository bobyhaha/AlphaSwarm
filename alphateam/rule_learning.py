"""Evolving a society's credit rules and substrate (ARC world with explore=1; flag --rule-learning W).

The goal is a society that finds which of its actions and combinations are useful, with credit assignment as the means.
The credit weights and what the substrate shows are guesses, so they are a rule set the society can revise (see
explore.RULES and BOUNDS). What is never revised is the world's score (levels completed out of levels to win) and the
facts credit is computed from; credit is a means, so a rule set is judged only by the score.

Variation within a run: every W ticks a reviser reads the agents' rule proposals (substrate op=propose), which actions,
combinations and kinds of entry have paid, what agents used and reused, and the score's progress in the windows after
each earlier revision, and returns a bounded rule set, which the harness enforces and posts.
Selection across runs: a run exports its final rules, which a next run can start from (world option rules=<file>); the
score's area under the curve (mean score over ticks) compares runs, since levels are rare (after Agensh, which reads
time-to-threshold as well as the final score).
There is no revert within a run: with a few first completions per run, a window's progress is too noisy to judge a
change, so the reviser sees the progress after each revision as evidence instead.
"""
from __future__ import annotations

import json

from .explore import BOUNDS, RULES, VIEWS, clean_rules, rules_text
from .kernel import clip

SYSTEM = ("You revise the rules of a society of AI agents exploring unfamiliar games. The game score is fixed and is the "
          "only judge: levels completed out of levels to win. Credit is a means to it: it tells agents which actions pay "
          "and buys them extra game steps. Change a rule only for a reason in the evidence, and say what result would show "
          "the change helped. Return JSON only.")


class RuleLearner:
    def __init__(self, window: int):
        self.window = max(2, int(window))
        self.history: list[dict] = []
        self._seen_proposals = 0

    def _progress(self, k, start: int) -> dict:
        w = k.env
        firsts = [f for f in w.firsts if f["t"] > start]
        scores = [s["metrics"]["score"] for s in k.snapshots if s["t"] > start]
        return {"first_completions": len(firsts), "levels": sum(w.best.values()),
                "score_now": scores[-1] if scores else w.score()["score"],
                "mean_score_in_window": round(sum(scores) / len(scores), 4) if scores else None,
                "novel_frontier_states": sum(1 for c in w.credit_log if c["t"] > start and c["reason"].startswith("new state"))}

    def evidence(self, k) -> dict:
        w = k.env
        start = k.tick - self.window
        props = [e for e in k.boards["global"].entries if e.kind == "proposal"]
        new, self._seen_proposals = props[self._seen_proposals:], len(props)
        for h in self.history:
            h.setdefault("progress_after", None)
        if self.history:  # what happened in the window after the last revision
            self.history[-1]["progress_after"] = self._progress(k, self.history[-1]["t"])
        usage = {}
        for e in k.events:
            if e["ok"] and e["op"] != "intent" and e["t"] > start:
                op = e["op"].split(".")[1] if e["op"].startswith("lab.") else e["op"].split(".")[0]
                usage[op] = usage.get(op, 0) + 1
        return {"tick": k.tick, "of": k.total_ticks, "rules_now": w.rules, "bounds": BOUNDS, "views": list(VIEWS),
                "progress_this_window": self._progress(k, start),
                "earlier_revisions": [{"t": h["t"], "changed": h["changed"], "why": h["why"][:200],
                                       "progress_after": h.get("progress_after")} for h in self.history[-4:]],
                "proposals_from_agents": [f"{e.author} t{e.tick}: {clip(e.content, 300)}" for e in new[-12:]],
                "what_paid": sorted(([c, round(v[0], 2), v[1]] for c, v in w.combo_credit.items()), key=lambda x: -x[1])[:10],
                "credit_by_action": {op: [round(v[0], 2), v[1]] for op, v in w.op_credit.items()},
                "actions_used_this_window": usage, "actions_never_used": w.unused_actions(k),
                "kinds_of_entry": w.kind_stats(k), "agent_defined_kinds": w.kinds,
                "plans": len(w.plans), "findings": len(w.findings),
                "dead_ends": sum(f["outcome"] == "dead_end" for f in w.findings)}

    async def propose(self, k, backend, ev: dict) -> dict:
        if not hasattr(backend, "complete"):  # scripted agents (tests): a small, legal change
            r = json.loads(json.dumps(k.env.rules))
            r["credit"]["novel"] = round(min(BOUNDS["novel"][1], r["credit"]["novel"] * 1.5), 3)
            return {"rules": r, "why": "scripted: more novelty credit", "expect": "more new frontier states"}
        prompt = ("Evidence about the society so far:\n" + json.dumps(ev, indent=1, default=str) +
                  "\n\nReturn JSON {\"rules\": {\"credit\": {...}, \"substrate\": {\"plan_ttl\": n, \"items\": n, \"show\": "
                  "[views], \"rank\": {\"cites\": w, \"credit\": w, \"recency\": w}, \"promoted_kinds\": [agent-defined kinds "
                  "to show every turn]}}, \"why\": \"the evidence for each change\", \"expect\": \"what progress in the game "
                  "score would show it helped\"}. Keep every value within its bounds; leave a rule as it is unless the "
                  "evidence argues for changing it.")
        # reasoning models spend much of the budget thinking: a small limit returns no content at all
        text = await backend.complete(k, SYSTEM, prompt, max_tokens=8000, who="organizer", json_mode=True)
        if not text.strip():
            raise ValueError("the reviser returned no content")
        return json.loads(text[text.find("{"): text.rfind("}") + 1])

    async def on_tick_end(self, k, backend):
        if k.tick % self.window or k.tick >= k.total_ticks or not getattr(k.env, "explore", False):
            return
        ev = self.evidence(k)
        try:
            out = await self.propose(k, backend, ev)
        except Exception as exc:  # a failed revision keeps the rules
            self.history.append({"t": k.tick, "changed": {}, "why": f"(revision failed: {exc})", "expect": "",
                                 "rules": k.env.rules})
            return
        before = k.env.rules
        new = clean_rules(out.get("rules") or {})
        changed = _diff(before, new)
        if changed:
            k.env.set_rules(k, new, str(out.get("why") or ""))
        self.history.append({"t": k.tick, "changed": changed, "why": str(out.get("why") or "")[:1500],
                             "expect": str(out.get("expect") or "")[:600], "rules": new,
                             "progress_before": ev["progress_this_window"]})

    def export(self) -> list[dict]:
        return self.history


def _diff(a: dict, b: dict, path: str = "") -> dict:
    out = {}
    for key in set(a) | set(b):
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            out.update(_diff(x, y, f"{path}{key}."))
        elif x != y:
            out[f"{path}{key}"] = [x, y]
    return out


__all__ = ["RuleLearner", "RULES", "rules_text"]
