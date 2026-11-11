"""Continual organizational learning inside a run (design world).

Every `window` ticks:
1. every agent reflects (one short call each): what about how the society is organized helped or hurt its work in the
   last window, and one concrete change. The reflection is posted to its private board and kept in its strategy notes;
2. if the window just finished scored clearly worse (by more than `margin`) than the window before it, under the
   previous strategy, the society reverts to that previous strategy; with stall=True a window that did not improve by
   `margin` either is marked stalled, and the proposer is told to change the strategy (a plateau is not a reason to
   keep it: a strategy that freezes the design looks stable and never triggers a revert); with judge='delta' each
   strategy is judged by what happened under it, the score change from the start to the end of its own window (revert
   when it fell by more than `margin`, stalled when it rose by less), so a drop that happened before a change is not
   charged to it;
3. a proposer (the same model, acting for the society) reads every reflection, the score of each window and the record
   of earlier strategies and decisions, and writes the teamwork strategy for the next window;
4. the harness enforces the strategy at once: it builds the teams, hands out decision rights, sets the phase rule and
   the default board for results, and shows the norms and roles to every agent.
Agents keep every primitive throughout; the strategy adds structure, it does not take actions away.

A teamwork strategy is organization as an executable, testable artifact (the idea of Pappu et al. 2026, 'Self-Organizing
Agent Teams Learn to Reason Together', where a team learns reusable strategies of phases, roles and information flow).
It is problem-independent: it never names knobs, blocks or agents.
  teams        none | clusters (agents whose blocks share 3 or more knobs, i.e. strongly coupled blocks) | all
  council      the teams form a council (a group of the teams that acts through their delegates); needs clusters
  rights       open | team_shared (a knob shared by several blocks is held by the team spanning all of its users, or the
               council when no team does) | team_all (also each block's own knobs, by its team) |
               one_user (each shared knob is held by one agent whose block uses it) |
               top_credit (each shared knob is held by the agent whose block uses it with the most credit, when the
               world keeps durable credit; authority earned by track record)
  explore      ticks at the start of the window during which nobody may set knobs (0-3)
  results_to   private | team | global: where simulation results go unless the agent says otherwise
  norms, delegate_role, member_role: text shown to every agent
"""
from __future__ import annotations

import asyncio
import json
import random
import re

from .kernel import Board, Group, clip

TEAMS = ("none", "clusters", "all")
RIGHTS = ("open", "team_shared", "team_all", "one_user", "top_credit")
FLOW = ("private", "team", "global")
SELF_ORG = {"teams": "none", "council": False, "rights": "open", "explore": 0, "results_to": "private", "norms": "",
            "delegate_role": "", "member_role": ""}
IDS = re.compile(r"\b[kKbBaAgG]\d+\b")  # instance-specific names never enter a strategy

REFLECT_SYSTEM = "You are one agent in a society of AI agents. Reflect briefly and concretely on how the society is organized."
PROPOSE_SYSTEM = ("You are the organizational designer of a society of AI agents, acting for its members. You revise how "
                  "the society is organized, using every member's reflection and the measured outcome of earlier "
                  "strategies. Window scores are noisy and depend on the state the previous window left, so weigh the "
                  "evidence. Reply with JSON only.")
SPEC = __doc__[__doc__.index("  teams"):]


def clean(s: dict | None) -> dict:
    """A valid strategy from whatever the proposer wrote (unknown values fall back to self-organization)."""
    s = s or {}
    out = dict(SELF_ORG)
    for key, allowed in (("teams", TEAMS), ("rights", RIGHTS), ("results_to", FLOW)):
        if str(s.get(key)) in allowed:
            out[key] = str(s[key])
    out["council"] = bool(s.get("council")) and out["teams"] == "clusters"
    try:
        out["explore"] = max(0, min(3, int(s.get("explore") or 0)))
    except (TypeError, ValueError):
        pass
    for key in ("norms", "delegate_role", "member_role"):
        out[key] = IDS.sub("", str(s.get(key) or "")).strip()[:500]
    if out["teams"] == "none":
        out["results_to"] = "private" if out["results_to"] == "team" else out["results_to"]
        if out["rights"] in ("team_shared", "team_all"):
            out["rights"] = "one_user"
    return out


def describe(s: dict) -> str:
    parts = {"none": "no strategy teams", "clusters": "teams of agents whose blocks are strongly coupled (share 3+ knobs)",
             "all": "one team of everyone"}[s["teams"]]
    if s["council"]:
        parts += ", plus a council of the teams acting through their delegates"
    parts += "; rights: " + {"open": "anyone may set any unheld knob",
                             "team_shared": "each shared knob is held by the team spanning its users (or the council)",
                             "team_all": "teams hold their blocks' knobs and the shared knobs they span (the council the rest)",
                             "one_user": "each shared knob is held by one agent whose block uses it",
                             "top_credit": "each shared knob is held by the agent with the most credit among those whose "
                                           "blocks use it"}[s["rights"]]
    if s["explore"]:
        parts += f"; no knob may be set in the first {s['explore']} tick(s) of the window"
    parts += f"; simulation results go to {'your team board' if s['results_to'] == 'team' else s['results_to']} by default"
    return parts


def clusters(env) -> list[list[int]]:
    """Connected components of blocks that share at least 3 knobs (strong coupling)."""
    n = len(env.blocks)
    idx = [set(int(i) for i in b["idx"]) for b in env.blocks]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a in range(n):
        for b in range(a + 1, n):
            if len(idx[a] & idx[b]) >= 3:
                parent[find(a)] = find(b)
    comps: dict[int, list[int]] = {}
    for b in range(n):
        comps.setdefault(find(b), []).append(b)
    return sorted(comps.values())


def _num(x: str) -> int:
    return int(x[1:])


class OrgLearner:
    def __init__(self, window: int, margin: float = 0.05, stall: bool = False, judge: str = "level"):
        self.window, self.margin, self.stall, self.judge = max(2, int(window)), margin, stall, judge
        self.strategy = dict(SELF_ORG)
        self.prev: tuple[dict, float] | None = None  # the previous window's strategy and score
        self.groups: list[str] = []
        self.held: dict[int, str] = {}  # knob -> holder this strategy assigned
        self.history: list[dict] = []
        self.start = 0

    # ------------------------------------------------------------------ the loop

    async def on_tick_end(self, k, backend):
        if k.tick % self.window or k.tick >= k.total_ticks:
            return
        scores = [s["metrics"]["score"] for s in k.snapshots[self.start + 1: k.tick + 1] if s["metrics"].get("score") is not None]
        score = sum(scores) / len(scores) if scores else 0.0
        entry = {"t": k.tick, "window": [self.start + 1, k.tick], "score": round(score, 4), "strategy": self.strategy,
                 "activity": self._activity(k)}
        base = self.strategy
        if self.judge == "delta":  # what changed under this strategy, start to end of its own window
            first = k.snapshots[self.start]["metrics"].get("score") or 0.0
            delta = (k.snapshots[k.tick]["metrics"].get("score") or 0.0) - first
            entry["delta"] = round(delta, 4)
            if self.history and delta < -self.margin:
                base = self.prev[0] if self.prev else dict(SELF_ORG)
                entry["decision"] = f"reverted: the score fell {delta:+.3f} under this strategy"
            elif self.stall and self.history and delta < self.margin:
                entry["decision"] = f"stalled: the score moved only {delta:+.3f} under this strategy"
            else:
                entry["decision"] = "kept" if self.history else "baseline"
        elif self.prev and score < self.prev[1] - self.margin:
            base = self.prev[0]
            entry["decision"] = f"reverted: {score:.3f} < previous window {self.prev[1]:.3f} - {self.margin}"
        else:
            entry["decision"] = "kept" if self.prev else "baseline"
            if self.stall and self.prev and score < self.prev[1] + self.margin:
                entry["decision"] = f"stalled: {score:.3f} is not {self.margin} above the previous window {self.prev[1]:.3f}"
        reflections = await self._reflect(k, backend, score)
        try:
            proposal = await self._propose(k, backend, base, reflections)
            new, entry["rationale"] = clean(proposal.get("strategy")), str(proposal.get("rationale") or "")[:1200]
        except Exception as exc:  # a failed proposal keeps the base strategy
            new, entry["rationale"] = base, f"(proposal failed: {exc})"
        entry["reflections"] = reflections
        entry["next"] = new
        self.history.append(entry)
        self.prev = (self.strategy, score) if not entry["decision"].startswith("reverted") else self.prev
        self.apply(k, new)
        self.start = k.tick

    def _activity(self, k) -> dict:
        env = k.env
        sets = [s for s in env.sets if self.start < s["t"] <= k.tick]
        shared = [s for s in sets if s["level"] != "private"]
        claims = [c for c in env.claims_log if self.start < c["t"] <= k.tick]
        return {"knob_sets": len(sets), "shared_sets": len(shared), "shared_reversals": sum(s["reversal"] for s in shared),
                "sets_for_groups": sum(bool(s["group"]) for s in sets), "claims_by_agents": sum(c["holder"].startswith("A") for c in claims),
                "claims_by_groups": sum(c["holder"].startswith("G") for c in claims),
                "groups_agents_formed": sum(1 for g in k.active_groups() if g.id not in self.groups)}

    async def _reflect(self, k, backend, score: float) -> dict[str, str]:
        aids = [a for a in k.active_agents()]
        trend = ", ".join(f"t{s['t']} {s['metrics']['score']:.2f}" for s in k.snapshots[self.start + 1: k.tick + 1])
        prompt = ("{obs}\n\nThe society's current teamwork strategy: " + describe(self.strategy) +
                  f".\nDesign score over the last {self.window} ticks: {trend}.\nIn at most 80 words: what about how the "
                  "society is organized (structure, who decides which knobs, phases, information flow, norms) helped or "
                  "hurt your work in these ticks, and one concrete change to the organization you would make.")

        async def one(aid):
            if not hasattr(backend, "complete"):  # scripted agents (tests)
                text = f"(mock reflection by {aid}) shared knobs keep flipping; give each shared knob one owner"
            else:
                try:
                    text = await backend.complete(k, REFLECT_SYSTEM, prompt.format(obs=clip(k.observe(aid, consume=False), 6000)),
                                                  max_tokens=3000, who=aid)  # reasoning models think first
                    if not text.strip():
                        return aid, "(no reflection: empty reply)"
                except Exception as exc:
                    return aid, f"(no reflection: {exc.__class__.__name__})"
            text = text.strip()[:700]
            k.post(f"private:{aid}", aid, "lesson", f"Reflection at t{k.tick}: {text}", notify=False)
            k.agents[aid].notes = (k.agents[aid].notes + [f"t{k.tick} reflection: {clip(text, 300)}"])[-3:]
            return aid, text
        return dict(await asyncio.gather(*(one(a) for a in aids)))

    async def _propose(self, k, backend, base: dict, reflections: dict[str, str]) -> dict:
        if not hasattr(backend, "complete"):
            cycle = [{"teams": "clusters", "council": True, "rights": "team_shared", "results_to": "team",
                      "norms": "delegates settle shared knobs after hearing members"}, {"rights": "one_user"}]
            return {"rationale": "(mock proposal)", "strategy": cycle[len(self.history) % 2]}
        record = [{"window": h["window"], "strategy": describe(h["strategy"]) + (f"; norms: {h['strategy']['norms']}" if
                   h["strategy"]["norms"] else ""), "mean_score": h["score"], "activity": h["activity"],
                   "decision_after": h["decision"]} | ({"score_change_in_window": h["delta"]} if "delta" in h else {})
                  for h in self.history]
        ledger = ""
        if getattr(k.env, "durable", False) and k.env.effective:
            eff = sorted(k.env.effective.items(), key=lambda kv: -kv[1])
            grp = sorted(((g, c[0]) for g, c in k.env.dcredit.items() if g in k.groups), key=lambda kv: -kv[1])
            ledger = ("CREDIT LEDGER (durable credit: what each agent's lasting changes and posted evidence contributed; "
                      "agents are anonymous here as A-numbers): " + ", ".join(f"{a} {v:+.3f}" for a, v in eff) +
                      (". Groups: " + ", ".join(f"{g} {v:+.3f}" for g, v in grp) if grp else "") + "\n\n")
        prompt = (ledger + f"STRATEGY LANGUAGE\n{SPEC}\nRECORD OF WINDOWS (each window ran under the strategy shown):\n"
                  f"{json.dumps(record, indent=1)}\n\nSTRATEGY TO BUILD ON (after any revert): {json.dumps(base)}\n\n"
                  "REFLECTIONS OF EVERY MEMBER ON THE LAST WINDOW:\n" +
                  "\n".join(f"- {a}: {clip(t, 400)}" for a, t in reflections.items()) +
                  ("\n\nTHE SOCIETY IS STALLED: the last window did not improve on the one before. Change the strategy "
                   "(structure, rights, phases or flow), not only its words: a strategy under which few knobs change "
                   "looks stable but cannot improve." if self.history and self.history[-1]["decision"].startswith("stalled")
                   else "") +
                  f"\n\nWrite the strategy for the next {self.window} ticks ({k.total_ticks - k.tick} ticks are left). Keep "
                  "what the evidence says works; change what the reflections and outcomes say hurts. Never name specific "
                  "knobs, blocks or agents. Reply as JSON: {\"rationale\": \"...\", \"strategy\": {\"teams\": ..., "
                  "\"council\": ..., \"rights\": ..., \"explore\": ..., \"results_to\": ..., \"norms\": \"...\", "
                  "\"delegate_role\": \"...\", \"member_role\": \"...\"}}")
        # reasoning models spend much of the budget thinking: a small limit returns no content at all
        text = await backend.complete(k, PROPOSE_SYSTEM, prompt, max_tokens=8000, who="organizer", json_mode=True)
        if not text.strip():
            raise ValueError("the organizer returned no content")
        return json.loads(text[text.find("{"): text.rfind("}") + 1])

    # ------------------------------------------------------------------ enforcing a strategy

    def _group(self, k, name: str, purpose: str, members: list[str], delegates: list[str]) -> str:
        gid = k.new_id("G")
        k.groups[gid] = Group(gid, name, purpose, members[0], k.tick, open=False, delegates=set(delegates),
                              protocol="Set up by the society's teamwork strategy; the strategy manages membership.")
        k.boards[f"mem:{gid}"] = Board(f"mem:{gid}", gid)
        k.boards[f"pub:{gid}"] = Board(f"pub:{gid}", gid)
        for m in members:
            k._add_member(gid, m)
        self.groups.append(gid)
        return gid

    def apply(self, k, s: dict):
        env = k.env
        for gid in self.groups:  # take down the previous strategy's structure
            if gid in k.groups:
                k._dissolve(gid)
        for i, holder in self.held.items():
            if env.rights.get(i) == holder:
                del env.rights[i]
        self.groups, self.held = [], {}
        teams, team_of, council = [], {}, None
        if s["teams"] != "none":
            comps = [list(range(len(env.blocks)))] if s["teams"] == "all" else clusters(env)
            for comp in comps:
                members = sorted((a for a, bs in env.instruments.items() if set(bs) & set(comp) and k.agents[a].active), key=_num)
                if not members:
                    continue
                labels = ", ".join(env.block_label[b] for b in comp)
                gid = self._group(k, f"Team {len(teams) + 1}", f"Blocks {labels}: settle the knobs these blocks share.",
                                  members, [members[0]])
                teams.append((gid, set(comp), members))
                for a in members:
                    team_of.setdefault(a, gid)
            if s["council"] and len(teams) > 1:
                ids = [g for g, _, _ in teams]
                council = self._group(k, "Council", "Settle the knobs shared across teams.", ids, ids)
        if s["rights"] != "open":
            rng = random.Random(f"rights:{k.tick}")
            for i in range(env.D):
                users = set(env.users[i])
                if len(users) < 2 and s["rights"] != "team_all":
                    continue
                holder = None
                if s["rights"] in ("team_shared", "team_all"):
                    holder = next((g for g, comp, _ in teams if users <= comp), council)
                if holder is None:
                    cand = sorted((a for a, bs in env.instruments.items() if set(bs) & users and k.agents[a].active), key=_num)
                    credit = getattr(env, "effective", {})
                    if s["rights"] == "top_credit" and cand and credit:
                        holder = max(cand, key=lambda a: (credit.get(a, 0.0), -_num(a)))
                    else:
                        holder = rng.choice(cand) if cand else None
                if holder:
                    env.rights[i] = holder
                    self.held[i] = holder
        env.no_sets_until = k.tick + s["explore"]
        env.default_to = ({a: g for a, g in team_of.items()} if s["results_to"] == "team" else
                          {a: "global" for a in env.instruments} if s["results_to"] == "global" else {})
        self.strategy = s
        text = (f"TEAMWORK STRATEGY (from t{k.tick + 1}; revised every {self.window} ticks from every member's reflection "
                f"and the measured outcome; the harness enforces its structure): {describe(s)}." +
                (f"\nNorms: {s['norms']}" if s["norms"] else "") +
                (f"\nDelegates (one per team, shown in the team's card) should: {s['delegate_role']}" if s["delegate_role"] else "") +
                (f"\nTeam members should: {s['member_role']}" if s["member_role"] else ""))
        k.stakes = text
        k.post("global", "organizer", "decision", text, notify=False)
        for gid, comp, members in teams:
            for a in members:
                k.post(f"private:{a}", "organizer", "note", f"You are in {gid} (blocks {', '.join(env.block_label[b] for b in sorted(comp))}"
                       f"; members {', '.join(members)}); {'you are its delegate' if a == members[0] else 'delegate ' + members[0]}"
                       + (f"; {gid} sits on the council {council}" if council else "") + ".", notify=False)

    def export(self) -> list[dict]:
        return self.history
