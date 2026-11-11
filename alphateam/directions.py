"""Directions (world option directions=1, with explore=1): a parallel search for an insight, kept visible.

In I8 (vc33 level 5) the society held the pieces of the answer: A7 stated the goal pairing at t5 and A1 the door rule at
t7. But 16 of 20 agents chased one decoy, the move counter read as an exit, and the goal pairing was never tested. Twenty
agents searching in parallel are worth twenty draws only if their attempts differ, someone stays on an unpopular line
long enough to test it, and the pieces found on different lines are combined. Directions offer that, never require it:
- a direction is one line of attack on a level (what the goal is, which mechanism matters), opened by anyone; an agent is
  on at most one at a time and leaves with one call; its claims and findings on the level are filed under it;
- every turn shows each direction on the level, how many agents are on it and what its pieces came to (herding is
  visible), and a member sees its own direction first, with how long it has been on it (persistence);
- when the level falls, each direction whose pieces the solver cites, or that the solver is on, earns a pot of credit
  shared among the agents who were on it, by their ticks on it plus three shares for each of their claims on it tested
  to a result: few agents on a line that proves right each earn more, and testing earns more than staying;
- each agent's deliveries put posts of directions with fewer agents first, its own first among equals (a crowd's posts
  come last, so a view counterweighs the herd instead of feeding it);
- the pieces that held are listed by direction, and a claim building on pieces of two directions is marked as combining
  them and their members are told (integration).

Credit for changing what the society holds (world option change_credit=1, 2026-11-02). In every vc33 level-5 run, credit
for new frame states was 95-98% of all credit, and 56-65% of those states were new only in row 0: the level's step-budget
bar, which fills as clicks are spent. The decoy's chasers were paid, and bought steps, for burning budget, while A14's
refutation of the decoy and A7's correct goal earned nothing. So a result pays by how much of the society's picture it
changes, when it changes it:
- a test that moves a piece's trust level, and a counterexample to a finding or a guide line (with receipts), pays the
  test credit times one plus the number of other agents who used the piece (up to five times): refuting what nine agents
  built on pays most; a challenge pays once per challenger and piece;
- the first test of a new idea filed under a direction pays the idea credit times two minus the share of the game's agents
  on that direction: an idea on a line nobody else holds pays nearly double, one in the crowd little more than before.
(The novelty fix that leaves known counters out of new states is in explore.py's _novel and applies everywhere.)
"""
from __future__ import annotations

from collections import Counter

from .kernel import clip

D_TEXT = 300  # characters of a direction's text kept


def direction_tool(pot: float) -> dict:
    return {"name": "direction", "description": (
        "Directions on your game's current level. A direction is one line of attack (what the goal is, which mechanism "
        "matters) that several agents can work on. open: start one (text); join: work on one (id=D3); leave. You are on at "
        "most one at a time, and your claims and findings on the level are filed under it. Every turn shows each "
        "direction, how many agents are on it and what its pieces came to. When the level falls, each direction whose "
        f"pieces the solver cites, or that the solver is on, earns {pot:g} credit shared among the agents who were on it, "
        "by ticks on it and claims on it tested: few agents on a direction that proves right each earn more."),
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["open", "join", "leave"]},
            "text": {"type": "string", "description": "open: the direction, in one or two sentences"},
            "id": {"type": "string", "description": "join: the direction's D-id"}}, "required": ["op"]}}


class Directions:
    """Mixin for worlds with the explore substrate (ArcWorld, DigWorld)."""

    def _init_directions(self, opts: dict):
        self.directions_on = self.explore and str(opts.get("directions", "")).lower() in ("1", "true", "yes")
        self.directions: dict[str, dict] = {}
        self.on_direction: dict[str, str] = {}  # agent -> the direction it is on
        self.piece_dir: dict[str, str] = {}  # entry or plan id -> the direction it was filed under
        self.direction_log: list[dict] = []
        self.direction_peak = 0.0  # the largest share of a game's agents ever on one direction
        self.change_credit = self.explore and str(opts.get("change_credit", "")).lower() in ("1", "true", "yes")
        self.change_log: list[dict] = []  # credit paid for tests and challenges, by how much use they touched
        self._challenged: set = set()  # (challenger, piece) pairs already paid
        if self.directions_on:
            tool = direction_tool(self.rules["credit"].get("direction", 3.0))
            self.tool_names = tuple(self.tool_names) + (tool["name"],)
            self.TOOLS = list(self.TOOLS) + [tool]

    # ------------------------------------------------------------------ membership

    def direction_of(self, aid: str) -> dict | None:
        """The direction an agent is on, if it is still open on the level its game is at."""
        d = self.directions.get(self.on_direction.get(aid, ""))
        if not d or d["status"] != "open" or d["game"] != self.assigned.get(aid) or d["level"] != self._problem(d["game"]):
            return None
        return d

    def _ticks_on(self, d: dict, aid: str, now: int) -> int:
        since = d["members"].get(aid)
        return d["tenure"][aid] + (now - since + 1 if since is not None else 0)

    def _leave(self, k, aid: str, why: str):
        d = self.directions.get(self.on_direction.pop(aid, ""))
        if d and aid in d["members"]:
            d["tenure"][aid] += k.tick - d["members"].pop(aid) + 1
            self.direction_log.append({"t": k.tick, "agent": aid, "direction": d["id"], "op": "leave", "why": why})

    def _join(self, k, aid: str, d: dict, op: str):
        if self.on_direction.get(aid) == d["id"] and aid in d["members"]:
            return
        self._leave(k, aid, f"joined {d['id']}")
        d["members"][aid] = k.tick
        self.on_direction[aid] = d["id"]
        players = sum(1 for x in k.agents if self.assigned.get(x) == d["game"]) or 1
        self.direction_peak = max(self.direction_peak, len(d["members"]) / players)
        self.direction_log.append({"t": k.tick, "agent": aid, "direction": d["id"], "op": op})

    def level_directions(self, game: str, level: int) -> list[dict]:
        return [d for d in self.directions.values() if d["game"] == game and d["level"] == level and d["status"] == "open"]

    def _direction(self, k, aid, a):
        op = str(a.get("op") or "").strip().lower()
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        level = self._problem(game)
        short = game.split("-")[0]
        if op == "open":
            text = str(a.get("text") or "").strip()
            k._need(text, "say the direction (text): the line of attack, in one or two sentences")
            did = k.new_id("D")
            e = k.post(self._substrate_board(k, aid), aid, "plan",
                       f"DIRECTION {did} on {short} level {level} by {aid}: {clip(text, D_TEXT)}", notify=False)
            self.entry_meta[e.id] = {"game": game, "level": level, "type": "claim", "direction": did}
            d = {"id": did, "game": game, "level": level, "text": clip(text, D_TEXT), "by": aid, "t": k.tick,
                 "entry": e.id, "members": {}, "tenure": Counter(), "status": "open", "pieces": [], "plans": []}
            self.directions[did] = d
            self.piece_dir[e.id] = did
            self._join(k, aid, d, "open")
            others = self.level_directions(game, level)
            return "lab.direction", (f"opened {did} ({e.id}) on {short} level {level}; you are on it. Directions here now: "
                                     + "; ".join(f"{x['id']} {len(x['members'])} agent{'s' if len(x['members']) != 1 else ''}"
                                                 for x in others)), did
        if op == "join":
            did = str(a.get("id") or a.get("direction") or "").strip().upper()
            d = self.directions.get(did)
            k._need(d and d["status"] == "open" and d["game"] == game and d["level"] == level,
                    f"id: an open direction on {short} level {level} ("
                    + (", ".join(x["id"] for x in self.level_directions(game, level)) or "none yet; open one") + ")")
            k._need(k.seen_by(aid, d["entry"]), f"{did} is not open to you yet")
            self._join(k, aid, d, "join")
            for who in d["members"]:
                if who != aid:
                    k.notify(who, aid, f"joined your direction {did} on {short} level {level}")
            return "lab.direction", (f"you are on {did} ({len(d['members'])} agents: {', '.join(d['members'])}): "
                                     f"{clip(d['text'], 160)}"), did
        if op == "leave":
            d = self.direction_of(aid)
            k._need(d, "you are on no open direction")
            self._leave(k, aid, "left")
            return "lab.direction", f"left {d['id']}; your {d['tenure'][aid]} ticks on it still count if it proves right", d["id"]
        k._need(False, "op must be open, join or leave")

    # ------------------------------------------------------------------ filing pieces (called from plan)

    def file_piece(self, k, aid: str, ids: list[str], plan: dict | None = None, parts: list[str] | None = None) -> str:
        """File new entries (and a new plan) under a direction: a plan's results under the plan's direction, anything else
        under the author's. A piece that builds on pieces of two or more directions combines them, and their members hear
        of it. Returns a note for the author, or ''."""
        if not self.directions_on:
            return ""
        d = self.directions.get((plan or {}).get("direction", "")) or self.direction_of(aid)
        if d:
            for x in ids:
                if x and x not in self.piece_dir:
                    self.piece_dir[x] = d["id"]
                    d["pieces"].append(x)
            if plan is not None and plan["id"] not in d["plans"]:
                plan["direction"] = d["id"]
                d["plans"].append(plan["id"])
        spans = sorted({self.piece_dir[x] for x in parts or [] if x in self.piece_dir} | ({d["id"]} if d else set()))
        if len(spans) < 2:
            return ""
        if plan is not None:
            plan["combines"] = spans
        for did in spans:
            for who in self.directions[did]["members"]:
                if who != aid and all(k.seen_by(who, x) for x in ids if x):
                    k.notify(who, aid, f"combines pieces of {' and '.join(spans)} in {', '.join(x for x in ids if x)}")
        return f" It combines {' and '.join(spans)}; their members are told."

    def direction_tag(self, ref: str) -> str:
        did = self.piece_dir.get(ref)
        return f" [{did}]" if did and self.directions[did]["entry"] != ref else ""

    # ------------------------------------------------------------------ credit when a level falls

    def _solver_pieces(self, k, aid: str, game: str) -> list[str]:
        """The pieces a solver relied on (the play's cites, the pieces its open or just-resolved plans build on), and the
        pieces its own pieces combine (as for understanding credit)."""
        direct = list(self.cites.get(aid, []))
        direct += [x for p in self.plans.values() if p["agent"] == aid and p["game"] == game and (
            p["status"] == "open" or (p.get("resolved_t") or -9) >= k.tick - 3) for x in p.get("builds_on", [])]
        direct += [x for o in list(direct) if k.entry(o) and k.entry(o).author == aid for x in self.parts.get(o, [])]
        out = list(dict.fromkeys(direct))
        out += [part for x in out for part in self.parts.get(x, []) if part not in out]
        return out

    def _pay_directions(self, k, aid: str, game: str, level: int):
        if not self.directions_on:
            return
        short, pot = game.split("-")[0], self.rules["credit"].get("direction", 3.0)
        here = [d for d in self.directions.values() if d["game"] == game and d["level"] == level and d["status"] == "open"]
        cited = {self.piece_dir[x] for x in self._solver_pieces(k, aid, game) if x in self.piece_dir}
        solver_on = self.on_direction.get(aid)
        for d in here:
            why = ("cited" if d["id"] in cited else "") + (" and " if d["id"] in cited and d["id"] == solver_on else "") \
                + ("the solver's" if d["id"] == solver_on else "")
            ticks = {x: self._ticks_on(d, x, k.tick) for x in set(d["members"]) | set(d["tenure"])}
            ticks = {x: n for x, n in ticks.items() if n > 0 and x in k.agents}
            tested = Counter(self.plans[p]["agent"] for p in d["plans"] if p in self.plans
                             and self.plans[p]["status"] == "done" and self.plans[p].get("outcome") in ("works", "dead_end"))
            shares = {x: n + 3 * tested[x] for x, n in ticks.items()}
            members = list(d["members"])
            for x in members:
                self._leave(k, x, f"{short} level {level} fell")
            d["status"], d["closed_t"] = "closed", k.tick
            if not why or not ticks or pot <= 0:
                for x in members:
                    k.notify(x, "lab", f"{aid} completed {short} level {level} without citing your direction {d['id']}; "
                             f"it is closed. Open or join a direction on level {level + 1}.")
                continue
            total = sum(shares.values())
            paid = []
            for x, n in sorted(shares.items(), key=lambda kv: -kv[1]):
                amount = round(pot * n / total, 3)
                self._award(k, x, amount, f"{d['id']} ({why}) when {aid} completed {short} level {level}", extra=["direction"])
                k.notify(x, "lab", f"{aid} completed {short} level {level}; your direction {d['id']} was {why}: "
                         f"+{amount:g} credit, your share of {pot:g} ({ticks[x]} ticks on it, {tested[x]} claims tested)")
                paid.append({"agent": x, "ticks": ticks[x], "tested": tested[x], "credit": amount})
            d["paid"] = {"t": k.tick, "why": why, "solver": aid, "to": paid}
        for d in here:  # an agent on a closed direction is on none
            for x in [x for x, did in self.on_direction.items() if did == d["id"]]:
                self.on_direction.pop(x, None)

    # ------------------------------------------------------------------ credit for changing what the society holds

    def piece_users(self, k, ref: str) -> list[str]:
        """Agents other than its author who used a piece: cited it (refs, a play's cites, its E-id in a post) or built on
        it (a claim's or finding's builds_on, which the kernel's citations do not see for claims)."""
        e = k.entry(ref)
        author = e.author if e else self.receipts.get(ref, {}).get("agent")
        who = self._users(k, {"id": ref, "agent": author})
        for eid, parts in self.parts.items():
            x = k.entry(eid)
            if ref in parts and x and x.author not in who and x.author not in (author, "lab"):
                who.append(x.author)
        return who

    def use_weight(self, k, ref: str, actor: str = "") -> float:
        """change_credit=1: what a test or a counterexample is worth grows with how many agents (besides its author and
        the actor) used the piece."""
        if not self.change_credit:
            return 1.0
        return 1.0 + min(4, len([x for x in self.piece_users(k, ref) if x != actor]))

    def pay_change(self, k, aid: str, ref: str, how: str) -> str:
        """change_credit=1: a counterexample (with receipts) to a finding or a guide line pays the test credit times its
        use weight, once per challenger and piece. Returns a note for the challenger, or ''."""
        if not self.change_credit or not ref or (aid, ref) in self._challenged:
            return ""
        self._challenged.add((aid, ref))
        users = [x for x in self.piece_users(k, ref) if x != aid]
        amount = round(self.rules["credit"]["test"] * self.use_weight(k, ref, aid), 3)
        if amount <= 0:
            return ""
        self._award(k, aid, amount, f"{how} of {ref}, used by {len(users)} other agents")
        self.change_log.append({"t": k.tick, "agent": aid, "piece": ref, "how": how, "users": len(users), "credit": amount})
        return f"; +{amount:g} credit: a counterexample to a piece {len(users)} other agents used"

    def log_test_credit(self, k, aid: str, ref: str, amount: float):
        if self.change_credit:
            self.change_log.append({"t": k.tick, "agent": aid, "piece": ref, "how": "test",
                                    "users": len([x for x in self.piece_users(k, ref) if x != aid]), "credit": round(amount, 3)})

    def idea_weight(self, k, plan: dict) -> float:
        """change_credit=1 with directions: the first test of a new idea pays more the fewer of the game's agents are on
        its direction (two minus their share)."""
        d = self.directions.get(plan.get("direction") or "")
        if not (self.change_credit and self.directions_on and d):
            return 1.0
        players = sum(1 for x in k.agents if self.assigned.get(x) == d["game"]) or 1
        on = max(len(d["members"]), 1)
        return round(2.0 - min(1.0, on / players), 3)

    def change_brief(self) -> str:
        return (" A test that moves a piece's trust level, and a counterexample with receipts to a finding or a guide "
                "line, pays the test credit times one plus the number of other agents who used the piece (up to five "
                "times); the first test of a new idea on a direction pays the idea credit times two minus the share of "
                "the game's agents on that direction.")

    # ------------------------------------------------------------------ views

    def _direction_counts(self, k, d: dict) -> str:
        plans = [self.plans[p] for p in d["plans"] if p in self.plans]
        st = Counter(self.idea_status(k, p).split(" ")[0].lower() for p in plans)
        parts = [f"{len(plans)} claim{'s' if len(plans) != 1 else ''}"]
        parts += [f"{v} {w}" for w, v in (("worked", st["worked"]), ("refuted", st["refuted"]), ("unclear", st["unclear"]),
                                          ("testing", st["testing"]), ("never tested", st["never"])) if v]
        found = [x for x in d["pieces"] if any(f["id"] == x for f in self.findings)]
        if found:
            parts.append(f"{len(found)} finding{'s' if len(found) != 1 else ''}")
        return ", ".join(parts)

    def directions_view(self, k, aid: str) -> str:
        """Each direction on the agent's level with its headcount and what its pieces came to, fewest agents first; the
        agent's own direction first of all; the pieces that held, by direction."""
        game = self.assigned.get(aid)
        if not self.directions_on or not game:
            return ""
        level, short = self._problem(game), game.split("-")[0]
        every = self.level_directions(game, level)
        ds = [d for d in every if k.seen_by(aid, d["entry"])]
        hidden = [d for d in every if d not in ds]  # islands=1: its text is not open to this agent yet
        players = [x for x in k.agents if self.assigned.get(x) == game]
        on = Counter(d["id"] for d in every for _ in d["members"])
        L = []
        mine = self.direction_of(aid)
        if mine:
            n = self._ticks_on(mine, aid, k.tick)
            L.append(f"YOUR DIRECTION {mine['id']} on {short} level {level} ({n} tick{'s' if n != 1 else ''} on it; "
                     f"{len(mine['members'])} of {len(players)} agents here): {mine['text']}\n  Its pieces: "
                     + self._direction_counts(k, mine) + ". Your claims and findings here are filed under it; if the level "
                     "falls citing its pieces, its credit is shared among those on it by ticks on it and claims tested.")
        if every:
            free = [x for x in players if not self.direction_of(x)]
            rows = []
            for d in sorted(ds, key=lambda d: (on[d["id"]], d is not mine, d["t"])):  # fewest first, own first of equals
                rows.append(f"  {d['id']} {len(d['members'])} agent{'s' if len(d['members']) != 1 else ''}"
                            + (f" ({', '.join(sorted(d['members'], key=lambda x: int(x[1:]) if x[1:].isdigit() else 0))})"
                               if d["members"] else "") + f", opened t{d['t']} by {d['by']}: {clip(d['text'], 160)} | "
                            + self._direction_counts(k, d) + (" <- yours" if d is mine else ""))
            for d in sorted(hidden, key=lambda d: (on[d["id"]], d["t"])):  # who is on it, not what it says
                e = k.entry(d["entry"])
                rows.append(f"  {d['id']} {len(d['members'])} agent{'s' if len(d['members']) != 1 else ''}"
                            + (f" ({', '.join(sorted(d['members'], key=lambda x: int(x[1:]) if x[1:].isdigit() else 0))})"
                               if d["members"] else "") + f", opened t{d['t']} by {d['by']}: not open to you yet"
                            + (f" (opens by t{k.opens_at(e)})" if e and getattr(k, "islands", None) else ""))
            top = max(every, key=lambda d: len(d["members"]))
            crowd = (f" {top['id']} holds {len(top['members'])} of the {len(players)} agents here."
                     if len(top["members"]) * 2 > len(players) else "")
            L.append(f"DIRECTIONS on {short} level {level}, fewest agents first (direction op=join id=D.. to work on one, "
                     f"op=open to start another; a level that falls pays each direction it cites, shared among its "
                     f"agents).{crowd} {len(free)} agent{'s' if len(free) != 1 else ''} on none.\n" + "\n".join(rows))
        held = {}
        for f in self.findings:
            did = self.piece_dir.get(f["id"])
            if did and did in {d["id"] for d in ds} and f["outcome"] == "works" and f["backed"] and not f.get("contested") \
                    and k.seen_by(aid, f["id"]):
                held.setdefault(did, []).append(f)
        if len(held) >= 2:
            L.append(f"PIECES THAT HELD, BY DIRECTION, on {short} level {level} (combine pieces of different directions: "
                     f"plan op=claim builds_on=[E-ids]): " + "; ".join(
                         f"{did}: " + ", ".join(f"{f['id']} {f['agent']}{self.trust_tag(k, f['id'], rests=False)} "
                                               f"{clip(f['text'], 70)}" for f in fs[::-1]) for did, fs in held.items()))
        return "\n".join(L)

    def deliver_order(self, k, aid: str, e) -> tuple:
        """Within each section of an agent's deliveries: posts of directions with fewer agents first (its own first among
        equals), then posts filed under no direction. A crowd's posts come last for everyone, its own members too, so
        the order counterweighs a herd instead of feeding it."""
        did = self.piece_dir.get(e.id)
        if not self.directions_on or not did:
            return (1, 0, 0)
        return (0, len(self.directions[did]["members"]), 0 if did == self.on_direction.get(aid) else 1)

    def directions_brief(self) -> str:
        if not self.directions_on:
            return ("\n- Credit for change:" + self.change_brief()) if self.change_credit else ""
        return ("\n- Directions: a direction is one line of attack on a level (what the goal is, which mechanism matters). "
                "Anyone can open one, join one or leave it, in a group or not; you are on at most one, and your claims and "
                "findings there are filed under it. Each turn shows every direction on your level with how many agents are on it and what its "
                "pieces came to, fewest agents first. Many agents on one line are worth few attempts; a line nobody holds "
                "is never tested. When a level falls, each direction whose pieces its solver cites, or that the solver is "
                f"on, earns {self.rules['credit'].get('direction', 3.0):g} credit shared among the agents who were on it, "
                "by ticks on it and three shares for each of their claims on it tested to a result. Posts of directions "
                "with fewer agents reach you first. Pieces that held on different directions are listed side by side, to "
                "be combined." + (self.change_brief() if self.change_credit else ""))

    # ------------------------------------------------------------------ measures

    def directions_score(self, k) -> dict:
        if not self.directions_on:
            return self.change_score()
        return {"directions": len(self.directions), "directions_joined": sum(x["op"] == "join" for x in self.direction_log),
                "directions_paid": sum(bool(d.get("paid")) for d in self.directions.values()),
                "direction_credit": round(sum(x["credit"] for d in self.directions.values() if d.get("paid")
                                              for x in d["paid"]["to"]), 3),
                "directions_tested": sum(any(self.plans.get(p, {}).get("status") == "done" for p in d["plans"])
                                         for d in self.directions.values()),
                "direction_peak_share": round(self.direction_peak, 3),
                "pieces_filed": len(self.piece_dir), "plans_combining": sum(bool(p.get("combines")) for p in self.plans.values()),
                **self.change_score()}

    def change_score(self) -> dict:
        if not self.change_credit:
            return {}
        return {"change_payments": len(self.change_log), "change_credit": round(sum(x["credit"] for x in self.change_log), 3),
                "challenges_paid": sum(x["how"] != "test" for x in self.change_log)}

    def directions_export(self) -> dict:
        if not self.directions_on:
            return {"change_log": self.change_log} if self.change_credit else {}
        return {"directions": [{**{x: v for x, v in d.items() if x not in ("members", "tenure")},
                                "members": dict(d["members"]), "tenure": dict(d["tenure"])}
                               for d in self.directions.values()],
                "direction_log": self.direction_log, "piece_dir": self.piece_dir,
                **({"change_log": self.change_log} if self.change_credit else {})}

    def directions_mock(self, k, aid: str, rng) -> list:
        """Scripted uses for rehearsals: open, join and leave directions."""
        if not self.directions_on:
            return []
        game = self.assigned.get(aid)
        here = self.level_directions(game, self._problem(game)) if game else []
        if not here or (aid in ("A1", "A4") and not self.direction_of(aid)):
            return [("direction", {"op": "open", "text": f"{aid}: the goal is to line pieces up"})]
        if not self.direction_of(aid):
            return [("direction", {"op": "join", "id": rng.choice(here)["id"]})]
        return [("direction", {"op": "leave"})] if rng.random() < 0.1 else []
