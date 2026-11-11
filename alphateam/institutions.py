"""Institutions agents can use to build their own structure in the design world: leadership and contracts.

take_lead (world option leadership=1): an agent takes the lead of a scope, a set of blocks that includes its own. The lead
of a scope holds every knob whose users all lie inside it; where led scopes nest, a knob is held by the lead of the
smallest led scope covering all its users, so the lead of a larger scope holds exactly the knobs shared between the
smaller scopes inside it. A hierarchy appears when agents lead nested scopes: team leads hold their blocks' knobs, the
lead of a super-block holds the knobs its blocks share, and so on up. A lead can simulate the combined cost of its whole
scope, which no single agent can. A scope that already has a lead can be taken only by an agent with more credit
(when the world keeps durable credit); otherwise only after its lead steps down.

take_lead in team mode (leadership=team): like a CEO or PI. The lead founds a team (a group it directs) and keeps
working on details itself. Members join by hiring contract (paid a salary each credit review) or by joining the open
team. A member who takes the lead founds a sub-team nested inside its team, and a lead who is hired brings its team in as
a division, so teams nest recursively. A team holds the knobs whose users all lie in its members' blocks (the smallest
such team, so sub-teams hold their own area and the parent the knobs between them); its lead, and anyone the lead adds
as a delegate with protocol, may set them. Under economy=1 the lead earns project_mult times the fall in the team's
combined cost and pays the salaries.

contract (world option contracts=1, needs durable credit): a binding deal over a knob, paid in credit. An agent offers
another agent a payment to set a knob to a value and keep it there for some ticks; on acceptance the harness sets the
knob (the acceptor must be allowed to set it), locks it for the agreed ticks and moves the payment. This is the
Coasean remedy for the world's externality: the block that a shared knob hurts more can pay the other to settle it.
"""
from __future__ import annotations

import random
from collections import Counter

from .kernel import ActionError

LEAD_TOOL = {
    "name": "take_lead",
    "description": "Take the lead of a scope: a set of blocks that includes your own (default: just your block). The lead "
                   "of a scope holds every knob used only by blocks in it; where led scopes nest, each knob is held by the "
                   "lead of the smallest led scope covering all its users, so leads of larger scopes hold the knobs shared "
                   "between smaller ones. A lead can simulate its whole scope's combined cost (simulate with area=true). A "
                   "scope that already has a lead can be taken only by an agent with more credit. op step_down gives up a lead; op appoint "
                   "(with to) makes another agent the lead of a scope strictly inside one you lead (delegation).",
    "parameters": {"type": "object", "properties": {
        "scope": {"type": "array", "items": {"type": "string"}, "description": "block labels, e.g. [\"B3\", \"B7\"]"},
        "op": {"type": "string", "enum": ["take", "step_down", "appoint"]},
        "to": {"type": "string", "description": "appoint: the agent to lead the scope"}}}}

CONTRACT_TOOL = {
    "name": "contract",
    "description": "A binding deal over a knob, paid in credit. offer: ask another agent (to) to set a knob to a value and "
                   "keep it there for some ticks (1-10), for a payment from your credit (pay). accept or decline: the agent "
                   "offered to answers with the contract id. On accept the harness sets the knob, locks it for the agreed "
                   "ticks (nobody may change it) and moves the payment. cancel: withdraw your open offer.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["offer", "accept", "decline", "cancel"]},
        "to": {"type": "string"}, "knob": {"type": "string"}, "value": {"type": "integer"},
        "ticks": {"type": "integer"}, "pay": {"type": "number"}, "id": {"type": "string"}},
        "required": ["op"]}}


TEAM_TOOL = {
    "name": "take_lead",
    "description": "Lead a team, like a CEO or PI: found a team (a group you direct) and keep working on details yourself. "
                   "If you are already in a team, the team you found is a sub-team nested inside it. Grow it by hiring "
                   "(contract op=hire, paid a salary each credit review) or let agents join it. Your team holds every knob "
                   "used only by its members' blocks (sub-teams hold their own area, the parent the knobs between them); "
                   "you, and anyone you add as a delegate with protocol, may set them; you can simulate the team's combined "
                   "cost (simulate with area=true). op step_down hands the team to a member (to) or dissolves it.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["found", "step_down"]},
        "name": {"type": "string"}, "to": {"type": "string"}}}}


def _num(x: str) -> int:
    return int(x[1:])


class Institutions:
    """Mixed into DesignWorld."""

    def _init_institutions(self, opts: dict):
        self.team_mode = str(opts.get("leadership", "")).lower() == "team"
        self.leadership = str(opts.get("leadership", "")).lower() in ("1", "true", "yes") or self.team_mode
        self.teams: dict[str, dict] = {}  # team group id -> {"lead", "parent", "t"}
        self.jobs: dict[str, dict] = {}  # member (agent, or a sub-team's group) -> {"team", "salary", "until", "id"}
        self.team_rights: dict[int, str] = {}
        # scope_teams=1 (with leadership=1): every led scope has a team group: its lead directs it (sole delegate), its
        # members are the agents of its blocks not inside a smaller led scope plus the teams of the maximal led scopes
        # inside it, so led scopes nest into teams of teams; the team's memory board is its shared substrate
        self.scope_teams = str(opts.get("scope_teams", "")).lower() in ("1", "true", "yes")
        self.scope_groups: dict[frozenset, str] = {}
        self.contracts_on = str(opts.get("contracts", "")).lower() in ("1", "true", "yes")
        # chair=1 (with leadership=team): the chaired control arm. Before the run begins the harness founds one team with
        # every agent in it and makes one agent its lead: an imposed coordinator, to compare with societies where nobody
        # is placed in charge (alphateam/paired.py, frame arm="chaired"; DESIGN.md section 20)
        self.chair = str(opts.get("chair", "")).lower() in ("1", "true", "yes")
        self.chairman: str | None = None
        self.chair_team: str | None = None
        if self.chair and not self.team_mode:
            raise ValueError("chair=1 needs leadership=team")
        self.leads: dict[frozenset, str] = {}  # scope (block indices) -> lead
        self.lead_rights: dict[int, str] = {}
        self.lead_log: list[dict] = []
        self.contracts: dict[str, dict] = {}
        self.locks: dict[int, tuple] = {}  # knob -> (contract id, last locked tick)
        extra = [t for t, on in ((TEAM_TOOL if self.team_mode else LEAD_TOOL, self.leadership),
                                 (CONTRACT_TOOL, self.contracts_on)) if on]
        if extra:
            self.tool_names = tuple(self.tool_names) + tuple(t["name"] for t in extra)
            self.TOOLS = list(self.TOOLS) + extra

    # ------------------------------------------------------------------ leadership

    def _scope_label(self, scope) -> str:
        return "{" + ",".join(sorted((self.block_label[b] for b in scope), key=lambda s: int(s[1:]))) + "}"

    def _take_lead(self, k, aid, a):
        if self.team_mode:
            return self._team_lead(k, aid, a)
        if str(a.get("op") or "take") == "step_down":
            mine = [s for s, h in self.leads.items() if h == aid]
            k._need(mine, "you lead no scope")
            for s in mine:
                del self.leads[s]
                self.lead_log.append({"t": k.tick, "op": "step_down", "agent": aid, "scope": sorted(s)})
            self._recompute_lead_rights()
            self._sync_scope_teams(k)
            return "lab.step_down", f"stepped down from {', '.join(self._scope_label(s) for s in mine)}", None
        op = str(a.get("op") or "take")
        mine = set(self.instruments.get(aid, []))
        k._need(mine, "you have no block")
        labels = a.get("scope") or [self.block_label[b] for b in mine]
        if isinstance(labels, str):
            labels = [x.strip() for x in labels.replace(",", " ").split() if x.strip()]
        index = {lab: b for b, lab in enumerate(self.block_label)}
        bad = [x for x in labels if str(x).upper() not in index]
        k._need(not bad, f"no block {', '.join(map(str, bad))} (blocks are B1..B{len(self.blocks)})")
        scope = frozenset(index[str(x).upper()] for x in labels)
        if op == "appoint":  # delegation: hand a scope inside your own to another agent
            to = str(a.get("to") or "").strip().upper()
            k._need(to in self.instruments and to != aid, "appoint another agent (to)")
            k._need(any(scope < s for s, h in self.leads.items() if h == aid), "you can appoint only to a scope strictly "
                    "inside one you lead")
            k._need(scope & set(self.instruments[to]), f"{to}'s block is not in that scope")
            prev = self.leads.get(scope)
            self.leads[scope] = to
            self.lead_log.append({"t": k.tick, "op": "appoint", "agent": aid, "to": to, "scope": sorted(scope), "from": prev})
            self._recompute_lead_rights()
            self._sync_scope_teams(k)
            k.notify(to, aid, f"{aid} appointed you lead of {self._scope_label(scope)}")
            return "lab.appoint", f"appointed {to} lead of {self._scope_label(scope)}", self._scope_label(scope)
        k._need(scope & mine, "a scope you lead must include your own block")
        held = self.leads.get(scope)
        if held and held != aid:
            if getattr(self, "durable", False):
                mine_c, theirs = self.effective.get(aid, 0.0), self.effective.get(held, 0.0)
                k._need(mine_c > theirs, f"{self._scope_label(scope)} is led by {held}, who has more credit "
                                         f"({theirs:+.3f} vs your {mine_c:+.3f})")
                k.notify(held, aid, f"{aid} took over the lead of {self._scope_label(scope)} with more credit")
            else:
                raise ActionError(f"{self._scope_label(scope)} is led by {held}; it must step down first")
        self.leads[scope] = aid
        self.lead_log.append({"t": k.tick, "op": "take", "agent": aid, "scope": sorted(scope), "from": held})
        self._recompute_lead_rights()
        self._sync_scope_teams(k)
        knobs = [self.name[i] for i, h in self.lead_rights.items() if h == aid]
        for b in scope:
            for other, bs in self.instruments.items():
                if b in bs and other != aid:
                    k.notify(other, aid, f"{aid} now leads {self._scope_label(scope)} (your block is inside it)")
        return "lab.take_lead", (f"you now lead {self._scope_label(scope)} and hold {len(knobs)} knobs: "
                                 f"{', '.join(knobs[:20])}"), self._scope_label(scope)

    def _sync_scope_teams(self, k):
        if not self.scope_teams:
            return
        from .kernel import Board, Group
        for scope, gid in list(self.scope_groups.items()):  # scopes no longer led lose their team
            if scope not in self.leads:
                if k.groups[gid].dissolved is None:
                    k._dissolve(gid)
                del self.scope_groups[scope]
        for scope in sorted(self.leads, key=len):  # smaller scopes first, so their teams exist before their parents'
            lead = self.leads[scope]
            gid = self.scope_groups.get(scope)
            if gid is None or k.groups[gid].dissolved is not None:
                gid = k.new_id("G")
                k.groups[gid] = Group(gid, f"Team {self._scope_label(scope)}", f"Blocks {self._scope_label(scope)}.", lead,
                                      k.tick, open=False, delegates={lead},
                                      protocol="The lead directs the team and decides its knobs; members report evidence "
                                               "and carry out the lead's directions; the team board is its shared memory.")
                k.boards[f"mem:{gid}"] = Board(f"mem:{gid}", gid)
                k.boards[f"pub:{gid}"] = Board(f"pub:{gid}", gid)
                self.scope_groups[scope] = gid
            k.groups[gid].delegates = {lead}
            inner = [t for t in self.leads if t < scope]
            maximal = [t for t in inner if not any(t < u for u in inner)]
            covered = set().union(*maximal) if maximal else set()
            want = {self.scope_groups[t] for t in maximal} | {lead} | {
                a for a, bs in self.instruments.items() if set(bs) & (scope - covered) and k.agents[a].active}
            have = set(k.groups[gid].members)
            for m in have - want:
                k._remove_member(gid, m)
            for m in want - have:
                if m != gid and not (m in k.groups and gid in k.agents_inside_groups(m)):
                    k._add_member(gid, m)
                    if m in self.instruments and m != lead:
                        k.notify(m, lead, f"you are in team {gid} led by {lead} ({self._scope_label(scope)}); "
                                          f"its board mem:{gid} is the team's shared memory")

    def _recompute_lead_rights(self):
        for i, h in self.lead_rights.items():
            if self.rights.get(i) == h:
                del self.rights[i]
        self.lead_rights = {}
        for i in range(self.D):
            users = set(self.users[i])
            cover = [s for s in self.leads if users <= s]
            if cover:
                s = min(cover, key=lambda s: (len(s), sorted(s)))
                self.rights[i] = self.leads[s]
                self.lead_rights[i] = self.leads[s]

    # ------------------------------------------------------------------ teams (CEO / PI)

    def _install_chair(self, k):
        """The chaired control arm: one team with every agent in it, led by one agent, before the run begins."""
        agents = sorted(self.instruments, key=_num)
        self.chairman = random.Random(f"chair:{getattr(self, 'world_seed', 0)}:{self.D}:{len(agents)}").choice(agents)
        _, _, gid = self._team_lead(k, self.chairman, {"op": "found", "name": "The whole team"})
        self.chair_team = gid
        for a in agents:
            if a != self.chairman:
                k._add_member(gid, a)
        self._recompute_team_rights(k)
        self.lead_log.append({"t": k.tick, "op": "chair", "agent": self.chairman, "team": gid})

    def _team_of(self, k, member: str) -> str | None:
        """The innermost team a member (agent or team) belongs to directly."""
        direct = [g for g in k.member_of.get(member, ()) if g in self.teams and k.groups[g].dissolved is None]
        return min(direct, key=lambda g: len(k.agents_inside(g))) if direct else None

    def _team_lead(self, k, aid, a):
        from .kernel import Board, Group
        led = [g for g, t in self.teams.items() if t["lead"] == aid and k.groups[g].dissolved is None]
        if str(a.get("op") or "found") == "step_down":
            k._need(led, "you lead no team")
            g = led[0]
            to = str(a.get("to") or "").strip().upper()
            if to:
                k._need(k.inside(to, g) and to in self.instruments, f"{to} is not in {g}")
                self.teams[g]["lead"] = to
                k.groups[g].delegates = {to}
                k.notify(to, aid, f"{aid} handed you the lead of team {g}")
                msg = f"handed {g} to {to}"
            else:
                k._dissolve(g)
                msg = f"dissolved {g}"
            self.lead_log.append({"t": k.tick, "op": "step_down", "agent": aid, "team": g, "to": to or None})
            self._recompute_team_rights(k)
            return "lab.step_down", msg, g
        if led:
            raise ActionError(f"you already lead {led[0]}")
        parent = self._team_of(k, aid)
        gid = k.new_id("G")
        name = str(a.get("name") or f"Team of {aid}")[:60]
        k.groups[gid] = Group(gid, name, f"Led by {aid}.", aid, k.tick, open=True, delegates={aid},
                              protocol=f"{aid} leads this team and decides its knobs; members follow the lead's directions.")
        k.boards[f"mem:{gid}"] = Board(f"mem:{gid}", gid)
        k.boards[f"pub:{gid}"] = Board(f"pub:{gid}", gid)
        k._add_member(gid, aid)
        if parent:  # a member who takes the lead founds a sub-team nested in its team
            k._remove_member(parent, aid)
            k._add_member(parent, gid)
        self.teams[gid] = {"lead": aid, "parent": parent, "t": k.tick}
        self.lead_log.append({"t": k.tick, "op": "found", "agent": aid, "team": gid, "parent": parent})
        self._recompute_team_rights(k)
        return "lab.take_lead", (f"you lead team {gid}" + (f", a sub-team of {parent}" if parent else "") +
                                 "; hire members with contract op=hire or let agents join"), gid

    def team_scope(self, k, g: str) -> set[int]:
        return {b for a in k.agents_inside(g) for b in self.instruments.get(a, [])}

    def _recompute_team_rights(self, k):
        for i, h in self.team_rights.items():
            if self.rights.get(i) == h:
                del self.rights[i]
        self.team_rights = {}
        live = [g for g in self.teams if k.groups[g].dissolved is None]
        scopes = {g: self.team_scope(k, g) for g in live}
        for i in range(self.D):
            users = set(self.users[i])
            cover = [g for g in live if users <= scopes[g]]
            if len(users) < 2:  # under a chair, a block's own knobs stay with its agents: the chair's team holds what is shared
                cover = [g for g in cover if g != getattr(self, "chair_team", None)]
            if cover:
                g = min(cover, key=lambda g: (len(scopes[g]), g))
                self.rights[i] = g
                self.team_rights[i] = g

    def team_depth(self, k) -> int:
        live = [g for g in self.teams if k.groups[g].dissolved is None]
        return max((k.depth(g) for g in live), default=0)

    def lead_depth(self) -> int:
        """Longest chain of strictly nested led scopes (1 = leads exist but none nests in another)."""
        scopes = sorted(self.leads, key=len)
        depth = {}
        for s in scopes:
            depth[s] = 1 + max((depth[t] for t in depth if t < s), default=0)
        return max(depth.values(), default=0)

    def superior(self, aid: str) -> str | None:
        mine = [s for s, h in self.leads.items() if h == aid]
        if not mine:
            return None
        big = max(mine, key=len)
        above = [s for s in self.leads if big < s]
        return self.leads[min(above, key=len)] if above else None

    def _simulate_area(self, k, aid, a):
        if self.team_mode:
            led = [g for g, t in self.teams.items() if t["lead"] == aid and k.groups[g].dissolved is None]
            k._need(led, "only a team lead can simulate its team's area (take_lead first)")
            scope = sorted(self.team_scope(k, led[0]))
        else:
            mine = [s for s, h in self.leads.items() if h == aid]
            k._need(mine, "only a lead can simulate an area (take_lead first)")
            scope = sorted(max(mine, key=len))
        budget = self.sim_budget(aid)
        k._need(self.sims[(aid, k.tick)] < budget, f"you have used all {budget} simulations this tick")
        changes = a.get("changes") or {}
        k._need(isinstance(changes, dict) and len(changes) <= self.max_changes, "changes must be an object of at most "
                f"{self.max_changes} knob values")
        x2 = self.x.copy()
        shown = []
        for ref, v in changes.items():
            i = self._knob(ref)
            x2[i] = int(v)
            shown.append(f"{self.name[i]}={int(v)}")
        board = k.board_id(aid, a.get("to") or self.default_to.get(aid, "private"))
        k._need(k.can_write(aid, board), f"you cannot write to {board}")
        self.sims[(aid, k.tick)] += 1
        self.used[aid] += 1
        if shown and getattr(self, "evidence_share", 0):
            self.evidence.append({"t": k.tick, "agent": aid, "knobs": [self._knob(r) for r in changes], "board": board})
        now = sum(self.block_cost(b, self.x) for b in scope)
        text = f"Area simulation of {self._scope_label(scope)} by {aid} at t{k.tick}: combined cost now {now:.1f}"
        if shown:
            hyp = sum(self.block_cost(b, x2) for b in scope)
            text += f"; with {', '.join(shown)}: {hyp:.1f} ({hyp - now:+.1f})"
        k.post(board, aid, "data", text + ".", notify=False)
        return "lab.simulate_area", text + f" [simulations left this tick: {budget - self.sims[(aid, k.tick)]}]", None

    # ------------------------------------------------------------------ contracts

    def _contract(self, k, aid, a):
        op = str(a.get("op") or "")
        if op == "hire":
            k._need(self.team_mode, "hiring needs teams (take_lead)")
            led = [g for g, t in self.teams.items() if t["lead"] == aid and k.groups[g].dissolved is None]
            k._need(led, "found a team first (take_lead)")
            to = str(a.get("to") or "").strip().upper()
            k._need(to in self.instruments and to != aid, "hire another agent (to)")
            try:
                salary, ticks = float(a.get("pay") or a.get("salary") or 0), int(a.get("ticks") or 9)
            except (TypeError, ValueError):
                raise ActionError("pay (salary per credit review) is a number, ticks an integer") from None
            k._need(salary >= 0 and 1 <= ticks <= 40, "salary >= 0, ticks 1-40")
            cid = f"C{len(self.contracts) + 1}"
            self.contracts[cid] = {"id": cid, "t": k.tick, "from": aid, "to": to, "kind": "hire", "team": led[0],
                                   "salary": salary, "ticks": ticks, "status": "open", "knob": None}
            k.notify(to, aid, f"contract {cid}: {aid} offers you a job in team {led[0]} for {salary} credit per credit "
                              f"review over {ticks} ticks (contract op=accept id={cid}, or decline)")
            return "lab.contract_hire", f"offered {cid} (a job in {led[0]}) to {to}", cid
        if op == "quit":
            k._need(self.team_mode, "quit applies to team jobs")
            mine = self._team_of(k, aid)
            k._need(mine, "you are in no team")
            member = aid
            led = [g for g, t in self.teams.items() if t["lead"] == aid and k.groups[g].dissolved is None]
            if led and self.teams[led[0]]["parent"]:  # a sub-team lead quits with its team
                member, mine = led[0], self.teams[led[0]]["parent"]
            k._need(self.teams.get(mine, {}).get("lead") != aid, "a lead steps down instead (take_lead op=step_down)")
            k._remove_member(mine, member)
            self.jobs.pop(member, None)
            if member in self.teams:
                self.teams[member]["parent"] = None
            self._recompute_team_rights(k)
            return "lab.contract_quit", f"left {mine}", mine
        if op == "offer":
            to = str(a.get("to") or "").strip().upper()
            k._need(to in self.instruments and to != aid, "offer to another agent (to)")
            i = self._knob(a.get("knob"))
            try:
                v, ticks, pay = int(a.get("value")), int(a.get("ticks") or 3), float(a.get("pay") or 0)
            except (TypeError, ValueError):
                raise ActionError("value and ticks are integers, pay is a number") from None
            k._need(0 <= v <= 10 and 1 <= ticks <= 10 and pay >= 0, "value 0-10, ticks 1-10, pay >= 0")
            have = self.dcredit.get(aid, [0.0])[0]
            k._need(have >= pay, f"you have {have:+.3f} credit, less than the payment {pay}")
            cid = f"C{len(self.contracts) + 1}"
            self.contracts[cid] = {"id": cid, "t": k.tick, "from": aid, "to": to, "knob": i, "value": v, "ticks": ticks,
                                   "pay": pay, "status": "open"}
            k.notify(to, aid, f"contract {cid}: {aid} offers {pay} credit if you set {self.name[i]}={v} and keep it "
                              f"{ticks} ticks (contract op=accept id={cid}, or decline)")
            return "lab.contract_offer", f"offered {cid} to {to}", cid
        c = self.contracts.get(str(a.get("id") or "").strip().upper())
        k._need(c, "no such contract (id)")
        k._need(c["status"] == "open", f"{c['id']} is {c['status']}")
        if op == "cancel":
            k._need(c["from"] == aid, "only the offerer can cancel")
            c["status"] = "cancelled"
            return "lab.contract_cancel", f"cancelled {c['id']}", c["id"]
        k._need(c["to"] == aid, f"{c['id']} was offered to {c['to']}")
        if op == "decline":
            c["status"] = "declined"
            k.notify(c["from"], aid, f"{aid} declined {c['id']}")
            return "lab.contract_decline", f"declined {c['id']}", c["id"]
        k._need(op == "accept", "op is offer, hire, accept, decline, cancel or quit")
        if c.get("kind") == "hire":
            g = c["team"]
            k._need(k.groups[g].dissolved is None, f"{g} no longer exists")
            led = [t for t, v in self.teams.items() if v["lead"] == aid and k.groups[t].dissolved is None]
            member = led[0] if led else aid  # a lead who is hired brings its team in as a division
            k._need(member != g and not k.inside(aid, g), f"you are already in {g}")
            old = self._team_of(k, member)
            if old:
                k._remove_member(old, member)
            k._add_member(g, member)
            if member in self.teams:
                self.teams[member]["parent"] = g
            self.jobs[member] = {"team": g, "salary": c["salary"], "until": k.tick + c["ticks"], "id": c["id"], "agent": aid}
            c.update(status="accepted", accepted=k.tick)
            self._recompute_team_rights(k)
            k.notify(c["from"], aid, f"{aid} accepted {c['id']} and joined {g}" + (f" with its team {member}" if led else ""))
            return "lab.contract_accept", f"joined {g}" + (f" bringing your team {member} as a division" if led else ""), c["id"]
        have = self.dcredit.get(c["from"], [0.0])[0]
        k._need(have >= c["pay"], f"{c['from']} no longer has the credit to pay ({have:+.3f})")
        i = c["knob"]
        if self.x[i] != c["value"]:
            self._set_knob(k, aid, {"knob": self.name[i], "value": c["value"]})  # checks rights and locks
        else:
            self._check_lock(k, i)
        self.locks[i] = (c["id"], k.tick + c["ticks"])
        self.dcredit.setdefault(c["from"], [0.0, 0])[0] -= c["pay"]
        self.dcredit.setdefault(aid, [0.0, 0])[0] += c["pay"]
        c.update(status="accepted", accepted=k.tick)
        k.notify(c["from"], aid, f"{aid} accepted {c['id']}: {self.name[i]}={c['value']} locked until t{k.tick + c['ticks']}")
        return "lab.contract_accept", (f"accepted {c['id']}: set {self.name[i]}={c['value']}, locked until "
                                       f"t{k.tick + c['ticks']}; received {c['pay']} credit"), c["id"]

    def _institutions_tick(self, k):
        if self.team_mode:
            self._recompute_team_rights(k)

    def _team_rewards(self, k, since: int, gap: float, notes) -> None:
        """At a credit review: the lead of each team of 2+ blocks earns project_mult x the fall in its combined cost;
        salaries move from leads to members."""
        for g, t in self.teams.items():
            if k.groups[g].dissolved is not None:
                continue
            scope = self.team_scope(k, g)
            if getattr(self, "economy", False) and len(scope) >= 2:
                r = self.project_mult * sum(self.block_cost(b, self.x_reviewed) - self.block_cost(b, self.x) for b in scope) / gap
                self.dcredit.setdefault(t["lead"], [0.0, 0])[0] += r
                notes[t["lead"]].append(f"as lead of {g} you earned {r:+.3f} ({self.project_mult:g}x the fall in its cost)")
        for member, j in list(self.jobs.items()):
            if k.tick > j["until"] or not k.inside(j["agent"], j["team"]):
                self.jobs.pop(member)
                continue
            lead = self.teams[j["team"]]["lead"]
            pay = min(j["salary"], max(0.0, self.dcredit.get(lead, [0.0])[0]))
            self.dcredit.setdefault(lead, [0.0, 0])[0] -= pay
            self.dcredit.setdefault(j["agent"], [0.0, 0])[0] += pay
            notes[j["agent"]].append(f"salary from {lead} ({j['team']}): {pay:+.3f}")

    def _check_lock(self, k, i: int):
        lock = self.locks.get(i)
        if lock and k.tick <= lock[1]:
            raise ActionError(f"{self.name[i]} is locked by contract {lock[0]} until t{lock[1]}")

    # ------------------------------------------------------------------ views and records

    def _institutions_brief(self) -> str:
        out = ""
        if self.team_mode:
            out += ("\n- Teams: take_lead founds a team you lead, like a CEO or PI: you direct it and still work on "
                    "details yourself. If you are in a team when you take the lead, yours becomes a sub-team inside it, so "
                    "teams nest. Grow a team by hiring (contract op=hire with a salary per credit review) or by letting "
                    "agents join; members can quit (contract op=quit). A team holds every knob used only by its members' "
                    "blocks (the smallest such team: sub-teams hold their own area, the parent the knobs between them); "
                    "its lead and any delegates the lead adds with protocol may set them. A lead can simulate its team's "
                    "combined cost (simulate with area=true). view_design shows which blocks use a knob.")
            if self.chair and self.chairman:
                out += (f"\n- This society starts as one team. Before the run began, {self.chairman} was made its lead and "
                        "every agent a member. The team holds every knob that several of its members' blocks share, so only "
                        "its lead, and anyone the lead adds as a delegate with protocol, can set those; the lead or a "
                        "delegate can also hand them to a member with a task (delegate) whose scope names them. A knob "
                        "that only one block uses is not held by the team.")
        elif self.leadership:
            out += ("\n- Leadership: take_lead makes you the lead of a scope (a set of blocks that includes yours). The lead "
                    "of a scope holds every knob used only by blocks in it; where led scopes nest, a knob is held by the lead "
                    "of the smallest led scope covering all its users, so leads of larger scopes hold the knobs shared "
                    "between smaller scopes. A lead can simulate its scope's combined cost (simulate with area=true). A led "
                    "scope can be taken over only by an agent with more credit. view_design shows which blocks use a knob." +
                    (" Every led scope is a team, like a lab under its PI: its members are the agents of its blocks (a led "
                     "scope inside it joins as a sub-team), its lead directs it and still works on details, and its board "
                     "mem:G is the team's shared memory for directions and evidence." if self.scope_teams else ""))
        if getattr(self, "local", False):
            out += ("\n- Local communication: agents cannot post on the global board (design reviews still appear there). "
                    "Information travels through links, groups, shared boards and leads.")
        if self.contracts_on:
            out += ("\n- Contracts: contract lets you pay another agent credit to set a knob to a value and keep it there "
                    "for some ticks; on acceptance the harness sets the knob, locks it and moves the payment.")
        return out

    def _institutions_observe(self, k, aid: str) -> str:
        bits = []
        if self.team_mode:
            led = [g for g, t in self.teams.items() if t["lead"] == aid and k.groups[g].dissolved is None]
            for g in led:
                subs = [m for m in k.groups[g].members if m in self.teams]
                bits.append(f"You lead team {g} ({len(k.agents_inside(g))} agents, blocks "
                            f"{self._scope_label(self.team_scope(k, g))}, {sum(h == g for h in self.team_rights.values())} knobs"
                            + (f", sub-teams {', '.join(subs)}" if subs else "") +
                            (f"; it is a division of {self.teams[g]['parent']}" if self.teams[g]["parent"] else "") + ")")
            mine = self._team_of(k, aid)
            if mine and not led:
                bits.append(f"You are in team {mine}, led by {self.teams[mine]['lead']}")
        mine = [s for s, h in self.leads.items() if h == aid]
        if mine:
            sup = self.superior(aid)
            bits.append(f"You lead {', '.join(self._scope_label(s) for s in mine)}" + (f"; your superior is {sup}" if sup else ""))
        b = self.instruments.get(aid, [])
        over = sorted({h for s, h in self.leads.items() if set(b) & s and h != aid})
        if over:
            bits.append(f"Leads over your block: {', '.join(over)}")
        offers = [c for c in self.contracts.values() if c["to"] == aid and c["status"] == "open"]
        if offers:
            bits.append("Contract offers to you: " + "; ".join(
                f"{c['id']} from {c['from']}: {self.name[c['knob']]}={c['value']} for {c['ticks']} ticks, pay {c['pay']}"
                for c in offers[:4]))
        return (" " + ". ".join(bits) + ".") if bits else ""

    def _institutions_score(self) -> dict:
        if self.team_mode:
            live = [g for g in self.teams if self._k.groups[g].dissolved is None] if getattr(self, "_k", None) else []
            return {"teams": len(live), "team_depth": self.team_depth(self._k) if live else 0,
                    "team_knobs": len(self.team_rights), "in_teams": round(sum(
                        any(self._k.inside(a, g) for g in live) for a in self.instruments) / max(1, len(self.instruments)), 4),
                    "jobs": len(self.jobs), "contracts_offered": len(self.contracts),
                    "contracts_accepted": sum(c["status"] == "accepted" for c in self.contracts.values())}
        return {"leads": len(self.leads), "lead_depth": self.lead_depth(), "lead_agents": len(set(self.leads.values())),
                "lead_knobs": len(self.lead_rights),
                "contracts_offered": len(self.contracts),
                "contracts_accepted": sum(c["status"] == "accepted" for c in self.contracts.values())}

    def _institutions_export(self) -> dict:
        return {"teams": self.teams, "jobs": self.jobs,
                "leads": [{"scope": [self.block_label[b] for b in sorted(s)], "lead": h} for s, h in self.leads.items()],
                "lead_log": self.lead_log,
                "contracts": [dict(c, knob=self.name[c["knob"]]) for c in self.contracts.values()],
                "lead_holders": Counter(self.lead_rights.values())}
