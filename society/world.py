"""The kernel of the agent society.

Only the action space and a small fixed kernel are designed here: substrates
(boards) with permissions, an audit log, a resource ledger, and the rules that
make actions well defined. Everything organizational (who teams up, how deep
the hierarchy gets, how a team decides, which new actions exist) is left to
the agents.

Structure is G_t = (V_t, E_t, H_t):
  V  agents and teams (a team plus its sub-teams is a superagent)
  E  links, board shares, critiques, assignments (lateral graph)
  H  containment: each agent has at most one home team, each team at most one
     parent, so H is a forest and every subtree is a well-defined superagent.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

PRIMITIVES = ("read", "write", "share", "summarize", "reflect", "connect", "plan", "design_action")
ENTRY_KINDS = ("note", "result", "question", "proposal", "plan", "summary", "critique", "reflection", "data", "decision")
CONNECT_OPS = ("link", "unlink", "join", "leave", "form_team", "invite", "nest", "split", "merge",
               "dissolve", "recharter", "govern", "admit", "expel", "vote")
PLAN_OPS = ("decompose", "assign", "claim", "release", "submit", "verify", "revise")
DESIGN_OPS = ("define", "revise", "adopt", "drop")
RULES = ("any", "rep", "majority", "consensus")
STRUCTURAL = {"form_team", "join", "leave", "nest", "split", "merge", "dissolve", "expel", "admit"}
MOTION_TTL = 3
MAX_TEXT = 4000
MENTION = re.compile(r"@([AT]\d+)\b")
ACTION_NAME = re.compile(r"^[a-z][a-z0-9_]{2,30}$")


class ActionError(Exception):
    """An action the kernel refuses; the message goes back to the agent."""


def clip(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [s for s in re.split(r"[,\s]+", x) if s]
    return [str(s).strip() for s in x if str(s).strip()]


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Entry:
    id: str
    board: str
    author: str
    tick: int
    kind: str
    content: str
    refs: list[str] = field(default_factory=list)
    via: str | None = None


@dataclass
class Board:
    id: str
    owner: str
    entries: list[Entry] = field(default_factory=list)
    readers: set[str] = field(default_factory=set)
    writers: set[str] = field(default_factory=set)


@dataclass
class Agent:
    id: str
    name: str
    persona: str
    team: str | None = None
    links: set[str] = field(default_factory=set)
    reflections: list[str] = field(default_factory=list)
    actions: set[str] = field(default_factory=set)
    inbox: list[dict] = field(default_factory=list)
    influence: Counter = field(default_factory=Counter)  # team id -> coordination acts in that team


@dataclass
class Team:
    id: str
    name: str
    charter: str
    founder: str
    parent: str | None
    created: int
    rule: str = "any"
    reps: set[str] = field(default_factory=set)
    open: bool = True
    dissolved: int | None = None


@dataclass
class Block:
    id: str
    title: str
    spec: str
    parent: str | None
    created_by: str
    created: int
    owner: str | None = None
    status: str = "open"  # open | claimed | submitted | done
    deliverable: str = ""
    submitted_by: str | None = None
    reviews: list[dict] = field(default_factory=list)


@dataclass
class CustomAction:
    name: str
    description: str
    procedure: str
    params: list[str]
    author: str
    created: int
    version: int = 1
    uses: int = 0


@dataclass
class Motion:
    id: str
    team: str
    proposer: str
    action: str
    args: dict
    summary: str
    created: int
    yes: set[str] = field(default_factory=set)
    no: set[str] = field(default_factory=set)
    status: str = "open"  # open | passed | failed | expired


class World:
    def __init__(self, mission: str, people: list[dict], max_actions: int = 4,
                 allowed: dict[str, set[str] | None] | None = None):
        self.mission = mission
        self.max_actions = max_actions
        # Which primitives (and which ops inside them) exist in this condition.
        self.allowed = allowed or {p: None for p in PRIMITIVES}
        self.tick = 0
        self.total_ticks = 0
        self.token_budget: int | None = None
        self.agents: dict[str, Agent] = {}
        self.teams: dict[str, Team] = {}
        self.boards: dict[str, Board] = {"global": Board("global", "society")}
        self.blocks: dict[str, Block] = {"M0": Block("M0", "Mission", mission, None, "society", 0)}
        self.custom: dict[str, CustomAction] = {}
        self.motions: dict[str, Motion] = {}
        self.events: list[dict] = []
        self.snapshots: list[dict] = []
        self.ledger: dict[str, Counter] = defaultdict(Counter)  # agent -> category -> tokens
        self.counts = Counter()  # society-level action counts by category
        self._ids = Counter()
        self.summarizer = None  # async (world, agent_id, text, focus) -> str
        self.custom_runner = None  # async (world, agent_id, CustomAction, args) -> str
        for i, p in enumerate(people, 1):
            aid = f"A{i}"
            self.agents[aid] = Agent(aid, p["name"], p.get("persona", ""))
            self.boards[f"private:{aid}"] = Board(f"private:{aid}", aid)

    # ------------------------------------------------------------------ ids & structure

    def new_id(self, prefix: str) -> str:
        self._ids[prefix] += 1
        return f"{prefix}{self._ids[prefix]}"

    def active_teams(self) -> list[Team]:
        return [t for t in self.teams.values() if t.dissolved is None]

    def children(self, tid: str | None) -> list[str]:
        return [t.id for t in self.active_teams() if t.parent == tid]

    def direct_members(self, tid: str) -> list[str]:
        return [a.id for a in self.agents.values() if a.team == tid]

    def subtree_teams(self, tid: str) -> list[str]:
        out, stack = [], [tid]
        while stack:
            t = stack.pop()
            out.append(t)
            stack.extend(self.children(t))
        return out

    def subtree_agents(self, tid: str) -> list[str]:
        inside = set(self.subtree_teams(tid))
        return [a.id for a in self.agents.values() if a.team in inside]

    def ancestors(self, tid: str | None) -> list[str]:
        out = []
        while tid:
            out.append(tid)
            tid = self.teams[tid].parent
        return out

    def agent_teams(self, aid: str) -> list[str]:
        """Home team first, then every enclosing superagent."""
        return self.ancestors(self.agents[aid].team)

    def principals(self, aid: str) -> set[str]:
        return {aid, "*", *self.agent_teams(aid)}

    def lead(self, tid: str) -> str | None:
        """De facto lead: measured from coordination behavior, never assigned."""
        members = self.subtree_agents(tid)
        if not members:
            return None
        best = max(members, key=lambda m: (self.agents[m].influence[tid], m == self.teams[tid].founder))
        return best if self.agents[best].influence[tid] > 0 else None

    def label(self, x: str | None) -> str:
        if x in self.agents:
            return f"{x}/{self.agents[x].name}"
        if x in self.teams:
            return f"{x} '{self.teams[x].name}'"
        return str(x)

    # ------------------------------------------------------------------ resolution & permissions

    def _team(self, ref) -> str:
        tid = str(ref or "").strip().removeprefix("team:")
        if tid not in self.teams or self.teams[tid].dissolved is not None:
            raise ActionError(f"no active team '{ref}'")
        return tid

    def _agent(self, ref) -> str:
        aid = str(ref or "").strip().removeprefix("private:").removeprefix("agent:")
        if aid not in self.agents:
            raise ActionError(f"no agent '{ref}'")
        return aid

    def _principal(self, ref) -> str:
        ref = str(ref or "").strip()
        return self._agent(ref) if ref.removeprefix("agent:").startswith("A") else self._team(ref)

    def _block(self, ref) -> str:
        bid = str(ref or "").strip()
        if bid not in self.blocks:
            raise ActionError(f"no block '{ref}'")
        return bid

    def board_id(self, aid: str, ref) -> str:
        ref = str(ref or "").strip()
        if ref in ("global", "society", "all"):
            return "global"
        if ref in ("private", "me", "self", ""):
            return f"private:{aid}"
        if ref == "team":
            if not self.agents[aid].team:
                raise ActionError("you have no home team")
            return f"team:{self.agents[aid].team}"
        if re.fullmatch(r"(private:|agent:)?A\d+", ref):
            return f"private:{self._agent(ref)}"
        if re.fullmatch(r"(team:)?T\d+", ref):
            tid = ref.removeprefix("team:")
            if f"team:{tid}" not in self.boards:
                raise ActionError(f"no team board '{ref}'")
            return f"team:{tid}"
        raise ActionError(f"unknown board '{ref}'")

    def can_read(self, aid: str, bid: str) -> bool:
        b = self.boards[bid]
        if bid == "global" or b.owner == aid:
            return True
        if bid.startswith("team:") and b.owner in self.agent_teams(aid):
            return True
        return bool(self.principals(aid) & (b.readers | b.writers))

    def can_write(self, aid: str, bid: str) -> bool:
        b = self.boards[bid]
        if bid == "global" or b.owner == aid:
            return True
        if bid.startswith("team:") and b.owner in self.agent_teams(aid):
            return True
        return bool(self.principals(aid) & b.writers)

    def _need(self, ok: bool, msg: str):
        if not ok:
            raise ActionError(msg)

    def can_manage(self, aid: str, block_id: str) -> bool:
        owner = self.blocks[block_id].owner
        return owner is None or owner == aid or owner in self.agent_teams(aid)

    # ------------------------------------------------------------------ messaging

    def notify(self, target: str, frm: str, text: str, **extra):
        if target in self.agents:
            if target != frm:
                self.agents[target].inbox.append({"t": self.tick, "from": frm, "text": clip(text, 500), **extra})
        elif target in self.teams:
            for m in self.subtree_agents(target):
                self.notify(m, frm, text, **extra)

    def post(self, bid: str, author: str, kind: str, content: str, refs=None, via=None) -> Entry:
        e = Entry(self.new_id("E"), bid, author, self.tick, kind, str(content)[:MAX_TEXT],
                  [str(r) for r in as_list(refs)][:12], via)
        self.boards[bid].entries.append(e)
        for m in set(MENTION.findall(e.content)):
            if m in self.agents or (m in self.teams and self.teams[m].dissolved is None):
                self.notify(m, author, f"mentioned you in {e.id} on {bid}: {clip(e.content, 300)}")
        if bid.startswith("private:") and self.boards[bid].owner != author:
            self.notify(self.boards[bid].owner, author, f"wrote {e.id} on your private board: {clip(e.content, 300)}")
        return e

    def fmt(self, e: Entry, limit: int | None = None) -> str:
        body = e.content if limit is None else clip(e.content, limit)
        via = f" via x_{e.via}" if e.via else ""
        refs = f" refs={','.join(e.refs)}" if e.refs else ""
        return f"[{e.id} t{e.tick} {self.label(e.author)} {e.kind}{via}{refs}] {body}"

    # ------------------------------------------------------------------ accounting

    def charge(self, aid: str, category: str, tokens: int):
        self.ledger[aid][category] += int(tokens)

    def total_tokens(self) -> int:
        return sum(sum(c.values()) for c in self.ledger.values())

    def budget_left(self) -> int | None:
        return None if self.token_budget is None else self.token_budget - self.total_tokens()

    # ------------------------------------------------------------------ dispatch

    async def act(self, aid: str, name: str, args: dict | None, via: str | None = None) -> tuple[bool, str]:
        args = dict(args or {})
        op_label, target = name, None
        try:
            if name in PRIMITIVES and name in self.allowed:
                op = str(args.get("op", "")).strip()
                ops = self.allowed[name]
                if ops is not None and op and op not in ops:
                    raise ActionError(f"op '{op}' is not available in this society")
                op_label, msg, target = await getattr(self, f"_do_{name}")(aid, args, via)
            elif name.startswith("x_") and "design_action" in self.allowed:
                op_label, msg = await self._run_custom(aid, name[2:], args, via)
            else:
                raise ActionError(f"unknown action '{name}'")
            ok = True
        except ActionError as exc:
            ok, msg = False, f"refused: {exc}"
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            ok, msg = False, f"refused: malformed arguments ({exc})"
        if ok and op_label.startswith("connect.") and op_label.split(".")[1] in STRUCTURAL:
            self.counts["reorganizations"] += 1
        self.counts[op_label.split(".")[0]] += 1
        self.events.append({
            "t": self.tick, "agent": aid, "op": op_label, "ok": ok, "target": target, "via": via,
            "args": {k: clip(v if isinstance(v, str) else json.dumps(v), 240) for k, v in args.items() if not k.startswith("_")},
            "msg": clip(msg, 400),
        })
        return ok, msg

    def log_intent(self, aid: str, text: str):
        if text.strip():
            self.events.append({"t": self.tick, "agent": aid, "op": "intent", "ok": True, "target": None,
                                "via": None, "args": {}, "msg": clip(text, 400)})

    # ------------------------------------------------------------------ primitives

    async def _do_read(self, aid, a, via):
        bid = self.board_id(aid, a.get("board", "global"))
        self._need(self.can_read(aid, bid), f"you cannot read {bid}")
        entries = self.boards[bid].entries
        if a.get("kind"):
            entries = [e for e in entries if e.kind == a["kind"]]
        if a.get("author"):
            entries = [e for e in entries if e.author == a["author"]]
        last = max(1, min(30, int(a.get("last") or 8)))
        shown = entries[-last:]
        body = "\n".join(self.fmt(e, 1500) for e in shown) or "(no entries)"
        self.charge(aid, "read", est_tokens(body))
        return "read", f"{bid}: {len(entries)} matching entries, showing the last {len(shown)}\n{body}", bid

    async def _do_write(self, aid, a, via):
        bid = self.board_id(aid, a.get("board", "private"))
        self._need(self.can_write(aid, bid), f"you cannot write to {bid}")
        content = str(a.get("content", "")).strip()
        self._need(bool(content), "content is empty")
        kind = a.get("kind") if a.get("kind") in ENTRY_KINDS else "note"
        e = self.post(bid, aid, kind, content, a.get("refs"), via)
        if bid != f"private:{aid}":
            self.charge(aid, "communication", est_tokens(content))
        scope = bid.split(":")[0]
        return f"write.{scope}", f"posted {e.id} to {bid}", bid

    async def _do_share(self, aid, a, via):
        bid = self.board_id(aid, a.get("board", "private"))
        b = self.boards[bid]
        if bid.startswith("private:"):
            self._need(b.owner == aid, "you can only share your own private board")
        elif bid.startswith("team:"):
            self._need(b.owner in self.agent_teams(aid), "you can only share boards of teams you are in")
        else:
            raise ActionError("the global board is already shared with everyone")
        mode = a.get("mode", "read")
        targets = ["*" if t in ("all", "*", "everyone") else self._principal(t) for t in as_list(a.get("with"))]
        self._need(bool(targets), "say who to share with")
        for t in targets:
            if mode == "revoke":
                b.readers.discard(t)
                b.writers.discard(t)
            elif mode == "write":
                b.writers.add(t)
            else:
                b.readers.add(t)
            if t != "*" and mode != "revoke":
                self.notify(t, aid, f"shared {bid} with you ({mode}). Read it with board='{bid.split(':')[1]}'.")
        return "share", f"{bid}: {mode} for {', '.join(targets)}", targets[0]

    async def _do_summarize(self, aid, a, via):
        src = self.board_id(aid, a.get("source", "team"))
        self._need(self.can_read(aid, src), f"you cannot read {src}")
        dest = self.board_id(aid, a.get("dest", "private"))
        self._need(self.can_write(aid, dest), f"you cannot write to {dest}")
        sources = [src]
        if a.get("include_subteams") and src.startswith("team:"):
            sources += [f"team:{t}" for t in self.subtree_teams(src[5:])[1:] if self.can_read(aid, f"team:{t}")]
        parts, refs = [], []
        for s in sources:
            entries = self.boards[s].entries[-40:]
            if entries:
                parts.append(f"## {s}\n" + "\n".join(self.fmt(e, 1200) for e in entries))
                refs.append(entries[-1].id)
        self._need(bool(parts), "nothing to summarize")
        text = "\n\n".join(parts)
        summary = await self.summarizer(self, aid, text, str(a.get("focus", "")))
        e = self.post(dest, aid, "summary", summary, refs, via)
        if dest.startswith("team:"):
            self.agents[aid].influence[dest[5:]] += 1
        return "summarize", f"posted summary {e.id} to {dest} (from {', '.join(sources)}):\n{clip(summary, 600)}", dest

    async def _do_reflect(self, aid, a, via):
        target = str(a.get("target", "self")).strip()
        content = str(a.get("content", "")).strip()
        self._need(bool(content), "content is empty")
        if a.get("proposal"):
            content += f"\nPROPOSAL: {a['proposal']}"
        me = self.agents[aid]
        if target in ("self", "me", aid):
            me.reflections.append(clip(content, 600))
            self.post(f"private:{aid}", aid, "reflection", content, via=via)
            return "reflect.self", "noted in your self-model (shown at the top of future turns)", aid
        self.charge(aid, "critique", est_tokens(content))
        if target in ("society", "all", "global", "everyone"):
            e = self.post("global", aid, "critique", content, via=via)
            return "reflect.society", f"posted {e.id} to global", "society"
        if target.removeprefix("agent:") in self.agents:
            other = self._agent(target)
            shared = [t for t in self.agent_teams(aid) if t in self.agent_teams(other)]
            bid = f"team:{shared[0]}" if shared else f"private:{aid}"
            e = self.post(bid, aid, "critique", f"@{other} {content}", via=via)
            if not shared:
                self.notify(other, aid, f"critique of you: {content}")
            return "reflect.agent", f"critique {e.id} delivered to {other} (on {bid})", other
        tid = self._team(target)
        if tid in self.agent_teams(aid):
            e = self.post(f"team:{tid}", aid, "critique", content, via=via)
            return "reflect.team", f"posted {e.id} on team:{tid}", tid
        self.notify(tid, aid, f"outside critique of {tid}: {content}")
        return "reflect.team", f"critique delivered to every member of {tid}", tid

    # ------------------------------------------------------------------ connect: structure & governance

    def _authorize(self, aid: str, tid: str, action: str, args: dict, summary: str) -> str | None:
        """Return None if `aid` may act for `tid` now, else open a motion and return a message.

        Each team picks its own decision rule; the kernel only enforces it.
        """
        mid = args.get("_motion")
        if mid in self.motions and self.motions[mid].team == tid and self.motions[mid].status == "passed":
            return None
        t = self.teams[tid]
        voters = set(self.subtree_agents(tid))
        if aid in voters and (t.rule == "any" or (t.rule == "rep" and aid in t.reps)):
            return None
        m = Motion(self.new_id("V"), tid, aid, action, {k: v for k, v in args.items() if k != "_motion"},
                   summary, self.tick)
        if aid in voters:
            m.yes.add(aid)
        self.motions[m.id] = m
        if self._passes(m) is True:
            m.status = "passed"
            return None
        self.notify(tid, aid, f"motion {m.id} in {tid} (rule '{t.rule}'): {summary}. "
                              f"Vote with connect op=vote motion={m.id} vote=yes|no.")
        return f"motion {m.id} opened in {tid} under rule '{t.rule}' ({self._tally(m)}); it executes when it passes"

    def _tally(self, m: Motion) -> str:
        return f"{len(m.yes)} yes / {len(m.no)} no of {len(self.subtree_agents(m.team))} voters"

    def _passes(self, m: Motion) -> bool | None:
        t = self.teams[m.team]
        voters = set(self.subtree_agents(m.team))
        yes, no = m.yes & voters, m.no & voters
        if t.rule == "any":
            return True if yes else None
        if t.rule == "rep":
            reps = t.reps & voters or voters  # a team whose reps all left falls back to any member
            return True if yes & reps else None
        if t.rule == "majority":
            if len(yes) * 2 > len(voters):
                return True
            return False if len(no) * 2 >= len(voters) else None
        if no:
            return False
        return True if yes >= voters else None

    async def _execute_motion(self, m: Motion) -> str:
        m.status = "passed"
        ok, msg = await self.act(m.proposer, m.action, {**m.args, "_motion": m.id}, via=None)
        return f"motion {m.id} passed and executed: {msg}" if ok else f"motion {m.id} passed but failed: {msg}"

    def _expire_motions(self):
        for m in self.motions.values():
            if m.status == "open" and self.tick - m.created > MOTION_TTL:
                m.status = "expired"

    def _dissolve(self, tid: str):
        t = self.teams[tid]
        for m in self.direct_members(tid):
            self.agents[m].team = t.parent
        for c in self.children(tid):
            self.teams[c].parent = t.parent
        t.dissolved = self.tick
        for b in self.blocks.values():
            if b.owner == tid and b.status != "done":
                b.owner = None
                b.status = "open" if b.status == "claimed" else b.status

    def _cleanup(self, tid: str | None):
        """Teams with no agents left inside them dissolve; walk upward."""
        while tid and self.teams[tid].dissolved is None and not self.subtree_agents(tid):
            parent = self.teams[tid].parent
            for t in self.subtree_teams(tid):
                self._dissolve(t)
            tid = parent

    def _move(self, aid: str, tid: str | None):
        old = self.agents[aid].team
        self.agents[aid].team = tid
        if old != tid:
            self._cleanup(old)

    async def _do_connect(self, aid, a, via):
        op = str(a.get("op", "")).strip()
        me = self.agents[aid]
        target = str(a.get("target", "") or "").strip()

        if op in ("link", "unlink"):
            tgt = self._principal(target)
            self._need(tgt != aid, "cannot link to yourself")
            if op == "link":
                me.links.add(tgt)
                if tgt in self.agents:
                    self.agents[tgt].links.add(aid)
                self.notify(tgt, aid, f"{self.label(aid)} linked with {'you' if tgt in self.agents else tgt}")
            else:
                me.links.discard(tgt)
                if tgt in self.agents:
                    self.agents[tgt].links.discard(aid)
            return f"connect.{op}", f"{op} {self.label(tgt)}: done", tgt

        if op == "vote":
            mid = str(a.get("motion", "")).strip()
            self._need(mid in self.motions, f"no motion '{mid}'")
            m = self.motions[mid]
            self._need(m.status == "open", f"motion {mid} is {m.status}")
            self._need(aid in self.subtree_agents(m.team), f"only members of {m.team} vote on {mid}")
            (m.yes if str(a.get("vote", "yes")).lower().startswith("y") else m.no).add(aid)
            verdict = self._passes(m)
            if verdict is True:
                return "connect.vote", await self._execute_motion(m), m.team
            if verdict is False:
                m.status = "failed"
                self.notify(m.proposer, aid, f"motion {mid} failed")
                return "connect.vote", f"motion {mid} failed ({self._tally(m)})", m.team
            return "connect.vote", f"vote recorded on {mid} ({self._tally(m)})", m.team

        if op == "leave":  # exit is always free
            old = me.team
            self._need(old is not None, "you have no home team")
            self._move(aid, None)
            self.notify(old, aid, f"{self.label(aid)} left {old}")
            return "connect.leave", f"you left {old} and are now a free agent", old

        if op == "form_team":
            name, charter = clip(a.get("name"), 60), str(a.get("charter") or "").strip()[:1500]
            self._need(bool(name and charter), "form_team needs a name and a charter")
            parent = a.get("parent")
            parent = None if str(parent or "").lower() in ("", "none", "null", "root", "society") else self._team(parent)
            if parent:
                self._need(parent in self.agent_teams(aid), "you can only create a sub-team inside a team you are in")
            tid = self.new_id("T")
            self.teams[tid] = Team(tid, name, charter, aid, parent, self.tick)
            self.boards[f"team:{tid}"] = Board(f"team:{tid}", tid)
            self.post(f"team:{tid}", aid, "plan", f"Charter: {charter}")
            self._move(aid, tid)
            me.influence[tid] += 2
            for inv in as_list(a.get("invite")):
                if inv in self.agents and inv != aid:
                    self.notify(inv, aid, f"invites you to join {tid} '{name}': {clip(charter, 200)}", invite=tid)
            where = f" inside {self.label(parent)}" if parent else " at top level"
            return "connect.form_team", f"created {tid} '{name}'{where}; you are its first member", tid

        if op == "join":
            tid = self._team(target)
            self._need(me.team != tid, f"{tid} is already your home team")
            t = self.teams[tid]
            if not t.open:
                pending = self._authorize(aid, tid, "connect", {"op": "admit", "target": aid, "team": tid},
                                          f"admit {self.label(aid)}")
                if pending:
                    return "connect.join", f"{tid} is closed; requested admission. {pending}", tid
            old = me.team
            self._move(aid, tid)
            self.notify(tid, aid, f"{self.label(aid)} joined {tid}")
            return "connect.join", f"you moved into {self.label(tid)}" + (f" (left {old})" if old else ""), tid

        if op == "invite":
            tid = self._team(a.get("team") or me.team)
            self._need(tid in self.agent_teams(aid), "you can only invite into teams you are in")
            invitees = [self._agent(x) for x in as_list(target or a.get("invite"))]
            self._need(bool(invitees), "who should be invited?")
            for inv in invitees:
                self.notify(inv, aid, f"invites you to join {self.label(tid)}: {clip(self.teams[tid].charter, 200)}", invite=tid)
            me.influence[tid] += 1
            return "connect.invite", f"invited {', '.join(invitees)} to {tid}", invitees[0]

        if op == "admit":
            team_into = self._team(a.get("team") or me.team)
            newcomer = str(a.get("target") or "").strip()
            pending = self._authorize(aid, team_into, "connect", {**a, "team": team_into}, f"admit {self.label(newcomer)}")
            if pending:
                return "connect.admit", pending, team_into
            if newcomer in self.agents:
                self._move(newcomer, team_into)
            else:
                child = self._team(newcomer)
                self._need(team_into not in self.subtree_teams(child), "that would create a cycle")
                if a.get("merge"):
                    for m in self.direct_members(child):
                        self.agents[m].team = team_into
                    self._dissolve(child)
                else:
                    self.teams[child].parent = team_into
            self.notify(newcomer, aid, f"{newcomer} admitted into {self.label(team_into)}")
            return "connect.admit", f"admitted {self.label(newcomer)} into {team_into}", newcomer

        # Everything below acts on behalf of a team, so the team's own rule applies.
        tid = self._team(target or me.team)
        t = self.teams[tid]

        if op == "nest":
            parent = a.get("parent")
            parent = None if str(parent or "").lower() in ("", "none", "null", "root", "society") else self._team(parent)
            if parent is None and t.parent and t.parent in self.agent_teams(aid) and tid not in self.agent_teams(aid):
                pending = self._authorize(aid, t.parent, "connect", a, f"detach sub-team {tid}")
            else:
                self._need(tid in self.agent_teams(aid), f"you are not inside {tid}")
                if parent:
                    self._need(parent not in self.subtree_teams(tid), "that would create a cycle")
                pending = self._authorize(aid, tid, "connect", a, f"attach {tid} under {parent or 'society (top level)'}")
            if pending:
                return "connect.nest", pending, tid
            if parent and not self.teams[parent].open and aid not in self.subtree_agents(parent):
                pending = self._authorize(aid, parent, "connect", {"op": "admit", "target": tid, "team": parent},
                                          f"admit team {self.label(tid)} as a sub-team")
                if pending:
                    return "connect.nest", f"{parent} is closed; requested admission. {pending}", tid
            t.parent = parent
            me.influence[tid] += 1
            self.notify(parent or tid, aid, f"{self.label(tid)} is now nested under {self.label(parent) if parent else 'society'}")
            return "connect.nest", f"{tid} now sits under {self.label(parent) if parent else 'society (top level)'}", parent or "society"

        self._need(tid in self.agent_teams(aid), f"you are not inside {tid}")

        if op == "merge":
            into = self._team(a.get("into") or a.get("parent"))
            self._need(into not in self.subtree_teams(tid), "cannot merge a team into its own sub-team")
            pending = self._authorize(aid, tid, "connect", a, f"merge {tid} into {into}")
            if pending:
                return "connect.merge", pending, into
            if not self.teams[into].open and aid not in self.subtree_agents(into):
                pending = self._authorize(aid, into, "connect", {"op": "admit", "target": tid, "team": into, "merge": True},
                                          f"merge team {self.label(tid)} into this team")
                if pending:
                    return "connect.merge", f"{into} is closed; requested merge. {pending}", into
            for m in self.direct_members(tid):
                self.agents[m].team = into
            for c in self.children(tid):
                self.teams[c].parent = into
            t.dissolved = self.tick
            for b in self.blocks.values():
                if b.owner == tid:
                    b.owner = into
            self.notify(into, aid, f"{self.label(tid)} merged into {self.label(into)}")
            return "connect.merge", f"merged {tid} into {self.label(into)}", into

        if op == "split":
            members = [self._agent(m) for m in as_list(a.get("members"))]
            name, charter = clip(a.get("name"), 60), str(a.get("charter") or "").strip()[:1500]
            self._need(bool(members and name and charter), "split needs members, a name and a charter")
            self._need(all(self.agents[m].team == tid for m in members), f"members must be direct members of {tid}")
            pending = self._authorize(aid, tid, "connect", a, f"split {', '.join(members)} off into new team '{name}'")
            if pending:
                return "connect.split", pending, tid
            new = self.new_id("T")
            parent = tid if a.get("as_subteam", True) else t.parent
            self.teams[new] = Team(new, name, charter, aid, parent, self.tick)
            self.boards[f"team:{new}"] = Board(f"team:{new}", new)
            self.post(f"team:{new}", aid, "plan", f"Charter: {charter}")
            for m in members:
                self.agents[m].team = new
                self.notify(m, aid, f"you were moved into new team {new} '{name}' split from {tid}")
            self._cleanup(tid)
            me.influence[tid] += 1
            return "connect.split", f"split {', '.join(members)} into {new} '{name}' under {self.label(parent) if parent else 'society'}", new

        if op == "dissolve":
            pending = self._authorize(aid, tid, "connect", a, f"dissolve {tid}")
            if pending:
                return "connect.dissolve", pending, tid
            members = self.subtree_agents(tid)
            self._dissolve(tid)
            for m in members:
                self.notify(m, aid, f"{self.label(tid)} was dissolved")
            return "connect.dissolve", f"dissolved {tid}; its members and sub-teams moved up a level", tid

        if op == "recharter":
            pending = self._authorize(aid, tid, "connect", a, f"rename/recharter {tid}")
            if pending:
                return "connect.recharter", pending, tid
            t.name = clip(a.get("name"), 60) or t.name
            t.charter = str(a.get("charter") or t.charter)[:1500]
            self.post(f"team:{tid}", aid, "plan", f"Revised charter: {t.charter}")
            me.influence[tid] += 1
            return "connect.recharter", f"{tid} is now '{t.name}'", tid

        if op == "govern":
            rule = a.get("rule", t.rule)
            self._need(rule in RULES, f"rule must be one of {RULES}")
            pending = self._authorize(aid, tid, "connect", a, f"set decision rule of {tid} to '{rule}'")
            if pending:
                return "connect.govern", pending, tid
            t.rule = rule
            if a.get("reps") is not None:
                t.reps = {self._agent(r) for r in as_list(a.get("reps"))}
            if a.get("open") is not None:
                t.open = str(a.get("open")).lower() not in ("false", "0", "no", "closed")
            self.notify(tid, aid, f"{tid} now decides by '{t.rule}'"
                                  + (f" (reps {', '.join(sorted(t.reps))})" if t.reps else "")
                                  + f"; membership {'open' if t.open else 'by admission'}")
            me.influence[tid] += 1
            return "connect.govern", f"{tid}: rule={t.rule} reps={sorted(t.reps)} open={t.open}", tid

        if op == "expel":
            who = self._agent(a.get("member"))
            self._need(who in self.subtree_agents(tid) and who != aid, f"{who} is not another member of {tid}")
            pending = self._authorize(aid, tid, "connect", a, f"expel {self.label(who)}")
            if pending:
                return "connect.expel", pending, who
            self._move(who, None)
            self.notify(who, aid, f"you were expelled from {self.label(tid)}")
            return "connect.expel", f"expelled {who} from {tid}", who

        raise ActionError(f"unknown connect op '{op}' (ops: {', '.join(CONNECT_OPS)})")

    # ------------------------------------------------------------------ plan: hierarchical specs (chip-design style)

    async def _do_plan(self, aid, a, via):
        op = str(a.get("op", "")).strip()
        me = self.agents[aid]
        home = me.team

        if op == "decompose":
            parent = self._block(a.get("block") or "M0")
            self._need(self.can_manage(aid, parent), f"{parent} is owned by {self.label(self.blocks[parent].owner)}")
            items = a.get("blocks") or []
            if isinstance(items, str):
                items = json.loads(items)
            self._need(isinstance(items, list) and bool(items), "blocks must be a non-empty list of {title, spec}")
            made = []
            for it in items[:8]:
                it = {"title": it} if isinstance(it, str) else it
                title = clip(it.get("title"), 90)
                if not title:
                    continue
                owner = self._principal(it["owner"]) if it.get("owner") else None
                nid = self.new_id("B")
                self.blocks[nid] = Block(nid, title, str(it.get("spec", "")).strip()[:2000], parent, aid, self.tick,
                                         owner=owner, status="claimed" if owner else "open")
                if owner:
                    self.notify(owner, aid, f"assigned you block {nid} '{title}' (child of {parent})")
                made.append(nid)
            if home:
                me.influence[home] += len(made)
            return "plan.decompose", f"created {', '.join(made)} under {parent}", parent

        bid = self._block(a.get("block"))
        b = self.blocks[bid]

        if op == "assign":
            to = self._principal(a.get("to"))
            self._need(self.can_manage(aid, bid) or (b.parent and self.can_manage(aid, b.parent)),
                       f"you manage neither {bid} nor its parent")
            b.owner = to
            if b.status == "open":
                b.status = "claimed"
            self.notify(to, aid, f"assigned you block {bid} '{b.title}': {clip(b.spec, 300)}")
            if home:
                me.influence[home] += 1
            return "plan.assign", f"{bid} now owned by {self.label(to)}", to

        if op == "claim":
            self._need(b.owner is None, f"{bid} is owned by {self.label(b.owner)}; decompose it or ask them")
            b.owner, b.status = aid, "claimed"
            return "plan.claim", f"you own {bid} '{b.title}'", bid

        if op == "release":
            self._need(b.owner == aid or b.owner in self.agent_teams(aid), f"you do not own {bid}")
            if b.owner in self.teams:
                pending = self._authorize(aid, b.owner, "plan", a, f"release block {bid}")
                if pending:
                    return "plan.release", pending, bid
            b.owner, b.status = None, "open"
            return "plan.release", f"{bid} is open again", bid

        if op == "submit":
            deliverable = str(a.get("deliverable", "")).strip()
            self._need(bool(deliverable), "deliverable is empty")
            self._need(b.status != "done", f"{bid} is already done")
            if b.owner is None:
                b.owner = aid
            self._need(b.owner == aid or b.owner in self.agent_teams(aid), f"{bid} is owned by {self.label(b.owner)}")
            if b.owner in self.teams:  # submitting is a team speaking through its interface
                pending = self._authorize(aid, b.owner, "plan", a, f"submit deliverable for {bid}: {clip(deliverable, 120)}")
                if pending:
                    return "plan.submit", pending, bid
            b.deliverable, b.status, b.submitted_by = deliverable[:MAX_TEXT], "submitted", aid
            self.charge(aid, "communication", est_tokens(deliverable))
            for who in {b.created_by, self.blocks[b.parent].owner if b.parent else None} - {None, "society", aid}:
                self.notify(who, aid, f"submitted {bid} '{b.title}' for verification")
            return "plan.submit", f"{bid} submitted; it is done once someone else verifies it", bid

        if op == "verify":
            self._need(b.status == "submitted", f"{bid} is {b.status}, not submitted")
            self._need(aid != b.submitted_by, "you cannot verify your own submission")
            verdict = "accept" if str(a.get("verdict", "")).lower().startswith("acc") else "reject"
            notes = str(a.get("notes", "")).strip()[:1500]
            b.reviews.append({"t": self.tick, "by": aid, "verdict": verdict, "notes": notes})
            b.status = "done" if verdict == "accept" else "claimed"
            self.notify(b.submitted_by, aid, f"{verdict}ed {bid}: {clip(notes, 300)}")
            if home:
                me.influence[home] += 1
            return "plan.verify", f"{bid} {verdict}ed", b.submitted_by

        if op == "revise":
            self._need(self.can_manage(aid, bid) or (b.parent and self.can_manage(aid, b.parent)),
                       f"you manage neither {bid} nor its parent")
            b.title = clip(a.get("title"), 90) or b.title
            b.spec = str(a.get("spec") or b.spec)[:2000]
            if b.owner:
                self.notify(b.owner, aid, f"revised spec of {bid}: {clip(b.spec, 300)}")
            return "plan.revise", f"{bid} spec revised", bid

        raise ActionError(f"unknown plan op '{op}' (ops: {', '.join(PLAN_OPS)})")

    # ------------------------------------------------------------------ design_action: the evolving action space

    async def _do_design_action(self, aid, a, via):
        op = str(a.get("op", "")).strip()
        me = self.agents[aid]
        name = str(a.get("name", "")).strip().lower().removeprefix("x_")
        if op == "define":
            self._need(bool(ACTION_NAME.match(name)), "name must be 3-31 chars of a-z, 0-9, _")
            self._need(name not in self.custom, f"x_{name} exists; adopt it or pick another name")
            desc, proc = str(a.get("description", "")).strip(), str(a.get("procedure", "")).strip()
            self._need(bool(desc and proc), "define needs a description and a procedure")
            params = [p for p in as_list(a.get("params")) if ACTION_NAME.match(p)][:5]
            self.custom[name] = CustomAction(name, desc[:400], proc[:2000], params, aid, self.tick)
            me.actions.add(name)
            return "design_action.define", f"defined x_{name}; it becomes one of your tools next turn and others can adopt it", name
        self._need(name in self.custom, f"no custom action x_{name}")
        ca = self.custom[name]
        if op == "revise":
            self._need(ca.author == aid, "only the author can revise; define a variant under a new name instead")
            ca.description = str(a.get("description") or ca.description)[:400]
            ca.procedure = str(a.get("procedure") or ca.procedure)[:2000]
            if a.get("params") is not None:
                ca.params = [p for p in as_list(a.get("params")) if ACTION_NAME.match(p)][:5]
            ca.version += 1
            return "design_action.revise", f"x_{name} is now v{ca.version}", name
        if op == "adopt":
            me.actions.add(name)
            return "design_action.adopt", f"adopted x_{name}; available from your next turn", name
        if op == "drop":
            me.actions.discard(name)
            return "design_action.drop", f"dropped x_{name}", name
        raise ActionError(f"unknown design_action op '{op}' (ops: {', '.join(DESIGN_OPS)})")

    async def _run_custom(self, aid, name, args, via):
        self._need(via is None, "custom actions cannot invoke other custom actions")
        self._need(name in self.custom, f"no custom action x_{name}")
        self._need(name in self.agents[aid].actions, f"adopt x_{name} before using it")
        ca = self.custom[name]
        ca.uses += 1
        result = await self.custom_runner(self, aid, ca, args)
        return f"x.{name}", result

    # ------------------------------------------------------------------ observation

    def observe(self, aid: str, consume_inbox: bool = True) -> str:
        me = self.agents[aid]
        self._expire_motions()
        L = [f"TICK {self.tick} of {self.total_ticks}. You are {aid} ({me.name}). {me.persona or 'You have no assigned specialty.'}",
             f"\nMISSION (block M0): {self.mission}"]
        left = self.budget_left()
        spent = self.total_tokens()
        cats = Counter()
        for c in self.ledger.values():
            cats.update(c)
        breakdown = ", ".join(f"{k} {v * 100 // max(spent, 1)}%" for k, v in cats.most_common(5))
        L.append(f"\nCOSTS: society has used {spent:,} tokens ({breakdown or 'none yet'}); "
                 f"you {sum(self.ledger[aid].values()):,}; reorganizations so far {self.counts['reorganizations']}"
                 + (f"; budget left {max(left, 0):,} tokens" if left is not None else ""))
        if me.reflections:
            L.append("\nYOUR SELF-NOTES:\n" + "\n".join(f"- {r}" for r in me.reflections[-4:]))
        path = " < ".join(self.label(t) for t in self.agent_teams(aid)) or "none (free agent)"
        L.append(f"\nYOUR POSITION: home team {path}. Links: {', '.join(self.label(x) for x in sorted(me.links)) or 'none'}. "
                 f"Adopted custom actions: {', '.join('x_' + n for n in sorted(me.actions)) or 'none'}.")
        if me.inbox:
            L.append("\nINBOX (new since your last turn):\n" + "\n".join(
                f"- t{n['t']} from {self.label(n['from'])}: {n['text']}" for n in me.inbox[-15:]))
            if consume_inbox:
                me.inbox = []
        open_motions = [m for m in self.motions.values() if m.status == "open" and aid in self.subtree_agents(m.team)]
        if open_motions:
            L.append("\nOPEN MOTIONS YOU CAN VOTE ON:\n" + "\n".join(
                f"- {m.id} in {m.team} by {m.proposer}: {m.summary} ({self._tally(m)})" for m in open_motions))
        L.append("\nSOCIETY MAP (a team with its sub-teams is a superagent; indentation = nesting):")
        tree: list[str] = []
        self._render_teams(None, 0, tree, aid)
        L.extend(tree or ["  (no teams exist)"])
        free = [x for x in self.agents.values() if x.team is None]
        if free:
            L.append("  Free agents: " + ", ".join(x.id for x in free))
        L.append("AGENTS: " + "; ".join(f"{x.id} {x.name} ({clip(x.persona, 70) or 'no specialty'})" for x in self.agents.values()))
        L.append("\nBLOCK TREE (status, owner; M0 is the mission):")
        L.extend(self._render_blocks(aid))
        if "design_action" in self.allowed:
            if self.custom:
                L.append("\nCUSTOM ACTIONS IN CIRCULATION:")
                for ca in sorted(self.custom.values(), key=lambda c: -c.uses):
                    adopters = sum(ca.name in x.actions for x in self.agents.values())
                    mark = " [adopted]" if ca.name in me.actions else ""
                    L.append(f"- x_{ca.name} v{ca.version} by {ca.author}, {adopters} adopters, {ca.uses} uses{mark}: {clip(ca.description, 160)}")
            else:
                L.append("\nCUSTOM ACTIONS IN CIRCULATION: none yet.")
        L.append("\nBOARDS YOU CAN SEE (latest entries; use read for more):")
        for bid, n in self._visible_boards(aid):
            entries = self.boards[bid].entries[-n:]
            L.append(f"[{bid}] {len(self.boards[bid].entries)} entries")
            L.extend("  " + self.fmt(e, 320) for e in entries)
        L.append(f"\nYou may take up to {self.max_actions} actions this tick.")
        return "\n".join(L)

    def _render_teams(self, parent, depth, out, aid):
        me = self.agents[aid]
        for tid in sorted(self.children(parent), key=lambda x: int(x[1:])):
            t = self.teams[tid]
            mark = "  ← your home team" if tid == me.team else ""
            gov = f"rule {t.rule}" + (f" reps {','.join(sorted(t.reps))}" if t.rule == "rep" else "") + ("" if t.open else ", closed")
            out.append(f"{'  ' * (depth + 1)}- {tid} '{t.name}'{mark}: members {', '.join(self.direct_members(tid)) or '—'}; "
                       f"{len(self.subtree_agents(tid))} agents inside; lead {self.lead(tid) or '—'}; {gov}; "
                       f"charter: {clip(t.charter, 150)}")
            self._render_teams(tid, depth + 1, out, aid)

    def _render_blocks(self, aid) -> list[str]:
        mine = self.principals(aid)
        relevant = {b.id for b in self.blocks.values() if b.owner in mine or (b.parent and self.blocks[b.parent].owner in mine)}
        show_all = len(self.blocks) <= 40
        out = []

        def walk(bid, depth):
            b = self.blocks[bid]
            if show_all or depth <= 2 or bid in relevant:
                spec = f" — {clip(b.spec, 120)}" if bid != "M0" else ""
                out.append(f"{'  ' * (depth + 1)}{bid} [{b.status}{', ' + str(b.owner) if b.owner else ''}] {b.title}{spec}")
            for c in [x.id for x in self.blocks.values() if x.parent == bid]:
                walk(c, depth + 1)

        walk("M0", 0)
        return out

    def _visible_boards(self, aid) -> list[tuple[str, int]]:
        out = [("global", 6)]
        for i, t in enumerate(self.agent_teams(aid)):
            out.append((f"team:{t}", 8 if i == 0 else 4))
        out.append((f"private:{aid}", 4))
        seen = {b for b, _ in out}
        for bid in self.boards:
            if bid not in seen and self.can_read(aid, bid) and self.boards[bid].entries:
                if bid.startswith("team:") and self.teams[bid[5:]].dissolved is not None:
                    continue
                out.append((bid, 3))
        return out

    # ------------------------------------------------------------------ snapshots & export

    def metrics(self) -> dict:
        act = self.active_teams()
        edges = {tuple(sorted((x.id, y))) for x in self.agents.values() for y in x.links if y in self.agents}
        top = {x.id: (self.agent_teams(x.id) or [None])[-1] for x in self.agents.values()}
        within = sum(1 for p, q in edges if top[p] and top[p] == top[q])
        status = Counter(b.status for b in self.blocks.values())
        cats = Counter()
        for c in self.ledger.values():
            cats.update(c)
        return {
            "teams": len(act),
            "depth": max((len(self.ancestors(t.id)) for t in act), default=0),
            "free": sum(1 for x in self.agents.values() if x.team is None),
            "largest": max((len(self.subtree_agents(t.id)) for t in act if t.parent is None), default=0),
            "links": len(edges),
            "links_within": within,
            "custom": len(self.custom),
            "adoptions": sum(len(x.actions) for x in self.agents.values()),
            "custom_uses": sum(c.uses for c in self.custom.values()),
            "blocks": len(self.blocks),
            "done": status["done"],
            "submitted": status["submitted"],
            "entries": sum(len(b.entries) for b in self.boards.values()),
            "reorganizations": self.counts["reorganizations"],
            "motions": len(self.motions),
            "tokens": self.total_tokens(),
            "token_split": dict(cats),
        }

    def snapshot(self) -> dict:
        return {
            "t": self.tick,
            "agents": [{"id": x.id, "team": x.team, "links": sorted(x.links)} for x in self.agents.values()],
            "teams": [{"id": t.id, "name": t.name, "charter": t.charter, "parent": t.parent, "founder": t.founder,
                       "created": t.created, "rule": t.rule, "reps": sorted(t.reps), "open": t.open,
                       "members": self.direct_members(t.id), "lead": self.lead(t.id)} for t in self.active_teams()],
            "blocks": [{"id": b.id, "title": b.title, "parent": b.parent, "owner": b.owner, "status": b.status}
                       for b in self.blocks.values()],
            "custom": [{"name": c.name, "author": c.author, "version": c.version, "uses": c.uses,
                        "adopters": sum(c.name in x.actions for x in self.agents.values())} for c in self.custom.values()],
            "motions": [{"id": m.id, "team": m.team, "status": m.status, "summary": m.summary} for m in self.motions.values()
                        if m.created == self.tick or m.status == "open"],
            "metrics": self.metrics(),
        }

    def export(self, meta: dict) -> dict:
        return {
            "meta": {**meta, "mission": self.mission, "ticks": self.tick, "max_actions": self.max_actions},
            "agents": [{"id": x.id, "name": x.name, "persona": x.persona, "reflections": x.reflections,
                        "tokens": dict(self.ledger[x.id])} for x in self.agents.values()],
            "teams": [{"id": t.id, "name": t.name, "charter": t.charter, "founder": t.founder, "created": t.created,
                       "dissolved": t.dissolved} for t in self.teams.values()],
            "blocks": [b.__dict__ for b in self.blocks.values()],
            "custom": [{**c.__dict__} for c in self.custom.values()],
            "entries": [e.__dict__ for b in self.boards.values() for e in b.entries],
            "motions": [{**m.__dict__, "yes": sorted(m.yes), "no": sorted(m.no)} for m in self.motions.values()],
            "events": self.events,
            "snapshots": self.snapshots,
        }
