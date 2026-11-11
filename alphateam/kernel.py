"""The alphateam kernel: substrates, groups, delegation, attention, a ledger and an audit log.

Design the primitives of organization, not the organization. The kernel provides
composable primitives in three families and enforces only what keeps them well
defined (permissions, cycles, budgets):

  information   read, search, write, share, summarize, critique
  coordination  connect (follow, form/join/leave groups, invite, admit, remove, dissolve),
                link (a private two-way channel, open only when both agents agree), delegate (tasks)
  adaptation    reflect, protocol (a group rewrites its own rules), design_action (new actions)

Nothing prescribes roles, teams, stages, hierarchy, leaders or decision rules.
A group may contain agents and other groups, so superagents (a group with its
nested groups) and hierarchies of any depth can emerge, and membership may
overlap: the structure is a DAG, not a tree. A group's own protocol says who may
act for it; by default anyone inside it may.

Communication has real costs. Each agent sees a bounded, local view each turn
(its groups, its inbox, what it follows, headers of new posts), and anything
more must be read or searched for, which spends a per-tick attention budget.
"""
from __future__ import annotations

import inspect
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

INFORMATION = ("read", "search", "write", "share", "summarize", "critique")
COORDINATION = ("connect", "link", "delegate", "message")
ADAPTATION = ("reflect", "protocol", "design_action")
PRIMITIVES = INFORMATION + COORDINATION + ADAPTATION
OPTIONAL = ("rules",)  # offered only where a society's rules are open to proposals (alphateam/recipe.py attach)
CONNECT_OPS = ("follow", "unfollow", "form_group", "join", "leave", "invite", "admit", "remove", "dissolve")
DELEGATE_OPS = ("post", "take", "refuse", "deliver", "review", "drop")
LINK_OPS = ("open", "send", "close")
DESIGN_OPS = ("define", "revise", "adopt", "drop")
SEARCH_KINDS = ("agents", "groups", "posts", "tasks", "actions")
KINDS = ("note", "result", "data", "question", "proposal", "plan", "summary", "critique", "decision", "lesson")
STRUCTURAL = {"form_group", "join", "leave", "admit", "remove", "dissolve"}
MAX_TEXT = 3000
INBOX_CAP = 40
MENTION = re.compile(r"@([AG]\d+)\b")
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
    persona: str = ""
    active: bool = True
    follows: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    actions: set[str] = field(default_factory=set)
    inbox: deque = field(default_factory=lambda: deque(maxlen=INBOX_CAP))
    missed: int = 0  # inbox items pushed out unread
    attention: int = 0  # tokens of reading left this tick
    seen: dict = field(default_factory=dict)  # board id -> entries already shown
    pushed: set = field(default_factory=set)  # sync: entry ids already delivered to this agent (each is delivered once)


@dataclass
class Group:
    id: str
    name: str
    purpose: str
    founder: str
    created: int
    members: set[str] = field(default_factory=set)  # agent ids and group ids
    delegates: set[str] = field(default_factory=set)  # who may act for the group; empty = anyone inside
    open: bool = True
    protocol: str = ""
    invited: set[str] = field(default_factory=set)
    requests: set[str] = field(default_factory=set)
    dissolved: int | None = None


@dataclass
class Link:
    """A private channel between two agents. It carries messages only after both have opened it."""
    pair: tuple[str, str]
    requester: str
    created: int
    status: str = "pending"  # pending | open | closed
    opened: int | None = None

    @property
    def board(self) -> str:
        return f"link:{self.pair[0]}~{self.pair[1]}"

    def other(self, aid: str) -> str:
        return self.pair[1] if self.pair[0] == aid else self.pair[0]


@dataclass
class Task:
    id: str
    title: str
    spec: str
    parent: str | None
    requester: str
    to: str | None
    created: int
    status: str = "open"  # open | taken | delivered | accepted | dropped | refused | lapsed
    taker: str | None = None
    deliverable: str = ""
    reviews: list[dict] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)  # what the task covers: a world may give its taker authority over it
    accept: str = ""  # what would show it is done
    until: int | None = None  # the tick it lapses, when it has one
    held: float = 0.0  # credit held for it, in worlds with credit
    payer: str | None = None
    evidence: list[str] = field(default_factory=list)  # what delivery points to: receipts, entries
    seen: set = field(default_factory=set)  # agents shown it, for reading where delegation stops


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


class Kernel:
    def __init__(self, n_agents: int, max_actions: int = 4, attention: int = 3000,
                 allowed: dict[str, set[str] | None] | None = None, personas: list[str] | None = None):
        self.max_actions = max_actions
        self.attention = attention
        self.allowed = allowed or {p: None for p in PRIMITIVES}
        self.tick = 0
        self.total_ticks = 0
        self.token_budget: int | None = None
        self.cost_budget: float | None = None  # USD; an operator's cap, not shown to agents
        self.cost_fn = None  # () -> USD spent so far, set by the runner
        self.isolated = False  # no-communication control condition
        self.env = None  # the task world (see worlds/)
        self.agents: dict[str, Agent] = {}
        self.groups: dict[str, Group] = {}
        self.member_of: dict[str, set[str]] = defaultdict(set)  # agent or group id -> groups it is a direct member of
        self.boards: dict[str, Board] = {"global": Board("global", "society"), "library": Board("library", "society")}
        self.tasks: dict[str, Task] = {}
        self.task_log: list[dict] = []
        self.links: dict[tuple[str, str], Link] = {}
        self.links_of: dict[str, set] = defaultdict(set)  # agent -> keys of its links
        self.custom: dict[str, CustomAction] = {}
        self.events: list[dict] = []
        # everything, unclipped: what each agent was shown, what it said and did, and what came back (trace.jsonl).
        # events above cut arguments at 160 characters and results at 240; whoever reads a run to say why it went
        # as it did needs the whole of it. Written, never read back: it cannot change what a society does.
        self.trace: list[dict] = []
        self.snapshots: list[dict] = []
        self.by_author: dict[str, list[Entry]] = defaultdict(list)
        self.cited: Counter = Counter()  # entry id -> how many later posts cite it (refs)
        self.index: dict[str, Entry] = {}  # entry id -> entry
        self.retrieved: dict[str, dict] = defaultdict(dict)  # agent -> {entry id: tick last read in full}
        self.running_custom: dict[str, str] = {}  # agent -> custom action it is running
        self.custom_kinds: set[str] = set()  # kinds of entry defined by agents (world option explore)
        self.rank = {"cites": 0.25, "credit": 0.0, "recency": 0.0}  # substrate search weights (a world may evolve them)
        self.substrate = False  # ScaleMAS-style search: relevance + citations, bounded top k (Qi et al. 2026)
        # the whole board (world option board=1, after ScaleMAS's store and Agensh's turn-start delivery, in AlphaTeam's
        # terms): every entry others post reaches each agent once, grouped by type with duplicates counted; search ranks
        # by BM25 relevance plus a citation weight that each citing post moves toward 1 and that decays every tick
        self.full_board = False
        self.cite_alpha, self.cite_decay = 0.3, 0.98
        self.weight: dict[str, tuple[float, int]] = {}  # entry id -> (citation weight, tick it was last moved)
        self.citers: dict[str, list[str]] = defaultdict(list)  # entry id -> what cites it (entry ids, or plays)
        self._terms: dict[str, tuple[Counter, int]] = {}  # entry id -> term counts and length, for BM25
        self._used: set = set()  # (agent, entry, tick) already counted by cite_use
        # sync (world option sync=1, after Agensh's shared context, in AlphaTeam's terms): there are no private boards; the
        # closing note of each turn is public; each turn delivers what others posted since, most relevant first, each
        # entry once; inbox messages and the world's urgent entries reach an agent during its turn, in tool results
        self.sync = False
        self.private_boards = False  # with sync: agents keep private boards and groups their memory boards (world option private=1)
        self.push_items = 8  # sync: entries (after collapsing duplicates) delivered at the start of a turn
        self.continuity = 0  # ticks of its own conversation each agent keeps (world option continuity=W; 0: fresh each tick)
        self.final_moves = False  # the last reply of a tick may still make game moves (world option final_moves=1)
        # islands (world option islands=1, with sync and board=1): an agent's posts reach its island (itself and everyone
        # who shares a group with it) at once, and everyone else when they open: `embargo` ticks after they were posted,
        # when the world says so (e.g. their level fell), or when someone in the island publishes them (share entry=E..,
        # `quota` per `window` ticks per agent). Evidence (posts by the world, play data) reaches everyone at once.
        # Nothing is hidden for good: every post opens to everyone by tick + embargo and is then delivered like any other.
        self.islands: dict | None = None  # {"embargo": int, "quota": int, "window": int, "cap": int}
        self.published: dict[str, tuple[int, str]] = {}  # entry id -> (tick, who published it)
        self.publishes: dict[str, list[int]] = defaultdict(list)  # agent -> ticks of its publishes
        # teaming (world option teaming=1): the prompt says what working with others does, and each turn's view has a
        # MEMBERSHIP block: the agent's island, who holds posts it cannot see yet, groups with places, and the moves
        self.teaming = False
        # messages (world option messages=1): a direct message to an agent or a group, delivered during the recipient's
        # turn if it is acting (with sync) or at its next turn; every message is kept in messages_log
        self.messages_on = False
        self.messages_log: list[dict] = []
        self.message_limit = 8  # per agent per tick
        self.turned_away: dict[str, tuple[int, str]] = {}  # agent -> (tick, group) of its last join refused for size
        self.critiques: Counter = Counter()  # target -> critiques received
        self.ledger: dict[str, Counter] = defaultdict(Counter)  # agent -> category -> tokens
        self.usage: Counter = Counter()  # provider token classes for cost estimates
        self.counts: Counter = Counter()
        self.tick_counts: Counter = Counter()  # reset every tick, for per-tick metrics
        self._ids: Counter = Counter()
        self._ver = 0
        self._up: dict[str, frozenset] = {}
        self._down: dict[str, tuple] = {}
        self._depths: tuple[int, dict] = (-1, {})
        self._total = 0
        self.summarizer = None  # async (kernel, agent_id, text, focus) -> str
        self.custom_runner = None  # async (kernel, agent_id, CustomAction, args) -> str
        for i in range(1, n_agents + 1):
            aid = f"A{i}"
            persona = personas[i - 1] if personas and i - 1 < len(personas) else ""
            self.agents[aid] = Agent(aid, persona)
            self.boards[f"private:{aid}"] = Board(f"private:{aid}", aid)

    # ------------------------------------------------------------------ structure

    def new_id(self, prefix: str) -> str:
        self._ids[prefix] += 1
        return f"{prefix}{self._ids[prefix]}"

    def _changed(self):
        self._ver += 1
        self._up.clear()
        self._down.clear()

    def active_groups(self) -> list[Group]:
        return [g for g in self.groups.values() if g.dissolved is None]

    def active_agents(self) -> list[str]:
        return [a for a, x in self.agents.items() if x.active]

    def containers(self, x: str) -> frozenset:
        """Every group that x (an agent or a group) is inside, directly or through nested groups."""
        hit = self._up.get(x)
        if hit is None:
            out, stack = set(), list(self.member_of.get(x, ()))
            while stack:
                g = stack.pop()
                if g not in out:
                    out.add(g)
                    stack.extend(self.member_of.get(g, ()))
            hit = self._up[x] = frozenset(out)
        return hit

    def inside(self, aid: str, gid: str) -> bool:
        return gid in self.containers(aid)

    def agents_inside(self, gid: str) -> tuple[str, ...]:
        hit = self._down.get(gid)
        if hit is None:
            out, seen, stack = {}, set(), [gid]
            while stack:
                g = stack.pop()
                if g in seen:
                    continue
                seen.add(g)
                for m in self.groups[g].members:
                    if m in self.groups:
                        stack.append(m)
                    else:
                        out[m] = None
            hit = self._down[gid] = tuple(out)
        return hit

    def depth(self, gid: str) -> int:
        """Levels of nesting below and including this group (a group of agents only has depth 1)."""
        if self._depths[0] != self._ver:
            self._depths = (self._ver, {})
        memo = self._depths[1]
        if gid not in memo:
            memo[gid] = 1  # groups cannot contain themselves, but guard anyway
            memo[gid] = 1 + max((self.depth(m) for m in self.groups[gid].members if m in self.groups), default=0)
        return memo[gid]

    def can_act_for(self, aid: str, gid: str, _seen=None) -> bool:
        g = self.groups.get(gid)
        if g is None or g.dissolved is not None:
            return False
        if not g.delegates:
            return self.inside(aid, gid)
        _seen = (_seen or set()) | {gid}
        return any(d == aid or (d in self.groups and d not in _seen and self.can_act_for(aid, d, _seen)) for d in g.delegates)

    def contacts(self, gid: str) -> list[str]:
        """Who hears a group's mail: its delegates, else its direct agent members, else its sub-groups' contacts."""
        g = self.groups[gid]
        direct = [d for d in g.delegates if d in self.agents] or [m for m in g.members if m in self.agents]
        if direct:
            return direct
        out = []
        for m in g.members:
            if m in self.groups:
                out += self.contacts(m)
        return out

    def _add_member(self, gid: str, mid: str):
        self.groups[gid].members.add(mid)
        self.member_of[mid].add(gid)
        self.groups[gid].invited.discard(mid)
        self.groups[gid].requests.discard(mid)
        self._changed()

    def _remove_member(self, gid: str, mid: str):
        g = self.groups[gid]
        g.members.discard(mid)
        g.delegates.discard(mid)
        self.member_of[mid].discard(gid)
        self._changed()
        if not g.members:
            self._dissolve(gid)

    def _dissolve(self, gid: str):
        g = self.groups[gid]
        if g.dissolved is not None:
            return
        g.dissolved = self.tick
        for m in list(g.members):
            self.member_of[m].discard(gid)
        g.members.clear()
        for parent in list(self.member_of.get(gid, ())):
            self._remove_member(parent, gid)
        for t in self.tasks.values():
            if t.status == "taken" and t.to == gid:
                t.status, t.taker = "open", None
        self.counts["dissolved"] += 1
        self.tick_counts["dissolved"] += 1
        self._changed()

    def deactivate(self, aid: str):
        """An agent leaves the society for good (used by churn shocks)."""
        a = self.agents[aid]
        a.active = False
        for gid in list(self.member_of.get(aid, ())):
            self._remove_member(gid, aid)
        for t in self.tasks.values():
            if t.taker == aid and t.status == "taken":
                t.status, t.taker = "open", None

    def label(self, x: str | None) -> str:
        if x in self.groups:
            return f"{x} '{self.groups[x].name}'"
        return str(x)

    # ------------------------------------------------------------------ resolution & permissions

    def _group(self, ref) -> str:
        gid = str(ref or "").strip().removeprefix("mem:").removeprefix("pub:").removeprefix("group:")
        if gid not in self.groups or self.groups[gid].dissolved is not None:
            raise ActionError(f"no active group '{ref}'")
        return gid

    def _agent(self, ref) -> str:
        aid = str(ref or "").strip().removeprefix("private:")
        if aid not in self.agents or not self.agents[aid].active:
            raise ActionError(f"no active agent '{ref}'")
        return aid

    def _member_ref(self, ref) -> str:
        ref = str(ref or "").strip()
        return self._group(ref) if ref.removeprefix("group:").startswith("G") else self._agent(ref)

    def principals(self, aid: str) -> set[str]:
        return {aid, "*", *self.containers(aid)}

    def board_id(self, aid: str, ref) -> str:
        ref = str(ref or "").strip()
        if ref in ("global", "society", "all"):
            return "global"
        if ref == "library":
            return "library"
        if self.sync and not self.private_boards and ref in ("private", "me", "self", ""):
            return "global"  # everything is public: what would have been a private note goes to the global board
        if ref in ("private", "me", "self", ""):
            return f"private:{aid}"
        if self.sync and not self.private_boards and re.fullmatch(r"(private:)?A\d+", ref):
            raise ActionError("there are no private boards: everything is on the shared boards (global, library, groups)")
        if re.fullmatch(r"(private:)?A\d+", ref):
            return f"private:{self._agent(ref)}"
        m = re.fullmatch(r"(mem:|pub:|group:)?(G\d+)", ref)
        if m:
            gid = self._group(m.group(2))
            return f"pub:{gid}" if m.group(1) == "pub:" else f"mem:{gid}"
        if ref.startswith("link:"):
            ids = re.findall(r"A\d+", ref)
            other = next((x for x in ids if x != aid), None)
            ln = self.links.get(self._pair(aid, other)) if other else None
            self._need(ln is not None, f"you have no link with {other or ref}")
            return ln.board
        raise ActionError(f"unknown board '{ref}' (use global, library, private, A<n>, G<n>, pub:G<n> or link:A<n>)")

    def _pair(self, a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b), key=lambda x: int(x[1:])))

    def can_read(self, aid: str, bid: str) -> bool:
        if self.isolated:  # the no-communication control: each agent sees only its own board
            return bid == f"private:{aid}"
        if bid in ("global", "library") or bid.startswith("pub:"):
            return True
        if bid.startswith("link:"):
            return aid in bid[5:].split("~")
        if self.sync and not self.private_boards and bid.startswith("private:"):
            return False
        if self.sync and not self.private_boards and bid.startswith("mem:"):
            return True  # everything is public: a group's memory board is readable by all, writable by its members
        b = self.boards[bid]
        if b.owner == aid or (bid.startswith("mem:") and self.inside(aid, b.owner)):
            return True
        return bool(self.principals(aid) & (b.readers | b.writers))

    def can_write(self, aid: str, bid: str) -> bool:
        if self.isolated:
            return bid == f"private:{aid}"
        if bid == "global" and getattr(self, "local", False) and aid in self.agents:
            return False  # local communication: information travels through links, groups and leads
        if bid in ("global", "library") or bid.startswith("pub:"):
            return True
        if bid.startswith("link:"):
            pair = tuple(bid[5:].split("~"))
            return aid in pair and self.links.get(pair) is not None and self.links[pair].status == "open"
        if self.sync and not self.private_boards and bid.startswith("private:"):
            return False
        b = self.boards[bid]
        if b.owner == aid or (bid.startswith("mem:") and self.inside(aid, b.owner)):
            return True
        return bool(self.principals(aid) & b.writers)

    def _need(self, ok: bool, msg: str):
        if not ok:
            raise ActionError(msg)

    # ------------------------------------------------------------------ islands (world option islands=1)

    def island(self, aid: str) -> set[str]:
        """An agent and every agent that shares a group with it (directly or through nesting)."""
        out = {aid}
        for g in self.containers(aid):
            out.update(self.agents_inside(g))
        return out

    def opens_at(self, e: Entry) -> int:
        """The tick by which an entry is open to everyone, whatever else happens."""
        return e.tick + (self.islands or {}).get("embargo", 0)

    def is_open(self, e: Entry) -> bool:
        """Open to everyone: islands off, a world post or play data, published, past its embargo, or opened by the world."""
        if not self.islands or (e.author not in self.agents and e.author not in self.groups) or e.kind == "data" \
                or e.id in self.published:
            return True
        if self.tick >= self.opens_at(e):
            return True
        fn = getattr(self.env, "entry_open", None)
        return bool(fn and fn(self, e))

    def visible(self, aid: str, e: Entry) -> bool:
        """Whether an agent may see an entry now (board permissions aside). Messages addressed to it (its links, its
        groups' public boards) always reach it."""
        if not self.islands or aid not in self.agents or e.author == aid or self.is_open(e):
            return True
        if e.board.startswith("link:") or (e.board.startswith("pub:") and self.inside(aid, e.board[4:])):
            return True
        if f"@{aid}" in e.content and re.search(rf"@{aid}\b", e.content):  # addressed to it: a message, not a broadcast
            return True
        return e.author in self.island(aid) or (e.author in self.groups and self.inside(aid, e.author))

    def seen_by(self, aid: str, eid: str | None) -> bool:
        """visible() by entry id; ids that are not entries (receipts, plans without one) count as visible."""
        e = self.index.get(str(eid or ""))
        return e is None or self.visible(aid, e)

    def publish_left(self, aid: str) -> int:
        isl = self.islands or {}
        recent = [t for t in self.publishes.get(aid, []) if t > self.tick - isl.get("window", 1)]
        return max(0, isl.get("quota", 0) - len(recent))

    def membership(self, aid: str) -> str:
        """teaming=1: the decision every turn offers. Who is on the agent's island, who holds posts it cannot see yet (by
        author, with their kinds and levels, not their text), which groups have places, and the moves."""
        isl = sorted(self.island(aid) - {aid}, key=lambda x: int(x[1:]))
        lines = ["MEMBERSHIP (this turn, like every turn: stay as you are, or change who you work with)"]
        lines.append(("- Your island: " if self.islands else "- Who shares a group with you: ")
                     + (f"you and {', '.join(isl)}" if isl else "just you (you share no group)")
                     + (f". Posts reach your island at once and others {self.islands['embargo']} ticks later (sooner if "
                        f"their level falls or one of you publishes them: share entry=E.., {self.publish_left(aid)} left)"
                        if self.islands else ""))
        if self.islands:
            held: dict[str, list[Entry]] = defaultdict(list)
            for e in self.hidden_from(aid):
                held[e.author].append(e)
            label = getattr(self.env, "entry_label", None)
            rows = []
            for who, es in sorted(held.items(), key=lambda x: (-len(x[1]), x[0]))[:8]:
                kinds = Counter(label(self, e) if label else e.kind for e in es)
                rows.append(f"{who} {len(es)} (" + ", ".join(f"{n} {k}" if n > 1 else k for k, n in kinds.most_common(4))
                            + f"; opens by t{min(self.opens_at(e) for e in es)})")
            lines.append("- Not yet open to you: " + ("; ".join(rows) if rows else "nothing"))
        cap = (self.islands or {}).get("cap")
        rows = []
        for g in sorted(self.active_groups(), key=lambda g: int(g.id[1:])):
            n = len(self.agents_inside(g.id))
            word = "purpose" if getattr(self.env, "directions_on", False) else "direction"  # directions=1 has its own
            rows.append(f"{g.id} '{clip(g.name, 40)}'" + (f", {word}: {clip(g.purpose, 120)}" if g.purpose else "")
                        + f" ({', '.join(self.agents_inside(g.id)[:6])}"
                        + (f"; {cap - n} places" if cap and n < cap else "; full" if cap else "")
                        + ("" if g.open else "; closed: ask to join") + ")")
        lines.append("- Groups: " + ("; ".join(rows[:8]) if rows else "none yet"))
        alone = sorted((a for a in self.active_agents() if not self.member_of.get(a) and a != aid), key=lambda a: int(a[1:]))
        away = sorted((a for a, (t, _) in self.turned_away.items() if t >= self.tick - 3 and not self.member_of.get(a)
                       and a != aid),
                      key=lambda a: int(a[1:]))
        lines.append(f"- In no group: {', '.join(alone) if alone else 'nobody else'}"
                     + (f"; turned away by a full group lately: {', '.join(away)}" if away else ""))
        lines.append("- Moves (one tool call each): connect op=form_group name=.. purpose=(its "
                     + ("purpose" if getattr(self.env, "directions_on", False) else "direction") + ") invite=[A..]; "
                     "connect op=join target=G..; connect op=invite target=A.. group=G..; connect op=leave target=G..; "
                     + ("message to=A12 (or G1) text=.. reaches them during their turn." if self.messages_on else
                        "writing @A12 in any post reaches A12 at once."))
        return "\n".join(lines)

    def hidden_from(self, aid: str) -> list[Entry]:
        """Entries by other islands that are not open yet: each opens by its embargo tick at the latest."""
        if not self.islands:
            return []
        return [e for e in self.boards["global"].entries if not self.visible(aid, e)]

    # ------------------------------------------------------------------ messaging

    def notify(self, target: str, frm: str, text: str):
        if target in self.agents:
            a = self.agents[target]
            if target != frm and a.active:
                if len(a.inbox) == a.inbox.maxlen:
                    a.missed += 1
                a.inbox.append({"t": self.tick, "from": frm, "text": clip(text, 300)})
        elif target in self.groups and self.groups[target].dissolved is None:
            for m in self.contacts(target):
                self.notify(m, frm, text)

    def post(self, bid: str, author: str, kind: str, content: str, refs=None, via=None, notify=True) -> Entry:
        e = Entry(self.new_id("E"), bid, author, self.tick, kind, str(content)[:MAX_TEXT],
                  [str(r) for r in as_list(refs)][:12], via)
        self.boards[bid].entries.append(e)
        self.by_author[author].append(e)
        self.index[e.id] = e
        cites = {r for r in e.refs if re.fullmatch(r"E\d+", r) and r != e.id}
        for r in cites:
            self.cited[r] += 1
        if self.full_board:  # an entry named in the text counts as used when someone else wrote it
            named = {r for r in re.findall(r"\bE\d+\b", e.content) if r != e.id and r not in cites and r in self.index
                     and self.index[r].author != author}
            for r in named:
                self.cited[r] += 1
            for r in cites | named:  # only use by others moves the weight (engine refs to one's own plan do not)
                if r in self.index and self.index[r].author != author:
                    self.citers[r].append(e.id)
                    self._cite(r)
        if notify:
            for m in set(MENTION.findall(e.content)):
                if self.islands and m in self.agents and not self.visible(m, e):
                    continue  # another island: it reaches them when it opens
                if self.islands and m in self.groups:
                    for c in self.contacts(m):
                        if self.visible(c, e):
                            self.notify(c, author, f"mentioned {m} in {e.id} on {bid}: {clip(e.content, 200)}")
                elif m in self.agents or m in self.groups:
                    self.notify(m, author, f"mentioned you in {e.id} on {bid}: {clip(e.content, 200)}")
            owner = self.boards[bid].owner
            if bid.startswith("pub:"):
                self.notify(owner, author, f"wrote {e.id} to your group's public board {bid}: {clip(e.content, 200)}")
            elif bid.startswith("private:") and owner != author:
                self.notify(owner, author, f"wrote {e.id} on your private board: {clip(e.content, 200)}")
        return e

    def cite_weight(self, eid: str) -> float:
        """An entry's citation weight in [0, 1): each citing post or play moves it a fraction cite_alpha toward 1, and it
        decays by cite_decay every tick, so what the society keeps using ranks first and what it stopped using fades."""
        w, t = self.weight.get(eid, (0.0, self.tick))
        return w * self.cite_decay ** max(0, self.tick - t)

    def _cite(self, eid: str):
        w = self.cite_weight(eid)
        self.weight[eid] = (w + self.cite_alpha * (1 - w), self.tick)

    def cite_use(self, aid: str, ids, what: str):
        """A citation that is not a post (e.g. a play naming the pieces it acts on): counted like a citing post."""
        for r in dict.fromkeys(ids):
            e = self.index.get(r)
            if e is not None and e.author != aid and (aid, r, self.tick) not in self._used:  # once per agent per tick
                self._used.add((aid, r, self.tick))
                self.cited[r] += 1
                self.citers[r].append(what)
                self._cite(r)

    def _bm25(self, words: list[str], entries: list[Entry]) -> dict[str, float]:
        """BM25 relevance (k1 1.5, b 0.75) of each entry to the query words; only entries containing a word score."""
        for e in entries:
            if e.id not in self._terms:
                toks = re.findall(r"[a-z0-9]+", e.content.lower())
                self._terms[e.id] = (Counter(toks), len(toks))
        n = len(entries) or 1
        avg = sum(self._terms[e.id][1] for e in entries) / n or 1.0
        q = list(dict.fromkeys(t for w in words for t in re.findall(r"[a-z0-9]+", w)))
        df = {t: sum(1 for e in entries if t in self._terms[e.id][0]) for t in q}
        out = {}
        for e in entries:
            tf, ln = self._terms[e.id]
            s = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * ln / avg))
                    for t in q if tf[t])
            if s > 0:
                out[e.id] = s
        return out

    def kinds_allowed(self) -> tuple[str, ...]:
        return KINDS + tuple(sorted(self.custom_kinds))

    def entry(self, eid: str) -> Entry | None:
        return self.index.get(str(eid).strip())

    def fmt(self, e: Entry, limit: int | None = 600) -> str:
        body = e.content if limit is None else clip(e.content, limit)
        via = f" via x_{e.via}" if e.via else ""
        refs = f" refs={','.join(e.refs)}" if e.refs else ""
        return f"[{e.id} t{e.tick} {e.author} {e.kind}{via}{refs}] {body}"

    def header(self, e: Entry) -> str:
        cited = f" cited {self.cited[e.id]}x" if self.substrate and self.cited[e.id] else ""
        ran = self.env.entry_note(e.id) if self.env is not None and hasattr(self.env, "entry_note") else ""
        return f"{e.id} t{e.tick} {e.author} {e.kind}{cited}{ran}: {clip(e.content, 90)}"

    # ------------------------------------------------------------------ sync (world option sync=1)

    def end_turn(self, aid: str, note: str):
        """An agent's closing text: a note to its future self, on its private board or, with sync, on the global board."""
        if note.strip() and self.agents[aid].active:
            self.post("global" if self.sync and not self.private_boards else f"private:{aid}", aid, "note", note.strip()[:1200], notify=False)

    def gist(self, e: Entry, n: int = 140) -> str:
        """An entry's id, author and first line: what an agent receives without reading the entry."""
        first = next((ln for ln in e.content.splitlines() if ln.strip()), "")
        return f"{e.id} t{e.tick} {e.author} {e.kind}: {clip(first.strip(), n)}"

    def _rank(self, aid: str, e: Entry) -> int:
        """0 urgent (also delivered during a turn), 1 relevant now, 2 related, 3 other; the world decides relevance."""
        fn = getattr(self.env, "sync_rank", None)
        return fn(self, aid, e) if fn else 2

    def _push(self, aid: str) -> str:
        """What others posted on the global board since this agent's last turn, most relevant first, each entry once;
        entries with the same text (ids and ticks aside) are collapsed into one line with a count."""
        a = self.agents[aid]
        entries = self.boards["global"].entries
        start = a.seen.get("global", 0)
        a.seen["global"] = len(entries)
        a.seen["global!"] = max(a.seen.get("global!", 0), len(entries))
        # posts that mention this agent already reached it as inbox messages
        fresh = [e for e in entries[start:] if e.author != aid and e.id not in a.pushed
                 and not re.search(rf"@{aid}\b", e.content)]
        held = ""
        if self.islands:  # everything not yet delivered that is visible now, posts that just opened among them
            fresh = [e for e in entries if e.author != aid and e.id not in a.pushed and self.visible(aid, e)]
            hidden = self.hidden_from(aid)
            if hidden:
                held = (f"\n({len(hidden)} posts by other islands are not open to you yet; each opens by its embargo tick "
                        f"(the next by t{min(self.opens_at(e) for e in hidden)}), sooner if its level falls or its island "
                        f"publishes it, and then reaches you here.)")
        a.pushed.update(e.id for e in fresh)
        if not fresh:
            return "NEW SINCE YOUR LAST TURN: nothing." + held
        if self.full_board:
            return self._deliver_all(aid, fresh) + held
        groups: dict[str, list] = {}
        order = {e.id: i for i, e in enumerate(fresh)}
        for e in sorted(fresh, key=lambda e: (self._rank(aid, e), -order[e.id])):  # most relevant, then newest
            key = re.sub(r"\b([aerpgc]\d+|t\d+)\b", "", e.content.lower())
            groups.setdefault(" ".join(key.split())[:100], []).append(e)
        rows = list(groups.values())
        shown, rest = rows[:self.push_items], rows[self.push_items:]
        lines = []
        for g in shown:
            more = f" (x{len(g)}: also {', '.join(sorted({x.author for x in g[1:]}))})" if len(g) > 1 else ""
            lines.append(f"- {self.gist(g[0])}{more}")
        self.counts["pushed"] += sum(len(g) for g in shown)
        self.counts["collapsed"] += sum(len(g) - 1 for g in shown)
        if rest:
            near = sum(1 for g in rest if self._rank(aid, g[0]) <= 2)
            lines.append(f"- +{sum(len(g) for g in rest)} more ({near} related to your work): read global or search "
                         f"what=posts to see them")
        return f"NEW SINCE YOUR LAST TURN ({len(fresh)} posts by others, most relevant to you first; each is shown once, "\
               f"read entry=<id> for the full text):\n" + "\n".join(lines)

    def _deliver_all(self, aid: str, fresh: list[Entry]) -> str:
        """board=1: every entry others posted since this agent's last turn, each once, grouped by type (the world names
        the groups and how much of each entry's first line to show); entries with the same text are one line with a
        count. Nothing is left out: the full text is one read away."""
        fn = getattr(self.env, "deliver_group", None)
        tag = getattr(self.env, "entry_tag", None)  # e.g. what the entry can be trusted at (trust=1)
        sections: dict[tuple, dict[str, list]] = {}
        for e in fresh:
            order, label, n = fn(self, aid, e) if fn else (5, "Posts", 120)
            key = " ".join(re.sub(r"\b([aerpgc]\d+|t\d+)\b", "", e.content.lower()).split())[:100]
            sections.setdefault((order, label, n), {}).setdefault(key, []).append(e)
        lines = []
        okey = getattr(self.env, "deliver_order", None)  # e.g. directions=1: within a section, the agent's own direction first
        for (order, label, n), groups in sorted(sections.items()):
            rows = []
            for g in (sorted(groups.values(), key=lambda g: min(okey(self, aid, x) for x in g)) if okey else groups.values()):
                more = f" (x{len(g)}: also {', '.join(sorted({x.author for x in g[1:]}))})" if len(g) > 1 else ""
                rows.append(f"- {self.gist(g[0], n)}{tag(self, g[0]) if tag else ''}{more}")
            lines.append(f"{label} ({sum(len(g) for g in groups.values())}):\n" + "\n".join(rows))
            self.counts["pushed"] += sum(len(g) for g in groups.values())
            self.counts["collapsed"] += sum(len(g) - 1 for g in groups.values())
        return (f"NEW SINCE YOUR LAST TURN: all {len(fresh)} posts by others, each shown once, grouped by type (read "
                f"entry=<id> for the full text; search what=posts finds anything older):\n" + "\n".join(lines))

    def interrupts(self, aid: str, limit: int = 6) -> str:
        """With sync, delivered during a turn (appended to a tool result): new inbox messages, and urgent entries others
        posted since the agent last looked (the world decides what is urgent, e.g. a dead end on the agent's problem)."""
        if not self.sync or aid not in self.agents:
            return ""
        a = self.agents[aid]
        lines = []
        while a.inbox and len(lines) < limit:
            n = a.inbox.popleft()
            lines.append(f"- t{n['t']} {n['from']}: {n['text']}")
        entries = self.boards["global"].entries
        start = max(a.seen.get("global!", 0), a.seen.get("global", 0))
        for i in range(start, len(entries)):
            if len(lines) >= limit:
                break
            e = entries[i]
            if self.islands and not self.visible(aid, e):
                continue  # not open to this agent yet: the turn-start delivery brings it when it opens
            a.seen["global!"] = i + 1
            if e.author != aid and e.id not in a.pushed and self._rank(aid, e) == 0:
                a.pushed.add(e.id)
                lines.append(f"- {self.gist(e)}")
        if not lines:
            return ""
        self.counts["interrupts"] += len(lines)
        return "\n[NEW WHILE YOU WORK]\n" + "\n".join(lines)

    # ------------------------------------------------------------------ accounting

    def charge(self, aid: str, category: str, tokens: int):
        self.ledger[aid][category] += int(tokens)
        self.tick_counts[f"tok_{category}"] += int(tokens)
        self._total += int(tokens)

    def total_tokens(self) -> int:
        return self._total

    def budget_left(self) -> int | None:
        return None if self.token_budget is None else self.token_budget - self.total_tokens()

    def exhausted(self) -> bool:
        """True once the token budget or the operator's spending cap (USD) is used up."""
        left = self.budget_left()
        if left is not None and left <= 0:
            return True
        return self.cost_budget is not None and self.cost_fn is not None and (self.cost_fn() or 0) >= self.cost_budget

    def spend_attention(self, aid: str, text: str) -> bool:
        cost = est_tokens(text)
        a = self.agents[aid]
        if cost > a.attention:
            return False
        a.attention -= cost
        self.charge(aid, "read", cost)
        return True

    def begin_turn(self, aid: str):
        self.agents[aid].attention = self.attention

    # ------------------------------------------------------------------ dispatch

    async def act(self, aid: str, name: str, args: dict | None, via: str | None = None) -> tuple[bool, str]:
        args = dict(args or {}) if isinstance(args, dict) else {}
        target = None
        # refusals are logged under the same family.op label a success would get
        if self.env is not None and name in self.env.tool_names:
            op_label = f"lab.{str(args.get('op') or name).strip()}"
        else:
            family = {"design_action": "design"}.get(name, name)
            op_label = f"{family}.{str(args['op']).strip()}" if args.get("op") else family
        try:
            self._need(self.agents[aid].active, "you have left the society")
            if (name in PRIMITIVES or name in OPTIONAL) and name in self.allowed:
                op = str(args.get("op", "")).strip()
                ops = self.allowed[name]
                if ops is not None and op and op not in ops:
                    raise ActionError(f"op '{op}' is not available in this society")
                op_label, msg, target = await getattr(self, f"_do_{name}")(aid, args, via)
            elif self.env is not None and name in self.env.tool_names:
                res = self.env.act(self, aid, name, args)
                if inspect.isawaitable(res):  # slow benchmark tools (running code, simulating) don't block other agents
                    res = await res
                op_label, msg, target = res
            elif name.startswith("x_") and "design_action" in self.allowed:
                op_label, msg = await self._run_custom(aid, name[2:], args, via)
            else:
                raise ActionError(f"unknown action '{name}'")
            ok = True
        except ActionError as exc:
            ok, msg = False, f"refused: {exc}"
        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
            ok, msg = False, f"refused: malformed arguments ({exc})"
        family = op_label.split(".")[0]
        if ok and family == "connect" and op_label.split(".")[1] in STRUCTURAL:
            self.counts["reorganizations"] += 1
            self.tick_counts["reorganizations"] += 1
        self.counts[family] += 1
        self.tick_counts[family] += 1
        self.events.append({
            "t": self.tick, "agent": aid, "op": op_label, "ok": ok, "target": target, "via": via,
            "args": {k: clip(v if isinstance(v, str) else json.dumps(v), 160) for k, v in args.items()},
            "msg": clip(msg, 240),
        })
        self.trace.append({"t": self.tick, "agent": aid, "kind": "action", "op": op_label, "ok": ok, "args": args, "result": msg})
        return ok, msg

    def log_intent(self, aid: str, text: str):
        if text and text.strip():
            self.events.append({"t": self.tick, "agent": aid, "op": "intent", "ok": True, "target": None,
                                "via": None, "args": {}, "msg": clip(text, 300)})

    # ------------------------------------------------------------------ information

    async def _do_read(self, aid, a, via):
        bid = self.board_id(aid, a.get("board", "global"))
        self._need(self.can_read(aid, bid), f"you cannot read {bid}")
        entries = self.boards[bid].entries
        if self.islands:
            want = str(a.get("entry") or "").strip()
            e0 = self.index.get(want)
            self._need(not e0 or self.visible(aid, e0), f"{want} is by another island and opens to you by "
                       f"t{self.opens_at(e0) if e0 else 0} (sooner if its level falls or its island publishes it)")
            entries = [e for e in entries if self.visible(aid, e)]
        if a.get("kind"):
            entries = [e for e in entries if e.kind == a["kind"]]
        if a.get("author"):
            entries = [e for e in entries if e.author == str(a["author"]).strip()]
        if a.get("contains"):
            words = str(a["contains"]).lower().split()
            entries = [e for e in entries if all(w in e.content.lower() for w in words)]
        if a.get("entry"):
            entries = [e for e in entries if e.id == str(a["entry"]).strip()]
        last = max(1, min(30, int(a.get("last") or 8)))
        limit = None if a.get("full") else 600
        lines, shown = [], 0
        for e in reversed(entries[-last:]):
            line = self.fmt(e, limit)
            if not self.spend_attention(aid, line):
                lines.append("(attention exhausted for this tick)")
                break
            lines.append(line)
            shown += 1
            self.retrieved[aid][e.id] = self.tick
        body = "\n".join(reversed(lines)) or "(no entries)"
        return "read", f"{bid}: {len(entries)} matching, showing {shown} (attention left {self.agents[aid].attention})\n{body}", bid

    async def _do_search(self, aid, a, via):
        what = str(a.get("what", "posts")).strip()
        words = str(a.get("query", "")).lower().split()
        limit = max(1, min(20, int(a.get("limit") or 8)))
        match = (lambda text: all(w in text.lower() for w in words)) if words else (lambda text: True)
        if what == "agents":
            pool = [x for x in self.active_agents() if x != aid]
            if not words:
                random.Random(self.tick * 7919 + int(aid[1:])).shuffle(pool)
            followers = Counter(f for y in self.agents.values() for f in y.follows)
            ids = [x for x in pool if match(self.agent_key(x))]  # cheap match first, full cards only for what is shown
            hits = [self.agent_card(x, followers) for x in ids[:limit]] + ids[limit:]
        elif what == "groups":
            cards = (self.group_card(g.id) for g in sorted(self.active_groups(), key=lambda g: -len(self.agents_inside(g.id))))
            hits = [c for c in cards if match(c)]
        elif what == "tasks":
            hits = [self.task_card(t) for t in reversed(self.tasks.values()) if t.status != "dropped" and match(self.task_card(t))]
        elif what == "actions":
            hits = [f"x_{c.name} v{c.version} by {c.author}, {sum(c.name in x.actions for x in self.agents.values())} adopters, "
                    f"{c.uses} uses: {clip(c.description, 140)}" for c in self.custom.values() if match(c.name + " " + c.description)]
        elif what == "posts" and self.full_board:  # weight + BM25 relevance / its best value; the weight only reorders
            pool = [e for bid, b in self.boards.items() if b.entries and self.can_read(aid, bid) for e in b.entries
                    if self.visible(aid, e)]
            rel = self._bm25(words, pool) if words else {e.id: 1.0 for e in pool}
            top = max(rel.values(), default=1.0)
            scored = sorted(((self.cite_weight(e.id) + rel[e.id] / top, int(e.id[1:]), e) for e in pool if e.id in rel),
                            key=lambda x: (-x[0], -x[1]))
            hits = []
            tag = getattr(self.env, "entry_tag", None)
            for _, _, e in scored[:limit]:
                cites = [r for r in e.refs if r in self.index][:8]
                by = self.citers.get(e.id, [])[-8:]
                hits.append(self.header(e) + f" [{e.board}]" + (tag(self, e) if tag else "")
                            + (f" cites {', '.join(cites)}" if cites else "")
                            + (f"; cited by {', '.join(by)}" if by else ""))
            hits += [None] * (len(scored) - len(hits))  # counted, not shown
        elif what == "posts" and self.substrate:  # relevance (query words present) + citations, newest breaks ties
            scored = []
            for bid, b in self.boards.items():
                if b.entries and self.can_read(aid, bid):
                    for e in b.entries:
                        if not self.visible(aid, e):
                            continue
                        text = e.content.lower()
                        rel = sum(w in text for w in words) / len(words) if words else 1.0
                        if rel > 0:
                            w = self.rank
                            credit = getattr(self.env, "credit", None) or {}
                            earned = credit.get(e.author, 0.0)
                            earned = earned[0] if isinstance(earned, (list, tuple)) else earned  # a world may keep [credit, changes]
                            scored.append((rel + w["cites"] * math.log1p(self.cited[e.id])
                                           + w["credit"] * math.log1p(max(0.0, earned))
                                           + w["recency"] / (1 + self.tick - e.tick), int(e.id[1:]), e))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            hits = [self.header(e) + f" [{e.board}]" for _, _, e in scored]
        elif what == "posts":
            hits = []
            for bid, b in self.boards.items():
                if b.entries and self.can_read(aid, bid):
                    hits += [e for e in b.entries if match(e.content) and self.visible(aid, e)]
            hits = [self.header(e) + f" [{e.board}]" for e in sorted(hits, key=lambda e: -int(e.id[1:]))]
        elif self.env is not None and what in self.env.search_kinds:
            hits = self.env.search(self, aid, what, words)
        else:
            kinds = SEARCH_KINDS + (tuple(self.env.search_kinds) if self.env else ())
            raise ActionError(f"search what must be one of {', '.join(kinds)}")
        out, shown = [], 0
        for h in hits[:limit]:
            if not self.spend_attention(aid, h):
                out.append("(attention exhausted for this tick)")
                break
            out.append("- " + h)
            shown += 1
        return f"search.{what}", f"{len(hits)} {what} match, showing {shown}:\n" + ("\n".join(out) or "(none)"), None

    def agent_key(self, x: str) -> str:
        """The searchable text of an agent: id, persona, world tags (e.g. instruments), groups, track record."""
        tags = self.env.agent_tags(x) if self.env else ""
        record = self.env.track_record(x) if self.env else ""
        return f"{x} {self.agents[x].persona} {tags} {' '.join(self.member_of.get(x, ()))} {record}"

    def agent_card(self, x: str, followers: Counter | None = None) -> str:
        groups = sorted(self.member_of.get(x, ()), key=lambda g: int(g[1:]))
        followers = followers[x] if followers is not None else sum(x in y.follows for y in self.agents.values())
        tags = self.env.agent_tags(x) if self.env else ""
        record = self.env.track_record(x) if self.env else ""
        return (f"{x}{' (' + clip(self.agents[x].persona, 60) + ')' if self.agents[x].persona else ''} {tags} | groups "
                f"{','.join(groups) or 'none'} | followers {followers} | links {len(self.open_links(x))}"
                f"{' | ' + record if record else ''}")

    def group_card(self, gid: str) -> str:
        g = self.groups[gid]
        subs = [m for m in g.members if m in self.groups]
        parents = sorted(self.member_of.get(gid, ()))
        record = self.env.group_record(gid) if self.env is not None and hasattr(self.env, "group_record") else ""
        return (f"{gid} '{g.name}': {len(self.agents_inside(gid))} agents inside, {len(g.members)} members"
                f"{' | ' + record + ' |' if record else ''}"
                f"{' (' + str(len(subs)) + ' groups)' if subs else ''}{', inside ' + ','.join(parents) if parents else ''}, "
                f"{'open' if g.open else 'closed'}, delegates {','.join(sorted(g.delegates)) or 'anyone inside'}; "
                f"purpose: {clip(g.purpose, 120)}{'; protocol: ' + clip(g.protocol, 120) if g.protocol else ''}")

    def task_card(self, t: Task) -> str:
        return (f"{t.id} [{t.status}] '{t.title}' from {t.requester} to {t.to or 'anyone'}"
                f"{' taken by ' + t.taker if t.taker else ''}{' parent ' + t.parent if t.parent else ''}"
                f"{' scope ' + ', '.join(t.scope) if t.scope else ''}{f' until t{t.until}' if t.until else ''}"
                f"{f' pay {t.held:g}' if t.held else ''}{' done when: ' + clip(t.accept, 100) if t.accept else ''}: {clip(t.spec, 140)}")

    async def _do_write(self, aid, a, via):
        bid = self.board_id(aid, a.get("board", "private"))
        self._need(self.can_write(aid, bid), f"you cannot write to {bid}")
        content = str(a.get("content", "")).strip()
        self._need(bool(content), "content is empty")
        kind = a.get("kind") if a.get("kind") in self.kinds_allowed() else "note"
        if self.env is not None and hasattr(self.env, "check_entry"):
            self.env.check_entry(kind, content)  # agent-defined kinds may carry an interface (required fields)
        e = self.post(bid, aid, kind, content, a.get("refs"), via)
        self.charge(aid, "communication", est_tokens(content))
        return f"write.{bid.split(':')[0]}", f"posted {e.id} to {bid}", bid

    async def _do_message(self, aid, a, via):
        """messages=1: a direct message to an agent or a group, kept in messages_log; with islands it is not held back."""
        self._need(self.messages_on, "there is no message action in this society")
        to = str(a.get("to") or a.get("target") or "").strip().upper().removeprefix("@")
        text = " ".join(str(a.get("text") or a.get("content") or "").split())
        self._need(bool(text), "text: what to say")
        self._need(len(text) <= 800, "a message is at most 800 characters")
        sent = sum(1 for m in self.messages_log if m["from"] == aid and m["t"] == self.tick)
        self._need(sent < self.message_limit, f"at most {self.message_limit} messages per tick")
        if to in self.agents:
            self._need(to != aid, "that is you")
            whom = [to]
        else:
            gid = self._group(to)
            whom = [x for x in self.agents_inside(gid) if x != aid]
            self._need(bool(whom), f"nobody else is in {gid}")
        mid = self.new_id("M")
        self.messages_log.append({"id": mid, "t": self.tick, "from": aid, "to": to, "text": text,
                                  "reply_to": str(a.get("reply_to") or "") or None})
        for x in whom:
            me = self.agents[x]
            if me.active:
                if len(me.inbox) == me.inbox.maxlen:
                    me.missed += 1
                me.inbox.append({"t": self.tick, "from": aid, "text": f"MESSAGE {mid}" + (f" to {to}" if to != x else "")
                                 + f": {text}"})
        self.counts["messages"] += 1
        self.tick_counts["messages"] += 1
        return "message", f"sent {mid} to {to} ({', '.join(whom)})", to

    async def _do_share(self, aid, a, via):
        if a.get("entry"):  # islands: publish one of your island's entries to everyone now
            self._need(bool(self.islands), "there are no islands: every post already reaches everyone")
            eid = str(a["entry"]).strip().upper()
            e = self.index.get(eid)
            self._need(e is not None, f"no entry {eid}")
            self._need(e.author in self.island(aid) or (e.author in self.groups and self.inside(aid, e.author)),
                       f"{eid} is by {e.author}, not by you or your island")
            self._need(not self.is_open(e), f"{eid} is already open to everyone")
            self._need(self.publish_left(aid) > 0, f"you may publish {self.islands['quota']} entries per "
                       f"{self.islands['window']} ticks and have used them; {eid} opens to everyone by t{self.opens_at(e)}")
            self.published[eid] = (self.tick, aid)
            self.publishes[aid].append(self.tick)
            self.counts["published"] += 1
            self.tick_counts["published"] += 1
            return "share.publish", f"published {eid} to everyone ({self.publish_left(aid)} more this window)", eid
        bid = self.board_id(aid, a.get("board", "private"))
        b = self.boards[bid]
        if bid.startswith("private:"):
            self._need(b.owner == aid, "you can only share your own private board")
        elif bid.startswith("mem:"):
            self._need(self.can_act_for(aid, b.owner), f"you may not act for {b.owner}")
        else:
            raise ActionError(f"{bid} is already public")
        whom = ["*" if w in ("all", "*", "everyone") else self._member_ref(w) for w in as_list(a.get("with"))]
        self._need(bool(whom), "share with whom?")
        mode = a.get("mode", "read")
        for w in whom:
            if mode == "revoke":
                b.readers.discard(w)
                b.writers.discard(w)
            else:
                (b.writers if mode == "write" else b.readers).add(w)
                if w != "*":
                    self.notify(w, aid, f"shared {bid} with you ({mode}); use read board='{bid.split(':', 1)[1]}'")
        return f"share.{mode}", f"{bid}: {mode} for {', '.join(whom)}", bid

    async def _do_summarize(self, aid, a, via):
        sources = [self.board_id(aid, s) for s in as_list(a.get("sources") or a.get("source"))]
        self._need(bool(sources), "give sources")
        for s in sources:
            self._need(self.can_read(aid, s), f"you cannot read {s}")
        dest = self.board_id(aid, a.get("to", "private"))
        self._need(self.can_write(aid, dest), f"you cannot write to {dest}")
        text = "\n".join(self.fmt(e, 500) for s in sources for e in self.boards[s].entries[-40:])
        self._need(bool(text.strip()), "nothing to summarize")
        self._need(self.summarizer is not None, "no summarizer attached")
        summary = await self.summarizer(self, aid, text, str(a.get("focus", "")))
        e = self.post(dest, aid, "summary", summary, [self.boards[s].entries[-1].id for s in sources if self.boards[s].entries], via)
        return "summarize", f"posted summary {e.id} to {dest}:\n{clip(summary, 500)}", dest

    async def _do_critique(self, aid, a, via):
        target = str(a.get("target", "society")).strip()
        content = str(a.get("content", "")).strip()
        self._need(bool(content), "content is empty")
        text = content + (f"\nProposal: {a['proposal']}" if a.get("proposal") else "")
        if target in ("society", "all", "everyone"):
            e = self.post("global", aid, "critique", text, via=via)
            where = "global"
        elif target.removeprefix("group:").startswith("G"):
            gid = self._group(target)
            e = self.post(f"pub:{gid}", aid, "critique", text, via=via)
            where = f"pub:{gid}"
        else:
            tid = self._agent(target)
            self._need(tid != aid, "use reflect for yourself")
            self.notify(tid, aid, f"critique: {text}")
            e, where = None, tid
        self.critiques[where if where != "global" else "society"] += 1
        self.charge(aid, "critique", est_tokens(text))
        return f"critique.{'agent' if e is None else where.split(':')[0]}", f"critique delivered to {where}", where

    # ------------------------------------------------------------------ coordination

    def _as(self, aid: str, a: dict) -> str:
        """The member an action is on behalf of: yourself, or a group you may act for."""
        who = str(a.get("as") or "").strip()
        if not who or who == aid:
            return aid
        gid = self._group(who)
        self._need(self.can_act_for(aid, gid), f"you may not act for {gid} (see its delegates)")
        return gid

    async def _do_connect(self, aid, a, via):
        op = str(a.get("op", "")).strip()
        me = self.agents[aid]
        if op in ("follow", "unfollow"):
            target = self._member_ref(a.get("target"))
            self._need(target != aid, "cannot follow yourself")
            if op == "follow":
                me.follows.add(target)
                self.notify(target, aid, f"{aid} follows you")
            else:
                me.follows.discard(target)
            return f"connect.{op}", f"{op}ed {target}", target
        if op == "form_group":
            name = clip(a.get("name"), 60)
            self._need(bool(name), "a group needs a name")
            founder = self._as(aid, a)
            gid = self.new_id("G")
            self.groups[gid] = Group(gid, name, str(a.get("purpose") or "").strip()[:800], aid, self.tick,
                                     open=a.get("open", True) is not False, protocol=str(a.get("protocol") or "")[:800])
            self.boards[f"mem:{gid}"] = Board(f"mem:{gid}", gid)
            self.boards[f"pub:{gid}"] = Board(f"pub:{gid}", gid)
            self._add_member(gid, founder)
            invited = []
            for inv in as_list(a.get("invite")):
                try:
                    m = self._member_ref(inv)
                except ActionError:
                    continue
                if m != founder:
                    self.groups[gid].invited.add(m)
                    self.notify(m, aid, f"invites you to join {gid} '{name}': {clip(self.groups[gid].purpose, 160)}")
                    invited.append(m)
            self.counts["formed"] += 1
            self.tick_counts["formed"] += 1
            return "connect.form_group", f"formed {gid} '{name}' with member {founder}" + (
                f"; invited {', '.join(invited)}" if invited else ""), gid
        gid = self._group(a.get("target") if op in ("join", "leave", "dissolve") else (a.get("group") or a.get("to") or a.get("from")))
        g = self.groups[gid]
        if op == "join":
            member = self._as(aid, a)
            self._need(member not in g.members, f"{member} is already in {gid}")
            self._need(member != gid and (member not in self.groups or gid not in self.agents_inside_groups(member)),
                       f"{gid} is inside {member}; joining would create a cycle")
            if not self.fits(gid, self._agents_of(member)):
                raise ActionError(self._full(gid, member))
            if g.open or member in g.invited:
                self._add_member(gid, member)
                self.notify(gid, aid, f"{member} joined {gid}")
                return "connect.join", f"{member} joined {gid}", gid
            g.requests.add(member)
            self.notify(gid, aid, f"{member} asks to join closed group {gid}; admit with connect op=admit")
            return "connect.request", f"{gid} is closed; requested admission for {member}", gid
        if op == "leave":
            member = self._as(aid, a)
            self._need(member in g.members, f"{member} is not a member of {gid}")
            self._remove_member(gid, member)
            return "connect.leave", f"{member} left {gid}" + (" (it dissolved)" if g.dissolved is not None else ""), gid
        if op == "invite":
            target = self._member_ref(a.get("target"))
            self._need(self.inside(aid, gid), f"you are not inside {gid}")
            self._need(g.open or self.can_act_for(aid, gid), f"{gid} is closed; only those who may act for it can invite")
            g.invited.add(target)
            self.notify(target, aid, f"invites you to join {gid} '{g.name}': {clip(g.purpose, 160)} (connect op=join target={gid})")
            return "connect.invite", f"invited {target} to {gid}", target
        self._need(self.can_act_for(aid, gid), f"you may not act for {gid} (delegates: {','.join(sorted(g.delegates)) or 'anyone inside'})")
        if op == "admit":
            target = self._member_ref(a.get("target"))
            self._need(target in g.requests or target in g.invited, f"{target} has not asked to join {gid}")
            if not self.fits(gid, self._agents_of(target)):
                raise ActionError(self._full(gid, target))
            self._add_member(gid, target)
            self.notify(target, aid, f"you were admitted to {gid}")
            return "connect.admit", f"admitted {target} to {gid}", gid
        if op == "remove":
            target = self._member_ref(a.get("target"))
            self._need(target in g.members, f"{target} is not a member of {gid}")
            self._remove_member(gid, target)
            self.notify(target, aid, f"you were removed from {gid}")
            return "connect.remove", f"removed {target} from {gid}", gid
        if op == "dissolve":
            self._dissolve(gid)
            return "connect.dissolve", f"dissolved {gid}", gid
        raise ActionError(f"unknown connect op '{op}'")

    def _agents_of(self, x: str) -> set[str]:
        return set(self.agents_inside(x)) if x in self.groups else {x}

    def _full(self, gid: str, who: str) -> str:
        """The refusal for a full group: who else was turned away and what starts another group."""
        if who in self.agents:
            self.turned_away[who] = (self.tick, gid)
        others = sorted((a for a, (t, _) in self.turned_away.items() if a != who and t >= self.tick - 3
                         and not self.member_of.get(a)), key=lambda a: int(a[1:]))
        return (f"{gid} (or a group containing it) would have more than {(self.islands or {}).get('cap')} agents"
                + (f"; also turned away lately and still in no group: {', '.join(others)}" if others else "")
                + "; connect op=form_group name=.. purpose=.. invite=[..] starts another group")

    def fits(self, gid: str, newcomers: set[str]) -> bool:
        """islands: whether gid and every group containing it stay within the group-size cap with these agents added."""
        if not self.islands:
            return True
        return all(len(set(self.agents_inside(g)) | newcomers) <= self.islands["cap"] for g in {gid, *self.containers(gid)})

    def agents_inside_groups(self, gid: str) -> set[str]:
        """Groups nested (at any depth) inside gid."""
        out, stack = set(), [gid]
        while stack:
            for m in self.groups[stack.pop()].members:
                if m in self.groups and m not in out:
                    out.add(m)
                    stack.append(m)
        return out

    async def _do_link(self, aid, a, via):
        op = str(a.get("op", "")).strip() or ("send" if a.get("content") else "open")  # a message with no op is a send
        other = self._agent(a.get("target"))
        self._need(other != aid, "you cannot link to yourself")
        key = self._pair(aid, other)
        ln = self.links.get(key)
        level = self.same_level(aid, other)
        if level and op in ("open", "send") and (ln is None or ln.status != "open"):
            # joining a group is consent to talk with the others at that level, so no handshake is needed
            ln = self.links[key] = Link(key, aid, self.tick, "open", self.tick)
            self.boards.setdefault(ln.board, Board(ln.board, f"{key[0]}~{key[1]}"))
            self.links_of[aid].add(key)
            self.links_of[other].add(key)
            self.counts["links_opened"] += 1
            self.tick_counts["links_opened"] += 1
            if op == "open":
                return "link.open", f"link with {other} is open (you are at the same level of {level})", other
        if op == "open":
            if ln is None or ln.status == "closed":
                ln = self.links[key] = Link(key, aid, self.tick)
                self.boards.setdefault(ln.board, Board(ln.board, f"{key[0]}~{key[1]}"))
                self.links_of[aid].add(key)
                self.links_of[other].add(key)
                note = clip(a.get("note") or a.get("content"), 200)
                self.notify(other, aid, f"asks to open a link with you{': ' + note if note else ''} "
                                        f"(to agree: link op=open target={aid}; otherwise ignore it)")
                self.tick_counts["link_requests"] += 1
                return "link.request", f"asked {other} to link; messages can flow once {other} agrees", other
            if ln.status == "pending" and ln.requester != aid:
                ln.status, ln.opened = "open", self.tick
                self.notify(other, aid, f"agreed to your link; send with link op=send target={aid}")
                self.counts["links_opened"] += 1
                self.tick_counts["links_opened"] += 1
                return "link.open", f"link with {other} is open; read it with board='link:{other}'", other
            raise ActionError(f"your link with {other} is already {ln.status}" +
                              (f" (waiting for {other} to agree)" if ln.status == "pending" else ""))
        self._need(ln is not None and ln.status == "open",
                   f"no open link with {other}: both of you must agree with link op=open")
        if op == "send":
            content = str(a.get("content", "")).strip()
            self._need(bool(content), "content is empty")
            e = self.post(ln.board, aid, "note", content, a.get("refs"), via, notify=False)
            self.notify(other, aid, f"[link] {clip(content, 300)}")
            self.charge(aid, "communication", est_tokens(content))
            self.tick_counts["link_messages"] += 1
            return "link.send", f"sent {e.id} to {other}", other
        if op == "close":
            ln.status = "closed"
            self.notify(other, aid, f"closed the link with you")
            self.tick_counts["links_closed"] += 1
            return "link.close", f"closed your link with {other}", other
        raise ActionError(f"unknown link op '{op}'")

    def at_level(self, aid: str, gid: str) -> bool:
        """Whether aid acts at gid's level: a direct member, or someone who speaks for one of its member groups."""
        g = self.groups[gid]
        return aid in g.members or any(m in self.groups and self.can_act_for(aid, m) for m in g.members)

    def same_level(self, a: str, b: str) -> str | None:
        """A group in which both agents act at the same level, if any; such agents may talk without asking."""
        for gid in sorted(self.containers(a) & self.containers(b), key=lambda g: int(g[1:])):
            if self.groups[gid].dissolved is None and self.at_level(a, gid) and self.at_level(b, gid):
                return gid
        return None

    def open_links(self, aid: str) -> list[str]:
        return [self.links[key].other(aid) for key in self.links_of.get(aid, ()) if self.links[key].status == "open"]

    async def _do_delegate(self, aid, a, via):
        op = str(a.get("op", "")).strip()
        self._need(op in DELEGATE_OPS, f"delegate has no op '{op}': its ops are {', '.join(DELEGATE_OPS)}")
        if op == "post":
            title = clip(a.get("title"), 90)
            self._need(bool(title), "a task needs a title")
            requester = self._as(aid, a)
            parent = str(a.get("parent") or "").strip() or None
            self._need(parent is None or parent in self.tasks, f"no task '{parent}'")
            to = str(a.get("to") or "").strip()
            to = None if to.lower() in ("", "anyone", "any", "all", "open") else self._member_ref(to)
            scope = [str(x).strip() for x in as_list(a.get("scope")) if str(x).strip()][:30]
            whole = self.tasks[parent].scope if parent else []
            self._need(not (scope and whole) or self._inside_scope(scope, whole), f"scope: what {parent} covers is {', '.join(whole)}")
            scope = scope or list(whole)
            pay, until = a.get("pay"), a.get("until")
            until = int(until) if until else (self.tick + 6 if scope or pay else None)
            self._need(until is None or until >= self.tick, "until: a tick from now on")
            tid = self.new_id("K")
            t = Task(tid, title, str(a.get("spec") or "")[:1500], parent, requester, to, self.tick, scope=scope,
                     accept=clip(a.get("accept"), 300) if a.get("accept") else "", until=until)
            self._task_hook("post", t, aid, a)
            self.tasks[tid] = t
            self._task_log("post", t, aid)
            if to:
                self.notify(to, aid, f"task {tid} '{title}' for you: {clip(a.get('spec'), 200)}")
            return "delegate.post", f"posted task {tid}" + (f" to {to}" if to else " (open to anyone)"), tid
        tid = str(a.get("task") or "").strip()
        self._need(tid in self.tasks, f"no task '{tid}'" + (f": tasks are K ids, and {tid} is a post" if tid[:1] in ("P", "R", "E") else ""))
        t = self.tasks[tid]
        if op == "take":
            self._need(t.status == "open", f"{tid} is {t.status}")
            self._need(t.to is None or t.to == aid or self.inside(aid, t.to), f"{tid} is addressed to {t.to}")
            self._task_hook("take", t, aid, a)
            t.status, t.taker = "taken", aid
            self._task_log("take", t, aid)
            self.notify(t.requester, aid, f"{aid} took task {tid}")
            return "delegate.take", f"you took {tid}", tid
        if op == "refuse":
            self._need(t.status == "open", f"{tid} is {t.status}")
            self._need(t.to == aid or (t.to is not None and self.inside(aid, t.to)), f"{tid} is not addressed to you")
            self._end_task(t, "refused", aid, a)
            self.notify(t.requester, aid, f"refused task {tid}: {clip(a.get('notes'), 200)}")
            return "delegate.refuse", f"refused {tid}", tid
        if op == "deliver":
            self._need(t.taker == aid and t.status == "taken", f"you have not taken {tid}")
            t.deliverable = str(a.get("deliverable") or "").strip()[:MAX_TEXT]
            self._need(bool(t.deliverable), "deliverable is empty")
            t.evidence = [str(x).strip() for x in as_list(a.get("evidence")) if str(x).strip()][:12]
            t.status = "delivered"
            self._task_log("deliver", t, aid)
            self.charge(aid, "communication", est_tokens(t.deliverable))
            self.notify(t.requester, aid, f"delivered {tid}: {clip(t.deliverable, 200)}")
            return "delegate.deliver", f"delivered {tid}", tid
        if op == "review":
            self._need(t.status == "delivered", f"{tid} is {t.status}")
            self._need(aid != t.taker, "someone other than the taker must review")
            self._need(t.requester == aid or self.inside(aid, t.requester), f"only {t.requester} reviews {tid}")
            verdict = a.get("verdict", "accept")
            self._need(verdict in ("accept", "reject"), "verdict is accept or reject")
            t.reviews.append({"by": aid, "t": self.tick, "verdict": verdict, "notes": clip(a.get("notes"), 600)})
            if verdict == "accept":
                self._end_task(t, "accepted", aid, a)
            else:
                t.status = "taken"
                self._task_log("reject", t, aid)
            self.notify(t.taker, aid, f"{verdict}ed {tid}: {clip(a.get('notes'), 200)}")
            return f"delegate.{verdict}", f"{verdict}ed {tid}", tid
        if op == "drop":
            if t.taker == aid and t.status == "taken":
                self._task_hook("release", t, aid, a)
                t.status, t.taker = "open", None
                self._task_log("release", t, aid)
                return "delegate.release", f"released {tid}", tid
            self._need(t.requester == aid or self.inside(aid, t.requester), f"only the taker or {t.requester} can drop {tid}")
            self._need(t.status in ("open", "taken", "delivered"), f"{tid} is {t.status}")
            self._end_task(t, "dropped", aid, a)
            if t.taker:
                self.notify(t.taker, aid, f"revoked task {tid}: {clip(a.get('notes'), 200)}")
            return "delegate.drop", f"dropped {tid}", tid
        raise ActionError(f"unknown delegate op '{op}'")

    def _inside_scope(self, part: list[str], whole: list[str]) -> bool:
        fn = getattr(self.env, "in_scope", None)
        return fn(part, whole) if fn else set(part) <= set(whole)

    def _task_hook(self, op: str, t: Task, aid: str, a: dict):
        """A world's part in delegated work: holding pay, giving the taker what the task covers. It may refuse."""
        fn = getattr(self.env, "on_task", None)
        if fn:
            fn(self, op, t, aid, a)

    def _task_log(self, op: str, t: Task, aid: str):
        self.task_log.append({"t": self.tick, "op": op, "task": t.id, "by": aid, "requester": t.requester, "to": t.to, "taker": t.taker,
                              "parent": t.parent, "scope": t.scope, "held": t.held})

    def _end_task(self, t: Task, status: str, aid: str, a: dict | None = None):
        """A task ends, and what was handed on from it ends with it, so revoking or lapsing leaves nothing dangling upward or downward."""
        for child in self.tasks.values():
            if child.parent == t.id and child.status in ("open", "taken", "delivered"):
                self._end_task(child, "dropped", aid, a)
                if child.taker:
                    self.notify(child.taker, aid, f"{child.id} ended with {t.id}")
        t.status = status
        self._task_hook(status, t, aid, a or {})
        self._task_log(status, t, aid)

    def delegation_measures(self) -> dict:
        """Where delegation stops: offered, seen, taken, delivered, accepted, chosen again. A zero here says at which step."""
        tasks = list(self.tasks.values())
        done = Counter((t.requester, t.taker) for t in tasks if t.status == "accepted")

        def hops(t) -> int:
            n = 0
            while t.parent and t.parent in self.tasks:
                t, n = self.tasks[t.parent], n + 1
            return n

        return {"offered": len(tasks), "to_a_group": sum(t.to in self.groups for t in tasks), "seen_by_someone": sum(bool(t.seen - {t.requester}) for t in tasks),
                "taken": sum(any(e["task"] == t.id and e["op"] == "take" for e in self.task_log) for t in tasks), "refused": sum(t.status == "refused" for t in tasks),
                "delivered": sum(any(e["task"] == t.id and e["op"] == "deliver" for e in self.task_log) for t in tasks), "accepted": sum(t.status == "accepted" for t in tasks),
                "chosen_again": sum(n > 1 for n in done.values()), "lapsed": sum(t.status == "lapsed" for t in tasks), "revoked": sum(t.status == "dropped" for t in tasks),
                "with_scope": sum(bool(t.scope) for t in tasks), "with_pay": sum(any(e["task"] == t.id and e["held"] for e in self.task_log) for t in tasks),
                "deepest": max((hops(t) for t in tasks), default=0)}

    def expire_tasks(self):
        """Tasks past their tick end. Work delivered and never reviewed counts as accepted: nothing delivered is lost."""
        for t in list(self.tasks.values()):
            if t.until is None or self.tick <= t.until or t.status not in ("open", "taken", "delivered"):
                continue
            ended = "accepted" if t.status == "delivered" else "lapsed"
            self._end_task(t, ended, t.requester)
            self.notify(t.requester, "lab", f"task {t.id} {ended} at its end tick")
            if t.taker:
                self.notify(t.taker, "lab", f"task {t.id} {ended} at its end tick")

    # ------------------------------------------------------------------ adaptation

    async def _do_reflect(self, aid, a, via):
        content = str(a.get("content", "")).strip()
        self._need(bool(content), "content is empty")
        notes = self.agents[aid].notes
        notes.append(clip(content, 500))
        del notes[:-5]
        self.charge(aid, "critique", est_tokens(content))
        return "reflect", "updated your strategy notes", aid

    async def _do_rules(self, aid, a, via):
        """The rules a society runs under, as settings later societies can start differently (alphateam/recipe.py)."""
        book = getattr(self, "rulebook", None)
        self._need(book is not None, "the rules of this society are not open to proposals")
        return book.act(self, aid, a, via)

    async def _do_protocol(self, aid, a, via):
        gid = self._group(a.get("group"))
        g = self.groups[gid]
        self._need(self.can_act_for(aid, gid), f"you may not act for {gid} (delegates: {','.join(sorted(g.delegates)) or 'anyone inside'})")
        changed = []
        if a.get("text") is not None:
            g.protocol = str(a["text"]).strip()[:1500]
            changed.append("protocol")
        if a.get("delegates") is not None:
            ds = set()
            for d in as_list(a["delegates"]):
                d = self._member_ref(d)
                self._need(d in g.members, f"delegate {d} must be a member of {gid}")
                ds.add(d)
            g.delegates = ds
            changed.append("delegates")
        if a.get("open") is not None:
            g.open = bool(a["open"])
            changed.append("open" if g.open else "closed")
        if a.get("purpose"):
            g.purpose = str(a["purpose"]).strip()[:800]
            changed.append("purpose")
        if a.get("name"):
            g.name = clip(a["name"], 60)
            changed.append("name")
        self._need(bool(changed), "nothing to change (text, delegates, open, purpose, name)")
        self.notify(gid, aid, f"{aid} changed {gid}'s {', '.join(changed)}")
        self.counts["protocol_changes"] += 1
        self.tick_counts["protocol_changes"] += 1
        return "protocol", f"{gid}: changed {', '.join(changed)}", gid

    async def _do_design_action(self, aid, a, via):
        op = str(a.get("op", "")).strip()
        name = str(a.get("name", "")).strip().removeprefix("x_").lower()
        me = self.agents[aid]
        if op == "define":
            self._need(bool(ACTION_NAME.fullmatch(name)), "name must be 3-31 chars of a-z, 0-9, _")
            self._need(name not in self.custom, f"x_{name} exists; revise or adopt it")
            self._need(sum(c.author == aid for c in self.custom.values()) < 3, "you have defined 3 actions already")
            procedure = str(a.get("procedure", "")).strip()
            self._need(len(procedure) >= 20, "the procedure must describe concrete steps over primitives")
            self.custom[name] = CustomAction(name, clip(a.get("description") or procedure, 300), procedure[:1500],
                                             as_list(a.get("params"))[:6], aid, self.tick)
            me.actions.add(name)
            self.post("library", aid, "proposal", f"Defined new action x_{name}: {clip(a.get('description') or procedure, 300)}", notify=False)
            return "design.define", f"defined x_{name}; it is now one of your tools", name
        self._need(name in self.custom, f"no custom action '{name}'")
        ca = self.custom[name]
        if op == "revise":
            self._need(name in me.actions, "adopt it before revising")
            ca.procedure = str(a.get("procedure") or ca.procedure)[:1500]
            ca.description = clip(a.get("description") or ca.description, 300)
            ca.version += 1
            return "design.revise", f"x_{name} is now v{ca.version}", name
        if op == "adopt":
            me.actions.add(name)
            return "design.adopt", f"adopted x_{name}", name
        if op == "drop":
            me.actions.discard(name)
            return "design.drop", f"dropped x_{name}", name
        raise ActionError(f"unknown design_action op '{op}'")

    async def _run_custom(self, aid, name, args, via):
        self._need(via is None, "custom actions cannot call custom actions")
        self._need(name in self.custom, f"no custom action '{name}'")
        self._need(name in self.agents[aid].actions, f"adopt x_{name} first")
        self._need(self.custom_runner is not None, "no executor attached")
        ca = self.custom[name]
        ca.uses += 1
        self.running_custom[aid] = name  # so a world can credit what an invented action earns, and its inventor
        try:
            out = await self.custom_runner(self, aid, ca, args)
        finally:
            self.running_custom.pop(aid, None)
        return f"x_{name}", out

    # ------------------------------------------------------------------ observation

    def _new(self, aid: str, bid: str, k: int) -> tuple[int, list[Entry]]:
        """Entries on bid this agent has not been shown yet: (count, newest k)."""
        a = self.agents[aid]
        entries = self.boards[bid].entries
        start = a.seen.get(bid, 0)
        a.seen[bid] = len(entries)
        fresh = [e for e in entries[start:] if e.author != aid]
        return len(fresh), fresh[-k:]

    def observe(self, aid: str, consume: bool = True) -> str:
        """A bounded, local view: nobody sees the whole society."""
        me = self.agents[aid]
        L = [f"TICK {self.tick} of {self.total_ticks}. You are {aid}.{' ' + me.persona if me.persona else ''} "
             f"Society: {len(self.active_agents())} agents, {len(self.active_groups())} groups. "
             + (f"You have {self.max_calls} replies this tick and may take any number of actions in each reply"
                if getattr(self, "max_calls", None) else f"You may take up to {self.max_actions} actions this tick")
             + f"; reading budget {me.attention} tokens."]
        spent = self.total_tokens()
        left = self.budget_left()
        L.append(f"COSTS: society {spent:,} tokens, you {sum(self.ledger[aid].values()):,}; reorganizations "
                 f"{self.counts['reorganizations']}" + (f"; budget left {max(left, 0):,}" if left is not None else ""))
        if me.notes:
            L.append("YOUR STRATEGY NOTES:\n" + "\n".join(f"- {n}" for n in me.notes[-3:]))
        if self.sync:
            L.append(self.recent_work(aid))
        if self.env is not None:
            L.append(self.env.observe(self, aid))
        if me.inbox:
            items = list(me.inbox)
            L.append(f"INBOX ({len(items)} new{', ' + str(me.missed) + ' older pushed out' if me.missed else ''}):\n" +
                     "\n".join(f"- t{n['t']} {n['from']}: {n['text']}" for n in (items if self.full_board else items[-10:])))
            if consume:
                me.inbox.clear()
                me.missed = 0
        mine = sorted(self.member_of.get(aid, ()), key=lambda g: int(g[1:]))
        outer = sorted(self.containers(aid) - set(mine), key=lambda g: int(g[1:]))
        if mine:
            L.append("YOUR GROUPS (direct):")
            for gid in mine[:6]:
                g = self.groups[gid]
                n, fresh = self._new(aid, f"mem:{gid}", 4) if consume else (0, [])
                np, pub = self._new(aid, f"pub:{gid}", 2) if consume else (0, [])
                L.append(f"- {self.group_card(gid)}")
                L.extend(f"    mem new: {self.header(e)}" for e in fresh)
                if n > len(fresh):
                    L.append(f"    (+{n - len(fresh)} more new on mem:{gid})")
                L.extend(f"    pub new: {self.header(e)}" for e in pub)
                if np > len(pub):
                    L.append(f"    (+{np - len(pub)} more new on pub:{gid})")
            if len(mine) > 6:
                L.append(f"- and {len(mine) - 6} more: {', '.join(mine[6:])}")
        else:
            L.append("YOUR GROUPS: none.")
        if self.teaming:
            L.append(self.membership(aid))
        elif self.islands:
            isl = sorted(self.island(aid) - {aid}, key=lambda x: int(x[1:]))
            L.append("YOUR ISLAND: " + (f"you and {', '.join(isl)} (everyone who shares a group with you)" if isl else
                                        "just you (you share no group)")
                     + f". Posts by your island reach each other at once; everyone else receives them "
                     f"{self.islands['embargo']} ticks after they were posted, when their level falls, or when one of you "
                     f"publishes them (share entry=E.., {self.islands['quota']} per {self.islands['window']} ticks each; you "
                     f"have {self.publish_left(aid)} left). Receipts, the engine's notes, play data and published solutions "
                     f"reach everyone at once. A group holds at most {self.islands['cap']} agents.")
        if outer:
            L.append("ALSO INSIDE (through nesting): " + "; ".join(
                f"{g} '{self.groups[g].name}' ({len(self.agents_inside(g))} agents)" for g in outer[:6]))
            for gid in outer[:3]:
                if consume:
                    n, fresh = self._new(aid, f"mem:{gid}", 2)
                    L.extend(f"    {gid} mem new: {self.header(e)}" for e in fresh)
        if me.follows:
            lines = []
            for f in sorted(me.follows)[:12]:
                if f in self.agents:
                    recent = [e for e in self.by_author[f][-3:] if e.tick >= self.tick - 1 and self.can_read(aid, e.board)
                              and self.visible(aid, e)]
                    lines += [f"- {self.header(e)} [{e.board}]" for e in recent]
                elif f in self.groups and self.groups[f].dissolved is None and consume:
                    _, pub = self._new(aid, f"pub:{f}", 2)
                    lines += [f"- {f}: {self.header(e)}" for e in pub]
            L.append(f"FOLLOWING {', '.join(sorted(me.follows)[:12])}:" + ("\n" + "\n".join(lines[:10]) if lines else " nothing new"))
        mine_links = [self.links[key] for key in sorted(self.links_of.get(aid, ())) if self.links[key].status != "closed"]
        if mine_links:
            parts = []
            for ln in mine_links[:12]:
                other = ln.other(aid)
                if ln.status == "pending":
                    parts.append(f"{other} (pending: {'waiting for them' if ln.requester == aid else 'they asked; open to agree'})")
                else:
                    entries = self.boards[ln.board].entries
                    n = sum(e.author != aid for e in entries[me.seen.get(ln.board, 0):])
                    if consume:
                        me.seen[ln.board] = len(entries)
                    parts.append(f"{other} (open{', ' + str(n) + ' new' if n else ''})")
            L.append("YOUR LINKS: " + "; ".join(parts) + (f"; and {len(mine_links) - 12} more" if len(mine_links) > 12 else ""))
        tasks = [t for t in self.tasks.values() if t.status in ("open", "taken") and t.to and (t.to == aid or self.inside(aid, t.to))]
        tasks += [t for t in self.tasks.values() if t.status == "delivered" and (t.requester == aid or self.inside(aid, t.requester))]
        tasks += [t for t in self.tasks.values() if t.taker == aid and t.status == "taken"]
        if tasks:
            seen, lines = set(), []
            for t in tasks:
                if t.id not in seen:
                    seen.add(t.id)
                    lines.append("- " + self.task_card(t))
            L.append("TASKS FOR YOU OR YOUR GROUPS:\n" + "\n".join(lines[:8]))
        open_tasks = sum(t.status == "open" and t.to is None for t in self.tasks.values())
        anyone = [t for t in self.tasks.values() if t.status == "open" and t.to is None and t.requester != aid]
        if anyone:
            L.append("TASKS OPEN TO ANYONE:\n" + "\n".join("- " + self.task_card(t) for t in anyone))
        for t in tasks + anyone:
            t.seen.add(aid)
        for bid, label in (("global", "GLOBAL"), ("library", "LIBRARY")):
            if consume and self.sync and bid == "global":
                L.append(self._push(aid))
            elif consume:
                n, fresh = self._new(aid, bid, 5 if bid == "global" else 2)
                L.append(f"{label} BOARD: {n} new posts" + (" (newest shown):\n" + "\n".join(f"- {self.header(e)}" for e in fresh) if fresh else ""))
        own = self.boards[f"private:{aid}"].entries
        L.append(("" if self.sync else f"YOUR PRIVATE BOARD: {len(own)} entries" + (f", newest: {self.header(own[-1])}" if own else "") + ". ") +
                 f"Open tasks for anyone: {open_tasks}. Custom actions in circulation: {len(self.custom)}"
                 + (f" (yours: {', '.join('x_' + n for n in sorted(me.actions))})" if me.actions else "") + ".")
        return "\n".join(L)

    def recent_work(self, aid: str) -> str:
        """With sync, an agent's memory between turns: its own latest notes, and what the world keeps of its work."""
        board = f"private:{aid}" if self.private_boards else "global"
        notes = [e for e in self.by_author[aid] if e.kind == "note" and e.board == board][-3:]
        lines = [f"- {e.id} t{e.tick}: {clip(e.content, 400)}" for e in notes]
        fn = getattr(self.env, "recent_work", None)
        extra = fn(self, aid) if fn else ""
        head = "YOUR RECENT WORK:" if getattr(self, "private_boards", False) else "YOUR RECENT WORK (public, like everything you write):"
        return "\n".join([head] + lines + ([extra] if extra else [])) if lines or extra else "YOUR RECENT WORK: none yet."

    # ------------------------------------------------------------------ export

    def structure(self) -> dict:
        groups = []
        for g in self.active_groups():
            groups.append({"id": g.id, "name": g.name, "members": sorted(g.members, key=lambda m: (m[0], int(m[1:]))),
                           "delegates": sorted(g.delegates), "open": g.open, "inside": len(self.agents_inside(g.id)),
                           "depth": self.depth(g.id), "purpose": clip(g.purpose, 200), "protocol": clip(g.protocol, 300),
                           "founder": g.founder, "created": g.created})
        return {"groups": groups, "follows": {a: sorted(x.follows) for a, x in self.agents.items() if x.follows},
                "links": [list(ln.pair) for ln in self.links.values() if ln.status == "open"],
                "active": [a for a, x in self.agents.items() if x.active]}

    def export_custom(self) -> list[dict]:
        return [{"name": c.name, "description": c.description, "procedure": c.procedure, "params": c.params,
                 "author": c.author, "created": c.created, "version": c.version, "uses": c.uses,
                 "adopters": sum(c.name in x.actions for x in self.agents.values())} for c in self.custom.values()]
