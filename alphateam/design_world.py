"""A design world: one shared design, many local objectives that conflict (chip-design style).

The society settles a single design: D integer knobs (0-10). The knobs form a hierarchy:
  - every block has a few private knobs;
  - every super-block has a few interface knobs shared by all of its blocks;
  - a few top knobs are shared across super-blocks.
The hidden cost of the design is the sum of block costs. Each block cost is a convex quadratic (with cross terms) in the
block's private knobs, its super-block's interface knobs and some top knobs, and blocks prefer different values for the
knobs they share. So the best value of an interface knob depends on all the blocks of its super-block, and the best value
of a block's private knobs depends on where the shared knobs end up. Like integrating blocks in a chip, a good design
needs the blocks that share a knob to settle it together.

Each agent can simulate one block (its instrument), a few times per tick: its block's cost under the current design,
optionally with a few knob values changed hypothetically. Anyone can set any knob of the one shared design. Every few
ticks a design review posts the score of the current design on the global board. Knob and block names are shuffled,
so nothing about the hierarchy leaks through naming.

Defaults (sized for ~50 agents): 14 blocks x 4 private knobs, 4 super-blocks x 4 interface knobs, 3 top knobs (each
block sees 2), shared knobs weighted 3x private ones. Calibration with idealized scripted agents (each step one block
sets one of its knobs to its own best value): selfish blocks reach about 0.44 and stay there while shared knobs keep
flipping; minimizing the total cost reaches 1.0. So the last ~0.55 needs the blocks that share knobs to settle them
together.

Score: the share of the achievable improvement over the starting design that the current design has realized,
(C_start - C) / (C_start - C_opt), clamped to [0, 1]. C_opt is the integer optimum of the convex quadratic, found by
exact box-constrained coordinate descent on the continuous problem followed by integer coordinate descent with single
and pair moves from several starts (exact on small instances; tested against brute force).

Shocks: 'T:shift' rewrites the cost targets of one super-block's blocks (and so its best design). 'T:rewire' moves half of the
blocks to other super-blocks, so which blocks share interface knobs changes. 'T:leave:N' takes the agent holding the most knobs away
for N ticks; what it held is free while it is gone.

Dependency (world option dependency=independent): the same number of knobs and blocks, with nothing shared, so each block can be
settled alone. The default is coupled. dependency=moving is coupled with a rewire at tick rewire_at and a leave right after it. The
three are the review's test of when structure is worth having (DESIGN.md section 19): nothing in the brief says which one an agent
is in, or that anyone is looking for hierarchy.

Delegation: a kernel task whose scope names knobs hands those decisions over: while the task lives its taker holds them, and they go
back when it ends. Only whoever holds a knob, or anyone while nobody does, can hand it over.

Credit (world option credit=1): every knob change is credited to whoever made it (and to the group it was made for)
with the change in the design's score it caused at that moment, so credits add up to the score. Design reviews post the
ledger and tell each agent what its changes since the last review earned; nothing is shown at the moment of a change.

Durable credit (credit=durable, credit_every=E): credit for what lasts, shared with groups, with consequences. Every E ticks
each knob whose value changed since the last credit review is credited to whoever set its current value (the group, if
set for one) with how much better the design is now than with that knob back at its value at the last review; values
that were reversed or overwritten earn nothing. A group's credit is shared equally among the agents inside it, and an
agent's credit is its own plus its shares. Credit sets resources: simulations per tick are the base plus one per 0.05 of
credit (at least 1, at most base + 6), updated at each credit review.
Options on top: evidence_share=F gives that fraction of a change's positive credit to the agents whose simulations of
that knob (hypothetical changes, posted before the change on a board the setter can read) informed it; claim_rent=R
charges every holder R credit per held knob at each credit review; group_area=1 also credits every group of two or more
agents with the improvement, since the last credit review, in the summed cost of the blocks its agents simulate;
claim_price=P makes credit a currency for authority: claiming a knob costs P of the claimant's own credit (an agent's,
or a group's when claimed for one), and a claimant without that much credit cannot claim; rights_by_credit=1 makes
authority follow track record: at each credit review every shared knob is held by the agent, among those whose blocks
use it, with the most credit (none while no user has positive credit).
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict, deque

import numpy as np

from .institutions import Institutions
from .kernel import ActionError

LO, HI = 0, 10
LEVELS = ("private", "interface", "top")


class DesignWorld(Institutions):
    tool_names = ("simulate", "set_knob", "view_design")
    forkable = True  # all of its state is in memory, so a running society can be copied for a branch test (alphateam/paired.py)
    search_kinds = ("knobs",)
    has_modules = True
    TOOLS = [
        {"name": "simulate",
         "description": "Simulate your block: its cost under the current shared design (lower is better), and optionally its "
                        "cost with up to 4 knob values changed hypothetically (the shared design is not changed). Results "
                        "are posted to the board in 'to' (default your private board). Limited simulations per tick.",
         "parameters": {"type": "object", "properties": {
             "changes": {"type": "object", "description": "hypothetical knob values, e.g. {\"k12\": 7, \"k40\": 3}"},
             "block": {"type": "string", "description": "which of your blocks (only if you simulate several)"},
             "to": {"type": "string", "description": "board for the result: 'private', a group id, 'global', ..."}}}},
        {"name": "set_knob",
         "description": "Set one knob of the shared design (integer 0-10). Anyone may set any knob; the latest value stands. "
                        "Optionally on behalf of a group you may act for.",
         "parameters": {"type": "object", "properties": {
             "knob": {"type": "string", "description": "e.g. k12"}, "value": {"type": "integer"},
             "as": {"type": "string", "description": "optional group id"}},
             "required": ["knob", "value"]}},
        {"name": "view_design",
         "description": "Show knobs of the shared design: current value, who set it last and when. Default: your block's knobs.",
         "parameters": {"type": "object", "properties": {
             "knobs": {"type": "array", "items": {"type": "string"}, "description": "knob ids, or [\"all\"]"}}}},
    ]

    OWN_TOOL = {"name": "own_knob",
                "description": "Claim exclusive control of a knob, for yourself or for a group you may act for (as). While "
                               "held, only the holder (or anyone who may act for the holding group) can set it. op release "
                               "gives it up.",
                "parameters": {"type": "object", "properties": {
                    "knob": {"type": "string"}, "op": {"type": "string", "enum": ["claim", "release"]},
                    "as": {"type": "string", "description": "optional group id to hold it"}}, "required": ["knob"]}}

    def __init__(self, n_agents: int, seed: int = 0, single: bool = False, world_agents: int | None = None,
                 opts: dict | None = None):
        opts = opts or {}
        # decision rights: knob index -> holder (agent or group id). ownership=1 adds own_knob for the agents;
        # authority=1 hands the prescribed org chart the rights (team: its block's private knobs; division: its
        # super-block's interface knobs; program: the top knobs).
        self.rights: dict[int, str] = {}
        self.dictated: dict[int, str] = {}  # the rights the holder rule assigned (set_holders can take them back)
        self.ownership = str(opts.get("ownership", "")).lower() in ("1", "true", "yes", "group")
        self.group_only = str(opts.get("ownership", "")).lower() == "group"  # only groups may hold knobs
        self.authority = str(opts.get("authority", "")).lower() in ("1", "true", "yes")
        # control: each knob held by one random agent whose block uses it ("agent"), or only the shared knobs ("agent_shared")
        self.dictators = str(opts.get("authority", "")).lower() in ("agent", "agent_shared")
        self.dictate_private = str(opts.get("authority", "")).lower() == "agent"
        # economy=1: each agent's reward is its own credit, earned from its own block's cost; the lead of a scope of two
        # or more blocks earns project_mult times the fall in the scope's combined cost (collective projects pay more)
        self.economy = str(opts.get("economy", "")).lower() in ("1", "true", "yes")
        self.project_mult = float(opts.get("project_mult", 2.0))
        self.economy_share = float(opts.get("economy_share", 0) or 0)  # profit sharing: each agent's share of the total's fall
        self.durable = str(opts.get("credit", "")).lower() == "durable" or self.economy
        self.credit_on = str(opts.get("credit", "")).lower() in ("1", "true", "yes") or self.durable
        if self.ownership:
            self.tool_names = type(self).tool_names + ("own_knob",)
            self.TOOLS = type(self).TOOLS + [self.OWN_TOOL]
        self._init_institutions(opts)
        self.rng = random.Random(seed)
        self.world_seed = seed
        npr = np.random.default_rng(seed)
        size = world_agents or n_agents
        per_block = float(opts.get("agents_per_block", 3.5))
        n_blocks = int(opts.get("blocks", max(2, round(size / per_block))))
        n_super = int(opts.get("supers", max(1, round(n_blocks / 4))))
        n_super = min(n_super, n_blocks)
        self.n_private = int(opts.get("private", 4))
        self.n_interface = int(opts.get("interface", 4))
        self.n_top = int(opts.get("top", 3))
        self.dependency = str(opts.get("dependency", "coupled"))
        self.force_budget = str(opts.get("force_budget", "")).lower() in ("1", "true", "yes")
        self.contest_facts = str(opts.get("contest", "")).lower() in ("1", "true", "yes")  # reviews say which shared knobs keep moving
        extra = 0
        if self.dependency == "independent":  # the same knobs, none shared: what was shared becomes private
            extra = n_super * self.n_interface + self.n_top
            self.n_interface = self.n_top = 0
        self.top_per_block = min(self.n_top, int(opts.get("top_per_block", 2)))
        self.sims_per_tick = int(opts.get("sims", 3))
        self.max_changes = 4
        self.weight = {lv: float(opts.get(f"w_{lv}", w)) for lv, w in (("private", 0.5), ("mid", 1.0), ("interface", 1.5),
                                                                          ("top", 1.5))}
        # a deeper hierarchy (subs=S): each super-block's blocks split into S sub-groups that share `mid` knobs of their own
        self.n_subs = int(opts.get("subs", 0) or 0)
        self.n_mid = int(opts.get("mid", 3))
        self.levels = ("private", "mid", "interface", "top") if self.n_subs else LEVELS
        self.validate_every = int(opts.get("review_every", 3))
        self.credit_every = int(opts.get("credit_every", self.validate_every))
        self.single = single
        self.world_agents = size
        self.n_super = n_super
        # knobs, in an internal order; agent-facing names are shuffled
        level, owner = [], []  # owner: block for private, super-block for interface, -1 for top
        supers = [b % n_super for b in range(n_blocks)]
        for b in range(n_blocks):
            mine = self.n_private + extra // n_blocks + (b < extra % n_blocks)
            level += ["private"] * mine
            owner += [b] * mine
        for s in range(n_super):
            level += ["interface"] * self.n_interface
            owner += [s] * self.n_interface
        self.block_sub = {}
        if self.n_subs:
            for s in range(n_super):
                members = [b for b in range(n_blocks) if supers[b] == s]
                size = -(-len(members) // self.n_subs)
                for c in range(self.n_subs):
                    chunk = members[c * size:(c + 1) * size]
                    for b in chunk:
                        self.block_sub[b] = (s, c)
                    if len(chunk) >= 2:  # a sub-group of one block would only add private knobs
                        level += ["mid"] * self.n_mid
                        owner += [(s, c)] * self.n_mid
        level += ["top"] * self.n_top
        owner += [-1] * self.n_top
        self.D = len(level)
        self.level = np.array(level)
        self.owner = owner
        names = list(range(1, self.D + 1))
        self.rng.shuffle(names)
        self.name = [f"k{x}" for x in names]
        self.index = {nm: i for i, nm in enumerate(self.name)}
        labels = list(range(1, n_blocks + 1))
        self.rng.shuffle(labels)
        self.block_label = [f"B{x}" for x in labels]
        top_ids = [i for i in range(self.D) if level[i] == "top"]
        self.blocks = []
        for b in range(n_blocks):
            s = supers[b]
            idx = [i for i in range(self.D) if level[i] == "private" and owner[i] == b]
            idx += [i for i in range(self.D) if level[i] == "mid" and owner[i] == self.block_sub.get(b)]
            idx += [i for i in range(self.D) if level[i] == "interface" and owner[i] == s]
            first = top_ids[b % len(top_ids)] if top_ids else None  # every top knob is used by some block
            rest = [i for i in top_ids if i != first]
            tops = ([first] + self.rng.sample(rest, min(len(rest), self.top_per_block - 1))) if top_ids else []
            self.blocks.append({"id": b, "super": s, "idx": np.array(idx + sorted(tops))})
        for i in top_ids[n_blocks:]:  # fewer blocks than top knobs: give the rest to random blocks
            blk = self.blocks[self.rng.randrange(n_blocks)]
            if i not in blk["idx"]:
                blk["idx"] = np.append(blk["idx"], i)
        self.modules = [{"id": b, "name": self.block_label[b], "super": blk["super"], "level": 0, "outputs": []}
                        for b, blk in enumerate(self.blocks)]
        self.users = defaultdict(list)  # knob -> blocks whose cost depends on it
        for b, blk in enumerate(self.blocks):
            for i in blk["idx"]:
                self.users[int(i)].append(b)
        self._npr = npr
        for blk in self.blocks:
            self._make_cost(blk)
        self.x = np.array([self.rng.randint(LO, HI) for _ in range(self.D)], dtype=float)
        self.x_start = self.x.copy()
        self._solve()
        # agents and their blocks (instruments)
        aids = [f"A{i + 1}" for i in range(n_agents)]
        if single:
            self.instruments = {aids[0]: list(range(n_blocks))}
        else:
            order = aids[:]
            self.rng.shuffle(order)
            self.instruments = {a: [j % n_blocks] for j, a in enumerate(order)}
        self.home = {a: [self.block_label[b] for b in bs] for a, bs in self.instruments.items()}  # used by the mock backend
        self.sims: Counter = Counter()  # (agent, tick) -> simulations used
        self.used: Counter = Counter()  # agent -> simulations in total
        self.focus: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self.last_set: dict[int, tuple] = {}  # knob -> (agent, group, tick)
        self.last_delta: dict[int, float] = {}
        self.sets: list[dict] = []
        self.claims_log: list[dict] = []
        self.reviews: list[dict] = []
        self.credit: dict[str, list] = defaultdict(lambda: [0.0, 0])  # agent or group -> [score credited, changes]
        self.credit_shown: dict[str, tuple] = {}  # the ledger as of the last review (what agents see)
        self.dcredit: dict[str, list] = {}  # durable credit: agent or group -> [credit, changes credited]
        self.effective: dict[str, float] = {}  # agent -> own durable credit plus shares of its groups' (last review)
        self.sim_bonus: dict[str, int] = {}
        self.x_reviewed = self.x_start.copy()  # the design at the last credit review
        self.evidence_share = float(opts.get("evidence_share", 0) or 0)
        self.claim_rent = float(opts.get("claim_rent", 0) or 0)
        self.group_area = str(opts.get("group_area", "")).lower() in ("1", "true", "yes")
        self.claim_price = float(opts.get("claim_price", 0) or 0)
        self.rights_by_credit = str(opts.get("rights_by_credit", "")).lower() in ("1", "true", "yes")
        self.credit_held: dict[int, str] = {}  # knob -> holder assigned by credit
        self.evidence: list[dict] = []  # simulations with hypothetical changes: who, which knobs, which board, when
        self.no_sets_until = 0  # a teamwork strategy's exploration phase: no knob may be set up to this tick
        self.default_to: dict[str, str] = {}  # a teamwork strategy's default board for an agent's simulation results
        self.shocks: list[dict] = []
        self.delegated: dict[int, str] = {}  # knob -> taker of a live task that covers it
        self.task_knobs: dict[str, list[int]] = {}
        self.away: dict[str, int] = {}  # agent -> the tick it comes back
        plan = list(opts.get("shocks") or [])
        if self.dependency == "moving":
            at = int(opts.get("rewire_at", 12))
            plan += [f"{at}:rewire", f"{at + 1}:leave:{int(opts.get('leave_for', 4))}"]
        self.shock_plan = self._parse_shocks(plan)

    # ------------------------------------------------------------------ hidden cost

    def _make_cost(self, blk: dict):
        """A convex quadratic in the block's knobs with cross terms; shared knobs weigh more than private ones, so how
        the blocks that share a knob settle it matters most."""
        idx = blk["idx"]
        n = len(idx)
        scale = np.array([self.weight[self.level[i]] for i in idx])
        A = self._npr.normal(0, 1, (n, n)) / np.sqrt(n)
        Q = (A @ A.T) * np.outer(scale, scale) + 0.3 * np.diag(scale ** 2)
        blk["Q"] = Q
        blk["t"] = self._npr.uniform(LO, HI, n)  # each block's preferred values, also for shared knobs (conflicts)

    def _assemble(self):
        H = np.zeros((self.D, self.D))
        g = np.zeros(self.D)
        c = 0.0
        for blk in self.blocks:
            idx, Q, t = blk["idx"], blk["Q"], blk["t"]
            H[np.ix_(idx, idx)] += Q
            g[idx] += Q @ t
            c += float(t @ Q @ t)
        self.H, self.g, self.c0 = H, g, c

    def cost(self, x: np.ndarray) -> float:
        return float(x @ self.H @ x - 2 * self.g @ x + self.c0)

    def block_cost(self, b: int, x: np.ndarray) -> float:
        blk = self.blocks[b]
        d = x[blk["idx"]] - blk["t"]
        return float(d @ blk["Q"] @ d)

    def _continuous_opt(self) -> np.ndarray:
        """Box-constrained minimum of the convex quadratic by projected coordinate descent (exact for convex problems)."""
        x = np.clip(np.linalg.solve(self.H, self.g), LO, HI)
        for _ in range(500):
            prev = x.copy()
            for i in range(self.D):
                x[i] = min(HI, max(LO, (self.g[i] - self.H[i] @ x + self.H[i, i] * x[i]) / self.H[i, i]))
            if np.max(np.abs(x - prev)) < 1e-9:
                break
        return x

    def _integer_descent(self, x: np.ndarray) -> np.ndarray:
        """Integer local search: exact single-coordinate moves, then pair moves of ±1, until neither improves."""
        x = np.clip(np.round(x), LO, HI)
        grad = self.H @ x - self.g  # half the gradient
        while True:
            improved = False
            for i in range(self.D):  # the best integer for one coordinate of a convex quadratic is its rounded vertex
                v = min(HI, max(LO, round(x[i] - grad[i] / self.H[i, i])))
                d = v - x[i]
                if d and 2 * d * grad[i] + d * d * self.H[i, i] < -1e-9:
                    x[i] = v
                    grad += self.H[:, i] * d
                    improved = True
            if improved:
                continue
            nz = np.argwhere(np.triu(np.abs(self.H) > 1e-12, 1))
            for i, j in nz:
                for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    if not (LO <= x[i] + di <= HI and LO <= x[j] + dj <= HI):
                        continue
                    delta = (2 * (di * grad[i] + dj * grad[j]) + di * di * self.H[i, i] + dj * dj * self.H[j, j]
                             + 2 * di * dj * self.H[i, j])
                    if delta < -1e-9:
                        x[i] += di
                        x[j] += dj
                        grad += self.H[:, i] * di + self.H[:, j] * dj
                        improved = True
                        break
            if not improved:
                return x

    def _solve(self, starts: int = 6):
        self._assemble()
        cands = [self._integer_descent(self._continuous_opt())]
        for _ in range(starts):
            cands.append(self._integer_descent(np.array([self.rng.randint(LO, HI) for _ in range(self.D)], dtype=float)))
        best = min(cands, key=self.cost)
        self.x_opt, self.c_opt = best, self.cost(best)
        self.c_start = self.cost(self.x_start)

    # ------------------------------------------------------------------ run hooks

    def _parse_shocks(self, specs) -> dict[int, list[str]]:
        plan = defaultdict(list)
        for s in specs:
            t, _, kind = str(s).partition(":")
            if kind.split(":")[0] not in ("shift", "rewire", "leave"):
                raise ValueError(f"design world shocks are T:shift, T:rewire or T:leave:N, got '{s}'")
            plan[int(t)].append(kind)
        return plan

    def _assign_holders(self):
        """Rights without hierarchy: one agent whose block uses the knob holds it (the same draw for a given world)."""
        rng = random.Random(f"dictators:{self.D}:{len(self.instruments)}")
        for i in range(self.D):
            users = [a for a, bs in sorted(self.instruments.items()) if set(bs) & set(self.users[i])]
            if users and (self.dictate_private or str(self.level[i]) != "private"):
                who = rng.choice(users)
                if i not in self.rights:  # a knob someone already holds (a claim, a lead, a team) stays theirs
                    self.rights[i] = self.dictated[i] = who

    def set_holders(self, k, mode: str):
        """Change who holds knobs in a running society (a recipe revision tried on a branch): none, shared or all."""
        for i, h in self.dictated.items():
            if self.rights.get(i) == h:
                del self.rights[i]
        self.dictated = {}
        self.dictators = mode in ("shared", "all")
        self.dictate_private = mode == "all"
        if self.dictators:
            self._assign_holders()

    def setup(self, k):
        if self.dictators:
            self._assign_holders()
        if self.authority:
            by_name = {g.name: g.id for g in k.active_groups()}
            for i in range(self.D):
                lv, o = str(self.level[i]), self.owner[i]
                gid = by_name.get(f"Module team {o}" if lv == "private" else f"Division {o}" if lv == "interface" else "Program")
                if gid:
                    self.rights[i] = gid
        if getattr(self, "chair", False):  # the chaired control arm (alphateam/institutions.py)
            self._install_chair(k)
        for aid, bs in self.instruments.items():
            k.post(f"private:{aid}", "lab", "note", self._block_note(bs), notify=False)

    def _block_note(self, bs: list[int]) -> str:
        lines = [f"{self.block_label[b]} depends on knobs {', '.join(self.name[i] for i in self.blocks[b]['idx'])}"
                 for b in bs[:16]]
        return "You can simulate " + ("these blocks:\n" if len(bs) > 1 else "block ") + "\n".join(lines)

    def holder(self, i: int) -> str | None:
        return self.delegated.get(i) or self.rights.get(i)

    def in_scope(self, part: list[str], whole: list[str]) -> bool:
        return set(part) <= set(whole)

    def on_task(self, k, op: str, t, aid: str, a: dict):
        """A task whose scope names knobs hands those decisions over, for as long as it lives."""
        k._need(op != "post" or not a.get("pay"), "this world has no pay on tasks; contract moves credit")
        if op == "take" and t.scope:
            knobs = [self._knob(x) for x in t.scope]
            for i in knobs:
                held = self.holder(i)
                mine = held in k.groups and (k.can_act_for(t.requester, held) if getattr(self, "chair", False) else
                                             k.inside(t.requester, held))  # under a chair: its lead and delegates only
                k._need(held is None or held == t.requester or mine,
                        f"{self.name[i]} is held by {held}; only its holder can hand it over")
            for i in knobs:
                self.delegated[i] = aid
            self.task_knobs[t.id] = knobs
        elif op in ("accepted", "dropped", "lapsed", "release"):
            for i in self.task_knobs.pop(t.id, []):
                self.delegated.pop(i, None)

    def _rewire(self, k):
        """Half of the blocks move to other super-blocks: who shares interface knobs with whom changes, and with it the best design."""
        blocks = list(self.blocks)
        self.rng.shuffle(blocks)
        moved = blocks[:max(1, len(blocks) // 2)]
        for blk in moved:
            old, new = blk["super"], self.rng.choice([s for s in range(self.n_super) if s != blk["super"]] or [blk["super"]])
            keep = [int(i) for i in blk["idx"] if not (str(self.level[i]) == "interface" and self.owner[i] == old)]
            blk["super"] = new
            blk["idx"] = np.array(keep + [i for i in range(self.D) if str(self.level[i]) == "interface" and self.owner[i] == new])
            self._make_cost(blk)
            self.modules[blk["id"]]["super"] = new
        self.users = defaultdict(list)
        for b, blk in enumerate(self.blocks):
            for i in blk["idx"]:
                self.users[int(i)].append(b)
        for aid, bs in self.instruments.items():
            if set(bs) & {blk["id"] for blk in moved}:
                k.notify(aid, "lab", "Your block's dependencies changed. " + self._block_note(bs))
        if getattr(self, "leadership", False):
            self._recompute_lead_rights()
        return len(moved)

    def _leave(self, k, ticks: int):
        counts = Counter(h for h in self.rights.values() if h in k.agents)
        chair = getattr(self, "chairman", None)
        if chair and k.agents[chair].active:  # under a chair, the agent with the most decisions is the chair
            who = chair
        else:
            who = counts.most_common(1)[0][0] if counts else sorted(k.active_agents(), key=lambda a: int(a[1:]))[0]
        k.agents[who].active = False
        self.away[who] = k.tick + ticks
        for i, h in list(self.rights.items()):
            if h == who:
                del self.rights[i]
        return who

    def on_tick_start(self, k):
        for who, back in list(self.away.items()):
            if k.tick >= back:
                k.agents[who].active = True
                del self.away[who]
                k.notify(who, "lab", "You are back. What you held was released while you were away.")
        for kind in self.shock_plan.get(k.tick, []):
            start = self.x_start
            if kind == "rewire":
                n = self._rewire(k)
                self._solve()
                self.x_start = start
                self.c_start = self.cost(start)
                self.shocks.append({"t": k.tick, "kind": "rewire", "moved": n})
                continue
            if kind.startswith("leave"):
                self.shocks.append({"t": k.tick, "kind": "leave", "agent": self._leave(k, int(kind.split(":")[1] or 4))})
                continue
            s = self.rng.randrange(self.n_super)
            for blk in self.blocks:
                if blk["super"] == s:
                    blk["t"] = self._npr.uniform(LO, HI, len(blk["idx"]))
            self._solve()
            self.x_start = start
            self.c_start = self.cost(start)
            self.shocks.append({"t": k.tick, "kind": "shift", "super": s,
                                "blocks": sum(blk["super"] == s for blk in self.blocks)})

    def on_tick_end(self, k):
        self._k = k
        self._institutions_tick(k)
        if self.validate_every and k.tick % self.validate_every == 0:
            s = self.normalized(self.x)
            self.reviews.append({"t": k.tick, "score": round(s, 4)})
            prev = self.reviews[-2]["score"] if len(self.reviews) > 1 else None
            k.post("global", "lab", "result",
                   f"Design review at t{k.tick}: score {s:.3f} (share of the achievable improvement over the starting "
                   f"design that the current design realizes; 1 = best possible, 0 = no better than the start, "
                   f"negative = worse than the start)" +
                   (f", previous review {prev:.3f}." if prev is not None else ".") + self._contest_line(k), notify=False)
        if self.credit_on and self.credit_every and k.tick % self.credit_every == 0:
            if self.durable:
                self._durable_review(k)
            else:
                self._post_credit(k, k.tick - self.credit_every)

    def _contest_line(self, k, window: int = 3) -> str:
        """A measured fact for the review post, under contest=1: the shared knobs that keep moving, and who moved them. No advice."""
        if not self.contest_facts:
            return ""
        by = defaultdict(list)
        for s in self.sets:
            if s["t"] > k.tick - window and s["level"] != "private":
                by[s["knob"]].append(s)
        busy = sorted(((n, v) for n, v in by.items() if len(v) >= 2), key=lambda kv: -len(kv[1]))[:5]
        if not busy:
            return ""
        lines = [f"{n} moved {len(v)} times ({sum(x['reversal'] for x in v)} reversals) by {', '.join(sorted({x['agent'] for x in v}))}, "
                 f"from {v[0]['from']:g} to {v[-1]['to']:g}, used by {', '.join(self.block_label[b] for b in self.users[self._knob(n)])}"
                 for n, v in busy]
        return f" Shared knobs that moved most in the last {window} ticks: " + "; ".join(lines) + "."

    def _durable_review(self, k):
        """Credit what lasted since the last credit review, share group credit, and set each agent's simulations."""
        since = k.tick - self.credit_every
        gap = max(1e-9, self.c_start - self.c_opt)
        now = self.cost(self.x)
        notes = defaultdict(list)
        for i, (aid, group, t) in sorted(self.last_set.items()):
            if t <= since or self.economy:
                continue
            v0 = self.x_reviewed[i]
            gain = 0.0
            if self.x[i] != v0:
                y = self.x.copy()
                y[i] = v0
                gain = (self.cost(y) - now) / gap
            who = group or aid
            providers = sorted({e["agent"] for e in self.evidence if i in e["knobs"] and e["t"] <= t and e["agent"] != aid
                                and k.can_read(aid, e["board"])}) if self.evidence_share and gain > 0 else []
            kept = gain * (1 - self.evidence_share) if providers else gain
            c = self.dcredit.setdefault(who, [0.0, 0])
            c[0] += kept
            c[1] += 1
            notes[aid].append(f"{self.name[i]} {v0:g}->{self.x[i]:g} {kept:+.3f}" + (f" for {group}" if group else "") +
                              (f" ({gain - kept:+.3f} to the evidence of {', '.join(providers[:4])})" if providers else ""))
            for a in providers:
                self.dcredit.setdefault(a, [0.0, 0])[0] += (gain - kept) / len(providers)
                notes[a].append(f"your simulations of {self.name[i]} informed {aid}'s change: {(gain - kept) / len(providers):+.3f}")
        for x in self.sets:  # changes that did not last earn nothing; say so
            if x["t"] > since and self.last_set[self.index[x["knob"]]][0] != x["agent"] and not self.economy:
                notes[x["agent"]].append(f"{x['knob']} ->{x['to']:g} overwritten 0")
        self._team_rewards(k, since, gap, notes)
        if self.economy:  # individual rewards: your own block, and for leads the scope they lead
            social = self.economy_share * (self.cost(self.x_reviewed) - self.cost(self.x)) / gap
            for aid, bs in self.instruments.items():
                r = sum(self.block_cost(b, self.x_reviewed) - self.block_cost(b, self.x) for b in bs) / gap + social
                c = self.dcredit.setdefault(aid, [0.0, 0])
                c[0] += r
                c[1] += 1
                notes[aid].append(f"your block's cost change since t{since} earned you {r:+.3f}")
            for scope, lead in self.leads.items():
                if len(scope) < 2:
                    continue
                r = self.project_mult * sum(self.block_cost(b, self.x_reviewed) - self.block_cost(b, self.x)
                                            for b in scope) / gap
                self.dcredit.setdefault(lead, [0.0, 0])[0] += r
                notes[lead].append(f"as lead of {self._scope_label(scope)} you earned {r:+.3f} "
                                   f"({self.project_mult:g}x the fall in its combined cost)")
        if self.group_area:  # a group earns what happened to its own area, whoever moved the knobs
            gap = max(1e-9, self.c_start - self.c_opt)
            for g in k.active_groups():
                members = [a for a in k.agents_inside(g.id) if a in self.instruments]
                if len(members) < 2:
                    continue
                area = sorted({b for a in members for b in self.instruments[a]})
                gain = sum(self.block_cost(b, self.x_reviewed) - self.block_cost(b, self.x) for b in area) / gap
                self.dcredit.setdefault(g.id, [0.0, 0])[0] += gain
                for a in members:
                    notes[a].append(f"{g.id}'s blocks ({', '.join(self.block_label[b] for b in area[:6])}) improved "
                                    f"{gain:+.3f}, shared among its {len(members)} agents")
        self.x_reviewed = self.x.copy()
        if self.claim_rent:
            for i, holder in self.rights.items():
                self.dcredit.setdefault(holder, [0.0, 0])[0] -= self.claim_rent
        for aid in self.instruments:
            if not k.agents[aid].active:
                continue
            groups = [g for g in k.containers(aid) if g in self.dcredit]
            self.effective[aid] = self.dcredit.get(aid, [0.0])[0] + sum(
                self.dcredit[g][0] / max(1, len(k.agents_inside(g))) for g in groups)
            self.sim_bonus[aid] = max(1 - self.sims_per_tick, min(6, round(self.effective[aid] / 0.05)))
        if self.rights_by_credit:  # authority follows track record
            for i, h in self.credit_held.items():
                if self.rights.get(i) == h:
                    del self.rights[i]
            self.credit_held = {}
            for i in range(self.D):
                users = [a for a, bs in self.instruments.items() if set(bs) & set(self.users[i]) and a in self.effective]
                if len(self.users[i]) < 2 or not users:
                    continue
                best = max(users, key=lambda a: (self.effective[a], -int(a[1:])))
                if self.effective[best] > 0:
                    self.rights[i] = best
                    self.credit_held[i] = best
            held = Counter(self.credit_held.values())
            if held:
                k.post("global", "lab", "result", f"Authority at t{k.tick} (each shared knob is held by its user with the most "
                       "credit): " + "; ".join(f"{a} holds {n}" for a, n in held.most_common(8)), notify=False)
        ranked = sorted(self.effective, key=lambda a: -self.effective[a])
        fmt = lambda a: f"{a} {self.effective[a]:+.3f} ({self.sim_budget(a)} sims)"
        groups = sorted((g for g in self.dcredit if g in k.groups), key=lambda g: -self.dcredit[g][0])
        k.post("global", "lab", "result",
               f"Durable credit at t{k.tick} (changes since t{since} that lasted; reversed or overwritten ones earn 0; group "
               "credit is shared among the agents inside; simulations per tick follow credit): " +
               "; ".join(fmt(a) for a in ranked[:10]) + (" … lowest: " + "; ".join(fmt(a) for a in ranked[-4:]) if len(ranked) > 10 else "") +
               (". Groups: " + "; ".join(f"{g} {self.dcredit[g][0]:+.3f} shared by {len(k.agents_inside(g))}" for g in groups[:8])
                if groups else ". No group has credit yet."), notify=False)
        for aid in self.effective:
            k.notify(aid, "lab", (f"credit since t{since}: " + ", ".join(notes[aid][-5:]) + "; " if notes[aid] else "") +
                     f"your credit is now {self.effective[aid]:+.3f} (own plus group shares), so you get "
                     f"{self.sim_budget(aid)} simulations per tick")

    def _post_credit(self, k, since: int):
        """The credit ledger on the global board, and each recent changer's own account in its inbox."""
        fmt = lambda who: f"{who} {self.credit[who][0]:+.3f} ({self.credit[who][1]} changes)"
        agents = sorted((w for w in self.credit if w not in k.groups), key=lambda w: -self.credit[w][0])
        groups = sorted((w for w in self.credit if w in k.groups), key=lambda w: -self.credit[w][0])
        self.credit_shown = {w: tuple(v) for w, v in self.credit.items()}
        if not agents:
            return
        top, low = agents[:10], [w for w in agents[-5:] if w not in agents[:10]]
        k.post("global", "lab", "result",
               f"Credit ledger at t{k.tick} (each knob change is credited with the change in the design's score it caused "
               f"when it was made; credits add up to the score): " + "; ".join(fmt(w) for w in top) +
               (" … lowest: " + "; ".join(fmt(w) for w in low) if low else "") +
               (". Groups: " + "; ".join(fmt(g) for g in groups[:8]) if groups else ". No changes were made for groups."),
               notify=False)
        recent = defaultdict(list)
        for x in self.sets:
            if x["t"] > since:
                recent[x["agent"]].append(x)
        for aid, xs in recent.items():
            gain = sum(x["gain"] for x in xs)
            k.notify(aid, "lab", f"your knob changes since t{since} earned {gain:+.3f} credit (" +
                     ", ".join(f"{x['knob']} {x['from']:g}->{x['to']:g} {x['gain']:+.3f}" for x in xs[-5:]) +
                     f"); your total {self.credit[aid][0]:+.3f}")

    # ------------------------------------------------------------------ actions

    def act(self, k, aid: str, name: str, a: dict):
        return getattr(self, f"_{name}")(k, aid, a)

    def _knob(self, ref) -> int:
        nm = str(ref or "").strip().lower()
        if re.fullmatch(r"\d+", nm):
            nm = f"k{nm}"
        if nm not in self.index:
            raise ActionError(f"no knob '{ref}' (knobs are k1..k{self.D})")
        return self.index[nm]

    def _note_focus(self, k, aid: str, blocks: dict[int, float]):
        for b, w in blocks.items():
            self.focus[aid].append((k.tick, b, w))

    def _simulate(self, k, aid, a):
        if a.get("area") and self.leadership:
            return self._simulate_area(k, aid, a)
        mine = self.instruments.get(aid, [])
        k._need(mine, "you have no block to simulate")
        if a.get("block"):
            lab = str(a["block"]).strip().upper()
            bs = [b for b in mine if self.block_label[b] == lab]
            k._need(bs, f"you cannot simulate {a['block']} (yours: {', '.join(self.block_label[b] for b in mine[:8])})")
            b = bs[0]
        else:
            k._need(len(mine) == 1, "say which block to simulate")
            b = mine[0]
        budget = self.sim_budget(aid)
        k._need(self.sims[(aid, k.tick)] < budget, f"you have used all {budget} simulations this tick")
        changes = a.get("changes") or {}
        k._need(isinstance(changes, dict), "changes must be an object like {\"k12\": 7}")
        k._need(len(changes) <= self.max_changes, f"at most {self.max_changes} hypothetical changes per simulation")
        x2 = self.x.copy()
        shown = []
        for ref, v in changes.items():
            i = self._knob(ref)
            v = int(v)
            k._need(LO <= v <= HI, f"knob values are integers {LO}-{HI}")
            x2[i] = v
            shown.append(f"{self.name[i]}={v}")
        board = k.board_id(aid, a.get("to") or self.default_to.get(aid, "private"))
        k._need(k.can_write(aid, board), f"you cannot write to {board}")
        self.sims[(aid, k.tick)] += 1
        self.used[aid] += 1
        if shown:
            self.evidence.append({"t": k.tick, "agent": aid, "knobs": [self._knob(r) for r in changes], "board": board})
        k.tick_counts["experiments"] += 1
        now = self.block_cost(b, self.x)
        text = f"Simulation of {self.block_label[b]} by {aid} at t{k.tick}: cost now {now:.1f}"
        if shown:
            hyp = self.block_cost(b, x2)
            text += f"; with {', '.join(shown)}: {hyp:.1f} ({hyp - now:+.1f})"
        k.post(board, aid, "data", text + ".", notify=False)
        self._note_focus(k, aid, {b: 1.0})
        return "lab.simulate", text + f" [simulations left this tick: {budget - self.sims[(aid, k.tick)]}]", self.block_label[b]

    def sim_budget(self, aid: str | None = None) -> int:
        """Simulations per tick; the single agent gets the whole society's; with durable credit, credit adds or removes some."""
        return self.sims_per_tick * (self.world_agents if self.single else 1) + (self.sim_bonus.get(aid, 0) if aid else 0)

    def _set_knob(self, k, aid, a):
        k._need(not self.no_sets_until or k.tick > self.no_sets_until, f"no knob may be set before t{self.no_sets_until + 1} (the teamwork "
                "strategy's exploration phase)")
        i = self._knob(a.get("knob"))
        try:
            v = int(a.get("value"))
        except (TypeError, ValueError):
            raise ActionError("value must be an integer 0-10") from None
        k._need(LO <= v <= HI, f"knob values are integers {LO}-{HI}")
        self._check_lock(k, i)
        group = None
        if a.get("as"):
            group = k._group(a["as"])
            k._need(k.can_act_for(aid, group), f"you may not act for {group}")
        holder = self.holder(i)
        if holder and holder != aid and not (holder in k.groups and k.can_act_for(aid, holder)):
            raise ActionError(f"{self.name[i]} is held by {holder}; only " +
                              (f"those who may act for {holder}" if holder in k.groups else holder) + " can set it")
        if holder in k.groups and not group:
            group = holder
        old = float(self.x[i])
        if v == old:
            return "lab.set_knob", f"{self.name[i]} is already {v}", self.name[i]
        before = self.cost(self.x)
        d = v - old
        reversal = i in self.last_delta and self.last_delta[i] * d < 0
        users = self.users[i]
        mine = set(self.instruments.get(aid, []))
        span = set()
        if group:
            for x in k.agents_inside(group):
                span |= set(self.instruments.get(x, [])) & set(users)
        self.x[i] = v
        self.last_delta[i] = d
        prev = self.last_set.get(i)
        self.last_set[i] = (aid, group, k.tick)
        gain = (before - self.cost(self.x)) / max(1e-9, self.c_start - self.c_opt)  # score change this set caused
        self.sets.append({"t": k.tick, "knob": self.name[i], "level": str(self.level[i]), "agent": aid, "group": group,
                          "from": old, "to": v, "reversal": reversal, "informed": bool(mine & set(users)),
                          "spanning": len(span) >= 2, "gain": round(gain, 5)})
        for who in (aid, group) if group else (aid,):
            self.credit[who][0] += gain
            self.credit[who][1] += 1
        k.tick_counts["published"] += 1
        self._note_focus(k, aid, {b: 1.0 / len(users) for b in users})
        if prev and prev[0] != aid:
            k.notify(prev[0], aid, f"{aid} changed {self.name[i]} from {old:g} to {v} (you had set it at t{prev[2]})")
        msg = f"set {self.name[i]} from {old:g} to {v}"
        if prev:
            msg += f" (last set by {prev[0]}{'/' + prev[1] if prev[1] else ''} at t{prev[2]})"
        return "lab.set_knob", msg + ".", self.name[i]

    def _own_knob(self, k, aid, a):
        i = self._knob(a.get("knob"))
        holder = self.rights.get(i)
        if str(a.get("op") or "claim") == "release":
            k._need(holder and (holder == aid or holder in k.groups and k.can_act_for(aid, holder)),
                    f"you do not hold {self.name[i]}")
            del self.rights[i]
            return "lab.release_knob", f"released {self.name[i]}", self.name[i]
        k._need(holder is None, f"{self.name[i]} is already held by {holder}")
        k._need(a.get("as") or not self.group_only, "only groups can hold knobs: claim it as a group you may act for (as)")
        who = aid
        if a.get("as"):
            who = k._group(a["as"])
            k._need(k.can_act_for(aid, who), f"you may not act for {who}")
        if self.claim_price:
            have = self.dcredit.get(who, [0.0])[0]
            k._need(have >= self.claim_price, f"claiming a knob costs {self.claim_price} credit; {who} has {have:+.3f}"
                    + (" (a group's own credit, from its area and its changes)" if who in k.groups else " (your own credit)"))
            self.dcredit[who][0] -= self.claim_price
        self.rights[i] = who
        self.claims_log.append({"t": k.tick, "knob": self.name[i], "level": str(self.level[i]), "agent": aid, "holder": who})
        return "lab.own_knob", f"{self.name[i]} is now held by {who}", self.name[i]

    def _view_design(self, k, aid, a):
        refs = a.get("knobs") or []
        if refs in (["all"], "all"):
            ids = list(range(self.D))
        elif refs:
            ids = [self._knob(r) for r in refs[:40]]
        else:
            ids = sorted({int(i) for b in self.instruments.get(aid, [])[:4] for i in self.blocks[b]["idx"]})
        ids.sort(key=lambda i: int(self.name[i][1:]))
        return "lab.view_design", "\n".join(self._knob_line(i) for i in ids), None

    def _knob_line(self, i: int) -> str:
        who = self.last_set.get(i)
        by = f"set by {who[0]}{'/' + who[1] if who[1] else ''} t{who[2]}" if who else "starting value"
        held = f", held by {self.holder(i)}" if self.holder(i) else ""
        used = f", used by {','.join(self.block_label[b] for b in self.users[i])}" if self.leadership else ""
        return f"{self.name[i]} = {self.x[i]:g} ({by}{held}{used})"

    # ------------------------------------------------------------------ views

    def search(self, k, aid, what, words):
        ids = [self.index[w] for w in words if w in self.index]
        pool = ids or list(range(self.D))
        if "mine" in words:
            mine = {int(i) for b in self.instruments.get(aid, []) for i in self.blocks[b]["idx"]}
            pool = [i for i in pool if i in mine]
        authors = {w.upper() for w in words if re.fullmatch(r"a\d+", w)}
        if authors:
            pool = [i for i in pool if i in self.last_set and self.last_set[i][0] in authors]
        if "changed" in words or "recent" in words:
            pool = sorted((i for i in pool if i in self.last_set), key=lambda i: -self.last_set[i][2])
        else:
            pool.sort(key=lambda i: int(self.name[i][1:]))
        return [self._knob_line(i) for i in pool]

    def agent_tags(self, aid: str) -> str:
        bs = self.instruments.get(aid, [])
        if not bs:
            return ""
        if len(bs) > 1:
            return f"simulates {len(bs)} blocks"
        b = bs[0]
        return f"simulates {self.block_label[b]} (knobs {','.join(self.name[i] for i in self.blocks[b]['idx'])})"

    def track_record(self, aid: str) -> str:
        n = sum(s["agent"] == aid for s in self.sets)
        return f"record: {n} knob changes" if n else ""

    def group_record(self, gid: str) -> str:
        n = sum(s["group"] == gid for s in self.sets)
        return f"group record: {n} knob changes" if n else ""

    def observe(self, k, aid: str) -> str:
        bs = self.instruments.get(aid, [])
        last = self.reviews[-1] if self.reviews else None
        left = self.sim_budget(aid) - self.sims[(aid, k.tick)]
        head = f"DESIGN: you simulate {', '.join(self.block_label[b] for b in bs[:6])}{' and more' if len(bs) > 6 else ''}. "
        if len(bs) == 1:
            head += "Its knobs now: " + "; ".join(self._knob_line(int(i)) for i in self.blocks[bs[0]]["idx"]) + ". "
        shown = self.credit_shown.get(aid)
        mine = f" Your credit {shown[0]:+.3f} from {shown[1]} changes (as of the last review)." if shown else ""
        if self.durable and aid in self.effective:
            own = self.dcredit.get(aid, [0.0])[0]
            mine = (f" Your credit {self.effective[aid]:+.3f} (own {own:+.3f}, group shares {self.effective[aid] - own:+.3f}); "
                    f"simulations per tick {self.sim_budget(aid)}.")
        return (head + f"Simulations left this tick {left}. "
                + (f"Last design review t{last['t']}: score {last['score']:.3f}." if last else "No design review yet.") + mine
                + self._institutions_observe(k, aid))

    def brief(self) -> str:
        return f"""THE MISSION
The society settles one shared design: {self.D} knobs k1..k{self.D}, each an integer 0-10 (they start at arbitrary values). The design is made of {len(self.blocks)} blocks. Each block has a cost that depends on some of the knobs (lower is better); the design's cost is the sum of all block costs. Some knobs matter to one block only; others are shared by several blocks, which may want different values for them. {'Each agent pursues its own reward (below); nobody is rewarded for the total.' if self.economy else 'The goal is the lowest total cost.'}
- Each agent can simulate one block (its instrument): simulate returns that block's cost under the current design, and optionally its cost with a few knob values changed hypothetically. Nobody sees other blocks' costs except through what others report. Simulations per tick are limited.
- set_knob changes the one shared design; the latest value stands. view_design and search what=knobs show current values and who set them.{self._rights_brief()}
- Every {self.validate_every} ticks a design review posts the current design's score on the global board: the share of the achievable improvement over the starting design that it realizes (1 = best possible). The final score is the design at the end.{self._credit_brief()}{self._institutions_brief()}"""

    def _credit_brief(self) -> str:
        if self.economy:
            return (f"\n- YOUR REWARD is your own credit. Every {self.credit_every} ticks you earn credit equal to how much "
                    "your own block's cost fell since the last review (as a share of the design's achievable improvement; "
                    "negative if it rose). The lead of a scope of two or more blocks (take_lead) also earns "
                    f"{self.project_mult:g} times the fall in the scope's combined cost: a collective project pays more than "
                    "its parts, but needs someone to coordinate it. Credit buys resources: your simulations per tick are "
                    f"{self.sims_per_tick} plus one per 0.05 of credit (1 to {self.sims_per_tick + 6}), and taking over a led "
                    "scope needs more credit than its lead. " +
                    ("Agents can pay each other through contracts. " if self.contracts_on or not getattr(self, "told_rules", False) else "") +
                    (f"Profit sharing: every agent also earns {self.economy_share:g} times the fall in the design's total "
                     "cost (as a share of the achievable improvement), so everyone gains when the whole design improves."
                     if self.economy_share else "The design's total cost is measured, but it is nobody's reward.") +
                    (self._rules_brief() if getattr(self, "told_rules", False) else ""))
        if self.durable:
            return (f"\n- Credit and resources: every {self.credit_every} ticks, each knob whose value changed since the last "
                    "credit review is credited to whoever set its current value (to the group, if set for a group with as, "
                    "or by a group's holder) with the part of the change that lasted: how much better the design is now "
                    "than with that knob back at its value at the last credit review. Reversed or overwritten changes earn "
                    "nothing. A group's credit is shared equally among the agents inside it; your credit is your own plus "
                    f"your shares. Your simulations per tick are {self.sims_per_tick} plus one per 0.05 of credit (at least "
                    f"1, at most {self.sims_per_tick + 6}), updated at each credit review, and the ledger is posted on the "
                    "global board." +
                    (f" Evidence earns credit: {self.evidence_share:.0%} of a change's positive credit goes to the agents whose "
                     "simulations of that knob with hypothetical values were posted, before the change, on a board the "
                     "agent who made it can read." if self.evidence_share else "") +
                    (f" Holding a knob costs {self.claim_rent} credit per credit review." if self.claim_rent else "") +
                    (" Groups also earn from their area: at each credit review, every group of two or more agents is "
                     "credited with how much the summed cost of the blocks its agents simulate fell since the last review "
                     "(whoever moved the knobs), shared equally among the agents inside it." if self.group_area else "") +
                    (f" Claiming a knob with own_knob costs {self.claim_price} credit, paid from the claimant's own credit "
                     "(yours, or the group's when you claim for a group with as); without enough credit you cannot claim."
                     if self.claim_price else "") +
                    (" Authority follows credit: at each credit review, every knob shared by several blocks is held by the "
                     "agent with the most credit among those whose blocks use it (only that agent can set it until the next "
                     "review); while no user of a knob has positive credit, anyone may set it." if self.rights_by_credit else ""))
        if not self.credit_on:
            return ""
        return ("\n- Credit: every knob change is credited to whoever made it (and to the group it was made for, with as) "
                "with the change in the design's score it caused when it was made, so credits add up to the score. Each "
                "design review posts the credit ledger of agents and groups and tells every agent what its own recent "
                "changes earned.")

    def _rules_brief(self) -> str:
        """The rules in force that the reward text above does not cover. Said under a recipe (told_rules), where any of
        them can be switched on: a rule agents are held to is a rule they are told."""
        return ((f" Holding a knob costs {self.claim_rent:g} credit per credit review." if self.claim_rent else "") +
                (" Groups also earn from their area: at each credit review, every group of two or more agents is credited "
                 "with how much the summed cost of the blocks its agents simulate fell since the last review, shared "
                 "equally among the agents inside it." if self.group_area else "") +
                (f" Claiming a knob with own_knob costs {self.claim_price:g} credit, paid from the claimant's own credit; "
                 "without enough credit you cannot claim." if self.claim_price and self.ownership else "") +
                (" Authority follows credit: at each credit review, every knob shared by several blocks is held by the "
                 "agent with the most credit among those whose blocks use it (only that agent can set it until the next "
                 "review); while no user of a knob has positive credit, anyone may set it." if self.rights_by_credit else ""))

    def _rights_brief(self) -> str:
        return self._holding_brief() + (" A task (delegate) whose scope names knobs hands them to its taker for as long as it lives; only "
                                        "whoever holds a knob, or anyone while nobody does, can hand it over.")

    def _holding_brief(self) -> str:
        if self.dictators:
            which = "Every knob" if self.dictate_private else "Every knob shared by several blocks"
            return (f"\n- {which} is held by one agent whose block depends on it: only that agent can set it (view_design "
                    "shows holders)." + ("" if self.dictate_private else " Anyone may set the other knobs."))
        if self.authority:
            return ("\n- Knobs are held by the organization's groups: a team holds its block's knobs, a division the knobs "
                    "shared within its blocks, the program the knobs shared by all. Only agents who may act for the holding "
                    "group can set a knob (view_design shows holders).")
        if self.ownership and self.group_only:
            return ("\n- Anyone may set a knob nobody holds. own_knob lets a group claim exclusive control of a knob (agents "
                    "cannot hold knobs themselves); while held, only those who may act for the holding group can set it.")
        if self.ownership:
            return ("\n- Anyone may set a knob nobody holds. own_knob lets an agent or a group claim exclusive control of a "
                    "knob; while held, only the holder (or those who may act for the holding group) can set it.")
        return " Anyone may set any knob."

    # ------------------------------------------------------------------ scoring (hidden from agents)

    def normalized(self, x: np.ndarray, clamp: bool = False) -> float:
        """Share of the achievable improvement realized: 1 at the optimum, 0 at the starting design, negative when the
        design is worse than where it started (unclamped, so damage stays visible)."""
        gap = self.c_start - self.c_opt
        v = float((self.c_start - self.cost(x)) / gap) if gap > 0 else 1.0
        return min(1.0, max(0.0, v)) if clamp else v

    def level_gaps(self, x: np.ndarray) -> dict[str, float]:
        """Improvement still available by fixing one level's knobs to the optimum (others as they are), per achievable gap."""
        gap = max(1e-9, self.c_start - self.c_opt)
        c = self.cost(x)
        out = {}
        for lv in self.levels:
            y = x.copy()
            m = self.level == lv
            y[m] = self.x_opt[m]
            out[lv] = round((c - self.cost(y)) / gap, 4)
        return out

    def score(self) -> dict:
        shared = [s for s in self.sets if s["level"] != "private"]
        gaps = self.level_gaps(self.x)
        return {"score": round(self.normalized(self.x), 4), "score_clamped": round(self.normalized(self.x, clamp=True), 4),
                "coverage": round(len(self.last_set) / self.D, 4),
                "score_base": None, "score_composite": None, "structure_f1": None, "val": None,
                "gap_private": gaps["private"], "gap_interface": gaps["interface"], "gap_top": gaps["top"],
                "gap_mid": gaps.get("mid"),
                "at_optimum": round(float(np.mean(self.x == self.x_opt)), 4),
                "knob_sets": len(self.sets), "shared_sets": len(shared),
                "shared_reversals": sum(s["reversal"] for s in shared),
                "shared_informed": round(sum(s["informed"] for s in shared) / len(shared), 4) if shared else None,
                "shared_by_spanning_group": round(sum(s["spanning"] for s in shared) / len(shared), 4) if shared else None,
                "sets_for_groups": sum(bool(s["group"]) for s in self.sets),
                "knobs_held_by_groups": sum(h.startswith("G") for h in self.rights.values()),
                "knobs_held_by_agents": sum(h.startswith("A") for h in self.rights.values()),
                **({f"tasks_{key}": v for key, v in self._k.delegation_measures().items()} if getattr(self, "_k", None) else {}),
                "knobs_delegated": len(self.delegated)} | self._institutions_score()

    def author_scores(self) -> dict:
        return {}  # no per-agent validated record in this world (the expertise metric stays empty)

    def dominant_modules(self, tick: int, window: int = 3) -> dict[str, int]:
        out = {}
        for aid, items in self.focus.items():
            w = defaultdict(float)
            for t, m, x in items:
                if t > tick - window:
                    w[m] += x
            if w:
                out[aid] = max(w.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return out

    def home_module(self, aid: str) -> int:
        return self.instruments[aid][0]

    def team_purpose(self, m: dict | None) -> str:
        if m is None:
            return "Lower the design's total cost."
        b = m["id"]
        return f"Settle the knobs of block {self.block_label[b]} ({', '.join(self.name[i] for i in self.blocks[b]['idx'])})."

    def agent_export(self, aid: str) -> dict:
        return {"instruments": [self.block_label[b] for b in self.instruments.get(aid, [])], "record": self.track_record(aid),
                "experiments": self.used[aid]}

    def export(self) -> dict:
        knobs = [{"knob": self.name[i], "level": str(self.level[i]), "owner": self.owner[i], "value": float(self.x[i]),
                  "start": float(self.x_start[i]), "optimum": float(self.x_opt[i]),
                  "blocks": [self.block_label[b] for b in self.users[i]],
                  "last_set": list(self.last_set[i]) if i in self.last_set else None} for i in range(self.D)]
        return {"kind": "design", "structure": "design", "D": self.D, "n_blocks": len(self.blocks), "n_super": self.n_super,
                "modules": self.modules, "c_start": self.c_start, "c_opt": self.c_opt, "c_final": self.cost(self.x),
                "score": self.normalized(self.x), "level_gaps": self.level_gaps(self.x), "knobs": knobs,
                "sets": self.sets, "reviews": self.reviews, "shocks": self.shocks, "claims": self.claims_log,
                "credit": {w: [round(c, 5), n] for w, (c, n) in self.credit.items()}, "credit_on": self.credit_on,
                **self._institutions_export(), "dependency": self.dependency,
                "task_log": self._k.task_log if getattr(self, "_k", None) else [],
                "durable_credit": {w: [round(c, 5), n] for w, (c, n) in self.dcredit.items()},
                "effective_credit": {a: round(v, 5) for a, v in self.effective.items()}, "sim_bonus": self.sim_bonus,
                "rights": {self.name[i]: h for i, h in self.rights.items()},
                "params": {"private": self.n_private, "interface": self.n_interface, "top": self.n_top,
                           "top_per_block": self.top_per_block, "sims_per_tick": self.sims_per_tick,
                           "review_every": self.validate_every}}

    # ------------------------------------------------------------------ scripted agents (mock backend)

    def mock_actions(self, k, aid, t, rng):
        """A selfish block engineer: simulate, then move one of its block's knobs toward the block's own optimum."""
        b = self.instruments[aid][0]
        blk = self.blocks[b]
        i = int(rng.choice(list(blk["idx"])))
        best = min(range(LO, HI + 1), key=lambda v: self.block_cost(b, np.where(np.arange(self.D) == i, v, self.x)))
        return [("simulate", {"changes": {self.name[i]: best}} if len(self.instruments[aid]) == 1 else
                 {"changes": {self.name[i]: best}, "block": self.block_label[b]}),
                ("set_knob", {"knob": self.name[i], "value": best}),
                ("view_design", {})]


def describe(w: DesignWorld) -> str:
    """One line summarizing a world's size (for logs and reports)."""
    per_block = Counter(b for bs in w.instruments.values() for b in bs)
    return (f"{w.D} knobs ({len(w.blocks)} blocks x {w.n_private} private, {w.n_super} super-blocks x {w.n_interface} "
            f"interface, {w.n_top} top; each block sees {w.top_per_block} top knobs); {len(w.instruments)} agents, "
            f"{min(per_block.values())}-{max(per_block.values())} per block; C_start {w.c_start:.0f}, C_opt {w.c_opt:.0f}")
