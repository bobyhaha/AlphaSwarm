"""What the worlds of the society thesis share (code_world.py, science_world.py): the law, wealth goals, roles, trace, pay.

GOAL.md: agents get actions they can extend, and credit assignment is the central means. The user's frame (2026-11-03): accuracy
is the only goal, any level of reflection and change is allowed, and credit can make an agent's own wealth, its team's or the
society's the goal. So a world built on this class has:
- a law: everything a review may change, in one bounded dictionary, with the goal deciding whose wealth a payment is;
- the attribution kept whatever the goal does with it, and a ledger of which actions and combinations paid;
- role (a title of one's own), trace (read what another agent has been doing) and pay (give, or fine at a cost to oneself);
- practices: offered text from the society's review, with agent and group ids stripped, because a review never says who does what.

A world supplies LAW, LAW_BOUNDS, LAW_CHOICES, SHOW, WORDS, GAME and law_words(), and the hooks alphateam/reflection.py reads:
progress(), window_facts() and held_log.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from .explore import Explore
from .kernel import ActionError, as_list, clip

GOALS = ("individual", "team", "society")
RULE_KEYS = ("lead_share", "action_share", "steps_per_credit")
RULE_MAX = {"lead_share": 0.6, "action_share": 0.6, "steps_per_credit": 4.0}
IDS = re.compile(r"\b[AGOPET]\d+\b")  # who does what is never written into a practice
WORKING = frozenset(("shell", "probe", "publish", "head", "answer", "calc", "open_task"))
READING = frozenset(("read", "search", "trace"))

TRACE_TOOL = {
    "name": "trace",
    "description": "Read what another agent has been doing: its commands and their results, newest last. Open to everyone; spends "
                   "your reading budget.",
    "parameters": {"type": "object", "properties": {
        "agent": {"type": "string"}, "last": {"type": "integer", "description": "how many items (default 10, max 40)"}},
        "required": ["agent"]}}
ROLE_TOOL = {
    "name": "role",
    "description": "Say what you do in this society: a short title and what others can bring you. Everyone sees it beside your "
                   "name. Change it whenever you like; nobody assigns it.",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "does": {"type": "string"},
                                                    "team": {"type": "string", "description": "the group whose wealth yours is, under a team goal"}},
                   "required": ["title"]}}
PAY_TOOL = {
    "name": "pay",
    "description": "Move credit. op=give (default) gives some of yours to another agent at once. op=fine spends some of yours to take "
                   "more from another agent (the rate is in the law); what is taken goes to nobody, and the fine is public.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["give", "fine"]}, "to": {"type": "string"}, "amount": {"type": "number"},
        "why": {"type": "string"}}, "required": ["to", "amount"]}}


def bound_law(given, base: dict, bounds: dict, choices: dict, whole: tuple = ()) -> dict:
    """A bounded law: numbers within their bounds, choices among those offered; anything else stays as it is in base."""
    out = dict(base)
    given = given if isinstance(given, dict) else {}
    for key, value in given.items():
        if key in choices:
            out[key] = value if value in choices[key] else out[key]
        elif key in bounds:
            try:
                low, high = bounds[key]
                value = min(high, max(low, float(value)))
            except (TypeError, ValueError):
                continue
            out[key] = int(round(value)) if key in whole else round(value, 3)
    return out


class SocietyWorld(Explore):
    GAME = "society"
    LAW: dict = {}
    LAW_BOUNDS: dict = {}
    LAW_CHOICES: dict = {}
    LAW_WHOLE: tuple = ()
    SHOW: tuple = ()
    SHOW_DEFAULT: tuple = ()
    WORDS: dict = {}
    final_words = "act"
    quiet_closing = True  # receipts and traces hold what a move did, so a reply with no text posts no note
    has_modules = False
    supports_replay = False

    def _init_society(self, n_agents: int, single: bool, world_agents: int | None, opts: dict):
        on = lambda key, default="1": str(opts.get(key, default)).lower() in ("1", "true", "yes")
        self.single = single
        self.aids = [f"A{i + 1}" for i in range(n_agents)]
        self.scale = (world_agents or n_agents) if single else 1
        self.incentive = on("credit")
        self.market_on = self.incentive and not single
        self.goal_locked = on("goal_locked", "0")
        self.ways_on = on("ways", "0")  # organizational examples in the brief: an arm, off by default, so structure owes them nothing
        self.force_budget = on("force_budget", "0")  # agents may not end a tick with replies left, so a budget sweep spends its budget
        # what Explore expects of a world: one problem that never falls, so nothing it holds is ever swept off the table
        self.games = [self.GAME]
        self.assigned = {a: self.GAME for a in self.aids}
        self.best, self.win = {self.GAME: 0}, {self.GAME: 1}
        self.state = {(a, self.GAME): {"levels": 0} for a in self.aids}
        self.firsts: list[dict] = []
        self.home = {a: [self.GAME] for a in self.aids}
        self.tasks = {self.GAME: self.GAME}
        self.sync = on("sync", "0")
        self.focus = defaultdict(list)
        self.used = Counter()
        self.workdir = None
        self.started = False
        self._k = None
        for key in ("explore", "hypotheses", "trust", "guide"):
            opts.setdefault(key, "1")
        opts.setdefault("delegation", "0" if single else "1")
        opts["model"], opts["carry"] = "0", "0"
        opts["no_contracts"] = "1"  # delegate with pay does what contract did, and it can be handed on
        self._init_explore(opts)
        c = self.rules["credit"]
        c.update(finding=0.0, dead_end=0.0, idea=0.0, test=0.0, understanding=0.0, direction=0.0, reuse=0.0, priority=0.0, novel=0.0,
                 steps_per_credit=float(opts.get("steps_per_credit", 1.0)), lead_share=float(opts.get("lead_share", 0.0)))
        self.rules["substrate"]["show"] = list(self.SHOW_DEFAULT)
        self.rules["substrate"]["items"] = 50  # views list everything
        self.law = self.clean_law(json.loads(opts["law"]) if str(opts.get("law", "")).startswith("{") else {}, self.LAW)
        self.law = self.clean_law({key: opts[key] for key in self.LAW if key in opts}, self.law)
        self.law_v, self.law_log = 1, []
        self.norms, self.norms_t = str(opts.get("norms") or ""), 0
        self.steps_per_tick = self.law["steps"] * self.scale
        self.TOOLS = [self._in_words(t) for t in self.TOOLS if t["name"] != "focus"]
        self.steps: Counter = Counter()
        self.attributed: Counter = Counter()  # agent -> credit it earned by attribution, whatever the goal does with it
        self.held_log: list[dict] = []
        self.payments: list[dict] = []
        self.feedback: dict[str, list] = defaultdict(list)
        self.titles: dict[str, dict] = {}
        self.title_log: list[dict] = []
        self.traces: dict[str, list] = defaultdict(list)  # agent -> its work with results
        self.reads: Counter = Counter()  # (reader, whose work) -> traces and group copies read
        self.suggestions: list[dict] = []  # what reviews suggested; nothing makes anyone follow it
        self.ids = Counter()

    def add_tools(self, tools: list[dict]):
        self.TOOLS = list(self.TOOLS) + tools + ([PAY_TOOL] if self.market_on else [])
        self.tool_names = tuple(t["name"] for t in self.TOOLS)

    def _in_words(self, tool: dict) -> dict:
        name = tool["name"]
        if name not in self.WORDS:
            return tool
        props = {key: value for key, value in tool["parameters"]["properties"].items() if key != "game"}
        return {**tool, "description": self.WORDS[name], "parameters": {**tool["parameters"], "properties": props}}

    def _plain(self, text: str) -> str:
        """Explore speaks of a game with levels; this world has one problem that never falls."""
        text = re.sub(rf"\b{re.escape(self.GAME)} level \d+\b", self.GAME, text)
        text = re.sub(r"\b(on|at) level \d+\b", "", text)
        text = re.sub(r"\bL1 ", "", text)
        text = re.sub(r"\blevel \d+\b", "the work", text)
        return text.replace("of this game across levels", f"of {self.GAME}")

    def _id(self, prefix: str) -> str:
        self.ids[prefix] += 1
        return f"{prefix}{self.ids[prefix]}"

    def setup(self, k):
        self._k = k
        k.rank = dict(self.rules["substrate"]["rank"])

    def act(self, k, aid: str, name: str, a: dict):
        self._now = name
        return getattr(self, f"_{name}")(k, aid, a)

    def _plan(self, k, aid, a):
        a = dict(a)
        if a.get("op") == "claim" and a.get("result") and not a.get("hypothesis"):  # a test written as a claim (seen in I9)
            a["op"] = "test"
            a.setdefault("entry", (as_list(a.get("builds_on")) or [""])[0])
        board = k.boards[self._substrate_board(k, aid)]
        start = len(board.entries)
        op, msg, target = super()._plan(k, aid, a)
        for e in board.entries[start:]:
            e.content = self._plain(e.content)
        return op, self._plain(re.sub(r"; \+0 credit for a test that moved it from \w+", "", msg)), target  # tests pay through use

    def steps_allowed(self, aid: str, tick: int) -> int:
        if not self.incentive:
            return self.steps_per_tick
        return super().steps_allowed(aid, tick)

    def _step(self, k, aid: str):
        allowed = self.steps_allowed(aid, k.tick)
        k._need(self.steps[(aid, k.tick)] < allowed, f"you have used all {allowed} steps this tick")
        self.steps[(aid, k.tick)] += 1
        self.used[aid] += 1

    def _peers(self, k, aid: str) -> list[str]:
        out = set()
        for gid in k.member_of.get(aid, ()):
            out |= set(k.agents_inside(gid))
        return sorted(out - {aid})

    def _log_work(self, k, aid: str, tool: str, given: str, result: str) -> str:
        self.traces[aid].append({"t": k.tick, "tool": tool, "in": clip(given, 300), "out": clip(result, 600)})
        return self._receipt(k, aid, self.GAME, 0, [f"{tool}: {clip(given, 120)}"], [clip(result, 200)], 0, start=[clip(given, 80)])

    def _trace(self, k, aid, a):
        who = str(a.get("agent") or "").strip().upper()
        k._need(who in k.agents, "agent: an agent id")
        rows = self.traces[who][-max(1, min(40, int(a.get("last") or 10))):]
        text = "\n".join(f"t{r['t']} {r['tool']}: {r['in']}\n  -> {r['out']}" for r in rows) or "(nothing yet)"
        k._need(k.spend_attention(aid, text), "your reading budget for this tick is spent")
        if who != aid:
            self.reads[(aid, who)] += 1
        return "lab.trace", f"{who}'s last {len(rows)} items:\n{text}", who

    def _role(self, k, aid, a):
        title = clip(str(a.get("title") or "").strip(), 60)
        k._need(title, "title: what you do, in a few words")
        pool = str(a.get("team") or "").strip().upper() or self.titles.get(aid, {}).get("pool")
        k._need(not pool or (pool in k.groups and k.inside(aid, pool)), f"{pool} is not a group of yours")
        self.titles[aid] = {"title": title, "does": clip(str(a.get("does") or "").strip(), 160), "t": k.tick, "by": aid, "pool": pool}
        self.title_log.append({"t": k.tick, "agent": aid, "title": title, "by": aid})
        return "lab.role", f"everyone now sees you as: {title}", aid

    # ------------------------------------------------------------------ credit

    def trust_weight(self, level: str) -> float:
        return {"reproduced": self.law["reproduced"], "seen": self.law["seen"], "said": self.law["said"]}.get(level, 0.0)

    def team_of(self, k, aid: str) -> list[str]:
        """Who an agent shares wealth with: everyone inside the group it declared with role team=G1, or else its smallest group, so
        that an umbrella over two teams changes nobody's income unless they choose it. Alone, itself."""
        chosen = self.titles.get(aid, {}).get("pool")
        groups = [g for g in k.member_of.get(aid, ()) if g in k.groups and k.groups[g].dissolved is None]
        if chosen and chosen in k.groups and k.groups[chosen].dissolved is None and k.inside(aid, chosen):
            return sorted(k.agents_inside(chosen), key=lambda a: int(a[1:]))
        if not groups:
            return [aid]
        return sorted(k.agents_inside(min(groups, key=lambda g: (len(k.agents_inside(g)), g))), key=lambda a: int(a[1:]))

    def _pay_out(self, k, rows: list[tuple]):
        """The goal in the law decides whose wealth a payment is. The attribution is kept whatever the goal does with it."""
        goal = self.law["goal"]
        for who, amount, why in rows:
            if amount <= 0:
                continue
            self.attributed[who] += amount
            if goal == "individual":
                self._award(k, who, amount, why)
                continue
            combo = self._combo(k, who)  # the ledger of what paid follows the earner's actions, whoever shares the credit
            key = "+".join(combo) or "(none)"
            self.combo_credit[key][0] += amount
            self.combo_credit[key][1] += 1
            for op in combo:
                self.op_credit[op][0] += amount
                self.op_credit[op][1] += 1
            mates = list(k.active_agents()) if goal == "society" else self.team_of(k, who)
            for mate in mates:
                self._award(k, mate, amount / len(mates), f"shared with {goal}: {why}", _depth=1, share=False, label="shared")

    def on_task(self, k, op: str, t, aid: str, a: dict):
        """Delegated work: pay is held while a task lives, drawn from the parent's held pay when its taker hands part of it on."""
        if op == "post":
            pay = round(float(a.get("pay") or 0), 2)
            if not pay:
                return
            k._need(self.incentive and pay > 0, "pay: a positive amount, where this society has credit")
            parent = k.tasks.get(t.parent) if t.parent else None
            if parent is not None and parent.taker == aid and parent.held >= pay:
                parent.held = round(parent.held - pay, 2)
                t.payer = parent.id
            else:
                k._need(pay <= self.credit[aid], f"you have {self.credit[aid]:.1f} credit")
                self.credit[aid] -= pay
                t.payer = aid
            t.held = pay
        elif op == "accepted" and t.held and t.taker:
            self._award(k, t.taker, t.held, f"paid for {t.id}", share=False)
            self.payments.append({"t": k.tick, "from": t.payer, "to": t.taker, "amount": t.held, "op": "task", "why": t.id})
            t.held = 0.0
        elif op in ("dropped", "lapsed", "refused") and t.held:
            back = k.tasks.get(t.payer)
            if back is not None:
                back.held = round(back.held + t.held, 2)
            elif t.payer:
                self.credit[t.payer] += t.held
            t.held = 0.0

    def delegation_measures(self, k) -> dict:
        return k.delegation_measures()

    def _tell(self, aid: str, text: str):
        self.feedback[aid].append(text)

    def _pay(self, k, aid, a):
        op, to, amount = str(a.get("op") or "give"), str(a.get("to") or "").strip().upper(), float(a.get("amount") or 0)
        k._need(to in k.agents and to != aid, "to: another agent")
        k._need(0 < amount <= self.credit[aid], f"you have {self.credit[aid]:.1f} credit")
        why = clip(str(a.get("why") or ""), 120)
        self.credit[aid] -= amount
        if op == "fine":
            taken = min(self.credit[to], amount * self.law["fine_ratio"])
            self.credit[to] -= taken
            self.payments.append({"t": k.tick, "from": aid, "to": to, "amount": amount, "taken": taken, "op": "fine", "why": why})
            k.notify(to, aid, f"fined you {taken:g} credit: {why}")
            k.post("global", "lab", "result", f"FINE: {aid} spent {amount:g} to take {taken:g} from {to}: {why}", notify=False)
            return "lab.fine", f"{to} lost {taken:g} credit; you spent {amount:g}", to
        k._need(op == "give", "op: give or fine")
        self._award(k, to, amount, f"from {aid}: {why}", share=False)
        self.payments.append({"t": k.tick, "from": aid, "to": to, "amount": amount, "op": "give", "why": why})
        k.notify(to, aid, f"paid you {amount:g} credit: {why}")
        return "lab.pay", f"paid {to} {amount:g} credit", to

    # ------------------------------------------------------------------ the law: what a review may change

    def clean_law(self, given, base: dict | None = None) -> dict:
        return bound_law(given, base if base is not None else self.LAW, self.LAW_BOUNDS, self.LAW_CHOICES, self.LAW_WHOLE)

    def law_words(self) -> str:
        raise NotImplementedError

    def set_law(self, k, law: dict | None = None, rules: dict | None = None, norms: str | None = None, why: str = "") -> dict:
        """The rules of the society: what a review may change. It never says who does what: a practice names no agent or group."""
        before = {"law": dict(self.law), "rules": {x: self.rules["credit"][x] for x in RULE_KEYS}, "norms": self.norms,
                  "show": list(self.rules["substrate"]["show"])}
        new = self.clean_law(law, self.law)
        if self.goal_locked:
            new["goal"] = before["law"]["goal"]
        self.law = new
        self.steps_per_tick = new["steps"] * self.scale
        for key in RULE_KEYS:
            try:
                value = float((rules or {}).get(key, self.rules["credit"][key]))
            except (TypeError, ValueError):
                continue
            self.rules["credit"][key] = min(RULE_MAX[key], max(0.0, value))
        show = (rules or {}).get("show")
        if isinstance(show, list):
            self.rules["substrate"]["show"] = [v for v in self.SHOW if v in show] or self.rules["substrate"]["show"]
        if norms is not None:
            self.norms, self.norms_t = clip(" ".join(IDS.sub("", str(norms)).split()), 700), k.tick
        changed = {x: [before["law"][x], new[x]] for x in new if before["law"][x] != new[x]}
        changed.update({x: [before["rules"][x], self.rules["credit"][x]] for x in RULE_KEYS if before["rules"][x] != self.rules["credit"][x]})
        if before["show"] != self.rules["substrate"]["show"]:
            changed["show"] = [before["show"], self.rules["substrate"]["show"]]
        if norms is not None and before["norms"] != self.norms:
            changed["practices"] = [before["norms"], self.norms]
        if changed:
            self.law_v += 1
            self.law_log.append({"t": k.tick, "v": self.law_v, "changed": changed, "why": clip(why, 400)})
            k.post("global", "lab", "decision", f"THE LAW v{self.law_v} from t{k.tick}: {self.law_words()}"
                   + (f" Practices: {self.norms}" if self.norms else "") + (f" Why: {clip(why, 400)}" if why else ""), notify=False)
        return {"before": before, "changed": changed}

    def suggest(self, k, items) -> list[str]:
        """A review's suggestions are posts: they say what the review saw, and it stays with the agents to act on them or not."""
        made = [clip(str(x).strip(), 300) for x in as_list(items) if str(x).strip()]
        for text in made:
            self.suggestions.append({"t": k.tick, "text": text})
        return made

    def action_facts(self, k, start: int) -> dict:
        """What agents did since tick start and what paid: facts for reflection, no advice."""
        ops = Counter()
        for e in k.events:
            if e["t"] > start and e["ok"] and e["op"] != "intent":
                ops[e["op"].split(".")[1] if e["op"].startswith("lab.") else e["op"].split(".")[0]] += 1
        work = sum(n for op, n in ops.items() if op in WORKING)
        reading = sum(n for op, n in ops.items() if op in READING)
        organizing = sum(ops.values()) - work - reading
        return {"action_mix": dict(ops.most_common()), "working_share": round(work / max(1, sum(ops.values())), 3),
                "organizing_share": round(organizing / max(1, sum(ops.values())), 3),
                "what_paid": sorted(([c, round(v[0], 2), v[1]] for c, v in self.combo_credit.items() if c != "shared"), key=lambda x: -x[1]),
                "credit_by_action": {op: [round(v[0], 2), v[1]] for op, v in self.op_credit.items() if op != "shared"},
                "actions_never_used": self.unused_actions(k), "who_reads_whom": [[a, b, n] for (a, b), n in self.reads.most_common(12)],
                "tasks": self.delegation_measures(k)}

    def group_facts(self, k, score_of) -> dict:
        """Groups as they stand, and what each produced per member against agents in no group."""
        agents = list(k.active_agents())
        groups = []
        for g in k.active_groups():
            inside = k.agents_inside(g.id)
            groups.append({"id": g.id, "name": g.name, "agents": list(inside), "lead": self.teams.get(g.id, {}).get("lead"),
                           "output_per_member": round(sum(score_of(a) for a in inside) / max(1, len(inside)), 2)})
        loose = [a for a in agents if not k.member_of.get(a)]
        return {"groups": groups, "in_no_group": loose, "output_per_agent_in_no_group": round(sum(score_of(a) for a in loose) / max(1, len(loose)), 2)}

    # ------------------------------------------------------------------ what agents see

    def agent_tags(self, aid: str) -> str:
        return self.titles.get(aid, {}).get("title", "")

    def unused_actions(self, k) -> list[str]:
        used = {e["op"].split(".")[1] if e["op"].startswith("lab.") else e["op"].split(".")[0]
                for e in k.events if e["ok"] and e["op"] != "intent"}
        alias = {"own": ("claim", "accept", "reject", "release", "pass"), "pay": ("pay", "fine"), "plan": ("plan", "test", "challenge")}
        return [n for n in super().unused_actions(k) if not used & set(alias.get(n, ()))]

    def sync_rank(self, k, aid: str, e) -> int:
        m = self.entry_meta.get(e.id) or {}
        return 0 if m.get("type") == "challenge" and m.get("target") == aid else 2

    def start_text(self, r: dict) -> str:
        return " ".join(r.get("start") or []) or "the start"

    def receipt_line(self, rid: str) -> str:
        r = self.receipts.get(rid)
        if not r:
            return f"{rid}: no such receipt"
        return f"{rid} (engine record, t{r['t']} {r['agent']} {r['moves'][0]}: {clip(r['steps'][0], 140)})"

    def recent_work(self, k, aid: str) -> str:
        mine = [r for r in self.receipts.values() if r["agent"] == aid][-3:]
        return ("Your latest receipts (cite them as evidence):\n" + "\n".join(f"- {self.receipt_line(r['id'])}" for r in mine)) if mine else ""

    def wealth_lines(self, k, aid: str) -> list[str]:
        if not self.incentive:
            return []
        team = self.team_of(k, aid)
        out = [f"Wealth: yours {self.credit[aid]:.1f}" + (f"; your team's {sum(self.credit[a] for a in team):.1f}" if len(team) > 1 else "")
               + f"; the society's {sum(self.credit.values()):.1f}; " + ", ".join(f"{a} {c:.1f}" for a, c in self.credit.most_common()) + "."]
        if self.law["goal"] != "individual" and self.attributed:
            out.append("What each contributed, by the engine's attribution: " + ", ".join(f"{a} {c:.1f}" for a, c in self.attributed.most_common()) + ".")
        return out

    def goal_text(self) -> str:
        return {"individual": "YOUR GOAL: maximize your own wealth. Credit paid to you is yours.",
                "team": "YOUR GOAL: maximize your team's wealth. Credit paid to a group's member goes to everyone in the group, equally: the "
                        "group it names with role team=G1, or else its smallest group. An agent in no group keeps its own.",
                "society": "YOUR GOAL: maximize the society's wealth. All credit, whoever earned it, is shared by all agents equally."
                }[self.law["goal"]]

    def author_scores(self) -> dict:
        return {}

    def dominant_modules(self, tick: int, window: int = 3) -> dict:
        return {}


__all__ = ["SocietyWorld", "GOALS", "RULE_KEYS", "bound_law", "ActionError"]
