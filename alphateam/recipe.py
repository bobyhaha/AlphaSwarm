"""Collaboration recipes: the rules a society starts from, as a typed, versioned bundle (DESIGN.md section 20).

What evolves in AlphaTeam is not a team but the rules of the game. A recipe has five parts. The system enforces the first
four; the fifth is advice:

  rights        who may do what       who holds shared decisions, whether claims are offered, what they cost
  information   who sees what         messages, what is delivered to whom, how much of its past an agent keeps
  incentives    who is paid for what  shares of the total, pay for leading, pay for evidence
  procedures    which steps exist     institutions on offer, how often reviews run, actions earlier societies designed
  advice        what agents are told  lessons from earlier societies, in their words

Three rules keep a recipe honest:
  - Rules, not assignments. A recipe never names an agent or a group, so the same recipe runs at any team size and
    leads and teams form anew in every run. Text that names one is refused.
  - Enforced parts carry the weight. A revision is a list of typed edits. One that changes only advice is marked,
    so that selection can first ask whether behaviour moved (alphateam/selection.py).
  - Some things never evolve. The world, the budget (agents, ticks, replies, simulations), the model, the score and
    the kernel's permissions and accounting are the frame. A recipe compiles to world options only, and only to the
    options its registry names.

A world registers its settings in REGISTRY. options() turns a recipe into world options; to_cfg() merges them into a
frame (a base run configuration); apply_live() changes the rules of a running society for a branch test, for the
settings that can change mid-run. Rulebook is what the kernel's optional `rules` action talks to: agents view the
rules, propose a change for later societies, back a proposal, or leave a lesson.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field

from .kernel import ACTION_NAME, ActionError, CustomAction, clip, est_tokens

PARTS = ("rights", "information", "incentives", "procedures", "advice")
ENFORCED = PARTS[:4]
# What a recipe may not carry: the name of an agent or a group, or of anything that exists in one run only (an entry,
# a proposal, a task). Best effort: it also catches "agent 7" and "a3", and it will rewrite the odd innocent "A4".
NAMES = re.compile(r"\b(?:agent|group)\s*#?\d+\b|\b[AG]\d+\b", re.I)
RUN_IDS = re.compile(r"\b[EPT]\d+\b")
# names that mean something in one task instance only (they are shuffled from instance to instance)
LOCAL = {"design": ((re.compile(r"\bknob\s*#?\d+\b|\bk\d+\b", re.I), "a knob"),
                    (re.compile(r"\bblock\s*#?\d+\b|\bB\d+\b", re.I), "a block"))}
MAX_ADVICE, MAX_LESSON, ADVICE_CHARS = 12, 400, 2400  # a recipe's advice has a size budget: every agent is shown all of it
STATUSES = ("untested", "confirmed", "refuted")
EDITS = ("set", "advise", "unadvise", "adopt", "unadopt")


class RecipeError(ValueError):
    """A recipe or an edit that the registry does not allow. The message says what is allowed."""


# ---------------------------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class Setting:
    key: str
    part: str
    kind: str  # bool | choice | int | float
    default: object
    doc: str  # one line, shown to agents: what it sets
    choices: tuple = ()
    lo: float | None = None
    hi: float | None = None
    opt: str | None = None  # the world option it compiles to (default: its key)
    encode: dict | None = None  # value -> option text (default: 1/0 for a bool, the number otherwise)
    needs: tuple = ()  # ((key, values), ...): it has an effect only while these other settings hold
    live: object = None  # (kernel, world, value) -> None changes a running society; None: only at the start of a run
    uses: tuple = ()  # counts of a run that show its mechanism in use: names of facts, or the starts of action labels

    def allowed(self) -> str:
        if self.kind == "bool":
            return "on | off"
        if self.kind == "choice":
            return " | ".join(self.choices)
        return f"{self.lo:g} to {self.hi:g}" + (" (whole numbers)" if self.kind == "int" else "")

    def coerce(self, value):
        """The value as the registry holds it, or RecipeError naming what is allowed."""
        try:
            if self.kind == "bool":
                if isinstance(value, bool):
                    return value
                text = str(value).strip().lower()
                if text in ("1", "true", "yes", "on"):
                    return True
                if text in ("0", "false", "no", "off"):
                    return False
                raise ValueError
            if self.kind == "choice":
                text = str(value).strip().lower()
                if text not in self.choices:
                    raise ValueError
                return text
            num = float(value)
            if isinstance(value, bool) or not math.isfinite(num):
                raise ValueError
            if self.kind == "int":
                if num != int(num):
                    raise ValueError
                num = int(num)
            else:
                num = round(num, 4) + 0.0  # no negative zero: equal settings are one recipe
            if not self.lo <= num <= self.hi:
                raise ValueError
            return num
        except (TypeError, ValueError, OverflowError):
            raise RecipeError(f"{self.key} takes {self.allowed()}, not {value!r}") from None

    def option(self, value) -> str:
        if self.encode is not None:
            return self.encode[value]
        if self.kind == "bool":
            return "1" if value else "0"
        if self.kind == "choice":
            return str(value)
        return f"{value:g}"


def show(value) -> str:
    return ("on" if value else "off") if isinstance(value, bool) else (f"{value:g}" if isinstance(value, float) else str(value))


def _world(name, cast=float):
    return lambda k, env, v: setattr(env, name, cast(v))


def _kernel(name, cast):
    return lambda k, env, v: setattr(k, name, cast(v))


def _rights_by_credit(k, env, v):
    env.rights_by_credit = bool(v)
    if not v:  # what credit had assigned is free again, and the holder rule (if on) takes back what credit had taken over
        for i, h in env.credit_held.items():
            if env.rights.get(i) == h:
                del env.rights[i]
        env.credit_held = {}
        if env.dictators:
            env.set_holders(k, "all" if env.dictate_private else "shared")


# The design world's rules (alphateam/design_world.py, alphateam/institutions.py). Defaults are the free arm of
# tools/launch_design.sh: equal rights and every institution on offer.
DESIGN = (
    Setting("holders", "rights", "choice", "none", choices=("none", "shared", "all"), opt="authority",
            encode={"none": "", "shared": "agent_shared", "all": "agent"},
            doc="who may set a knob: anyone (none); for each knob several blocks use, one agent drawn at random from "
                "those whose blocks use it (shared); or one such holder for every knob (all)",
            live=lambda k, env, v: env.set_holders(k, v)),
    Setting("claims", "rights", "choice", "agents", choices=("off", "agents", "groups"), opt="ownership",
            encode={"off": "", "agents": "1", "groups": "group"},
            doc="whether own_knob is offered: off; an agent or a group may claim a knob nobody holds (agents); only a "
                "group may (groups)"),
    Setting("claim_price", "rights", "float", 0.0, lo=0, hi=0.2, needs=(("claims", ("agents", "groups")),),
            doc="credit a claimant pays to claim a knob", live=_world("claim_price")),
    Setting("claim_rent", "rights", "float", 0.0, lo=0, hi=0.1,
            doc="credit every holder pays for each knob it holds, at each credit review", live=_world("claim_rent")),
    Setting("rights_by_credit", "rights", "bool", False,
            doc="at each credit review, every knob several blocks use goes to the agent with the most credit among "
                "those whose blocks use it", live=_rights_by_credit),
    Setting("messages", "information", "bool", True, doc="whether the message action is offered",
            live=_kernel("messages_on", bool)),
    Setting("delivery", "information", "choice", "top", choices=("top", "all"), opt="board", encode={"top": "", "all": "1"},
            doc="what reaches an agent at the start of a turn: the most relevant new posts (top) or every new post "
                "once (all)", live=lambda k, env, v: setattr(k, "full_board", bool(k.sync and v == "all"))),
    Setting("delivered", "information", "int", 8, lo=4, hi=16, opt="sync_items", needs=(("delivery", ("top",)),),
            doc="how many new posts are delivered at the start of a turn while delivery is top",
            live=_kernel("push_items", int)),
    Setting("moving_facts", "information", "bool", False, opt="contest",
            doc="each design review also says which shared knobs keep moving and who moved them",
            live=_world("contest_facts", bool)),
    Setting("memory_ticks", "information", "int", 2, lo=0, hi=4, opt="continuity",
            doc="ticks of its own conversation each agent keeps", live=_kernel("continuity", int)),
    Setting("shared_pay", "incentives", "float", 0.0, lo=0, hi=1, opt="economy_share",
            doc="each agent's share of the fall in the whole design's cost, paid on top of what its own block earns",
            live=_world("economy_share")),
    Setting("lead_pay", "incentives", "float", 2.0, lo=0, hi=4, opt="project_mult",
            needs=(("take_lead", ("scope", "team")),),
            doc="what a lead earns: this many times the fall in the combined cost of what it leads",
            live=_world("project_mult")),
    Setting("group_pay", "incentives", "bool", False, opt="group_area",
            doc="every group of two or more is also credited with the fall in the summed cost of its members' blocks",
            live=_world("group_area", bool)),
    Setting("credit_every", "incentives", "int", 3, lo=1, hi=6, doc="ticks between credit reviews",
            live=_world("credit_every", int)),
    Setting("take_lead", "procedures", "choice", "scope", choices=("off", "scope", "team"), opt="leadership",
            encode={"off": "", "scope": "1", "team": "team"},
            doc="whether take_lead is offered: off; lead a set of blocks and hold the knobs inside it (scope); found a "
                "team and hire into it (team)"),
    Setting("contracts", "procedures", "bool", True, doc="whether contract is offered: a binding, paid deal over a knob"),
    Setting("review_every", "procedures", "int", 1, lo=1, hi=6, doc="ticks between design reviews, which post the score",
            live=_world("validate_every", int)),
)

# The code world's rules (alphateam/code_world.py: ProgramBench xsv rebuilt by a society). Defaults are the plain arm of
# tools/launch_xsv.sh with every action on offer: no credit, no review, work landing at the end of a tick. The budget
# (agents, ticks, replies, steps, probes, sandboxes), the model and both scores are the frame and are not settings.
# No setting changes mid-run here: a code-world society cannot be copied, so every comparison is between fresh runs.
CODE = (
    Setting("own", "rights", "bool", True, uses=("claims", "lab.claim"),
            doc="whether the own action is offered: an agent or a group may hold file patterns, and a change to a held "
                "file waits for its holder to accept it"),
    Setting("own_ticks", "rights", "int", 6, lo=2, hi=12, needs=(("own", (True,)),), uses=("claims", "lab.claim"),
            doc="idle ticks after which a claim on files lapses"),
    Setting("messages", "information", "bool", True, uses=("message",), doc="whether the message action is offered"),
    Setting("private", "information", "bool", True,
            doc="whether agents and groups have boards of their own beside the public board"),
    Setting("teaming", "information", "bool", True,
            doc="whether each turn also shows what working in a group gives its members"),
    Setting("show_open", "information", "bool", False,
            doc="whether each turn also lists the commands not at HEAD that no copy has changed, and those some copy has "
                "changed, with whose"),
    Setting("memory_ticks", "information", "int", 2, lo=0, hi=4, opt="continuity",
            doc="ticks of its own conversation each agent keeps"),
    Setting("credit", "incentives", "bool", False,
            doc="whether verified progress at HEAD is paid as credit; credit buys steps and goes first at the gate"),
    Setting("goal", "incentives", "choice", "individual", choices=("individual", "team", "society"),
            needs=(("credit", (True,)),),
            doc="whose wealth a payment is: the agent's own (individual), its group's, shared equally (team), or "
                "everyone's, shared equally (society)"),
    Setting("piece_share", "incentives", "float", 0.25, lo=0, hi=0.6, needs=(("credit", (True,)),),
            doc="the share of a merge's value paid to the pieces it used (cited, built on or read)"),
    Setting("stream", "procedures", "bool", False,
            doc="when offered work lands at HEAD: as soon as it is offered (on) or at the end of the tick (off)"),
    Setting("offer_order", "procedures", "choice", "unblocks", choices=("unblocks", "credit", "open"),
            needs=(("stream", (False,)),),
            doc="the order in which a tick's offers land: work that other code depends on first (unblocks), highest "
                "credit first (credit), or agent order (open)"),
    Setting("on_conflict", "procedures", "choice", "keep", choices=("keep", "take_head", "hold_back"),
            doc="after a file of a copy conflicts with HEAD: the copy keeps the agent's version, which goes to the gate again "
                "with each later offer (keep); the copy takes HEAD's version and the agent's is moved to /public/unmerged "
                "(take_head); or the copy keeps the agent's version, left out of its offers until the agent changes it (hold_back)"),
    Setting("publish_replays", "information", "bool", False,
            doc="whether a publish also returns the logged checks replayed on the publishing agent's copy as it is then"),
    Setting("offer", "procedures", "choice", "asked", choices=("asked", "named", "every_tick"),
            doc="what reaches the gate: a copy's changed files when its agent publishes (asked); only the files a publish "
                "names (named); or every copy's changed files at the end of every tick, with or without a publish (every_tick)"),
    Setting("new_file", "procedures", "choice", "open", choices=("open", "first_copy"),
            doc="a file HEAD does not have: any copy may write it (open); or it is written in one copy at a time, the first "
                "copy to create it holding it until it lands, leaves that copy or three more ticks have passed, a version "
                "written elsewhere meanwhile being moved to /public/unmerged (first_copy)"),
    Setting("instructions", "information", "bool", True,
            doc="whether the texts an agent is shown carry their standing instructions (the working cycle, what to do with "
                "ideas, whom to message about an overlap) beside their facts"),
    Setting("delegation", "procedures", "bool", True, uses=("delegate.",),
            doc="whether the delegate action is offered: a task handed to an agent or a group, with pay if any"),
    Setting("toolbox", "procedures", "bool", True, uses=("tools", "lab.toolbox"),
            doc="whether the toolbox action is offered: a script an agent shares with everyone"),
    Setting("review_every", "procedures", "int", 0, lo=0, hi=12, opt="reflect",
            doc="ticks between the society's reviews of itself, in which any rule may change (0: no reviews)"),
)

REGISTRY: dict[str, dict[str, Setting]] = {"design": {s.key: s for s in DESIGN}, "code": {s.key: s for s in CODE}}


def registry(world: str) -> dict[str, Setting]:
    if world not in REGISTRY:
        raise RecipeError(f"no rules are registered for the '{world}' world (registered: {', '.join(REGISTRY)})")
    return REGISTRY[world]


# ---------------------------------------------------------------------------------------------- lessons

def anonymize(text: str, world: str | None = None) -> str:
    """A recipe sets rules for whoever is there, on whatever instance: the ids of this run's agents and groups, and
    the names that belong to this one task instance, are taken out of its text."""
    text = NAMES.sub(lambda m: "an agent" if m.group()[0] in "Aa" else "a group", str(text or ""))
    text = RUN_IDS.sub(lambda m: {"E": "an entry", "P": "a proposal", "T": "a task"}[m.group()[0]], text)
    for pattern, word in LOCAL.get(world or "", ()):
        text = pattern.sub(word, text)
    return text.strip()


def named(text: str, world: str | None = None) -> bool:
    text = text or ""
    return bool(NAMES.search(text) or RUN_IDS.search(text)) or any(p.search(text) for p, _ in LOCAL.get(world or "", ()))


def lesson(text: str, when: str = "", evidence=(), status: str = "untested", source: str = "", world: str | None = None) -> dict:
    """What the next society should try, under what conditions, and what happened that suggests it."""
    text, when = clip(anonymize(text, world), MAX_LESSON), clip(anonymize(when, world), MAX_LESSON // 2)
    key = hashlib.sha1(f"{text}|{when}".encode()).hexdigest()[:8]
    return {"id": f"L{key}", "text": text, "when": when, "evidence": [str(x) for x in evidence][:12],
            "status": status if status in STATUSES else "untested", "source": str(source or "")}


def _action(a: dict, world: str | None = None) -> dict:
    name = str(a.get("name", "")).strip()
    if not ACTION_NAME.match(name):
        raise RecipeError("a designed action needs a name of 3-31 lower-case letters, digits or _")
    procedure = anonymize(a.get("procedure", ""), world)
    if len(procedure) < 20:
        raise RecipeError("a designed action needs a procedure of at least 20 characters")
    return {"name": name, "description": clip(anonymize(a.get("description", ""), world), 300), "procedure": procedure[:1500],
            "params": [str(p) for p in (a.get("params") or [])][:6], "author": "heritage", "version": int(a.get("version", 1) or 1)}


# ---------------------------------------------------------------------------------------------- recipes

@dataclass
class Recipe:
    world: str
    settings: dict = field(default_factory=dict)
    advice: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    parent: str | None = None
    generation: int = 0
    origin: dict = field(default_factory=dict)  # how it came about: by whom, which edits, why, from which run

    def __post_init__(self):
        reg = registry(self.world)
        full = {key: s.default for key, s in reg.items()}
        for key, v in (self.settings or {}).items():
            if key not in reg:
                raise RecipeError(f"no setting '{key}' in the {self.world} world (settings: {', '.join(reg)})")
            full[key] = reg[key].coerce(v)
        self.settings = full

    @property
    def id(self) -> str:
        """A name for the content: two recipes with the same rules, lessons and actions are the same recipe."""
        reg = registry(self.world)
        body = {"world": self.world, "settings": {key: v for key, v in self.settings.items() if v != reg[key].default},
                "advice": sorted((x["text"], x.get("when", "")) for x in self.advice),
                "actions": sorted((x["name"], x["procedure"], tuple(x.get("params", []))) for x in self.actions)}
        return "R" + hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]

    def part(self, name: str) -> dict:
        reg = registry(self.world)
        return {key: v for key, v in self.settings.items() if reg[key].part == name}

    def to_json(self) -> dict:
        return {"id": self.id, "world": self.world, "settings": dict(self.settings), "advice": copy.deepcopy(self.advice),
                "actions": copy.deepcopy(self.actions), "parent": self.parent, "generation": self.generation,
                "origin": copy.deepcopy(self.origin)}

    @classmethod
    def from_json(cls, d: dict) -> "Recipe":
        return cls(d["world"], dict(d.get("settings") or {}), list(d.get("advice") or []), list(d.get("actions") or []),
                   d.get("parent"), int(d.get("generation", 0)), dict(d.get("origin") or {}))


def seed(world: str = "design", **settings) -> Recipe:
    """A starting recipe: the registry's defaults (the free arm), with any settings given."""
    return Recipe(world, settings, origin={"by": "seed"})


def validate(recipe: Recipe, locked=()) -> Recipe:
    """Raise RecipeError unless the recipe is one the system can enforce and carries no names."""
    reg = registry(recipe.world)
    for key, v in recipe.settings.items():
        reg[key].coerce(v)
    if len(recipe.advice) > MAX_ADVICE or len(heritage(recipe)) > ADVICE_CHARS:
        raise RecipeError(f"a recipe carries at most {MAX_ADVICE} lessons and {ADVICE_CHARS} characters of advice; retire a "
                          f"lesson first (unadvise)")
    for x in recipe.advice:
        for part in (x.get("text", ""), x.get("when", "")):
            if named(part, recipe.world):
                raise RecipeError(f"a lesson may not name an agent, a group or a part of one task instance: {clip(part, 80)}")
        if x.get("status", "untested") not in STATUSES:
            raise RecipeError(f"a lesson's status is one of {', '.join(STATUSES)}")
    seen = set()
    for a in recipe.actions:
        if a["name"] in seen:
            raise RecipeError(f"two designed actions are named {a['name']}")
        seen.add(a["name"])
        if named(a.get("procedure", ""), recipe.world) or named(a.get("description", ""), recipe.world):
            raise RecipeError(f"a designed action may not name an agent, a group or a part of one task instance: {a['name']}")
    for key in locked:
        if key not in reg:
            raise RecipeError(f"the frame locks '{key}', which is not a setting")
    return recipe


def apply_edits(recipe: Recipe, edits: list[dict], origin: dict | None = None, locked=()) -> Recipe:
    """A revision: the child of a recipe after a list of typed edits. Refuses what the registry does not allow."""
    reg = registry(recipe.world)
    settings, advice, actions = dict(recipe.settings), copy.deepcopy(recipe.advice), copy.deepcopy(recipe.actions)
    done = []
    for e in edits or []:
        op = str((e or {}).get("op", "")).strip()
        if op == "set":
            key = str(e.get("key", "")).strip()
            if key not in reg:
                raise RecipeError(f"no setting '{key}' (settings: {', '.join(reg)})")
            if key in locked:
                raise RecipeError(f"'{key}' is fixed in this frame")
            v = reg[key].coerce(e.get("value"))
            if v == settings[key]:
                raise RecipeError(f"{key} is already {show(v)}")
            done.append({"op": "set", "key": key, "value": v, "from": settings[key], "part": reg[key].part})
            settings[key] = v
        elif op == "advise":
            les = lesson(e.get("text", ""), e.get("when", ""), e.get("evidence") or (), e.get("status", "untested"),
                         e.get("source", ""), recipe.world)
            if len(les["text"]) < 20:
                raise RecipeError("a lesson needs at least 20 characters: what to try, when, and what suggests it")
            if any(x["id"] == les["id"] for x in advice):
                raise RecipeError("that lesson is already in the recipe")
            advice.append(les)
            done.append({"op": "advise", "id": les["id"], "part": "advice"})
        elif op == "unadvise":
            lid = str(e.get("id", "")).strip()
            if not any(x["id"] == lid for x in advice):
                raise RecipeError(f"no lesson {lid} in the recipe")
            advice = [x for x in advice if x["id"] != lid]
            done.append({"op": "unadvise", "id": lid, "part": "advice"})
        elif op == "adopt":
            act = _action(e.get("action") or {}, recipe.world)
            if any(a["name"] == act["name"] for a in actions):
                raise RecipeError(f"the recipe already carries an action named {act['name']}")
            actions.append(act)
            done.append({"op": "adopt", "name": act["name"], "part": "procedures"})
        elif op == "unadopt":
            name = str(e.get("name", "")).strip()
            if not any(a["name"] == name for a in actions):
                raise RecipeError(f"the recipe carries no action named {name}")
            actions = [a for a in actions if a["name"] != name]
            done.append({"op": "unadopt", "name": name, "part": "procedures"})
        else:
            raise RecipeError(f"unknown edit '{op}' (edits: {', '.join(EDITS)})")
    if not done:
        raise RecipeError("a revision needs at least one edit")
    return validate(Recipe(recipe.world, settings, advice, actions, parent=recipe.id, generation=recipe.generation + 1,
                           origin=dict(origin or {}, edits=done)), locked)


def changed(parent: Recipe, child: Recipe) -> list[str]:
    return [key for key in child.settings if child.settings[key] != parent.settings.get(key)]


def enforced(child: Recipe) -> bool:
    """True when the revision changes something the system enforces, not only what agents are told."""
    return any(e.get("part") in ENFORCED for e in child.origin.get("edits", []))


def inert(parent: Recipe, child: Recipe) -> str | None:
    """Why a revision cannot change anything the system does, or None. Free to check, before any run."""
    reg = registry(child.world)
    keys = changed(parent, child)
    same_actions = sorted(a["name"] + a["procedure"] for a in parent.actions) == sorted(a["name"] + a["procedure"] for a in child.actions)
    same_advice = sorted(x["id"] for x in parent.advice) == sorted(x["id"] for x in child.advice)
    if not keys and same_actions and same_advice:
        return "it changes nothing"
    if not keys or not same_actions or not same_advice:
        return None
    idle = []
    for key in keys:
        unmet = [f"{other} is {show(child.settings[other])}" for other, values in reg[key].needs
                 if child.settings[other] not in values]
        if not unmet:
            return None
        idle.append(f"{key} has no effect while {' and '.join(unmet)}")
    return "; ".join(idle)


def unused(parent: Recipe, child: Recipe, run) -> str | None:
    """Why a revision could not show in a society like the parent's last one, or None. It could not if all it does is
    take away, or tune, something no agent used in that run: a rule nobody invoked cannot have mattered. `run` is a
    full run under the parent (its facts and its counts of actions). Offering something new always may matter, and so
    does anything agents did use; a setting that names no counts is never judged this way."""
    counts, ops = getattr(run, "facts", None) or {}, getattr(run, "ops", None) or {}
    keys = changed(parent, child)
    if not ops or not keys:
        return None
    reg, idle = registry(child.world), []
    for key in keys:
        s = reg[key]
        if not s.uses or (s.kind == "bool" and child.settings[key] is True):
            return None
        n = sum(v for name, v in counts.items() if name in s.uses and isinstance(v, (int, float)) and not isinstance(v, bool))
        n += sum(v for op, v in ops.items() if any(op.startswith(u) for u in s.uses))
        if n:
            return None
        idle.append(f"{key} was not used by anyone in the parent's run")
    return "; ".join(idle)


# ---------------------------------------------------------------------------------------------- into a run

def options(recipe: Recipe, locked: dict | None = None) -> dict[str, str]:
    """The world options a recipe compiles to. A frame's locked settings win over the recipe's."""
    reg = registry(recipe.world)
    settings = dict(recipe.settings)
    for key, v in (locked or {}).items():
        settings[key] = reg[key].coerce(v)
    return {(s.opt or key): s.option(settings[key]) for key, s in reg.items()}


def to_cfg(recipe: Recipe, frame: dict) -> dict:
    """A run configuration: the frame (world, budget, fixed options) with the recipe's rules compiled into it."""
    if frame.get("world") != recipe.world:
        raise RecipeError(f"a {recipe.world} recipe cannot run in a {frame.get('world')} frame")
    cfg = copy.deepcopy(frame)
    opts = dict(cfg.get("world_opts") or {})
    opts.update(options(recipe, cfg.get("locked")))
    opts.update(cfg.get("force_opts") or {})  # what the frame's arm needs, whatever the recipe says
    cfg["world_opts"] = opts
    cfg["recipe"] = recipe.id
    return cfg


def heritage(recipe: Recipe) -> str:
    """The recipe's lessons as every agent is shown them: all of them, confirmed first. validate() keeps the whole
    within a size budget, so nothing a recipe carries goes unseen."""
    first = {"confirmed": 0, "untested": 1, "refuted": 2}
    order = sorted(recipe.advice, key=lambda x: first.get(x.get("status", "untested"), 1))
    if not order:
        return ""
    return ("LESSONS FROM EARLIER SOCIETIES (their words, left after their runs; confirmed means later societies did "
            "better with the lesson than without it, untested that nobody has checked, refuted that they did not):\n"
            + "\n".join(f"- [{x.get('status', 'untested')}] {x['text']}" + (f" (when: {x['when']})" if x.get("when") else "")
                        for x in order))


def attach(k, recipe: Recipe, locked: dict | None = None, proposals: bool = True):
    """Start a society under a recipe: its lessons, the actions it carries, and (optionally) the `rules` action."""
    k.recipe = recipe
    k.heritage = heritage(recipe)
    if k.env is not None:
        k.env.told_rules = True  # the world's brief then states every rule in force (design_world._rules_brief)
    for a in recipe.actions:
        k.custom.setdefault(a["name"], CustomAction(a["name"], a["description"], a["procedure"], list(a["params"]),
                                                    "heritage", 0, a.get("version", 1)))
    if proposals:
        k.rulebook = Rulebook(recipe, locked)
        k.allowed["rules"] = None
        last = getattr(k.env, "final_tools", None)  # a world that names what a tick's last reply may do: rules is among it
        if last is not None and "rules" not in last:
            k.env.final_tools = tuple(last) + ("rules",)
    return k


def live_ok(old: Recipe, new: Recipe) -> bool:
    """Can a running society switch from old to new? Only if every changed setting can change mid-run."""
    reg = registry(new.world)
    return all(reg[key].live is not None for key in changed(old, new))


def apply_live(k, old: Recipe, new: Recipe):
    """Change the rules of a running society (a branch test). Refuses a setting that only applies from the start."""
    reg = registry(new.world)
    keys = changed(old, new)
    for key in keys:
        if reg[key].live is None:
            raise RecipeError(f"'{key}' can only change at the start of a run")
    for key in keys:
        reg[key].live(k, k.env, new.settings[key])
    for a in new.actions:
        k.custom.setdefault(a["name"], CustomAction(a["name"], a["description"], a["procedure"], list(a["params"]),
                                                    "heritage", k.tick, a.get("version", 1)))
    k.recipe = new
    k.heritage = heritage(new)
    if getattr(k, "rulebook", None) is not None:
        k.rulebook.recipe = new
    return keys


def menu_edits(recipe: Recipe, rng: random.Random, locked=(), enforced_only: bool = True) -> list[dict]:
    """One random typed edit from the menu: a setting moved to a neighbouring value. For scripted and control runs."""
    reg = registry(recipe.world)
    keys = [key for key in reg if key not in locked]
    rng.shuffle(keys)
    for key in keys:
        s, now = reg[key], recipe.settings[key]
        if s.kind == "bool":
            return [{"op": "set", "key": key, "value": not now}]
        if s.kind == "choice":
            return [{"op": "set", "key": key, "value": rng.choice([c for c in s.choices if c != now])}]
        span = s.hi - s.lo
        step = max(1, round(span / 4)) if s.kind == "int" else span / 4
        value = min(s.hi, max(s.lo, now + rng.choice((-1, 1)) * step))
        if s.kind == "float":
            value = round(value, 4)
        if value != now:
            return [{"op": "set", "key": key, "value": value}]
    return []


# ---------------------------------------------------------------------------------------------- the rules action

RULES_TOOL = (
    "The rules this society runs under (who holds which decisions, what is shared with whom, how credit is paid, which "
    "actions are offered) are settings, and later societies can start with other values. view: the settings, their "
    "values now and the values allowed, and the proposals made so far. propose: one setting (key) and a value for later "
    "societies, with why and the ids of the entries that are your evidence; separate societies then run with and "
    "without the change, and it is kept only if those with it do better. back: add your support to a proposal (id). "
    "lesson: leave a lesson for later societies (text: what to try; when: under what conditions; evidence: ids of what "
    "happened here that suggests it).")


# What the rules action also offers after the last tick (alphateam/llm.py closing). By then the shared work is final, so
# looking back cannot change a run's score. During a run the action and its description are exactly as above.
CLOSING_TOOL = (
    RULES_TOOL + " After the last tick two more ops are open. record: what happened in this run, as the system kept "
    "it. what=timeline: the run a tick a line (what landed, what was turned away and why, holds, tasks). what=tick "
    "with tick=N: every action of that tick with what came back. what=agent with agent=A3: what that agent offered and "
    "what became of it. what=search with text=...: actions whose arguments or results contain the text. suggest: a "
    "rule that is not a setting yet: what the system should enforce or show differently (text), when it would have "
    "mattered here (when), and your evidence (ids of entries or receipts, or ticks written t6). The people who "
    "maintain the rules read every suggestion and build those they can as settings; a built suggestion is then tried "
    "like any proposal. back also takes a suggestion's id.")
CLOSING_OPS = ("view", "record", "propose", "suggest", "back", "lesson")


def _cited(k, x: str) -> bool:
    """Is x something an agent can cite as evidence here: an entry on a board, a receipt of the world (the code
    world's R-ids, which its own actions tell agents to cite), or a tick of this run written t6 (what an agent cites
    after looking at the run's record: in the first live closing round four lessons citing a tick were refused)."""
    tick = x[:1].lower() == "t" and x[1:].isdigit() and 1 <= int(x[1:]) <= k.tick
    return x in k.index or x.upper() in (getattr(k.env, "receipts", None) or ()) or tick


class Rulebook:
    """What the kernel's optional `rules` action talks to. It keeps this run's proposals and lessons."""

    def __init__(self, recipe: Recipe, locked: dict | None = None):
        self.recipe = recipe
        self.locked = dict(locked or {})
        self.proposals: dict[str, dict] = {}
        self.lessons: list[dict] = []
        self.suggestions: dict[str, dict] = {}  # rules that are not settings yet, in agents' words (after the last tick)
        self.trials: list[dict] = []  # branch tests run inside this society (alphateam/trial.py)
        self.note = ""  # when proposals can be tried inside this run: how (set by alphateam/trial.py, shown to agents)

    def now(self, key: str):
        """The value in force: the frame's, where the frame fixes the setting; else the recipe's."""
        reg = registry(self.recipe.world)
        return reg[key].coerce(self.locked[key]) if key in self.locked else self.recipe.settings[key]

    def view(self) -> str:
        reg = registry(self.recipe.world)
        lines = ["THE RULES THIS SOCIETY RUNS UNDER (each is a setting; later societies can start with other values):"]
        if self.note:
            lines.append(self.note)
        for part in ENFORCED:
            lines.append(f"{part}:")
            for key, s in reg.items():
                if s.part == part:
                    fixed = " (fixed in this run's frame)" if key in self.locked else f" [{s.allowed()}]"
                    lines.append(f"- {key} = {show(self.now(key))}{fixed}: {s.doc}")
        if self.proposals:
            lines.append("PROPOSALS SO FAR:")
            lines += [f"- {p['id']} by {p['by']} at t{p['t']}: {p['key']} = {show(p['value'])}, backed by "
                      f"{len(p['backers'])}: {clip(p['why'], 160)}" for p in self.ranked()]
        if self.suggestions:
            lines.append("RULES SUGGESTED SO FAR (not settings yet):")
            lines += [f"- {x['id']} by {x['by']}, backed by {len(x['backers'])}: {clip(x['text'], 200)}"
                      for x in sorted(self.suggestions.values(), key=lambda x: (-len(x["backers"]), x["id"]))]
        if self.trials:
            lines.append("TRIED IN THIS SOCIETY:")
            lines += [f"- {t['text']}" for t in self.trials]
        return "\n".join(lines)

    def ranked(self) -> list[dict]:
        """Proposals, most backed first; evidence breaks ties, then the earlier one."""
        return sorted(self.proposals.values(), key=lambda p: (-len(p["backers"]), -len(p["evidence"]), p["t"], p["id"]))

    def act(self, k, aid: str, a: dict, via=None):
        op = str(a.get("op") or "view").strip()
        if op == "view":
            text = self.view()
            k._need(k.spend_attention(aid, text), "not enough reading budget left this tick")
            return "rules.view", text, None
        if op == "propose":
            reg = registry(self.recipe.world)
            key = str(a.get("key", "")).strip()
            k._need(key in reg, f"no setting '{key}' (rules op=view lists them)")
            k._need(key not in self.locked, f"'{key}' is fixed in this run's frame")
            try:
                value = reg[key].coerce(a.get("value"))
            except RecipeError as exc:
                raise ActionError(str(exc)) from None
            k._need(value != self.now(key), f"{key} is already {show(value)}")
            why = str(a.get("why", "")).strip()
            k._need(len(why) >= 20, "say why in at least 20 characters: what happened here that suggests it")
            evidence = [str(x).strip() for x in (a.get("evidence") or []) if str(x).strip()][:12]
            missing = [x for x in evidence if not _cited(k, x)]
            k._need(not missing, f"no entry, receipt or tick {', '.join(clip(x, 40) for x in missing)}: evidence is a list of ids")
            same = next((p for p in self.proposals.values() if p["key"] == key and p["value"] == value), None)
            if same:
                same["backers"].add(aid)
                return "rules.back", f"{same['id']} already proposes {key} = {show(value)}; your support was added", same["id"]
            pid = k.new_id("P")
            e = k.post("global", aid, "proposal", f"Rule proposal {pid}: {key} = {show(value)} (now "
                       f"{show(self.now(key))}) for later societies. Why: {why}", refs=evidence, via=via)
            self.proposals[pid] = {"id": pid, "by": aid, "t": k.tick, "key": key, "value": value, "why": clip(why, 600),
                                   "evidence": evidence, "backers": {aid}, "entry": e.id}
            k.charge(aid, "critique", est_tokens(why))
            return "rules.propose", f"{pid} proposed: {key} = {show(value)} (posted as {e.id})", pid
        if op == "back":
            pid = str(a.get("id", "")).strip().upper()
            book = self.proposals if pid in self.proposals else self.suggestions
            k._need(pid in book, f"no proposal or suggestion {pid} (rules op=view lists them)")
            book[pid]["backers"].add(aid)
            return "rules.back", f"you back {pid}; it now has {len(book[pid]['backers'])} backers", pid
        if op in ("record", "suggest"):
            k._need(getattr(k, "closing_now", False), f"{op} opens after the last tick of a run")
        if op == "record":
            from . import record as rec  # no society runs under that module: it only reads what a run kept
            what = str(a.get("what") or "timeline").strip()
            if what == "timeline":
                text = rec.live_timeline(k)
            elif what == "tick":
                k._need(str(a.get("tick", "")).strip().lstrip("t").isdigit(), "what=tick takes tick=N")
                t = int(str(a.get("tick")).strip().lstrip("t"))
                lines = rec.event_lines(k.events, round_=t, chars=260)
                text = "\n".join(lines[:80]) + (f"\n... {len(lines) - 80} more" if len(lines) > 80 else "") if lines \
                    else f"nothing happened in tick {t}"
            elif what == "agent":
                who = str(a.get("agent", "")).strip().upper()
                k._need(who in k.agents, "what=agent takes agent=A3")
                text = rec.agent_text(k.env.export(), k.events, who)
            elif what == "search":
                q = str(a.get("text", "")).strip()
                k._need(len(q) >= 2, "what=search takes text=...")
                lines = rec.event_lines(k.events, grep=q, chars=260)
                text = "\n".join(lines[:40]) + (f"\n... {len(lines) - 40} more" if len(lines) > 40 else "") if lines \
                    else f"no action mentions '{clip(q, 60)}'"
            else:
                raise ActionError("what is one of timeline, tick, agent, search")
            text = text if len(text) <= 9000 else text[:9000] + "\n..."  # as written: a line a tick (clip would join the lines)
            k._need(k.spend_attention(aid, text), "not enough reading budget left")
            return "rules.record", text, None
        if op == "suggest":
            text = str(a.get("text", "")).strip()
            k._need(len(text) >= 30, "a suggestion needs at least 30 characters: what the system should enforce or show differently")
            evidence = [str(x).strip() for x in (a.get("evidence") or []) if str(x).strip()][:12]
            missing = [x for x in evidence if not _cited(k, x)]
            k._need(not missing, f"no entry, receipt or tick {', '.join(clip(x, 40) for x in missing)}: evidence is a list of ids")
            k._need(not any(x["text"] == clip(text, 900) for x in self.suggestions.values()), "that was already suggested")
            sid = k.new_id("S")
            when = clip(str(a.get("when", "")).strip(), 300)
            e = k.post("global", aid, "proposal", f"Rule suggestion {sid} for later societies: {clip(text, 600)}"
                       + (f" When: {when}" if when else ""), refs=evidence, via=via)
            self.suggestions[sid] = {"id": sid, "by": aid, "t": k.tick, "text": clip(text, 900), "when": when,
                                     "evidence": evidence, "backers": {aid}, "entry": e.id}
            k.charge(aid, "critique", est_tokens(text))
            return "rules.suggest", f"{sid} suggested (posted as {e.id}): it will be read by the people who maintain the rules", sid
        if op == "lesson":
            text = str(a.get("text", "")).strip()
            k._need(len(text) >= 20, "a lesson needs at least 20 characters: what to try, and what happened that suggests it")
            evidence = [str(x).strip() for x in (a.get("evidence") or []) if str(x).strip()][:12]
            missing = [x for x in evidence if not _cited(k, x)]
            k._need(not missing, f"no entry, receipt or tick {', '.join(clip(x, 40) for x in missing)}: evidence is a list of ids")
            les = lesson(text, a.get("when", ""), evidence, world=self.recipe.world)
            k._need(not any(x["id"] == les["id"] for x in self.lessons), "that lesson was already left")
            e = k.post("global", aid, "lesson", f"Lesson for later societies: {les['text']}"
                       + (f" When: {les['when']}" if les["when"] else ""), refs=evidence, via=via)
            self.lessons.append(dict(les, by=aid, t=k.tick, entry=e.id))
            k.charge(aid, "critique", est_tokens(text))
            return "rules.lesson", f"lesson left for later societies (posted as {e.id})", e.id
        raise ActionError("op is one of view, propose, back, lesson" + (", record, suggest" if getattr(k, "closing_now", False) else ""))

    def export(self) -> dict:
        return {"recipe": self.recipe.id,
                "proposals": [dict(p, backers=sorted(p["backers"], key=lambda x: int(x[1:]))) for p in self.ranked()],
                "lessons": copy.deepcopy(self.lessons), "trials": copy.deepcopy(self.trials),
                "suggestions": [dict(x, backers=sorted(x["backers"], key=lambda b: int(b[1:])))
                                for x in sorted(self.suggestions.values(), key=lambda x: (-len(x["backers"]), x["id"]))]}
