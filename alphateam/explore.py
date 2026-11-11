"""Exploration, a substrate of work, plans and evidence, and delegation, for the ARC world (world option explore=1).

The project's goal is a society that finds which of its actions, and which combinations of them, are useful, with
credit assignment as the means. This module offers actions and computes credit; nothing in it is required.

Substrate (after Queen v3): plans, findings and engine evidence are entries on the shared boards.
- Every play gets an engine receipt (R-id): the copy, the level it started on, the moves and what changed. Receipts are
  engine-owned, so a finding that cites them is shown as backed by engine records; one that cites none is a claim.
- plan op=claim posts an intention on a problem (a game's next unsolved level) with a hypothesis. It is shown to everyone
  working on that problem until it expires, so others can test something else. plan op=done closes it with a finding
  and the receipts that show it.
- The open problems (each game's frontier level), how many copies are there and their open plans are shown every turn.
- The kernel's retrieval log (entries an agent read in full) gives the edges along which reuse credit flows.

Exploration (after Evolving Agent Teams): agents choose problems (focus switches game) and credit pays for exploring
where nobody has been: the first copy to complete a level, frame states at a frontier no copy has seen (cells that change
on almost every step, such as timers, are masked), findings backed by receipts, and posts the first solver read. Credit
earned in the last two ticks buys extra game steps.

Delegation (world option delegation=1): a lead, like a CEO or PI, founds a team and keeps playing; members join; a
member who founds a team creates a sub-team, so teams nest. A lead assigns each direct child a role and a job, shown at
the top of the child's every turn; children report up; a tracker the lead names keeps a note pinned for the whole team.
A lead earns a share of the credit its children earn.

Action ledger: each credit award is attributed to the actions its earner used that tick and the tick before; the society
sees which actions and combinations have paid, and which actions nobody has used yet.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from .game_model import GameModels, load_inherited
from .model_sandbox import check
from .directions import Directions
from .kernel import ActionError, as_list, clip

PLAN_TOOL = {
    "name": "plan",
    "description": "Plans and findings on the shared substrate. claim: announce what you will test on a problem (default: "
                   "your game's next unsolved level): a hypothesis and the result that would show it (criteria); everyone on "
                   "that problem sees it while it is open, and those with open plans there are told. finding: post an "
                   "intermediate finding with its receipts, keeping the plan open. done: close your plan with what changed, "
                   "the idea, the outcome and the receipts (R-ids from play) that show it. drop: abandon it. challenge: contest "
                   "someone's finding with a counterexample (finding + receipts); it is shown as contested until its author "
                   "posts a revision (done or finding with entry=<the contested finding>).",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["claim", "finding", "done", "drop", "challenge"]},
        "entry": {"type": "string", "description": "challenge: the finding (E-id) a counterexample contradicts; done or "
                                                   "finding: the finding of yours this one revises"},
        "hypothesis": {"type": "string", "description": "claim: what you will test, concretely"},
        "criteria": {"type": "string", "description": "claim: the result that would show it worked"},
        "game": {"type": "string", "description": "claim: the game (default yours)"},
        "ttl": {"type": "integer", "description": "claim: ticks the plan stays open (1-5, default 3)"},
        "finding": {"type": "string", "description": "finding or done: what you found"},
        "outcome": {"type": "string", "enum": ["works", "dead_end", "unclear"], "description": "done: did it work"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "done: receipt ids, e.g. [\"R12\"]"}},
        "required": ["op"]}}
FOCUS_TOOL = {
    "name": "focus",
    "description": "Switch to another game: you play your own copy of it (new copies start at level 1; replay published "
                   "solutions to reach the frontier). Your old copy is kept if you come back.",
    "parameters": {"type": "object", "properties": {"game": {"type": "string"}}, "required": ["game"]}}
LEAD_TOOL = {
    "name": "lead",
    "description": "Lead a team like a CEO or PI and keep playing yourself. found: found a team (if you are in a team, it is "
                   "a sub-team). join/leave: join an open team or leave yours. assign: give a direct child (a member, or "
                   "the lead of a sub-team) a role and a job, shown at the top of its every turn. report: tell your lead "
                   "what you did and found. tracker: name who keeps your team's pinned note. pin: (tracker or lead) set "
                   "the note every team member sees each turn. step_down: hand the team to a member (to) or dissolve it.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["found", "join", "leave", "assign", "report", "tracker", "pin", "step_down"]},
        "team": {"type": "string", "description": "join: the team's group id"},
        "to": {"type": "string", "description": "assign/tracker/step_down: the agent"},
        "role": {"type": "string"}, "job": {"type": "string"},
        "text": {"type": "string", "description": "report or pin: the text"},
        "name": {"type": "string"}},
        "required": ["op"]}}

SUBSTRATE_TOOL = {
    "name": "substrate",
    "description": "Shape the shared substrate. define_kind: add a kind of entry anyone can then write (write kind=<name>), "
                   "with its purpose and, optionally, its fields: an interface every entry of that kind must fill (lines "
                   "'field: value'), so independent findings fit together. adopt: declare you will use a kind; once enough "
                   "agents adopt it, it is agreed and shown every turn. revise_kind: change an adopted kind's fields or "
                   "purpose. propose: suggest a change to the credit rules or to what the substrate shows each turn; the "
                   "rules are revised from the proposals and from what has paid, judged only by the game score.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["define_kind", "adopt", "revise_kind", "propose"]},
        "fields": {"type": "array", "items": {"type": "string"}, "description": "define_kind/revise_kind: field names"},
        "name": {"type": "string", "description": "the kind's short name, letters and underscores"},
        "purpose": {"type": "string", "description": "define_kind: what entries of this kind are for"},
        "text": {"type": "string", "description": "propose: the change and why"}},
        "required": ["op"]}}
CONTRACT_TOOL = {
    "name": "contract",
    "description": "Pay another agent credit for a job. offer: to, job, pay. accept or decline: the offer's id (on accept the "
                   "pay is taken from the offerer and held). done: the offerer releases the pay to the taker. cancel: the "
                   "offerer withdraws an offer, or takes back the held pay of an accepted job that is not done.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["offer", "accept", "decline", "done", "cancel"]},
        "to": {"type": "string"}, "job": {"type": "string"}, "pay": {"type": "number"},
        "id": {"type": "string", "description": "the contract id, e.g. C3"}},
        "required": ["op"]}}

# carry=1: the plan tool can also explain a published solution, and a finding can name the findings it builds on
CARRY_PLAN_TOOL = {**PLAN_TOOL, "description": PLAN_TOOL["description"] + (
    " explain: attach an explanation (finding + receipts) to a published solution (entry=<its E-id>), so those who "
    "replay it learn why it works. A finding or done can name the findings it builds on (builds_on); if one of them is "
    "contested, yours is flagged."), "parameters": {**PLAN_TOOL["parameters"], "properties": {
        **PLAN_TOOL["parameters"]["properties"],
        "op": {"type": "string", "enum": ["claim", "finding", "done", "drop", "challenge", "explain"]},
        "builds_on": {"type": "array", "items": {"type": "string"},
                      "description": "finding or done: E-ids of findings this one builds on"}}}}
MODEL_TOOL = {
    "name": "model",
    "description": "The society's shared model of a game, one piece of code everyone builds together: step(grid, action, x, "
                   "y) -> (new_grid, event) predicts what one move does (grid: 64x64 numpy int array of colors 0-15, "
                   "indexed grid[row, column]; action 1-7, with x (column) and y (row) for ACTION6 clicks, else None; "
                   "event '' or 'level_up' or 'game_over'). Every move any copy makes is recorded as a transition, and each "
                   "version is checked against all of them. view: the current version, what it explains, its code and "
                   "transitions it gets wrong. case: one transition (id T12) with the frame before and after around the "
                   "change. test: check your code without merging. propose: merge it; accepted only if written on the "
                   "current version (base), it still explains every transition the current version explains, and it "
                   "explains at least one more; if refused because the model changed, view the new version, take in its "
                   "changes and propose again. Credit is paid for moves recorded after your merge that your version "
                   "predicts and the one before it did not. Code: numpy as np (common functions), no imports or files.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["view", "case", "test", "propose"]},
        "game": {"type": "string", "description": "default your game"},
        "id": {"type": "string", "description": "case: a transition id, e.g. T12"},
        "base": {"type": "integer", "description": "propose: the version your code was written on"},
        "code": {"type": "string", "description": "test or propose: the full source, defining step(grid, action, x, y)"},
        "why": {"type": "string", "description": "propose: what your change adds"}},
        "required": ["op"]}}

# a guide line is a piece of the shared picture: long enough for one mechanic with its receipts (in the F runs 31 of 58
# attempted lines were refused at 240 characters and 23 were never sent again). A piece is never refused for space: in the
# F2 runs two guides were full of early-level lines by t7 and t12, 14 frontier lines were refused (a vc33 win hypothesis
# among them), and none was edited in, since compressing others' lines counted as removing them. Past GUIDE_MAX the
# oldest lines are shown shortened (view shows them whole), and merge combines lines without claiming any was wrong.
GUIDE_LINE, GUIDE_MAX, GUIDE_STORE, GUIDE_SHORT = 600, 4000, 16000, 120
GUIDE_TOOL = {
    "name": "guide",
    "description": "The society's guide to a game: one durable, revisable text of what is understood about it (what the "
                   "objects and actions do, what completes a level, what was ruled out), kept across levels and shown every "
                   "turn to everyone on that game, its lines numbered. view: the full guide, its versions and who wrote each "
                   "line. add: append one line (a fact, a rule, a dead end) with the receipts that show it (evidence); "
                   "appends are combined with everyone else's, so no base is needed. merge: combine two or more lines (lines: "
                   "their numbers) into one shorter line that keeps what they say; it takes the place of the first, their "
                   "authors become its co-authors and are told, and the old lines stay in the guide's history. challenge: "
                   "flag someone's line with a counterexample (line, evidence, why); it is shown as contested until its "
                   "author revises it. edit: replace it with your revision, written on the current version (base); if it "
                   "changed underneath, take in the new version and edit again. Your lines are yours to change; removing a "
                   "line someone else wrote needs receipts (evidence) showing it is wrong. Nothing is lost for space: past "
                   f"{GUIDE_MAX} characters the oldest lines are shown shortened until they are merged.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["view", "add", "merge", "challenge", "edit"]},
        "game": {"type": "string", "description": "default your game"},
        "line": {"type": "string", "description": f"add: one line, at most {GUIDE_LINE} characters; merge: the combined line; "
                                                  "challenge: the line (its text, or its number)"},
        "lines": {"type": "array", "items": {"type": "integer"}, "description": "merge: the numbers of the lines to combine"},
        "base": {"type": "integer", "description": "edit: the version you revised"},
        "text": {"type": "string", "description": f"edit: the whole new guide (at most {GUIDE_STORE} characters)"},
        "why": {"type": "string", "description": "edit: what your revision changes"},
        "evidence": {"type": "array", "items": {"type": "string"},
                     "description": "receipts (R-ids): behind an added line, a counterexample, or removals in an edit"}},
        "required": ["op"]}}

# The rules a society can evolve (credit weights and what the substrate shows), with their bounds. What is never evolved:
# the world's score (levels completed out of levels to win) and the facts credit is computed from.
RULES = {"credit": {"priority": 5.0, "novel": 0.2, "novel_cap": 2.0, "finding": 0.3, "dead_end": 0.3, "reuse": 1.0,
                    "lead_share": 0.3, "action_share": 0.2, "steps_per_credit": 2.0, "model": 0.3, "model_cap": 3.0,
                    "idea": 0.5, "understanding": 2.0, "test": 0.3, "direction": 3.0},
         "substrate": {"plan_ttl": 3, "items": 4, "agree": 2,
                       "show": ["problems", "plans", "findings", "dead_ends", "untried", "interfaces", "runs", "paid",
                                "unused"],
                       "rank": {"cites": 0.25, "credit": 0.0, "recency": 0.0}, "promoted_kinds": []}}
BOUNDS = {"priority": (0, 15), "novel": (0, 1), "novel_cap": (0, 6), "finding": (0, 2), "dead_end": (0, 2), "reuse": (0, 4),
          "lead_share": (0, 0.6), "action_share": (0, 0.6), "steps_per_credit": (0, 4), "model": (0, 2), "model_cap": (0, 6),
          "idea": (0, 2), "understanding": (0, 6), "test": (0, 2), "direction": (0, 10), "plan_ttl": (1, 8),
          "items": (1, 8),
          "agree": (1, 6),
          "cites": (0, 2), "credit": (0, 2), "recency": (0, 2)}
VIEWS = ("problems", "guide", "plans", "findings", "dead_ends", "carried", "model", "untried", "interfaces", "runs", "paid",
         "unused", "kinds", "cycle")
# A working cycle offered as a view (inspired by Agensh's worker loop, in AlphaTeam's terms): the rule loop can show or hide
# it, so whether it helps is judged by the score like any other rule.
CYCLE = ("A working cycle that may pay (yours to use, change or ignore): see what is known, dead and untried on your "
         "problem; claim what you will test and what result would show it; play; check your receipts against the claim; "
         "post the result with its receipts so others can build on it; when the frontier moves, start from it.")
KIND_NAME = __import__("re").compile(r"[a-z][a-z_]{1,23}")

# hypotheses=1: an idea is a plan's hypothesis. In the F2 runs the same few ideas were tested again and again (one after
# it had been refuted), and pieces found by different agents were rarely combined. Two ideas count as the same when their
# content words overlap (Jaccard) by SAME_IDEA or more; the engine cannot judge meaning, so a close match is pointed out,
# never refused, and only decides whether testing it earns credit for a new idea.
SAME_IDEA = 0.5
IDEA_WORD = re.compile(r"[a-z]{3,}|\d+")
IDEA_STOP = frozenset("the and that this with will from when then into each what which test tests testing level levels "
                      "game copy move moves click clicks action actions see shows show whether until after before does "
                      "not are for but all its can one two may might should would could than more also only there "
                      "their them they have has had been being because".split())
E_ID = re.compile(r"\bE\d+\b")
HYP_PLAN = (" Every claim on a level stays listed there under IDEAS with what came of it (testing, never tested, refuted, "
            "unclear, worked). A claim can name the pieces it combines or builds on (builds_on: E-ids of others' findings, "
            "explanations or claims); their authors are told.")
# trust=1: what a piece can be trusted at is computed by the engine from evidence, never voted. In the vc33 depth run one
# agent's run fired a gate on level 4; eight agents re-clicked the same cell from other states, saw nothing, and a guide
# line called it "settled" not a gate. A refutation counts only from the state the result came from, and every piece
# shows what it rests on, so a conclusion can be traced back to the assumptions under it.
R_ID = re.compile(r"\bR\d+\b")
TRUST_RANK = {"contested": 0, "said": 1, "seen": 2, "reproduced": 3, "engine": 3}
TRUST_PLAN = (" test: test a result before trusting it (entry=<E-id or R-id>). With no evidence, the engine runs the "
              "receipts behind it again, from the exact state each run started in, on a spare copy, and reports whether "
              "every step gives the recorded result. With result=holds or fails and evidence (receipts of your own runs), "
              "your run is recorded as a test of it, marked as repeating one of its runs (same starting state, same moves) "
              "or not; only a failure that repeats its run contests it. A test that changes what a piece can be trusted at "
              "earns credit.")
TRUST_PROPS = {"result": {"type": "string", "enum": ["holds", "fails"],
                          "description": "test: what your own run (evidence) showed about the entry"}}
HYP_CITES = {"type": "array", "items": {"type": "string"},
             "description": "E-ids of the pieces (findings, explanations, claims, guide posts) these moves rely on; if they "
                            "complete a level first, their authors earn credit for understanding (E-ids in your note count too)"}


def fit_rows(rows, budget: int, steps=(130, 50, 0)) -> list[str]:
    """Rows of (head, text, tail), most important first. Every row is shown; past the budget the texts of the later rows
    are shortened step by step, down to the head alone (an id and what came of it), and none is dropped."""
    rows = [[h, clip(t, steps[0]), tail] for h, t, tail in rows]
    size = lambda: sum(len(h) + len(t) + len(tail) + 3 for h, t, tail in rows)
    for step in steps[1:]:
        for row in reversed(rows):
            if size() <= budget:
                break
            row[1] = clip(row[1], step) if step else ""
    return [h + (f": {t}" if t else "") + tail for h, t, tail in rows]


def idea_words(text: str) -> frozenset:
    return frozenset(w for w in IDEA_WORD.findall(str(text).lower()) if w not in IDEA_STOP)


def same_idea(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def clean_rules(r) -> dict:
    """A bounded rule set, filling anything missing from the defaults."""
    import copy
    import json
    if isinstance(r, str) and r.strip():
        r = json.loads(open(r).read() if r.strip().endswith(".json") else r)
    r = r if isinstance(r, dict) else {}
    out = copy.deepcopy(RULES)
    clip_to = lambda key, v, d: min(BOUNDS[key][1], max(BOUNDS[key][0], float(v))) if isinstance(v, (int, float)) else d
    for key, d in RULES["credit"].items():
        out["credit"][key] = round(clip_to(key, (r.get("credit") or {}).get(key), d), 3)
    sub = r.get("substrate") or {}
    for key in ("plan_ttl", "items", "agree"):
        out["substrate"][key] = int(clip_to(key, sub.get(key), RULES["substrate"][key]))
    if isinstance(sub.get("show"), list):
        out["substrate"]["show"] = [v for v in VIEWS if v in sub["show"]]
    for key, d in RULES["substrate"]["rank"].items():
        out["substrate"]["rank"][key] = round(clip_to(key, (sub.get("rank") or {}).get(key), d), 3)
    if isinstance(sub.get("promoted_kinds"), list):
        out["substrate"]["promoted_kinds"] = [str(x) for x in sub["promoted_kinds"]][:3]
    return out


def rules_text(r: dict, model: bool = False, hyp: bool = False) -> str:
    c, s = r["credit"], r["substrate"]
    return (f"credit: first to complete a level +{c['priority']:g}; each unseen frontier state +{c['novel']:g} (up to "
            f"{c['novel_cap']:g} per tick); finding backed by receipts +{c['finding']:g}, dead end backed by receipts "
            f"+{c['dead_end']:g}; each author the first solver read +{c['reuse']:g}; "
            + (f"the first to test a new idea on a level, with receipts from after its claim, +{c['idea']:g}; each piece a "
               f"first solver cites +{c['understanding']:g} to its author, and the pieces it built on "
               f"+{c['understanding'] / 2:g}; " if hyp else "")
            + "royalties paid out of the earner's credit: "
            f"{c['lead_share']:g} to its lead, {c['action_share']:g} to the author of an invented action it used; "
            + (f"each move recorded after your model merge that your version predicts and the version before it did not "
               f"+{c['model']:g} (up to {c['model_cap']:g} per merge per tick); " if model else "")
            + f"each credit point earned in the last two ticks buys {c['steps_per_credit']:g} steps. "
            f"Substrate: plans stay open {s['plan_ttl']} ticks; a kind is agreed once {s['agree']} agents adopt it; each turn shows {', '.join(s['show']) or 'nothing'} "
            f"({s['items']} items each); search ranks posts by relevance + {s['rank']['cites']:g} log(1+citations) + "
            f"{s['rank']['credit']:g} log(1+author credit) + {s['rank']['recency']:g}/(1+age)"
            + (f"; promoted kinds {', '.join(s['promoted_kinds'])}" if s["promoted_kinds"] else ""))


class Explore(Directions):
    """Mixin for ArcWorld."""

    def _init_explore(self, opts: dict):
        on = lambda key: str(opts.get(key, "")).lower() in ("1", "true", "yes")
        self.explore = on("explore")
        self.delegation = on("delegation")
        self.receipts: dict[str, dict] = {}
        self.plans: dict[str, dict] = {}
        self.credit: Counter = Counter()
        self.earned: Counter = Counter()  # (agent, tick) -> credit earned
        self.credit_log: list[dict] = []
        self.combo_credit: dict[str, list] = defaultdict(lambda: [0.0, 0])
        self.op_credit: dict[str, list] = defaultdict(lambda: [0.0, 0])
        self.seen_states: dict[str, set] = defaultdict(set)
        self.changes: dict[str, np.ndarray] = {}
        self.changes_n: Counter = Counter()
        self.mask: dict[str, np.ndarray] = {}
        self.novel_tick: Counter = Counter()
        self.novel_count: Counter = Counter()
        self.finding_tick: set = set()
        self.switches = 0
        self.teams: dict[str, dict] = {}
        self.roles: dict[str, dict] = {}
        self.reports: dict[str, dict] = {}
        self.trackers: dict[str, str] = {}
        self.pins: dict[str, dict] = {}
        self.lead_log: list[dict] = []
        self.rules = clean_rules(opts.get("rules"))
        self.entry_meta: dict[str, dict] = {}  # entry id -> game, level and type (claim, finding, dead_end, challenge, solution)
        if getattr(self, "sync", False) and not opts.get("rules"):  # sync keeps the default view small
            self.rules["substrate"]["show"] = [v for v in self.rules["substrate"]["show"] if v not in ("paid", "unused")]
        if on("cycle") and "cycle" not in self.rules["substrate"]["show"]:
            self.rules["substrate"]["show"].append("cycle")
        self.rules_version, self.rules_t = 1, 0
        self.contracts_on = self.explore and not on("no_contracts")
        self.contracts: dict[str, dict] = {}
        self.kinds: dict[str, dict] = {}
        self.kind_reuse: Counter = Counter()
        self._running_entry: dict[str, str] = {}  # agent -> entry or receipt whose moves it is running
        self.entry_runs: dict[str, dict] = defaultdict(lambda: {"runs": 0, "credit": 0.0, "by": set()})
        self.findings: list[dict] = []
        self.tried: dict[tuple, set] = defaultdict(set)  # (game, level) -> actions and click regions tried
        # carry=1: nothing learned is swept off the table when a level falls. Findings stay in view at later levels until a
        # receipt there contradicts them, open plans move to the next level, solutions are published with the solver's
        # findings and with explanations, and a contested finding flags those built on it.
        self.carry = self.explore and on("carry")
        self.solution_of: dict[tuple, str] = {}  # (game, level solved) -> the published solution's entry
        self.explanations: dict[str, list] = defaultdict(list)  # solution entry -> explanation entries
        self.plans_carried = 0
        # model=1: the society's shared model of each game (alphateam/game_model.py)
        self.model_on = self.explore and on("model")
        # inherit=<run.json> (or models=<run.json>): a society starts from the models and guides an earlier one assembled
        inherit = opts.get("inherit")
        self.models = GameModels(self.games, load_inherited(opts.get("models") or inherit)) if self.model_on else None
        self.model_credit_log: list[dict] = []
        # guide=1: one durable, revisable text per game, the society's understanding in language (after VISTA's game guide)
        self.guide_on = self.explore and on("guide")
        # lessons=1 (with guide=1): one more guide, shared by all games, for what one game taught that may hold in others
        # (kinds of goals, counters, controls), so understanding transfers between games and, inherited, between runs
        self.lessons_on = self.guide_on and on("lessons")
        keys = list(self.games) + (["lessons"] if self.lessons_on else [])
        self.guides = {g: [{"v": 0, "author": "lab", "t": 0, "text": "", "why": "empty", "lines": {}}] for g in keys}
        self.guide_evidence: dict[str, dict] = {g: {} for g in keys}  # line -> receipts behind it
        self.guide_contested: dict[str, dict] = {g: {} for g in keys}  # line -> the counterexample against it
        # hypotheses=1: every idea claimed on a level is kept with what came of it and shown to everyone there, so agents
        # see which ideas are untried and which pieces nobody has combined (diversity from information, not assignment).
        # The first to test a new idea with receipts earns credit, and when a level falls, the pieces its solver cites earn
        # credit for understanding, as do (half as much) the pieces those were built on: paid once a result confirms them.
        self.hyp_on = self.explore and on("hypotheses")
        # board=1 (with the kernel's full board): the per-level lists show every finding and dead end, who used each, and the
        # backed findings nobody else has used; guide and lessons lines are entries, ranked by use and never refused
        self.board_on = self.explore and on("board")
        self.guide_ids: dict[str, dict] = {g: {} for g in keys}  # guide line -> the entry that added it
        # island_guides=1 (with islands): each island keeps its own guide; a line belongs to the island that added it
        self.island_guides = self.guide_on and on("island_guides")
        self.guide_island: dict[str, dict] = {g: {} for g in keys}  # guide line -> island key (a group id or an agent)
        # unique_credit=1: when a level falls, ideas tested by one island only earn their testers credit
        self.unique_credit = self.explore and on("unique_credit")
        self.unique_log: list[dict] = []
        self.parts: dict[str, list] = {}  # entry -> the entries it combines or builds on
        self.cites: dict[str, list] = {}  # agent -> the entries cited by the play it is making
        self.shown_explanations: dict[str, set] = defaultdict(set)  # agent -> explanations shown to it in replays
        self.understanding_log: list[dict] = []
        self.refuted_repeats = 0  # plays told that their moves repeat a refuted test
        # trust=1: every piece shows what it can be trusted at (said, seen in k states, reproduced, contested) and what it
        # rests on; plan op=test reproduces a result (the engine reruns its receipts where the world allows) or records a
        # run of one's own as a test of it
        self.trust_on = self.explore and on("trust")
        self.tests: dict[str, list] = defaultdict(list)  # tested entry or receipt -> its tests
        self.engine_tests: Counter = Counter()  # (agent, tick) -> engine reruns asked for
        self.trust_rev = 0  # bumped whenever a test, challenge or new piece can change a trust level
        self._trust_cache: dict = {}
        if self.guide_on and inherit:
            import json
            from pathlib import Path
            old = (json.loads(Path(inherit).read_text())["world"].get("guides") or {})
            for g in keys:
                text = ((old.get(g.split("-")[0]) or [{}])[-1].get("text") or "").strip()[:GUIDE_STORE]
                if text:
                    lines = {ln.strip(): "inherited" for ln in text.splitlines() if ln.strip()}
                    self.guides[g] = [{"v": 0, "author": "inherited", "t": 0, "text": text,
                                       "why": "inherited from an earlier society", "lines": lines}]
        for flag, view in ((self.carry, "carried"), (self.model_on, "model"), (self.guide_on, "guide")):
            if flag and view not in self.rules["substrate"]["show"]:
                self.rules["substrate"]["show"].append(view)
        plan_tool = CARRY_PLAN_TOOL if self.carry else PLAN_TOOL
        if self.hyp_on:
            props = {**plan_tool["parameters"]["properties"], "builds_on": {
                "type": "array", "items": {"type": "string"},
                "description": "claim, finding or done: E-ids of the pieces (others' findings, explanations, claims) this "
                               "one combines or builds on"}}
            plan_tool = {**plan_tool, "description": plan_tool["description"] + HYP_PLAN,
                         "parameters": {**plan_tool["parameters"], "properties": props}}
        if self.trust_on:
            props = plan_tool["parameters"]["properties"]
            props = {**props, **TRUST_PROPS, "op": {**props["op"], "enum": props["op"]["enum"] + ["test"]},
                     "entry": {**props["entry"], "description": props["entry"]["description"]
                               + "; test: the E-id or R-id to test"}}
            plan_tool = {**plan_tool, "description": plan_tool["description"] + TRUST_PLAN,
                         "parameters": {**plan_tool["parameters"], "properties": props}}
        guide_tool = GUIDE_TOOL
        if self.lessons_on:
            guide_tool = {**GUIDE_TOOL, "description": GUIDE_TOOL["description"] + " game=lessons is the guide shared by all "
                          "games: what one game taught that may hold in others.",
                          "parameters": {**GUIDE_TOOL["parameters"], "properties": {
                              **GUIDE_TOOL["parameters"]["properties"],
                              "game": {"type": "string", "description": "default your game; 'lessons' for the guide "
                                                                       "shared by all games"}}}}
        extra = ([plan_tool, FOCUS_TOOL, SUBSTRATE_TOOL] if self.explore else []) + ([LEAD_TOOL] if self.delegation else []) \
            + ([CONTRACT_TOOL] if self.contracts_on else []) + ([MODEL_TOOL] if self.model_on else []) \
            + ([guide_tool] if self.guide_on else [])
        self.tool_names = tuple(self.tool_names) + tuple(t["name"] for t in extra)
        self.TOOLS = list(self.TOOLS) + extra
        if self.hyp_on:  # a play can cite the pieces its moves rely on
            self.TOOLS = [{**t, "parameters": {**t["parameters"], "properties": {**t["parameters"]["properties"],
                                                                                   "cites": HYP_CITES}}}
                          if t["name"] == "play" else t for t in self.TOOLS]
        self._init_directions(opts)  # directions=1 (alphateam/directions.py)

    # ------------------------------------------------------------------ credit

    def steps_allowed(self, aid: str, tick: int) -> int:
        if not self.explore:
            return self.steps_per_tick
        recent = self.earned[(aid, tick - 1)] + self.earned[(aid, tick - 2)]
        return self.steps_per_tick + min(self.steps_per_tick, int(round(self.rules['credit']['steps_per_credit'] * recent)))

    def _combo(self, k, aid: str) -> list[str]:
        ops = {self._now} if getattr(self, "_now", None) else set()
        if k.running_custom.get(aid):
            ops.add(f"x_{k.running_custom[aid]}")
        for e in reversed(k.events):
            if e["t"] < k.tick - 1:
                break
            if e["agent"] == aid and e["ok"] and e["op"] != "intent":
                op = e["op"]
                ops.add(op.split(".")[1] if op.startswith("lab.") else op.split(".")[0])
        return sorted(ops)

    def _award(self, k, aid: str, amount: float, reason: str, _depth: int = 0, extra: list | None = None,
               share: bool = True, label: str = "lead share"):
        """Credit for something the engine observed. Shares are royalties paid out of it, never minted on top: a lead
        receives lead_share of what its children earn, and an invented action's author action_share of what others earn
        with it, so using someone's action or joining a team costs something and lasts only where it pays."""
        if amount <= 0 or aid not in k.agents:
            return
        running = self._running_entry.get(aid) if _depth == 0 else None
        if running:  # credit earned while running the moves written in an entry or a receipt
            self.entry_runs[running]["credit"] += amount
        combo = sorted(set(self._combo(k, aid) + (extra or []) + ([f"run:{self._entry_kind(k, running)}"] if running else []))) \
            if _depth == 0 else [label]
        royalties = []
        author = self._entry_author(k, running) if running and share else None
        if author and author != aid:
            royalties.append((author, self.rules["credit"]["action_share"], f"{aid} earned credit running {running}",
                              "entry royalty", False))
        lead = self._lead_of(k, aid)
        if share and self.delegation and lead and lead != aid and _depth < 4:
            royalties.append((lead, self.rules["credit"]["lead_share"], f"share of {aid}'s credit", "lead share", True))
        if share and _depth == 0:
            for op in combo:
                ca = k.custom.get(op[2:]) if op.startswith("x_") else None
                if ca and ca.author != aid and ca.author in k.agents:
                    royalties.append((ca.author, self.rules["credit"]["action_share"],
                                      f"{aid} earned credit using x_{ca.name}", "inventor share", False))
        total = sum(r[1] for r in royalties)
        scale = min(1.0, 0.9 / total) if total else 1.0  # the earner keeps at least a tenth
        paid = [(to, amount * rate * scale, why, lab, up) for to, rate, why, lab, up in royalties]
        net = amount - sum(x[1] for x in paid)
        self.credit[aid] += net
        self.earned[(aid, k.tick)] += net
        self.credit_log.append({"t": k.tick, "agent": aid, "credit": round(net, 3), "gross": round(amount, 3),
                                "reason": reason, "combo": combo})
        key = "+".join(combo) or "(none)"
        self.combo_credit[key][0] += amount  # the ledger keeps what the combination earned in all
        self.combo_credit[key][1] += 1
        for op in combo:
            self.op_credit[op][0] += amount
            self.op_credit[op][1] += 1
        for to, value, why, lab, up in paid:
            self._award(k, to, value, why, _depth + 1, share=up, label=lab)

    def _novel(self, k, aid: str, game: str, frame: np.ndarray, before: np.ndarray, level: int):
        """Credit a frame state at the game's frontier that no copy has seen, ignoring cells that change on most steps."""
        if game not in self.mask:  # first learn which cells are volatile (timers, counters), then freeze the mask
            if game not in self.changes:
                self.changes[game] = np.zeros(frame.shape, int)
            self.changes[game] += (frame != before)
            self.changes_n[game] += 1
            if self.changes_n[game] < 50:
                return
            self.mask[game] = self.changes[game] > 0.3 * self.changes_n[game]
        counters = getattr(self, "counters", None)  # step bars the evidence engine found (evidence=1) are never new
        masked = np.where(self.mask[game] | counters.mask(game) if counters is not None else self.mask[game], -1, frame)
        key = (level, hash(masked.tobytes()))
        if key in self.seen_states[game]:
            return
        self.seen_states[game].add(key)
        c = self.rules["credit"]
        self.novel_count[game] += 1
        if level == self.best[game] and c["novel"] > 0 and self.novel_tick[(aid, k.tick)] + c["novel"] <= c["novel_cap"] + 1e-9:
            self.novel_tick[(aid, k.tick)] += c["novel"]
            self._award(k, aid, c["novel"], f"new state in {game.split('-')[0]} level {level + 1}")

    def _first_solve(self, k, aid: str, game: str, level: int):
        short = game.split("-")[0]
        if self.unique_credit:
            self._pay_unique(k, game, level)
        self._pay_directions(k, aid, game, level)  # directions=1: each direction the solver cites or is on
        for p in self.plans.values():
            if p["status"] == "open" and p["game"] == game and p["level"] == level and p["agent"] != aid:
                if self.carry:  # the plan and its interim findings move to the next level instead of going stale
                    p["level"], p["carried_from"] = level + 1, level
                    p["until"] = max(p["until"], k.tick + 1)
                    self.plans_carried += 1
                    k.notify(p["agent"], "lab", f"{aid} completed {short} level {level}. Your plan {p['id']} and its interim "
                             f"findings are carried to level {level + 1} (replay the published solution to get there; drop "
                             f"the plan if it no longer applies).")
                else:
                    k.notify(p["agent"], "lab", f"{aid} completed {short} level {level}, the problem of your plan "
                             f"{p['id']}; its solution is published (replay reaches the new frontier).")
        sid = self.solution_of.get((game, level))
        if self.carry and sid and k.entry(sid):  # publish the understanding with the moves
            mine = [f for f in self.findings if f["agent"] == aid and f["game"] == game and f["backed"]][-3:]
            if mine:
                k.entry(sid).content += "\nThe solver's findings on this game: " + "; ".join(
                    f"{f['id']} (level {f['level']}): {clip(f['text'], 120)}" for f in mine)
            k.notify(aid, "lab", f"you were first to complete {short} level {level}; others will replay your moves "
                     f"({sid}). Explain why they work (plan op=explain entry={sid}, finding, evidence = your receipts) so "
                     f"those who replay them understand the level, not just pass it."
                     + (" If what you learned may hold in other games, add it to the lessons (guide op=add game=lessons)."
                        if self.lessons_on else ""))
        used = [eid for eid, t in k.retrieved.get(aid, {}).items() if t >= k.tick - 3]
        used += [r for p in self.plans.values() if p["agent"] == aid for r in p.get("refs", [])]
        authors, kinds = [], set()
        for eid in used:
            e = k.entry(eid)
            if not e:
                continue
            kinds.add(f"read:{'solution' if e.author == 'lab' else e.kind}")
            if e.author not in ("lab", aid) and e.author in k.agents and e.author not in authors:
                authors.append(e.author)
                self.kind_reuse[e.kind] += 1
        # pieces of the shared picture in the solver's view count as used, not only what it read in full: guide lines
        # on this game written before this tick, and findings on it delivered to the solver (sync) in the last 5 ticks
        in_view = []
        if self.guide_on:
            for g in [game] + (["lessons"] if self.lessons_on else []):
                for line, who in self.guides[g][-1]["lines"].items():
                    first = next((v["t"] for v in self.guides[g] if line in v["lines"]), k.tick)
                    if first < k.tick and line not in self.guide_contested[g]:
                        in_view += [x for x in who.split(",") if x not in in_view]  # a merged line has co-authors
            if in_view:
                kinds.add("read:guide")
        delivered = getattr(k.agents[aid], "pushed", set())
        for eid in delivered:
            e, meta = k.entry(eid), self.entry_meta.get(eid)
            if e and meta and meta["game"] == game and e.tick >= k.tick - 5 and meta["type"] in (
                    "finding", "dead_end", "explanation", "model", "guide") and e.author not in in_view:
                in_view.append(e.author)
                kinds.add(f"delivered:{meta['type']}")
        for eid in self.shown_explanations.get(aid, ()):  # an explanation read in a replay pays when a later level falls
            e, meta = k.entry(eid), self.entry_meta.get(eid)
            if e and meta and meta["game"] == game and e.author not in in_view:
                in_view.append(e.author)
                kinds.add("replayed:explanation")
        for x in in_view:
            if x not in authors and x not in ("lab", aid) and x in k.agents:
                authors.append(x)
        self._award(k, aid, self.rules["credit"]["priority"], f"first to complete {game.split('-')[0]} level {level}",
                    extra=sorted(kinds))
        for x in authors[:4]:
            self._award(k, x, self.rules["credit"]["reuse"],
                        f"{aid} used their work before completing {game.split('-')[0]} level {level}")
        if self.hyp_on:
            self._pay_understanding(k, aid, game, level)

    def _pay_unique(self, k, game: str, level: int):
        """unique_credit=1: tested ideas on a level that fell, which no other island tested, earn their testers 0.5 each
        (at most 3 per island), whether they held or not."""
        tested = [p for p in self.plans.values() if p["game"] == game and p["level"] == level
                  and (p.get("finding") or p["status"] == "done") and p["agent"] in k.agents]
        isl = {p["id"]: self.island_key(k, p["agent"]) for p in tested}
        paid = Counter()
        for p in sorted(tested, key=lambda p: p["t"]):
            w = idea_words(p["hypothesis"])
            if any(isl[q["id"]] != isl[p["id"]] and same_idea(w, idea_words(q["hypothesis"])) >= 0.33 for q in tested):
                continue
            if paid[isl[p["id"]]] >= 6:  # 3 credit per island
                continue
            paid[isl[p["id"]]] += 1
            self._award(k, p["agent"], 0.5, f"an idea only its island tested on {game.split('-')[0]} level {level} ({p['id']})")
            self.unique_log.append({"t": k.tick, "game": game, "level": level, "plan": p["id"], "agent": p["agent"],
                                    "island": isl[p["id"]]})

    def _pay_understanding(self, k, aid: str, game: str, level: int):
        """Credit for understanding, paid once a result confirms it: the pieces the solver cites (in the play that
        completed the level, E-ids in its note, the pieces its plans build on) earn their authors credit, and the pieces
        those were built on half as much. Only entries others posted before this tick count."""
        short, c = game.split("-")[0], self.rules["credit"]["understanding"]
        if c <= 0:
            return
        direct = list(self.cites.get(aid, []))
        direct += [x for p in self.plans.values() if p["agent"] == aid and p["game"] == game and (
            p["status"] == "open" or (p.get("resolved_t") or -9) >= k.tick - 3) for x in p.get("builds_on", [])]
        own = [x for x in direct if (k.entry(x) and k.entry(x).author == aid)]
        direct += [x for o in own for x in self.parts.get(o, [])]  # the solver's own pieces lead to what they combined
        pay, seen = [], set()

        def add(eid: str, amount: float, why: str):
            e = k.entry(eid)
            if e and eid not in seen and e.tick < k.tick and e.author not in ("lab", aid) and e.author in k.agents:
                seen.add(eid)
                pay.append((e.author, amount, eid, why))

        for eid in dict.fromkeys(direct):
            add(eid, c, f"{aid} cited {eid} completing {short} level {level}")
        for _, _, eid, _ in list(pay):
            for part in self.parts.get(eid, []):
                add(part, c / 2, f"{aid} completed {short} level {level} citing {eid}, which builds on {part}")
        pay = pay[:6]
        for to, amount, eid, why in pay:
            self._award(k, to, amount, why, extra=["cited"])
            k.notify(to, "lab", f"{aid} completed {short} level {level} citing {eid}, yours or built on yours: "
                     f"+{amount:g} credit for understanding")
        if pay:
            self.understanding_log.append({"t": k.tick, "solver": aid, "game": short, "level": level,
                                           "paid": [{"to": to, "entry": eid, "credit": amount} for to, amount, eid, _ in pay]})

    def repeats_refuted(self, game: str, level: int, moves: list[str], n: int = 6) -> list[tuple]:
        """Refuted tests that these moves repeat: agents' dead ends on this level whose receipts share a run of n moves
        with them. In the F2 runs about a fifth of frontier moves re-tested refuted ideas, mostly by playing, not claiming,
        and often after reading the refutation; this is said in the play's result, as information."""
        if len(moves) < n:
            return []
        mine = {tuple(moves[i:i + n]) for i in range(len(moves) - n + 1)}
        out = []
        for f in self.findings:
            if f["outcome"] != "dead_end" or f["agent"] == "lab" or f["game"] != game or f["level"] != level:
                continue
            for r in f.get("evidence", []):
                rc = self.receipts.get(r)
                if rc and rc["levels_after"] < rc["level_start"] and any(  # a receipt that completed a level refutes nothing
                        tuple(rc["moves"][i:i + n]) in mine for i in range(len(rc["moves"]) - n + 1)):
                    out.append((f, r))
                    break
        return out[-2:]

    def _piece_ids(self, k, given, text: str = "") -> list[str]:
        """Entries named as pieces: given ids and E-ids written in a text, those that exist."""
        ids = [str(v).strip().upper() for v in as_list(given) if str(v).strip()] + E_ID.findall(str(text or ""))
        return [x for x in dict.fromkeys(ids) if k.entry(x)][:6]

    def idea_status(self, k, p: dict) -> str:
        if p["status"] == "done":
            return {"works": "WORKED", "dead_end": "REFUTED"}.get(p.get("outcome"), "UNCLEAR") + (
                f" ({p['finding']})" if p.get("finding") else "")
        if p["status"] == "open" and p["until"] >= k.tick:
            return f"testing until t{p['until']}" + (f", interim {p['finding']}" if p.get("finding") else "")
        return ("never tested" if not p.get("finding") else f"stopped after interim {p['finding']}") + (
            " (dropped)" if p["status"] == "dropped" else " (expired)")

    def similar_ideas(self, game: str, level: int, hypothesis: str, n: int = 3) -> list[dict]:
        words = idea_words(hypothesis)
        scored = [(same_idea(words, idea_words(p["hypothesis"])), p) for p in self.plans.values()
                  if p["game"] == game and p["level"] == level]
        return [p for s, p in sorted(scored, key=lambda x: (-x[0], -x[1]["t"])) if s >= SAME_IDEA][:n]

    def ideas_view(self, k, game: str, level: int, budget: int = 5000, aid: str | None = None) -> str:
        ideas = [p for p in self.plans.values() if p["game"] == game and p["level"] == level
                 and (aid is None or k.seen_by(aid, p.get("entry")))]
        if not ideas:
            return ""
        status = {p["id"]: self.idea_status(k, p) for p in ideas}
        order = lambda p: (["testing", "never", "stopped", "UNCLEAR", "REFUTED", "WORKED"].index(
            next(w for w in ("testing", "never", "stopped", "UNCLEAR", "REFUTED", "WORKED") if status[p["id"]].startswith(w))),
            -p["t"])
        count = Counter(status[p["id"]].split(" ")[0].lower() for p in ideas)
        label = {"never": "never tested", "stopped": "stopped early"}
        rows = [(f"  {p['id']} {p['agent']} t{p['t']}{self.direction_tag(p['entry'])} {status[p['id']]}"
                 + (f", carried from level {p['carried_from']}" if p.get("carried_from") else "")
                 + (f", builds on {', '.join(p['builds_on'])}" if p.get("builds_on") else ""), p["hypothesis"], "")
                for p in sorted(ideas, key=order)]
        return (f"IDEAS on {game.split('-')[0]} level {level}, every claim here and what came of it ("
                + ", ".join(f"{v} {label.get(w, w)}" for w, v in count.most_common()) + ")"
                + (". Test one nobody has, take over one never tested, or combine pieces (claim builds_on=[E-ids])"
                   if getattr(self, "instructions", True) else "") + ":\n" + "\n".join(fit_rows(rows, budget)))

    def _users(self, k, f) -> list[str]:
        """Agents other than its author who cited a piece (in a post's refs, builds_on or text, or in a play's cites)."""
        who = []
        for c in k.citers.get(f["id"], []):
            x = k.index[c].author if c in k.index else c.split()[0]
            if x not in (f["agent"], "lab") and x not in who:
                who.append(x)
        return who

    def _used_by(self, k, f) -> str:
        who = self._users(k, f)
        return f" [used by {', '.join(who[:5])}{f' +{len(who) - 5}' if len(who) > 5 else ''}]" if who else ""

    def note_tried(self, game: str, level: int, label: str):
        m = __import__("re").fullmatch(r"click\((\d+),(\d+)\)", label)
        self.tried[(game, level)].add(f"click r{int(m.group(2)) // 8} c{int(m.group(1)) // 8}" if m else label)

    def untried_line(self, aid: str, game: str) -> str:
        """What no copy has tried at the game's frontier: accepted actions, and 8x8 click regions (an idle prompt as information)."""
        level = self.best[game]
        tried = self.tried[(game, level)]
        acts = (self.state.get((aid, game)) or {}).get("actions") or []
        plain = [x for x in acts if x not in ("ACTION6", "RESET") and x not in tried]
        out = []
        if plain:
            out.append("actions " + ", ".join(plain))
        if "ACTION6" in acts:
            regions = [(r, c) for r in range(8) for c in range(8) if f"click r{r} c{c}" not in tried]
            if regions:
                sample = ", ".join(f"({c * 8 + 4},{r * 8 + 4})" for r, c in regions[:6])
                out.append(f"clicks in {len(regions)} of 64 8x8 regions, e.g. at {sample}")
        return (f"UNTRIED by any copy at {game.split('-')[0]} level {level + 1}: " + "; ".join(out)) if out else ""

    # ------------------------------------------------------------------ receipts (called from play)

    def _receipt(self, k, aid: str, game: str, level0: int, moves: list[str], lines: list[str], level1: int,
                 **extra) -> str:
        """extra: where the play started (start: the moves since the level began; instance: which game instance, when
        copies differ) and what each step did (results), so the run can be told apart from runs in other states and
        rerun exactly."""
        rid = k.new_id("R")
        self.receipts[rid] = {"id": rid, "agent": aid, "game": game, "t": k.tick, "level_start": level0 + 1,
                              "moves": moves, "steps": lines, "levels_after": level1, **extra}
        self.trust_rev += 1
        return rid

    def start_text(self, r: dict) -> str:
        """The state a run started in, in words: the level start plus the moves made since it began."""
        s = list(r.get("start") or [])
        base = f"the start of level {r['level_start']}"
        if not s:
            return base
        return base + (f" + {' '.join(s)}" if len(s) <= 6 else f" + {len(s)} moves (last: {' '.join(s[-4:])})")

    def evidence_marks(self, game: str, level: int, text: str) -> str:
        return ""  # worlds that keep evidence (arc with evidence=1) say what recorded runs show about the moves a text names

    def receipt_line(self, rid: str) -> str:
        r = self.receipts.get(rid)
        if not r:
            return f"{rid}: no such receipt"
        done = f", completed level {r['level_start']}" if r["levels_after"] >= r["level_start"] else ", level unchanged"
        start = f", from {self.start_text(r)}" if self.trust_on and r.get("start") is not None else ""
        return (f"{rid} (engine record, t{r['t']} {r['agent']} on {r['game'].split('-')[0]} level {r['level_start']}: "
                f"{len(r['moves'])} moves {' '.join(r['moves'][:12])}{' ...' if len(r['moves']) > 12 else ''}{done}{start})")

    # ------------------------------------------------------------------ plan and focus

    def _problem(self, game: str) -> int:
        return self.best[game] + 1

    def _plan(self, k, aid, a):
        # result=holds|fails with no op and no hypothesis is a test of a piece (op=test), not a claim missing its hypothesis
        op = str(a.get("op") or ("test" if self.trust_on and a.get("result") and not a.get("hypothesis") else "claim"))
        mine = [p for p in self.plans.values() if p["agent"] == aid and p["status"] == "open"]
        if op == "claim":
            game = self._game_ref(a.get("game")) if a.get("game") else self.assigned.get(aid)
            k._need(game, "which game?")
            hyp = str(a.get("hypothesis") or "").strip()
            k._need(hyp, "say what you will test (hypothesis)" + (
                f"; you gave result={clip(str(a['result']), 40)}, which belongs to op=test (record whether a piece held)"
                if a.get("result") else ""))
            untested = [p for p in mine if not p.get("finding")]  # one open plan each: a new claim drops the last
            for p in mine:
                p["status"] = "dropped"
            dropped = (f" This dropped your open {', '.join(p['id'] for p in untested)} (never tested); it stays under "
                       f"IDEAS for anyone to take over, or claim it again." if untested else "")
            ttl = max(1, min(8, int(a.get("ttl") or self.rules["substrate"]["plan_ttl"])))
            level = self._problem(game)
            pid = k.new_id("P")
            others = [p for p in self.open_plans(k, game) if p["agent"] != aid and k.seen_by(aid, p.get("entry"))]
            similar = [p for p in self.similar_ideas(game, level, hyp, n=12) if k.seen_by(aid, p.get("entry"))][:3] \
                if self.hyp_on else []
            parts = self._piece_ids(k, a.get("builds_on"), hyp) if self.hyp_on else []
            e = k.post(self._substrate_board(k, aid), aid, "plan",
                       f"PLAN {pid} on {game.split('-')[0]} level {level} (until t{k.tick + ttl}): {clip(hyp, 400)}", notify=False)
            criteria = str(a.get("criteria") or "").strip()
            self.entry_meta[e.id] = {"game": game, "level": level, "type": "claim"}
            self.plans[pid] = {"id": pid, "agent": aid, "game": game, "level": level, "hypothesis": hyp, "t": k.tick,
                               "until": k.tick + ttl, "status": "open", "entry": e.id, "refs": [],
                               "criteria": clip(criteria, 200), "builds_on": parts}
            if criteria:
                e.content += f" Criteria: {clip(criteria, 200)}"
            if parts:  # combining pieces: whose pieces they are is legible, and their authors hear of it
                self.parts[e.id] = parts
                e.refs += parts
                e.content += " Builds on: " + ", ".join(f"{x} ({k.entry(x).author})" for x in parts)
                for who in dict.fromkeys(k.entry(x).author for x in parts):
                    if who in k.agents and who != aid and k.seen_by(who, e.id):
                        k.notify(who, aid, f"claimed {pid} on {game.split('-')[0]} level {level} building on your "
                                 f"{', '.join(x for x in parts if k.entry(x).author == who)}: {clip(hyp, 120)}")
            filed = self.file_piece(k, aid, [e.id], plan=self.plans[pid], parts=parts)
            # those already on the problem (and able to see the claim) are told, so overlaps can be resolved by messages
            for p in [p for p in others if k.seen_by(p["agent"], e.id)]:
                k.notify(p["agent"], aid, f"claimed {pid} on {game.split('-')[0]} level {level}: {clip(hyp, 120)}. If it "
                         f"overlaps your {p['id']}, message @{aid} to split the work.")
            same = "; ".join(f"{p['agent']}: {clip(p['hypothesis'], 80)}" for p in others[:4])
            near = [g for g in k.active_groups() if getattr(k, "teaming", False) and g.purpose and not k.inside(aid, g.id)
                    and same_idea(idea_words(hyp), idea_words(g.purpose)) >= 0.33]
            close = "; ".join(f"{p['id']} {p['agent']} {self.idea_status(k, p)}: {clip(p['hypothesis'], 100)}"
                              for p in similar)
            return "lab.plan", (f"claimed {pid} ({e.id}) on {game.split('-')[0]} level {level} until t{k.tick + ttl}." +
                                (f" Others' open plans on it: {same}" if same else " Nobody else has an open plan on it.") +
                                (f" Similar ideas already claimed on this level: {close}. If yours is the same, build on "
                                 f"what they found or test another idea; testing it again earns no credit for a new idea."
                                 if close else "") +
                                "".join(f" It is close to {g.id}'s {'purpose' if self.directions_on else 'direction'} "
                                        f"('{clip(g.purpose, 100)}'; "
                                        f"{', '.join(k.agents_inside(g.id)[:5])})." for g in near[:2]) + filed + dropped), pid
        if op == "explain":
            k._need(self.carry, "op must be claim, finding, done, drop or challenge")
            sol = k.entry(str(a.get("entry") or "").strip().upper())
            meta = self.entry_meta.get(sol.id) if sol else None
            k._need(meta and meta["type"] == "solution", "entry: the published solution (an E-id) you explain")
            text = str(a.get("finding") or "").strip()
            ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
            k._need(text, "say why the solution works (finding)")
            k._need(all(r in self.receipts for r in ev), "evidence: receipt ids (R-ids) from play")
            short = meta["game"].split("-")[0]
            e = k.post(self._substrate_board(k, aid), aid, "result", f"EXPLANATION of the {short} level {meta['level']} "
                       f"solution ({sol.id}): {clip(text, 900)}\n" + ("Evidence: " + "; ".join(
                           self.receipt_line(r) for r in ev[:3]) if ev else "No engine evidence cited: a claim."),
                       refs=[sol.id] + ev, notify=False)
            self.entry_meta[e.id] = {"game": meta["game"], "level": meta["level"], "type": "explanation"}
            self.explanations[sol.id].append(e.id)
            if ev and (aid, k.tick) not in self.finding_tick:
                self.finding_tick.add((aid, k.tick))
                self._award(k, aid, self.rules["credit"]["finding"], "explanation backed by receipts")
            return "lab.explain", f"posted explanation {e.id} of {sol.id}" + (" backed by engine records" if ev else ""), e.id
        if op == "test":
            return self._test(k, aid, a)
        if op == "challenge":
            target = next((f for f in self.findings if f["id"] == str(a.get("entry") or "").strip().upper()), None)
            k._need(target, "entry: the finding you contest (an E-id of a FINDING)")
            k._need(target["agent"] != aid, "revise your own finding instead")
            text = str(a.get("finding") or "").strip()
            ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
            k._need(text and ev and all(r in self.receipts for r in ev), "a challenge needs the counterexample (finding) "
                    "and the receipts that show it")
            later = min(self.receipts[r]["level_start"] for r in ev)
            if self.carry and later > target["level"]:  # true where it was found, but not at this level: retire it here
                target["superseded_at"] = later
                e = k.post(self._substrate_board(k, aid), aid, "critique", f"DOES NOT HOLD at level {later}: {target['id']} "
                           f"({target['agent']}, level {target['level']}): {clip(text, 600)}\nEvidence: "
                           + "; ".join(self.receipt_line(r) for r in ev[:3]), refs=[target["id"]] + ev, notify=False)
                self.entry_meta[e.id] = {"game": target["game"], "level": later, "type": "dead_end"}
                k.notify(target["agent"], aid, f"showed that your finding {target['id']} does not hold at level {later} "
                         f"({e.id}); it stays true for level {target['level']}")
                return "lab.challenge", (f"{target['id']} no longer carried forward from level {later} ({e.id})"
                                         + self.pay_change(k, aid, target["id"], "counterexample")), e.id
            e = k.post(self._substrate_board(k, aid), aid, "critique", f"CHALLENGE to {target['id']} ({target['agent']}): "
                       f"{clip(text, 600)}\nEvidence: " + "; ".join(self.receipt_line(r) for r in ev[:3]),
                       refs=[target["id"]] + ev, notify=False)
            self.entry_meta[e.id] = {"game": target["game"], "level": target["level"], "type": "challenge",
                                     "target": target["agent"]}
            target.setdefault("challenges", []).append({"by": aid, "entry": e.id, "t": k.tick})
            target["contested"] = True
            if self.carry:  # those who built on it are told, and their findings are flagged
                for f in self.findings:
                    if target["id"] in f.get("builds_on", []):
                        f["built_on_contested"] = target["id"]
                        k.notify(f["agent"], aid, f"{target['id']}, which your finding {f['id']} builds on, is contested ({e.id})")
            k.notify(target["agent"], aid, f"challenged your finding {target['id']} with a counterexample ({e.id}); "
                     f"revise it with plan op=done or op=finding, entry={target['id']}")
            return "lab.challenge", f"challenged {target['id']} ({e.id})" + self.pay_change(
                k, aid, target["id"], "counterexample"), e.id
        if op == "drop":
            k._need(mine, "you have no open plan")
            for p in mine:
                p["status"] = "dropped"
            return "lab.plan", f"dropped {', '.join(p['id'] for p in mine)}", None
        k._need(op in ("done", "finding"), "op must be claim, finding, done or drop")
        finding = str(a.get("finding") or "").strip()
        k._need(finding, "say what you found (finding)")
        ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
        bad = [r for r in ev if r not in self.receipts]
        k._need(not bad, f"no such receipts: {', '.join(bad)}")
        outcome = str(a.get("outcome") or "unclear").strip().lower()
        outcome = outcome if outcome in ("works", "dead_end", "unclear") else "unclear"
        plan = mine[0] if mine else None
        if plan and op == "done":
            plan["status"] = "done"
            plan["outcome"] = outcome
        game = plan["game"] if plan else self.assigned.get(aid)
        backed = [r for r in ev if self.receipts[r]["game"] == game]
        label = {"works": "FINDING", "dead_end": "DEAD END", "unclear": "FINDING (unclear)"}[outcome]
        if op == "finding":
            label = "INTERIM " + label
        body = (f"{label} on {game.split('-')[0]} level {self._problem(game) if not plan else plan['level']}"
                + (f" ({plan['id']}: {clip(plan['hypothesis'], 120)}"
                   + (f"; criteria: {clip(plan['criteria'], 100)}" if plan.get("criteria") else "") + ")" if plan else "")
                + f": {clip(finding, 900)}\n"
                + ("Evidence: " + "; ".join(self.receipt_line(r) for r in backed[:3]) if backed else
                   "No engine evidence cited: a claim."))
        marks = self.evidence_marks(game, plan["level"] if plan else self._problem(game), finding)
        if marks:
            body += f"\nEngine records: {marks}"
        known = {f["id"]: f for f in self.findings}
        builds_on = [x for x in (str(v).strip().upper() for v in as_list(a.get("builds_on"))) if x in known][:6] \
            if self.carry else []
        parts = list(dict.fromkeys(self._piece_ids(k, a.get("builds_on"), finding) + (plan or {}).get("builds_on", [])))[:6] \
            if self.hyp_on else []
        refs = list(dict.fromkeys(([plan["entry"]] if plan else []) + backed + builds_on + parts))
        e = k.post(self._substrate_board(k, aid), aid, "result", body, refs=refs, notify=False)
        self.entry_meta[e.id] = {"game": game, "level": plan["level"] if plan else self._problem(game),
                                 "type": "dead_end" if outcome == "dead_end" else "finding"}
        if parts:
            self.parts[e.id] = parts
        if plan:
            plan["finding"] = e.id
        filed = self.file_piece(k, aid, [e.id], plan=plan, parts=parts)
        idea = ""
        if self.hyp_on and plan and op == "done" and outcome in ("works", "dead_end"):
            # diversity that pays: the first to test an idea on a level, with receipts made after claiming it
            fresh = [r for r in backed if self.receipts[r]["t"] >= plan["t"]]
            words = idea_words(plan["hypothesis"])
            tested = [p for p in self.plans.values() if p is not plan and p["game"] == plan["game"]
                      and p["level"] == plan["level"] and p.get("resolved_t") is not None]
            close = max(tested, key=lambda p: same_idea(words, idea_words(p["hypothesis"])), default=None)
            if not fresh:
                idea = "; no receipt from after your claim, so the idea counts as untested"
            elif close and same_idea(words, idea_words(close["hypothesis"])) >= SAME_IDEA:
                plan["resolved_t"], plan["same_as"] = k.tick, close["id"]
                idea = (f"; the idea was tested before ({close['id']} {close['agent']}, {self.idea_status(k, close)}), so no "
                        f"credit for a new idea")
            else:
                plan["resolved_t"], plan["new_idea"] = k.tick, True
                amount = round(self.rules["credit"]["idea"] * self.idea_weight(k, plan), 3)  # change_credit=1
                self._award(k, aid, amount, f"first to test an idea on {game.split('-')[0]} level {plan['level']}")
                idea = f"; first test of this idea on the level: +{amount:g} credit"
        revises = next((f for f in self.findings if f["id"] == str(a.get("entry") or "").strip().upper()
                        and f["agent"] == aid), None)
        if revises:  # a revision settles the challenges to the finding it revises
            revises["contested"], revises["revised_by"] = False, e.id
            e.refs.append(revises["id"])
            for c in revises.get("challenges", []):
                k.notify(c["by"], aid, f"revised {revises['id']} as {e.id} after your challenge")
        self.findings.append({"id": e.id, "agent": aid, "game": game, "level": plan["level"] if plan else self._problem(game),
                              "outcome": outcome, "text": clip(finding, 160), "backed": bool(backed), "t": k.tick,
                              "revises": revises["id"] if revises else None, "plan": plan["id"] if plan else None,
                              "builds_on": builds_on, "evidence": backed[:3]})
        bad = next((x for x in builds_on if known[x].get("contested")), None)
        if bad:
            self.findings[-1]["built_on_contested"] = bad
        if backed and (aid, k.tick) not in self.finding_tick:
            self.finding_tick.add((aid, k.tick))
            key = "dead_end" if outcome == "dead_end" else "finding"
            self._award(k, aid, self.rules["credit"][key], f"{key.replace('_', ' ')} backed by receipts")
        return "lab.plan", f"posted finding {e.id}" + (" backed by engine records" if backed else " (no receipts: a claim)") \
            + idea + (f". Engine records on the moves it names: {marks}" if marks else "") + filed, e.id

    # ------------------------------------------------------------------ trust (trust=1)

    supports_replay = False  # worlds whose copies can be rerun exactly from a recorded state say so
    state_noun = "state"

    def replay_receipt(self, rid: str) -> tuple:
        return None, "this world cannot rerun recorded runs"

    def start_key(self, rid: str) -> tuple:
        """Two runs start from the same state when they are on the same instance of the same level with the same moves
        made since it began (copies of a level start identically unless the world gives each copy its own instance)."""
        r = self.receipts[rid]
        return r["game"], r["level_start"], r.get("instance"), tuple(r.get("start") or ())

    def _trust_index(self, k) -> dict:
        key = (len(k.index), self.trust_rev, len(self.findings))
        if self._trust_cache.get("key") != key:
            guide = {eid: (g, line) for g, ids in self.guide_ids.items() for line, eid in ids.items()}
            self._trust_cache = {"key": key, "findings": {f["id"]: f for f in self.findings}, "guide": guide,
                                 "levels": {}, "deps": {}}
        return self._trust_cache

    def receipts_behind(self, k, ref: str) -> list[str]:
        """The engine records a piece rests on: a receipt itself, a finding's evidence, a guide line's receipts, or the
        receipts an entry cites or names."""
        if ref in self.receipts:
            return [ref]
        idx = self._trust_index(k)
        f = idx["findings"].get(ref)
        if f is not None:
            return [r for r in f.get("evidence", []) if r in self.receipts]
        if ref in idx["guide"]:
            g, line = idx["guide"][ref]
            return [r for r in self.guide_evidence[g].get(line, []) if r in self.receipts]
        e = k.entry(ref)
        if e is None:
            return []
        return [r for r in dict.fromkeys(list(e.refs or []) + R_ID.findall(e.content)) if r in self.receipts]

    def trust_of(self, k, ref: str) -> dict:
        """What a piece can be trusted at, from evidence: said (no engine record), seen (engine records from k distinct
        starting states), reproduced (rerun by the engine with the same result, or a run by someone else that held),
        contested (a rerun failed, or a test that repeated its run failed, or a counterexample is standing). Engine posts
        and published solutions are facts. Failures under other conditions are counted, not ruled on."""
        idx = self._trust_index(k)
        if ref in idx["levels"]:
            return idx["levels"][ref]
        e = k.entry(ref) if ref.startswith("E") else None
        author = self.receipts[ref]["agent"] if ref in self.receipts else e.author if e else None
        if author == "lab" or self.entry_meta.get(ref, {}).get("type") == "solution":
            out = {"level": "engine", "states": 0, "tests": 0, "other": 0, "receipts": []}
        else:
            rids = self.receipts_behind(k, ref)
            tests = self.tests.get(ref, [])
            f = idx["findings"].get(ref)
            g = idx["guide"].get(ref)
            standing = bool(f and f.get("contested")) or bool(g and g[1] in self.guide_contested[g[0]])
            bad = [t for t in tests if t["result"] == "fails" and (t["how"] == "replay" or t["same_state"])]
            good = [t for t in tests if t["result"] == "holds" and (t["how"] == "replay" or t["by"] != author)]
            other = [t for t in tests if t["result"] == "fails" and t["how"] == "run" and not t["same_state"]]
            level = "contested" if bad or standing else "reproduced" if good else "seen" if rids else "said"
            out = {"level": level, "states": len({self.start_key(r) for r in rids}), "tests": len(tests),
                   "other": len(other), "receipts": rids}
        idx["levels"][ref] = out
        return out

    def deps(self, k, ref: str) -> list[str]:
        """The pieces an entry rests on: what it cites, combines or names (claims it closes are not evidence)."""
        idx = self._trust_index(k)
        if ref in idx["deps"]:
            return idx["deps"][ref]
        e = k.entry(ref) if ref.startswith("E") else None
        ids = [] if e is None else [x for x in dict.fromkeys(list(e.refs or []) + self.parts.get(ref, []) + E_ID.findall(e.content))
                                    if x.startswith("E") and x != ref and k.entry(x)
                                    and self.entry_meta.get(x, {}).get("type") != "claim"]
        idx["deps"][ref] = ids
        return ids

    def rests_on(self, k, ref: str, depth: int = 3) -> dict:
        """The trust levels of what a piece rests on directly, its weakest link, and any contested piece within depth
        steps under it."""
        direct = {x: self.trust_of(k, x)["level"] for x in self.deps(k, ref)}
        weak = min(direct.values(), key=TRUST_RANK.get, default=None)
        contested, seen, frontier = [], {ref}, list(direct)
        for _ in range(depth):
            nxt = []
            for x in frontier:
                if x in seen:
                    continue
                seen.add(x)
                if self.trust_of(k, x)["level"] == "contested":
                    contested.append(x)
                nxt += self.deps(k, x)
            frontier = nxt
        return {"direct": direct, "weakest": weak, "weak_ids": [x for x, v in direct.items() if v == weak],
                "contested": contested}

    def trust_tag(self, k, ref: str, rests: bool = True) -> str:
        if not self.trust_on:
            return ""
        t = self.trust_of(k, ref)
        if t["level"] == "engine":
            return ""
        n, noun = t["states"], self.state_noun
        s = {"said": "said: no run behind it", "seen": f"seen in {n} {noun}{'s' if n != 1 else ''}",
             "reproduced": "reproduced", "contested": "CONTESTED"}[t["level"]]
        if t["other"]:
            s += f"; failed in {t['other']} test{'s' if t['other'] != 1 else ''} under other conditions"
        if rests:
            r = self.rests_on(k, ref)
            if r["contested"]:
                s += f"; rests on contested {', '.join(r['contested'][:2])}"
            elif r["weakest"] == "said":
                s += f"; rests on unbacked {', '.join(r['weak_ids'][:2])}"
        return f" [{s}]"

    def entry_tag(self, k, e) -> str:
        """Shown after an entry where it is delivered or found (kernel): its trust level and weakest support."""
        d = self.direction_tag(e.id)  # directions=1: the direction it was filed under
        if not self.trust_on or e.author == "lab" or e.kind == "data" or \
                self.entry_meta.get(e.id, {}).get("type") in ("claim", "solution"):
            return d
        return self.trust_tag(k, e.id) + d

    def _test(self, k, aid, a):
        """Test before trust: rerun the receipts behind a piece from the states they started in (worlds that allow it),
        or record one's own run as a test of it, marked as from the same starting state as its runs or not."""
        k._need(self.trust_on, "there is no test in this society's rules")
        ref = str(a.get("entry") or "").strip().upper()
        k._need(ref in self.receipts or (ref.startswith("E") and k.entry(ref)), "entry: the E-id or R-id to test")
        target_author = self.receipts[ref]["agent"] if ref in self.receipts else k.entry(ref).author
        rids = self.receipts_behind(k, ref)
        ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
        result = str(a.get("result") or a.get("outcome") or "").strip().lower()
        before = self.trust_of(k, ref)["level"]
        noun = self.state_noun
        if not ev:
            k._need(self.supports_replay, "this world cannot rerun recorded runs: test what it says on your own copy, "
                                          "then give result (holds or fails) and evidence (your receipts)")
            k._need(rids, f"{ref} cites no engine records, so there is no run to reproduce: test what it says on your "
                          f"own copy, then give result and evidence")
            k._need(self.engine_tests[(aid, k.tick)] < 3, "at most 3 engine reruns per tick")
            self.engine_tests[(aid, k.tick)] += 1
            outs = [(r, *self.replay_receipt(r)) for r in rids[:3]]
            ran = [o for o in outs if o[1] is not None]
            k._need(ran, "; ".join(f"{o[0]}: {o[2]}" for o in outs))
            result = "fails" if any(o[1] is False for o in ran) else "holds"
            why = "; ".join(f"{o[0]}: {o[2]}" for o in outs)
            test = {"by": aid, "t": k.tick, "result": result, "how": "replay", "same_state": True,
                    "evidence": [o[0] for o in ran], "why": clip(why, 400)}
            text = (f"TEST of {ref} ({target_author}), rerun by the engine for {aid}: "
                    + ("REPRODUCED" if result == "holds" else "NOT REPRODUCED") + f". {clip(why, 700)}")
        else:
            k._need(result in ("holds", "fails"), "result: holds or fails (what your run showed)")
            bad = [r for r in ev if r not in self.receipts or self.receipts[r]["agent"] != aid]
            k._need(not bad, f"evidence: receipts of your own runs ({', '.join(bad)} are not yours or do not exist)")
            k._need(target_author != aid, "it is yours: test someone else's result, or revise your own")
            # a repeat: the same starting state and the recorded moves again; anything else tests other conditions
            same = [x for x in rids if any(self.start_key(r) == self.start_key(x) and
                                           self.receipts[r]["moves"][:len(self.receipts[x]["moves"])] == self.receipts[x]["moves"]
                                           for r in ev)]
            why = clip(str(a.get("finding") or a.get("why") or ""), 500)
            test = {"by": aid, "t": k.tick, "result": result, "how": "run", "same_state": bool(same), "evidence": ev[:3],
                    "why": why}
            where = (f"repeating {', '.join(same[:2])} (same starting {noun}, same moves)" if same else
                     f"not a repeat of its runs ({', '.join(rids[:2])}): another starting {noun} or other moves" if rids
                     else "(it cites no runs)")
            text = (f"TEST of {ref} ({target_author}) by {aid}: {result.upper()}, {where}: {why}\nEvidence: "
                    + "; ".join(self.receipt_line(r) for r in ev[:3]))
        self.tests[ref].append(test)
        self.trust_rev += 1
        after = self.trust_of(k, ref)["level"]
        meta = self.entry_meta.get(ref) or {}
        game = meta.get("game") or (self.receipts[ref]["game"] if ref in self.receipts else self.assigned.get(aid))
        e = k.post(self._substrate_board(k, aid), aid, "result", text,
                   refs=list(dict.fromkeys([ref] + test["evidence"] + rids)), notify=False)
        self.entry_meta[e.id] = {"game": game, "level": meta.get("level") or (
            self.receipts[ref]["level_start"] if ref in self.receipts else self._problem(game) if game in self.best else 0),
            "type": "test"}
        test["entry"] = e.id
        if target_author in k.agents and target_author != aid:
            k.notify(target_author, aid, f"tested your {ref}: {result} ({e.id}); it is now {after}")
        paid = ""
        if after != before:
            amount = round(self.rules["credit"]["test"] * self.use_weight(k, ref, aid), 3)  # change_credit=1: by use
            self._award(k, aid, amount, f"test that moved {ref} from {before} to {after}")
            self.log_test_credit(k, aid, ref, amount)
            paid = f"; +{amount:g} credit for a test that moved it from {before}"
        return "lab.test", f"{ref} is now {after} ({e.id}){paid}" + (f". {clip(why, 300)}" if test["how"] == "replay" else ""), e.id

    def _engine_dead_end(self, k, aid: str, game: str, level: int, moves: list[str], rid: str, text: str = ""):
        """An engine-owned dead end on a level: a copy reached GAME_OVER (with the moves that led there), or lost its
        progress to a RESET."""
        short = game.split("-")[0]
        text = text or (f"GAME_OVER on {short} level {level} after {len(moves)} moves from the level start ({aid}'s "
                        f"copy); the last ones: {' '.join(moves[-12:]) or '(none)'}")
        kind = text.split(" ")[0]  # GAME_OVER or RESET: one engine dead end of each kind per level is the fact
        same = [f for f in self.findings if f["agent"] == "lab" and f["game"] == game and f["level"] == level
                and f["text"].split(" ")[0] == kind]
        if same:
            same[0]["repeats"] = same[0].get("repeats", 1) + 1
            return
        e = k.post(self._substrate_board(k, aid), "lab", "result", f"DEAD END (engine) on {short} level {level}: {text}\n"
                   f"Evidence: {self.receipt_line(rid)}", refs=[rid], notify=False)
        self.entry_meta[e.id] = {"game": game, "level": level, "type": "dead_end"}
        self.findings.append({"id": e.id, "agent": "lab", "game": game, "level": level, "outcome": "dead_end",
                              "text": clip(text, 160), "backed": True, "t": k.tick, "revises": None, "plan": None,
                              "builds_on": []})

    def entry_label(self, k, e) -> str:
        """What kind of post an entry is, and its level, without its text (teaming=1: who holds what)."""
        meta = self.entry_meta.get(e.id) or {}
        kind = {"claim": "claim", "guide": "guide line", "explanation": "explanation", "challenge": "challenge",
                "dead_end": "dead end", "model": "model change"}.get(meta.get("type"), e.kind)
        if kind == "result" and any(f["id"] == e.id for f in self.findings):
            kind = "finding"
        return f"{kind}{' on level ' + str(meta['level']) if meta.get('level') else ''}"

    def entry_open(self, k, e) -> bool:
        """islands=1: a post about a level opens to everyone once that level has fallen; solutions are open at once."""
        meta = self.entry_meta.get(e.id)
        if not meta:
            return False
        if meta.get("type") == "solution":
            return True
        return meta.get("game") in self.best and bool(meta.get("level")) and self.best[meta["game"]] >= meta["level"]

    def _substrate_board(self, k, aid: str) -> str:
        if getattr(k, "local", False):
            team = self._team_of(k, aid)
            return f"mem:{team}" if team else f"private:{aid}"
        return "global"

    def _game_ref(self, ref) -> str | None:
        ref = str(ref or "").strip().lower()
        return next((g for g in self.games if g.startswith(ref)), None) if ref else None

    def _focus(self, k, aid, a):
        game = self._game_ref(a.get("game"))
        k._need(game, f"game must be one of {', '.join(g.split('-')[0] for g in self.games)}")
        k._need(game != self.assigned.get(aid), "you already play it")
        self.assigned[aid] = game
        self._copy(aid, game)
        self.switches += 1
        st = self.state[(aid, game)]
        return "lab.focus", (f"you now play your copy of {game.split('-')[0]} (level {st['levels'] + 1}, accepted actions "
                             f"{', '.join(st['actions'])}); the society's frontier there is level {self._problem(game)}"), game

    def open_plans(self, k, game: str) -> list[dict]:
        return [p for p in self.plans.values() if p["game"] == game and p["status"] == "open" and p["until"] >= k.tick
                and p["level"] == self._problem(game)]

    # ------------------------------------------------------------------ delegation

    def _team_of(self, k, aid: str) -> str | None:
        """The team an agent is a direct member of (for a sub-team lead: the team it leads)."""
        led = self._led(k, aid)
        if led:
            return led
        return next((g for g in sorted(k.member_of.get(aid, ())) if g in self.teams and k.groups[g].dissolved is None), None)

    def _led(self, k, aid: str) -> str | None:
        return next((g for g, t in self.teams.items() if t["lead"] == aid and k.groups[g].dissolved is None), None)

    def _lead_of(self, k, aid: str) -> str | None:
        led = self._led(k, aid)
        team = self.teams[led]["parent"] if led else self._team_of(k, aid)
        if not team or team not in self.teams or k.groups[team].dissolved is not None:
            return None
        return self.teams[team]["lead"]

    def children(self, k, gid: str) -> list[str]:
        out = []
        for m in sorted(k.groups[gid].members):
            if m in k.agents and m != self.teams[gid]["lead"]:
                out.append(m)
            elif m in self.teams and k.groups[m].dissolved is None:
                out.append(self.teams[m]["lead"])
        return out

    def _lead(self, k, aid, a):
        from .kernel import Board, Group
        op = str(a.get("op") or "found")
        led = self._led(k, aid)
        to = str(a.get("to") or "").strip().upper()
        if op == "found":
            k._need(not led, f"you already lead {led}")
            parent = self._team_of(k, aid)
            gid = k.new_id("G")
            k.groups[gid] = Group(gid, str(a.get("name") or f"Team of {aid}")[:60], f"Led by {aid}.", aid, k.tick, open=True,
                                  delegates={aid}, protocol=f"{aid} leads this team: it assigns roles and jobs to its children.")
            k.boards[f"mem:{gid}"] = Board(f"mem:{gid}", gid)
            k.boards[f"pub:{gid}"] = Board(f"pub:{gid}", gid)
            k._add_member(gid, aid)
            if parent:
                k._remove_member(parent, aid)
                k._add_member(parent, gid)
            self.teams[gid] = {"lead": aid, "parent": parent, "t": k.tick}
            self.lead_log.append({"t": k.tick, "op": "found", "agent": aid, "team": gid, "parent": parent})
            return "lab.lead", f"you lead team {gid}" + (f", a sub-team of {parent}" if parent else "") + \
                "; others can join it (lead op=join team=" + gid + ")", gid
        if op == "join":
            gid = str(a.get("team") or "").strip().upper()
            k._need(gid in self.teams and k.groups[gid].dissolved is None, f"{gid or '?'} is not a team")
            k._need(not led, "you lead a team: found a sub-team inside the other team instead, or step down first")
            k._need(not self._team_of(k, aid), f"you are already in team {self._team_of(k, aid)}: leave it first")
            k._need(k.groups[gid].open, f"{gid} is not open")
            k._need(k.fits(gid, {aid}), f"{gid} (or a team containing it) would have more than "
                    f"{(k.islands or {}).get('cap')} agents")
            k._add_member(gid, aid)
            k.notify(self.teams[gid]["lead"], aid, f"{aid} joined your team {gid}")
            self.lead_log.append({"t": k.tick, "op": "join", "agent": aid, "team": gid})
            return "lab.lead", f"you joined {gid}, led by {self.teams[gid]['lead']}", gid
        if op == "leave":
            team = self._team_of(k, aid)
            k._need(team and not led, "you are in no team (a lead steps down instead)")
            k._remove_member(team, aid)
            self.roles.pop(aid, None)
            return "lab.lead", f"you left {team}", team
        if op == "report":
            text = str(a.get("text") or "").strip()
            k._need(text, "report what?")
            boss = self._lead_of(k, aid)
            k._need(boss, "you have no lead to report to")
            self.reports[aid] = {"t": k.tick, "text": clip(text, 500), "to": boss}
            k.notify(boss, aid, f"report: {clip(text, 200)}")
            team = self.teams[led]["parent"] if led else self._team_of(k, aid)
            k.post(f"mem:{team}", aid, "summary", f"REPORT to {boss}: {clip(text, 800)}", notify=False)
            return "lab.lead", f"reported to {boss}", boss
        if op == "pin":
            text = str(a.get("text") or "").strip()
            k._need(text, "pin what?")
            team = next((g for g, t in self.trackers.items() if t == aid and k.groups[g].dissolved is None), None) or led
            k._need(team, "only a team's lead or tracker can pin its note")
            self.pins[team] = {"by": aid, "t": k.tick, "text": clip(text, 900)}
            k.post(f"mem:{team}", aid, "summary", f"PINNED NOTE: {clip(text, 900)}", notify=False)
            return "lab.lead", f"pinned the note of {team}", team
        k._need(led, "you lead no team")
        if op == "assign":
            k._need(to in self.children(k, led), f"{to or '?'} is not a direct child of {led} (its members and sub-team leads)")
            role, job = str(a.get("role") or "").strip(), str(a.get("job") or "").strip()
            k._need(role or job, "give a role or a job")
            self.roles[to] = {"by": aid, "team": led, "role": clip(role, 200), "job": clip(job, 400), "t": k.tick}
            k.notify(to, aid, f"assigned you: role {clip(role, 80)}; job {clip(job, 160)}")
            self.lead_log.append({"t": k.tick, "op": "assign", "agent": aid, "to": to, "team": led})
            return "lab.lead", f"assigned {to}", to
        if op == "tracker":
            k._need(to == aid or to in self.children(k, led), f"{to or '?'} is not you or a direct child of {led}")
            self.trackers[led] = to
            k.notify(to, aid, f"you keep the pinned note of team {led} (lead op=pin text=...)")
            return "lab.lead", f"{to} keeps {led}'s pinned note", to
        k._need(op == "step_down", f"unknown op {op}")
        if to:
            k._need(to in self.children(k, led) and to in k.groups[led].members, f"{to} is not a member of {led}")
            self.teams[led]["lead"] = to
            k.groups[led].delegates = {to}
            k.notify(to, aid, f"{aid} handed you the lead of team {led}")
            msg = f"handed {led} to {to}"
        else:
            k._dissolve(led)
            msg = f"dissolved {led}"
        self.lead_log.append({"t": k.tick, "op": "step_down", "agent": aid, "team": led, "to": to or None})
        return "lab.lead", msg, led

    def team_depth(self, k) -> int:
        live = [g for g in self.teams if k.groups[g].dissolved is None]
        return max((k.depth(g) for g in live), default=0)

    # ------------------------------------------------------------------ substrate and contracts

    def _substrate(self, k, aid, a):
        op = str(a.get("op") or "")
        if op == "define_kind":
            name = str(a.get("name") or "").strip().lower()
            k._need(KIND_NAME.fullmatch(name), "name: 2-24 lowercase letters and underscores, starting with a letter")
            k._need(name not in self.kinds and name not in k.kinds_allowed(), f"kind {name} already exists")
            purpose = str(a.get("purpose") or "").strip()
            k._need(purpose, "say what entries of this kind are for (purpose)")
            fields = self._fields(k, a)
            self.kinds[name] = {"by": aid, "t": k.tick, "purpose": clip(purpose, 200), "fields": fields,
                                "adopters": [aid], "version": 1}
            k.custom_kinds.add(name)
            k.post("global", "lab", "note", f"New kind of entry '{name}' (defined by {aid}): {clip(purpose, 200)}"
                   + (f"; fields {', '.join(fields)} (one line 'field: value' each)" if fields else "")
                   + f". Write one with write kind={name}; adopt it with substrate op=adopt name={name}.", notify=False)
            return "lab.define_kind", f"defined kind {name}; write entries with write kind={name}", name
        if op in ("adopt", "revise_kind"):
            name = str(a.get("name") or "").strip().lower()
            kind = self.kinds.get(name)
            k._need(kind, f"no agent-defined kind {name or '?'}")
            if op == "adopt":
                k._need(aid not in kind["adopters"], f"you already adopted {name}")
                kind["adopters"].append(aid)
                agreed = len(kind["adopters"]) == self.rules["substrate"]["agree"]
                if agreed:
                    k.post("global", "lab", "note", f"Kind '{name}' is agreed ({', '.join(kind['adopters'])} adopted it)"
                           + (f": fields {', '.join(kind['fields'])}" if kind["fields"] else "") + ".", notify=False)
                return "lab.adopt", f"adopted {name} ({len(kind['adopters'])} adopters" + ("; now agreed" if agreed else "") + ")", name
            k._need(aid in kind["adopters"], f"adopt {name} before revising it")
            if a.get("fields") is not None:
                kind["fields"] = self._fields(k, a)
            if str(a.get("purpose") or "").strip():
                kind["purpose"] = clip(str(a["purpose"]).strip(), 200)
            kind["version"] += 1
            for x in kind["adopters"]:
                if x != aid:
                    k.notify(x, aid, f"revised kind {name} to v{kind['version']}: fields {', '.join(kind['fields']) or 'none'}")
            return "lab.revise_kind", f"{name} is now v{kind['version']}", name
        k._need(op == "propose", "op must be define_kind, adopt, revise_kind or propose")
        text = str(a.get("text") or "").strip()
        k._need(text, "propose what?")
        e = k.post("global", aid, "proposal", f"RULES PROPOSAL: {clip(text, 800)}", notify=False)
        return "lab.propose", f"posted proposal {e.id}; the rules are revised from proposals and from what has paid", e.id

    MOVE = re.compile(r"ACTION[1-7]|RESET|click\(\s*(?:\d+|\{[a-z_]+\})\s*,\s*(?:\d+|\{[a-z_]+\})\s*\)", re.I)

    def entry_moves(self, k, aid: str, ref: str, args: dict) -> list[str]:
        """The moves written in an entry (or a receipt), with placeholders like {x} filled from args."""
        ref = str(ref).strip().upper()
        if ref.startswith("R"):
            k._need(ref in self.receipts, f"no receipt {ref}")
            moves = list(self.receipts[ref]["moves"])
        else:
            e = k.entry(ref)
            k._need(e and k.can_read(aid, e.board) and k.visible(aid, e), f"no entry {ref} you can read")
            moves = [m.replace(" ", "") for m in self.MOVE.findall(e.content)]
            moves = [m if m.lower().startswith("click") else m.upper() for m in moves]
        k._need(moves, f"{ref} has no moves (ACTION1-7, RESET or click(x,y))")
        try:
            moves = [re.sub(r"\{([a-z_]+)\}", lambda m: str(int(args[m.group(1)])), x) for x in moves]
        except (KeyError, ValueError, TypeError) as exc:
            raise ActionError(f"{ref} has placeholders; give their values in args ({exc})")
        return moves

    def _entry_author(self, k, ref: str | None) -> str | None:
        if not ref:
            return None
        if ref.startswith("R"):
            return self.receipts.get(ref, {}).get("agent")
        e = k.entry(ref)
        return e.author if e else None

    def _entry_kind(self, k, ref: str) -> str:
        if ref.startswith("R"):
            return "receipt"
        e = k.entry(ref)
        return e.kind if e else "entry"

    def entry_note(self, eid: str) -> str:
        r = self.entry_runs.get(eid)
        return f" ran {r['runs']}x, earned {r['credit']:.1f}" if r and r["runs"] else ""

    def runs_line(self, k, aid: str, game: str, n: int) -> str:
        rows = []
        for ref, r in sorted(self.entry_runs.items(), key=lambda x: -x[1]["credit"]):
            if r["runs"] and self._entry_game(k, ref) in (game, None):
                rows.append(f"{ref} by {self._entry_author(k, ref)}: ran {r['runs']}x by {len(r['by'])} agents, earned {r['credit']:.1f}")
        return ("ENTRIES RUN AS MOVES on your game (play entry=<id>): " + "; ".join(rows[:n])) if rows else ""

    def _entry_game(self, k, ref: str) -> str | None:
        if ref.startswith("R"):
            return self.receipts.get(ref, {}).get("game")
        e = k.entry(ref)
        return next((g for g in self.games if e and g.split("-")[0] in e.content.lower()), None)

    def _fields(self, k, a) -> list[str]:
        fields = [str(f).strip().lower() for f in (a.get("fields") or []) if str(f).strip()][:8]
        k._need(all(KIND_NAME.fullmatch(f) for f in fields), "fields: short lowercase names")
        return fields

    def check_entry(self, kind: str, content: str):
        """An entry of an agent-defined kind must fill the kind's fields (its interface)."""
        spec = self.kinds.get(kind)
        if not spec or not spec.get("fields"):
            return
        text = content.lower()
        missing = [f for f in spec["fields"] if not (re.search(rf"(^|\n)\s*{f}\s*:", text) or f'"{f}"' in text)]
        if missing:
            raise ActionError(f"a '{kind}' entry needs the fields {', '.join(spec['fields'])} (one line 'field: value' "
                              f"each); missing {', '.join(missing)}")

    def agreed_kinds(self) -> list[str]:
        return [n for n, s in self.kinds.items() if len(s.get("adopters", [])) >= self.rules["substrate"]["agree"]]

    def _contract(self, k, aid, a):
        op = str(a.get("op") or "")
        if op == "offer":
            to = str(a.get("to") or "").strip().upper()
            k._need(to in k.agents and to != aid, "offer to another agent")
            job = str(a.get("job") or "").strip()
            k._need(job, "describe the job")
            pay = round(float(a.get("pay") or 0), 2)
            k._need(0 < pay <= self.credit[aid], f"pay must be positive and at most your credit ({self.credit[aid]:.1f})")
            cid = k.new_id("C")
            self.contracts[cid] = {"id": cid, "from": aid, "to": to, "job": clip(job, 300), "pay": pay, "status": "offered",
                                   "t": k.tick}
            k.notify(to, aid, f"offers you contract {cid}: {clip(job, 160)} for {pay:g} credit (contract op=accept id={cid})")
            return "lab.contract", f"offered {cid} to {to}", cid
        cid = str(a.get("id") or "").strip().upper()
        c = self.contracts.get(cid)
        k._need(c, f"no contract {cid or '?'}")
        if op in ("accept", "decline"):
            k._need(c["to"] == aid and c["status"] == "offered", f"{cid} is not an open offer to you")
            if op == "decline":
                c["status"] = "declined"
                k.notify(c["from"], aid, f"declined {cid}")
                return "lab.contract", f"declined {cid}", cid
            k._need(self.credit[c["from"]] >= c["pay"], f"{c['from']} no longer has {c['pay']:g} credit")
            self.credit[c["from"]] -= c["pay"]
            c["status"], c["accepted"] = "accepted", k.tick
            k.notify(c["from"], aid, f"accepted {cid}; the pay is held until you mark it done")
            return "lab.contract", f"accepted {cid}: {c['job']} (pay {c['pay']:g} held)", cid
        k._need(c["from"] == aid, f"only {c['from']} can do that with {cid}")
        if op == "done":
            k._need(c["status"] == "accepted", f"{cid} is not an accepted job")
            c["status"] = "paid"
            self._award(k, c["to"], c["pay"], f"paid by {aid} for {cid}", share=False)
            k.notify(c["to"], aid, f"released the pay of {cid}: {c['pay']:g} credit")
            return "lab.contract", f"paid {c['to']} {c['pay']:g} for {cid}", cid
        k._need(op == "cancel" and c["status"] in ("offered", "accepted"), f"{cid} cannot be cancelled")
        if c["status"] == "accepted":
            self.credit[aid] += c["pay"]
        c["status"] = "cancelled"
        k.notify(c["to"], aid, f"cancelled {cid}")
        return "lab.contract", f"cancelled {cid}", cid

    def set_rules(self, k, rules: dict, why: str = ""):
        self.rules = clean_rules(rules)
        self.rules_version += 1
        self.rules_t = k.tick
        k.rank = dict(self.rules["substrate"]["rank"])
        k.post("global", "lab", "decision", f"RULES v{self.rules_version} from t{k.tick}: {rules_text(self.rules, self.model_on, self.hyp_on)}"
               + (f" Why: {clip(why, 400)}" if why else ""), notify=False)

    def kind_stats(self, k) -> dict:
        posts, reads = Counter(), Counter()
        for e in k.index.values():
            if e.author != "lab":
                posts[e.kind] += 1
        for seen in k.retrieved.values():
            for eid in seen:
                e = k.entry(eid)
                if e and e.author != "lab":
                    reads[e.kind] += 1
        return {kind: {"posts": posts[kind], "reads": reads[kind], "reused_by_first_solvers": self.kind_reuse[kind]}
                for kind in set(posts) | set(reads)}

    # ------------------------------------------------------------------ views

    def explore_observe(self, k, aid: str) -> str:
        L = []
        if self.delegation:
            r = self.roles.get(aid)
            if r and k.groups[r["team"]].dissolved is None:
                L.append(f"YOUR ROLE (from {r['by']}, lead of {r['team']}, t{r['t']}): {r['role'] or '-'}; job: {r['job'] or '-'}")
            led = self._led(k, aid)
            if led:
                rows = []
                for c in self.children(k, led)[:10]:
                    rr, rep = self.roles.get(c), self.reports.get(c)
                    rows.append(f"  {c}: " + (f"role {clip(rr['role'], 60)}; job {clip(rr['job'], 80)}" if rr else "no role yet")
                                + (f"; report t{rep['t']}: {clip(rep['text'], 160)}" if rep else "; no report yet"))
                L.append(f"YOUR TEAM {led} ({len(k.agents_inside(led))} agents"
                         + (f", sub-team of {self.teams[led]['parent']}" if self.teams[led]["parent"] else "")
                         + f"; tracker {self.trackers.get(led, 'none')}); your direct children:" + ("\n" + "\n".join(rows) if rows else " none yet"))
            for g in sorted(k.containers(aid)):
                if g in self.pins and k.groups[g].dissolved is None:
                    p = self.pins[g]
                    L.append(f"PINNED in {g} by {p['by']} (t{p['t']}): {p['text']}")
        if self.contracts_on:
            mine = [c for c in self.contracts.values() if aid in (c["from"], c["to"]) and c["status"] in ("offered", "accepted")]
            if mine:
                L.append("YOUR CONTRACTS: " + "; ".join(f"{c['id']} {c['from']}->{c['to']} {c['status']} pay {c['pay']:g}: "
                                                        f"{clip(c['job'], 80)}" for c in mine[:5]))
        if not self.explore:
            return "\n".join(L)
        show, n = self.rules["substrate"]["show"], self.rules["substrate"]["items"]
        mine = self.assigned.get(aid)
        if "problems" in show:
            rows = []
            for g in self.games:
                at = [x for (x, gg), st in self.state.items() if gg == g and self.assigned.get(x) == g and st["levels"] == self.best[g]]
                rows.append(f"  {g.split('-')[0]} level {self._problem(g)}: {len(at)} copies there, "
                            f"{len(self.open_plans(k, g))} open plans" + (" <- yours" if g == mine else ""))
            L.append("OPEN PROBLEMS (each game's next unsolved level):\n" + "\n".join(rows))
        view = self.directions_view(k, aid)
        if view:
            L.append(view)
        if mine:
            level = self._problem(mine)
            plans = self.open_plans(k, mine)
            if "plans" in show and self.hyp_on:  # every idea on the level and what came of it, open plans among them
                view = self.ideas_view(k, mine, level, aid=aid)
                if view:
                    L.append(view)
            elif "plans" in show and plans:
                L.append(f"OPEN PLANS on {mine.split('-')[0]} level {level}:\n" +
                         "\n".join(f"  {p['id']} {p['agent']} until t{p['until']}: {clip(p['hypothesis'], 140)}"
                                   for p in [p for p in plans if k.seen_by(aid, p.get("entry"))][-n:]))
            here = [f for f in self.findings if f["game"] == mine and f["level"] == level and k.seen_by(aid, f["id"])]
            for view, label, pick in (("findings", "FINDINGS", lambda f: f["outcome"] != "dead_end"),
                                      ("dead_ends", "DEAD ENDS", lambda f: f["outcome"] == "dead_end")):
                if self.board_on and view in show and any(pick(f) for f in here):  # all of them, newest first
                    rows = [f for f in here if pick(f)][::-1]
                    L.append(f"{label} on {mine.split('-')[0]} level {level}, all {len(rows)}, newest first:\n" + "\n".join(
                        fit_rows([(f"  {f['id']} {f['agent']}{'' if f['backed'] else ' (claim)'}"
                                   + (f" (happened {f['repeats']}x)" if f.get("repeats") else "")
                                   + (" CONTESTED by " + ", ".join(c["by"] for c in f["challenges"]) if f.get("contested") else "")
                                   + (f" (revised as {f['revised_by']})" if f.get("revised_by") else "")
                                   + self.trust_tag(k, f["id"]), f["text"],
                                   self._used_by(k, f)) for f in rows], 3000, steps=(160, 60, 0))))
                    continue
                rows = [f for f in here if pick(f)][-n:]
                if view in show and rows:
                    L.append(f"{label} on {mine.split('-')[0]} level {level}: " + "; ".join(
                        f"{f['id']} {f['agent']}{'' if f['backed'] else ' (claim)'}"
                        + (f" (happened {f['repeats']}x)" if f.get("repeats") else "")
                        + (" CONTESTED by " + ", ".join(c["by"] for c in f["challenges"]) if f.get("contested") else "")
                        + (f" (builds on contested {f['built_on_contested']})" if f.get("built_on_contested") else "")
                        + (f" (revised as {f['revised_by']})" if f.get("revised_by") else "") + self.trust_tag(k, f["id"])
                        + f": {f['text']}" for f in rows))
            if self.board_on and "findings" in show:
                unused = [f for f in self.findings if f["game"] == mine and f["level"] <= level and f["backed"]
                          and f["outcome"] != "dead_end" and f["agent"] != "lab" and f["t"] < k.tick and k.seen_by(aid, f["id"])
                          and not f.get("superseded_at") and not f.get("revised_by") and not self._users(k, f)]
                if unused:
                    L.append(f"BACKED FINDINGS NOBODY ELSE HAS USED YET on {mine.split('-')[0]} ({len(unused)}; combine "
                             f"them, claim builds_on=[E-ids], or test whether they matter here): " + "; ".join(
                                 f"{f['id']} L{f['level']} {f['agent']}{self.trust_tag(k, f['id'], rests=False)}: "
                                 f"{clip(f['text'], 70)}" for f in unused[::-1]))
            expired = [p for p in self.plans.values() if p["game"] == mine and p["level"] == level and p["status"] == "open"
                       and p["until"] < k.tick and not p.get("finding") and k.seen_by(aid, p.get("entry"))][-n:]
            if "plans" in show and expired and not self.hyp_on:  # plans that ran out without a result can be taken over
                L.append("EXPIRED WITHOUT A RESULT (free to take over): " + "; ".join(
                    f"{p['id']} {p['agent']}: {clip(p['hypothesis'], 90)}" for p in expired))
        if mine and self.guide_on and "guide" in show:
            cur = self.guides[mine][-1]
            L.append(f"GUIDE to {mine.split('-')[0]} v{cur['v']}" + (f" (by {cur['author']}, t{cur['t']})" if cur["v"] else "")
                     + f", the society's understanding of this game across levels (guide op=add to add a line with "
                     f"its receipts, op=merge to combine lines into one, op=challenge to contest one, op=edit "
                     f"base={cur['v']} to revise):\n" + (self.guide_text(mine, fit=GUIDE_MAX, k=k, aid=aid) if cur["text"] else "(empty)"))
            if self._own_guide(mine):
                me = self.island_key(k, aid)
                others = Counter(self.guide_island[mine][ln] for ln in cur["lines"] if ln in self.guide_island[mine]
                                 and self.guide_island[mine][ln] != me and self._line_seen(k, aid, mine, ln))
                L.append("OTHER ISLANDS' GUIDES (open lines; guide op=view island=G1 or A5 shows one): "
                         + (", ".join(f"{x} {n} lines" for x, n in others.most_common(8)) if others else "none open yet"))
                meet = self.islands_meet(k, aid, mine)
                if meet:
                    L.append("WHERE ISLANDS' GUIDES MEET (lines of different islands on the same subject; do they agree? a "
                             "receipt settles which holds):\n" + "\n".join(meet))
        if self.lessons_on and "guide" in show:
            cur = self.guides["lessons"][-1]
            L.append(f"LESSONS shared by all games v{cur['v']}, what one game taught that may hold in others (guide op=add "
                     f"game=lessons to add one with its receipts):\n"
                     + (self.guide_text("lessons", fit=GUIDE_MAX // 2, k=k, aid=aid) if cur["text"] else "(empty)"))
        if mine and self.carry and "carried" in show:
            level = self._problem(mine)
            carried = [f for f in self.findings if f["game"] == mine and f["level"] < level and f["backed"]
                       and not f.get("superseded_at") and not f.get("revised_by") and k.seen_by(aid, f["id"])]
            if carried and self.board_on:
                carried = carried[::-1]
                L.append(f"CARRIED FORWARD from earlier levels of {mine.split('-')[0]}, all {len(carried)}, newest first "
                         f"(they hold at level {level} until a receipt there contradicts them; challenge one with such a "
                         f"receipt to retire it here):\n" + "\n".join(fit_rows(
                             [(f"  {f['id']} L{f['level']} {f['agent']}{' DEAD END' if f['outcome'] == 'dead_end' else ''}"
                               + (" CONTESTED" if f.get("contested") else "") + self.trust_tag(k, f["id"]), f["text"],
                               self._used_by(k, f)) for f in carried], 2500, steps=(160, 60, 0))))
                carried = []
            carried = carried[-n:]
            if carried:
                L.append(f"CARRIED FORWARD from earlier levels of {mine.split('-')[0]} (they hold at level {level} until a "
                         f"receipt there contradicts them; challenge one with such a receipt to retire it here): " + "; ".join(
                             f"{f['id']} L{f['level']} {f['agent']}{' DEAD END' if f['outcome'] == 'dead_end' else ''}"
                             + (" CONTESTED" if f.get("contested") else "") + self.trust_tag(k, f["id"])
                             + f": {f['text']}" for f in carried))
        if mine and self.model_on and "model" in show:
            cur = self.models.current(mine)
            got, total = self.models.coverage(mine)
            here = [c for c in self.models.transitions[mine] if c["level"] == self._problem(mine)]
            ok_here = sum(self.models.results.get((mine, cur["v"]), {}).get(c["id"], False) for c in here)
            L.append(f"SHARED MODEL of {mine.split('-')[0]} v{cur['v']} (by {cur['author']}, t{cur['t']}): explains {got} of "
                     f"{total} recorded moves, {ok_here} of {len(here)} at level {self._problem(mine)} (checked at the end of "
                     f"each tick). model op=view shows its code and what it gets wrong; propose a change on v{cur['v']}.")
        if mine and "untried" in show:
            line = self.untried_line(aid, mine)
            if line:
                L.append(line)
        if "interfaces" in show:
            for name in self.agreed_kinds()[:3]:
                spec = self.kinds[name]
                rows = [e for e in k.index.values() if e.kind == name and k.can_read(aid, e.board) and k.visible(aid, e)][-n:]
                L.append(f"AGREED KIND '{name}' (fields {', '.join(spec['fields']) or 'free'}; {len(spec['adopters'])} "
                         f"adopters)" + (": " + "; ".join(k.header(e) for e in rows) if rows else ": no entries yet"))
        if "runs" in show and mine:
            line = self.runs_line(k, aid, mine, n)
            if line:
                L.append(line)
        if "kinds" in show or self.rules["substrate"]["promoted_kinds"]:
            for kind in self.rules["substrate"]["promoted_kinds"]:
                rows = [e for e in k.index.values() if e.kind == kind and k.can_read(aid, e.board) and k.visible(aid, e)][-n:]
                if rows:
                    L.append(f"LATEST '{kind}' ENTRIES: " + "; ".join(k.header(e) for e in rows))
        mine_plan = next((p for p in self.plans.values() if p["agent"] == aid and p["status"] == "open"), None)
        steps = self.steps_allowed(aid, k.tick)
        L.append(f"YOUR CREDIT {self.credit[aid]:.1f} (society total {sum(self.credit.values()):.1f}); steps this tick "
                 f"{steps}" + (f" ({self.steps_per_tick} + {steps - self.steps_per_tick} for recent credit)"
                               if steps > self.steps_per_tick else "")
                 + (f"; your open plan {mine_plan['id']} until t{mine_plan['until']}" if mine_plan else "; no open plan")
                 + f". Rules v{self.rules_version}" + (f" (revised t{self.rules_t})" if self.rules_version > 1 else "") + ".")
        paid = sorted(self.combo_credit.items(), key=lambda x: -x[1][0])[:n]
        if "paid" in show and paid:
            L.append("WHAT HAS PAID (credit earned by agents who, that tick or the one before, used these actions or read these "
                     "kinds; an association, not a proof): " + "; ".join(f"{c} {v[0]:.1f} ({v[1]}x)" for c, v in paid))
        unused = self.unused_actions(k)
        if "unused" in show and unused:
            L.append("ACTIONS NOBODY HAS USED YET: " + ", ".join(unused))
        return "\n".join(L)

    # ------------------------------------------------------------------ the game guide (guide=1)

    def guide_text(self, game: str, authors: bool = False, fit: int | None = None, k=None, aid: str | None = None) -> str:
        """The current guide, numbered, each line marked with its receipts and any counterexample against it. With fit,
        the oldest lines are shown shortened (then, if still too long, left out) until it fits; none is removed. With
        islands, lines another island added and that are not open yet are left out of aid's view (numbers are kept)."""
        cur, ev, bad = self.guides[game][-1], self.guide_evidence[game], self.guide_contested[game]
        rows = []
        for i, ln in enumerate((x.strip() for x in cur["text"].splitlines() if x.strip()), 1):
            if k is not None and aid and not self._line_seen(k, aid, game, ln):
                continue
            if k is not None and aid and self._own_guide(game) and self.guide_island[game].get(ln, self.island_key(k, aid)) \
                    != self.island_key(k, aid):
                continue  # another island's line: listed under OTHER ISLANDS' GUIDES
            mark = (f" [{', '.join(ev[ln][:3])}]" if ev.get(ln) else "") + (
                f" [CONTESTED by {bad[ln]['by']}: {', '.join(bad[ln]['evidence'][:2])}: {clip(bad[ln]['why'], 80)}]"
                if ln in bad else "")
            engine = self.evidence_marks(game, self._guide_level(game), ln)
            mark += f" [engine records: {engine}]" if engine else ""
            eid = self.guide_ids[game].get(ln) if self.board_on else None
            if self.trust_on and k is not None and self.guide_ids[game].get(ln):
                mark += self.trust_tag(k, self.guide_ids[game][ln])
            rows.append([f"{i}. " + (f"({eid}) " if eid else "") + (f"[{cur['lines'].get(ln, '?')}] " if authors else ""),
                         ln, mark])
        size = lambda: sum(len(a) + len(b) + len(c) + 1 for a, b, c in rows)
        if self.board_on and k is not None and fit:  # ranked by use: the least used lines are shortened first, none hidden
            use = lambda i: (k.cite_weight(self.guide_ids[game].get(rows[i][1], "")), i)
            order = sorted(range(len(rows)), key=use)
            for step in (GUIDE_SHORT, 40, 0):
                for i in order:
                    if size() <= fit:
                        break
                    rows[i][1] = clip(rows[i][1], step) if step else "(shortened: guide op=view)"
            return (f"(over {fit} characters: the least used lines are shortened; guide op=view shows every line whole, "
                    f"op=merge combines lines; cite a line by its E-id when you use it)\n" if size() > fit or any(
                        r[1].endswith("…") or r[1].startswith("(shortened") for r in rows) else "") + "\n".join(
                a + b + c for a, b, c in rows)
        shortened = hidden = 0
        for row in rows:
            if not fit or size() <= fit:
                break
            if len(row[1]) > GUIDE_SHORT:
                row[1], shortened = row[1][:GUIDE_SHORT] + "...", shortened + 1
        while fit and rows and size() > fit:
            rows.pop(0)
            hidden += 1
        head = ""
        if shortened or hidden:
            head = (f"(over {fit} characters: " + ", ".join(x for x in (
                f"{shortened} older lines shortened" if shortened else "", f"the {hidden} oldest not shown" if hidden else "")
                if x) + "; guide op=view shows every line whole, op=merge combines lines)\n")
        return head + "\n".join(a + b + c for a, b, c in rows)

    def island_key(self, k, aid: str) -> str:
        """The island an agent writes for: its lowest-numbered group, or itself when it is in no group."""
        groups = sorted(k.member_of.get(aid, ()), key=lambda g: int(g[1:]))
        return groups[0] if groups else aid

    def _own_guide(self, game: str) -> bool:
        return self.island_guides and game != "lessons"

    def islands_meet(self, k, aid: str, game: str, n: int = 4) -> list[str]:
        """Open lines of different islands' guides on the same subject (word overlap >= 0.33), most alike first."""
        lines = [ln for ln in self.guide_island[game] if ln in self.guides[game][-1]["lines"] and self._line_seen(k, aid, game, ln)]
        pairs = []
        for i, a in enumerate(lines):
            for b in lines[i + 1:]:
                if self.guide_island[game][a] != self.guide_island[game][b]:
                    sim = same_idea(idea_words(a), idea_words(b))
                    if sim >= 0.33:
                        pairs.append((sim, a, b))
        out = []
        for sim, a, b in sorted(pairs, reverse=True)[:n]:
            out.append(f"  {self.guide_island[game][a]} ({self.guide_ids[game].get(a, '?')}): {clip(a, 150)}\n"
                       f"  {self.guide_island[game][b]} ({self.guide_ids[game].get(b, '?')}): {clip(b, 150)}")
        return out

    def _line_seen(self, k, aid: str, game: str, line: str) -> bool:
        return k.seen_by(aid, self.guide_ids[game].get(line))

    def _guide_level(self, game: str) -> int:
        return self._problem(game) if game in self.best else 0  # the lessons guide belongs to no level

    def _guide(self, k, aid, a):
        k._need(self.guide_on, "there is no guide in this society")
        ref = str(a.get("game") or "").strip().lower()
        game = "lessons" if self.lessons_on and ref in ("lessons", "all") else \
            self._game_ref(ref) if ref else self.assigned.get(aid)
        k._need(game in self.guides, "which game?" + (" (or game=lessons)" if self.lessons_on else ""))
        short, versions = game.split("-")[0], self.guides[game]
        cur = versions[-1]
        if str(a.get("op") or "view") == "view" and a.get("island") and self._own_guide(game):
            isl = str(a["island"]).strip().upper()
            rows = [f"{i}. ({self.guide_ids[game].get(ln, '?')}) {ln}" for i, ln in enumerate(
                (x.strip() for x in cur["text"].splitlines() if x.strip()), 1)
                if self.guide_island[game].get(ln) == isl and self._line_seen(k, aid, game, ln)]
            return "lab.guide", (f"{isl}'s guide to {short} (lines open to you):\n" + ("\n".join(rows) or "(none open)")), short
        if str(a.get("op") or "view") == "view":
            hist = "; ".join(f"v{v['v']} {v['author']} t{v['t']}: {clip(v['why'], 60)}" for v in versions[1:][-5:]) or "none yet"
            return "lab.guide", (f"GUIDE to {short} v{cur['v']} (edits: {hist}). Each line with its author, receipts and "
                                 f"any counterexample:\n"
                                 + (self.guide_text(game, authors=True, k=k, aid=aid) or "(empty: write what you understand about this "
                                                                           "game)")), short
        if str(a.get("op")) == "add":  # a compatible contribution: combined with the current version as it stands
            line = " ".join(str(a.get("line") or "").split())
            k._need(line, "line: what to add")
            k._need(len(line) <= GUIDE_LINE, f"a line is at most {GUIDE_LINE} characters: split it into two lines")
            k._need(line not in cur["lines"], "that line is already in the guide")
            text = (cur["text"] + "\n" + line).strip()
            k._need(self.board_on or len(text) <= GUIDE_STORE, f"the guide holds at most {GUIDE_STORE} characters: merge lines first")
            ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
            k._need(all(r in self.receipts for r in ev), "evidence: receipt ids (R-ids) from play")
            if ev:
                self.guide_evidence[game][line] = ev
            versions.append({"v": cur["v"] + 1, "author": aid, "t": k.tick, "text": text, "why": "add: " + clip(line, 120),
                             "lines": {**cur["lines"], line: aid}, "removed": [], "evidence": ev})
            e = k.post("global", aid, "result", f"GUIDE {short} v{cur['v'] + 1} by {aid}, added: {line}", refs=ev, notify=False)
            self.entry_meta[e.id] = {"game": game, "level": self._guide_level(game), "type": "guide"}
            self.guide_ids[game][line] = e.id
            self.guide_island[game][line] = self.island_key(k, aid)
            engine = self.evidence_marks(game, self._guide_level(game), line)
            return "lab.guide", (f"added to the {short} guide, now v{cur['v'] + 1} ({e.id})"
                                 + (f". Engine records on the moves it names: {engine}" if engine else "") + (
                f"; it is over {GUIDE_MAX} characters, so everyone sees its least used lines shortened (cite lines by E-id "
                f"when you use them; op=merge combines lines)" if len(text) > GUIDE_MAX and self.board_on else
                f"; it is over {GUIDE_MAX} characters, so everyone sees its oldest lines shortened until lines are merged "
                f"(op=merge)" if len(text) > GUIDE_MAX else "")), short
        if str(a.get("op")) == "merge":  # combining pieces: one line keeps what several said, without claiming any was wrong
            lines = [x.strip() for x in cur["text"].splitlines() if x.strip()]
            try:
                nums = sorted({int(x) for x in (a.get("lines") or [])})
            except (TypeError, ValueError):
                nums = []
            k._need(len(nums) >= 2 and all(0 < n <= len(lines) for n in nums),
                    f"lines: the numbers of two or more lines to combine (1-{len(lines)}, as numbered in the guide)")
            line = " ".join(str(a.get("line") or "").split())
            old = [lines[n - 1] for n in nums]
            k._need(all(self._line_seen(k, aid, game, x) for x in old), "you can only merge lines that are open to you")
            k._need(line, "line: the combined line, keeping what the merged lines say")
            k._need(len(line) <= GUIDE_LINE, f"a line is at most {GUIDE_LINE} characters")
            k._need(len(line) < sum(map(len, old)), "the combined line must be shorter than the lines it replaces together")
            k._need(line not in cur["lines"] or line in old, "that line is already in the guide")
            ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
            k._need(all(r in self.receipts for r in ev), "evidence: receipt ids (R-ids) from play")
            who = []
            for x in old:
                ev += [r for r in self.guide_evidence[game].get(x, []) if r not in ev]
                who += [y for y in cur["lines"].get(x, "?").split(",") if y not in who]
            who += [aid] if aid not in who else []  # combining is work too
            if ev:
                self.guide_evidence[game][line] = ev
            kept = [line if i == nums[0] else x for i, x in enumerate(lines, 1) if i == nums[0] or i not in nums]
            versions.append({"v": cur["v"] + 1, "author": aid, "t": k.tick, "text": "\n".join(kept),
                             "why": "merge: " + clip(str(a.get("why") or line), 120),
                             "lines": {**{x: cur["lines"].get(x, aid) for x in kept if x != line}, line: ",".join(who)},
                             "removed": [], "merged": old, "evidence": ev})
            parts = [self.guide_ids[game][x] for x in old if self.guide_ids[game].get(x)]
            e = k.post("global", aid, "result", f"GUIDE {short} v{cur['v'] + 1} by {aid}, merged lines "
                       f"{', '.join(map(str, nums))}" + (f" ({', '.join(parts)})" if parts else "") + f" into line {nums[0]}: "
                       f"{line}", refs=parts + [r for r in ev if r not in parts], notify=False)
            self.entry_meta[e.id] = {"game": game, "level": self._guide_level(game), "type": "guide"}
            self.guide_ids[game][line] = e.id
            self.guide_island[game][line] = self.island_key(k, aid)
            if parts and self.hyp_on:
                self.parts[e.id] = parts
            told = {y for y in who if y in k.agents} | {self.guide_contested[game][x]["by"] for x in old
                                                        if x in self.guide_contested[game]}
            for y in sorted(t for t in told - {aid} if k.seen_by(t, e.id)):
                k.notify(y, aid, f"merged {len(old)} {short} guide lines, among them yours or one you contested, into "
                         f"line {nums[0]}: \"{clip(line, 100)}\" ({e.id}); if it lost something, add it back or challenge it")
            return "lab.guide", f"merged {len(old)} lines into line {nums[0]}; the {short} guide is now v{cur['v'] + 1} ({e.id})", short
        if str(a.get("op")) == "challenge":  # evidence decides: a counterexample marks the line for everyone
            ref = " ".join(str(a.get("line") or "").split())
            lines = [x.strip() for x in cur["text"].splitlines() if x.strip()]
            line = lines[int(ref) - 1] if ref.isdigit() and 0 < int(ref) <= len(lines) else (ref if ref in lines else None)
            k._need(line and self._line_seen(k, aid, game, line), "line: the line you contest (its text, or its number in "
                    "guide op=view), one that is open to you")
            who = cur["lines"].get(line, "?")
            k._need(who != aid, "it is your line: edit it instead")
            ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
            k._need(ev and all(r in self.receipts for r in ev), "a challenge needs the receipts of your counterexample")
            why = clip(str(a.get("why") or ""), 300)
            self.guide_contested[game][line] = {"by": aid, "evidence": ev, "why": why, "t": k.tick}
            e = k.post("global", aid, "critique", f"GUIDE {short}: CHALLENGE to the line \"{clip(line, 120)}\" ({who}): "
                       f"{why}\nEvidence: " + "; ".join(self.receipt_line(r) for r in ev[:3]), refs=ev, notify=False)
            self.entry_meta[e.id] = {"game": game, "level": self._guide_level(game), "type": "guide"}
            for y in who.split(","):
                if y in k.agents and y != aid and k.seen_by(y, e.id):
                    k.notify(y, aid, f"challenged your {short} guide line \"{clip(line, 80)}\" with {', '.join(ev)} "
                             f"({e.id}); revise or remove it with guide op=edit")
            return "lab.guide", f"the line is marked contested ({e.id})" + self.pay_change(
                k, aid, self.guide_ids[game].get(line), "counterexample to a guide line"), short
        k._need(str(a.get("op")) == "edit", "op must be view, add, merge, challenge or edit")
        k._need(a.get("base") is not None, f"base: the version you revised (now v{cur['v']})")
        k._need(int(a["base"]) == cur["v"], f"the guide changed underneath: it is v{cur['v']} now (by {cur['author']}: "
                f"{clip(cur['why'], 120)}). View it, take its changes into yours, and edit again on v{cur['v']}.")
        text = str(a.get("text") or "").strip()
        k._need(text, "text: the whole new guide")
        k._need(len(text) <= GUIDE_STORE, f"the guide must stay under {GUIDE_STORE} characters: keep it compact")
        new_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        unseen = [ln for ln in cur["lines"] if ln not in new_lines and (not self._line_seen(k, aid, game, ln) or (
            self._own_guide(game) and self.guide_island[game].get(ln, self.island_key(k, aid)) != self.island_key(k, aid)))]
        if unseen:  # islands: an edit cannot remove what the editor cannot see; those lines stay as they are
            new_lines += unseen
            text = "\n".join(new_lines)
        removed = [ln for ln, who in cur["lines"].items() if ln not in new_lines and who != aid]
        ev = [str(r).strip().upper() for r in (a.get("evidence") or []) if str(r).strip()]
        k._need(not removed or (ev and all(r in self.receipts for r in ev)),
                f"you removed {len(removed)} line(s) others wrote (e.g. '{clip(removed[0], 80) if removed else ''}'): keep "
                f"them, or give receipts (evidence) showing they are wrong")
        lines = {ln: cur["lines"].get(ln, aid) for ln in new_lines}
        versions.append({"v": cur["v"] + 1, "author": aid, "t": k.tick, "text": text,
                         "why": clip(str(a.get("why") or ""), 300), "lines": lines, "removed": removed, "evidence": ev})
        e = k.post("global", aid, "result", f"GUIDE {short} v{cur['v'] + 1} by {aid}: {clip(str(a.get('why') or ''), 300)}"
                   + (f" (removed {len(removed)} line(s), evidence {', '.join(ev)})" if removed else ""), refs=ev, notify=False)
        self.entry_meta[e.id] = {"game": game, "level": self._guide_level(game), "type": "guide"}
        for ln in new_lines:
            self.guide_ids[game].setdefault(ln, e.id)
        for who in sorted({y for ln in removed for y in cur["lines"][ln].split(",")} - {aid}):
            if who in k.agents and k.seen_by(who, e.id):
                k.notify(who, aid, f"removed your line(s) from the {short} guide with evidence {', '.join(ev)} ({e.id})")
        return "lab.guide", f"guide to {short} is now v{cur['v'] + 1} ({e.id})", short

    # ------------------------------------------------------------------ the shared model (model=1)

    async def _model(self, k, aid, a):
        k._need(self.model_on, "there is no shared model in this society")
        game = self._game_ref(a.get("game")) if a.get("game") else self.assigned.get(aid)
        k._need(game in self.models.versions, "which game?")
        m, short, op = self.models, game.split("-")[0], str(a.get("op") or "view")
        cur = m.current(game)
        if op == "case":
            ref = str(a.get("id") or "").strip().upper().lstrip("T")
            k._need(ref.isdigit() and int(ref) < len(m.transitions[game]), "id: a transition id, e.g. T12")
            return "lab.model", m.case_text(game, int(ref)), short
        if op == "view":
            got = await m.explained(game, cur["v"])
            wrong = [c for c in m.transitions[game] if not got.get(c["id"])]
            wrong.sort(key=lambda c: (-(c["level"] == self._problem(game)), -c["n"], -c["t"]))
            res = await m.run(cur["code"], wrong[:4], detail=4, game=game)
            why = {d["i"]: d["msg"] for d in res.get("details", [])}
            n, total = m.coverage(game)
            hist = "; ".join(f"v{h['v']} {h['author']} t{h['t']} +{h['gain']}" for h in m.versions[game][1:][-6:]) or "none yet"
            rows = "\n".join(f"- {m.describe(c)}\n  the model: {why.get(i, 'wrong')}" for i, c in enumerate(wrong[:4]))
            note = f" Note: {m.mask_note[game]}." if m.mask_note[game] else ""
            return "lab.model", (f"SHARED MODEL of {short} v{cur['v']} by {cur['author']} (t{cur['t']}; merges: {hist}). "
                                 f"Explains {n} of {total} recorded moves.{note}" + (f" Some it gets wrong (case shows the frames):\n{rows}" if rows else "")
                                 + f"\nCODE v{cur['v']}:\n{cur['code']}"), short
        k._need(op in ("test", "propose"), "op must be view, case, test or propose")
        code = str(a.get("code") or "")
        bad = check(code)
        k._need(not bad, f"the code is not allowed: {bad}")
        if op == "propose":
            k._need(a.get("base") is not None, f"base: the version your code was written on (now v{cur['v']})")
            k._need(int(a["base"]) == cur["v"], f"the model changed underneath: it is v{cur['v']} now (by {cur['author']}: "
                    f"{clip(cur['why'], 120)}). View it, take its changes into your code, and propose again on v{cur['v']}.")
        old = dict(await m.explained(game, cur["v"]))
        res = await m.run(code, m.transitions[game], detail=3, game=game)
        if res.get("error"):
            raise ActionError(res["error"])
        new = {c["id"]: ok for c, ok in zip(m.transitions[game], res["ok"])}
        broken = [m.transitions[game][t] for t, ok in old.items() if ok and not new.get(t)]
        gained = [m.transitions[game][t] for t, ok in new.items() if ok and not old.get(t)]
        total = len(m.transitions[game])
        report = (f"your code explains {sum(new.values())} of {total} recorded moves (v{cur['v']}: {sum(old.values())}); "
                  f"it breaks {len(broken)} the current version explains and adds {len(gained)}.")
        if broken:
            report += " Broken, e.g.: " + "; ".join(m.describe(c) for c in broken[:2])
        if op == "test":
            return "lab.model", "TEST " + report, short
        k._need(not broken, "not merged: " + report + " A merge must keep everything that already works.")
        k._need(gained, "not merged: " + report + " A merge must explain at least one more move.")
        latest = m.current(game)
        k._need(latest["v"] == cur["v"], f"the model changed while yours was checked: it is v{latest['v']} now (by "
                f"{latest['author']}). Take in its changes and propose again on v{latest['v']}.")
        v = cur["v"] + 1
        m.versions[game].append({"v": v, "author": aid, "t": k.tick, "code": code, "why": clip(str(a.get("why") or ""), 300),
                                 "parent": cur["v"], "gain": len(gained)})
        m.results[(game, v)] = new
        e = k.post("global", aid, "result", f"MODEL {short} v{v} by {aid}: explains {sum(new.values())} of {total} recorded "
                   f"moves (+{len(gained)} over v{cur['v']}): {clip(str(a.get('why') or ''), 300)}", notify=False)
        self.entry_meta[e.id] = {"game": game, "level": self._problem(game), "type": "model"}
        return "lab.model", f"merged as v{v} ({e.id}): " + report, short

    async def model_tick_end(self, k):
        """Check the current and recent versions on this tick's moves, and pay authors for moves recorded after their
        merge that their version predicts and the version before it did not."""
        c = self.rules["credit"]
        for game in self.games:
            m = self.models
            await m.explained(game, m.current(game)["v"])
            for ver in m.versions[game][1:]:
                if ver["t"] < k.tick - 5 or ver["author"] not in k.agents:
                    continue
                mine, before = await m.explained(game, ver["v"]), await m.explained(game, ver["parent"])
                gain = [t for t in m.transitions[game] if t["t"] > ver["t"] and mine.get(t["id"])
                        and not before.get(t["id"]) and (game, ver["v"], t["id"]) not in m.credited]
                if not gain:
                    continue
                for t in gain:
                    m.credited.add((game, ver["v"], t["id"]))
                amount = min(c["model_cap"], c["model"] * len(gain))
                self.model_credit_log.append({"t": k.tick, "agent": ver["author"], "game": game.split("-")[0],
                                              "version": ver["v"], "moves": len(gain), "credit": amount})
                self._award(k, ver["author"], amount, f"model v{ver['v']} of {game.split('-')[0]} predicted {len(gain)} "
                            f"new moves", _depth=0)

    def model_score(self) -> dict:
        cov = []
        for g in self.games:
            n, total = self.models.coverage(g)
            cov.append(n / total if total else 0.0)
        return {"model_merges": sum(len(v) - 1 for v in self.models.versions.values()),
                "model_coverage": round(sum(cov) / len(cov), 4) if cov else 0.0}

    def record_of(self, aid: str) -> str:
        mine = [f for f in self.findings if f["agent"] == aid]
        games = Counter(f["game"].split("-")[0] for f in mine)
        firsts = [f"{x['game'].split('-')[0]} L{x['level']}" for x in self.firsts if x["agent"] == aid]
        parts = [f"credit {self.credit[aid]:.1f}"]
        if mine:
            parts.append(f"{len(mine)} findings ({', '.join(f'{g} {n}' for g, n in games.most_common(3))}), "
                         f"{sum(f.get('contested', False) for f in mine)} contested")
        if firsts:
            parts.append("first to complete " + ", ".join(firsts))
        return "; ".join(parts)

    def unused_actions(self, k) -> list[str]:
        used = {e["op"].split(".")[1] if e["op"].startswith("lab.") else e["op"].split(".")[0]
                for e in k.events if e["ok"] and e["op"] != "intent"}
        offered = [n for n in k.allowed if n != "design_action"] + list(self.tool_names)
        if "design_action" in k.allowed:
            offered.append("design")
        alias = {"substrate": ("define_kind", "adopt", "revise_kind", "propose"), "contract": ("contract",)}
        return [n for n in offered if n not in used and n not in ("reflect",) and not any(x in used for x in alias.get(n, ()))]

    def explore_brief(self) -> str:
        out = ""
        if self.explore:
            out += ("\n- Problems, plans and findings. The open problems are each game's next unsolved level; each turn shows "
                    "how many copies are at each, and for yours the open plans, findings, dead ends and what no copy has "
                    "tried. A plan lives on the substrate from claim to result: the claim says what you will test and what "
                    "result would show it; findings along the way and the result (works, dead end or unclear) cite the "
                    "receipts that show them. Others on the problem see each step and are told of new claims there, and "
                    "everyone with an open plan on a level is told when it is completed. focus moves you to another game "
                    "(your own copy). The society gains most when its members explore different things.\n"
                    "- Every play gets an engine receipt (R-id): the moves, from which level, and what changed. A finding "
                    "that cites receipts is shown as backed by engine records; one that cites none is a claim.\n"
                    f"- The rules now in force (v{self.rules_version}): {rules_text(self.rules, self.model_on, self.hyp_on)}. Repeating a path another copy "
                    "already took earns nothing. Each turn shows which actions and combinations have paid so far.\n"
                    "- substrate: define a new kind of entry anyone can write, or propose a change to these rules or to what "
                    "the substrate shows. play entry=<E-id or R-id> runs the moves written in an entry you can read or in "
                    "a receipt, filling placeholders such as {x} from args, so move sequences that work can be written, "
                    "cited, revised and reused; an entry's author receives a royalty out of what others earn running it. "
                    "A kind can carry fields, an interface its entries must fill, so independent "
                    "findings fit together; once enough agents adopt a kind it is agreed and shown every turn. The rules are "
                    "revised from proposals and from what has paid, and are judged only by the game score.")
            if "cycle" in self.rules["substrate"]["show"]:
                out += "\n- " + CYCLE
            if self.carry:
                out += ("\n- Nothing learned is lost when a level falls. Findings from earlier levels stay in view, carried "
                        "forward, until a receipt at the new level contradicts them (challenge with such a receipt retires "
                        "one there). Open plans move to the next level with their interim findings. A published solution "
                        "comes with the solver's findings and with explanations (plan op=explain): replay to reach the "
                        "level, and read why the moves work. A finding can name the findings it builds on; if one is "
                        "contested, yours is flagged.")
            if self.guide_on:
                out += ("\n- The guide: one short, durable text per game holding what the society understands about it, "
                        "kept across levels and shown every turn. Before a play, say what you expect to see; afterwards, "
                        "what changed, expected or not. Surprises are what the guide should be revised with. Add a line "
                        "with the receipts that show it; a counterexample marks someone's line contested until its "
                        "author revises it. Merge lines into one shorter line that keeps what they say: their authors "
                        "become its co-authors and are told, and the old lines stay in the history. Edit it on the "
                        "current version; your own lines are yours to change, and removing a line someone else wrote "
                        "needs receipts showing it is wrong. No line is refused for space: a long guide shows its oldest "
                        "lines shortened until they are merged.")
            if self.lessons_on:
                out += ("\n- Lessons: one more guide, shared by all games (guide game=lessons), for what one game taught "
                        "that may hold in others: kinds of goals, counters and budgets, how controls behave, ways of "
                        "probing that worked. Everyone sees it every turn.")
            if self.hyp_on:
                out += ("\n- Ideas: every claim on a level stays listed under IDEAS with what came of it (testing, never "
                        "tested, refuted, unclear, worked), so everyone sees which ideas nobody has tested and which "
                        "pieces nobody has combined. A claim close to an earlier one is pointed out, never refused. The "
                        "first to test a new idea on a level (plan done, with receipts from after the claim) earns credit; "
                        "testing an idea again earns none. Ideas found by different agents can be combined: name the "
                        "pieces in builds_on, and their authors are told. When a level falls, each piece its solver cites "
                        "(play cites, E-ids in the play's note, or the pieces its plan builds on) earns its author credit "
                        "for understanding, and the pieces that one was built on half as much.")
            if self.trust_on:
                noun = self.state_noun
                out += (f"\n- Test before you trust. Every finding, dead end, guide line and note shows what it can be "
                        f"trusted at, worked out by the engine from evidence, never by votes: said (no engine record behind "
                        f"it), seen in k {noun}s (receipts from k different starting {noun}s), reproduced (rerun with the "
                        f"same result, or a run by someone else that held), CONTESTED (a rerun failed, or a test that "
                        f"repeated its run failed, or a counterexample stands). It also shows its weakest support (rests on "
                        f"unbacked or contested E-ids), so a conclusion can be traced to what it assumes. Receipts say which "
                        f"{noun} a run started from. What a move does can depend on the {noun}: a run that does not repeat "
                        f"a result's moves from its starting {noun} does not refute it; it is counted as a failure under "
                        f"other conditions. "
                        f"plan op=test checks a piece before you build on it"
                        + (": with no evidence the engine reruns its receipts exactly from where they started; "
                           if self.supports_replay else ": run what it says on your own copy, then report result and "
                                                        "evidence; ")
                        + "a test that changes what a piece can be trusted at earns credit.")
            if self.model_on:
                out += ("\n- The shared model: the society builds one program per game that predicts what each move does "
                        "(model). Every move any copy makes is recorded, every version is checked against all of them, and a "
                        "change is merged only if it breaks nothing the current version explains and explains more. Pieces "
                        "found by different agents fit together there, and the next level starts from everything it "
                        "already explains.")
        if self.contracts_on:
            out += ("\n- contract: pay another agent credit for a job. The pay is held when the job is accepted and released "
                    "when you mark it done.")
        if self.delegation:
            out += ("\n- lead: like a CEO or PI, found a team (a group you direct) and keep playing yourself; others join it; a "
                    "member who founds a team creates a sub-team, so teams nest. A lead assigns each direct child (a member, "
                    "or the lead of a sub-team) a role and a job, which the child sees at the top of every turn; children "
                    "report up; a lead names a tracker, whose pinned note every team member sees each turn. A lead receives a "
                    "royalty out of what its children earn (the rate is in the rules).")
        return out + self.directions_brief()

    def explore_score(self, k) -> dict:
        if not (self.explore or self.delegation):
            return {}
        live = [g for g in self.teams if k.groups[g].dissolved is None]
        return {"plans": len(self.plans), "findings": len(self.findings),
                "dead_ends": sum(f["outcome"] == "dead_end" for f in self.findings),
                "backed_findings": sum(f["backed"] for f in self.findings),
                "switches": self.switches, "credit_total": round(sum(self.credit.values()), 2),
                "teams": len(live), "team_depth": self.team_depth(k), "roles": len(self.roles),
                "reports": len(self.reports), "pins": len(self.pins),
                "in_teams": sum(1 for a in k.agents if self._team_of(k, a)),
                "contracts": len(self.contracts), "contracts_paid": sum(c["status"] == "paid" for c in self.contracts.values()),
                "challenges": sum(len(f.get("challenges", [])) for f in self.findings),
                "revisions": sum(bool(f.get("revises")) for f in self.findings),
                "kinds_defined": len(self.kinds), "kinds_agreed": len(self.agreed_kinds()), "rules_version": self.rules_version,
                "entry_runs": sum(r["runs"] for r in self.entry_runs.values()),
                "entries_run_by_others": sum(1 for eid, r in self.entry_runs.items() if r["by"] - {self._entry_author(self._k, eid)}),
                "novel_states": sum(self.novel_count.values()),
                **({"plans_carried": self.plans_carried, "explanations": sum(map(len, self.explanations.values())),
                    "superseded": sum(bool(f.get("superseded_at")) for f in self.findings)} if self.carry else {}),
                **(self.model_score() if self.model_on else {}),
                **({"guide_edits": sum(len(v) - 1 for v in self.guides.values()),
                    "guide_chars": sum(len(v[-1]["text"]) for v in self.guides.values()),
                    "guide_merges": sum("merged" in x for v in self.guides.values() for x in v)} if self.guide_on else {}),
                **({"lesson_lines": len([x for x in self.guides["lessons"][-1]["text"].splitlines() if x.strip()])}
                   if self.lessons_on else {}),
                **(self.ideas_score(k) if self.hyp_on else {}),
                **(self.trust_score(k) if self.trust_on else {}), **self.directions_score(k)}

    def trust_score(self, k) -> dict:
        pieces = [f["id"] for f in self.findings if f["agent"] != "lab"]
        levels = Counter(self.trust_of(k, x)["level"] for x in pieces)
        tests = [t for v in self.tests.values() for t in v]
        return {"tests": len(tests), "tests_engine": sum(t["how"] == "replay" for t in tests),
                "tests_repeats": sum(t["how"] == "run" and t["same_state"] for t in tests),
                "tests_other_conditions": sum(t["how"] == "run" and not t["same_state"] for t in tests),
                "pieces_said": levels["said"], "pieces_seen": levels["seen"], "pieces_reproduced": levels["reproduced"],
                "pieces_contested": levels["contested"], "state_notes": getattr(self, "state_notes", 0)}

    def trust_export(self, k) -> dict:
        """Every piece with what it can be trusted at and what it rests on, for tracing a conclusion back to its
        assumptions after the run (tools/trace_claims.py)."""
        ids = [e.id for e in k.index.values() if e.author != "lab" and e.kind != "data"]
        out = {}
        for x in ids:
            t, d = self.trust_of(k, x), self.deps(k, x)
            if t["level"] == "said" and not d and x not in self.tests:
                continue
            out[x] = {"level": t["level"], "states": t["states"], "other": t["other"], "receipts": t["receipts"][:6],
                      "rests_on": d[:10]}
        return {"tests": {x: v for x, v in self.tests.items()}, "trust": out,
                "receipt_starts": {r: {"agent": v["agent"], "level": v["level_start"], "start": list(v.get("start") or []),
                                       "instance": v.get("instance"), "moves": v["moves"]}
                                   for r, v in self.receipts.items()}}

    def ideas_score(self, k) -> dict:
        refuted = {p["id"] for p in self.plans.values() if p.get("outcome") == "dead_end"}
        return {"ideas_new": sum(bool(p.get("new_idea")) for p in self.plans.values()),
                "ideas_retested": sum(bool(p.get("same_as")) for p in self.plans.values()),
                "ideas_retested_refuted": sum(p.get("same_as") in refuted for p in self.plans.values()),
                "ideas_combining": sum(len({k.entry(x).author for x in p.get("builds_on", []) if k.entry(x)} - {p["agent"]}) >= 2
                                       for p in self.plans.values()),
                "pieces_with_parts": len(self.parts), "refuted_repeats": self.refuted_repeats,
                "understanding_solves": len(self.understanding_log),
                "understanding_credit": round(sum(x["credit"] for u in self.understanding_log for x in u["paid"]), 2)}

    def explore_export(self, k) -> dict:
        return {"plans": list(self.plans.values()), "findings": self.findings, "credit": dict(self.credit),
                "credit_log": self.credit_log[-2000:], "combo_credit": {c: v for c, v in self.combo_credit.items()},
                "op_credit": dict(self.op_credit), "unused_actions": self.unused_actions(k), "teams": self.teams,
                "roles": self.roles, "reports": self.reports, "trackers": self.trackers, "pins": self.pins,
                "lead_log": self.lead_log, "receipts": len(self.receipts), "switches": self.switches,
                "contracts": list(self.contracts.values()), "kinds": self.kinds, "kind_stats": self.kind_stats(k),
                "rules": self.rules, "rules_version": self.rules_version,
                "entry_runs": {e: {"runs": r["runs"], "credit": round(r["credit"], 3), "by": sorted(r["by"])}
                               for e, r in self.entry_runs.items()},
                **({"models": self.models.export(), "model_credit": self.model_credit_log} if self.model_on else {}),
                **({"guides": {g.split("-")[0]: [{k: v for k, v in x.items() if k != "lines"} for x in vs]
                               for g, vs in self.guides.items()}} if self.guide_on else {}),
                **({"explanations": dict(self.explanations), "solutions": {f"{g.split('-')[0]}|{lv}": e for (g, lv), e
                                                                          in self.solution_of.items()}} if self.carry else {}),
                **({"understanding": self.understanding_log, "parts": self.parts} if self.hyp_on else {}),
                **(self.trust_export(k) if self.trust_on else {}),
                **({"messages": k.messages_log} if getattr(k, "messages_on", False) else {}),
                **({"guide_island": {g.split("-")[0]: d for g, d in self.guide_island.items()}} if self.island_guides else {}),
                **({"unique_credit": self.unique_log} if self.unique_credit else {}), **self.directions_export(),
                **({"islands": {"config": k.islands, "published": {x: list(v) for x, v in k.published.items()},
                                "opened_by_level": sorted(e.id for e in k.index.values() if e.author in k.agents
                                                          and e.id not in k.published and self.entry_open(k, e))}}
                   if getattr(k, "islands", None) else {})}

    def explore_mock(self, k, aid, t, rng) -> list:
        """Scripted uses of every action, for rehearsals (t is the mock's task label; the tick is k.tick)."""
        acts, t = self.directions_mock(k, aid, rng), k.tick
        if self.delegation:
            led, team = self._led(k, aid), self._team_of(k, aid)
            if t == 1 and aid in ("A1", "A3"):
                acts.append(("lead", {"op": "found"}))
            elif t >= 2 and not team and self.teams:
                acts.append(("lead", {"op": "join", "team": sorted(self.teams)[0]}))
            elif led:
                kids = self.children(k, led)
                if kids:
                    acts.append(("lead", {"op": "assign", "to": rng.choice(kids), "role": "explorer", "job": "test clicks"}))
                    acts.append(("lead", {"op": "tracker", "to": kids[0]}))
            elif team:
                acts.append(("lead", {"op": "report", "text": f"t{t}: tried moves"}))
                if self.trackers.get(team) == aid:
                    acts.append(("lead", {"op": "pin", "text": "what we know so far"}))
        if self.explore:
            pieces = [f["id"] for f in self.findings if f["agent"] not in (aid, "lab")][-2:] if self.hyp_on else []
            if self.lessons_on and t == 3 and aid == "A2":
                acts.insert(0, ("guide", {"op": "add", "game": "lessons", "line": "edge counters are move budgets"}))
            acts.append(("plan", {"op": "claim", "hypothesis": f"ACTION{rng.randint(1, 5)} moves the piece",
                                  "criteria": "the piece moves one cell", **({"builds_on": pieces} if pieces else {})}))
            mine = [r for r, v in self.receipts.items() if v["agent"] == aid][-1:]
            if mine and rng.random() < 0.3:
                acts.append(("plan", {"op": "finding", "finding": "partial: it moved once", "evidence": mine}))
            last = [r for r, v in self.receipts.items() if v["agent"] == aid][-1:]
            if last:
                acts.append(("plan", {"op": "done", "finding": "it moved", "evidence": last,
                                      "outcome": rng.choice(["works", "dead_end"])}))
            if rng.random() < 0.1:
                acts.append(("focus", {"game": rng.choice(self.games).split("-")[0]}))
            if t == 2 and aid == "A2":
                acts.append(("substrate", {"op": "define_kind", "name": "mechanic", "purpose": "what an action does",
                                           "fields": ["action", "effect"]}))
            if t >= 3 and aid in ("A2", "A5"):
                acts.append(("write", {"board": "global", "kind": "mechanic", "content": "action: ACTION1\neffect: moves up"}))
            if t == 3 and aid == "A5":
                acts.append(("substrate", {"op": "adopt", "name": "mechanic"}))
            if t == 3 and aid == "A6":
                acts.append(("write", {"board": "global", "content": "probe: ACTION1 ACTION2 ACTION1"}))
            runnable = [e for e in k.boards["global"].entries if e.content.startswith("probe:")]
            if t >= 4 and runnable and rng.random() < 0.5:
                acts.append(("play", {"entry": runnable[-1].id}))
            theirs = [f for f in self.findings if f["agent"] != aid and not f.get("contested")]
            if t == 4 and aid == "A7" and theirs and last:
                acts.append(("plan", {"op": "challenge", "entry": theirs[-1]["id"], "finding": "it did not move", "evidence": last}))
            if t == 3 and aid == "A4":
                acts.append(("substrate", {"op": "propose", "text": "raise novelty credit"}))
            others = [f for f in self.findings if f["agent"] not in (aid, "lab") and f.get("evidence")]
            if self.trust_on and others and t >= 2 and rng.random() < 0.4:
                if self.supports_replay and rng.random() < 0.5:
                    acts.append(("plan", {"op": "test", "entry": others[-1]["id"]}))
                elif last:
                    acts.append(("plan", {"op": "test", "entry": others[-1]["id"], "result": rng.choice(["holds", "fails"]),
                                          "evidence": last, "finding": "my run"}))
        if self.guide_on and aid in ("A3", "A4"):
            g = self.assigned.get(aid)
            cur = self.guides[g][-1]
            acts.append(("guide", {"op": "edit", "base": cur["v"], "text": cur["text"] + f"\n{aid} t{t}: ACTION1 moves up",
                                   "why": "rehearsal"}))
        if self.model_on and aid in ("A1", "A2"):
            g = self.assigned.get(aid)
            cur = self.models.current(g)
            acts.append(("model", {"op": "view"}))
            if t >= 2:  # a no-op edit: refused because it explains nothing new
                acts.append(("model", {"op": "propose", "base": cur["v"], "code": cur["code"] + "\n# same\n", "why": "rehearsal"}))
        if self.carry and self.solution_of and t >= 2:
            sid = next(iter(self.solution_of.values()))
            mine = [r for r, v in self.receipts.items() if v["agent"] == aid][-1:]
            if mine and rng.random() < 0.2:
                acts.append(("plan", {"op": "explain", "entry": sid, "finding": "the clicks line the blocks up", "evidence": mine}))
        if self.contracts_on and t == 3 and self.credit[aid] >= 0.5 and len(k.agents) > 1:
            other = rng.choice([x for x in k.agents if x != aid])
            acts.append(("contract", {"op": "offer", "to": other, "job": "map the level", "pay": 0.5}))
        if self.contracts_on:
            for c in self.contracts.values():
                if c["to"] == aid and c["status"] == "offered":
                    acts.append(("contract", {"op": "accept", "id": c["id"]}))
                elif c["from"] == aid and c["status"] == "accepted":
                    acts.append(("contract", {"op": "done", "id": c["id"]}))
        return acts
