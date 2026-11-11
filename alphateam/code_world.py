"""A code world: rebuild a real program from its behaviour, as a test of whether a society organizes itself.

The agents rebuild ProgramBench's xsv (a CSV tool with 21 commands) in Python. The work splits (commands) and couples (the
help layer, shared CSV code, the dispatcher), HEAD is one shared artifact, and hidden tests judge it, so organization has a
real job here, which ARC's single frontier never gave it. See GOAL.md: the claim to test is that, given enough primitives
(agents can add their own) and incentives, an efficient collective and an emergent structure form.

Six things are offered, none assigned (the user's list, 2026-11-03):
1. Primitive tools. The kernel's (boards, groups that nest, links, messages, tasks, protocols, reflect, design_action) and the
   Explore layer's (plan, guide, lead, contract, substrate) stay; this world adds shell, probe, publish and head (the work),
   own (authority over part of the code), toolbox (a script for everyone, an agent-made primitive), pay, role (a title of
   one's own) and trace (read what another agent has been doing).
2. Boards: private and group boards (world option private=1), plus the one public board, on which every post reaches everyone
   once.
3. Trust: each piece shows what it can be trusted at (said, seen in k inputs, reproduced, contested), from receipts and tests.
4. The credit engine: credit is paid only for verified progress at HEAD, and it flows along what made the progress possible.
5. Hierarchy: groups nest, leads get a royalty, a group can hold code. Nothing makes a hierarchy; it is measured if one forms.
6. Reflection: alphateam/reflection.py reviews the society every few ticks. Any rule may change (the law below, the practices, what
   each turn shows), and only accuracy judges a change. Who does what is never assigned: structure stays the agents' own choice.

The law is everything the review may change, with bounds; what it may not change is the accuracy score and the facts credit
is computed from. The goal in the law says whose wealth credit is: an agent's own, its team's, or the society's (shared
equally, a flat collective reward). Wealth has consequences: recent credit buys steps, offers land in order of credit, and it
pays others.

Some of this comes from reading Avalon-M1 (General Reasoning, 2026), in AlphaTeam's terms; see DESIGN.md §19.

The evidence layer is te-next's code (the harness of the ten-agent xsv baseline): the reference behind a probe meter, the
regression gate, the held-out set and the hidden grader, so scores compare directly with those runs. The world option fake=1
uses FakeHarness, a small in-memory program, for tests and rehearsals. Nothing here shows hidden tests to anyone, and only
counts ever leave the grader.
"""
from __future__ import annotations

import asyncio
import copy
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from .explore import fit_rows
from .kernel import ActionError, as_list, clip
from .society import GOALS, ROLE_TOOL, RULE_KEYS, TRACE_TOOL, SocietyWorld, bound_law

HARNESS = "/Users/baiyu/asp-runs/te-next"
TASK = "datasets/programbench-xsv-py-v2/task.json"
HELD_OUT = "validation/programbench-xsv-py-v2r.jsonl"
GRADER = "/Users/baiyu/asp-runs/pbx/scripts/native_eval.sh"
GRADES = Path("/Users/baiyu/asp-runs/alphateam-grades")


def score_native(grader: str, root: Path, submission: Path, tid: str) -> dict:
    """A packaged program on the hidden tests: the grader's script, then its own summary. Only counts come back.
    Module-level so that a saved program can be graded again (tools/regrade.py) without a harness or a run."""
    done = subprocess.run(["sh", grader, str(submission), str(submission.parent / "native")], cwd=root,
                          capture_output=True, timeout=7200)
    if done.returncode:
        return {"status": f"grader exit {done.returncode}"}
    code = ("import json, os, re, sys\nfrom pathlib import Path\nfrom codetest.programbench_score import summarize\n"
            "m = re.search(r'PROGRAMBENCH_HF_REVISION=(\\w+)', Path(sys.argv[3]).read_text())\n"
            "os.environ.setdefault('HF_HUB_OFFLINE', '1')\n"
            "if m: os.environ.setdefault('PROGRAMBENCH_HF_REVISION', m.group(1))\n"
            "s = summarize(sys.argv[1], sys.argv[2])\n"
            "print(json.dumps({k: s.get(k) for k in ('status', 'passed', 'total', 'resolved', 'expected_tests')}))\n")
    scored = subprocess.run([str(Path(root) / ".venv/bin/python"), "-c", code, str(submission), tid, grader], cwd=root,
                            capture_output=True, text=True, timeout=600)
    if scored.returncode:
        return {"status": "scoring failed", "error": scored.stderr.strip().splitlines()[-1][:160] if scored.stderr.strip() else ""}
    return json.loads(scored.stdout.strip().splitlines()[-1])
GAME = "xsv"
HIDDEN_TESTS = 1182

ORDERS = ("unblocks", "credit", "open")
# Everything the review may change, besides the norms, the teams and the Explore layer's lead_share, action_share and
# steps_per_credit. gain and held_out make the value of a merge; the shares are paid out of it, never minted.
LAW = {"goal": "individual", "gain": 0.3, "held_out": 1.0, "check_share": 0.15, "piece_share": 0.25, "tool_share": 0.1,
       "owner_share": 0.15, "tester_share": 0.25, "said": 0.2, "seen": 0.6, "reproduced": 1.0, "steps": 12, "own_ticks": 6,
       "own_cap": 40, "offer_order": "unblocks", "fine_ratio": 2.0}
LAW_BOUNDS = {"gain": (0, 2), "held_out": (0, 4), "check_share": (0, .5), "piece_share": (0, .6), "tool_share": (0, .4),
              "owner_share": (0, .5), "tester_share": (0, .6), "said": (0, 1), "seen": (0, 1), "reproduced": (0, 1),
              "steps": (4, 24), "own_ticks": (2, 12), "own_cap": (1, 200), "fine_ratio": (0, 4)}
WHOLE = ("steps", "own_ticks", "own_cap")
SHOW = ("plans", "findings", "dead_ends", "guide", "interfaces", "paid", "unused", "untried", "cycle", "kinds")

TOOL_USE = re.compile(r"/public/tools/([a-z][a-z0-9_-]{1,30})")
GROUP_READ = re.compile(r"/public/group/(A\d+)")
IMPORTS = re.compile(r"^\s*(?:from\s+(?:prog\.|\.)(\w+)\s+import|from\s+(?:prog|\.)\s+import\s+([\w, ]+)|import\s+prog\.(\w+))", re.M)
TOOL_NAME = re.compile(r"[a-z][a-z0-9_-]{1,30}")

SHELL_TOOL = {
    "name": "shell",
    "description": "Run one shell command in your own copy of the program, at /workspace in a fresh container with no network. "
                   "One command may read several files, write several, build and check: batch what belongs together. Your "
                   "source files persist; the container, /tmp and installed packages do not. Read-only: /inputs/head (HEAD), "
                   "/public/probes.jsonl (every logged observation of the reference), /public/replay.py (builds ./executable "
                   "and compares it with the logged observations; --help), /public/board.json (the state of HEAD), "
                   "/public/tools/ (tools agents shared), /public/proposals/ (changes waiting for an owner), /public/unmerged/ "
                   "(work the gate turned down or that conflicted: nothing is lost), /public/group/<agent>/ (working copies of "
                   "agents who share a group with you), /reference-docs, "
                   "/task/SPEC.md. Uses one step; returns a receipt (R-id) to cite as evidence.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
PROBE_TOOL = {
    "name": "probe",
    "description": "Run the reference program, the only way to reach it, from the budget everyone shares. Give the normal "
                   "command line, not the executable: args, stdin, input files and files to capture. An observation is logged "
                   "for everyone and becomes a check the gate protects; an equivalent repeat is free. Uses one step; returns "
                   "a receipt (R-id).",
    "parameters": {"type": "object", "properties": {
        "args": {"type": "array", "items": {"type": "string"}, "description": "e.g. [\"count\", \"data.csv\"]"},
        "stdin": {"type": "string"},
        "files": {"type": "object", "additionalProperties": {"type": "string"}, "description": "input file name -> text"},
        "capture": {"type": "array", "items": {"type": "string"}, "description": "output files to capture"}},
        "required": ["args"]}}
PUBLISH_TOOL = {
    "name": "publish",
    "description": "Offer your changed files to HEAD at the end of this tick. The gate merges them three-way, builds, and replays "
                   "every logged observation; a file lands only if nothing that matched before breaks. cites names the posts "
                   "(E-ids) this builds on: they share its credit, by what each can be trusted at.",
    "parameters": {"type": "object", "properties": {
        "note": {"type": "string", "description": "what this change does, why it works and what you ruled out"},
        "cites": {"type": "array", "items": {"type": "string"}}}, "required": []}}


def publish_tool(offer: str, replays: bool = False) -> dict:
    """The publish action as the rules in force make it (recipe settings `offer` and `publish_replays`). With the
    starting rules it is what it always was."""
    if replays:
        t = copy.deepcopy(publish_tool(offer))
        t["description"] += " Its result also shows the logged checks replayed on your copy as it is at that moment."
        return t
    if offer == "named":
        t = copy.deepcopy(PUBLISH_TOOL)
        t["description"] = ("Offer files of your copy to HEAD at the end of this tick: paths names them (patterns allowed, * for "
                            "every changed file); only those are offered, and the note does not select files. "
                            + PUBLISH_TOOL["description"].split(". ", 1)[1])
        t["parameters"]["properties"]["paths"] = {"type": "array", "items": {"type": "string"},
                                                  "description": "the changed files to offer: paths or patterns"}
        t["parameters"]["required"] = ["paths"]
        return t
    if offer == "every_tick":
        t = copy.deepcopy(PUBLISH_TOOL)
        t["description"] = ("Add a note and cites to this tick's offer of your changed files; they are offered at the end of every "
                            "tick with or without it. A publish also offers again a file the gate turned away. "
                            + PUBLISH_TOOL["description"].split(". ", 1)[1])
        return t
    return PUBLISH_TOOL


HEAD_TOOL = {
    "name": "head",
    "description": "See the state of HEAD in full. what: failing (every feature with failing checks), merges (every merge so far), "
                   "owners (claims and proposals), tools (the toolbox), contracts, credit, roles, unmerged (work that did "
                   "not land), law (the rules in force and their history). The whole state is also in /public/board.json.",
    "parameters": {"type": "object", "properties": {
        "what": {"type": "string", "enum": ["failing", "merges", "owners", "tools", "contracts", "credit", "roles", "unmerged", "law"]}},
        "required": ["what"]}}
OWN_TOOL = {
    "name": "own",
    "description": "Offered, never required. op=claim holds paths (file patterns such as prog/cmd_join.py or prog/help*) for you, or "
                   "for a group of yours with for=G1: anyone else's change there becomes a proposal instead of landing. op=accept "
                   "lands a proposal for what you hold (through the gate); op=reject turns one down, and needs evidence: the receipts "
                   "of a run showing what is wrong with it. op=release gives a claim up; op=pass hands it to=A3, so responsibility "
                   "moves to one agent at a time. A claim lapses if its holder lands or decides nothing there for a few ticks.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["claim", "accept", "reject", "release", "pass"]},
        "paths": {"type": "array", "items": {"type": "string"}, "description": "with claim"},
        "for": {"type": "string", "description": "with claim: a group you belong to"},
        "why": {"type": "string"}, "id": {"type": "string", "description": "O-id of a claim, or P-id of a proposal"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "with reject: R-ids of your run that shows the problem"},
        "to": {"type": "string", "description": "with pass"}}, "required": ["op"]}}
TOOLBOX_TOOL = {
    "name": "toolbox",
    "description": "Add a primitive of your own to the work. op=add stores script under a name at /public/tools/<name>, executable, "
                   "for everyone's shell commands; while someone else lands a merge after using it, you share its credit. "
                   "op=revise replaces your own script.",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["add", "revise"]}, "name": {"type": "string"},
        "description": {"type": "string"}, "script": {"type": "string", "description": "a whole script, with its #! line"}},
        "required": ["op", "name"]}}
# The Explore layer's tools in this world's words
WORDS = {
    "plan": "Plans and findings on the shared substrate. claim: announce what you will do or test: a hypothesis and the result "
            "that would show it (criteria); everyone sees it while it is open, and every claim stays listed under IDEAS with what "
            "came of it. finding: post an intermediate finding with its receipts, keeping the plan open. done: close your plan "
            "with what changed, the outcome and the receipts (R-ids from shell or probe) that show it. drop: abandon it. "
            "challenge: contest someone's finding with a counterexample (finding + receipts); it is shown as contested until its "
            "author revises it (done or finding with entry=<the contested finding>). test: test someone's result before you "
            "trust it (entry=<E-id or R-id>): run what it says on your own copy, then give result=holds or fails, evidence (your "
            "receipts) and finding (what you saw). A claim or finding can name the pieces it builds on (builds_on: E-ids); their "
            "authors are told, and they share the credit of what is built on them.",
    "guide": "The society's guide to the program: one durable, revisable text of what is understood about it (how commands behave, "
             "how the code is laid out, what was ruled out), shown every turn to everyone, its lines numbered. view: the full "
             "guide, its versions and who wrote each line. add: append one line with the receipts that show it (evidence). merge: "
             "combine lines (lines: their numbers) into one shorter line (line). challenge: flag someone's line with a "
             "counterexample (line, evidence, why). edit: replace it with your revision, written on the current version (base).",
    "lead": "Found and run a team. found: found a team (if you are in a team, it is a sub-team). "
            "join/leave: join an open team or leave yours. assign: give a direct child (a member, or the lead of a sub-team) a role "
            "and a job, shown at the top of its every turn. report: tell your lead what you did and found. tracker: name who keeps "
            "your team's pinned note. pin: (tracker or lead) set the note every team member sees each turn. step_down: hand the "
            "team to a member (to) or dissolve it.",
    "substrate": "Shape the shared substrate. define_kind: add a kind of entry anyone can then write (write kind=<name>), with its "
                 "purpose and, optionally, its fields: an interface every entry of that kind must fill (lines 'field: value'). adopt: "
                 "declare you will use a kind; once enough agents adopt it, it is agreed and shown every turn. revise_kind: change an "
                 "adopted kind's fields or purpose. propose: suggest a change to anything about how this society is run (the credit "
                 "law, the norms, teams, what the substrate shows); proposals are read at every review, and only accuracy on the "
                 "hidden tests judges a change.",
}

STRUCTURES = """- Each agent or team holds a part of the code (own), and others send it proposals.
- A lead who works like a CEO or PI: directs a team, delegates jobs and roles to its children, keeps working on details itself; a child who leads a sub-team does the same, so teams nest.
- One agent builds the shared skeleton first while others probe the reference and write checks; the rest then extend it.
- One agent holds everything, and the others test, review and diagnose for it, handing it concrete failures.
- Testers or reviewers find what fails and hand a specific bug to whoever can fix it.
- An integrator takes work assembled apart and makes it land together.
- A market: jobs with prices (delegate with pay), and tools others pay for by using them."""
CYCLE = ("A working cycle that may pay (yours to use, change or ignore): see what HEAD, the guide and the board already hold, and what is "
         "dead or untried; claim what you will do and what result would show it; do it; check your receipts against the claim; "
         "publish with why it works and what you ruled out, so others can build on it; when HEAD moves, start from it.")


def clean_law(given, base: dict | None = None) -> dict:
    """A bounded law, filling anything missing from base (default LAW)."""
    return bound_law(given, base or LAW, LAW_BOUNDS, {"goal": GOALS, "offer_order": ORDERS}, WHOLE)


def law_text(law: dict, credit: dict | None = None) -> str:
    who = {"individual": "the agent it is paid to", "team": "all members of the payee's group, equally (an agent in no group keeps it)",
           "society": "all agents, equally"}[law["goal"]]
    c = credit or {}
    return (f"a merge is worth {law['gain']:g} per logged check it newly passes + {law['held_out']:g} per held-out check HEAD newly "
            f"passes. Out of that: {law['check_share']:g} to the authors of the probes behind those checks, {law['piece_share']:g} "
            f"to the pieces it used (cited, built on or read), each weighted by trust (reproduced {law['reproduced']:g}, seen "
            f"{law['seen']:g}, said {law['said']:g}, contested 0), of which {law['tester_share']:g} goes to whoever tested the piece, "
            f"{law['tool_share']:g} to the authors of tools used, {law['owner_share']:g} to the owner who accepted it; the rest to "
            f"the agent who landed it"
            + (f", less {c['lead_share']:g} to its lead" if c.get("lead_share") else "")
            + f". Credit goes to {who}. {law['steps']} steps a tick"
            + (f", plus {c['steps_per_credit']:g} per credit earned in the last two ticks, at most double" if c.get("steps_per_credit") else "")
            + ". Offers land " + {"unblocks": "by how much code at HEAD depends on what they change, so work that unblocks others comes first, "
                                              "then highest credit", "credit": "highest credit first", "open": "in agent order"}[law["offer_order"]]
            + f". A claim lapses after "
            f"{law['own_ticks']} idle ticks; a holder covers at most {law['own_cap']} files. A fine takes {law['fine_ratio']:g} per "
            f"credit spent.")


def middle(text: str, limit: int) -> str:
    """Head and tail of a long output: the end of a run is usually what matters."""
    return text if len(text) <= limit else f"{text[:limit // 2]}\n[... {len(text) - limit} characters cut ...]\n{text[-limit // 2:]}"


class Slots:
    """How many containers may run at once."""

    def __init__(self, n: int):
        self.sem = threading.BoundedSemaphore(n)

    def __enter__(self):
        self.sem.acquire()
        return self

    def __exit__(self, *args):
        self.sem.release()


class Copy:
    """One agent's working copy, the HEAD version it is based on, and its harness-owned git."""

    def __init__(self, root: Path):
        self.root, self.source, self.base, self.git = root, root / "source", root / "base", root / "git"
        self.tree, self.conflicts = None, {}


class Harness:
    """te-next's evidence layer. Nothing here calls a model."""

    def __init__(self, opts: dict):
        self.root = Path(str(opts.get("harness") or HARNESS)).resolve()
        for path in (str(self.root / "vendor"), str(self.root)):  # as te-next's run.py does
            if path not in sys.path:
                sys.path.insert(0, path)
        from codetest import benchmarks, evidence, sandbox
        self.ev, self.sb = evidence, sandbox
        self.task = benchmarks.Task.load(self.root / str(opts.get("task") or TASK))
        self.held_path = self.root / str(opts.get("held_out") or HELD_OUT)
        self.grader = str(opts.get("grader") or GRADER)
        self.probe_cap, self.trials = int(opts.get("probes", 480)), int(opts.get("gate_trials", 12))
        self.sandboxes = int(opts.get("sandboxes", 6))
        self.copies: dict[str, Copy] = {}
        # recipe setting on_conflict=take_head: a file whose rebase conflicts takes HEAD's version in the copy, and the
        # agent's version is handed to the world to keep (moved: agent -> path -> the agent's version)
        self.take_head = str(opts.get("on_conflict", "keep")) == "take_head"
        self.moved: dict[str, dict] = {}

    def standing(self, aid: str) -> list[str]:
        """The files of a copy that conflicted with HEAD and that its agent has not changed since."""
        c = self.copies[aid]
        self._resolve(c)
        return sorted(c.conflicts)

    def start(self, out: Path, agents: list[str], emit):
        ev, sb = self.ev, self.sb
        self.out = out
        self.repo = sb.Repository(out / "head", self.task.seed)
        self.slots = Slots(self.sandboxes)
        cfg = SimpleNamespace(probe_cap=self.probe_cap, society_critical_dependents=3)
        self.hub = ev.EvidenceHub(cfg, out, ev.DockerExecutor(self.task, self.slots, lambda: None), emit)
        self.gate = ev.HeadIntegrator(self.hub, self.repo, out, emit, gate=True, trials=self.trials)
        for aid in agents:
            c = self.copies[aid] = Copy(out / "workers" / aid)
            sb.copy_source(self.task.seed, c.source)
            sb.copy_source(self.repo.path, c.base)
        self.hub.bootstrap()
        self.hub.refresh_head(self.repo)
        self.held = [json.loads(x) for x in self.held_path.read_text().splitlines() if x.strip()]
        self.sync()

    def version(self) -> str:
        return self.repo.version

    def sync(self):
        """Point every copy's .git and /inputs/head at HEAD; its files stay as they are."""
        version = self.repo.version
        tree = self.out / "head-trees" / version[:12]
        if not tree.exists():
            self.repo.export(version, tree)
        for c in self.copies.values():
            c.tree = tree
            self.sb.WorkspaceGit(c.git).sync(tree, version)

    def sync_one(self, aid: str):
        version = self.repo.version
        tree = self.out / "head-trees" / version[:12]
        if not tree.exists():
            self.repo.export(version, tree)
        c = self.copies[aid]
        c.tree = tree
        self.sb.WorkspaceGit(c.git).sync(tree, version)

    def shell(self, aid: str, command: str, board: dict, extras: dict) -> tuple[dict, list]:
        c = self.copies[aid]
        with tempfile.TemporaryDirectory(dir=c.root, prefix="tool-") as temp:
            public = self.hub.write_public(Path(temp) / "public", board)
            for rel, src in extras.items():
                self._mount(src, public / rel)
            with self.slots:
                with self.task.sandbox(c.source, reference=False, git_dir=c.git) as box:
                    box.put(public, "/public", readonly=True)
                    box.put(c.tree, "/inputs/head", readonly=True)
                    result = box.execute(command)
                    if not box.alive:
                        return {**result, "killed": "time or output limit: this command's file changes were discarded"}, []
                    changed = Path(temp) / "source"
                    box.snapshot(changed)
            (changed / "executable").unlink(missing_ok=True)
            paths = self.ev.changed_items(c.source, changed)
            notes = (c.source / "PROJECT_NOTES.md", changed / "PROJECT_NOTES.md")
            if notes[1].exists() and notes[1].read_bytes() != (notes[0].read_bytes() if notes[0].exists() else b""):
                paths = sorted(set(paths) | {"PROJECT_NOTES.md"})
            shutil.rmtree(c.source)
            self.sb.copy_source(changed, c.source)
        return result, paths

    def _mount(self, src: Path, target: Path):
        """A read-only copy of a folder other agents may be changing right now. A file that vanished mid-copy only means a rebase got
        there first, so it tries again and, if the folder will not hold still, leaves it out of this one command."""
        for attempt in range(3):
            try:
                if target.exists():
                    shutil.rmtree(target)
                self.sb.copy_source(src, target)
                return
            except OSError:
                time.sleep(0.2 * (attempt + 1))
        shutil.rmtree(target, ignore_errors=True)

    def probe(self, raw: dict, aid: str, tick: int) -> dict:
        return self.hub.request(raw, purpose="agent", requested_by=aid, team=None, charter=None, round_no=tick, slot=None)

    def view(self) -> dict:
        table = self.hub.table(3)
        cov = self.hub.coverage()
        return {"coverage": self.hub.coverage_line(), "implemented": cov.get("implemented", 0), "commands": cov.get("total", 0),
                "failing": sorted(({"id": f["id"], "fail": f["head_fail"], "checks": f["obs"], "critical": f["critical"]}
                                   for f in table.values() if f["head_fail"]), key=lambda x: (-x["critical"], -x["fail"])),
                "checks_by_feature": sorted(((f["id"], f["obs"]) for f in table.values()), key=lambda x: (x[1], x[0])),
                "mismatches": self.hub.mismatch_rows(200), "checks": len(self.hub.rows()), "probes_left": self.hub.remaining,
                "head": self.repo.version[:12]}

    def missing(self) -> list[str]:
        """The commands the program's --help lists that HEAD does not implement yet (the harness's own coverage line)."""
        return list(self.hub.coverage().get("missing") or [])

    def untried(self) -> dict[str, list[str]]:
        """For each command, the options its --help lists that no logged observation has used: what nobody has tried yet."""
        seen = defaultdict(set)
        for row in self.hub.rows():
            args = row["input"]["args"]
            if args:
                seen[args[0]].update(str(a).split("=")[0] for a in args[1:])
        out = {}
        for name, options in self.hub.model.commands.items():
            left = [o.get("long") or o.get("short") for o in options if not {o.get("long"), o.get("short")} & seen[name]]
            left = [x for x in left if x not in ("--help", "-h")]
            if left:
                out[name] = left
        return out

    def dependents(self) -> Counter:
        """How many modules at HEAD import each one: the pieces that connect many others."""
        out = Counter()
        for path in (Path(self.repo.path) / "prog").glob("*.py"):
            names = set()
            for found in IMPORTS.findall(path.read_text(errors="ignore")):
                names.update(x.strip() for x in ",".join(found).split(",") if x.strip())
            for name in names - {path.stem}:
                out[f"prog/{name}.py"] += 1
        return out

    def author_of(self, oid: str) -> str | None:
        return next((r.get("requested_by") for r in self.hub.rows() if r["oid"] == oid), None)

    def changed(self, aid: str) -> list[str]:
        c = self.copies[aid]
        return self.ev.changed_items(c.base, c.source)

    def head_files(self) -> list[str]:
        return sorted(self.ev.item_map(self.repo.path))

    def drop(self, aid: str, path: str):
        """Take a file that HEAD does not have out of a copy (new_file=first_copy)."""
        target = self.copies[aid].source / path
        if target.is_file():
            target.unlink()

    def files_of(self, aid: str, paths: list[str]) -> tuple[dict, dict]:
        """Current and base versions of some of an agent's changed files."""
        c = self.copies[aid]
        ours, base = self.ev.item_map(c.source), self.ev.item_map(c.base)
        return {p: ours.get(p) for p in paths}, {p: base.get(p) for p in paths}

    def _resolve(self, c: Copy):
        """A conflicted file the agent changed after the notice merges against the HEAD version it was shown."""
        for path, info in list(c.conflicts.items()):
            target = c.source / path
            current = (target.read_bytes(), bool(target.stat().st_mode & 0o111)) if target.is_file() else None
            if self.ev.blob_sha(current) != info["kept"]:
                self.ev.write_items(c.base, {path: info["head"]})
                del c.conflicts[path]

    def offer(self, aid: str, round_no: int, hold: list[str]) -> dict:
        """The agent's changed files through the gate, without those held for an owner."""
        c = self.copies[aid]
        self._resolve(c)
        if not hold:
            return self.gate.offer(aid, c.source, c.base, round_no, point="end")
        with tempfile.TemporaryDirectory(prefix="alphateam-offer-") as temp:
            branch = Path(temp) / "branch"
            self.sb.copy_source(c.source, branch)
            base = self.ev.item_map(c.base)
            self.ev.write_items(branch, {p: base.get(p) for p in hold})
            return self.gate.offer(aid, branch, c.base, round_no, point="end")

    def offer_files(self, label: str, files: dict, base_files: dict, round_no: int) -> dict:
        """A proposal through the gate, merged three-way against the HEAD of now."""
        with tempfile.TemporaryDirectory(prefix="alphateam-proposal-") as temp:
            branch, base = Path(temp) / "branch", Path(temp) / "base"
            self.sb.copy_source(self.repo.path, branch)
            self.sb.copy_source(self.repo.path, base)
            self.ev.write_items(branch, files)
            self.ev.write_items(base, base_files)
            return self.gate.offer(label, branch, base, round_no, point="end")

    def rebase(self) -> dict:
        """Every copy onto HEAD, file by file. A conflict keeps the agent's version and is reported."""
        out = {aid: self.rebase_one(aid) for aid in self.copies}
        return {aid: paths for aid, paths in out.items() if paths}

    def rebase_one(self, aid: str) -> list[str]:
        ev, sb, c = self.ev, self.sb, self.copies[aid]
        self._resolve(c)
        known = set(c.conflicts)
        head, base, ours = (ev.item_map(p) for p in (self.repo.path, c.base, c.source))
        branch, new_base, conflicts, moved = dict(head), dict(head), {}, {}
        for path in ev.changed_items(base, ours):
            ok, value = ev.merge3(base.get(path), ours.get(path), head.get(path))
            if ok:
                branch[path] = value
            elif self.take_head:  # the copy takes HEAD's version (branch and base already hold it); the agent's goes to the world
                moved[path] = ours.get(path)
            else:
                branch[path], new_base[path] = ours.get(path), base.get(path)
                conflicts[path] = {"kept": ev.blob_sha(ours.get(path)), "head": head.get(path)}
        notes = (c.source / "PROJECT_NOTES.md").read_bytes() if (c.source / "PROJECT_NOTES.md").exists() else None
        for folder, values in ((c.source, branch), (c.base, new_base)):
            with tempfile.TemporaryDirectory(prefix="alphateam-rebase-") as temp:
                staged = Path(temp) / "tree"
                sb.copy_source(self.repo.path, staged)
                ev.write_items(staged, {p: values.get(p) for p in set(head) | set(values)})
                shutil.rmtree(folder)
                sb.copy_source(staged, folder)
        if notes is not None:
            (c.source / "PROJECT_NOTES.md").write_bytes(notes)
        c.conflicts = conflicts
        if moved:
            self.moved.setdefault(aid, {}).update(moved)
        return sorted((set(conflicts) - known) | set(moved))  # a conflict is reported once, not at every catch-up

    def held_out(self) -> dict:
        built, matches, _ = self.hub.matches(self.repo.path, "validation", rows=self.held)
        return {"built": built, "passed": sum(1 for ok in matches.values() if ok), "total": len(matches)}

    def grade(self) -> dict:
        """HEAD on the hidden tests, offline. Only counts ever leave the grader."""
        tid = self.task.data["id"]
        base = GRADES / f"{self.out.parent.name}-{self.repo.version[:8]}"  # no spaces: the grader's script needs a plain path
        submission = base / "submission"
        root = submission / tid
        root.mkdir(parents=True, exist_ok=True)
        self.repo.package(root / "submission.tar.gz")
        self.sb.safe_extract((root / "submission.tar.gz").read_bytes(), root / "source")
        return self.score_native(submission)

    def score_native(self, submission: Path) -> dict:
        return score_native(self.grader, self.root, submission, self.task.data["id"])


class FakeHarness:
    """A tiny in-memory program for tests and rehearsals: a file 'feature_N' passes check N; a file named 'broken' regresses."""

    def __init__(self, opts: dict):
        self.head: dict[str, str] = {"prog/cli.py": "x", "prog/helptext.py": "h"}
        self.copies: dict[str, dict] = {}
        self.bases: dict[str, dict] = {}
        self.logged: dict[str, str] = {}
        self.depends: dict[str, int] = {}
        self.probes_left, self.commit = int(opts.get("probes", 480)), 1
        # on_conflict=take_head, on a simple model of a conflict: a file changed in a copy whose version at HEAD is neither
        # the copy's base nor the copy's own. Without the setting this harness knows no conflicts, as before.
        self.take_head = str(opts.get("on_conflict", "keep")) == "take_head"
        self.hold_back = str(opts.get("on_conflict", "keep")) == "hold_back"
        self.moved: dict[str, dict] = {}
        self.conflicts: dict[str, dict] = defaultdict(dict)  # hold_back: agent -> path -> its version when it conflicted

    def start(self, out: Path, agents: list[str], emit):
        self.out = out
        for a in agents:
            self.copies[a], self.bases[a] = dict(self.head), dict(self.head)

    def _clash(self, aid: str, files: dict) -> list[str]:
        base = self.bases[aid]
        return sorted(p for p, text in files.items() if (self.take_head or self.hold_back) and p in self.head
                      and self.head[p] != base.get(p) and self.head[p] != text)

    def standing(self, aid: str) -> list[str]:
        for p in [p for p, text in self.conflicts[aid].items() if self.copies[aid].get(p) != text]:
            del self.conflicts[aid][p]  # the agent changed it: it merges against the HEAD of now
            self.bases[aid][p] = self.head[p]
        return sorted(self.conflicts[aid])

    def version(self) -> str:
        return f"commit{self.commit:06d}"

    def sync(self):
        pass

    def shell(self, aid: str, command: str, board: dict, extras: dict) -> tuple[dict, list]:
        changed = []
        for path, text in re.findall(r"write (\S+) (\S+)", command):
            self.copies[aid][path] = text
            changed.append(path)
        return {"output": f"ran {clip(command, 60)}", "exit_code": 0}, changed

    def probe(self, raw: dict, aid: str, tick: int) -> dict:
        oid = "o" + "_".join(raw["args"])
        if oid in self.logged:
            return {"status": "duplicate", "oid": oid, "cost": 0}
        self.logged[oid], self.probes_left = aid, self.probes_left - 1
        return {"status": "reproduced", "oid": oid, "observation": {"stdout": "ok"}, "features": ["cmd:" + raw["args"][0]], "cost": 1}

    def view(self) -> dict:
        return {"coverage": f"{sum(p.startswith('feature') for p in self.head)} features at HEAD", "implemented": 1, "commands": 2,
                "failing": [], "checks_by_feature": [("cmd:count", len(self.logged))], "mismatches": [], "checks": len(self.logged),
                "probes_left": self.probes_left, "head": self.version()[:12]}

    def missing(self) -> list[str]:
        """Listed pieces with no file at HEAD: in this small model a piece `x` is the file feature_x (tests set `listed`)."""
        return [name for name in getattr(self, "listed", ()) if f"feature_{name}" not in self.head]

    def untried(self) -> dict[str, list[str]]:
        return {"count": ["--human-readable"]}

    def dependents(self) -> Counter:
        return Counter(self.depends)

    def author_of(self, oid: str) -> str | None:
        return self.logged.get(oid)

    def changed(self, aid: str) -> list[str]:
        a, b = self.copies[aid], self.bases[aid]
        return sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))

    def head_files(self) -> list[str]:
        return sorted(self.head)

    def drop(self, aid: str, path: str):
        self.copies[aid].pop(path, None)

    def files_of(self, aid: str, paths: list[str]) -> tuple[dict, dict]:
        return {p: (self.copies[aid].get(p, "").encode(), False) for p in paths}, \
            {p: (self.bases[aid][p].encode(), False) if p in self.bases[aid] else None for p in paths}

    def _land(self, label: str, files: dict) -> dict:
        accepted, rejected, gains = [], [], []
        for path, text in sorted(files.items()):
            if "broken" in path:
                rejected.append({"file": path, "reason": "regression", "regressed_oids": ["o1"]})
                continue
            self.head[path] = text
            self.commit += 1
            accepted.append(path)
            m = re.match(r"feature_(\w+)", Path(path).name)
            if m:
                gains.append("o" + m.group(1))
        return {"accepted": accepted, "rejected": rejected, "gains": gains, "head_after": self.version()}

    def offer(self, aid: str, round_no: int, hold: list[str]) -> dict:
        files = {p: self.copies[aid][p] for p in self.changed(aid) if p not in hold and p in self.copies[aid]}
        clash = self._clash(aid, files)
        row = self._land(aid, {p: text for p, text in files.items() if p not in clash})
        row["rejected"] += [{"file": p, "reason": "conflict"} for p in clash]
        return row

    def offer_files(self, label: str, files: dict, base_files: dict, round_no: int) -> dict:
        return self._land(label, {p: v[0].decode() for p, v in files.items() if v})

    def rebase(self) -> dict:
        out = {a: self.rebase_one(a) for a in self.copies}
        return {a: paths for a, paths in out.items() if paths}

    def rebase_one(self, aid: str) -> list[str]:
        mine = {p: self.copies[aid][p] for p in self.changed(aid) if p in self.copies[aid]}
        clash, old = self._clash(aid, mine), self.bases[aid]
        if self.hold_back:  # the copy keeps the agent's version on its old base, so it stays a conflict until the agent changes it
            known = set(self.standing(aid))
            for p in clash:
                self.conflicts[aid][p] = mine[p]
            keep = {p: old[p] for p in self.conflicts[aid] if p in old}
            self.copies[aid] = {**self.head, **mine}
            self.bases[aid] = {**{p: v for p, v in self.head.items() if p not in self.conflicts[aid]}, **keep}
            return sorted(set(clash) - known)
        for p in clash:  # take_head: the copy takes HEAD's version; the agent's is handed to the world to keep
            self.moved.setdefault(aid, {})[p] = (mine.pop(p).encode(), False)
        self.copies[aid], self.bases[aid] = {**self.head, **mine}, dict(self.head)
        return clash

    def sync_one(self, aid: str):
        pass

    def held_out(self) -> dict:
        return {"built": True, "passed": sum(Path(p).name.startswith("feature_") for p in self.head), "total": 179}

    def grade(self) -> dict:
        return {"status": "fake", "passed": 100 * sum(Path(p).name.startswith("feature_") for p in self.head), "total": 1182}


class CodeWorld(SocietyWorld):
    GAME = GAME
    LAW, LAW_BOUNDS, LAW_WHOLE, SHOW, WORDS = LAW, LAW_BOUNDS, WHOLE, SHOW, WORDS
    LAW_CHOICES = {"goal": GOALS, "offer_order": ORDERS}
    SHOW_DEFAULT = ("plans", "findings", "dead_ends", "guide", "interfaces", "paid", "unused", "untried", "cycle")
    tool_names = ("shell", "probe", "publish", "head", "trace", "role")
    TOOLS = [SHELL_TOOL, PROBE_TOOL, PUBLISH_TOOL, HEAD_TOOL, TRACE_TOOL, ROLE_TOOL]
    final_tools = ("shell", "probe", "publish", "write", "plan", "guide", "message", "delegate", "own", "toolbox", "pay", "role")
    move_tools = ("shell", "probe", "publish")
    search_kinds = ("tools", "owners")
    state_noun = "input"  # a shell command runs on one agent's own copy, so nothing here can be rerun exactly
    memory_chars = 4000  # with few replies a tick, what a command printed last tick is what the next edit is written from

    def __init__(self, n_agents: int, seed: int = 0, single: bool = False, world_agents: int | None = None,
                 opts: dict | None = None):
        opts = dict(opts or {})
        on = lambda key, default="1": str(opts.get(key, default)).lower() in ("1", "true", "yes")
        self.h = FakeHarness(opts) if on("fake", "0") else Harness(opts)
        self.trial_window = max(1, int(opts.get("trial_window", 4)))
        self.own_on, self.toolbox_on = on("own") and not single, on("toolbox") and not single
        self.held_every = str(opts.get("held_every", "tick"))
        self.stream = on("stream", "0")  # land work as soon as it is offered, as the ten-agent baseline does, instead of at the end of a tick
        # two rules proposed on 2026-11-10 by readers of the record of ten runs (DESIGN.md section 22). Each defaults to what
        # the world always did. on_conflict: what a copy holds after one of its files conflicts with HEAD. offer: what reaches
        # the gate (asked: a copy's changed files when its agent publishes; named: only the files a publish names;
        # every_tick: every copy's changed files at the end of every tick).
        self.take_head = str(opts.get("on_conflict", "keep")) == "take_head"
        # two rules the societies themselves suggested after the five baseline runs of 2026-11-10 (their closing rounds).
        # on_conflict=hold_back: a file that conflicted with HEAD stays as the agent's version and is left out of its copy's
        # offers until the agent changes it. publish_replays: a publish also returns the logged checks replayed on the copy.
        self.hold_back = str(opts.get("on_conflict", "keep")) == "hold_back"
        if str(opts.get("on_conflict", "keep")) not in ("keep", "take_head", "hold_back"):
            raise ValueError(f"on_conflict is keep, take_head or hold_back, not {opts.get('on_conflict')!r}")
        self.publish_replays = on("publish_replays", "0")
        self.offer_rule = str(opts.get("offer", "asked"))
        if self.offer_rule not in ("asked", "named", "every_tick"):
            raise ValueError(f"offer is asked, named or every_tick, not {self.offer_rule!r}")
        # show_open: each turn also says which commands not at HEAD no copy has changed, and which some copy has, with whose
        self.show_open = on("show_open", "0")
        # two rules proposed on 2026-11-10 by readers of the record of 25 runs (round 2; DESIGN.md section 22).
        # new_file=first_copy: a file HEAD does not have is written in one copy at a time; the first copy to create it holds
        # it until it lands, leaves that copy, or three more ticks have passed. instructions=off: the texts an agent is shown
        # keep their facts and drop their standing instructions (the working cycle, what to do with ideas, whom to message).
        self.first_copy = str(opts.get("new_file", "open")) == "first_copy" and not single
        if str(opts.get("new_file", "open")) not in ("open", "first_copy"):
            raise ValueError(f"new_file is open or first_copy, not {opts.get('new_file')!r}")
        self.instructions = on("instructions", "1")
        self.first: dict[str, dict] = {}  # path not at HEAD -> who created it first, when, until when; open: the hold has ended
        self.not_kept: list[dict] = []
        self.aside: dict[str, dict] = defaultdict(dict)  # agent -> path -> where its version went and when (take_head)
        self.turned: dict[tuple, str] = {}  # (agent, path) -> the version the gate last turned away (every_tick)
        self._init_society(n_agents, single, world_agents, opts)
        self.TOOLS = [publish_tool(self.offer_rule, self.publish_replays) if t["name"] == "publish" else t for t in self.TOOLS]
        self.add_tools(([OWN_TOOL] if self.own_on else []) + ([TOOLBOX_TOOL] if self.toolbox_on else []))
        self.queue: dict[str, dict] = {}
        self.accepted: list[str] = []
        self.merges: list[dict] = []
        self.rejections: list[dict] = []
        self.best_held = 0
        self.hidden: dict = {}
        self.claims: dict[str, dict] = {}
        self.claim_log: list[dict] = []
        self.proposals: dict[str, dict] = {}
        self.tools: dict[str, dict] = {}
        self.tool_log: list[dict] = []
        self.recent_tools: dict[str, dict] = defaultdict(dict)  # agent -> tool -> tick last used
        self.touched: dict[str, Counter] = defaultdict(Counter)  # agent -> file -> merged changes
        self.unmerged: list[dict] = []  # work that did not land: it survives and stays visible
        self._wip_t, self._wip = -1, {}
        self._overlaps: set = set()
        self.stale: set = set()  # copies behind HEAD, rebased before their next command
        self.gate_lock = asyncio.Lock()
        self._task_claims: dict[str, list] = {}  # task -> the claims it holds, with who held them before
        self.heritage = str(opts.get("heritage", "all"))  # all: what earlier societies established too; org: how they ran only
        self.inherited_tools: list[dict] = []
        self.gate_seconds: dict[int, float] = {}

    def law_words(self) -> str:
        return law_text(self.law, self.rules["credit"])

    def _start(self, k):
        out = Path(self.workdir) if self.workdir else Path(tempfile.mkdtemp(prefix="alphateam-code-"))
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.tools_dir, self.proposals_dir, self.unmerged_dir = out / "public-tools", out / "public-proposals", out / "public-unmerged"
        for folder in (self.tools_dir, self.proposals_dir, self.unmerged_dir):
            folder.mkdir(exist_ok=True)
        for t in self.inherited_tools:
            path = self.tools_dir / t["name"]
            path.write_text(t["script"])
            path.chmod(0o755)
        self.h.start(out, list(k.agents), lambda kind, **f: None)
        held = self.h.held_out()
        self.held_log.append({"t": 0, **held})
        self.best_held = held["passed"]
        self.started = True

    def on_tick_start(self, k):
        if not self.started:
            self._start(k)

    def on_tick_end(self, k):
        if k.tick >= (k.total_ticks or 0) > 0:  # the last call: whatever is still in a copy is offered, so finished work is not left out
            for aid in k.active_agents():
                if aid not in self.queue and self.h.changed(aid):
                    self.queue[aid] = {"note": "offered at the end of the run", "cites": []}
        elif self.offer_rule == "every_tick":  # the same, at the end of every tick: publishing does not take a reply
            for aid in k.active_agents():
                if aid not in self.queue and self.h.changed(aid):
                    self.queue[aid] = {"note": "offered at the end of the tick", "cites": [], "sweep": True}
        before, began = self.h.version(), time.time()
        landed = []
        for aid in self.stale:
            self._conflict_if(k, aid, self.h.rebase_one(aid))
            self.h.sync_one(aid)
        self.stale.clear()
        for aid in self._gate_order():
            landed += self._offer(k, aid, self.queue[aid])
        self.queue.clear()
        for pid in self.accepted:
            landed += self._land_proposal(k, pid)
        self.accepted.clear()
        landed += self._lapse(k)
        self.gate_seconds[k.tick] = round(time.time() - began, 1)
        self._verify(k, landed)
        if self.h.version() != before:
            for aid, paths in self.h.rebase().items():
                self._conflict(k, aid, paths)
            self.h.sync()
        elif self.take_head:  # a file turned away for conflict while HEAD stayed where it was: its copy still takes HEAD's version
            for aid in sorted({x["agent"] for x in self.rejections if x["t"] == k.tick and x["reason"] == "conflict"}):
                self._conflict_if(k, aid, self.h.rebase_one(aid))
                self.h.sync_one(aid)

    def _conflict_if(self, k, aid: str, paths: list[str]):
        if paths:
            self._conflict(k, aid, paths)

    async def _catch_up(self, k, aid: str):
        """Stream landing: a copy behind HEAD is rebased before its next command, one at a time with the gate."""
        if aid in self.stale:
            async with self.gate_lock:
                self.stale.discard(aid)
                paths = await asyncio.to_thread(self.h.rebase_one, aid)
                self._conflict_if(k, aid, paths)
                await asyncio.to_thread(self.h.sync_one, aid)

    async def _land_now(self, k, aid: str, item: dict) -> str:
        """Stream landing: the gate takes this offer now. The harness's work runs off the event loop; the society's books stay on it."""
        await self._catch_up(k, aid)
        async with self.gate_lock:
            before = len(self.merges)
            changed = self.h.changed(aid)
            if not changed:
                return "nothing to offer: your copy equals HEAD"
            skip = self._left_out(k, aid, changed, item)
            changed = [p for p in changed if p not in skip]
            held = [p for p in changed if (c := self._claim_on(p)) and not self._holds(k, aid, c)]
            if held:
                self._propose(k, aid, held, item)
            if len(held) == len(changed):
                return f"{', '.join(held)} went to their holder as a proposal" if held else "nothing of what you named differs from HEAD"
            row = await asyncio.to_thread(self.h.offer, aid, (k.tick - 1) // self.trial_window, held + skip)
            landed = self._after_gate(k, aid, row, item)
            if landed:
                result = await asyncio.to_thread(self.h.held_out)
                self._verify(k, landed, result)
            if len(self.merges) > before:
                self.stale |= set(k.active_agents()) - {aid}
                paths = await asyncio.to_thread(self.h.rebase_one, aid)
                self._conflict_if(k, aid, paths)
                await asyncio.to_thread(self.h.sync_one, aid)
            notes, self.feedback[aid] = self.feedback[aid], []
            return " | ".join(notes) or "offered"

    def _gate_order(self) -> list[str]:
        """Who goes first at the gate. By default work that unblocks others comes first, ahead of more of the same."""
        rank = lambda a: (-self.credit[a], int(a[1:]))
        how = self.law["offer_order"]
        if how == "open":
            return sorted(self.queue, key=lambda a: int(a[1:]))
        if how == "credit":
            return sorted(self.queue, key=rank)
        depends = self.h.dependents()
        return sorted(self.queue, key=lambda a: (-sum(depends.get(p, 0) for p in self.h.changed(a)), *rank(a)))

    def _keep_unmerged(self, k, aid: str, paths: list[str], why: str, files: dict | None = None) -> str:
        """Nothing an agent establishes is lost: what did not land stays visible to everyone. files: the versions to keep,
        when they are no longer in the copy (on_conflict=take_head)."""
        if files is None:
            files, _ = self.h.files_of(aid, paths)
        name = f"t{k.tick}-{aid}"
        for path, value in files.items():
            if value is not None:
                target = self.unmerged_dir / name / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value[0])
        row = {"t": k.tick, "agent": aid, "files": paths, "why": why, "at": f"/public/unmerged/{name}/"}
        # take_head: a file turned away at the gate is moved aside by the rebase that follows, in the same tick: one row, not two
        if not (self.take_head and any(u["t"] == k.tick and u["agent"] == aid and set(paths) <= set(u["files"]) for u in self.unmerged)):
            self.unmerged.append(row)
        return f"/public/unmerged/{name}/"

    def _conflict(self, k, aid: str, paths: list[str]):
        """The substrate makes a conflict visible and the parties settle it: both are told, and whoever came later reconciles."""
        moved = self.h.moved.pop(aid, None) if self.take_head else None
        at = self._keep_unmerged(k, aid, paths, "conflict with HEAD", {p: moved.get(p) for p in paths} if moved else None)
        others: dict[str, list] = defaultdict(list)
        for path in paths:
            who = next((m["agent"] for m in reversed(self.merges) if path in m["files"] and m["agent"] != aid), None)
            if who:
                others[who].append(path)
        by = "; ".join(f"{', '.join(files)} landed by {who}" for who, files in others.items())
        # A notice is cut at 300 characters (kernel.notify), so under the two new rules it says what became of the file first
        # and names each path once. The notice of the starting rules is left exactly as it was.
        landed = f" Landed by {', '.join(others)}." if others else ""
        if self.take_head:  # the later party's copy takes in what is at HEAD; its own version is kept where everyone can see it
            for p in paths:
                self.aside[aid][p] = {"at": at, "t": k.tick}
            k.notify(aid, "lab", f"conflict on rebase: your copy now holds HEAD's version of {', '.join(paths)}; your version was moved "
                                 f"to {at}.{landed} Your other changed files are untouched.")
        elif self.hold_back:
            k.notify(aid, "lab", f"conflict on rebase in {', '.join(paths)}: your version stays in your copy, and out of your offers "
                                 f"until you change it; it is also at {at}.{landed} HEAD's is in /inputs/head.")
        else:
            k.notify(aid, "lab", f"rebase onto HEAD conflicted in {', '.join(paths)}" + (f" ({by})" if by else "") + f": your version was kept, "
                                 f"and everyone can see it at {at}. HEAD's is in /inputs/head. You came later, so you reconcile: take in HEAD's "
                                 "change, re-check, and offer again" + (f"; message {', '.join(others)} to settle it." if others else "."))
        for who, files in others.items():
            k.notify(who, "lab", f"{aid}'s work conflicts with what you landed in {', '.join(files)} (its version is at {at}); it will "
                                 "reconcile with yours and may message you.")

    def progress(self) -> dict:
        """The latest held-out measurement: how the review reads accuracy. built=False means no signal, not a zero."""
        return self.held_log[-1] if self.held_log else {"t": 0, "built": False, "passed": 0, "total": 179}

    # ------------------------------------------------------------------ actions

    def _board(self, k) -> dict:
        return {**self.h.view(), "merges": self.merges, "owners": self._owners_rows(), "tools": self._tool_rows(),
                "contracts": list(self.contracts.values()), "roles": self.titles, "unmerged": self.unmerged,
                "credit": {a: round(v, 2) for a, v in self.credit.most_common()} if self.incentive else {}}

    def _mounts(self, k, aid: str) -> dict:
        extras = {"tools": self.tools_dir, "proposals": self.proposals_dir, "unmerged": self.unmerged_dir}
        for peer in self._peers(k, aid):
            source = getattr(self.h.copies.get(peer), "source", None)
            if source is not None:
                extras[f"group/{peer}"] = source
        return extras

    def _overlap_now(self, k, aid: str, changed: list[str]) -> str:
        """Two copies changing one file: both hear of it during their turns, once a tick, so they can settle it directly."""
        work = self.in_progress(k)
        if changed:
            work[aid] = self.h.changed(aid)
        told, lines = [], []
        for other, paths in work.items():
            same = sorted(set(paths) & set(work.get(aid, ())))
            fresh = [p for p in same if (k.tick, frozenset((aid, other)), p) not in self._overlaps]
            if other == aid or not fresh:
                continue
            self._overlaps.update((k.tick, frozenset((aid, other)), p) for p in fresh)
            k.notify(other, "lab", f"{aid} has also changed {', '.join(fresh)} in its copy"
                                   + ("; settle it directly if you need to" if self.instructions else ""))
            lines.append(f"{', '.join(fresh)} also changed in {other}'s copy")
            told.append(other)
        if not lines:
            return ""
        return f"\nOverlap: {'; '.join(lines)}." + (f" Message {', '.join(told)} to settle it directly." if self.instructions else "")

    def _holder(self, k, path: str) -> dict | None:
        """new_file=first_copy: the live hold on a file HEAD does not have, if any. A hold ends when the file lands, leaves
        its holder's copy or its last tick has passed; it is not replaced, and the file is then open as under the starting rules."""
        rec = self.first.get(path)
        if rec is None or rec.get("open"):
            return None
        if path in self.h.head_files() or k.tick > rec["until"] or path not in self.h.changed(rec["agent"]):
            rec["open"] = True
            return None
        return rec

    def _first_copy(self, k, aid: str) -> str:
        """new_file=first_copy, after a shell command: a file HEAD does not have and no copy has had is held by the copy
        that created it; the same file written meanwhile in another copy is moved to /public/unmerged and taken out of it."""
        head, out = set(self.h.head_files()), []
        for path in [p for p in self.h.changed(aid) if p not in head and not self._claim_on(p)]:
            rec = self._holder(k, path)
            if rec is None:
                if path not in self.first:
                    self.first[path] = {"agent": aid, "t": k.tick, "until": k.tick + 3}
                continue
            if rec["agent"] == aid:
                continue
            files, _ = self.h.files_of(aid, [path])
            if files.get(path) is None:
                continue
            at = self._keep_unmerged(k, aid, [path], f"held by {rec['agent']} until t{rec['until']}", files)
            self.h.drop(aid, path)
            self._wip_t = -1
            self.not_kept.append({"t": k.tick, "agent": aid, "file": path, "holder": rec["agent"], "at": at})
            out.append(f"Not kept: {path}. It is not at HEAD and {rec['agent']}'s copy has had it since t{rec['t']}; a file that is not "
                       f"at HEAD is written in one copy at a time, and {rec['agent']}'s hold lasts until it lands or t{rec['until']} "
                       f"ends. Your version is at {at}{path}.")
        if out:
            free = self.open_pieces(k).split(" Not at HEAD, changed in a copy")[0]
            out.append(free) if free else None
        return ("\n" + "\n".join(out)) if out else ""

    async def _shell(self, k, aid, a):
        command = str(a.get("command") or "")
        k._need(command.strip(), "give a shell command")
        self._step(k, aid)
        await self._catch_up(k, aid)
        for name in set(TOOL_USE.findall(command)) & set(self.tools):
            self.recent_tools[aid][name] = k.tick
            if self.tools[name]["author"] != aid:
                self.tools[name]["uses"] += 1
                self.tools[name]["users"].add(aid)
        for peer in set(GROUP_READ.findall(command)) - {aid}:
            self.reads[(aid, peer)] += 1
        try:
            result, changed = await asyncio.to_thread(self.h.shell, aid, command, self._board(k), self._mounts(k, aid))
        except OSError as exc:  # one failed sandbox costs one command, never the run
            raise ActionError(f"the sandbox failed ({exc.__class__.__name__}); run it again") from None
        text = middle(str(result.get("output", "")), 7000)
        head = f"exit {result.get('exit_code')}" + (f" ({result['status']})" if result.get("status") not in (None, "ok") else "")
        head += (f"; {result['killed']}" if result.get("killed") else "") + (f"; changed: {', '.join(changed)}" if changed else "")
        rid = self._log_work(k, aid, "shell", command, f"{head}\n{text}")
        kept = self._first_copy(k, aid) if self.first_copy else ""
        return "lab.shell", f"{rid}: {head}\n{text}{kept}{self._overlap_now(k, aid, changed)}", rid

    async def _probe(self, k, aid, a):
        files = a.get("files") or {}
        if isinstance(files, list):
            files = {f["path"]: f["content"] for f in files}
        raw = {"args": [str(x) for x in a.get("args") or []], "stdin": str(a.get("stdin") or ""), "files": files,
               "capture": [str(x) for x in a.get("capture") or []]}
        self._step(k, aid)
        result = await asyncio.to_thread(self.h.probe, raw, aid, k.tick)
        shown = clip(json.dumps(result, ensure_ascii=False), 5000)
        rid = self._log_work(k, aid, "probe", " ".join(raw["args"]), shown)
        if result.get("status") == "reproduced":
            k.post("global", aid, "data", f"PROBE {result.get('oid')} by {aid} ({rid}): xsv {clip(' '.join(raw['args']), 90)} "
                                          f"({', '.join(result.get('features') or [])})", notify=False)
        return "lab.probe", f"{rid}: {shown}", rid

    async def _publish(self, k, aid, a):
        item = {"note": clip(str(a.get("note") or ""), 500), "cites": [str(x) for x in as_list(a.get("cites"))]}
        cited = self._piece_ids(k, item["cites"], item["note"])
        if self.board_on and cited:
            k.cite_use(aid, cited, f"{aid} publish t{k.tick}")
        told = "queued: the gate takes offers at the end of this tick"
        if self.offer_rule == "named":
            changed = self.h.changed(aid)
            item["paths"] = [str(x).strip() for x in as_list(a.get("paths")) if str(x).strip()]
            named = self._named(changed, item["paths"])
            k._need(named, "publish offers the files you name in paths. Your copy changes: " + (", ".join(changed) or "nothing"))
            rest = [p for p in changed if p not in named]
            told = (f"queued for the gate at the end of this tick: {', '.join(named)}."
                    + (f" Not offered, only in your copy: {', '.join(rest)}." if rest else ""))
        elif self.offer_rule == "every_tick":
            told = "noted: the gate takes your copy at the end of this tick"
        if self.publish_replays:  # run by the world in the agent's copy; it takes no step and leaves a receipt like any command
            await self._catch_up(k, aid)
            try:
                result, _ = await asyncio.to_thread(self.h.shell, aid, "python3 /public/replay.py", self._board(k), self._mounts(k, aid))
                text = middle(str(result.get("output", "")), 1500)
            except OSError as exc:
                text = f"(the replay could not be run: {exc.__class__.__name__})"
            rid = self._log_work(k, aid, "shell", "python3 /public/replay.py (run by the world with your publish)", text)
            told += f"\n{rid}: the logged checks replayed on your copy as it is now:\n{text}"
        if self.stream:
            done = await self._land_now(k, aid, item)
            rest = [p for p in self.h.changed(aid) if item.get("paths") is not None and p not in self._named([p], item["paths"])]
            return "lab.publish", (done + (f" | not offered, only in your copy: {', '.join(rest)}" if rest else "")
                                   + (told[told.index("\n"):] if self.publish_replays else "")), None
        self.queue[aid] = item
        return "lab.publish", told, None

    @staticmethod
    def _named(changed: list[str], patterns: list[str]) -> list[str]:
        """The changed files an offer names (recipe setting offer=named): by path or pattern; * is every changed file."""
        return [p for p in changed if any(q == "*" or p == q or fnmatch.fnmatch(p, q) or fnmatch.fnmatch(p, f"prog/{q}") for q in patterns)]

    def _left_out(self, k, aid: str, changed: list[str], item: dict) -> list[str]:
        """Changed files this offer does not carry to the gate. They stay in the copy: not offered, not turned away.
        offer=named: what the publish did not name. offer=every_tick, for the offer the world makes at the end of a tick:
        a file the gate turned away and the agent has not changed since, and a file someone else holds."""
        out = [p for p in changed if p in set(self.h.standing(aid))] if self.hold_back else []
        if item.get("paths") is not None:
            named = self._named(changed, item["paths"])
            out += [p for p in changed if p not in named]
        elif item.get("sweep"):
            out += [p for p in changed if self.turned.get((aid, p)) == self._version(aid, p)
                    or ((c := self._claim_on(p)) and not self._holds(k, aid, c))]
        return sorted(set(out))

    def _version(self, aid: str, path: str) -> str:
        files, _ = self.h.files_of(aid, [path])
        return hashlib.sha256(repr(files.get(path)).encode()).hexdigest()

    def _head(self, k, aid, a):
        what = str(a.get("what") or "failing")
        if what == "failing":
            return "lab.head", clip(json.dumps(self.h.view()["failing"]), 6000), None
        rows = {"merges": self.merges, "owners": self._owners_rows(), "tools": self._tool_rows(),
                "contracts": list(self.contracts.values()), "credit": dict(self.credit.most_common()), "roles": self.titles,
                "unmerged": self.unmerged, "law": {"now": self.law_words(), "history": self.law_log}}
        k._need(what in rows, "what: failing, merges, owners, tools, contracts, credit, roles, unmerged or law")
        text = json.dumps(rows[what], default=str)
        return "lab.head", text if len(text) <= 12000 else clip(text, 12000) + " (the rest is in /public/board.json)", None

    # ------------------------------------------------------------------ the gate

    def _claim_on(self, path: str) -> dict | None:
        return next((c for c in self.claims.values() if any(fnmatch.fnmatch(path, p) for p in c["paths"])), None)

    def _holds(self, k, aid: str, c: dict) -> bool:
        return c["holder"] == aid or (c["holder"].startswith("G") and k.inside(aid, c["holder"]))

    def _offer(self, k, aid: str, item: dict) -> list[dict]:
        changed = self.h.changed(aid)
        if not changed:
            self._tell(aid, f"t{k.tick}: nothing to offer: your copy equals HEAD")
            return []
        skip = self._left_out(k, aid, changed, item)
        changed = [p for p in changed if p not in skip]
        held = [p for p in changed if (c := self._claim_on(p)) and not self._holds(k, aid, c)]
        if held:
            self._propose(k, aid, held, item)
        if len(held) == len(changed):
            return []
        return self._after_gate(k, aid, self.h.offer(aid, (k.tick - 1) // self.trial_window, held + skip), item)

    def _after_gate(self, k, aid: str, row: dict, item: dict, owner: str | None = None) -> list[dict]:
        accepted, gains = row["accepted"], row["gains"]
        for r in row["rejected"]:
            self.rejections.append({"t": k.tick, "agent": aid, "file": r["file"], "reason": r["reason"]})
            detail = f" (breaks {', '.join(r.get('regressed_oids', []))})" if r.get("regressed_oids") else ""
            tail = f": {clip(r.get('build_tail', ''), 300)}" if r.get("build_tail") else ""
            at = self._keep_unmerged(k, aid, [r["file"]], r["reason"])
            if self.offer_rule == "every_tick":  # the world does not offer this version again; a publish, or a change to it, does
                self.turned[(aid, r["file"])] = self._version(aid, r["file"])
            if self.take_head and r["reason"] == "conflict":
                self._tell(aid, f"t{k.tick}: {r['file']} rejected: conflict with what landed this tick; your version is at {at}; "
                                "your copy holds HEAD's version from the next rebase")
            elif self.hold_back and r["reason"] == "conflict":
                self._tell(aid, f"t{k.tick}: {r['file']} rejected: conflict (it stays in your copy and at {at}; it is left out "
                                "of your offers until you change it)")
            else:
                self._tell(aid, f"t{k.tick}: {r['file']} rejected: {r['reason']}{detail}{tail} (it stays in your copy and at {at})")
        if not accepted:
            return []
        for p in accepted:
            self.touched[aid][p] += 1
        for c in self.claims.values():
            if self._holds(k, aid, c) and any(fnmatch.fnmatch(p, q) for p in accepted for q in c["paths"]):
                c["last"] = k.tick
        m = {"t": k.tick, "agent": aid, "files": accepted, "gains": len(gains), "gain_ids": list(gains), "verified": 0, "credit": 0.0,
             "note": item.get("note", ""), "owner": owner, "cites": item.get("cites", []), "task": self._task_for(k, aid, accepted)}
        self.merges.append(m)
        if self.held_every == "merge" and not self.stream:
            self._verify(k, [m])
            return []
        return [m]

    def _verify(self, k, merges: list[dict], held: dict | None = None):
        """What these merges did for HEAD on checks nobody can see, once, and the credit for it. An unbuilt HEAD is no signal."""
        began = time.time()
        held = held or (self.h.held_out() if merges else self.held_log[-1])
        self.held_log.append({**held, "t": k.tick, "seconds": round(time.time() - began, 1)})
        if not merges:
            return
        new = max(0, held["passed"] - self.best_held) if held.get("built", True) else 0
        self.best_held = max(self.best_held, held["passed"]) if held.get("built", True) else self.best_held
        weights = [1 + m["gains"] for m in merges]
        for m, weight in zip(merges, weights):
            part = new * weight / sum(weights)
            value = self.law["gain"] * m["gains"] + self.law["held_out"] * part
            m.update(verified=round(part, 2), held_out=held["passed"], credit=round(value, 2))
            self._pay_merge(k, m, value)
            self._tell(m["agent"], f"t{k.tick}: merged {', '.join(m['files'])}: {m['gains']} logged checks newly pass; this "
                                   f"tick's merges made {new} checks you cannot see newly pass"
                       + (f"; worth {value:.1f} credit" if self.incentive else ""))
            k.post("global", "lab", "result", f"MERGE by {m['agent']}: {', '.join(m['files'])} (+{m['gains']} checks, "
                                              f"+{m['verified']:g} verified). {m['note']}", notify=False)

    # ------------------------------------------------------------------ the credit engine

    def _pieces(self, k, aid: str, m: dict) -> dict[str, float]:
        """What a merge used: cited or built on (1), read in full in the last three ticks (.5), and what those were built on (half)."""
        out = {x: 1.0 for x in self._piece_ids(k, m.get("cites"), m.get("note", ""))}
        for p in self.plans.values():
            if p["agent"] == aid and k.tick - p["t"] <= 3:
                out.update({x: 1.0 for x in p.get("builds_on", [])})
        for eid, t in k.retrieved[aid].items():
            if k.tick - t <= 2:
                out.setdefault(eid, 0.5)
        for eid, weight in list(out.items()):
            for part in self.parts.get(eid, []):
                out.setdefault(part, weight / 2)
        keep = {}
        for eid, weight in out.items():
            e = k.entry(eid)
            if e is not None and e.author in k.agents and e.author != aid and e.kind != "data":
                keep[eid] = weight
        return keep

    def _pay_merge(self, k, m: dict, value: float):
        """A merge's value, with shares for whoever made it possible, paid out of it: the agent who landed it keeps at least a tenth."""
        if value <= 0 or not self.incentive:
            return
        aid, law, rows = m["agent"], self.law, []
        authors = Counter(self.h.author_of(o) for o in m["gain_ids"])
        for who, n in authors.items():
            if who in self.aids and who != aid:
                rows.append((who, law["check_share"] * value * n / sum(authors.values()), f"{aid}'s merge passes {n} of your checks"))
        pieces = self._pieces(k, aid, m)
        total = max(1.0, sum(pieces.values()))
        m["pieces"] = {}
        for eid, weight in pieces.items():
            level = self.trust_of(k, eid)["level"]
            amount = law["piece_share"] * value * weight * self.trust_weight(level) / total
            author = k.entry(eid).author
            m["pieces"][eid] = {"author": author, "level": level, "weight": weight, "paid": round(amount, 3)}
            testers = sorted({t["by"] for t in self.tests.get(eid, []) if t["result"] == "holds" and t["by"] not in (author, aid)})
            for t in testers:
                rows.append((t, amount * law["tester_share"] / len(testers), f"you tested {eid}, which {aid}'s merge used"))
            rows.append((author, amount * (1 - law["tester_share"] if testers else 1), f"{aid}'s merge used your {eid}"))
        inventors = {self.tools[n]["author"] for n, t in self.recent_tools[aid].items() if k.tick - t <= 2 and n in self.tools
                     and self.tools[n]["author"] != aid}
        for who in inventors:
            rows.append((who, law["tool_share"] * value / len(inventors), f"{aid}'s merge used your tool"))
        owner = m.get("owner")
        if owner and owner != aid and owner in k.agents:
            rows.append((owner, law["owner_share"] * value, f"you accepted {aid}'s proposal"))
        rows = [r for r in rows if r[1] > 0]
        paid = sum(r[1] for r in rows)
        scale = min(1.0, 0.9 * value / paid) if paid else 1.0
        self._pay_out(k, [(who, amount * scale, why) for who, amount, why in rows] + [(aid, value - paid * scale, f"merge at t{k.tick}")])

    # ------------------------------------------------------------------ the law: what a review may change

    def window_facts(self, k, start: int) -> dict:
        """What happened since tick start, for reflection: facts, no advice."""
        merges = [m for m in self.merges if m["t"] > start]
        agents = {a: {"merges": 0, "logged": 0, "verified": 0.0, "credit": round(self.credit[a], 2), "attributed": round(self.attributed[a], 2),
                      "steps": sum(n for (x, t), n in self.steps.items() if x == a and t > start),
                      "role": self.titles.get(a, {}).get("title", "")} for a in k.active_agents()}
        files = defaultdict(set)
        for m in merges:
            row = agents.get(m["agent"])
            if row:
                row["merges"] += 1
                row["logged"] += m["gains"]
                row["verified"] = round(row["verified"] + m["verified"], 2)
            for f in m["files"]:
                files[f].add(m["agent"])
        levels = Counter(self.trust_of(k, f["id"])["level"] for f in self.findings if f["agent"] != "lab")
        now = self.progress()
        then = next((h for h in reversed(self.held_log) if h["t"] <= start), self.held_log[0] if self.held_log else now)
        table = {a: self.h.changed(a) for a in k.active_agents()}
        several = Counter(p for paths in table.values() for p in paths)
        return {"tick": k.tick, "of": k.total_ticks, "held_out": {"now": now.get("passed"), "total": now.get("total"), "built": now.get("built", True),
                                                                  "gain_this_window": (now.get("passed") or 0) - (then.get("passed") or 0)},
                "agents": agents, "merges": len(merges), "logged_gains": sum(m["gains"] for m in merges),
                "verified_gains": round(sum(m["verified"] for m in merges), 1),
                "files_landed_by_several": {f: sorted(w) for f, w in files.items() if len(w) > 1},
                "rejections": dict(Counter(r["reason"] for r in self.rejections if r["t"] > start)),
                "proposals": dict(Counter(p["status"] for p in self.proposals.values() if p["t"] > start)),
                "tools": self._tool_rows(), **self.group_facts(k, lambda a: agents[a]["verified"] if a in agents else 0),
                "verified_under_a_task": round(sum(m["verified"] for m in merges if m.get("task")), 1),
                **self.action_facts(k, start), "work_not_at_head": {a: paths for a, paths in table.items() if paths},
                "same_file_in_several_copies": {p: n for p, n in several.items() if n > 1},
                "unmerged_this_window": [u for u in self.unmerged if u["t"] > start],
                "trust_of_findings": dict(levels), "tests": sum(len(v) for v in self.tests.values()),
                "claims_never_tested": sum(1 for p in self.plans.values() if p["status"] != "done" and not p.get("finding")),
                "probes_left": self.h.view()["probes_left"], "own": self._owners_rows()}

    def culture(self) -> dict:
        """What a society hands to the next: how it was run, the tools it made, and what it established (the guide, its dead
        ends). Never code of the program itself."""
        tools = []
        for name, t in self.tools.items():
            path = self.tools_dir / name
            if path.exists():
                tools.append({"name": name, "description": t["description"], "author": t["author"], "script": path.read_text()})
        return {"law": self.law, "rules": {x: self.rules["credit"][x] for x in RULE_KEYS}, "show": self.rules["substrate"]["show"],
                "norms": self.norms, "tools": tools, "titles": sorted({v["title"] for v in self.titles.values()}),
                "law_log": self.law_log, "guide": self.guides[GAME][-1]["text"],
                "dead_ends": [f["text"] for f in self.findings if f["outcome"] == "dead_end" and not f.get("contested")]}

    def inherit_culture(self, k, c: dict):
        goal = self.law["goal"]
        self.law = clean_law(c.get("law"), self.law)
        if self.goal_locked:
            self.law["goal"] = goal
        self.steps_per_tick = self.law["steps"] * self.scale
        self.rules["credit"].update({x: v for x, v in (c.get("rules") or {}).items() if x in RULE_KEYS})
        if c.get("show"):
            self.rules["substrate"]["show"] = list(c["show"])
        self.norms = str(c.get("norms") or "")
        self.inherited_tools = [t for t in c.get("tools", []) if TOOL_NAME.fullmatch(t.get("name", ""))]
        for t in self.inherited_tools:
            self.tools[t["name"]] = {"author": "heritage", "description": t.get("description", ""), "t": 0, "uses": 0, "users": set()}
        k.post("global", "heritage", "lesson", "From the previous society: the law and practices in force are the ones it ended with"
               + (f"; roles it used: {', '.join(c['titles'])}" if c.get("titles") else "")
               + (f"; tools at /public/tools: {', '.join(t['name'] for t in self.inherited_tools)}" if self.inherited_tools else "") + ".",
               notify=False)
        if self.heritage != "all":  # organization alone, so that a lineage's score measures how it organizes
            return
        text = str(c.get("guide") or "").strip()
        if text:
            lines = {ln.strip(): "inherited" for ln in text.splitlines() if ln.strip()}
            self.guides[GAME] = [{"v": 0, "author": "inherited", "t": 0, "text": text, "why": "inherited from an earlier society", "lines": lines}]
        if c.get("dead_ends"):
            k.post("global", "heritage", "lesson", "Dead ends the previous society established:\n" + "\n".join(f"- {x}" for x in c["dead_ends"]),
                   notify=False)

    # ------------------------------------------------------------------ own: authority over part of the code

    def _owners_rows(self) -> list[dict]:
        return [{"id": c["id"], "holder": c["holder"], "paths": c["paths"], "since": c["t"], "why": c["why"],
                 "waiting": [p["id"] for p in self.proposals.values() if p["status"] == "open" and c["id"] in p["claims"]]}
                for c in self.claims.values()]

    def _covers(self, patterns: list[str]) -> set[str]:
        return {f for f in self.h.head_files() if any(fnmatch.fnmatch(f, p) for p in patterns)}

    def _own(self, k, aid, a):
        k._need(self.own_on, "no own in this society")
        op = str(a.get("op") or "").strip()
        if op == "claim":
            paths = [str(p).strip() for p in a.get("paths") or [] if str(p).strip()]
            k._need(paths, "paths: file patterns such as prog/cmd_join.py")
            k._need(not any(p.startswith("/") or ".." in p for p in paths), "paths are relative to the repository")
            holder = str(a.get("for") or aid).strip()
            if holder != aid:
                k._need(holder in k.groups and k.inside(aid, holder), f"{holder} is not a group of yours")
            mine = self._covers(paths)
            k._need(len(mine | self._covers([p for c in self.claims.values() if c["holder"] == holder for p in c["paths"]]))
                    <= self.law["own_cap"], f"a holder may cover at most {self.law['own_cap']} files")
            for c in self.claims.values():
                overlap = mine & self._covers(c["paths"]) or set(paths) & set(c["paths"])
                k._need(not overlap, f"{c['id']} held by {c['holder']} covers {sorted(overlap)[0]}")
            cid = self._id("O")
            self.claims[cid] = {"id": cid, "holder": holder, "by": aid, "paths": paths, "t": k.tick, "last": k.tick,
                                "why": clip(str(a.get("why") or ""), 160)}
            self.claim_log.append({"t": k.tick, "op": "claim", "id": cid, "holder": holder, "by": aid, "paths": paths})
            k.post("global", "lab", "result", f"CLAIM {cid}: {holder} holds {', '.join(paths)}: {self.claims[cid]['why']}", notify=False)
            return "lab.claim", f"you hold {', '.join(paths)} as {cid}; it lapses after {self.law['own_ticks']} ticks without a merge or a decision there", cid
        ref = str(a.get("id") or "").strip().upper()
        if op in ("accept", "reject"):
            p = self.proposals.get(ref)
            k._need(p and p["status"] == "open", f"no open proposal {ref}")
            k._need(any(self._holds(k, aid, self.claims[c]) for c in p["claims"] if c in self.claims), "you hold none of its claims")
            for c in p["claims"]:
                if c in self.claims:
                    self.claims[c]["last"] = k.tick
            p["decider"] = aid
            if op == "accept":
                p["status"] = "accepted"
                self.accepted.append(ref)
                return "lab.accept", f"{ref} goes to the gate at the end of this tick", ref
            ev = [str(r).strip().upper() for r in as_list(a.get("evidence")) if str(r).strip()]
            k._need(ev and all(r in self.receipts for r in ev), "a rejection needs evidence: the receipts (R-ids) of a run that shows "
                    "what is wrong with it. Holding code decides nothing by itself; a counterexample does")
            p["status"], p["rejection"] = "rejected", ev
            why = clip(str(a.get("why") or ""), 200)
            line = "; ".join(self.receipt_line(r) for r in ev)
            self._tell(p["by"], f"t{k.tick}: {aid} rejected your proposal {ref}: {why}. Evidence: {line}. It stays at /public/proposals/{ref}/")
            k.notify(p["by"], aid, f"rejected your proposal {ref}: {why}. Evidence: {line}")
            return "lab.reject", f"rejected {ref}", ref
        c = self.claims.get(ref)
        k._need(c and self._holds(k, aid, c), f"you hold no claim {ref}")
        if op == "release":
            self._end_claim(k, ref, "released")
            return "lab.release", f"released {ref}", ref
        if op == "pass":
            to = str(a.get("to") or "").strip()
            k._need(to in k.agents, "to: an agent")
            c.update(holder=to, last=k.tick)
            self.claim_log.append({"t": k.tick, "op": "pass", "id": ref, "holder": to, "by": aid})
            k.notify(to, aid, f"passed you claim {ref} over {', '.join(c['paths'])}")
            return "lab.pass", f"{to} now holds {ref}", ref
        raise ActionError("op must be claim, accept, reject, release or pass")

    def _end_claim(self, k, cid: str, how: str):
        c = self.claims.pop(cid)
        self.claim_log.append({"t": k.tick, "op": how, "id": cid, "holder": c["holder"]})
        for p in self.proposals.values():   # nothing is lost: what waited on this claim goes to the gate
            if p["status"] == "open" and cid in p["claims"] and not any(x in self.claims for x in p["claims"]):
                p["status"] = "accepted"
                self.accepted.append(p["id"])

    def in_scope(self, part: list[str], whole: list[str]) -> bool:
        files = self._covers(whole) | set(whole)
        return all(p in whole or (self._covers([p]) and self._covers([p]) <= files) for p in part)

    def on_task(self, k, op: str, t, aid: str, a: dict):
        """A task with a scope is a decision handed over: while it lives its taker holds those files, and when it ends they go back."""
        super().on_task(k, op, t, aid, a)
        if not t.scope:
            return
        if op == "take":
            k._need(not [p for p in t.scope if p.startswith("/") or ".." in p], "a task's scope holds paths relative to the repository")
            mine = self._covers(t.scope)
            moved = []
            for c in self.claims.values():
                if mine & self._covers(c["paths"]) or set(t.scope) & set(c["paths"]):
                    k._need(c["holder"] == t.requester or (c["holder"].startswith("G") and k.inside(t.requester, c["holder"])) or self._holds(k, aid, c),
                            f"{c['id']} held by {c['holder']} covers part of it; only its holder can hand that over")
                    moved.append(c)
            self._task_claims[t.id] = [(c["id"], c["holder"]) for c in moved]
            for c in moved:
                c.update(holder=aid, last=k.tick, task=t.id)
                self.claim_log.append({"t": k.tick, "op": "delegate", "id": c["id"], "holder": aid, "by": t.requester, "task": t.id})
            rest = [p for p in t.scope if not any(p in c["paths"] or (self._covers([p]) & self._covers(c["paths"])) for c in moved)]
            if rest:
                cid = self._id("O")
                self.claims[cid] = {"id": cid, "holder": aid, "by": t.requester, "paths": rest, "t": k.tick, "last": k.tick,
                                    "why": f"task {t.id}", "task": t.id}
                self._task_claims[t.id].append((cid, None))
                self.claim_log.append({"t": k.tick, "op": "delegate", "id": cid, "holder": aid, "by": t.requester, "task": t.id, "paths": rest})
        elif op in ("accepted", "dropped", "lapsed", "release"):
            for cid, holder in self._task_claims.pop(t.id, []):
                if cid in self.claims:
                    if holder is None:
                        self._end_claim(k, cid, "task ended")
                    else:
                        self.claims[cid].update(holder=holder, last=k.tick)
                        self.claims[cid].pop("task", None)

    def _task_for(self, k, aid: str, files: list[str]) -> str | None:
        """The live task of this agent's that covers some of these files: what links a merge to a delegation."""
        for tid in self._task_claims:
            t = k.tasks[tid]
            if t.taker == aid and any(fnmatch.fnmatch(f, p) for f in files for p in t.scope):
                return tid
        return None

    def _lapse(self, k) -> list[dict]:
        landed = []
        for cid, c in list(self.claims.items()):
            if not c.get("task") and k.tick - c["last"] >= self.law["own_ticks"]:
                self._end_claim(k, cid, "lapsed")
                k.post("global", "lab", "result", f"LAPSED {cid}: {', '.join(c['paths'])} is free again", notify=False)
        for pid in self.accepted:
            landed += self._land_proposal(k, pid)
        self.accepted.clear()
        return landed

    def _propose(self, k, aid: str, held: list[str], item: dict):
        files, base = self.h.files_of(aid, held)
        pid = self._id("P")
        claims = sorted({c["id"] for p in held if (c := self._claim_on(p))})
        self.proposals[pid] = {"id": pid, "by": aid, "t": k.tick, "files": files, "base": base, "paths": held, "claims": claims,
                               "note": item.get("note", ""), "cites": item.get("cites", []), "status": "open"}
        folder = self.proposals_dir / pid
        for path, value in files.items():
            if value is not None:
                target = folder / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value[0])
        self._tell(aid, f"t{k.tick}: {', '.join(held)} held by {', '.join(claims)}: went to its holder as proposal {pid}")
        for cid in claims:
            holder = self.claims[cid]["holder"]
            who = k.agents_inside(holder) if holder.startswith("G") else (holder,)
            for x in who:
                k.notify(x, aid, f"proposal {pid} for {', '.join(held)}: {item.get('note', '')} (read /public/proposals/{pid}/; "
                                 "own op=accept or op=reject)")

    def _land_proposal(self, k, pid: str) -> list[dict]:
        p = self.proposals[pid]
        row = self.h.offer_files(p["by"], p["files"], p["base"], (k.tick - 1) // self.trial_window)
        p["landed"] = bool(row["accepted"])
        return self._after_gate(k, p["by"], row, {"note": p["note"], "cites": p["cites"]}, owner=p.get("decider"))

    # ------------------------------------------------------------------ toolbox: primitives agents make

    def _tool_rows(self) -> list[dict]:
        return [{"name": n, "author": t["author"], "description": t["description"], "uses_by_others": t["uses"],
                 "users": sorted(t["users"])} for n, t in self.tools.items()]

    def _toolbox(self, k, aid, a):
        k._need(self.toolbox_on, "no toolbox in this society")
        op, name = str(a.get("op") or "").strip(), str(a.get("name") or "").strip().lower()
        k._need(TOOL_NAME.fullmatch(name), "name: letters, digits, - and _, starting with a letter")
        script = str(a.get("script") or "")
        k._need(script.strip() and len(script) <= 20000, "script: a whole script of at most 20000 characters")
        if op == "add":
            k._need(name not in self.tools, f"{name} exists")
            k._need(sum(t["author"] == aid for t in self.tools.values()) < 5, "you have shared 5 tools already")
            self.tools[name] = {"author": aid, "description": clip(str(a.get("description") or ""), 200), "t": k.tick, "uses": 0,
                                "users": set()}
        elif op == "revise":
            k._need(name in self.tools and self.tools[name]["author"] == aid, f"{name} is not your tool")
        else:
            raise ActionError("op must be add or revise")
        path = self.tools_dir / name
        path.write_text(script)
        path.chmod(0o755)
        self.tool_log.append({"t": k.tick, "op": op, "name": name, "author": aid})
        if op == "add":
            k.post("global", "lab", "result", f"TOOL {name} by {aid}: {self.tools[name]['description']} (/public/tools/{name})", notify=False)
        return "lab.toolbox", f"/public/tools/{name} is ready for everyone's commands", name

    # ------------------------------------------------------------------ pay: credit moves between agents

    # ------------------------------------------------------------------ what agents see

    def search(self, k, aid, what, words):
        rows = self._tool_rows() if what == "tools" else self._owners_rows()
        return [clip(json.dumps(r), 240) for r in rows if not words or any(w in json.dumps(r).lower() for w in words)]

    def track_record(self, aid: str) -> str:
        n = sum(m["agent"] == aid for m in self.merges)
        verified = sum(m["verified"] for m in self.merges if m["agent"] == aid)
        return f"record: {n} merges, {verified:g} verified" if n else ""

    def group_record(self, gid: str) -> str:
        k, inside = self._k, ()
        if k is not None and gid in k.groups:
            inside = k.agents_inside(gid)
        n = sum(m["agent"] in inside for m in self.merges)
        return f"{n} merges by its members" if n else ""

    def deliver_group(self, k, aid: str, e) -> tuple:
        if e.author == "lab":
            return 0, "HEAD: merges, claims, jobs, tools and the law", 220
        kind = (self.entry_meta.get(e.id) or {}).get("type")
        if kind in ("dead_end", "challenge"):
            return 1, "Dead ends and challenges", 180
        if kind == "claim":
            return 2, "Ideas claimed", 180
        if kind in ("finding", "test"):
            return 3, "Findings and tests", 180
        if kind == "guide":
            return 4, "Guide", 160
        if e.kind == "data":
            return 6, "Probes", 90
        return 5, "Posts and notes", 220

    def open_pieces(self, k) -> str:
        """Recipe setting show_open: of the commands not at HEAD, which no copy has changed and which some copy has, with
        whose. A fact about the copies, already inferable from 'In other copies' by its absence; no advice."""
        missing = self.h.missing()
        if not missing:
            return ""
        work = self.in_progress(k)

        def touches(path: str, name: str) -> bool:
            stem = Path(path).stem
            return stem == name or stem.endswith("_" + name) or stem.startswith(name + "_")

        who = {name: sorted((a for a, paths in work.items() if any(touches(p, name) for p in paths)), key=lambda a: int(a[1:]))
               for name in missing}
        free = [name for name in missing if not who[name]]
        taken = [f"{name} ({', '.join(who[name])})" for name in missing if who[name]]
        return ("Not at HEAD and changed in no copy: " + (", ".join(free) or "none") + "."
                + (" Not at HEAD, changed in a copy: " + "; ".join(taken) + "." if taken else ""))

    def in_progress(self, k) -> dict[str, list[str]]:
        """What each copy has changed and HEAD does not hold yet."""
        if self._wip_t != k.tick:
            self._wip_t, self._wip = k.tick, {a: self.h.changed(a) for a in k.active_agents()}
        return self._wip

    def observe(self, k, aid: str) -> str:
        v = self.h.view()
        L = []
        if self.norms:
            L.append(f"PRACTICES the society adopted at its review at t{self.norms_t} (yours to use, change or ignore; substrate op=propose "
                     f"to change them): {self.norms}")
        L.append(f"{GAME}: HEAD {v['head']}. {v['coverage']}. {v['checks']} logged checks; probes left {v['probes_left']}.")
        if self.show_open:
            line = self.open_pieces(k)
            if line:
                L.append(line)
        if v["failing"]:
            L.append("Failing at HEAD (* connects many other features): " + ", ".join(
                f"{'*' if f.get('critical') else ''}{f['id']} {f['fail']}/{f['checks']}" for f in v["failing"]))
        if v["mismatches"]:
            L.append("What HEAD gets wrong, newest first:\n" + "\n".join(fit_rows(
                [(f"  {m['oid']} xsv {clip(' '.join(m['args']), 60)}", m["diff"], "") for m in v["mismatches"]], 2500, steps=(150, 60, 0))))
        depends = self.h.dependents()
        if depends:
            L.append("Connecting pieces (modules at HEAD, by how many others import them): " + ", ".join(f"{p} {n}" for p, n in depends.most_common()))
        work = self.in_progress(k)
        others = {a: paths for a, paths in work.items() if a != aid and paths}
        if others:
            held = {p: r for p in {p for paths in work.values() for p in paths} if (r := self._holder(k, p))} if self.first_copy else {}
            mark = lambda a, p: f"{p} (held until t{held[p]['until']})" if p in held and held[p]["agent"] == a else p  # noqa: E731
            L.append("In other copies, not at HEAD yet: " + "; ".join(f"{a} {', '.join(mark(a, p) for p in paths)}" for a, paths in sorted(others.items(), key=lambda x: int(x[0][1:]))))
        mine = work.get(aid, self.h.changed(aid))
        if self.first_copy:
            own = [f"{p} until t{r['until']}" for p in mine if (r := self._holder(k, p)) and r["agent"] == aid]
            if own:
                L.append("Held for you (not at HEAD, first written in your copy): " + ", ".join(own) + ".")
        conflicts = self.h.standing(aid) if self.hold_back else getattr(self.h.copies.get(aid), "conflicts", None)
        for p in [p for p in self.aside[aid] if p in mine]:  # shown until the agent next changes that path
            del self.aside[aid][p]
        L.append(f"Your copy: {'changes in ' + ', '.join(mine) if mine else 'same as HEAD'}."
                 + (f" Conflicts with HEAD{', left out of your offers until you change them' if self.hold_back else ''}: "
                    f"{', '.join(sorted(conflicts))}." if conflicts else "")
                 + (" Moved aside by a conflict (HEAD's version is in your copy): "
                    + "; ".join(f"{p} at t{v['t']}, your version at {v['at']}" for p, v in sorted(self.aside[aid].items())) + "."
                    if self.aside[aid] else ""))
        overlap = {p: [a for a, paths in others.items() if p in paths] for p in mine}
        overlap = {p: who for p, who in overlap.items() if who}
        if overlap:
            L.append(("Overlap with your changes, for you and them to settle directly: " if self.instructions else "Overlap with your changes: ") + "; ".join(f"{p} also in {', '.join(who)}" for p, who in overlap.items()))
        if self.feedback[aid]:
            L.append("Gate: " + " | ".join(self.feedback[aid]))
            self.feedback[aid] = []
        if self.titles:
            L.append("Roles: " + "; ".join(f"{a} {t['title']}" for a, t in sorted(self.titles.items(), key=lambda x: int(x[0][1:]))))
        if self.claims:
            L.append("Owners: " + "; ".join(f"{c['id']} {c['holder']} {','.join(c['paths'])}" for c in self.claims.values()))
        waiting = [p for p in self.proposals.values() if p["status"] == "open"
                   and any(c in self.claims and self._holds(k, aid, self.claims[c]) for c in p["claims"])]
        if waiting:
            L.append("Proposals for you: " + "; ".join(f"{p['id']} from {p['by']} {','.join(p['paths'])}" for p in waiting))
        if self.unmerged:
            L.append(f"Work that did not land, kept at /public/unmerged: {len(self.unmerged)} offers (head what=unmerged).")
        if self.tools:
            L.append("Toolbox: " + "; ".join(f"{n} by {t['author']} ({t['uses']} uses): {clip(t['description'], 70)}" for n, t in self.tools.items()))
        peers = self._peers(k, aid)
        if peers:
            L.append(f"Your group's copies are at /public/group/: {', '.join(peers)}.")
        if "untried" in self.rules["substrate"]["show"]:
            untried = self.h.untried()
            if untried:
                L.append("NOT TRIED YET: options no logged observation has used: " + "; ".join(f"{c} {' '.join(o)}" for c, o in sorted(untried.items())))
            L.append("Logged checks per feature, fewest first: " + ", ".join(f"{x} {n}" for x, n in v["checks_by_feature"]))
        L.append(f"Steps left this tick: {self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]}.")
        L += self.wealth_lines(k, aid)
        return "\n".join(L) + "\n" + self._plain(self.explore_observe(k, aid))

    def brief(self) -> str:
        spec = ""
        task = getattr(self.h, "task", None)
        if task is not None:
            spec = (task.root / "SPEC.md").read_text()
        credit = ""
        if self.incentive:
            credit = (f"\n\nCREDIT\n{self.goal_text()}\n"
                      "- Credit is paid only for verified progress at HEAD, never for activity. Held-out checks are checks nobody can see; "
                      "HEAD is replayed on them after each tick's merges, and only a new best counts.\n"
                      f"- The law in force (v{self.law_v}): {self.law_words()}\n"
                      "- Each turn shows which actions and combinations have paid so far, and which actions nobody has used yet.\n"
                      "- Wealth has consequences: it buys steps, it breaks ties at the gate, and it pays others.")
        offered = ["role: a title of your own, seen by everyone. trace: read what any agent has been doing.",
                   "groups: agents who share a group read each other's working copies at /public/group/<agent>/, and a group's memory board "
                   "is theirs. Your private board is yours."]
        if self.own_on:
            offered.append("own: hold part of the code, alone or for a group. Others' changes there become proposals you land or turn down, and "
                           "turning one down takes evidence. What waited on a lapsed claim goes to the gate.")
        offered.append("delegate: hand a job or a decision to an agent or a group, with what it covers and when it ends; it can be refused, "
                       "handed on in part, or revoked. With scope=[file patterns], its taker holds those files for as long as the task lives."
                       + (" pay is held until the job is accepted." if self.market_on else ""))
        if self.market_on:
            offered.append("pay: give credit, or fine someone at a cost to yourself.")
        offered.append("plan, guide, lead, substrate, message, link, reflect, protocol: described in their tools.")
        ways = (f"\n\nSOME WAYS TO ORGANIZE (any of these, none of them, or your own; nothing here is a recommendation, and nobody assigns any "
                f"of it)\n{STRUCTURES}") if self.ways_on else ""
        trust = ("\n\nTRUST\nEvery finding and guide line shows what it can be trusted at, worked out from evidence, never votes: said (no "
                 "engine record), seen in k inputs (receipts from k different commands or probes), reproduced (a test by someone else "
                 "held), CONTESTED (a counterexample stands). What rests on a contested piece is flagged. plan op=test checks a piece on "
                 "your own copy. The credit a piece earns from what is built on it depends on this level, so testing pieces pays both "
                 "their authors and their testers.")
        cycle = f"\n\n{CYCLE}" if "cycle" in self.rules["substrate"]["show"] and self.instructions else ""
        landing = "at once" if self.stream else "at the end of the tick"
        first = "" if self.stream else " Work that unblocks others is taken first unless the law says otherwise."
        offers = {"asked": f"publish offers your changed files to HEAD {landing}.",
                  "named": f"publish offers the changed files you name to HEAD {landing}; the rest stay in your copy.",
                  "every_tick": "At the end of every tick the gate takes whatever your copy has changed since it last judged it. "
                                "publish adds your note and cites to that offer, and offers again a file the gate turned away."
                  }[self.offer_rule]
        newfile = (" A file that is not at HEAD is written in one copy at a time: the first copy to create it holds it until it "
                   "lands, or until three more ticks have passed; a version written elsewhere meanwhile is moved to "
                   "/public/unmerged.") if self.first_copy else ""
        last = "" if self.offer_rule == "every_tick" else " Whatever is still in your copy at the very end is offered once more."
        stays = ("what did not land stays in its copy, except a file that conflicts with HEAD: the copy then holds HEAD's version of "
                 "it, and the agent's version is moved to /public/unmerged, where everyone can see it"
                 if self.take_head else "what did not land stays in its copy and is visible to everyone at /public/unmerged"
                 + ("; a file that conflicts with HEAD is left out of its copy's offers until its agent changes it" if self.hold_back else ""))
        if self.publish_replays:
            offers += " A publish also returns the logged checks replayed on your copy as it is at that moment."
        extend = ["design_action adds an action to the society (a procedure over the primitives, which others can adopt, revise or drop)",
                  "substrate define_kind adds a kind of entry, with fields if pieces must fit together"]
        if self.toolbox_on:
            extend.insert(0, "toolbox adds a script for everyone's shell commands, at /public/tools")
        return f"""THE MISSION
The society rebuilds a real program from its behaviour.

{spec}
HEAD is the society's one program and its final answer. At the end HEAD is graded on {HIDDEN_TESTS} hidden tests that nobody sees. Accuracy on them is the only thing that finally counts.

THE PICTURE
A hard problem is a large jigsaw puzzle. Many agents can solve local pieces at once; combining them into one coherent program is just as important, and it is where societies fail. Pieces are commands, shared code, findings about the reference and dead ends. HEAD is the assembled region: a placed piece is never swept off the table, and a piece joins only where it fits everything already placed. Pieces that connect many others matter first. Progress is what is assembled and coherent at HEAD, not how much lies in working copies. HEAD, the guide and the board hold what the society has already established.

HOW THE WORK GOES
- You have your own working copy. shell runs commands in it; up to {self.steps_per_tick} steps (shell or probe) per tick, more with recent credit.
- probe is the only way to run the reference. Everyone shares one budget of probes, each logged observation reaches everyone, and it becomes a check: the gate will not let HEAD lose it.
- {offers} The gate merges them three-way, builds, and replays every logged check; a file lands only if nothing that passed before breaks.{first} A result should carry its understanding: why it works, what it shows, what was ruled out.{last}
- Nothing is lost when someone else's work lands. After a merge every copy is rebased onto HEAD; {stays}.{newfile}
- Conflicts are visible and settled by the parties. Each turn shows what is in other copies and where it overlaps yours; a message reaches another agent during its turn. Evidence decides what is correct, never rank or votes: a concrete counterexample outweighs any holder. Whoever contributes later reconciles: take in what is at HEAD, re-check, then offer again, rather than overwrite it. Responsibility for a piece moves explicitly, to one agent at a time.
- head shows the failing features, every merge, owners, tools, jobs, credit, roles, unmerged work and the law.{credit}{trust}{cycle}

ACTIONS ARE YOURS TO EXPLORE AND EXTEND
Which actions and combinations of actions are useful is for this society to find out; nothing here is required. You can add to them: {'; '.join(extend)}.

OFFERED, NEVER REQUIRED
""" + "\n".join("- " + x for x in offered) + f"""{ways}

THE SOCIETY REVIEWS ITSELF
Every few ticks each agent reflects, and the society can change its credit law, its practices and what each turn shows. A change that does not help accuracy is reverted. A review never says who does what: who works with whom stays yours to decide."""

    # ------------------------------------------------------------------ results

    async def finalize(self, k):
        self.hidden = await asyncio.to_thread(self.h.grade)
        (self.out / "evaluation.json").write_text(json.dumps(self.hidden))
        print(f"hidden tests: {self.hidden.get('passed')} of {self.hidden.get('total')}", flush=True)

    def structure_measures(self, k) -> dict:
        """Who built, who was read, how work was divided: the society's shape, as measured."""
        lands = Counter(m["agent"] for m in self.merges)
        verified = Counter()
        for m in self.merges:
            verified[m["agent"]] += m["verified"]
        total = sum(verified.values()) or 1
        share = sorted((v / total for v in verified.values()), reverse=True)
        read = Counter()
        for (_, who), n in self.reads.items():
            read[who] += n
        shared = sum(1 for f, n in Counter(p for c in self.touched.values() for p in c).items() if n > 1)
        acts = self.action_facts(k, 0)
        return {"builders": len(lands), "top_builder_share": round(share[0], 3) if share else 0.0, "lands": dict(lands),
                "most_read": read.most_common(3), "reads": sum(read.values()), "files_shared": shared,
                "files_landed": len({p for c in self.touched.values() for p in c}), "roles_taken": len(self.titles),
                "action_mix": acts["action_mix"], "working_share": acts["working_share"], "organizing_share": acts["organizing_share"],
                "teams": len(self.teams), "suggestions": len(self.suggestions), "unmerged": len(self.unmerged),
                "tasks": self.delegation_measures(k), "verified_under_a_task": round(sum(m["verified"] for m in self.merges if m.get("task")), 1),
                "work_not_at_head": sum(len(self.h.changed(a)) for a in k.active_agents())}

    def score(self) -> dict:
        held = self.held_log[-1] if self.held_log else {"passed": 0, "total": 179}
        k = self._k
        view = self.h.view() if self.started else None
        return {"score": round(held["passed"] / max(1, held["total"]), 4),
                "coverage": round(view["implemented"] / max(1, view["commands"]), 4) if view else 0.0,
                "score_base": None, "score_composite": None, "structure_f1": None, "val": None,
                **(self.explore_score(k) if k is not None else {}),
                "held_out": held["passed"], "hidden": self.hidden.get("passed"), "merges": len(self.merges),
                "rejections": len(self.rejections), "steps": sum(self.used.values()),
                "claims": sum(x["op"] == "claim" for x in self.claim_log), "proposals": len(self.proposals),
                "proposals_landed": sum(bool(p.get("landed")) for p in self.proposals.values()), "tools": len(self.tools),
                "tool_uses_by_others": sum(t["uses"] for t in self.tools.values()), "payments": len(self.payments),
                "law_version": self.law_v,
                **({key: k.counts[key] for key in ("pushed", "collapsed", "interrupts")} if k is not None and getattr(k, "sync", False) else {})}

    def agent_export(self, aid: str) -> dict:
        return {"instruments": [GAME], "record": self.track_record(aid), "experiments": self.used[aid],
                "role": self.titles.get(aid), "attributed": round(self.attributed[aid], 3)}

    def export(self) -> dict:
        k = self._k
        return {"kind": "code", "structure": "code", "merges": self.merges, "rejections": self.rejections, "held_out": self.held_log,
                "hidden": self.hidden, "claims": self.claim_log, "owners_at_end": self._owners_rows(),
                "proposals": [{key: v for key, v in p.items() if key not in ("files", "base")} for p in self.proposals.values()],
                "tools": self._tool_rows(), "tool_log": self.tool_log, "payments": self.payments, "attributed": dict(self.attributed),
                "touched": {a: dict(c) for a, c in self.touched.items()}, "steps": dict(self.used),
                "law": self.law, "law_log": self.law_log, "norms": self.norms, "suggestions": self.suggestions, "titles": self.titles,
                "title_log": self.title_log, "reads": [[a, b, n] for (a, b), n in self.reads.most_common()], "unmerged": self.unmerged,
                "task_log": k.task_log if k is not None else [], "delegations": [
                    {key: getattr(t, key) for key in ("id", "title", "requester", "to", "taker", "parent", "status", "scope", "accept", "until", "evidence")}
                    for t in (k.tasks.values() if k is not None else ())],
                "traces": dict(self.traces), "gate_seconds": self.gate_seconds, "incentive": self.incentive, "own": self.own_on,
                "toolbox": self.toolbox_on, "market": self.market_on,
                **({"structure_measures": self.structure_measures(k)} if k is not None else {}),
                **(self.explore_export(k) if k is not None else {})}

    # ------------------------------------------------------------------ scripted agents (mock backend)

    def mock_actions(self, k, aid, t, rng):
        n, tick = int(aid[1:]), k.tick
        real = isinstance(self.h, Harness)
        name = f"prog/mock_{n % 3}_{tick}.py" if real else f"feature_{n}_{tick}"
        acts = [("shell", {"command": f"ls prog | head -5; printf '# scripted change\\n' > {name}" if real else f"write {name} v{tick}"}),
                ("probe", {"args": ["count", f"{n}{tick}.csv"], "files": {f"{n}{tick}.csv": "a,b\n1,2\n"}}),
                ("publish", {"note": f"scripted change {tick}", "cites": []})]
        if tick == 1 and n == 1 and self.own_on:
            acts.insert(0, ("own", {"op": "claim", "paths": ["prog/helptext.py"], "why": "the help layer"}))
        if tick == 1 and n == 2 and self.toolbox_on:
            acts.insert(0, ("toolbox", {"op": "add", "name": "fails", "description": "lists failing checks",
                                        "script": "#!/bin/sh\npython3 /public/replay.py --help\n"}))
        if tick == 1:
            acts.insert(0, ("role", {"title": "builder" if n % 2 else "tester"}))
        if tick == 2 and n == 2 and self.own_on:
            acts.insert(0, ("shell", {"command": "printf '# proposal\\n' >> prog/helptext.py" if real else "write prog/helptext.py h2"}))
        if tick == 2 and n == 3:
            acts.insert(0, ("shell", {"command": "/public/tools/fails || true"}))
        if tick == 2 and n == 4:
            acts.insert(0, ("trace", {"agent": "A1"}))
        if tick == 3 and n == 5 and self.market_on and self.credit[aid] > 1:
            acts.insert(0, ("pay", {"to": "A1", "amount": 0.5, "why": "the skeleton"}))
        return acts + [x for x in self.explore_mock(k, aid, t, rng) if x[0] not in ("play", "focus")]
