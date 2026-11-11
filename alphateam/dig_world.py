"""DiG-bench (Discovery in Games, arXiv 2608.12593) as a world: text games whose rules and win conditions must be discovered.

The 21 public games run on the benchmark's server (api.digbench.ai, a token from an email sign-in in DIGBENCH_API_TOKEN).
Each agent plays its own session of the game it was assigned. The server gives every session its own seed, so instances of
a game share their rules but can differ in their details, and a run cannot be repeated on another instance: what one agent
learns is a claim about the rules that others test on their own instances (trust=1 records those tests). The screen is a
short text string; a level has a step limit and running out (or reaching certain states) costs a life; some games have a
creative mode, a sandbox entered with its toggle action whose steps do not count. Progress on a game is the most levels any
instance completed; the score is the mean over games of levels completed / levels.

With copies=shared (2026-11-02) everyone assigned to a game plays one shared session of it, so the game is an object they
act on together, as the design world's one design is: a move by anyone changes it for all, and its steps and lives are
spent together. How they decide who moves is theirs to settle. A play is not sent when the session moved since the agent
last saw it: the agent is shown who did what, and decides on the current state (it reconciles before it acts).
With lease=1 as well, the right to move is offered as an institution (hold): an agent takes the shared session for a few
ticks, alone or for a group of its own, and the others are told who holds it, queue for it, and get it in turn when it is
released, passed on or runs out. Responsibility moves explicitly, to one holder at a time; nobody is made to use it.

The world option fake=1 replaces the server with two small local games (FakeDigServer), for tests and rehearsals.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque

from .explore import Explore
from .kernel import ActionError, clip

API = "https://api.digbench.ai/api/agent"
# the public endpoint sits behind Cloudflare, which turns away urllib's default User-Agent
USER_AGENT = "alphateam-society/1.0 (+https://digbench.ai)"

PLAY_TOOL = {
    "name": "play",
    "description": "Play your own instance of your game: a list of up to 15 actions, each one of the game's action tokens "
                   "(as listed in legal actions, e.g. \"1\", \"w\" or \"/\"). Returns, step by step, the screen after the "
                   "action (a short text), the level, lives and steps left, and any transition (a level cleared or "
                   "failed). RESET starts a new instance of the game at level 1 (give restart_game=true unless your game "
                   "is over); a new instance can differ in its details. Limited steps per tick.",
    "parameters": {"type": "object", "properties": {
        "actions": {"type": "array", "items": {"type": "string"}, "description": "e.g. [\"2\", \"2\", \"1\"]"},
        "show": {"type": "boolean", "description": "include the full state after the last step"},
        "entry": {"type": "string", "description": "instead of actions: run the moves written in an entry (E-id) or a receipt (R-id)"},
        "args": {"type": "object", "description": "with entry: values for placeholders such as {x}"},
        "restart_game": {"type": "boolean", "description": "allow RESET while your game is still going (it loses your levels)"},
        "to": {"type": "string", "description": "board for a one-line record of what happened"},
        "note": {"type": "string", "description": "in your last reply of a tick: your closing note (what these moves "
                                                  "test, what you learned, what next), kept as your public note"}},
        "required": []}}
SYNC_PLAY_TOOL = dict(PLAY_TOOL, description=PLAY_TOOL["description"] + " Your latest plays and their receipts are kept "
                      "in your recent work; give 'to' (e.g. global) to also post a one-line record.")
LOOK_TOOL = {
    "name": "look",
    "description": "See your instance now: the screen, level, lives, steps left, legal actions and mode, the game's task "
                   "description, and your last steps on this level (each action with the screen after it).",
    "parameters": {"type": "object", "properties": {"history": {"type": "integer", "description": "how many recent steps "
                                                                                                 "to show (default 12)"}}}}
REPLAY_TOOL = {
    "name": "replay",
    "description": "Run a published solution's moves on your instance from the start of the level it is on (your instance "
                   "must be at that level's start). Solutions were found on other instances of the game, so they may not "
                   "work on yours: the result shows. Long solutions continue on your next replay calls.",
    "parameters": {"type": "object", "properties": {
        "note": {"type": "string", "description": "in your last reply of a tick: your closing note, kept as your public note"}}}}
SHARED_NOTE = ("Everyone assigned to your game plays this one shared instance: a move by anyone changes it for all of them. "
               "If it moved since you last saw it, nothing is sent (no step spent) and you are shown what changed.")
HOLD_TOOL = {
    "name": "hold",
    "description": "Hold your game's shared instance so that only you, or a group of yours you name, can move it: for this "
                   "tick and up to {extra} more. The others are told who holds it, until when and why, and can queue. "
                   "op=take (when nobody holds it), op=request (queue: you get it when the holder releases it, passes it "
                   "or its time runs out), op=release, op=pass with to=<agent> (hand it on now).",
    "parameters": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["take", "request", "release", "pass"]},
        "ticks": {"type": "integer", "description": "with take or request: how many ticks, this one included (1-{n})"},
        "for": {"type": "string", "description": "with take: a group you belong to (G-id); then any of its members may move it"},
        "to": {"type": "string", "description": "with pass: the agent to hand it to"},
        "why": {"type": "string", "description": "what you will do with it, shown to the others"}},
        "required": ["op"]}}


def _shared(tool: dict) -> dict:
    """A tool's description for copies=shared: the instance is the game's, not the agent's."""
    d = tool["description"]
    for a, b in (("Play your own instance of your game", "Play your game's shared instance"),
                 ("See your instance now", "See your game's shared instance now"),
                 ("your last steps on this level", "the last steps anyone made on it"),
                 ("on your instance from", "on the shared instance from"), ("(your instance must", "(it must"),
                 ("may not work on yours", "may not work on it"), ("RESET starts a new instance of the game at level 1 "
                  "(give restart_game=true unless your game is over)", "RESET starts a new instance at level 1 for "
                  "everyone on the game (give restart_game=true unless the game is over)")):
        d = d.replace(a, b)
    return dict(tool, description=d + (" " + SHARED_NOTE if tool["name"] == "play" else ""))


class DigError(RuntimeError):
    """A call to the game server failed (after retries, or a request it refused)."""


class DigServer:
    """The DiG-bench agent API: list the games, start a session, step it. Stepping is idempotent: each step names the
    index it expects to apply (the last one + 1); resending an applied index returns the cached response."""

    def __init__(self, token: str, base: str = API, timeout: int = 60, retries: int = 5):
        self.token, self.base, self.timeout, self.retries = token, base.rstrip("/"), timeout, retries

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        last = ""
        for attempt in range(self.retries):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers={
                "Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                "Accept": "application/json", "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code < 500 and exc.code not in (408, 429):  # a refusal, not a hiccup
                    raise DigError(f"{method} {path}: HTTP {exc.code}: {body}")
                last = f"HTTP {exc.code}: {body}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = str(exc)
            time.sleep(min(2 ** attempt, 30))
        raise DigError(f"{method} {path} failed after {self.retries} attempts: {last}")

    def games(self) -> list[str]:
        return list(self._call("GET", "/games").get("games", []))

    def start(self, game: str, model_name: str, model_version: str) -> dict:
        return self._call("POST", "/sessions", {"game": game, "model_name": model_name, "model_version": model_version})

    def step(self, sid: str, index: int, action: str) -> dict:
        return self._call("POST", f"/sessions/{sid}/step", {"step_index": index, "action": action})


class _FakeGame:
    """Two small games with hidden rules, in the server's state format, for tests and rehearsals.
    T-1: a cursor A and a target * on a line; 1 moves left, 2 right, 3 stamps: on the target it clears the level,
    elsewhere it costs a life. T-2: a counter and a gate; 1 counts up (mod 5); 2 opens or closes the gate, but only when
    the counter shows 3; 3 passes the gate, clearing the level if it is open. Creative mode (/) is a free sandbox."""

    def __init__(self, game: str, seed: int):
        self.game, self.rng = game, random.Random(seed)
        self.max_level = 3 if game == "T-1" else 2
        self.limit = 10 if game == "T-1" else 12
        self.level, self.lives, self.mode, self.status = 1, 3, "survival", "in_progress"
        self._new_level()

    def _new_level(self):
        self.steps = self.limit
        if self.game == "T-1":
            self.pos, self.target = self.rng.sample(range(9), 2)
        else:
            self.count, self.open = self.rng.randrange(5), False
        self.saved = None

    def obs(self) -> str:
        if self.game == "T-1":
            return "".join("A" if i == self.pos else "*" if i == self.target else "_" for i in range(9))
        return f"count {self.count} gate {'open' if self.open else 'shut'}"

    def actions(self) -> list[str]:
        return ["1", "2", "3"] + (["/"] if self.game == "T-2" else [])

    def state(self, transition=None) -> dict:
        out = {"observation": self.obs(), "level": self.level, "max_level": self.max_level, "lives_left": self.lives,
               "steps_remaining": self.steps, "status": self.status, "done": self.status != "in_progress",
               "actions": self.actions() if self.status == "in_progress" else [], "transition": transition}
        if self.game == "T-2":
            out["mode"], out["creative_toggle"] = self.mode, "/"
        return out

    def step(self, a: str):
        if a == "/":
            if self.mode == "survival":
                self.saved, self.mode = (self.count, self.open), "creative"
                self.count, self.open = 0, False
            else:
                (self.count, self.open), self.mode = self.saved, "survival"
            return None
        if self.game == "T-1":
            if a in "12":
                self.pos = max(0, min(8, self.pos + (1 if a == "2" else -1)))
            elif self.pos == self.target:
                return self._clear()
            else:
                return self._fail()
        else:
            if a == "1":
                self.count = (self.count + 1) % 5
            elif a == "2" and self.count == 3:
                self.open = not self.open
            elif a == "3" and self.open and self.mode == "survival":
                return self._clear()
        if self.mode == "survival":
            self.steps -= 1
            if self.steps <= 0:
                return self._fail()
        return None

    def _clear(self):
        if self.level == self.max_level:
            self.status = "completed"
            return None
        self.level += 1
        self._new_level()
        return f"Level {self.level - 1} cleared — advancing to level {self.level}."

    def _fail(self):
        self.lives -= 1
        if self.lives <= 0:
            self.status = "game_over"
            return None
        self._new_level()
        return f"Level {self.level} failed — restarting level {self.level}."


class FakeDigServer:
    """The server's interface over local games: every session gets its own seed, like the real one."""

    def __init__(self, seed: int = 0):
        self.n, self.seed, self.sessions = 0, seed, {}

    def games(self) -> list[str]:
        return ["T-1", "T-2"]

    def start(self, game: str, model_name: str, model_version: str) -> dict:
        if game not in self.games():
            raise DigError(f"no game {game}")
        self.n += 1
        sid = f"s{self.n}"
        self.sessions[sid] = [_FakeGame(game, self.seed * 1000 + self.n), 0]
        return {"session_id": sid, "game": game, "step_index": 0, "state": self.sessions[sid][0].state(),
                "description": f"{game}: reach the highest level you can. Special actions: "
                               + ("'/' toggles creative mode." if game == "T-2" else "none.")}

    def step(self, sid: str, index: int, action: str) -> dict:
        g, idx = self.sessions[sid]
        if index != idx + 1:
            raise DigError(f"HTTP 409: step_index {index} != {idx + 1}")
        if g.status != "in_progress" or action not in g.actions():
            return {"step_index": idx, "invalid_action": True, "state": g.state()}
        tr = g.step(action)
        self.sessions[sid][1] = idx + 1
        return {"step_index": idx + 1, "invalid_action": False, "state": g.state(tr)}


def gid(name: str) -> str:
    """The society's name for a game: lower case without the dash (P-1 -> p1), as game names are matched elsewhere."""
    return str(name).strip().lower().replace("-", "")


class DigWorld(Explore):
    tool_names = ("play", "look")
    TOOLS = [PLAY_TOOL, LOOK_TOOL]
    search_kinds = ("games",)
    has_modules = False
    state_noun = "game instance"
    supports_replay = False  # each session has its own seed: a run cannot be repeated on another instance

    def __init__(self, n_agents: int, seed: int = 0, single: bool = False, world_agents: int | None = None,
                 opts: dict | None = None):
        opts = dict(opts or {})
        on = lambda key: str(opts.get(key, "")).lower() in ("1", "true", "yes")
        self.rng = random.Random(seed)
        if on("fake"):
            self.server = FakeDigServer(seed)
        else:
            token = os.environ.get("DIGBENCH_API_TOKEN", "").strip()
            if not token:
                raise ValueError("DiG-bench needs a game token in DIGBENCH_API_TOKEN (from digbench.ai/account/tokens)")
            self.server = DigServer(token, base=os.environ.get("DIGBENCH_SERVER_URL") or API)
        self.server_name = {gid(g): g for g in self.server.games()}
        wanted = [gid(g) for g in str(opts.get("games", "")).split(",") if g.strip()]
        if wanted:
            missing = [w for w in wanted if w not in self.server_name]
            if missing:
                raise ValueError(f"no such games on the server: {', '.join(missing)} "
                                 f"(available: {', '.join(self.server_name.values())})")
            self.games = wanted
        else:
            self.games = sorted(self.rng.sample(sorted(self.server_name), min(int(opts.get("n_games", 4)), len(self.server_name))))
        self.steps_per_tick = int(opts.get("steps", 15))
        self.publish = on("publish_solutions")
        self.model_name = str(opts.get("model_name") or "deepseek-v4.1-flash")
        self.label = str(opts.get("label") or "alphateam")
        opts["model"] = "0"  # the shared step-function model reads image frames; DiG screens are text
        if self.publish:
            self.tool_names = type(self).tool_names + ("replay",)
            self.TOOLS = type(self).TOOLS + [REPLAY_TOOL]
        self.single = single
        aids = [f"A{i + 1}" for i in range(n_agents)]
        order = aids[:]
        self.rng.shuffle(order)
        self.assigned = {a: self.games[j % len(self.games)] for j, a in enumerate(order)}
        if single:  # one agent plays one game with the society's total steps
            self.assigned = {aids[0]: self.games[0]}
            self.steps_per_tick *= (world_agents or n_agents)
        self.home = {a: [g] for a, g in self.assigned.items()}
        self.tasks = {g: g for g in self.games}
        self.sessions: dict[tuple, dict] = {}  # (agent, game) -> session id, step index, instance number
        self.frames: dict[tuple, str] = {}  # (agent, game) -> the current screen
        self.state: dict[tuple, dict] = {}
        self.history: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=60))  # (agent, game) -> (action, screen)
        self.descriptions: dict[str, str] = {}
        self.trace: dict[tuple, list] = defaultdict(list)  # (agent, game) -> moves since the current level began
        self.published: set = set()
        self.solutions: dict[tuple, list] = {}
        self.pending: dict[tuple, list] = {}
        self.steps = defaultdict(int)
        self.used = defaultdict(int)
        self.best: dict[str, int] = {g: 0 for g in self.games}
        self.win: dict[str, int] = {}
        self.firsts: list[dict] = []
        self.log: list[dict] = []
        self.focus = defaultdict(list)
        self.sync = on("sync")
        self.recent_plays: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
        self.instances = 0
        self.screens_seen: dict[str, set] = defaultdict(set)
        if self.sync:
            self.TOOLS = [SYNC_PLAY_TOOL if t is PLAY_TOOL else t for t in self.TOOLS]
        self.shared = str(opts.get("copies", "own")).strip().lower() == "shared"
        self.version: dict[tuple, int] = defaultdict(int)  # shared instance -> moves applied to it (RESET included)
        self.moves_on: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=60))  # shared instance -> who moved, what
        self.seen: dict[tuple, int] = {}  # (agent, shared instance) -> the version the agent last saw
        self._locks: dict[tuple, asyncio.Lock] = {}
        self.stale_refusals = 0
        if self.shared:
            self.TOOLS = [_shared(t) for t in self.TOOLS]
        self.lease_on = self.shared and on("lease")
        self.lease_ticks = max(1, min(5, int(opts.get("lease_ticks", 2))))
        self.leases: dict[str, dict] = {}  # game -> holder, group, until (last tick held), why, since
        self.lease_queue: dict[str, list] = defaultdict(list)  # game -> [{"agent", "why", "ticks"}]
        self.lease_log: list[dict] = []
        self.lease_refusals = 0
        if self.lease_on:
            n = self.lease_ticks
            tool = json.loads(json.dumps(HOLD_TOOL).replace("{extra}", str(n - 1)).replace("{n}", str(n)))
            self.tool_names = tuple(self.tool_names) + ("hold",)
            self.TOOLS = list(self.TOOLS) + [tool]
        self._init_explore(opts)
        self._now = None

    # ------------------------------------------------------------------ instances

    def _ck(self, aid: str, game: str) -> tuple:
        """The instance an agent plays: its own, or with copies=shared the one instance everyone on the game plays."""
        return ("shared", game) if self.shared else (aid, game)

    def _mirror(self, game: str):
        """A shared instance is also readable under each player's own key (the explore views look copies up per agent)."""
        ck = ("shared", game)
        if ck in self.state:
            for a, g in self.assigned.items():
                if g == game:
                    self.state[(a, g)], self.frames[(a, g)] = self.state[ck], self.frames[ck]

    def _start(self, aid: str, game: str):
        r = self.server.start(self.server_name[game], self.model_name, f"{self.label}{' isolated' if getattr(self, 'isolated', False) else ''}")
        key = self._ck(aid, game)
        self.instances += 1
        self.sessions[key] = {"sid": r["session_id"], "idx": int(r.get("step_index", 0)),
                              "instance": f"{'shared' if self.shared else aid}#{self.instances}"}
        self.descriptions.setdefault(game, str(r.get("description") or ""))
        self.trace[key] = []
        self.history[key].clear()
        self._record(key, r["state"], r.get("levels_beaten"))

    def _record(self, key, s: dict, beaten=None):
        """beaten: the server's levels_beaten when it sends one (creative-mode levels do not count there)."""
        mx = s.get("max_level")
        level = s.get("level") if isinstance(s.get("level"), int) else 1
        status = s.get("status") or "in_progress"
        done = mx if status == "completed" and isinstance(mx, int) else max(0, level - 1)
        done = beaten if isinstance(beaten, int) else done
        tr = s.get("transition") or ""
        self.frames[key] = str(s.get("observation", ""))
        self.state[key] = {"levels": done, "win": mx if isinstance(mx, int) else self.win.get(key[1], 0),
                           "state": {"completed": "WIN", "game_over": "GAME_OVER"}.get(status, "NOT_FINISHED"),
                           "actions": [str(x) for x in (s.get("actions") or [])], "lives": s.get("lives_left"),
                           "steps_left": s.get("steps_remaining"), "mode": s.get("mode"), "toggle": s.get("creative_toggle"),
                           "transition": tr, "failed": "fail" in tr.lower() or "lost" in tr.lower()}
        if isinstance(mx, int):
            self.win[key[1]] = max(self.win.get(key[1], 0), mx)
        if self.shared:
            self._mirror(key[1])

    def _copy(self, aid: str, game: str):
        key = self._ck(aid, game)
        if key not in self.sessions:
            self._start(aid, game)
        elif self.shared:
            self._mirror(game)
        return self.sessions[key]

    def _moved_since(self, aid: str, key) -> str:
        """With copies=shared: what others did to the instance since this agent last saw it ('' when nothing)."""
        v0 = self.seen.get((aid, key))
        if v0 is None or v0 >= self.version[key]:
            return ""
        since = [m for m in self.moves_on[key] if m["v"] > v0]
        if not since or all(m["agent"] == aid for m in since):
            return ""
        runs: list[dict] = []
        for m in since:  # consecutive moves by one agent in one tick read as one run
            if runs and runs[-1]["agent"] == m["agent"] and runs[-1]["t"] == m["t"]:
                runs[-1]["moves"].append(m["move"])
                runs[-1]["after"] = m["status"]
            else:
                runs.append({"agent": m["agent"], "t": m["t"], "moves": [m["move"]], "after": m["status"]})
        did = "; ".join(f"{r['agent']} t{r['t']}: {' '.join(r['moves'][-15:])} (then {r['after']})" for r in runs[-6:])
        return (f"NOT SENT: the shared instance of {key[1]} moved since you last saw it, so nothing was sent and no step "
                f"was spent. Since then: {did}. Now: {self._view(key, history=0)}\nDecide on this state, then play again.")

    def _moved(self, k, aid: str, key, move: str):
        if self.shared:
            self.version[key] += 1
            self.moves_on[key].append({"v": self.version[key], "t": k.tick, "agent": aid, "move": move,
                                       "status": self._status(key)})

    # ------------------------------------------------------------------ leases (copies=shared with lease=1)

    def _lease(self, k, game: str) -> dict | None:
        """The lease on a game's shared instance now; one whose time ran out passes to the first in the queue."""
        cur = self.leases.get(game)
        if cur and cur["until"] < k.tick:
            self._end_lease(k, game, "ran out")
            cur = self.leases.get(game)
        return cur

    def _grant(self, k, game: str, aid: str, until: int, why: str, group=None, how: str = "took"):
        self.leases[game] = {"holder": aid, "group": group, "until": until, "why": why, "since": k.tick}
        self.lease_log.append({"t": k.tick, "game": game, "agent": aid, "group": group, "until": until, "how": how,
                               "why": why})

    def _end_lease(self, k, game: str, how: str):
        """End a lease (released, ran out) and hand the instance to the first in the queue still on the game."""
        cur = self.leases.pop(game, None)
        if cur:
            self.lease_log.append({"t": k.tick, "game": game, "agent": cur["holder"], "how": how})
        q = self.lease_queue[game]
        while q:
            nxt = q.pop(0)
            if self.assigned.get(nxt["agent"]) == game:
                until = k.tick + nxt["ticks"] - 1
                self._grant(k, game, nxt["agent"], until, nxt["why"], how="queue")
                k.notify(nxt["agent"], "lab", f"you now hold the shared instance of {game} until the end of t{until}"
                         + (f" ({cur['holder']}'s hold {how})" if cur else "") + ": only you can move it until then; "
                         "hold op=release or op=pass when you are done")
                return

    def _lease_text(self, k, game: str, aid: str | None = None) -> str:
        cur = self._lease(k, game)
        if not cur:
            return (f"nobody holds the shared instance of {game} (hold op=take holds it for up to {self.lease_ticks} "
                    f"tick{'s' if self.lease_ticks > 1 else ''})")
        who = "you hold" if cur["holder"] == aid else f"{cur['holder']} holds"
        q = [x["agent"] for x in self.lease_queue[game]]
        return (f"{who} the shared instance of {game} until the end of t{cur['until']}"
                + (f" for its group {cur['group']}" if cur["group"] else "")
                + (f" ({cur['holder']}: {cur['why']})" if cur["why"] else "") + (f"; queue: {', '.join(q)}" if q else ""))

    def _may_move(self, k, aid: str, game: str) -> bool:
        cur = self._lease(k, game)
        if not cur or cur["holder"] == aid:
            return True
        g = cur["group"]
        return bool(g and g in k.member_of.get(aid, ()) and g in k.groups and k.groups[g].dissolved is None)

    def _hold(self, k, aid, a):
        k._need(self.lease_on, "holding the shared instance is not offered here")
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        op = str(a.get("op") or "take").strip().lower()
        why = clip(" ".join(str(a.get("why") or "").split()), 160)
        try:
            ticks = max(1, min(self.lease_ticks, int(a.get("ticks") or 1)))
        except (TypeError, ValueError):
            ticks = 1
        cur = self._lease(k, game)
        q = self.lease_queue[game]
        if op == "take":
            k._need(not cur or cur["holder"] == aid, self._lease_text(k, game, aid) + "; hold op=request queues you for it")
            group = None
            if a.get("for"):
                group = str(a["for"]).strip().upper()
                k._need(group in k.member_of.get(aid, ()) and group in k.groups and k.groups[group].dissolved is None,
                        f"you are not a member of a group {group}")
            if cur:  # the holder extends its hold, unless others are waiting
                k._need(not q, f"{', '.join(x['agent'] for x in q)} queued for it: release it or pass it on")
            self._grant(k, game, aid, k.tick + ticks - 1, why, group, how="extended" if cur else "took")
            return "lab.hold", self._lease_text(k, game, aid) + ". The others see this; their plays are not sent until then.", game
        if op == "request":
            k._need(cur and cur["holder"] != aid, "nobody else holds it: use op=take")
            if all(x["agent"] != aid for x in q):
                q.append({"agent": aid, "why": why, "ticks": ticks})
                k.notify(cur["holder"], aid, f"{aid} asks for the shared instance of {game} next" + (f": {why}" if why else ""))
            pos = 1 + [x["agent"] for x in q].index(aid)
            return "lab.hold", (f"queued for {game}: position {pos} after {cur['holder']} (holds until the end of "
                                f"t{cur['until']}); you will be told when it is yours"), game
        if op == "release":
            k._need(cur and cur["holder"] == aid, "you do not hold it")
            self._end_lease(k, game, "released")
            return "lab.hold", "released. " + self._lease_text(k, game, aid), game
        if op == "pass":
            k._need(cur and cur["holder"] == aid, "you do not hold it")
            to = str(a.get("to") or "").strip().upper()
            k._need(to != aid and to in k.agents and self.assigned.get(to) == game, f"{to or '?'} does not play {game}")
            self.lease_queue[game] = [x for x in q if x["agent"] != to]
            self._grant(k, game, to, cur["until"], f"passed on by {aid}", how="passed")
            k.notify(to, aid, f"{aid} passed you the shared instance of {game}: only you can move it until the end of "
                              f"t{cur['until']}")
            return "lab.hold", f"passed to {to}. " + self._lease_text(k, game, aid), game
        k._need(False, "op must be take, request, release or pass")

    def _status(self, key) -> str:
        st = self.state[key]
        return (f"level {st['levels'] + 1 if st['state'] == 'NOT_FINISHED' else st['levels']}/{st['win']}"
                f"{' ' + st['state'] if st['state'] != 'NOT_FINISHED' else ''}, lives {st['lives']}, steps left "
                f"{st['steps_left']}" + (f", {st['mode']} mode" if st.get("mode") and st["mode"] != "survival" else ""))

    def setup(self, k):
        self._k = k
        if self.explore:
            k.rank = dict(self.rules["substrate"]["rank"])
        for aid, game in self.assigned.items():
            self._copy(aid, game)
            if k.sync:
                continue
            st = self.state[(aid, game)]
            whose = "the shared instance" if self.shared else "your own instance"
            k.post(f"private:{aid}", "lab", "note", f"You play {whose} of game {game} ({st['win']} levels; legal "
                   f"actions now: {', '.join(st['actions'])}). Task: {self.descriptions.get(game, '')} The rules and what "
                   f"completes a level are unknown: find out by playing.", notify=False)

    def on_tick_start(self, k):
        if self.lease_on:
            for g in self.games:
                self._lease(k, g)  # holds that ran out pass to the queue before anyone's turn

    def on_tick_end(self, k):
        pass

    # ------------------------------------------------------------------ actions

    def act(self, k, aid: str, name: str, a: dict):
        self._now = name
        if name == "play":
            return self._play(k, aid, a)
        if name == "replay":
            return self._replay(k, aid, a)
        return getattr(self, f"_{name}")(k, aid, a)

    def _novel_screen(self, k, aid: str, game: str, level: int):
        """Credit a screen at the game's frontier that no instance has shown at that level and mode."""
        ck = self._ck(aid, game)
        key = (level, self.state[ck].get("mode"), self.frames[ck])
        if key in self.screens_seen[game]:
            return
        self.screens_seen[game].add(key)
        self.novel_count[game] += 1
        c = self.rules["credit"]
        if level == self.best[game] and c["novel"] > 0 and self.novel_tick[(aid, k.tick)] + c["novel"] <= c["novel_cap"] + 1e-9:
            self.novel_tick[(aid, k.tick)] += c["novel"]
            self._award(k, aid, c["novel"], f"new screen in {game} level {level + 1}")

    async def _play(self, k, aid, a):
        if self.hyp_on:
            self.cites[aid] = self._piece_ids(k, a.get("cites"), str(a.get("note") or ""))
            if self.board_on and self.cites[aid]:
                k.cite_use(aid, self.cites[aid], f"{aid} play t{k.tick}")
        if self.explore and a.get("entry"):
            ref = str(a["entry"]).strip().upper()
            moves = self.entry_moves(k, aid, ref, a.get("args") or {})
            self._running_entry[aid] = ref
            try:
                op, out, target = await self._play(k, aid, {**a, "entry": None, "actions": moves})
            finally:
                self._running_entry.pop(aid, None)
            self.entry_runs[ref]["runs"] += 1
            self.entry_runs[ref]["by"].add(aid)
            return op, f"ran {ref}: " + out, target
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        acts = a.get("actions") or []
        k._need(isinstance(acts, list) and acts, "actions must be a non-empty list of action tokens (or give an entry to run)")
        if not self.shared:
            return await self._play_on(k, aid, a, game, acts)
        key = self._ck(aid, game)
        async with self._locks.setdefault(key, asyncio.Lock()):  # one play at a time on a shared instance
            self._copy(aid, game)
            if self.lease_on and not self._may_move(k, aid, game):
                self.lease_refusals += 1
                return "lab.play", (f"NOT SENT: {self._lease_text(k, game, aid)}. No step was spent. hold op=request "
                                    f"queues you for it; or tell the holder what to try."), game
            moved = self._moved_since(aid, key)
            self.seen[(aid, key)] = self.version[key]
            if moved:
                self.stale_refusals += 1
                return "lab.play", moved, game
            out = await self._play_on(k, aid, a, game, acts)
            self.seen[(aid, key)] = self.version[key]
            return out

    async def _play_on(self, k, aid, a, game: str, acts: list):
        key = self._ck(aid, game)
        self._copy(aid, game)
        shared = self.shared
        allowed = self.steps_allowed(aid, k.tick)
        left = allowed - self.steps[(aid, k.tick)]
        k._need(left > 0, f"you have used all {allowed} steps this tick")
        st = self.state[key]
        lines, moves_done, results = [], [], []
        level_start, start = st["levels"], list(self.trace[key])
        instance = self.sessions[key]["instance"]
        game_over = None  # (level, moves from its start) when a life is lost or the game ends
        for item in acts[:min(15, left)]:
            label = str(item).strip()
            st = self.state[key]
            if label.upper() in ("RESET", "RESTART"):
                if st["state"] != "GAME_OVER" and not a.get("restart_game") and st["levels"] > 0:
                    whose = f"the {st['levels']} level(s) everyone on {game} completed" if shared else \
                        f"your {st['levels']} completed level(s)"
                    lines.append(f"RESET: refused: it starts a new instance at level 1 and loses {whose}; give "
                                 f"restart_game=true to do it anyway")
                    continue
                await asyncio.to_thread(self._start, aid, game)
                self.steps[(aid, k.tick)] += 1
                self.used[aid] += 1
                instance = self.sessions[key]["instance"]
                self._moved(k, aid, key, "RESET")
                lines.append(f"RESET: a new instance of {game}{' for everyone on it' if shared else ''}: "
                             f"{self.frames[key]!r} ({self._status(key)})")
                moves_done.append("RESET")
                results.append({"move": "RESET", "screen": self.frames[key], "levels": self.state[key]["levels"],
                                "state": self.state[key]["state"]})
                continue
            if st["state"] != "NOT_FINISHED":
                lines.append(f"{label}: the game is {'over' if st['state'] == 'GAME_OVER' else 'won'}; RESET starts a new instance")
                break
            if label not in st["actions"]:
                lines.append(f"{label}: not a legal action now (legal: {', '.join(st['actions']) or 'none'})")
                continue
            s = self.sessions[key]
            lv0, before = st["levels"], self.frames[key]
            try:
                r = await asyncio.to_thread(self.server.step, s["sid"], s["idx"] + 1, label)
            except DigError as exc:
                lines.append(f"{label}: the game server failed ({clip(str(exc), 160)}); try again")
                break
            self.steps[(aid, k.tick)] += 1
            self.used[aid] += 1
            if r.get("invalid_action"):
                self._record(key, r["state"], r.get("levels_beaten"))
                lines.append(f"{label}: refused by the game (legal now: {', '.join(self.state[key]['actions'])})")
                continue
            s["idx"] = int(r.get("step_index", s["idx"] + 1))
            self._record(key, r["state"], r.get("levels_beaten"))
            st = self.state[key]
            self._moved(k, aid, key, label)
            self.trace[key].append(label)
            self.history[key].append((f"{aid}: {label}" if shared else label, self.frames[key]))
            moves_done.append(label)
            results.append({"move": label, "screen": self.frames[key], "levels": st["levels"], "state": st["state"]})
            if self.explore:
                self.note_tried(game, lv0, label)
                self._novel_screen(k, aid, game, st["levels"])
            note = ""
            if st["levels"] > lv0:
                note = f" LEVEL {lv0 + 1} COMPLETED"
                moves, self.trace[key] = self.trace[key], []
                if self.publish and (game, st["levels"]) not in self.published:
                    self.published.add((game, st["levels"]))
                    self.solutions[(game, st["levels"])] = list(moves)
                    where = "the shared instance" if shared else f"{aid}'s instance"
                    e = k.post("global", "lab", "result", f"{game} level {st['levels']} was solved by {aid} with these "
                               f"{len(moves)} moves from the start of that level, on {where} (other instances "
                               f"may differ in their details); moves: {' '.join(moves)}", notify=False)
                    self.entry_meta[e.id] = {"game": game, "level": st["levels"], "type": "solution"}
                    self.solution_of[(game, st["levels"])] = e.id
                if st["levels"] > self.best[game]:
                    self.best[game] = st["levels"]
                    self.firsts.append({"t": k.tick, "game": game, "level": st["levels"], "agent": aid})
                    if self.explore:
                        self._first_solve(k, aid, game, st["levels"])
            changed = "" if self.frames[key] != before else " (screen unchanged)"
            lines.append(f"{label}: {self.frames[key]!r}{changed} ({self._status(key)}){note}"
                         + (f" [{st['transition']}]" if st["transition"] else ""))
            self.log.append({"t": k.tick, "agent": aid, "game": game, "action": label, "screen": self.frames[key],
                             "levels": st["levels"], "state": st["state"], "transition": st["transition"]})
            if (st["failed"] or st["state"] == "GAME_OVER") and st["levels"] == lv0:
                game_over = (lv0 + 1, list(self.trace[key]), "GAME_OVER" if st["state"] == "GAME_OVER" else "LIFE")
                self.trace[key] = []  # the level starts over
            if st["state"] != "NOT_FINISHED":
                break
        receipt = (" receipt " + self._receipt(k, aid, game, level_start, moves_done, lines, self.state[key]["levels"],
                                               start=start, instance=instance, results=results)
                   if self.explore and moves_done else "")
        if game_over and self.carry and receipt:
            lvl, moves, kind = game_over
            where = f"the shared instance, {aid} moving" if shared else f"{aid}'s instance"
            self._engine_dead_end(k, aid, game, lvl, moves, receipt.split()[-1], text=(
                f"{'GAME_OVER' if kind == 'GAME_OVER' else 'LIFE LOST'} on {game} level {lvl} after {len(moves)} moves "
                f"from the level start ({where}); the last ones: {' '.join(moves[-12:]) or '(none)'}"))
        if k.sync and not a.get("to"):
            self.recent_plays[aid].append(f"t{k.tick} {game}{receipt}: {clip(' | '.join(lines[-4:]), 400)}")
        else:
            board = k.board_id(aid, a.get("to", "private"))
            if k.can_write(aid, board):
                k.post(board, aid, "data", f"{aid} played {game}: " + " | ".join(lines[-6:]), notify=False)
        out = (f"{game}, {'the shared instance' if shared else 'your instance'}:\n" + "\n".join(lines) + f"\n[steps left this tick: "
               f"{self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]}]" + (f" [engine{receipt}]" if receipt else ""))
        rep = self.repeats_refuted(game, level_start + 1, moves_done) if self.hyp_on and not a.get("_replay") else []
        if rep:
            self.refuted_repeats += 1
            out += "\nNOTE: these moves repeat part of a refuted test: " + "; ".join(
                f"{r} behind dead end {f['id']} ({f['agent']}, t{f['t']}): {clip(f['text'], 120)}" for f, r in rep) + \
                ". Repeating it teaches nothing new unless something else is different; if so, say what in your note."
        if a.get("show"):
            out += "\n" + self._view(key)
        return "lab.play", out, game

    def entry_moves(self, k, aid: str, ref: str, args: dict) -> list[str]:
        """The moves of a receipt, or those written in an entry on a line 'moves: 1 2 2 3'."""
        ref = str(ref).strip().upper()
        if ref.startswith("R"):
            k._need(ref in self.receipts, f"no receipt {ref}")
            moves = [m for m in self.receipts[ref]["moves"] if m != "RESET"]
        else:
            e = k.entry(ref)
            k._need(e and k.can_read(aid, e.board), f"no entry {ref} you can read")
            m = re.search(r"moves?:\s*([^\n]+)", e.content, re.I)
            k._need(m, f"{ref} has no line 'moves: ...' (write the moves as e.g. 'moves: 1 2 2 3')")
            moves = [x.strip(".;") for x in re.split(r"[\s,]+", m.group(1).strip()) if x.strip(".;")]
        k._need(moves, f"{ref} has no moves")
        return moves

    async def _replay(self, k, aid, a):
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        key = self._ck(aid, game)
        self._copy(aid, game)
        queue = self.pending.get(key)
        intro = ""
        if not queue:
            target = self.state[key]["levels"] + 1
            moves = self.solutions.get((game, target))
            k._need(moves, f"no published solution yet for level {target} of {game}")
            k._need(not self.trace[key], f"{'the shared' if self.shared else 'your'} instance is not at the start of level "
                                         f"{target} ({len(self.trace[key])} moves were made on it); a solution runs from "
                                         f"a level's start")
            queue = list(moves)
            sid = self.solution_of.get((game, target))
            if self.carry and sid:
                exps = [k.entry(x) for x in self.explanations.get(sid, [])][-2:]
                self.shown_explanations[aid].update(x.id for x in exps if x)
                intro = (f"WHY THESE MOVES WORK ({sid}): " + (" | ".join(clip(x.content, 400) for x in exps if x)
                                                              if exps else "no explanation posted yet") + ".\n")
        left = self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]
        k._need(left > 0, "you have used all your steps this tick")
        now, rest = queue[:min(15, left)], queue[min(15, left):]
        self.pending[key] = rest
        lv0 = self.state[key]["levels"]
        _, out, ref = await self._play(k, aid, {"actions": now, "_replay": True})
        if out.startswith("NOT SENT"):  # the shared instance moved first: a later replay starts from its new state
            self.pending[key] = []
            return "lab.replay", out, ref
        done = self.state[key]["levels"] > lv0
        if done or self.state[key]["state"] != "NOT_FINISHED":
            self.pending[key] = []
        tail = (" Reached the next level." if done else f" {len(rest)} moves left: call replay again to continue."
                if rest and self.state[key]["state"] == "NOT_FINISHED" else
                f" The moves did not complete the level on {'the shared' if self.shared else 'your'} instance.")
        return "lab.replay", intro + out + tail, ref

    def _view(self, key, history: int = 12) -> str:
        st = self.state[key]
        hist = list(self.history[key])[-max(0, min(int(history), 60)):] if history else []
        whose = "the last {} steps on it, by anyone" if self.shared else "your last {} steps"
        return (f"screen: {self.frames[key]!r}\n{self._status(key)}; legal actions: {', '.join(st['actions']) or 'none'}"
                + (f"; creative mode toggle: {st['toggle']}" if st.get("toggle") else "")
                + (f"\nlast transition: {st['transition']}" if st["transition"] else "")
                + (f"\n{whose.format(len(hist))} (action -> screen):\n" + "\n".join(f"  {m} -> {s!r}" for m, s in hist)
                   if hist else ""))

    def _look(self, k, aid, a):
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        key = self._ck(aid, game)
        self._copy(aid, game)
        if self.shared:
            self.seen[(aid, key)] = self.version[key]
        return "lab.look", (f"{game} ({'the shared' if self.shared else 'your'} instance), task: "
                            f"{self.descriptions.get(game, '')}\n"
                            + self._view(key, a.get("history", 12) if a.get("history") is not None else 12)), None

    # ------------------------------------------------------------------ views

    def search(self, k, aid, what, words):
        out = []
        for g in self.games:
            players = sorted((x for x, gg in self.assigned.items() if gg == g), key=lambda x: int(x[1:]))
            plans = (f"; open plans on level {self._problem(g)}: " + "; ".join(f"{p['agent']} {p['hypothesis'][:60]}"
                                                                           for p in self.open_plans(k, g))
                     if self.explore and self.open_plans(k, g) else "")
            out.append(f"{g}: best {self.best[g]}/{self.win.get(g, '?')} levels; players {', '.join(players)}{plans}")
        return out

    def agent_tags(self, aid: str) -> str:
        g = self.assigned.get(aid)
        return f"plays {g}" if g else ""

    def track_record(self, aid: str) -> str:
        g = self.assigned.get(aid)
        if not g or (aid, g) not in self.state:
            return ""
        base = f"record: level {self.state[(aid, g)]['levels']} in {g}"
        return base + ("; " + self.record_of(aid) if self.explore else "")

    def group_record(self, gid: str) -> str:
        return ""

    def observe(self, k, aid: str) -> str:
        g = self.assigned.get(aid)
        if not g:
            return "DiG: no game."
        key = self._ck(aid, g)
        st = self.state.get(key, {})
        where = f"your instance: {self._status(key) if st else '?'}"
        if self.shared:
            self.seen[(aid, key)] = self.version[key]  # what the agent is shown now is what its next play is checked against
            others = sorted((x for x, gg in self.assigned.items() if gg == g and x != aid), key=lambda x: int(x[1:]))
            last = list(self.moves_on[key])[-10:]
            where = (f"the shared instance, with {', '.join(others) or 'nobody else'}: {self._status(key) if st else '?'}; "
                     f"last moves on it: {' '.join(m['agent'] + ':' + m['move'] for m in last) or 'none'}"
                     + (f"; HOLD: {self._lease_text(k, g, aid)}" if self.lease_on else ""))
        return (f"DiG: you play game {g} ({where}; screen "
                f"{self.frames.get(key, '')!r}); society's best on it: {self.best[g]} of {self.win.get(g, '?')} levels. "
                f"Steps left this tick {self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]}. Legal actions: "
                f"{', '.join(st.get('actions') or []) or 'none'}. Task: {self.descriptions.get(g, '')}"
                + ("\n" + self.explore_observe(k, aid) if (self.explore or self.delegation) else ""))

    def sync_rank(self, k, aid: str, e) -> int:
        mine = self.assigned.get(aid)
        if not mine:
            return 3
        on = self.state.get((aid, mine), {}).get("levels", 0) + 1
        m = self.entry_meta.get(e.id)
        if m:
            if m["type"] == "challenge" and m.get("target") == aid:
                return 0
            if m["game"] == "lessons":
                return 1
            if m["game"] != mine:
                return 3
            if m["type"] in ("solution", "explanation"):
                return 0 if m["level"] == on else 2 if m["type"] == "explanation" else 3
            if m["type"] in ("guide",):
                return 1
            if m["level"] in (on, self._problem(mine)):
                return 0 if m["type"] == "dead_end" else 1
            return 2
        if e.author == "lab":
            return 2
        named = next((g for g in self.games if g in e.content), None)
        return 2 if (named or self.assigned.get(e.author)) == mine else 3

    def deliver_group(self, k, aid: str, e) -> tuple:
        mine, m = self.assigned.get(aid), self.entry_meta.get(e.id)
        if m and m["game"] in (mine, "lessons"):
            t = m["type"]
            if t in ("dead_end", "challenge"):
                return 0, "Dead ends and challenges on your game", 160
            if t == "claim":
                return 1, "Ideas claimed on your game", 160
            if t in ("finding", "solution", "explanation", "test"):
                return 2, "Findings, tests, solutions and explanations on your game", 160
            return 3, "Guide and lessons", 160
        if m or (e.author != "lab" and self.assigned.get(e.author) not in (None, mine)):
            return 6, "Other games", 80
        if e.kind == "data" or e.content.startswith("My last moves this tick"):
            return 5, "Plays", 90
        return 4, "Notes and other posts", 120

    def recent_work(self, k, aid: str) -> str:
        plays = list(self.recent_plays.get(aid, ()))
        return ("Your latest plays (cite their receipts as evidence):\n" + "\n".join(f"- {p}" for p in plays)) if plays else ""

    def brief(self) -> str:
        games = ", ".join(f"{g} ({self.win.get(g, '?')} levels)" for g in self.games)
        tasks = "".join(f"\n  {g}: {self.descriptions[g]}" for g in self.games if self.descriptions.get(g))
        inst = ("Every agent plays its own instance of the game it was assigned: instances share their rules but can differ "
                "in their details, so a move sequence that worked on one may not work on another, while a rule found on one "
                "should hold on all. Your moves never disturb anyone else's instance, and what you learn reaches the others "
                "only if it is shared.")
        look = "look shows your instance now with your recent steps"
        if self.shared:
            inst = ("All agents assigned to a game play one shared instance of it: a move by anyone changes it for everyone "
                    "on that game, and its steps per level and its lives are spent by all of them together. If the instance "
                    "moved since you last saw it, your play is not sent (no step is spent) and you are shown who did what, so "
                    "you decide on its current state. A RESET starts a new instance, which can differ in its details, for "
                    "everyone on the game.")
            look = "look shows the shared instance now with the recent steps anyone made on it"
            if self.lease_on:
                look += (f". hold takes the shared instance for up to {self.lease_ticks} tick(s) so that only you, or a "
                         "group of yours you name, can move it; the others see who holds it, until when and why, can "
                         "queue (op=request), and get it in turn when it is released, passed on or runs out. Holding is "
                         "offered, never required")
        return f"""THE MISSION
The society plays {len(self.games)} unfamiliar text games: {games}. Nobody is told the rules or what completes a level; they must be discovered by experimenting. {inst}
- The screen is a short text string. Each level has a limited number of steps; running out costs a life, and some states cost a life too; when no lives are left the game is over (RESET starts a new instance at level 1). Some games have a creative mode, entered and left with its toggle action: a sandbox whose steps do not count against the level's limit.
- play sends actions (up to {self.steps_per_tick} steps per tick) and reports the screen after each; {look}. search what=games shows each game's best progress and its players.
- A game's progress is the most levels any instance has completed; the score is the mean over games of levels completed out of levels.
- What each game says about itself (its objective and special actions, not its rules):{tasks or ' nothing'}{self._publish_brief()}{self._sync_brief()}{self.explore_brief()}"""

    def _sync_brief(self) -> str:
        if not self.sync:
            return ""
        levels = ("\n- Levels: play results say which level your instance is on (level K of N); problems, plans and findings "
                  "name the level being worked on (level 1 is the first).")
        if self.board_on:
            return levels + ("\n- The board: everything is public and nothing posted is ever deleted. Everything others post "
                             "reaches you once, at the start of your next turn, grouped by type; dead ends on your level and "
                             "the published solution of the level your instance is on also reach you during your turn. "
                             "search what=posts ranks the whole history by relevance and by how much others have used each "
                             "entry. Name the E-ids of what you used: it helps others find them, and the pieces a solver "
                             "cites earn their authors credit.")
        return levels + ("\n- Sync: everything is public. Your latest plays are kept in your recent work (post them with "
                         "'to' if others should see them).")

    def _publish_brief(self) -> str:
        return ("\n- Solved levels are published: when any instance completes a level for the first time, the moves that "
                "solved it (from the start of that level) are posted on the global board, and replay runs them on your "
                "instance from the start of that level; on another instance they may not work, which itself tells you "
                "which details matter." if self.publish else "")

    # ------------------------------------------------------------------ scoring

    def score(self) -> dict:
        per = {g: self.best[g] / max(1, self.win.get(g, 1)) for g in self.games}
        return {"score": round(sum(per.values()) / len(per), 4), "coverage": round(sum(v > 0 for v in per.values()) / len(per), 4),
                "score_base": None, "score_composite": None, "structure_f1": None, "val": None,
                "levels": sum(self.best.values()), "steps": sum(self.used.values()), "instances": self.instances,
                **({"stale_refusals": self.stale_refusals} if self.shared else {}),
                **({"lease_refusals": self.lease_refusals, "holds": sum(1 for x in self.lease_log if "until" in x)}
                   if self.lease_on else {}),
                **({key: self._k.counts[key] for key in ("pushed", "collapsed", "interrupts")}
                   if getattr(getattr(self, "_k", None), "sync", False) else {}),
                **(self.explore_score(self._k) if getattr(self, "_k", None) else {})}

    def author_scores(self) -> dict:
        return {}

    def dominant_modules(self, tick: int, window: int = 3) -> dict:
        return {}

    def agent_export(self, aid: str) -> dict:
        g = self.assigned.get(aid)
        return {"instruments": [g] if g else [], "record": self.track_record(aid), "experiments": self.used[aid]}

    def export(self) -> dict:
        return {"kind": "dig", "structure": "dig", "games": self.games, "best": self.best, "win": self.win,
                "firsts": self.firsts, "assigned": self.assigned, "log": self.log[-5000:], "descriptions": self.descriptions,
                "copies": {f"{a}|{g}": st for (a, g), st in self.state.items()}, "copies_mode": "shared" if self.shared else "own",
                **({"moves_on": {g: list(m) for (_, g), m in self.moves_on.items()}} if self.shared else {}),
                **({"leases": self.lease_log} if self.lease_on else {}),
                **(self.explore_export(self._k) if (self.explore or self.delegation) and getattr(self, "_k", None) else {})}

    # ------------------------------------------------------------------ scripted agents (mock backend)

    def mock_actions(self, k, aid, t, rng):
        g = self.assigned.get(aid)
        st = self.state.get(self._ck(aid, g), {}) if g else {}
        acts = [x for x in (st.get("actions") or []) if x != "/"] or ["1"]
        moves = [rng.choice(acts) for _ in range(5)]
        if st.get("state") == "GAME_OVER":
            moves = ["RESET"] + moves
        hold = [("hold", {"op": rng.choice(["take", "request", "release"]), "ticks": 2})] if self.lease_on else []
        return hold + [("play", {"actions": moves}), ("look", {})] + self.explore_mock(k, aid, t, rng)
