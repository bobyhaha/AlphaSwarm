"""ARC-AGI-3 as a world: interactive games whose rules and goals must be discovered by exploring.

The public ARC-AGI-3 games (arc-agi package; the games run locally) are played in parallel copies. Each agent is assigned
one game and plays its own copy of it with `play`; nothing about a game (what the actions do, what wins a level) is told
to anyone. What one agent learns can help the others only through communication: boards, links, the substrate, a lead.
A game's progress is the most levels any of its copies has completed, and the score is the mean over games of levels
completed / levels to win. The isolated condition gives the same parallel copies without communication, so comparing the
two measures what sharing adds to parallel exploration (the case a research society is for).

Frames are 64x64 grids of colors 0-15, shown as 64 rows of hex digits. Actions are ACTION1-ACTION7 (ACTION6 is a click at
x,y) and RESET; each game lists the actions it accepts.
"""
from __future__ import annotations

import copy
import random
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .evidence import Counters, changes, object_at, pack, state_key, unpack
from .explore import Explore
from .kernel import ActionError, clip

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "ext" / "arcagi3_envs"
HEX = "0123456789abcdef"

PLAY_TOOL = {
    "name": "play",
    "description": "Play your copy of your game: a list of up to 15 actions, each 'ACTION1'..'ACTION7' or 'RESET'; a click is "
                   "{\"click\": [x, y]} (ACTION6 at column x, row y, 0-63). Returns, step by step, how many cells changed and "
                   "where, the level and state after each step, and, if show=true, the final frame. Results are posted to "
                   "the board in 'to' (default your private board). Limited steps per tick.",
    "parameters": {"type": "object", "properties": {
        "actions": {"type": "array", "items": {}, "description": "e.g. [\"ACTION1\", \"ACTION1\", {\"click\": [12, 30]}]"},
        "show": {"type": "boolean", "description": "include the final frame (64 rows of hex colors)"},
        "entry": {"type": "string", "description": "instead of actions: run the moves written in an entry (E-id) or a receipt (R-id)"},
        "args": {"type": "object", "description": "with entry: values for placeholders such as {x}"},
        "restart_game": {"type": "boolean", "description": "allow a RESET that would restart the whole game (a RESET at "
                                                           "the start of a level does; see the brief)"},
        "to": {"type": "string", "description": "board for a one-line record of what happened"},
        "note": {"type": "string", "description": "in your last reply of a tick: your closing note (what these moves "
                                                  "test, what you learned, what next), kept as your public note"}},
        "required": []}}
REPLAY_TOOL = {
    "name": "replay",
    "description": "Reproduce a published solution on your copy: resets your copy to the start of its current level, then "
                   "runs the moves that solved that level (as published on the global board). Long solutions continue "
                   "automatically on your next replay calls until done; each move uses one of your steps.",
    "parameters": {"type": "object", "properties": {
        "note": {"type": "string", "description": "in your last reply of a tick: your closing note, kept as your public "
                                                  "note"}}}}
# with sync there are no private boards: play records stay in the player's own recent work unless posted with 'to'
SYNC_PLAY_TOOL = dict(PLAY_TOOL, description=PLAY_TOOL["description"].replace(
    "Results are posted to the board in 'to' (default your private board).",
    "Your latest plays and their receipts are kept in your recent work; give 'to' (e.g. global) to also post a "
    "one-line record."))
LOOK_TOOL = {
    "name": "look",
    "description": "See the current frame of your copy: 64 rows of 64 hex digits (colors 0-f), or a region "
                   "[x0, y0, x1, y1] of it; plus its level, state and the actions it accepts.",
    "parameters": {"type": "object", "properties": {"region": {"type": "array", "items": {"type": "integer"}}}}}
# evidence=1: plays report what changed as objects and keep every frame; look can open any receipt's frames and say
# what moves did before from the state your copy is in, without playing
EVIDENCE_PLAY = {
    "description": " Each step says what changed as objects (moved, recoloured, grew or shrank, appeared, gone), and every "
                   "frame is kept in the play's receipt, in-between frames too. Before playing, say in expect what you "
                   "think will change; afterwards, name every change you see, expected or not.",
    "expect": {"type": "string", "description": "what you expect these moves to change, said before you see the result"}}
EVIDENCE_LOOK = {
    "description": " With receipt (an R-id, anyone's): see a frame that play kept, with step (the move's number in that "
                   "play; 0 = the frame before its first move) and frame (for a move that returned several frames; "
                   "default the last). With outcome (a list of actions, as in play): what those moves did before from "
                   "the state your copy is in now, as recorded in receipts, without playing them.",
    "receipt": {"type": "string"}, "step": {"type": "integer"}, "frame": {"type": "integer"},
    "outcome": {"type": "array", "items": {}}}


def with_evidence(tool: dict, extra: dict) -> dict:
    props = {k: v for k, v in extra.items() if k != "description"}
    return dict(tool, description=tool["description"] + extra["description"],
                parameters=dict(tool["parameters"], properties={**tool["parameters"]["properties"], **props}))


def frame_text(g: np.ndarray, region=None) -> str:
    x0, y0, x1, y1 = (0, 0, 63, 63) if not region else [max(0, min(63, int(v))) for v in region]
    rows = [f"{y:02d} " + "".join(HEX[int(c) & 15] for c in g[y, x0:x1 + 1]) for y in range(y0, y1 + 1)]
    return f"columns {x0}-{x1}, rows {y0}-{y1}:\n" + "\n".join(rows)


class ArcWorld(Explore):
    tool_names = ("play", "look")
    TOOLS = [PLAY_TOOL, LOOK_TOOL]
    search_kinds = ("games",)
    has_modules = False

    def __init__(self, n_agents: int, seed: int = 0, single: bool = False, world_agents: int | None = None,
                 opts: dict | None = None):
        from arc_agi import Arcade
        opts = opts or {}
        self.rng = random.Random(seed)
        ENV_DIR.mkdir(parents=True, exist_ok=True)
        self.arcade = Arcade(environments_dir=str(ENV_DIR))
        available = sorted(e.game_id for e in self.arcade.get_environments())
        wanted = [g.strip() for g in str(opts.get("games", "")).split(",") if g.strip()]
        if wanted:
            self.games = [next(g for g in available if g.startswith(w)) for w in wanted]
        else:
            self.games = sorted(self.rng.sample(available, min(int(opts.get("n_games", 4)), len(available))))
        self.steps_per_tick = int(opts.get("steps", 15))
        # publish_solutions=1: when a copy completes a level, the world posts the moves that solved it (from the start of
        # that level) on the global board, so others can replay them and build on the progress
        self.publish = str(opts.get("publish_solutions", "")).lower() in ("1", "true", "yes")
        self.trace: dict[tuple, list] = defaultdict(list)  # (agent, game) -> moves since the current level began
        self.published: set = set()
        self.solutions: dict[tuple, list] = {}  # (game, level reached) -> the moves that solved it
        self.pending: dict[tuple, list] = {}  # (agent, game) -> moves of a replay still to run
        if self.publish:
            self.tool_names = type(self).tool_names + ("replay",)
            self.TOOLS = type(self).TOOLS + [REPLAY_TOOL]
        self.single = single
        aids = [f"A{i + 1}" for i in range(n_agents)]
        order = aids[:]
        self.rng.shuffle(order)
        self.assigned = {a: self.games[j % len(self.games)] for j, a in enumerate(order)}
        if single:  # one agent plays every game, with the society's total steps
            self.assigned = {aids[0]: self.games[0]}
            self.steps_per_tick *= (world_agents or n_agents)
        self.home = {a: [g] for a, g in self.assigned.items()}  # used by the mock backend
        self.tasks = {g: g for g in self.games}
        self.copies: dict[tuple, object] = {}  # (agent, game) -> environment
        self.frames: dict[tuple, np.ndarray] = {}
        self.state: dict[tuple, dict] = {}
        self.steps = defaultdict(int)  # (agent, tick) -> steps used
        self.used = defaultdict(int)
        self.best: dict[str, int] = {g: 0 for g in self.games}  # game -> most levels any copy completed
        self.win: dict[str, int] = {}
        self.firsts: list[dict] = []  # first time any copy reached each level of each game
        self.log: list[dict] = []
        self.focus = defaultdict(list)
        # sync=1 (with the kernel's sync): one public substrate; each agent's plays are kept in its own recent work
        self.sync = str(opts.get("sync", "")).lower() in ("1", "true", "yes")
        self.recent_plays: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
        if self.sync:
            self.TOOLS = [SYNC_PLAY_TOOL if t is PLAY_TOOL else t for t in self.TOOLS]
        self._init_explore(opts)
        self._now = None
        self.evidence_on = self.explore and str(opts.get("evidence", "")).lower() in ("1", "true", "yes")
        if self.evidence_on:
            self.TOOLS = [with_evidence(t, EVIDENCE_PLAY) if t["name"] == "play" else
                          with_evidence(t, EVIDENCE_LOOK) if t["name"] == "look" else t for t in self.TOOLS]
        self.counters = Counters()
        self.fstore: list[bytes] = []  # every frame a copy has shown, packed; a frame id is its index
        self.cur_fid: dict[tuple, int] = {}  # (agent, game) -> the frame id of the copy's current frame
        self.obs_frames: dict[tuple, list] = {}  # (agent, game) -> the frames the last step returned (in-between ones too)
        self.trans: dict[tuple, list] = defaultdict(list)  # (game, level, move) -> every recorded run of that move
        self._tindex: dict[tuple, dict] = {}  # (state key, move) -> the first recorded run from that state
        self.targets: dict[tuple, dict] = defaultdict(dict)  # (game, level) -> what a move acts on -> its runs, live or not
        self.told: set = set()  # (agent, game, level, what a move acts on): each note once
        self._tversion = -1
        self.ev_stats = defaultdict(int)
        # start=<json file {game: [moves]}>: each copy plays these moves before the run (a test that starts at a later level)
        self.start_moves = {}
        # trust=1: plan op=test reruns a receipt on a spare copy, from the state its play started in, made from a snapshot of
        # the copy at that level's start (the games are deterministic)
        self.snaps: dict[tuple, tuple] = {}  # (agent, game, levels completed) -> (copy, frame) at that level's start
        self.supports_replay = self.trust_on
        if opts.get("start"):
            import json
            self.start_moves = json.loads(Path(str(opts["start"])).read_text())

    # ------------------------------------------------------------------ copies

    def _copy(self, aid: str, game: str):
        key = (aid, game)
        if key not in self.copies:
            env = self.arcade.make(game, seed=0)
            obs = env.reset()
            for item in self.start_moves.get(game.split("-")[0], []):  # not counted as steps, not in any receipt
                action, data, _ = self._parse(item)
                prev = np.array(obs.frame[-1]) if obs.frame else None
                obs = (env.step(action, data=data) if data else env.step(action)) or obs
                if prev is not None and obs.frame:
                    self.counters.learn(game, prev, np.array(obs.frame[-1]))
            self.copies[key] = env
            self._record(key, obs)
            self.best[game] = max(self.best[game], self.state[key]["levels"])  # levels a start file played count as reached
            self._snap(key, env)
            if self.evidence_on:
                self.cur_fid[key] = self._keep(self.frames[key])
        return self.copies[key]

    def _earlier_records(self, k, game: str):
        """The start file's moves, split by level, posted as the world's records of how those levels were solved (an
        earlier society's), so agents can replay or study them like any published solution. A start file may also
        carry that society's explanations ("explanations": {game: {level: [text, ...]}}), posted with the records."""
        moves = self.start_moves.get(game.split("-")[0], [])
        if not moves:
            return
        env = self.arcade.make(game, seed=0)
        obs, done, cur = env.reset(), 0, []
        for item in moves:
            action, data, label = self._parse(item)
            obs = (env.step(action, data=data) if data else env.step(action)) or obs
            cur.append(label)
            if int(obs.levels_completed) > done:
                done = int(obs.levels_completed)
                if (game, done) not in self.published:
                    self.published.add((game, done))
                    self.solutions[(game, done)] = list(cur)
                    e = k.post("global", "lab", "result", f"{game.split('-')[0]} level {done} was solved before this run "
                               f"(by an earlier society) with these {len(cur)} moves from the start of that level (replay "
                               f"them with play): {', '.join(cur)}", notify=False)
                    self.entry_meta[e.id] = {"game": game, "level": done, "type": "solution"}
                    self.solution_of[(game, done)] = e.id
                    for text in (self.start_moves.get("explanations") or {}).get(game.split("-")[0], {}).get(str(done), []):
                        x = k.post("global", "lab", "result", f"EXPLANATION of the {game.split('-')[0]} level {done} "
                                   f"solution ({e.id}), written by an agent of an earlier society (its E- and R-ids refer "
                                   f"to that society's records): {text}", refs=[e.id], notify=False)
                        self.entry_meta[x.id] = {"game": game, "level": done, "type": "explanation"}
                        self.explanations[e.id].append(x.id)
                cur = []

    def _snap(self, key, env):
        """trust=1: keep the copy as it is at the start of a level, so runs from that level can be rerun exactly."""
        if self.trust_on:
            self.snaps.setdefault((key[0], key[1], self.state[key]["levels"]), (copy.deepcopy(env), self.frames[key].copy()))

    def replay_receipt(self, rid: str) -> tuple:
        """Run a receipt's moves again on a spare copy, from the state its play started in: (True or False, what
        happened), or (None, why it cannot be run). The games are deterministic, so a mismatch means the record and the
        game disagree."""
        r = self.receipts.get(rid)
        snap = self.snaps.get((r["agent"], r["game"], r["level_start"] - 1)) if r else None
        if not r or snap is None or r.get("start") is None or not r.get("results"):
            return None, "no record of the state it started from"
        env, frame, levels = copy.deepcopy(snap[0]), snap[1], r["level_start"] - 1

        def run(label):
            nonlocal frame, levels
            action, data, _ = self._parse(label)
            obs = env.step(action, data=data) if data else env.step(action)
            if obs is None:
                return None
            g = np.array(obs.frame[-1]) if obs.frame else frame
            changed, frame = int((g != frame).sum()), g
            if int(obs.win_levels) > 0:  # a GAME_OVER frame reports 0/0: the copy keeps its levels
                levels = int(obs.levels_completed)
            return changed, levels, obs.state.name
        for m in r["start"]:
            run(m)
        for i, want in enumerate(r["results"], 1):
            got = run(want["move"])
            if got != (want["changed"], want["levels"], want["state"]):
                seen = f"{got[0]} cells changed, level {got[1]} {got[2]}" if got else "no response"
                return False, (f"step {i} {want['move']}: {seen}; the record says {want['changed']} cells changed, level "
                               f"{want['levels']} {want['state']}")
        return True, f"all {len(r['results'])} moves gave the recorded result again, from {self.start_text(r)}"

    def _keep(self, f) -> int:
        self.fstore.append(pack(f))
        return len(self.fstore) - 1

    def _record(self, key, obs):
        if self.evidence_on and obs.frame:
            self.obs_frames[key] = [np.array(f) for f in obs.frame]
        self.frames[key] = np.array(obs.frame[-1]) if obs.frame else self.frames.get(key, np.zeros((64, 64), int))
        prev = self.state.get(key, {})
        win = int(obs.win_levels)
        degenerate = win == 0  # a GAME_OVER frame reports "level 0/0": keep the copy's levels and the game's level count
        self.state[key] = {"levels": prev.get("levels", 0) if degenerate else int(obs.levels_completed),
                           "win": prev.get("win", self.win.get(key[1], 0)) if degenerate else win, "state": obs.state.name,
                           "actions": [f"ACTION{int(a)}" if int(a) > 0 else "RESET" for a in (obs.available_actions or [])]}
        if win > 0:
            self.win[key[1]] = max(self.win.get(key[1], 0), win)

    def setup(self, k):
        self._k = k
        if self.explore:
            k.rank = dict(self.rules["substrate"]["rank"])
        for aid, game in self.assigned.items():
            self._copy(aid, game)
            st = self.state[(aid, game)]
            if k.sync:  # no private boards: the game and its accepted actions are shown in every turn's view
                continue
            k.post(f"private:{aid}", "lab", "note", f"You play your own copy of game {game.split('-')[0]} (levels to win: "
                   f"{st['win']}; accepted actions: {', '.join(st['actions'])}). What the actions do and what completes "
                   f"a level are unknown: find out by playing.", notify=False)
        if self.publish and self.start_moves:  # levels a start file played: their moves become records like solutions
            for game in sorted(set(self.assigned.values())):
                self._earlier_records(k, game)

    def on_tick_start(self, k):
        pass

    def on_tick_end(self, k):
        if self.model_on:  # check the shared models on this tick's moves and pay for predictions (a coroutine)
            return self.model_tick_end(k)

    # ------------------------------------------------------------------ actions

    def act(self, k, aid: str, name: str, a: dict):
        self._now = name
        return getattr(self, f"_{name}")(k, aid, a)

    def _parse(self, item):
        from arcengine import GameAction
        if isinstance(item, dict) and "click" in item:
            x, y = item["click"]
            return GameAction.ACTION6, {"x": int(x), "y": int(y)}, f"click({int(x)},{int(y)})"
        s = str(item).strip().upper()
        click = re.fullmatch(r"CLICK\((\d+),\s*(\d+)\)", s.replace(" ", ""))
        if click:
            return GameAction.ACTION6, {"x": int(click.group(1)), "y": int(click.group(2))}, f"click({click.group(1)},{click.group(2)})"
        if s.isdigit():
            s = f"ACTION{s}"
        if s == "RESET":
            return GameAction.RESET, None, "RESET"
        if s.startswith("ACTION") and s[6:].isdigit() and 1 <= int(s[6:]) <= 7:
            return getattr(GameAction, s), None, s
        raise ActionError(f"unknown action {item!r}: use ACTION1..ACTION7, RESET or {{\"click\": [x, y]}}")

    def _play(self, k, aid, a):
        if self.hyp_on:  # the pieces these moves rely on, paid for understanding if they complete a level first
            self.cites[aid] = self._piece_ids(k, a.get("cites"), str(a.get("note") or ""))
            if self.board_on and self.cites[aid]:
                k.cite_use(aid, self.cites[aid], f"{aid} play t{k.tick}")
        if self.explore and a.get("entry"):  # run the moves written in an entry or a receipt
            ref = str(a["entry"]).strip().upper()
            moves = self.entry_moves(k, aid, ref, a.get("args") or {})
            self._running_entry[aid] = ref
            try:
                op, out, target = self._play(k, aid, {**a, "entry": None, "actions": moves})
            finally:
                self._running_entry.pop(aid, None)
            self.entry_runs[ref]["runs"] += 1
            self.entry_runs[ref]["by"].add(aid)
            return op, f"ran {ref}: " + out, target
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        acts = a.get("actions") or []
        k._need(isinstance(acts, list) and acts, "actions must be a non-empty list (or give an entry to run)")
        key = (aid, game)
        env = self._copy(aid, game)
        allowed = self.steps_allowed(aid, k.tick)
        left = allowed - self.steps[(aid, k.tick)]
        k._need(left > 0, f"you have used all {allowed} steps this tick")
        lines, moves_done, level_start, evs = [], [], self.state[key]["levels"], []
        start, results = list(self.trace[key]), []  # trust=1: where this play started and what each step did
        game_over = None  # (level, moves from the level start) when this play ends the copy's game
        lost_by_reset = None  # (level, level it went back to) when a RESET restarted the whole game
        for item in acts[:min(15, left)]:
            action, data, label = self._parse(item)
            if (label == "RESET" and self.carry and self.state[key]["levels"] > 0 and not self.trace[key]
                    and self.state[key]["state"] != "GAME_OVER" and not a.get("restart_game")):
                # at a level's start (not after a GAME_OVER), RESET restarts the whole game
                lines.append(f"RESET: refused: your copy is at the start of level {self.state[key]['levels'] + 1}, where "
                             f"RESET restarts the whole game and loses {self.state[key]['levels']} completed level(s); "
                             f"pass restart_game=true to do it anyway")
                continue
            before, lv0 = self.frames[key], self.state[key]["levels"]
            obs = env.step(action, data=data) if data else env.step(action)
            self.steps[(aid, k.tick)] += 1
            self.used[aid] += 1
            if obs is None:
                lines.append(f"{label}: no response")
                continue
            self._record(key, obs)
            self.trace[key].append(label)
            moves_done.append(label)
            if self.explore:
                self.note_tried(game, lv0, label)
            g = self.frames[key]
            if self.explore:
                self._novel(k, aid, game, g, before, self.state[key]["levels"])
            diff = np.argwhere(g != before)
            where = (f" in rows {diff[:, 0].min()}-{diff[:, 0].max()}, columns {diff[:, 1].min()}-{diff[:, 1].max()}"
                     if len(diff) else "")
            if self.evidence_on:  # what changed, as objects; the frames are kept for the receipt
                evs.append(self._evidence_step(key, game, lv0, label, data, before, g))
                where = (" (" + "; ".join(evs[-1]["report"][:4]) + ")" + (f" [{len(evs[-1]['frames'])} frames]"
                                                                          if len(evs[-1]["frames"]) > 1 else ""))
            st = self.state[key]
            if self.model_on:  # every move is a transition the shared model is checked against
                self.models.record(game, lv0 + 1, label, before, g, "level_up" if st["levels"] > lv0 else
                                   "game_over" if st["state"] == "GAME_OVER" else "", k.tick)
            note = ""
            if st["levels"] > lv0:
                note = f" LEVEL {st['levels']} COMPLETED"
                moves, self.trace[key] = self.trace[key], []
                if self.publish and (game, st["levels"]) not in self.published:
                    self.published.add((game, st["levels"]))
                    self.solutions[(game, st["levels"])] = list(moves)
                    e = k.post("global", "lab", "result", f"{game.split('-')[0]} level {st['levels']} was solved by {aid} "
                               f"with these {len(moves)} moves from the start of that level (replay them with play): "
                               f"{', '.join(moves)}", notify=False)
                    self.entry_meta[e.id] = {"game": game, "level": st["levels"], "type": "solution"}
                    self.solution_of[(game, st["levels"])] = e.id
                if st["levels"] > self.best[game]:
                    self.best[game] = st["levels"]
                    self.firsts.append({"t": k.tick, "game": game, "level": st["levels"], "agent": aid})
                    if self.explore:
                        self._first_solve(k, aid, game, st["levels"])
                self._snap(key, env)
            lines.append(f"{label}: {len(diff)} cells changed{where}; level {st['levels']}/{st['win']} {st['state']}{note}")
            results.append({"move": label, "changed": int(len(diff)), "levels": st["levels"], "state": st["state"]})
            self.log.append({"t": k.tick, "agent": aid, "game": game, "action": label, "changed": int(len(diff)),
                             "levels": st["levels"], "state": st["state"]})
            if st["state"] == "GAME_OVER" and label != "RESET":
                game_over = (lv0 + 1, list(self.trace[key]))
            if label == "RESET" and st["levels"] < lv0:
                lost_by_reset = (lv0 + 1, st["levels"] + 1)
            if label == "RESET" or st["state"] == "GAME_OVER":
                self.trace[key] = []  # a reset restarts the level, so its trace starts over
            if st["state"] in ("WIN", "GAME_OVER"):
                break
        short = game.split("-")[0]
        receipt = (" receipt " + self._receipt(k, aid, game, level_start, moves_done, lines, self.state[key]["levels"],
                                               **({"start": start, "results": results} if self.trust_on else {}))
                   if self.explore and moves_done else "")
        if game_over and self.carry and receipt:  # the engine publishes the failure as evidence, not left to notes
            self._engine_dead_end(k, aid, game, game_over[0], game_over[1], receipt.split()[-1])
        if lost_by_reset and self.carry and receipt:
            self._engine_dead_end(k, aid, game, lost_by_reset[0], [], receipt.split()[-1],
                                  text=f"RESET at the start of level {lost_by_reset[0]} restarted the whole game: "
                                       f"{aid}'s copy went back to level {lost_by_reset[1]}. RESET in the middle of a "
                                       f"level restarts only that level")
        if k.sync and not a.get("to"):  # kept in the player's own recent work, not posted
            self.recent_plays[aid].append(f"t{k.tick} {short}{receipt}: {clip(' | '.join(lines[-4:]), 400)}")
        else:
            board = k.board_id(aid, a.get("to", "private"))
            if k.can_write(aid, board):
                k.post(board, aid, "data", f"{aid} played {short}: " + " | ".join(lines[-6:]), notify=False)
        out = (f"{short}, your copy:\n" + "\n".join(lines) + f"\n[steps left this tick: "
               f"{self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]}]" + (f" [engine{receipt}]" if receipt else ""))
        if self.evidence_on and receipt:
            expect = clip(" ".join(str(a.get("expect") or "").split()), 300)
            notes = self._evidence_after(k, aid, game, receipt.split()[-1], evs, expect)
            if expect:
                out = f"You expected: {expect}\nWhat happened:\n" + out
            out += "".join("\nNOTE: " + x for x in notes)
        rep = self.repeats_refuted(game, level_start + 1, moves_done) if self.hyp_on and not a.get("_replay") else []
        if rep:  # information, not a block: the refutation is named where the moves are made
            self.refuted_repeats += 1
            out += "\nNOTE: these moves repeat part of a refuted test: " + "; ".join(
                f"{r} behind dead end {f['id']} ({f['agent']}, t{f['t']}): {clip(f['text'], 120)}" for f, r in rep) + \
                ". Repeating it teaches nothing new unless something else is different; if so, say what in your note."
        if a.get("show"):
            out += "\n" + frame_text(self.frames[key])
        return "lab.play", out, short

    def _replay(self, k, aid, a):
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        key = (aid, game)
        self._copy(aid, game)
        queue = self.pending.get(key)
        intro = ""
        if not queue:
            target = self.state[key]["levels"] + 1
            moves = self.solutions.get((game, target))
            k._need(moves, f"no published solution yet for level {target} of {game.split('-')[0]}")
            queue = ["RESET"] + list(moves)
            if self.carry:  # the understanding travels with the moves
                sid = self.solution_of.get((game, target))
                sol = k.entry(sid) if sid else None
                exps = [k.entry(x) for x in self.explanations.get(sid, [])][-2:]
                self.shown_explanations[aid].update(x.id for x in exps if x)
                solver = sol.content.split("The solver's findings on this game: ", 1)[1] if sol and \
                    "The solver's findings" in sol.content else ""
                intro = (f"WHY THESE MOVES WORK ({sid}): " + (" | ".join(clip(x.content, 400) for x in exps if x)
                                                              if exps else "no explanation posted yet")
                         + (f" | the solver's findings: {clip(solver, 400)}" if solver else "")
                         + ". After the replay, explain or challenge what you saw (plan op=explain or challenge).\n")
        left = self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]
        k._need(left > 0, "you have used all your steps this tick")
        now, rest = queue[:min(15, left)], queue[min(15, left):]
        self.pending[key] = rest
        lv0 = self.state[key]["levels"]
        _, out, ref = self._play(k, aid, {"actions": now, "_replay": True})
        done = self.state[key]["levels"] > lv0
        if done:
            self.pending[key] = []
        tail = (" Reached the next level." if done else f" {len(rest)} moves left: call replay again to continue."
                if rest else " Replay finished without reaching the next level.")
        if done and self.carry:
            known = [f for f in self.findings if f["game"] == game and f["backed"] and not f.get("superseded_at")][-3:]
            if known:
                tail += " Carried forward on this game: " + "; ".join(f"{f['id']} L{f['level']}: {clip(f['text'], 100)}"
                                                                      for f in known)
        return "lab.replay", intro + out + tail, ref

    # ------------------------------------------------------------------ evidence (evidence=1)

    def _evidence_step(self, key, game: str, lv0: int, label: str, data, before, g) -> dict:
        """One move: its frames kept, what it changed as objects, and the states it went from and to."""
        frames = self.obs_frames.get(key) or [g]
        b_fid = self.cur_fid.get(key)
        fids = [self._keep(f) for f in frames]
        self.cur_fid[key] = fids[-1]
        self.counters.learn(game, before, g)
        mask = self.counters.mask(game)
        self.ev_stats["reports"] += 1
        obj = object_at(before, data["x"], data["y"]) if data else None
        if obj and (obj[1] - obj[0] + 1) * (obj[3] - obj[2] + 1) > 400:
            obj = None  # a background or other large region is not one thing a click acts on
        return {"move": label, "level": lv0 + 1, "level_after": self.state[key]["levels"],
                "before": b_fid if b_fid is not None else self._keep(before), "frames": fids,
                "cells": int(((before != g) & ~mask).sum()), "xy": (data["x"], data["y"]) if data else None, "obj": obj,
                "path": tuple(self.trace[key][:-1]), "report": changes(before, g, mask)}

    def _reindex(self, game: str):
        """State keys leave out the counters; when a new counter is learned, every recorded run is keyed again."""
        if self._tversion == self.counters.version:
            return
        self._tversion, self._tindex = self.counters.version, {}
        for (g, level), runs in self.trans.items():
            mask = self.counters.mask(g)
            for r in runs:
                r["start"] = state_key(unpack(self.fstore[r["before"]]), mask, level)
                r["after"] = state_key(unpack(self.fstore[r["frames"][-1]]), mask, r["level_after"] + 1)
                self._tindex.setdefault((r["start"], r["move"]), r)

    @staticmethod
    def _same_target(r: dict, move: str, xy) -> bool:
        """The same move, or a click on the same object (a click acts on the region under it)."""
        if r["move"] == move:
            return True
        inside = lambda p, b: b is not None and p is not None and b[0] <= p[1] <= b[1] and b[2] <= p[0] <= b[3]
        return xy is not None and r["xy"] is not None and inside(xy, r["obj"])

    @staticmethod
    def _path_text(path) -> str:
        return ("from the level's start" if not path else
                "after " + ("... " if len(path) > 8 else "") + ", ".join(path[-8:]) + " (moves since the level's start)")

    def _evidence_after(self, k, aid: str, game: str, rid: str, evs: list[dict], expect: str) -> list[str]:
        """Keep the play's frames in its receipt, then say what the recorded runs of the same moves show: a move that did
        something from another state (its effect depends on the state), and moves already played from these states."""
        r = self.receipts[rid]
        r.update({"before": evs[0]["before"] if evs else None, "frames": [e["frames"] for e in evs],
                  "reports": [e["report"] for e in evs], "path": list(evs[0]["path"]) if evs else [], "expect": expect})
        r.setdefault("start", r["path"])  # the moves since the level began, as trust=1 reads a receipt's state
        self.ev_stats["expects"] += bool(expect)
        self._reindex(game)
        mask = self.counters.mask(game)
        for ev in evs:  # keyed with the counters as known now
            ev["start"] = state_key(unpack(self.fstore[ev["before"]]), mask, ev["level"])
            ev["after"] = state_key(unpack(self.fstore[ev["frames"][-1]]), mask, ev["level_after"] + 1)
        notes, repeats, differ = [], [], []
        for i, ev in enumerate(evs, 1):
            prior = self._tindex.get((ev["start"], ev["move"]))
            if prior:
                repeats.append(prior)
                if prior["after"] != ev["after"]:
                    differ.append((i, ev, prior))
            runs = [x for x in self.trans[(game, ev["level"])] if x["start"] != ev["start"] and ev["move"] != "RESET"
                    and self._same_target(x, ev["move"], ev["xy"])]
            live, dead = [x for x in runs if x["cells"]], [x for x in runs if not x["cells"]]
            # said only for a rare effect (at most a quarter of the runs did something): a handle that stops at its limit
            # is common and visible; a control that works only in one state is what gets voted down
            rare = lambda n_live, n: n_live <= max(1, n // 4)
            told = (aid, game, ev["level"], ev["obj"] or ev["move"])
            if told in self.told:
                continue
            if len(notes) < 2 and not ev["cells"] and live and rare(len(live), len(runs) + 1):
                self.told.add(told)
                x = live[0]
                notes.append(f"{ev['move']} changed nothing here, but {x['move']}" + (f" (the object at r{x['obj'][0]}-{x['obj'][1]} "
                             f"c{x['obj'][2]}-{x['obj'][3]})" if x["obj"] else "") + f" changed {x['cells']} cells in "
                             f"{x['rid']} step {x['step']} ({x['agent']} t{x['t']}: {x['report'][0]}), "
                             f"{self._path_text(x['path'])}. Its effect depends on the state; look receipt={x['rid']} "
                             f"step={x['step']} shows that frame.")
            elif len(notes) < 2 and ev["cells"] and dead and rare(len(live) + 1, len(runs) + 1):
                self.told.add(told)
                x = dead[0]
                notes.append(f"{ev['move']} changed cells here ({self._path_text(ev['path'])}) but nothing in {len(dead)} "
                             f"recorded run(s) from other states (e.g. {x['rid']} step {x['step']}, {x['agent']} t{x['t']}, "
                             f"{self._path_text(x['path'])}): its effect depends on the state, so say which state it "
                             f"needs when you post it.")
        self.ev_stats["state_notes"] += len(notes)
        if differ:
            i, ev, prior = differ[0]
            notes.append(f"step {i} ({ev['move']}) went from a state recorded in {prior['rid']} step {prior['step']} but "
                         f"ended differently: the frame does not show everything that matters.")
        elif evs and len(repeats) == len(evs):
            first = min(repeats, key=lambda x: x["t"])
            self.ev_stats["exact_repeats"] += 1
            notes.append(f"all {len(evs)} moves repeat recorded runs from the same states (e.g. {first['rid']} by "
                         f"{first['agent']} t{first['t']}), with the same results. look outcome=[...] gives what recorded "
                         f"moves did from your copy's state without using steps.")
        for i, ev in enumerate(evs, 1):
            rec = {**ev, "rid": rid, "step": i, "agent": aid, "t": k.tick}
            self.trans[(game, ev["level"])].append(rec)
            self._tindex.setdefault((ev["start"], ev["move"]), rec)
            t = self.targets[(game, ev["level"])].setdefault(("obj", ev["obj"]) if ev["obj"] else ("move", ev["move"]),
                                                             {"live": [], "n_live": 0, "dead": 0})
            if ev["cells"]:
                t["n_live"] += 1
                t["live"].append(rec) if len(t["live"]) < 3 else None
            else:
                t["dead"] += 1
        return notes

    def evidence_marks(self, game: str, level: int, text: str) -> str:
        """For a post or guide line that names clicks: recorded runs where the same click (or a click on the same object)
        did something from one state and nothing from another."""
        if not self.evidence_on or game not in self.best:
            return ""
        out, targets = [], self.targets[(game, level)]
        for x, y in dict.fromkeys((int(a), int(b)) for a, b in re.findall(r"\(\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)", str(text))):
            hits = [v for (kind, b), v in targets.items() if (kind == "move" and b == f"click({x},{y})") or
                    (kind == "obj" and b[0] <= y <= b[1] and b[2] <= x <= b[3])]
            live, dead = [r for v in hits for r in v["live"]], sum(v["dead"] for v in hits)
            n_live = sum(v["n_live"] for v in hits)
            if live and dead and n_live <= max(1, (n_live + dead) // 4):  # a rare effect, as in play's notes
                r = min(live, key=lambda r: r["t"])
                out.append(f"a click at ({x},{y}) changed {r['cells']} cells in {r['rid']} step {r['step']} ({r['agent']} "
                           f"t{r['t']}, {self._path_text(r['path'])}) and nothing in {dead} other run(s)")
            if len(out) == 2:
                break
        return "; ".join(out)

    def _look_evidence(self, k, aid: str, game: str, a: dict):
        key = (aid, game)
        if a.get("receipt"):
            rid = str(a["receipt"]).strip().upper()
            r = self.receipts.get(rid)
            k._need(r is not None and r.get("frames") is not None, f"no receipt {rid} with kept frames")
            n = len(r["moves"])
            step = int(a["step"]) if a.get("step") is not None else n
            k._need(0 <= step <= n, f"step: 0 (the frame before the first move) to {n}")
            self.ev_stats["receipt_looks"] += 1
            if step == 0:
                fid, head = r["before"], f"the frame before its first move ({self._path_text(r.get('path'))})"
            else:
                fids = r["frames"][step - 1]
                j = int(a["frame"]) if a.get("frame") is not None else len(fids)
                k._need(1 <= j <= len(fids), f"frame: 1 to {len(fids)} for step {step}")
                fid = fids[j - 1]
                head = (f"step {step} of {n}, {r['moves'][step - 1]}" + (f", frame {j} of {len(fids)}" if len(fids) > 1 else "")
                        + f"; what it changed: {'; '.join(r['reports'][step - 1])}")
            return "lab.look", (f"{rid} ({r['agent']} t{r['t']}, {game.split('-')[0]} level {r['level_start']}), {head}\n"
                                + frame_text(unpack(self.fstore[fid]), a.get("region"))), None
        moves = [self._parse(x)[2] for x in (a.get("outcome") or [])][:15]
        k._need(moves, "outcome: a list of actions, as in play")
        self._reindex(game)
        st = self.state[key]
        cur, out = state_key(self.frames[key], self.counters.mask(game), st["levels"] + 1), []
        for m in moves:
            r = self._tindex.get((cur, m))
            if not r:
                out.append(f"{m}: not recorded from this state; play it to find out")
                break
            out.append(f"{m}: {r['cells']} cells changed ({'; '.join(r['report'][:4])}) [{r['rid']} step {r['step']}, "
                       f"{r['agent']} t{r['t']}]")
            cur = r["after"]
        self.ev_stats["outcome_looks"] += 1
        self.ev_stats["outcome_hits"] += sum(not x.endswith("find out") for x in out)
        return "lab.look", (f"what these moves did before from the state your copy is in now (level {st['levels'] + 1}), "
                            f"as recorded in receipts; no steps used:\n" + "\n".join(out)), None

    def evidence_score(self) -> dict:
        return {f"ev_{k}": v for k, v in sorted(self.ev_stats.items())} | {"ev_frames": len(self.fstore)}

    def _look(self, k, aid, a):
        game = self.assigned.get(aid)
        k._need(game, "you have no game")
        key = (aid, game)
        self._copy(aid, game)
        if self.evidence_on and (a.get("receipt") or a.get("outcome")):
            return self._look_evidence(k, aid, game, a)
        st = self.state[key]
        return "lab.look", (f"{game.split('-')[0]} (your copy): level {st['levels']}/{st['win']} {st['state']}; accepted "
                            f"actions {', '.join(st['actions'])}\n" + frame_text(self.frames[key], a.get("region"))), None

    # ------------------------------------------------------------------ views

    def search(self, k, aid, what, words):
        out = []
        for g in self.games:
            players = sorted((x for x, gg in self.assigned.items() if gg == g), key=lambda x: int(x[1:]))
            plans = (f"; open plans on level {self._problem(g)}: " + "; ".join(f"{p['agent']} {p['hypothesis'][:60]}" for p in self.open_plans(k, g))
                     if self.explore and self.open_plans(k, g) else "")
            out.append(f"{g.split('-')[0]}: best {self.best[g]}/{self.win.get(g, '?')} levels; players {', '.join(players)}{plans}")
        return out

    def agent_tags(self, aid: str) -> str:
        g = self.assigned.get(aid)
        return f"plays {g.split('-')[0]}" if g else ""

    def track_record(self, aid: str) -> str:
        g = self.assigned.get(aid)
        if not g or (aid, g) not in self.state:
            return ""
        base = f"record: level {self.state[(aid, g)]['levels']} in {g.split('-')[0]}"
        return base + ("; " + self.record_of(aid) if self.explore else "")

    def group_record(self, gid: str) -> str:
        return ""

    def observe(self, k, aid: str) -> str:
        g = self.assigned.get(aid)
        if not g:
            return "ARC: no game."
        st = self.state.get((aid, g), {})
        copy = (f"on level {st.get('levels', 0) + 1} of {st.get('win', '?')}, {st.get('levels', 0)} completed" if k.sync
                else f"level {st.get('levels', 0)}/{st.get('win', '?')}")
        return (f"ARC: you play game {g.split('-')[0]} (your copy: {copy}, "
                f"{st.get('state', '?')}); society's best on it: {self.best[g]}. Steps left this tick "
                f"{self.steps_allowed(aid, k.tick) - self.steps[(aid, k.tick)]}."
                + (f" Accepted actions: {', '.join(st.get('actions') or [])}." if k.sync else "")
                + ("\n" + self.explore_observe(k, aid) if (self.explore or self.delegation) else ""))

    def sync_rank(self, k, aid: str, e) -> int:
        """What reaches an agent first (kernel sync): 0 urgent, also delivered during its turn (a dead end on the level
        its copy is on or the frontier, the published solution of the level its copy is on, a challenge to its finding);
        1 claims and findings on those levels; 2 the rest about its game; 3 other games."""
        mine = self.assigned.get(aid)
        if not mine:
            return 3
        on = self.state.get((aid, mine), {}).get("levels", 0) + 1  # the level the agent's copy is on
        m = self.entry_meta.get(e.id)
        if m:
            if m["type"] == "challenge" and m.get("target") == aid:
                return 0
            if m["game"] == "lessons":  # lessons are shared by all games
                return 1
            if m["game"] != mine:
                return 3
            if m["type"] in ("solution", "explanation"):
                return 0 if m["level"] == on else 2 if m["type"] == "explanation" else 3
            if m["type"] in ("model", "guide"):
                return 1
            if m["level"] in (on, self._problem(mine)):
                return 0 if m["type"] == "dead_end" else 1
            return 2
        if e.author == "lab":
            return 2
        named = next((g for g in self.games if g.split("-")[0] in e.content), None)
        return 2 if (named or self.assigned.get(e.author)) == mine else 3

    def deliver_group(self, k, aid: str, e) -> tuple:
        """board=1: the section an entry is delivered in, and how much of its first line is shown: what bears on the
        agent's game first (dead ends, ideas, findings and solutions, the guide and lessons), then notes, plays, other games."""
        mine, m = self.assigned.get(aid), self.entry_meta.get(e.id)
        if m and m["game"] in (mine, "lessons"):
            t = m["type"]
            if t in ("dead_end", "challenge"):
                return 0, "Dead ends and challenges on your game", 160
            if t == "claim":
                return 1, "Ideas claimed on your game", 160
            if t in ("finding", "solution", "explanation"):
                return 2, "Findings, solutions and explanations on your game", 160
            return 3, "Guide, lessons and the shared model", 160
        if m or (e.author != "lab" and self.assigned.get(e.author) not in (None, mine)):
            return 6, "Other games", 80
        if e.kind == "data" or e.content.startswith("My last moves this tick"):
            return 5, "Plays", 90
        return 4, "Notes and other posts", 120

    def recent_work(self, k, aid: str) -> str:
        plays = list(self.recent_plays.get(aid, ()))
        return ("Your latest plays (cite their receipts as evidence):\n" + "\n".join(f"- {p}" for p in plays)) if plays else ""

    def brief(self) -> str:
        games = ", ".join(f"{g.split('-')[0]} ({self.win.get(g, '?')} levels)" for g in self.games)
        return f"""THE MISSION
The society plays {len(self.games)} unfamiliar games: {games}. Nobody is told the rules, what the actions do or what completes a level; they must be discovered by experimenting. Each game is played in parallel copies: every agent plays its own copy of the game it was assigned, so its moves do not disturb anyone else's copy, and anything one agent learns reaches the others only if it is shared.
- play sends actions to your copy (up to {self.steps_per_tick} steps per tick) and reports what changed after each step; look shows your copy's current frame as 64 rows of hex colors (0-f). search what=games shows each game's best progress and its players.
- A game's progress is the most levels any of its copies has completed; the score is the mean over games of levels completed out of levels to win. A level reached by one copy can be reached by another by repeating the moves that worked.{self._publish_brief()}{self._sync_brief()}{self.explore_brief()}"""

    def _sync_brief(self) -> str:
        if not self.sync:
            return ""
        levels = ("\n- Levels: in play results 'level K/N' counts levels completed, so you are on level K+1; problems, "
                  "plans and findings name the level being worked on (level 1 is the first).")
        if self.board_on:
            return levels + ("\n- The board: everything is public and nothing posted is ever deleted. Everything others post "
                             "reaches you once, at the start of your next turn, grouped by type; dead ends on your level and "
                             "the published solution of the level your copy is on also reach you during your turn. Your "
                             "latest plays are kept in your recent work. search what=posts ranks the whole history by "
                             "relevance and by how much others have used each entry (cited it in refs, builds_on or cites, "
                             "or named its E-id; the weight fades when use stops), and shows what each result cites and "
                             "what cites it. Name the E-ids of what you used: it helps others find them, and the pieces a "
                             "solver cites earn their authors credit.")
        return levels + ("\n- Sync: everything is public. Your latest plays are kept in your recent work (post them with "
                         "'to' if others should see them). What reaches you first: dead ends on your level and the published "
                         "solution of the level your copy is on (these also reach you during your turn), then claims and "
                         "findings there, then other posts about your game, then other games.")

    def _publish_brief(self) -> str:
        return ("\n- Solved levels are published: when any copy completes a level for the first time, the moves that solved "
                "it (from the start of that level) are posted on the global board, and replay reproduces them on your "
                "copy (it resets to the level start and continues across ticks), so the society builds on each other's "
                "progress." if self.publish else "")

    # ------------------------------------------------------------------ scoring

    def score(self) -> dict:
        per = {g: self.best[g] / max(1, self.win.get(g, 1)) for g in self.games}
        return {"score": round(sum(per.values()) / len(per), 4), "coverage": round(sum(v > 0 for v in per.values()) / len(per), 4),
                "score_base": None, "score_composite": None, "structure_f1": None, "val": None,
                "levels": sum(self.best.values()), "steps": sum(self.used.values()),
                **({key: self._k.counts[key] for key in ("pushed", "collapsed", "interrupts")}
                   if getattr(getattr(self, "_k", None), "sync", False) else {}),
                **(self.explore_score(self._k) if getattr(self, "_k", None) else {}),
                **(self.evidence_score() if self.evidence_on else {})}

    def author_scores(self) -> dict:
        return {}

    def dominant_modules(self, tick: int, window: int = 3) -> dict:
        return {}

    def agent_export(self, aid: str) -> dict:
        g = self.assigned.get(aid)
        return {"instruments": [g] if g else [], "record": self.track_record(aid), "experiments": self.used[aid]}

    def export(self) -> dict:
        return {"kind": "arc", "structure": "arc", "games": self.games, "best": self.best, "win": self.win,
                "firsts": self.firsts, "assigned": self.assigned, "log": self.log[-3000:],
                **({"evidence": dict(self.ev_stats), "receipts_full": [  # the reports and expectations, not the frames
                    {k: v for k, v in r.items() if k not in ("frames", "before")} for r in self.receipts.values()]}
                   if self.evidence_on else {}),
                "copies": {f"{a}|{g}": st for (a, g), st in self.state.items()},
                **(self.explore_export(self._k) if (self.explore or self.delegation) and getattr(self, "_k", None) else {})}

    # ------------------------------------------------------------------ scripted agents (mock backend)

    def mock_actions(self, k, aid, t, rng):
        g = self.assigned.get(aid)
        acts = (self.state.get((aid, g), {}).get("actions") or ["ACTION1"]) if g else ["ACTION1"]
        moves = [rng.choice([x for x in acts if x != "ACTION6"] or ["ACTION1"]) for _ in range(5)]
        return [("play", {"actions": moves})] + self.explore_mock(k, aid, t, rng)
