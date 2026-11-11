"""The society's shared model of each ARC game: one assembled piece instead of findings side by side (world option model=1).

The goal's picture is a jigsaw: agents solve local pieces, and the pieces must be combined into one coherent picture. Here
the picture is a program, step(grid, action, x, y) -> (new_grid, event), that predicts what one move does in a game. Every
move any copy makes is recorded as a transition (the frame before, the move, the frame after or the level/game-over
event), and the engine checks each version of the model against all of them.

Merging follows the goal's rules for combining work and resolving conflicts:
- an edit names the version it was written on (base); if the model changed underneath, it is refused and the editor
  takes in the current version and tries again;
- it is merged only if it still explains every transition the current version explains (nothing that worked breaks)
  and explains at least one more;
- credit is paid prequentially: for transitions recorded after the merge that the new version predicts and the one
  before it did not, so memorizing past moves earns nothing.

Agent code runs in alphateam/model_sandbox.py, in a separate process with limits.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

from .model_sandbox import check

SANDBOX = Path(__file__).resolve().parent / "model_sandbox.py"
IDENTITY = ("def step(grid, action, x, y):\n"
            "    # version 0: nothing changes (a move that changes nothing is explained)\n"
            "    return grid, ''\n")
CLICK = re.compile(r"click\((\d+),(\d+)\)")


def encode(label: str):
    """A move label as (action, x, y)."""
    m = CLICK.fullmatch(label)
    if m:
        return 6, int(m.group(1)), int(m.group(2))
    return int(label[6:]), None, None


class GameModels:
    def __init__(self, games: list[str], inherit: dict | None = None):
        self.games = list(games)
        self.versions: dict[str, list[dict]] = {}
        for g in self.games:
            code, why = IDENTITY, "start: nothing changes"
            got = (inherit or {}).get(g.split("-")[0])
            if got and not check(got):
                code, why = got, "inherited from an earlier society"
            self.versions[g] = [{"v": 0, "author": "lab", "t": 0, "code": code, "why": why, "parent": None, "gain": 0}]
        self.transitions: dict[str, list[dict]] = {g: [] for g in self.games}
        self.index: dict[str, dict] = {g: {} for g in self.games}
        self.results: dict[tuple, dict] = {}  # (game, version) -> {transition id: explained}
        # edge lines (rows or columns 0, 1, 62, 63) that change on most moves are counters (move budgets); they are
        # learned from the first moves and then ignored when checking predictions
        self.masks: dict[str, np.ndarray | None] = {g: None for g in self.games}
        self.mask_note: dict[str, str] = {g: "" for g in self.games}
        self.credited: set = set()  # (game, version, transition id) already paid

    # ------------------------------------------------------------------ transitions

    def record(self, game: str, level: int, label: str, before: np.ndarray, after: np.ndarray, event: str, t: int):
        if game not in self.transitions or label == "RESET":
            return None
        b = np.asarray(before, dtype=np.uint8).tobytes()
        a = np.asarray(after, dtype=np.uint8).tobytes()
        key = (hashlib.sha1(b).hexdigest(), label)
        tid = self.index[game].get(key)
        if tid is not None:  # the same move from the same frame (replays repeat many): counted, kept once
            self.transitions[game][tid]["n"] += 1
            return tid
        action, x, y = encode(label)
        tid = len(self.transitions[game])
        self.transitions[game].append({"id": tid, "t": t, "level": level, "label": label, "action": action, "x": x,
                                       "y": y, "before": b, "after": None if event else a, "after_raw": a,
                                       "event": event, "n": 1})
        self.index[game][key] = tid
        if self.masks[game] is None:
            self._learn_mask(game)
        return tid

    def _learn_mask(self, game: str, n: int = 20):
        moves = [c for c in self.transitions[game] if not c["event"]]
        if len(moves) < n:
            return
        changed = [np.frombuffer(c["before"], np.uint8).reshape(64, 64) != np.frombuffer(c["after"], np.uint8).reshape(64, 64)
                   for c in moves]
        mask, lines = np.zeros((64, 64), bool), []
        for i in (0, 1, 62, 63):
            if sum(d[i, :].any() for d in changed) >= 0.5 * len(changed):
                mask[i, :] = True
                lines.append(f"row {i}")
            if sum(d[:, i].any() for d in changed) >= 0.5 * len(changed):
                mask[:, i] = True
                lines.append(f"column {i}")
        self.masks[game] = mask
        if lines:
            self.mask_note[game] = (f"{' and '.join(lines)} change on most moves (a counter), so predictions are "
                                    f"checked outside them")
            for key in [x for x in self.results if x[0] == game]:
                del self.results[key]  # results so far were checked with the counter; check again without it

    def current(self, game: str) -> dict:
        return self.versions[game][-1]

    def version(self, game: str, v: int) -> dict:
        return self.versions[game][v]

    # ------------------------------------------------------------------ evaluation (in the sandbox)

    async def run(self, code: str, cases: list[dict], detail: int = 0, game: str | None = None) -> dict:
        why = check(code)
        if why:
            return {"ok": [False] * len(cases), "details": [], "error": why}
        if not cases:
            return {"ok": [], "details": [], "error": None}
        mask = self.masks.get(game) if game else None
        job = pickle.dumps({"code": code, "detail": detail, "mask": mask.tobytes() if mask is not None else None, "cases": [
            {"before": c["before"], "after": c["after"], "event": c["event"], "action": c["action"], "x": c["x"],
             "y": c["y"]} for c in cases]})
        proc = await asyncio.create_subprocess_exec(sys.executable, "-I", str(SANDBOX), stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(job), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": [False] * len(cases), "details": [], "error": "the model took longer than 60 s"}
        try:
            res = json.loads(out.decode().strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": [False] * len(cases), "details": [],
                    "error": "the model crashed: " + err.decode()[-300:].strip()}
        if len(res.get("ok", [])) != len(cases):
            res["ok"] = (res.get("ok", []) + [False] * len(cases))[:len(cases)]
        return res

    async def explained(self, game: str, v: int) -> dict:
        """{transition id: explained} for version v over all recorded transitions (cached; new ones evaluated)."""
        got = self.results.setdefault((game, v), {})
        todo = [c for c in self.transitions[game] if c["id"] not in got]
        if todo:
            res = await self.run(self.version(game, v)["code"], todo, game=game)
            for c, ok in zip(todo, res["ok"]):
                got[c["id"]] = bool(ok)
        return got

    def coverage(self, game: str, v: int | None = None) -> tuple[int, int]:
        v = self.current(game)["v"] if v is None else v
        got = self.results.get((game, v), {})
        return sum(got.get(c["id"], False) for c in self.transitions[game]), len(self.transitions[game])

    # ------------------------------------------------------------------ descriptions

    def describe(self, c: dict) -> str:
        if c["event"]:
            what = "completed the level" if c["event"] == "level_up" else "GAME_OVER"
        else:
            b = np.frombuffer(c["before"], dtype=np.uint8).reshape(64, 64)
            a = np.frombuffer(c["after"], dtype=np.uint8).reshape(64, 64)
            diff = np.argwhere(a != b)
            if not len(diff):
                what = "nothing changed"
            else:
                cells = ", ".join(f"({int(x)},{int(y)}) {int(b[y, x]):x}->{int(a[y, x]):x}" for y, x in diff[:8])
                what = (f"{len(diff)} cells changed in rows {diff[:, 0].min()}-{diff[:, 0].max()}, columns "
                        f"{diff[:, 1].min()}-{diff[:, 1].max()}: {cells}{' ...' if len(diff) > 8 else ''}")
        return f"T{c['id']} level {c['level']} {c['label']} (t{c['t']}, seen {c['n']}x): {what}"

    def case_text(self, game: str, tid: int) -> str:
        c = self.transitions[game][tid]
        b = np.frombuffer(c["before"], dtype=np.uint8).reshape(64, 64)
        a = np.frombuffer(c["after_raw"], dtype=np.uint8).reshape(64, 64)
        diff = np.argwhere(a != b)
        if len(diff):
            y0, y1 = max(0, diff[:, 0].min() - 2), min(63, diff[:, 0].max() + 2)
            x0, x1 = max(0, diff[:, 1].min() - 2), min(63, diff[:, 1].max() + 2)
        else:
            cx, cy = (c["x"], c["y"]) if c["x"] is not None else (32, 32)
            y0, y1, x0, x1 = max(0, cy - 4), min(63, cy + 4), max(0, cx - 8), min(63, cx + 8)
        rows = lambda g: "\n".join(f"{y:02d} " + "".join("0123456789abcdef"[int(v) & 15] for v in g[y, x0:x1 + 1])
                                   for y in range(y0, y1 + 1))
        return (self.describe(c) + f"\nBEFORE, columns {x0}-{x1}:\n{rows(b)}\nAFTER"
                + (" (the first frame after the event)" if c["event"] else "") + f":\n{rows(a)}")

    def export(self) -> dict:
        out = {}
        for g in self.games:
            n, total = self.coverage(g)
            out[g.split("-")[0]] = {"code": self.current(g)["code"], "version": self.current(g)["v"],
                                    "explained": n, "transitions": total,
                                    "history": [{k: v for k, v in h.items() if k != "code"} for h in self.versions[g]]}
        return out


def load_inherited(path: str | None) -> dict:
    """Final models of an earlier run (its run.json), by short game name."""
    if not path:
        return {}
    w = json.loads(Path(path).read_text())["world"]
    return {g: m["code"] for g, m in (w.get("models") or {}).items()}
