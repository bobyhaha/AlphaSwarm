"""Evidence for a world of frames (world option `evidence=1`): what a move changed, said as objects, and every frame kept.

A play result used to say "57 cells changed in rows 0-54, columns 3-60": the key that slid across, the gate that moved and
the step bar all in one box. Here the engine names what changed (moved, recoloured, resized, appeared, gone), keeps every
frame a move returns (in-between ones too), and keys each frame by its state, so a move from a state can be looked up and
two runs of one move from different states can be told apart. Nothing here knows any game: objects are same-colour
connected regions, and a counter (a step bar) is an edge row or column that changes alone when nothing else does.
"""
from __future__ import annotations

import hashlib
import zlib
from collections import defaultdict

import numpy as np

HEX = "0123456789abcdef"
SMALL = 400  # regions up to this many cells can be recognised as moved; larger ones (backgrounds, liquids) are resized


def pack(f: np.ndarray) -> bytes:
    return zlib.compress(np.asarray(f, dtype=np.uint8).tobytes(), 6)


def unpack(b: bytes, shape=(64, 64)) -> np.ndarray:
    return np.frombuffer(zlib.decompress(b), dtype=np.uint8).reshape(shape).astype(int)


def where(b) -> str:
    return f"r{b[0]}-{b[1]} c{b[2]}-{b[3]}" if b[0] != b[1] or b[2] != b[3] else f"r{b[0]} c{b[2]}"


def components(f: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    """4-connected same-colour regions, leaving out masked (counter) cells."""
    h, w = f.shape
    grid = f.tolist()
    seen = mask.copy().tolist() if mask is not None else [[False] * w for _ in range(h)]
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y][x]:
                continue
            c, stack, cells = grid[y][x], [(y, x)], []
            seen[y][x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == c:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            ys, xs = [p[0] for p in cells], [p[1] for p in cells]
            box = (min(ys), max(ys), min(xs), max(xs))
            out.append({"c": c, "box": box, "n": len(cells), "cells": frozenset(cells),
                        "shape": frozenset((cy - box[0], cx - box[2]) for cy, cx in cells)})
    return out


def _edges(a, b) -> str:
    names = ("top", "bottom", "left", "right")
    moved = [f"{names[i]} {'r' if i < 2 else 'c'}{a[i]}->{b[i]}" for i in range(4) if a[i] != b[i]]
    return ", ".join(moved)


def changes(before: np.ndarray, after: np.ndarray, mask: np.ndarray | None = None, limit: int = 6) -> list[str]:
    """What a move changed, as objects. Counter cells (mask) are left out; say so when they are all that changed."""
    keep = ~mask if mask is not None else np.ones(before.shape, bool)
    diff = (before != after) & keep
    if not diff.any():
        return ["nothing changed but the counter" if (before != after).any() else "nothing changed"]
    if diff.sum() > 0.6 * keep.sum():
        return [f"the whole frame changed ({int(diff.sum())} cells): a new level, a reset or a redraw"]
    A, B = components(before, mask), components(after, mask)
    same = {(o["c"], o["cells"]) for o in A} & {(p["c"], p["cells"]) for p in B}
    A = [o for o in A if (o["c"], o["cells"]) not in same]
    B = [p for p in B if (p["c"], p["cells"]) not in same]
    used_a, used_b, lines = set(), set(), []
    # moved: the same colour and shape somewhere else; parts that moved together and touch are one object
    pairs = []
    for i, o in enumerate(A):
        if o["n"] > SMALL:
            continue
        cand = [(abs(p["box"][0] - o["box"][0]) + abs(p["box"][2] - o["box"][2]), j) for j, p in enumerate(B)
                if j not in used_b and p["c"] == o["c"] and p["shape"] == o["shape"]]
        if cand:
            j = min(cand)[1]
            used_a.add(i), used_b.add(j)
            pairs.append((B[j]["box"][0] - o["box"][0], B[j]["box"][2] - o["box"][2], o))
    groups = defaultdict(list)
    for dy, dx, o in pairs:
        groups[(dy, dx)].append(o)
    for (dy, dx), parts in groups.items():
        objs = []  # touching parts with one offset form one object
        for o in sorted(parts, key=lambda o: o["box"]):
            hit = next((g for g in objs if any(o["box"][0] <= q["box"][1] + 1 and q["box"][0] <= o["box"][1] + 1 and
                                               o["box"][2] <= q["box"][3] + 1 and q["box"][2] <= o["box"][3] + 1 for q in g)), None)
            (hit.append(o) if hit is not None else objs.append([o]))
        for g in objs:
            box = (min(o["box"][0] for o in g), max(o["box"][1] for o in g), min(o["box"][2] for o in g), max(o["box"][3] for o in g))
            way = " ".join(x for x in (f"{'down' if dy > 0 else 'up'} {abs(dy)}" if dy else "",
                                       f"{'right' if dx > 0 else 'left'} {abs(dx)}" if dx else "") if x)
            cols = ",".join(sorted({HEX[o["c"]] for o in g}))
            lines.append((0, f"moved {way}: a {box[1] - box[0] + 1}x{box[3] - box[2] + 1} object (colours {cols}) "
                             f"from {where(box)} to {where((box[0] + dy, box[1] + dy, box[2] + dx, box[3] + dx))}"))
    # recoloured: the same cells, another colour
    for i, o in enumerate(A):
        if i in used_a:
            continue
        j = next((j for j, p in enumerate(B) if j not in used_b and p["cells"] == o["cells"]), None)
        if j is not None:
            used_a.add(i), used_b.add(j)
            lines.append((1, f"recoloured {HEX[o['c']]} -> {HEX[B[j]['c']]}: {o['n']} cells at {where(o['box'])}"))
    # resized: the same colour, overlapping most (a region that grew, shrank or changed shape)
    over = sorted(((len(o["cells"] & p["cells"]), i, j) for i, o in enumerate(A) if i not in used_a
                   for j, p in enumerate(B) if j not in used_b and p["c"] == o["c"]), reverse=True)
    for n, i, j in over:
        if n == 0 or i in used_a or j in used_b:
            continue
        used_a.add(i), used_b.add(j)
        o, p = A[i], B[j]
        if o["n"] == p["n"] and o["n"] > SMALL:
            continue  # a large region (a background) only flowing around something that moved
        lines.append((3, f"colour {HEX[o['c']]} region at {where(o['box'])} {'grew' if p['n'] > o['n'] else 'shrank' if p['n'] < o['n'] else 'changed shape'}"
                         f" ({o['n']}->{p['n']} cells" + (f"; {_edges(o['box'], p['box'])})" if o["box"] != p["box"] else ")")))
    for j, p in enumerate(B):
        if j not in used_b:
            lines.append((2, f"appeared: colour {HEX[p['c']]}, {p['n']} cells at {where(p['box'])}"))
    for i, o in enumerate(A):
        if i not in used_a:
            lines.append((2, f"gone: colour {HEX[o['c']]}, {o['n']} cells at {where(o['box'])}"))
    lines = [s for _, s in sorted(lines, key=lambda x: x[0])]
    return lines[:limit] + ([f"and {len(lines) - limit} more changes"] if len(lines) > limit else [])


def object_at(f: np.ndarray, x: int, y: int) -> tuple | None:
    """The box of the same-colour region under a click, the thing a click acts on."""
    h, w = f.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    c, stack, seen = f[y, x], [(y, x)], {(y, x)}
    while stack and len(seen) < 4096:
        cy, cx = stack.pop()
        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and f[ny, nx] == c:
                seen.add((ny, nx))
                stack.append((ny, nx))
    ys, xs = [p[0] for p in seen], [p[1] for p in seen]
    return (min(ys), max(ys), min(xs), max(xs))


class Counters:
    """Per game, the edge rows and columns that change alone (step bars): left out of state keys and change reports."""

    def __init__(self, shape=(64, 64)):
        self.shape = shape
        self.seen: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        self.lines: dict[str, set] = defaultdict(set)
        self.version = 0

    def learn(self, game: str, before: np.ndarray, after: np.ndarray):
        d = np.argwhere(before != after)
        if not 0 < len(d) <= 4:
            return
        h, w = self.shape
        fits = [(axis, edge) for axis, edge in ((0, 0), (0, h - 1), (1, 0), (1, w - 1)) if (d[:, axis] == edge).all()]
        if len(fits) != 1:  # a corner cell lies on two edges: it does not say which one is the counter
            return
        key = fits[0]
        self.seen[game][key] += 1
        if self.seen[game][key] >= 2 and key not in self.lines[game]:
            self.lines[game].add(key)
            self.version += 1

    def mask(self, game: str) -> np.ndarray:
        m = np.zeros(self.shape, bool)
        for axis, edge in self.lines[game]:
            if axis == 0:
                m[edge, :] = True
            else:
                m[:, edge] = True
        return m


def state_key(f: np.ndarray, mask: np.ndarray, level: int) -> str:
    g = np.where(mask, 255, np.asarray(f, dtype=np.int16)).astype(np.uint8)
    return f"L{level}:" + hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest()
