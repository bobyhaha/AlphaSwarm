"""A science world with a hidden answer.

An unknown system has inputs x1..xN (integer settings 0-10) and outputs y1..yM, measured with noise. Its
dependency structure is set by `structure`:

  flat          every output is a formula of a few inputs drawn from anywhere
  modular       outputs in a module share a few inputs; modules barely interact
  hierarchical  modules are grouped into super-modules (with coupling inside a super-module), and on top of the
                base outputs sit composite outputs: integration outputs that are formulas of base outputs from
                several modules of one super-module, and top outputs that are formulas of integration outputs
                from several super-modules. Like integrating blocks in chip design, a good law for a composite
                needs good laws for its parts.

Input and output numbers are shuffled, so nothing about the structure leaks through naming. Each agent measures
only a few outputs and has a small experiment budget. Laws are formulas in inputs and may also name other
outputs, which then stand for the society's current law for that output; so a composite's law is only as good
as the laws it builds on. Laws are validated periodically on held-out conditions (public feedback and track
records) and scored at the end on a separate hidden test set. Because the true structure is known, we can ask
whether the organization the agents build mirrors it.

Shocks change the world mid-run: 'shift' rewrites the base laws of one super-module (and so everything built on
them), 'churn:F' removes a fraction F of the agents.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .expr import NAN, FormulaError, compile_formula, evaluate
from .kernel import ActionError, as_list, clip

COEFS = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT = 5
TEST_POINTS, VAL_POINTS = 400, 200
OUT_REF = re.compile(r"y?(\d+)")
REF = re.compile(r"([xy]?)(\d+)")


@dataclass
class Truth:
    formula: str
    fn: object
    parents: set[str]  # e.g. {"x3", "x9"} or, for a composite, {"y12", "y40"}


@dataclass
class Law:
    output: int
    formula: str
    fn: object
    parents: set[str]
    refs: set[int]  # outputs this law builds on
    author: str
    group: str | None
    tick: int
    val: float | None = None
    val_tick: int | None = None
    test: float | None = None


@dataclass
class Record:
    rid: int
    aid: str
    board: str
    entry: str
    x: bytes
    y: dict


def r_squared(truth: list[float], pred: list[float]) -> float:
    mean = sum(truth) / len(truth)
    sst = sum((t - mean) ** 2 for t in truth) or 1e-9
    sse = 0.0
    for t, p in zip(truth, pred):
        if p != p:
            return 0.0
        sse += (t - p) ** 2
    return max(0.0, 1 - sse / sst)


def least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        raise ActionError("the terms are collinear on this data (vary those inputs, or drop a term)")
    return beta


class HiddenLaws:
    tool_names = ("experiment", "analyze", "publish")
    search_kinds = ("laws",)
    TOOLS = [
        {"name": "experiment",
         "description": "Run 1-5 experiments and measure your instruments. design 'random' draws random settings on the inputs "
                        "listed in vary (or on all inputs if vary is omitted); design 'set' uses your settings (other inputs "
                        "stay at 5). Results are posted as data to the board in 'to' (default your private board). "
                        "Each run uses one unit of your experiment budget.",
         "parameters": {"type": "object", "properties": {
             "design": {"type": "string", "enum": ["random", "set"]},
             "runs": {"type": "integer", "description": "random design: number of runs, 1-5"},
             "vary": {"type": "array", "items": {"type": "string"}, "description": "random design: inputs to randomize, e.g. [\"x3\",\"x9\"]"},
             "settings": {"type": "array", "items": {"type": "object"}, "description": "set design: 1-5 mappings like {\"x3\": 7, \"x9\": 0}"},
             "to": {"type": "string", "description": "board to post the data to: 'private', a group id, 'global', ..."}},
             "required": ["design"]}},
        {"name": "analyze",
         "description": "Analyze all data on an output that is on boards you can read. op 'screen' ranks candidates by correlation "
                        "with the output: inputs (default all) and, if you list them, other outputs (their current published "
                        "laws are evaluated at each data point). op 'fit' fits intercept + coefficients × terms by least squares "
                        "and reports R², cross-validated R² and the fitted formula, ready to publish. Terms may use inputs and "
                        "other outputs' current laws.",
         "parameters": {"type": "object", "properties": {
             "op": {"type": "string", "enum": ["screen", "fit"]},
             "output": {"type": "string", "description": "e.g. y12; screen also accepts several, e.g. \"y12,y40,y51\""},
             "inputs": {"type": "array", "items": {"type": "string"}, "description": "screen: candidates such as \"x3\" or \"y40\" (default all inputs)"},
             "terms": {"type": "array", "items": {"type": "string"}, "description": "fit: 1-12 terms such as \"x3\", \"x3**2\", \"x3*x9\", \"max(0, x4 - 5)\", \"y40\""},
             "top": {"type": "integer"}},
             "required": ["op", "output"]}},
        {"name": "publish",
         "description": "Set the society's current law for an output (replaces the previous one). The formula may name other "
                        "outputs (e.g. y40), meaning their current laws. Optionally publish on behalf of a group you may act for.",
         "parameters": {"type": "object", "properties": {
             "output": {"type": "string"}, "formula": {"type": "string"},
             "as": {"type": "string", "description": "optional group id"}},
             "required": ["output", "formula"]}},
    ]

    def __init__(self, n_agents: int, seed: int = 0, structure: str = "hierarchical", instruments: int = 3,
                 redundancy: int | None = None, module_outputs: int = 4, module_inputs: int = 4,
                 experiments: int = 30, seed_runs: int = 3, noise: float = 0.3, validate_every: int = 3,
                 shocks: list[str] | None = None, single: bool = False, world_agents: int | None = None):
        if structure not in ("hierarchical", "modular", "flat"):
            raise ValueError("structure must be hierarchical, modular or flat")
        self.rng = random.Random(seed)
        self.noise_rng = random.Random(seed + 101)
        self.structure, self.noise, self.validate_every = structure, noise, validate_every
        n_world = world_agents or n_agents
        r = redundancy or max(2, round(math.sqrt(n_world) / 3))
        n_base = max(8, round(n_world * instruments / r))
        K = max(2, math.ceil(n_base / module_outputs))
        self.n_in = K * module_inputs
        hier = structure == "hierarchical"
        per_super = max(2, round(math.sqrt(K))) if hier else 1
        n_super_base = math.ceil(K / per_super)
        c1 = [max(1, round(len(range(s * per_super, min(K, (s + 1) * per_super))) / 2)) for s in range(n_super_base)] if hier else []
        c2 = max(1, n_super_base // 3) if hier and n_super_base >= 2 else 0
        self.n_out = n_base + sum(c1) + c2
        ids = list(range(1, self.n_out + 1))
        ins = list(range(1, self.n_in + 1))
        self.rng.shuffle(ids)
        self.rng.shuffle(ins)
        base, comp = ids[:n_base], ids[n_base:]
        self.modules = []
        self.module_of_out, self.module_of_in, self.level = {}, {}, {}
        for m in range(K):
            mo, mi = sorted(base[m::K]), sorted(ins[m * module_inputs:(m + 1) * module_inputs])
            self.modules.append({"id": m, "super": m // per_super, "level": 0, "outputs": mo, "inputs": mi})
            self.module_of_out.update({o: m for o in mo})
            self.module_of_in.update({i: m for i in mi})
            self.level.update({o: 0 for o in mo})
        for s, count in enumerate(c1):
            self._add_composite_module(sorted(comp[:count]), s, 1)
            comp = comp[count:]
        if c2:
            self._add_composite_module(sorted(comp), n_super_base, 2)
        self.n_super = max(mod["super"] for mod in self.modules) + 1
        self.has_modules = structure != "flat"
        self.truth: dict[int, Truth] = {}
        for o in sorted(self.level, key=lambda o: (self.level[o], o)):
            self.truth[o] = self._make_truth(o)
        erng = random.Random(seed + 202)
        self.test_x = [bytes(erng.randint(0, 10) for _ in range(self.n_in)) for _ in range(TEST_POINTS)]
        self.val_x = [bytes(erng.randint(0, 10) for _ in range(self.n_in)) for _ in range(VAL_POINTS)]
        self.truth_version = 0
        self._cache_truth()
        self.instruments: dict[str, list[int]] = {}
        self.budget: dict[str, int] = {}
        if single:
            self.instruments["A1"] = list(range(1, self.n_out + 1))
            self.budget["A1"] = max(experiments, round(n_world * instruments * experiments / self.n_out))
        else:
            cycle: list[int] = []
            for i in range(1, n_agents + 1):
                picks: list[int] = []
                while len(picks) < min(instruments, self.n_out):
                    if not cycle:
                        cycle = list(range(1, self.n_out + 1))
                        self.rng.shuffle(cycle)
                    o = cycle.pop()
                    if o not in picks:
                        picks.append(o)
                self.instruments[f"A{i}"] = sorted(picks)
                self.budget[f"A{i}"] = experiments
        self.measured_by = defaultdict(list)
        for aid, outs_ in self.instruments.items():
            for o in outs_:
                self.measured_by[o].append(aid)
        self.seed_runs = seed_runs
        self.used = Counter()
        self.records: list[Record] = []
        self.by_out: dict[int, list[int]] = defaultdict(list)
        self.laws: dict[int, Law] = {}
        self.laws_version = 0
        self.history: list[dict] = []
        self.validations: list[dict] = []
        self.shock_plan = self._parse_shocks(shocks or [])
        self.shocks: list[dict] = []
        self.focus: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
        self._records: tuple[dict, dict] | None = None
        self._score_key = None

    def _add_composite_module(self, outs: list[int], sup: int, level: int):
        mid = len(self.modules)
        self.modules.append({"id": mid, "super": sup, "level": level, "outputs": outs, "inputs": []})
        self.module_of_out.update({o: mid for o in outs})
        self.level.update({o: level for o in outs})

    # ------------------------------------------------------------------ the hidden system

    def _make_truth(self, o: int) -> Truth:
        rng = self.rng
        mod = self.modules[self.module_of_out[o]]
        if mod["level"] > 0:
            return self._make_composite(o, mod)
        if self.structure == "flat":
            parents = rng.sample(range(1, self.n_in + 1), 2 + (rng.random() < 0.5))
        else:
            parents = rng.sample(mod["inputs"], min(len(mod["inputs"]), 2 + (rng.random() < 0.5)))
            sibs = [m for m in self.modules if m["level"] == 0 and m["super"] == mod["super"] and m["id"] != mod["id"]]
            if self.structure == "hierarchical" and sibs and rng.random() < 0.45:
                parents.append(rng.choice(rng.choice(sibs)["inputs"]))
            others = [m for m in self.modules if m["level"] == 0 and m["super"] != mod["super"]]
            if others and rng.random() < 0.06:
                parents.append(rng.choice(rng.choice(others)["inputs"]))
        parents = list(dict.fromkeys(parents))
        terms = []
        for p in parents:
            a = rng.choice(COEFS) * rng.choice((-1, 1))
            u = rng.random()
            if u < 0.6:
                terms.append(f"{a:g}*x{p}")
            elif u < 0.8:
                terms.append(f"{a:g}*x{p}**2/10")
            else:
                terms.append(f"{a:g}*max(0, x{p} - {rng.randint(3, 7)})")
        if len(parents) >= 2 and rng.random() < 0.25:
            terms.append(f"{rng.choice(COEFS) * rng.choice((-1, 1)):g}*x{parents[0]}*x{parents[1]}/10")
        formula = (f"{rng.randint(0, 20)} + " + " + ".join(terms)).replace("+ -", "- ")
        fn, _, _ = compile_formula(formula, self.n_in)
        return Truth(formula, fn, {f"x{p}" for p in parents})

    def _make_composite(self, o: int, mod: dict) -> Truth:
        """An output built from outputs one level down, drawn from different modules (or super-modules)."""
        rng = self.rng
        if mod["level"] == 1:
            pool = [m for m in self.modules if m["level"] == 0 and m["super"] == mod["super"]]
        else:
            pool = [m for m in self.modules if m["level"] == 1]
        picked = rng.sample(pool, min(len(pool), 2 + (rng.random() < 0.4)))
        parents = [rng.choice(m["outputs"]) for m in picked]
        if len(parents) < 2:
            extra = [p for m in pool for p in m["outputs"] if p not in parents]
            parents += rng.sample(extra, min(1, len(extra)))
        terms = [f"{rng.choice((0.5, 1.0, 1.5, 2.0)) * rng.choice((-1, 1)):g}*y{p}" for p in parents]
        if len(parents) >= 2 and rng.random() < 0.3:
            terms.append(f"{rng.choice((0.5, 1.0)) * rng.choice((-1, 1)):g}*y{parents[0]}*y{parents[1]}/20")
        formula = (f"{rng.randint(0, 20)} + " + " + ".join(terms)).replace("+ -", "- ")
        fn, _, _ = compile_formula(formula, self.n_in, self.n_out)
        return Truth(formula, fn, {f"y{p}" for p in parents})

    def _tv(self, o: int, x: bytes, memo: dict) -> float:
        """True noiseless value of output o at x."""
        v = memo.get(o)
        if v is None:
            v = memo[o] = evaluate(self.truth[o].fn, x, lambda j: self._tv(j, x, memo))
        return v

    def _cache_truth(self):
        order = sorted(self.truth, key=lambda o: self.level[o])
        self.test_y = {o: [] for o in order}
        self.val_y = {o: [] for o in order}
        for xs, store in ((self.test_x, self.test_y), (self.val_x, self.val_y)):
            for x in xs:
                memo: dict = {}
                for o in order:
                    store[o].append(self._tv(o, x, memo))
        self.truth_version += 1

    def _measure(self, aid: str, x: bytes) -> dict:
        memo: dict = {}
        return {o: round(self._tv(o, x, memo) + self.noise_rng.gauss(0, self.noise), 3) for o in self.instruments[aid]}

    def _lv(self, o: int, x: bytes, memo: dict, stack: tuple = ()) -> float:
        """Value at x of the society's current law for output o (following references to other laws)."""
        if o in memo:
            return memo[o]
        law = self.laws.get(o)
        if law is None or o in stack or len(stack) > 8:
            return NAN
        v = memo[o] = evaluate(law.fn, x, lambda j: self._lv(j, x, memo, stack + (o,)))
        return v

    def _law_predictions(self, xs: list[bytes]) -> dict[int, list[float]]:
        preds: dict[int, list[float]] = {o: [] for o in self.laws}
        for x in xs:
            memo: dict = {}
            for o in self.laws:
                preds[o].append(self._lv(o, x, memo))
        return preds

    def _parse_shocks(self, specs: list[str]) -> dict[int, list[tuple]]:
        plan = defaultdict(list)
        for s in specs:
            parts = s.split(":")
            if len(parts) < 2 or parts[1] not in ("shift", "churn") or not parts[0].isdigit():
                raise ValueError(f"bad shock '{s}' (use T:shift or T:churn:FRACTION)")
            plan[int(parts[0])].append((parts[1], float(parts[2]) if len(parts) > 2 else 0.25))
        return plan

    # ------------------------------------------------------------------ kernel hooks

    def setup(self, k):
        """Seed every private substrate with a few free observations, so agents start with different data."""
        for aid in self.instruments:
            if self.seed_runs:
                xs = [bytes(self.rng.randint(0, 10) for _ in range(self.n_in)) for _ in range(self.seed_runs)]
                self._record(k, aid, f"private:{aid}", xs, "starting observations (random settings on all inputs)", None)

    def on_tick_start(self, k):
        for kind, arg in self.shock_plan.get(k.tick, []):
            if kind == "shift":
                sup = self.rng.choice(sorted({m["super"] for m in self.modules if m["level"] == 0}))
                outs = [o for m in self.modules if m["level"] == 0 and m["super"] == sup for o in m["outputs"]]
                for o in outs:
                    self.truth[o] = self._make_truth(o)
                self._cache_truth()
                self.shocks.append({"t": k.tick, "kind": "shift", "super": sup, "outputs": len(outs)})
            else:
                active = k.active_agents()
                gone = self.rng.sample(active, min(len(active) - 1, round(arg * len(active))))
                for aid in gone:
                    k.deactivate(aid)
                self.shocks.append({"t": k.tick, "kind": "churn", "removed": len(gone)})

    def on_tick_end(self, k):
        if self.validate_every and k.tick % self.validate_every == 0:
            self.validate(k)

    def validate(self, k):
        self._records = None
        preds = self._law_predictions(self.val_x)
        scored = []
        for o, law in self.laws.items():
            law.val = r_squared(self.val_y[o], preds[o])
            law.val_tick = k.tick
            scored.append((law.val, o, law))
        overall = sum(v for v, _, _ in scored) / self.n_out
        among = sum(v for v, _, _ in scored) / len(scored) if scored else 0.0
        self.validations.append({"t": k.tick, "overall": round(overall, 4), "published": len(scored)})
        scored.sort(key=lambda s: s[0])
        fmt = lambda s: f"y{s[1]} {s[0]:.2f} ({s[2].author}{'/' + s[2].group if s[2].group else ''})"
        text = (f"Lab validation at t{k.tick}: {len(scored)}/{self.n_out} outputs have a law. Mean R² over all outputs "
                f"{overall:.3f} (outputs without a law count 0); mean among published {among:.3f}.")
        if scored:
            text += "\nLowest: " + ", ".join(fmt(s) for s in scored[:5]) + "\nHighest: " + ", ".join(fmt(s) for s in scored[-5:][::-1])
            text += "\nPer-output results: search what=laws."
        k.post("global", "lab", "result", text, notify=False)
        by_author = defaultdict(list)
        for v, o, law in scored:
            by_author[law.author].append(f"y{o} {v:.2f}")
        for aid, items in by_author.items():
            k.notify(aid, "lab", f"validation of your laws: {', '.join(items[:8])}")

    # ------------------------------------------------------------------ actions

    def act(self, k, aid: str, name: str, a: dict):
        return getattr(self, f"_{name}")(k, aid, a)

    def _out(self, ref) -> int:
        m = OUT_REF.fullmatch(str(ref or "").strip())
        if not m or not 1 <= int(m.group(1)) <= self.n_out:
            raise ActionError(f"no output '{ref}' (outputs are y1..y{self.n_out})")
        return int(m.group(1))

    def _in(self, ref) -> int:
        m = REF.fullmatch(str(ref).strip())
        if not m or m.group(1) == "y" or not 1 <= int(m.group(2)) <= self.n_in:
            raise ActionError(f"no input '{ref}' (inputs are x1..x{self.n_in})")
        return int(m.group(2))

    def _cand(self, ref) -> tuple[str, int]:
        m = REF.fullmatch(str(ref).strip())
        if m and m.group(1) == "y":
            return "y", self._out(ref)
        return "x", self._in(ref)

    def _note_focus(self, k, aid: str, modules: dict[int, float]):
        total = sum(modules.values())
        for m, w in modules.items():
            self.focus[aid].append((k.tick, m, w / total))

    def _record(self, k, aid, board, xs, desc, shown):
        results = [self._measure(aid, x) for x in xs]
        first = len(self.records)
        lines = []
        for j, (x, y) in enumerate(zip(xs, results)):
            ins = " ".join(f"x{i}={x[i - 1]}" for i in shown) if shown else ""
            lines.append(f"run{j + 1} {ins + ' ' if ins else ''}→ " + ", ".join(f"y{o}={v:g}" for o, v in y.items()))
        content = (f"Experiment by {aid}: {len(xs)} run(s), {desc}.\n" + "\n".join(lines) +
                   f"\n(records R{first + 1}-R{first + len(xs)}; analyze can use them wherever this board is readable)")
        e = k.post(board, aid, "data", content, notify=False)
        for x, y in zip(xs, results):
            rec = Record(len(self.records), aid, board, e.id, x, y)
            self.records.append(rec)
            for o in y:
                self.by_out[o].append(rec.rid)
        return content

    def _experiment(self, k, aid, a):
        k._need(aid in self.instruments, "you have no instruments")
        board = k.board_id(aid, a.get("to", "private"))
        k._need(k.can_write(aid, board), f"you cannot write to {board}")
        left = self.budget[aid] - self.used[aid]
        design = str(a.get("design", "random")).strip()
        if design == "random":
            runs = max(1, min(5, int(a.get("runs") or 1)))
            vary = sorted({self._in(v) for v in as_list(a.get("vary"))})
            pool = vary or list(range(1, self.n_in + 1))
            xs = []
            for _ in range(runs):
                v = bytearray([DEFAULT] * self.n_in)
                for i in pool:
                    v[i - 1] = self.rng.randint(0, 10)
                xs.append(bytes(v))
            shown = vary if 0 < len(vary) <= 10 else None
            desc = (f"random settings on {', '.join(f'x{i}' for i in vary)}" if shown else
                    f"random settings on {len(vary)} chosen inputs" if vary else "random settings on all inputs")
            touched = vary
        elif design == "set":
            settings = a.get("settings")
            settings = [settings] if isinstance(settings, dict) else list(settings or [])
            k._need(0 < len(settings) <= 5, "give 1-5 settings, each a mapping like {\"x3\": 7, \"x9\": 0}")
            xs, keys = [], set()
            for s in settings:
                k._need(isinstance(s, dict), "each setting is a mapping like {\"x3\": 7}")
                v = bytearray([DEFAULT] * self.n_in)
                for key, val in s.items():
                    i = self._in(key)
                    v[i - 1] = max(0, min(10, round(float(val))))
                    keys.add(i)
                xs.append(bytes(v))
            touched = sorted(keys)
            shown = touched if len(touched) <= 10 else None
            desc = "chosen settings (other inputs at 5)"
        else:
            raise ActionError("design is 'random' or 'set'")
        k._need(len(xs) <= left, f"you have {left} experiments left")
        self.used[aid] += len(xs)
        k.tick_counts["experiments"] += len(xs)
        if touched and len(touched) < self.n_in:
            mods = Counter(self.module_of_in[i] for i in touched)
            self._note_focus(k, aid, {m: 0.5 * c for m, c in mods.items()})
        content = self._record(k, aid, board, xs, desc, shown)
        return "lab.experiment", f"{clip(content, 1400)}\nPosted to {board}. Experiments left {left - len(xs)}.", board

    def _data(self, k, aid, o):
        readable = {}
        out = []
        for rid in self.by_out[o]:
            rec = self.records[rid]
            ok = readable.get(rec.board)
            if ok is None:
                ok = readable[rec.board] = k.can_read(aid, rec.board)
            if ok:
                out.append(rec)
        return out, sum(readable.values())

    def _screen(self, k, aid, o: int, cands: list[tuple[str, int]], top: int) -> str:
        data, boards = self._data(k, aid, o)
        n = len(data)
        if n < 5:
            return f"screen y{o}: only {n} data points are readable to you (need 5)"
        names, cols, missing = [], [], []
        xs = [i for t, i in cands if t == "x"]
        if xs:
            X = np.frombuffer(b"".join(r.x for r in data), dtype=np.uint8).reshape(n, self.n_in)[:, [i - 1 for i in xs]]
            cols.append(X.astype(float))
            names += [f"x{i}" for i in xs]
        ys = [j for t, j in cands if t == "y" and j != o]
        if ys:
            vals = np.array([[self._lv(j, r.x, memo) for j in ys] for r in data for memo in [{}]])
            ok = ~np.isnan(vals).any(axis=0)
            missing = [f"y{j}" for j, good in zip(ys, ok) if not good]
            if ok.any():
                cols.append(vals[:, ok])
                names += [f"y{j}" for j, good in zip(ys, ok) if good]
        if not cols:
            return f"screen y{o}: no usable candidates" + (f" ({', '.join(missing)} have no published law)" if missing else "")
        M = np.hstack(cols)
        M = M - M.mean(axis=0)
        Y = np.array([r.y[o] for r in data])
        Y = Y - Y.mean()
        sx = np.sqrt((M ** 2).sum(axis=0))
        sy = float(np.sqrt((Y ** 2).sum())) or 1e-9
        corr = (M.T @ Y) / np.where(sx > 0, sx, 1) / sy
        scores = sorted(((float(corr[j]), name) for j, name in enumerate(names) if sx[j] > 0), key=lambda s: -abs(s[0]))
        self._note_focus(k, aid, {self.module_of_out[o]: 2})
        body = ", ".join(f"{name} r={r:+.2f}" for r, name in scores[:top]) or "none of these candidates varied in the data"
        return (f"screen y{o}: {n} data points from {boards} readable boards; {len(scores)} of {len(cands)} candidates "
                f"varied. Strongest by |correlation|: {body}" + (f" ({', '.join(missing)} have no published law)" if missing else ""))

    def _analyze(self, k, aid, a):
        op = str(a.get("op", "fit")).strip()
        if op == "screen":
            outs = [self._out(x) for x in as_list(a.get("output") or a.get("outputs"))][:6]
            k._need(bool(outs), "give an output, or a list of up to 6")
            cands = list(dict.fromkeys(self._cand(v) for v in as_list(a.get("inputs")))) or [("x", i) for i in range(1, self.n_in + 1)]
            top = max(1, min(20, int(a.get("top") or 8)))
            k.tick_counts["analyses"] += 1
            return "lab.screen", "\n".join(self._screen(k, aid, o, cands, top) for o in outs), ",".join(f"y{o}" for o in outs)
        o = self._out(a.get("output"))
        data, boards = self._data(k, aid, o)
        if op == "fit":
            raw = a.get("terms")
            terms = [t.strip() for t in (raw.split(";") if isinstance(raw, str) else raw or []) if str(t).strip()]
            k._need(1 <= len(terms) <= 12, "give 1-12 terms, e.g. [\"x3\", \"x3**2\", \"x3*x9\", \"max(0, x4 - 5)\", \"y40\"]")
            try:
                compiled = [compile_formula(t, self.n_in, self.n_out) for t in terms]
            except FormulaError as exc:
                raise ActionError(str(exc)) from None
            k._need(all(o not in outs for _, _, outs in compiled), f"terms for y{o} cannot use y{o} itself")
            fns = [fn for fn, _, _ in compiled]
            rows, y = [], []
            for r in data:
                memo: dict = {}
                row = [1.0] + [evaluate(f, r.x, lambda j: self._lv(j, r.x, memo)) for f in fns]
                if all(v == v for v in row):
                    rows.append(row)
                    y.append(r.y[o])
            n = len(rows)
            k._need(n >= len(terms) + 3, f"only {n} usable data points on y{o} are readable to you (need {len(terms) + 3}; "
                                         f"terms naming outputs need those outputs to have published laws)")
            X, Y = np.array(rows), np.array(y)
            beta = least_squares(X, Y)
            r2 = r_squared(y, list(X @ beta))
            folds = min(5, n)
            order = list(range(n))
            random.Random(n * 31 + o).shuffle(order)
            cv_pred = np.zeros(n)
            try:
                for f in range(folds):
                    test = order[f::folds]
                    train = sorted(set(range(n)) - set(test))
                    if len(train) < len(terms) + 2:
                        raise ActionError("too few points")
                    cv_pred[test] = X[test] @ least_squares(X[train], Y[train])
                cv = f"{r_squared(y, list(cv_pred)):.3f}"
            except ActionError:
                cv = "n/a"
            formula = f"{beta[0]:.4g}" + "".join(f" {'+' if b >= 0 else '-'} {abs(b):.4g}*({t})" for b, t in zip(beta[1:], terms))
            self._note_focus(k, aid, {self.module_of_out[o]: 3})
            k.tick_counts["analyses"] += 1
            return "lab.fit", (f"fit y{o} on {n} points from {boards} readable boards: R² {r2:.3f}, cross-validated R² {cv}\n"
                               f"formula: {formula}"), f"y{o}"
        raise ActionError("op is 'screen' or 'fit'")

    def _reaches(self, refs: set[int], target: int) -> bool:
        stack, seen = list(refs), set()
        while stack:
            j = stack.pop()
            if j == target:
                return True
            if j not in seen:
                seen.add(j)
                if j in self.laws:
                    stack.extend(self.laws[j].refs)
        return False

    def _publish(self, k, aid, a):
        o = self._out(a.get("output"))
        formula = str(a.get("formula") or "").strip()
        try:
            fn, ins, refs = compile_formula(formula, self.n_in, self.n_out)
        except FormulaError as exc:
            raise ActionError(str(exc)) from None
        k._need(not self._reaches(refs, o), f"laws would refer to each other in a circle (y{o} through {sorted(refs)})")
        group = None
        if a.get("as"):
            group = k._group(a["as"])
            k._need(k.can_act_for(aid, group), f"you may not act for {group}")
        prev = self.laws.get(o)
        self.laws[o] = Law(o, formula, fn, {f"x{i}" for i in ins} | {f"y{j}" for j in refs}, refs, aid, group, k.tick)
        self.laws_version += 1
        self._records = None
        self.history.append({"t": k.tick, "output": o, "author": aid, "group": group})
        self._note_focus(k, aid, {self.module_of_out[o]: 3})
        k.tick_counts["published"] += 1
        if prev and prev.author != aid:
            k.notify(prev.author, aid, f"{aid} replaced your law for y{o}")
        users = sorted({law.author for law in self.laws.values() if o in law.refs and law.author != aid})
        for u in users[:20]:
            k.notify(u, aid, f"{aid} published a new law for y{o}, which your law builds on")
        nxt = (k.tick // self.validate_every + 1) * self.validate_every if self.validate_every else None
        msg = f"published the society's law for y{o}"
        if prev:
            msg += f", replacing {prev.author}'s from t{prev.tick}" + (f" (validation R² {prev.val:.2f})" if prev.val is not None else "")
        return "lab.publish", msg + (f". Next validation at t{nxt}." if nxt else "."), f"y{o}"

    # ------------------------------------------------------------------ views

    def search(self, k, aid, what, words):
        outs = [self._out(w) for w in words if re.fullmatch(r"y\d+", w)]
        authors = {w.upper() for w in words if re.fullmatch(r"a\d+", w)}
        pool = outs or list(range(1, self.n_out + 1))
        if "mine" in words:
            pool = self.instruments.get(aid, [])
        if "missing" in words or "none" in words:
            pool = [o for o in pool if o not in self.laws]
        if authors:
            pool = [o for o in pool if o in self.laws and self.laws[o].author in authors]
        if "worst" in words:
            pool = sorted(pool, key=lambda o: (self.laws[o].val if o in self.laws and self.laws[o].val is not None else -1))
        elif not outs:
            pool = sorted(pool, key=lambda o: -(self.laws[o].tick if o in self.laws else -1))
        lines = []
        for o in pool:
            law = self.laws.get(o)
            if law:
                val = f"val R² {law.val:.2f} (t{law.val_tick})" if law.val is not None else "not yet validated"
                lines.append(f"y{o}: law by {law.author}{'/' + law.group if law.group else ''} t{law.tick}, {val}: {clip(law.formula, 110)}")
            else:
                lines.append(f"y{o}: no law yet; measured by {len(self.measured_by[o])} agents")
        return lines

    def agent_tags(self, aid: str) -> str:
        outs = self.instruments.get(aid, [])
        return "measures " + ",".join(f"y{o}" for o in outs[:6]) + ("…" if len(outs) > 6 else "")

    def _build_records(self):
        by_agent, by_group = defaultdict(list), defaultdict(list)
        for law in self.laws.values():
            if law.val is not None:
                by_agent[law.author].append(law.val)
                if law.group:
                    by_group[law.group].append(law.val)
        fmt = lambda v: f"{len(v)} validated laws, mean R² {sum(v) / len(v):.2f}"
        self._records = ({a: "record: " + fmt(v) for a, v in by_agent.items()},
                         {g: "group record: " + fmt(v) for g, v in by_group.items()})

    def track_record(self, aid: str) -> str:
        if self._records is None:  # rebuilt after each validation or publication
            self._build_records()
        return self._records[0].get(aid, "")

    def group_record(self, gid: str) -> str:
        if self._records is None:
            self._build_records()
        return self._records[1].get(gid, "")

    def observe(self, k, aid: str) -> str:
        outs = self.instruments.get(aid, [])
        mine = []
        for o in outs[:12]:
            law = self.laws.get(o)
            if law:
                val = f"val R² {law.val:.2f}" if law.val is not None else "not validated yet"
                mine.append(f"y{o} by {law.author} t{law.tick}, {val}")
            else:
                mine.append(f"y{o} none")
        last = self.validations[-1] if self.validations else None
        return (f"LAB: you measure {', '.join(f'y{o}' for o in outs[:12])}{' and more' if len(outs) > 12 else ''}. "
                f"Experiments left {self.budget.get(aid, 0) - self.used[aid]}/{self.budget.get(aid, 0)}. "
                f"Data on your outputs readable to you: see analyze.\n"
                f"Laws for your outputs: {'; '.join(mine) or 'none'}.\n"
                f"Society: {len(self.laws)}/{self.n_out} outputs have laws"
                + (f"; last validation t{last['t']}: mean R² over all outputs {last['overall']:.3f}." if last else "; no validation yet."))

    def brief(self) -> str:
        return f"""THE MISSION
You are scientists studying an unknown system with {self.n_in} inputs x1..x{self.n_in} (integer settings 0-10, default 5) and {self.n_out} outputs y1..y{self.n_out}. Each output is a fixed but unknown function, measured with noise; some outputs may be driven by other outputs rather than directly by inputs. The society's goal is an accurate published law for every output. At the end, each output's current law is scored on held-out conditions nobody sees (R²; outputs without a law score 0). The society's score is the average.
- Every agent measures only its own instruments (a few outputs) and has a small experiment budget, so nobody can cover much of the system alone. Your private board starts with a few observations only you have.
- experiment: set inputs (chosen values, or random values on inputs you pick, or on all inputs) and measure your outputs. Results are posted as data on a board you choose; anyone who can read that board can analyze them.
- analyze: 'screen' ranks inputs (or listed outputs) by correlation with an output; 'fit' fits a formula by least squares (with cross-validated R²). Both use all data on boards you can read.
- publish: sets the society's current law for an output. The latest law published for an output replaces the previous one.
- A formula may name another output (e.g. y12): it then stands for the society's current law for y12, so a law that builds on other laws is only as good as they are, and improves when they do.
- Every {self.validate_every} ticks the lab validates every published law on separate conditions and posts the results on the global board. Track records of agents and groups are public (search).
- Formulas use numbers, x1..x{self.n_in}, y1..y{self.n_out}, + - * / **, and max, min, abs, exp, log, sqrt."""

    # ------------------------------------------------------------------ scoring (hidden from agents)

    def score(self) -> dict:
        key = (self.laws_version, self.truth_version)
        if self._score_key != key:
            preds = self._law_predictions(self.test_x)
            for o, law in self.laws.items():
                law.test = r_squared(self.test_y[o], preds[o])
            self._score_key = key
        by_level = defaultdict(list)
        tp = fp = fn = 0
        for o in range(1, self.n_out + 1):
            law = self.laws.get(o)
            by_level[self.level[o]].append(law.test if law else 0.0)
            if self.level[o] == 0:
                true = self.truth[o].parents
                got = law.parents if law else set()
                tp += len(got & true)
                fp += len(got - true)
                fn += len(true - got)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        mean = lambda v: round(sum(v) / len(v), 4) if v else None
        return {"score": mean([s for v in by_level.values() for s in v]), "score_base": mean(by_level[0]),
                "score_composite": mean(by_level[1] + by_level[2]) if (by_level[1] or by_level[2]) else None,
                "coverage": round(len(self.laws) / self.n_out, 4),
                "structure_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                "val": self.validations[-1]["overall"] if self.validations else None}

    def dominant_modules(self, tick: int, window: int = 3) -> dict[str, int]:
        out = {}
        for aid, items in self.focus.items():
            w = Counter()
            for t, m, x in items:
                if t > tick - window:
                    w[m] += x
            if w:
                out[aid] = w.most_common(1)[0][0]
        return out

    def export(self) -> dict:
        self.score()
        return {
            "n_in": self.n_in, "n_out": self.n_out, "structure": self.structure, "n_super": self.n_super,
            "modules": self.modules, "levels": {o: lv for o, lv in self.level.items() if lv},
            "truth": {o: t.formula for o, t in self.truth.items()},
            "laws": [{"output": o, "formula": law.formula, "author": law.author, "group": law.group, "tick": law.tick,
                      "val": law.val, "test": law.test, "module": self.module_of_out[o], "level": self.level[o],
                      "refs": sorted(law.refs)} for o, law in sorted(self.laws.items())],
            "history": self.history, "validations": self.validations, "shocks": self.shocks,
            "instruments": self.instruments, "experiments_used": dict(self.used),
        }
