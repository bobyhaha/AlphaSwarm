"""Run an agent-written game model on recorded transitions, in a separate process with limits.

A model is Python source defining step(grid, action, x, y) -> (new_grid, event): grid is a 64x64 numpy int array of
colors 0-15, action 1-7 (x, y for ACTION6 clicks, else None), event '' | 'level_up' | 'game_over'. Returning a grid alone
means event ''. The code is checked (no imports, no names or attributes starting with '_', no file or reflection
functions) and runs with a whitelisted numpy, a CPU limit, no file writes and a time limit per call.

The parent sends a pickle on stdin ({"code", "cases", "detail"}; it is written by the harness, never by an agent) and
reads JSON on stdout: {"ok": [bool per case], "details": [...], "error": str | null}.
"""
from __future__ import annotations

import ast
import json
import sys

MAX_CODE = 8000
BAD_NAMES = {"open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input",
             "breakpoint", "help", "exit", "quit", "memoryview", "type", "object", "super", "classmethod",
             "staticmethod", "property", "dir", "id", "hash"}
BAD_ATTRS = {"tofile", "dump", "dumps", "ctypes", "load", "loads", "save", "savez", "savetxt", "loadtxt", "fromfile",
             "memmap", "lib", "os", "sys", "system", "mro", "f_globals", "gi_frame", "cr_frame", "tb_frame"}
BAD_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await,
             ast.With, ast.AsyncWith, ast.AsyncFor, ast.Yield, ast.YieldFrom)
NP_FUNCS = ("array", "asarray", "zeros", "ones", "full", "zeros_like", "ones_like", "full_like", "copy", "where",
            "argwhere", "nonzero", "count_nonzero", "any", "all", "sum", "unique", "roll", "flip", "fliplr", "flipud",
            "rot90", "transpose", "isin", "equal", "not_equal", "logical_and", "logical_or", "logical_not", "abs",
            "max", "min", "clip", "arange", "array_equal", "pad", "argmax", "argmin", "sign", "concatenate", "stack",
            "vstack", "hstack", "tile", "repeat", "mean", "bincount", "cumsum")
SAFE_BUILTINS = ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max", "min", "range",
                 "reversed", "round", "set", "sorted", "sum", "tuple", "zip", "map", "filter", "isinstance", "str",
                 "divmod", "pow", "frozenset", "ValueError", "IndexError", "KeyError", "Exception", "True", "False",
                 "None")


def check(code: str) -> str | None:
    """Why the code is not allowed, or None."""
    if not isinstance(code, str) or not code.strip():
        return "the code is empty"
    if len(code) > MAX_CODE:
        return f"the code is longer than {MAX_CODE} characters"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, BAD_NODES):
            return f"{type(node).__name__} is not allowed (no imports, classes, with, global or async)"
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in BAD_NAMES):
            return f"the name '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BAD_ATTRS):
            return f"the attribute '{node.attr}' is not allowed"
        if isinstance(node, (ast.FunctionDef, ast.arg)) and getattr(node, "name", getattr(node, "arg", "")).startswith("_"):
            return "names starting with '_' are not allowed"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "step" for n in tree.body):
        return "define a top-level function step(grid, action, x, y)"
    return None


def _limits():
    import resource
    for what, value in ((resource.RLIMIT_CPU, (20, 25)), (resource.RLIMIT_FSIZE, (0, 0))):
        try:
            resource.setrlimit(what, value)
        except (ValueError, OSError):
            pass


def main():
    import builtins
    import pickle
    import signal
    from types import SimpleNamespace

    import numpy as np

    job = pickle.loads(sys.stdin.buffer.read())  # written by the harness
    _limits()
    out = {"ok": [], "details": [], "error": None}
    why = check(job["code"])
    if why:
        out["error"] = why
        print(json.dumps(out))
        return
    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS if hasattr(builtins, n)}
    env = {"__builtins__": safe, "np": SimpleNamespace(**{f: getattr(np, f) for f in NP_FUNCS})}

    def alarm(*_):
        raise TimeoutError("a call took longer than 0.5 s")

    signal.signal(signal.SIGALRM, alarm)
    try:
        exec(compile(job["code"], "<model>", "exec"), env)
        step = env["step"]
    except Exception as exc:  # noqa: BLE001 - agent code
        out["error"] = f"the code failed to load: {type(exc).__name__}: {exc}"
        print(json.dumps(out))
        return
    detail = int(job.get("detail", 0))
    mask = np.frombuffer(job["mask"], dtype=bool).reshape(64, 64) if job.get("mask") else np.zeros((64, 64), bool)
    for i, c in enumerate(job["cases"]):
        grid = np.frombuffer(c["before"], dtype=np.uint8).reshape(64, 64).astype(int)
        msg, ok = "", False
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            res = step(grid.copy(), c["action"], c["x"], c["y"])
            signal.setitimer(signal.ITIMER_REAL, 0)
            pred, event = (res[0], res[1]) if isinstance(res, tuple) and len(res) == 2 else (res, "")
            event = event if isinstance(event, str) else ""
            if c["event"]:
                ok = event == c["event"]
                msg = "" if ok else f"predicted event '{event}', actual '{c['event']}'"
            elif event:
                msg = f"predicted event '{event}', but the move only changed the grid"
            else:
                pred = np.asarray(pred)
                after = np.frombuffer(c["after"], dtype=np.uint8).reshape(64, 64).astype(int)
                if pred.shape != (64, 64):
                    msg = f"returned shape {pred.shape}, not (64, 64)"
                else:
                    wrong = np.argwhere((pred != after) & ~mask)
                    ok = len(wrong) == 0
                    if not ok:
                        cells = ", ".join(f"({int(x)},{int(y)}) predicted {int(pred[y, x])} actual {int(after[y, x])}"
                                          for y, x in wrong[:6])
                        msg = f"{len(wrong)} cells wrong, e.g. {cells}"
        except Exception as exc:  # noqa: BLE001 - agent code
            signal.setitimer(signal.ITIMER_REAL, 0)
            msg = f"{type(exc).__name__}: {str(exc)[:160]}"
        out["ok"].append(bool(ok))
        if not ok and len(out["details"]) < detail:
            out["details"].append({"i": i, "msg": msg})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
