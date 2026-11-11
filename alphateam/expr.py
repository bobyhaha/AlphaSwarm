"""Safe arithmetic formulas over inputs x1..xN and outputs y1..yM, parsed with ast and evaluated without eval().

A compiled formula is fn(x, y): x is a sequence of input values indexed from 0 (x1 is x[0]) and y is a callable
returning the value of an output (for outputs defined by other outputs, or laws that build on other laws).
"""
from __future__ import annotations

import ast
import math
import re

VAR = re.compile(r"([xy])(\d+)")
MAX_LEN = 600
MAX_NODES = 160
NAN = float("nan")


class FormulaError(ValueError):
    pass


def _exp(v):
    return math.exp(min(v, 60.0))


def _log(v):
    return math.log(v) if v > 0 else NAN


def _sqrt(v):
    return math.sqrt(v) if v >= 0 else NAN


def _no_outputs(j):
    return NAN


FUNCS = {"max": max, "min": min, "abs": abs, "exp": _exp, "log": _log, "sqrt": _sqrt}


def compile_formula(text, n_inputs: int, n_outputs: int = 0):
    """Return (fn, inputs, outputs): the callable and the input and output numbers the formula uses."""
    text = str(text or "").strip().replace("^", "**")
    if not text:
        raise FormulaError("empty formula")
    if len(text) > MAX_LEN:
        raise FormulaError(f"formula longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"syntax error: {exc.msg}") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise FormulaError("formula too complex")
    inputs: set[int] = set()
    outputs: set[int] = set()
    names = f"x1..x{n_inputs}" + (f" and outputs y1..y{n_outputs}" if n_outputs else "")

    def build(node):
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            c = float(node.value)
            return lambda x, y: c
        if isinstance(node, ast.Name):
            m = VAR.fullmatch(node.id)
            limit = n_inputs if m and m.group(1) == "x" else n_outputs
            if not m or not 1 <= int(m.group(2)) <= limit:
                raise FormulaError(f"unknown variable '{node.id}' (use inputs {names})")
            j = int(m.group(2))
            if m.group(1) == "x":
                inputs.add(j)
                return lambda x, y: x[j - 1]
            outputs.add(j)
            return lambda x, y: y(j)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            f = build(node.operand)
            return (lambda x, y: -f(x, y)) if isinstance(node.op, ast.USub) else f
        if isinstance(node, ast.BinOp):
            a, b = build(node.left), build(node.right)
            if isinstance(node.op, ast.Add):
                return lambda x, y: a(x, y) + b(x, y)
            if isinstance(node.op, ast.Sub):
                return lambda x, y: a(x, y) - b(x, y)
            if isinstance(node.op, ast.Mult):
                return lambda x, y: a(x, y) * b(x, y)
            if isinstance(node.op, ast.Div):
                return lambda x, y: a(x, y) / b(x, y)
            if isinstance(node.op, ast.Pow):
                if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                        and abs(node.right.value) <= 4):
                    raise FormulaError("exponents must be numeric constants between -4 and 4")
                p = float(node.right.value)
                return lambda x, y: a(x, y) ** p
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
            fn = FUNCS[node.func.id]
            args = [build(arg) for arg in node.args]
            if not 1 <= len(args) <= 4:
                raise FormulaError(f"{node.func.id} takes 1-4 arguments")
            if len(args) == 1:
                g = args[0]
                return lambda x, y: fn(g(x, y))
            return lambda x, y: fn(*(g(x, y) for g in args))
        raise FormulaError(f"'{ast.unparse(node)}' is not allowed (use numbers, {names}, + - * / **, and {', '.join(FUNCS)})")

    fn = build(tree)
    return fn, inputs, outputs


def evaluate(fn, x, y=None) -> float:
    try:
        v = fn(x, y or _no_outputs)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return NAN
    if isinstance(v, complex) or v != v or v in (math.inf, -math.inf):
        return NAN
    return float(v)
