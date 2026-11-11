"""Small statistics for comparing societies: paired gains with a lower bound, and how different two runs' actions are.

A society run is costly and its score varies from run to run, so a revision is judged by paired runs (the same task
instance, start and budget for parent and child) and kept only when the lower bound of the mean paired gain is above a
threshold. No dependency beyond the standard library.
"""
from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist

_Z = NormalDist()


def t_quantile(p: float, df: int) -> float:
    """The p-quantile of Student's t with df degrees of freedom (exact for df 1 and 2, a series beyond)."""
    if not 0 < p < 1 or df < 1:
        raise ValueError("t_quantile needs 0 < p < 1 and df >= 1")
    if df == 1:
        return math.tan(math.pi * (p - 0.5))
    if df == 2:
        return (2 * p - 1) / math.sqrt(2 * p * (1 - p))
    z = _Z.inv_cdf(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def summary(xs, conf: float = 0.9) -> dict:
    """Mean, spread and one-sided bounds at `conf` for a list of numbers (paired gains, or one recipe's scores)."""
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "se": None, "lcb": None, "ucb": None}
    mean = sum(xs) / n
    if n < 2:
        return {"n": 1, "mean": round(mean, 5), "sd": None, "se": None, "lcb": None, "ucb": None}
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    t = t_quantile(conf, n - 1)
    return {"n": n, "mean": round(mean, 5), "sd": round(sd, 5), "se": round(se, 5),
            "lcb": round(mean - t * se, 5), "ucb": round(mean + t * se, 5)}


def pairs_needed(effect: float, sd_diff: float, conf: float = 0.9, power: float = 0.8) -> int:
    """Pairs for a one-sided paired test at `conf` to detect a mean gain `effect` with probability `power`,
    when single paired differences have standard deviation sd_diff (normal approximation, plus one for the t)."""
    if effect <= 0 or sd_diff <= 0:
        raise ValueError("effect and sd_diff must be positive")
    z = _Z.inv_cdf(conf) + _Z.inv_cdf(power)
    return max(2, math.ceil((z * sd_diff / effect) ** 2) + 1)


def divergence(a: Counter, b: Counter) -> float:
    """Jensen-Shannon divergence (base 2, between 0 and 1) of two counts of actions: 0 when the mixes are the same."""
    ta, tb = sum(a.values()), sum(b.values())
    if not ta or not tb:
        return 0.0 if ta == tb else 1.0
    out = 0.0
    for key in set(a) | set(b):
        p, q = a.get(key, 0) / ta, b.get(key, 0) / tb
        m = (p + q) / 2
        if p:
            out += 0.5 * p * math.log2(p / m)
        if q:
            out += 0.5 * q * math.log2(q / m)
    return round(max(0.0, min(1.0, out)), 5)


def area(scores: list, start: int = 0) -> float | None:
    """The progress score: the mean of a run's score over its ticks after `start` (the area under its progress
    curve, per tick). It rewards reaching a result sooner and moves less from run to run than the last tick alone."""
    vals = [s for s in scores[start:] if s is not None]
    return round(sum(vals) / len(vals), 5) if vals else None
