"""Real benchmarks as worlds for the society (see base.py)."""
from __future__ import annotations

import importlib

WORLDS = ("discoverybench", "scigym", "explorationbench")


def make_world(name: str, cfg: dict, n_agents: int, seed: int = 0, single: bool = False):
    if name not in WORLDS:
        raise ValueError(f"unknown world '{name}' (choose from hidden, {', '.join(WORLDS)})")
    world = importlib.import_module(f".{name}", __name__).make(cfg, n_agents, seed=seed, single=single)
    opts = cfg.get("world_opts") or {}
    world.claims_on = str(opts.get("claims", "")).lower() in ("1", "true", "yes")
    world.shuffle_search = opts.get("search_order") == "random"
    if int(opts.get("practice", 0) or 0) > 0:
        world.set_practice(int(opts["practice"]), str(opts.get("feedback", "")).lower() in ("1", "true", "yes"),
                           int(opts.get("feedback_every", 3)), seed)
    return world
