"""Offline backend for testing the pipeline at scale.

Its structure is SCRIPTED, not emergent: weighted heuristics that call every
primitive through the kernel, so runs exercise permissions, nesting, tasks,
custom actions and the science world without any API calls.
"""
from __future__ import annotations

import asyncio
import random
import re

from .kernel import CustomAction, Kernel


class MockBackend:
    def __init__(self, seed: int = 0, proposes: dict | None = None):
        self.rng = random.Random(seed)
        self.focus: dict[tuple, int] = {}  # (society, agent) -> output it works on
        self.proposes = proposes  # {"key": ..., "value": ...}: what the first agent proposes after the last tick, if anything

    async def closing(self, k: Kernel, aid: str):
        """After the last tick (alphateam/runner.py): scripted agents say nothing, unless a proposal was scripted. Then
        the first agent proposes it and the second backs it."""
        if not self.proposes or getattr(k, "rulebook", None) is None:
            return
        k.closing_now = True
        first, second = (sorted(k.active_agents(), key=lambda a: int(a[1:])) + [None, None])[:2]
        if self.proposes.get("suggest"):  # a rehearsal of looking back: the first agent reads the record and suggests a rule
            if aid == first:
                await k.act(aid, "rules", {"op": "record", "what": "timeline"})
                await k.act(aid, "rules", {"op": "suggest", "text": self.proposes["suggest"], "evidence": ["t1"]})
            return
        if aid == first:
            await k.act(aid, "rules", {"op": "propose", "key": self.proposes["key"], "value": self.proposes["value"],
                                       "why": "scripted: a rehearsal of what an agent might propose after a run"})
        elif aid == second:
            await asyncio.sleep(0)  # after the first agent's proposal
            same = next((p["id"] for p in k.rulebook.proposals.values() if p["key"] == self.proposes["key"]), None)
            if same:
                await k.act(aid, "rules", {"op": "back", "id": same})

    def fork(self, old: Kernel, new: Kernel) -> "MockBackend":
        """A backend for a copy of a running society (alphateam/paired.py): the same script from the same point."""
        import copy
        twin = copy.deepcopy(self)
        twin.focus = {(id(new) if kid == id(old) else kid, aid): f for (kid, aid), f in self.focus.items()}
        return twin

    async def turn(self, k: Kernel, aid: str):
        if not k.agents[aid].active:
            return
        k.begin_turn(aid)
        k.observe(aid)
        rng, env = self.rng, k.env
        if hasattr(env, "mock_actions"):  # a benchmark world (alphateam/bench)
            return await self._bench_turn(k, aid)
        outs = env.instruments[aid]
        f = self.focus.setdefault((id(k), aid), rng.choice(outs))
        n = 0

        async def do(_tool, **a):
            nonlocal n
            n += 1
            return await k.act(aid, _tool, a)

        groups_on = "connect" in k.allowed and (k.allowed["connect"] is None or "form_group" in k.allowed["connect"])
        if groups_on and not k.member_of.get(aid):
            target = next((g for g in k.active_groups() if g.name == f"y{f} study" and g.open), None)
            if target:
                await do("connect", op="join", target=target.id)
            else:
                await do("connect", op="form_group", name=f"y{f} study", purpose=f"Pool data on y{f} and publish its law.")
        while n < k.max_actions:
            mine = sorted(k.member_of.get(aid, ()), key=lambda g: int(g[1:]))
            board = mine[0] if mine else "private"
            left = env.budget[aid] - env.used[aid]
            r = rng.random()
            if r < 0.3 and left > 0:
                await do("experiment", design="random", runs=min(3, left), to=board)
            elif r < 0.55:
                ok, msg = await do("analyze", op="screen", output=f"y{f}")
                if ok and n < k.max_actions:
                    top = re.findall(r"(x\d+) r=", msg)[:3]
                    ok, msg = await do("analyze", op="fit", output=f"y{f}", terms=top)
                    cv = re.search(r"cross-validated R² ([\d.]+)", msg) if ok else None
                    law = env.laws.get(f)
                    if cv and float(cv.group(1)) > 0.5 and n < k.max_actions and (law is None or (law.val or 0) < float(cv.group(1))):
                        await do("publish", output=f"y{f}", formula=msg.split("formula: ", 1)[1].strip())
            elif r < 0.62 and mine and groups_on:
                g = mine[0]
                others = [x for x in k.active_groups() if x.id != g and x.name in {f"y{o} study" for o in outs if o != f}]
                if others and not k.member_of.get(g) and k.can_act_for(aid, g):
                    await do("connect", op="join", target=rng.choice(others).id, **{"as": g})
                else:
                    await do("search", what="groups", query=f"y{rng.choice(outs)}")
            elif r < 0.66:
                await do("write", board=f"pub:{mine[0]}" if mine and rng.random() < 0.5 else "global",
                         content=f"{aid}: working on y{f}; my data is on {board}.", kind="note")
            elif r < 0.69:
                target = rng.choice([g.id for g in k.active_groups()] or ["society"])
                await do("critique", target=target, content="Too many overlapping groups on the same outputs; merge efforts.",
                         proposal="Name one delegate per output.")
            elif r < 0.72:
                await do("reflect", content=f"Focus on y{f}; publish only when cross-validated R² beats the current law.")
            elif r < 0.75 and "design_action" in k.allowed:
                if k.custom and rng.random() < 0.7:
                    await do("design_action", op="adopt", name=rng.choice(list(k.custom)))
                else:
                    await do("design_action", op="define", name=f"pool_{aid.lower()}", description="Post my data to my group and screen it.",
                             procedure="1) experiment design=random to=my group. 2) analyze screen my output. 3) write the top inputs to the group.",
                             params=["output"])
            elif r < 0.77 and k.agents[aid].actions:
                await do(f"x_{rng.choice(sorted(k.agents[aid].actions))}", output=f"y{f}")
            elif r < 0.80 and mine and "protocol" in k.allowed:
                g = k.groups[mine[0]]
                if g.founder == aid:
                    await do("protocol", group=g.id, delegates=[aid], text="The founder speaks for the group; members post data to mem.")
                else:
                    await do("read", board=g.id, last=5)
            elif r < 0.83:
                open_tasks = [t for t in k.tasks.values() if t.status == "open" and t.to is None]
                delivered = [t for t in k.tasks.values() if t.status == "delivered" and t.requester == aid]
                mine_taken = [t for t in k.tasks.values() if t.taker == aid and t.status == "taken"]
                if delivered:
                    await do("delegate", op="review", task=delivered[0].id, verdict="accept", notes="ok")
                elif mine_taken:
                    await do("delegate", op="deliver", task=mine_taken[0].id, deliverable=f"screened y{f}; see my posts")
                elif open_tasks:
                    await do("delegate", op="take", task=rng.choice(open_tasks).id)
                else:
                    await do("delegate", op="post", title=f"Collect 10 runs varying inputs of y{f}", spec="Post data to my group.")
            elif r < 0.86:
                ok, msg = await do("search", what="agents", query=f"y{f}")
                hit = re.search(r"- (A\d+)", msg)
                if hit and n < k.max_actions:
                    await do("connect", op="follow", target=hit.group(1))
            elif r < 0.88 and mine:
                await do("summarize", sources=[mine[0]], to=f"pub:{mine[0]}", focus="what outsiders need")
            elif r < 0.90 and mine and groups_on:
                await do("connect", op="leave", target=mine[-1])
                self.focus[(id(k), aid)] = rng.choice(outs)
            elif r < 0.94 and "link" in k.allowed:
                asked = [k.links[key].requester for key in k.links_of.get(aid, ())
                         if k.links[key].status == "pending" and k.links[key].requester != aid]
                opened = k.open_links(aid)
                peers = [x for x in env.measured_by[f] if x != aid and k.agents[x].active]
                if asked:
                    await do("link", op="open", target=asked[0])
                elif opened and rng.random() < 0.6:
                    await do("link", op="send", target=rng.choice(opened), content=f"y{f}: my data is on {board}.")
                elif peers:
                    await do("link", op="open", target=rng.choice(peers), note=f"we both measure y{f}")
            else:
                await do("search", what="laws", query=f"y{f}")
        k.log_intent(aid, f"(mock) focusing on y{f}")

    async def _bench_turn(self, k: Kernel, aid: str):
        """Scripted: join or form a group per home task, then let the world script the work (open, tools, submit)."""
        rng, env = self.rng, k.env
        home = env.home.get(aid) or list(env.tasks)
        t = self.focus.setdefault((id(k), aid), rng.choice(home))
        n = 0
        groups_on = "connect" in k.allowed and (k.allowed["connect"] is None or "form_group" in k.allowed["connect"])
        if groups_on and not k.member_of.get(aid):
            target = next((g for g in k.active_groups() if g.name == f"{t} team" and g.open), None)
            if target:
                await k.act(aid, "connect", {"op": "join", "target": target.id})
            else:
                await k.act(aid, "connect", {"op": "form_group", "name": f"{t} team", "purpose": f"Answer {t}."})
            n += 1
        for name, args in env.mock_actions(k, aid, t, rng):
            if n >= k.max_actions:
                break
            ok, msg = await k.act(aid, name, args)
            n += 1
            k.interrupts(aid)  # sync: what an LLM agent would receive appended to this result
            if ok and name != "submit" and n < k.max_actions and k.member_of.get(aid) and rng.random() < 0.3:
                g = sorted(k.member_of[aid])[0]
                await k.act(aid, "write", {"board": g, "kind": "result", "content": f"{t}: {msg[:300]}"})
                n += 1
        if k.sync:
            k.end_turn(aid, f"{aid} t{k.tick}: played {t}; {n} actions")

    async def run_custom(self, k: Kernel, aid: str, ca: CustomAction, args: dict) -> str:
        mine = sorted(k.member_of.get(aid, ()))
        await k.act(aid, "experiment", {"design": "random", "runs": 1, "to": mine[0] if mine else "private"}, via=ca.name)
        return f"x_{ca.name}: ran one step"

    async def summarize(self, k: Kernel, aid: str, text: str, focus: str) -> str:
        return f"(mock summary for {focus or 'general'}) {len(text.splitlines())} posts; latest: {text.splitlines()[-1][:160] if text else ''}"

    async def retrospective(self, k: Kernel, metrics: list[dict]) -> str:
        return "(mock retrospective) Groups formed around single outputs; nesting followed shared instruments."

    async def generation_review(self, k: Kernel, societies: list[dict], n_experiments: int) -> dict:
        best = next(s for s in societies if s["selected"])
        return {"lessons": f"(mock review) society {best['society']} scored best.",
                "experiments": [f"(mock experiment {i + 1}) form a group per shared knob" for i in range(n_experiments)]}
