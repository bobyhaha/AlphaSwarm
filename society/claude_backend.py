"""Claude-driven agents: each agent's turn is a short tool-use loop over the action space."""
from __future__ import annotations

import asyncio
import json

import anthropic

from .actions import SYSTEM, tool_schemas
from .world import CustomAction, World, clip

CUSTOM_STEPS = 4

SUMMARY_SYSTEM = (
    "You compress a research society's board(s) into a summary for the readers described in the focus. "
    "Keep decisions, numbers, evidence, open questions, owners, and disagreements; drop chatter. "
    "Under 250 words. Refer to entries, blocks, agents and teams by id.")

RETRO_SYSTEM = (
    "You are writing the organizational retrospective of a society of AI research agents, to be handed to "
    "a future society working on a different mission. Focus on how they organized, not on the domain.")


class ClaudeBackend:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", concurrency: int = 8,
                 client: anthropic.AsyncAnthropic | None = None):
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model
        self.effort = effort
        self.sem = asyncio.Semaphore(concurrency)

    async def _create(self, world: World, who: str, category: str, **kwargs):
        async with self.sem:
            resp = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                **kwargs,
            )
        u = resp.usage
        world.charge(who, category, sum(getattr(u, f, 0) or 0 for f in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")))
        return resp

    async def _loop(self, world: World, aid: str, messages: list, tools: list, budget: int,
                    via: str | None, category: str) -> str:
        """Run tool calls until the agent stops or its action budget is spent; return its closing note."""
        used, note = 0, ""
        while True:
            try:
                kwargs = {"system": SYSTEM, "tools": tools, "messages": messages}
                if used >= budget:
                    kwargs["tool_choice"] = {"type": "none"}
                resp = await self._create(world, aid, category, **kwargs)
            except anthropic.AuthenticationError:
                raise
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                world.log_intent(aid, f"[turn cut short: {exc.__class__.__name__}]")
                return note
            if resp.stop_reason == "refusal":
                return "[declined]"
            note = " ".join(b.text for b in resp.content if b.type == "text").strip() or note
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls or used >= budget or resp.stop_reason == "max_tokens":
                return note
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for call in calls:
                if used >= budget:
                    ok, out = False, "refused: no actions left this tick"
                else:
                    ok, out = await world.act(aid, call.name, call.input, via=via)
                    used += 1
                result = {"type": "tool_result", "tool_use_id": call.id, "content": out}
                if not ok:
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})

    async def turn(self, world: World, aid: str):
        left = world.budget_left()
        if left is not None and left <= 0:
            return
        messages = [{"role": "user", "content": world.observe(aid)}]
        note = await self._loop(world, aid, messages, tool_schemas(world, aid), world.max_actions, None, "deliberation")
        world.log_intent(aid, note)

    async def run_custom(self, world: World, aid: str, ca: CustomAction, args: dict) -> str:
        task = (f"{world.observe(aid, consume_inbox=False)}\n\n"
                f"You are now carrying out the custom action x_{ca.name} (v{ca.version}, defined by {ca.author}).\n"
                f"Procedure:\n{ca.procedure}\n\nArguments: {json.dumps(args)}\n\n"
                f"Execute the procedure with at most {CUSTOM_STEPS} primitive actions, "
                f"then reply with one line on what it accomplished.")
        note = await self._loop(world, aid, [{"role": "user", "content": task}],
                                tool_schemas(world, aid, custom=False), CUSTOM_STEPS, ca.name, "custom")
        return note or f"x_{ca.name} ran"

    async def summarize(self, world: World, aid: str, text: str, focus: str) -> str:
        resp = await self._create(world, aid, "summarize", system=SUMMARY_SYSTEM,
                                  messages=[{"role": "user", "content": f"Focus: {focus or 'general'}\n\n{text}"}])
        return "".join(b.text for b in resp.content if b.type == "text").strip() or "(empty summary)"

    async def retrospective(self, world: World) -> str:
        traj = [{k: s["metrics"][k] for k in ("teams", "depth", "free", "links", "custom", "done", "reorganizations", "tokens")}
                | {"t": s["t"]} for s in world.snapshots]
        notes = [world.fmt(e, 500) for b in world.boards.values() for e in b.entries
                 if e.kind in ("critique", "reflection", "decision")][-60:]
        structure = [f"{t['id']} '{t['name']}' parent={t['parent']} rule={t['rule']} members={t['members']} lead={t['lead']}"
                     for t in world.snapshots[-1]["teams"]]
        custom = [f"x_{c.name} v{c.version} uses={c.uses}: {clip(c.description, 160)}" for c in world.custom.values()]
        prompt = (f"Mission: {world.mission}\n\nMetrics by tick:\n{json.dumps(traj)}\n\nFinal structure:\n" + "\n".join(structure)
                  + "\n\nCustom actions:\n" + ("\n".join(custom) or "none")
                  + "\n\nCritiques, reflections and decisions:\n" + "\n".join(notes)
                  + "\n\nWrite 5-8 transferable lessons about organizing (what structures, protocols and habits helped "
                    "or hurt, and under what conditions). Be specific and cite evidence from the run.")
        resp = await self._create(world, "society", "retrospective", system=RETRO_SYSTEM,
                                  messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()
