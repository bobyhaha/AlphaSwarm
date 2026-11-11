"""LLM-driven agents: one agent loop over two provider adapters.

- OpenAICompatible: OpenRouter (default; e.g. deepseek/deepseek-v4.1-flash) or DeepSeek's own API,
  both in Chat Completions format.
- AnthropicProvider: Claude through the Anthropic SDK.

Each agent's turn is a short tool-use loop: observe, call up to max_actions
primitives, then a closing call with tool_choice none for a one-line intent. With a per-call limit instead
(kernel max_calls), an agent gets that many replies per tick and each reply may contain any number of actions
(up to PER_REPLY, a guard against runaway output), then the closing call.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from dataclasses import dataclass, field

from .kernel import CustomAction, Kernel, clip
from .primitives import system_prompt, tool_schemas

CUSTOM_STEPS = 4
PER_REPLY = 50
TOOL_MARKUP = re.compile(r"<｜DSML｜|<\|tool[▁_ ]calls|<tool_call>")

SUMMARY_SYSTEM = (
    "You compress boards of a society of AI agents into a summary for the readers described in the focus. "
    "Keep findings, numbers, decisions, open questions, owners and disagreements; drop chatter. "
    "Under 200 words. Refer to agents, groups, entries, tasks and outputs by id.")

RETRO_SYSTEM = (
    "You write the organizational retrospective of a society of AI agents, for a future society facing a different "
    "instance of a similar mission. Focus on how they organized and coordinated, not on domain facts.")

REVIEW_SYSTEM = (
    "You review one generation of societies of AI agents that worked in parallel on the same instance of a mission, each "
    "trying a different organizational experiment, for the next generation. Learn from the comparison: attribute score "
    "differences to organizational differences only as far as the evidence supports, and say when it does not. Focus on "
    "how they organized and coordinated, not on domain facts. Reply with JSON only.")


class Fatal(Exception):
    """Configuration, authentication or credit errors: stop the run instead of skipping turns."""


@dataclass
class Call:
    id: str
    name: str
    args: dict | None
    raw: str = ""


@dataclass
class Reply:
    text: str
    calls: list[Call]
    finish: str  # ok | length | refused
    tokens: dict
    assistant: list = field(default_factory=list)


class OpenAICompatible:
    PRESETS = {
        "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                       "model": "deepseek/deepseek-v4.1-flash"},
        "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-flash"},
    }

    def __init__(self, preset: str = "openrouter", model: str | None = None, reasoning: bool = False,
                 max_tokens: int | None = None, client=None, run_id: str = "", reasoning_effort: str | None = None,
                 provider_ignore: list[str] | None = None):
        import openai
        self.openai = openai
        p = self.PRESETS[preset]
        if client is None:
            key = os.environ.get(p["key_env"])
            if not key:
                raise Fatal(f"set {p['key_env']} to use the {preset} backend")
            client = openai.AsyncOpenAI(api_key=key, base_url=p["base_url"], max_retries=5, timeout=300)
        self.client = client
        self.preset, self.model, self.reasoning = preset, model or p["model"], reasoning
        self.max_tokens = max_tokens or (16000 if reasoning else 3000)
        self.run_id = run_id
        self.reasoning_effort, self.provider_ignore = reasoning_effort, list(provider_ignore or [])
        # a malformed response body surfaces as a JSON decode error from the client; treat it like a dropped call
        self.transient = (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError, TimeoutError,
                          json.JSONDecodeError)

    def tools(self, schemas: list[dict]) -> list[dict]:
        return [{"type": "function", "function": s} for s in schemas]

    def first(self, text: str) -> list:
        return [{"role": "user", "content": text}]

    def results(self, results: list[tuple]) -> list:
        return [{"role": "tool", "tool_call_id": cid, "content": out} for cid, ok, out in results]

    def _extra(self, who: str) -> dict:
        if self.preset == "openrouter":
            extra = {"reasoning": {"effort": self.reasoning_effort} if self.reasoning_effort else {"enabled": bool(self.reasoning)}}
            if self.provider_ignore:
                extra["provider"] = {"ignore": self.provider_ignore}
            if who:
                extra["session_id"] = f"{self.run_id}-{who}"[:256]  # sticky provider routing keeps prompt caches warm
            return extra
        return {"thinking": {"type": "enabled" if self.reasoning else "disabled"}}

    async def create(self, system: str, messages: list, tools: list, final: bool, who: str,
                     temperature: float | None = None, json_mode: bool = False) -> Reply:
        oa = self.openai
        kwargs = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages,
                  "max_tokens": self.max_tokens, "extra_body": self._extra(who)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none" if final else "auto"
        resp = None
        for attempt in range(6):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
            except (oa.AuthenticationError, oa.PermissionDeniedError) as exc:
                raise Fatal(f"{self.preset}: {exc}") from None
            except oa.APIStatusError as exc:
                if exc.status_code == 402:
                    body = json.dumps(getattr(exc, "body", "") or "", default=str)
                    if "in_flight" in body and attempt < 5:  # OpenRouter's transient in-flight budget
                        await asyncio.sleep(2 + 2 ** attempt)
                        continue
                    raise Fatal(f"{self.preset}: insufficient credits or budget ({exc})") from None
                raise
            if resp.choices:
                break
            await asyncio.sleep(2 ** attempt)  # an error delivered in a 200 body
        if not resp or not resp.choices:
            raise TimeoutError("no choices returned")
        choice = resp.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                args = args if isinstance(args, dict) else None
            except json.JSONDecodeError:
                args = None
            calls.append(Call(tc.id, tc.function.name, args, raw))
        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw}}
                                       for c in calls]
        extra = getattr(msg, "model_extra", None) or {}
        for f in ("reasoning_content", "reasoning", "reasoning_details"):  # echo reasoning back unchanged in tool loops
            v = extra.get(f) if f in extra else getattr(msg, f, None)
            if v:
                assistant[f] = v
        u = resp.usage
        hit = 0
        if u is not None:
            uextra = getattr(u, "model_extra", None) or {}
            hit = uextra.get("prompt_cache_hit_tokens") or 0
            details = getattr(u, "prompt_tokens_details", None)
            if not hit and details is not None:
                hit = getattr(details, "cached_tokens", 0) or 0
        tokens = {"input": max(0, (u.prompt_tokens if u else 0) - hit), "cached": hit,
                  "output": u.completion_tokens if u else 0,
                  "cost": (getattr(u, "model_extra", None) or {}).get("cost") if u else None}
        finish = {"length": "length", "content_filter": "refused"}.get(choice.finish_reason, "ok")
        text = TOOL_MARKUP.split(msg.content or "", 1)[0]  # some models write tool-call markup into text when tools are off
        return Reply(text, calls, finish, tokens, [assistant])


class AnthropicProvider:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        import anthropic
        self.anthropic = anthropic
        self.client = client or anthropic.AsyncAnthropic()
        self.model, self.effort = model, effort
        self.transient = (anthropic.APIStatusError, anthropic.APIConnectionError)

    def tools(self, schemas: list[dict]) -> list[dict]:
        return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in schemas]

    def first(self, text: str) -> list:
        return [{"role": "user", "content": text}]

    def results(self, results: list[tuple]) -> list:
        blocks = []
        for cid, ok, out in results:
            b = {"type": "tool_result", "tool_use_id": cid, "content": out}
            if not ok:
                b["is_error"] = True
            blocks.append(b)
        return [{"role": "user", "content": blocks}]

    async def create(self, system: str, messages: list, tools: list, final: bool, who: str) -> Reply:
        kwargs = {"system": system, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            if final:
                kwargs["tool_choice"] = {"type": "none"}
        try:
            resp = await self.client.beta.messages.create(
                model=self.model, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": self.effort}, cache_control={"type": "ephemeral"}, **kwargs)
        except self.anthropic.AuthenticationError as exc:
            raise Fatal(f"anthropic: {exc}") from None
        u = resp.usage
        tokens = {"input": (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0),
                  "cached": getattr(u, "cache_read_input_tokens", 0) or 0, "output": u.output_tokens or 0, "cost": None}
        calls = [Call(b.id, b.name, b.input if isinstance(b.input, dict) else None) for b in resp.content if b.type == "tool_use"]
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        finish = {"refusal": "refused", "max_tokens": "length"}.get(resp.stop_reason, "ok")
        return Reply(text, calls, finish, tokens, [{"role": "assistant", "content": resp.content}])


REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_details")
# what the last reply of a tick may still do with final_moves: make game moves and record what they showed (in the F runs
# 82% of closing notes were the engine's move summaries, so the reasoning behind the moves left no public trace)
MOVE_TOOLS = ("play", "replay")
FINAL_TOOLS = MOVE_TOOLS + ("guide", "plan")


def tool_name(t: dict) -> str:
    return (t.get("function") or {}).get("name") or t.get("name") or ""


CLOSING_USES = 8  # how often an agent may use the rules action after the last tick (looking at the record counts)


def closing_text(k: Kernel) -> str:
    """What an agent is told after the last tick, when its society's rules are open to proposals (alphateam/recipe.py).
    It states what is still possible and what becomes of it. It asks for nothing."""
    return (f"The run is over: tick {k.tick} was the last one, and nothing more can be added to the shared work. One "
            f"action is still open to you, rules, at most {CLOSING_USES} times. With it you can look at what happened "
            "in this run as the system kept it (op=record), propose a value for a setting, suggest a rule that is not "
            "a setting yet, back a proposal or a suggestion, or leave a lesson. Later societies start from what is left "
            "here. A proposal, or a suggestion once it is built, is tried by running societies with and without the "
            "change, and it is kept only if those with it do better. Whether you use it is up to you; a reply without "
            "it ends your part.\n\n" + k.rulebook.view())


def closing_schema(schema: dict) -> dict:
    """The rules action as it is offered after the last tick: the same action, with the two ops that look back."""
    from .recipe import CLOSING_OPS, CLOSING_TOOL
    props = dict(schema["parameters"]["properties"])
    props["op"] = {"type": "string", "enum": list(CLOSING_OPS)}
    props.update(what={"type": "string", "enum": ["timeline", "tick", "agent", "search"]}, tick={"type": "integer"},
                 agent={"type": "string"})
    return {**schema, "description": CLOSING_TOOL, "parameters": {**schema["parameters"], "properties": props}}


def shorten(msg: dict, tool_chars: int = 600) -> dict:
    """An earlier tick's message as kept in an agent's memory: reasoning dropped, long tool results and observations
    clipped; tool calls kept whole so each result still matches its call."""
    m = {k: v for k, v in msg.items() if k not in REASONING_FIELDS}
    content = m.get("content")
    if isinstance(content, str):
        m["content"] = clip(content, {"tool": tool_chars, "user": 1500}.get(m.get("role"), 1500))
    elif isinstance(content, list):  # Anthropic blocks
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
                continue
            if isinstance(b, dict) and isinstance(b.get("content"), str):
                b = {**b, "content": clip(b["content"], 600)}
            elif isinstance(b, dict) and b.get("type") == "text":
                b = {**b, "text": clip(b.get("text", ""), 1500)}
            out.append(b)
        m["content"] = out
    return m


class LLMBackend:
    def __init__(self, provider, concurrency: int = 64, strong=None, strong_agents: set[str] | None = None):
        self.p = provider
        self.sem = asyncio.Semaphore(concurrency)
        # a mixed society: these agents use a stronger provider (e.g. the same model with reasoning on); nobody is told
        self.strong, self.strong_agents = strong, set(strong_agents or ())
        # continuity (kernel.continuity = W ticks, world option continuity=W): each agent keeps its own conversation from
        # its last W ticks, older parts shortened, instead of starting every tick from a fresh one
        self.memory: dict[str, list] = {}

    def fork(self, old, new) -> "LLMBackend":
        """A backend for a copy of a running society (alphateam/paired.py): the same provider and request limit, and
        each agent's kept conversation copied, so the two branches remember the same past and nothing after it."""
        import copy
        twin = copy.copy(self)
        twin.memory = copy.deepcopy(self.memory)
        return twin

    def _provider(self, who: str):
        return self.strong if self.strong is not None and who in self.strong_agents else self.p

    async def _create(self, k: Kernel, who: str, category: str, system: str, messages: list, tools: list, final: bool):
        async with self.sem:
            reply = await self._provider(who).create(system, messages, tools, final, who)
        t = reply.tokens
        k.charge(who, category, t["input"] + t["cached"] + t["output"])
        k.usage.update({"input": t["input"], "cached": t["cached"], "output": t["output"], "calls": 1})
        if t.get("cost"):
            k.usage["cost_microusd"] += round(float(t["cost"]) * 1e6)
        return reply

    async def complete(self, k: Kernel, system: str, prompt: str, model: str | None = None, max_tokens: int = 2000,
                       who: str = "grader", temperature: float | None = None, json_mode: bool = False,
                       effort: str | None = None) -> str:
        """One plain completion for world-side helpers (benchmark judges, milestone answerers). It may name a different
        model (e.g. the benchmark's judge); its cost counts toward the society's bill like any other call."""
        p = copy.copy(self._provider(who))
        if model:
            p.model = model
        if effort and hasattr(p, "reasoning_effort"):
            p.reasoning_effort = effort
        p.max_tokens = max_tokens
        extra = {}
        if temperature is not None:
            extra["temperature"] = temperature
        if json_mode:
            extra["json_mode"] = True
        async with self.sem:
            reply = await p.create(system, p.first(prompt), [], False, who, **extra)
        t = reply.tokens
        k.usage.update({"input": t["input"], "cached": t["cached"], "output": t["output"], "calls": 1})
        k.tick_counts["tok_grader"] += t["input"] + t["cached"] + t["output"]
        if t.get("cost"):
            k.usage["cost_microusd"] += round(float(t["cost"]) * 1e6)
        return reply.text

    async def _loop(self, k: Kernel, aid: str, messages: list, tools: list, budget: int, via: str | None, category: str,
                    max_calls: int | None = None) -> str:
        used, note, system, calls = 0, "", system_prompt(k), 0
        last_tools, move_tools = getattr(k.env, "final_tools", FINAL_TOOLS), getattr(k.env, "move_tools", MOVE_TOOLS)
        while True:
            final = used >= budget or (max_calls is not None and calls >= max_calls)
            # final_moves (world option): the last reply of a tick may still make game moves and record, with its closing note
            moves = [t for t in tools if tool_name(t) in last_tools] if final and used < budget and category == \
                "deliberation" and getattr(k, "final_moves", False) else []
            try:
                reply = await self._create(k, aid, category, system, messages, moves or tools, final and not moves)
                calls += 1
                k.trace.append({"t": k.tick, "agent": aid, "kind": "reply", "category": category, "text": reply.text,
                                "calls": [{"name": c.name, "args": c.args} for c in reply.calls], "finish": reply.finish})
            except Fatal:
                raise
            except self.p.transient as exc:
                k.log_intent(aid, f"[turn cut short: {exc.__class__.__name__}: {clip(exc, 160)}]")
                return note
            if reply.finish == "refused":
                return "[declined]"
            note = reply.text.strip() or note
            if not reply.calls and not final and category == "deliberation" and getattr(k.env, "force_budget", False):
                messages.extend(reply.assistant)  # a budget is a budget: stopping early is refused, as extended thinking is forced
                messages.extend(self.p.first(f"You have {max_calls - calls} replies left this tick. Keep working: check what you have, "
                                             "try what is still open, and improve what you can."))
                continue
            if not reply.calls or (final and not moves) or reply.finish == "length":
                return note
            messages.extend(reply.assistant)
            results, refused = [], {}  # identical calls refused earlier in this reply are not run again
            for n, c in enumerate(reply.calls):
                sig = (c.name, json.dumps(c.args, sort_keys=True, default=str)) if c.args is not None else None
                if sig in refused:
                    ok, out = False, (f"refused: the same call was refused earlier in this reply ({clip(refused[sig], 160)}); "
                                      f"change its arguments instead of repeating it")
                elif c.args is None:
                    ok, out = False, "refused: the arguments were not valid JSON"
                elif used >= budget:
                    ok, out = False, f"refused: you have used all {budget} actions for this tick"
                elif max_calls is not None and n >= PER_REPLY:
                    ok, out = False, f"refused: at most {PER_REPLY} actions per reply"
                elif moves and c.name not in last_tools:
                    ok, out = False, f"refused: the last reply of a tick may only use {', '.join(last_tools)}"
                else:
                    ok, out = await k.act(aid, c.name, c.args, via=via)
                    used += 1
                    if not ok and sig is not None:
                        refused[sig] = out.removeprefix("refused: ")
                left = (f"replies left this tick: {max(0, max_calls - calls)}" if max_calls is not None else
                        f"uses left: {budget - used}" if category == "closing" else f"actions left this tick: {budget - used}")
                if getattr(k, "final_moves", False) and max_calls is not None and calls >= max_calls and not moves:
                    words = getattr(k.env, "final_words", "make game moves and record what you learned")
                    left += f"; your last reply may still {words} ({', '.join(last_tools)}), with your closing note"
                results.append((c.id, ok, f"{out}\n[{left}]"))
            if results and category == "deliberation":  # sync: messages and urgent entries reach the agent mid-turn
                notices = k.interrupts(aid)
                if notices:
                    cid, ok, text = results[-1]
                    results[-1] = (cid, ok, text + notices)
            messages.extend(self.p.results(results))
            if moves:  # the moves of the last reply end the tick; its text is the closing note
                if reply.text.strip():
                    return reply.text.strip()
                # no text: the note given with the moves, else what those moves did, as the engine reported it
                noted = [str(c.args.get("note")).strip() for c in reply.calls
                         if c.args and c.name in move_tools and str(c.args.get("note") or "").strip()]
                done = [clip(" ".join(out.split("\n")[:3]), 300) for _, ok, out in results if ok]
                if noted:
                    return clip(" ".join(noted), 800) + (f" (last moves: {clip(' | '.join(done), 300)})" if done else "")
                if getattr(k.env, "quiet_closing", False):
                    return ""
                return "My last moves this tick: " + (" | ".join(done) if done else "(refused)")

    async def turn(self, k: Kernel, aid: str):
        if not k.agents[aid].active or k.exhausted():
            return
        k.begin_turn(aid)
        tools = self.p.tools(tool_schemas(k, aid))
        calls = getattr(k, "max_calls", None)
        keep = int(getattr(k, "continuity", 0) or 0)
        kept = int(getattr(k.env, "memory_chars", 600))  # worlds whose replies act on what the last tick's commands printed keep more of it
        past = [shorten(m, kept) for block in self.memory.get(aid, [])[-keep:] for m in block] if keep else []
        shown = k.observe(aid)
        if not getattr(k, "_traced_system", False):  # once a run: the system prompt and the actions on offer
            k._traced_system = True
            k.trace.append({"t": k.tick, "agent": aid, "kind": "system", "text": system_prompt(k),
                            "tools": [t.get("name") for t in tool_schemas(k, aid)]})
        k.trace.append({"t": k.tick, "agent": aid, "kind": "turn", "shown": shown})
        messages = past + self.p.first(shown)
        note = await self._loop(k, aid, messages, tools, 10 ** 6 if calls else k.max_actions, None,
                                "deliberation", max_calls=calls)
        if keep:  # this tick's exchange, closed by the agent's note, joins its memory
            block = messages[len(past):] + [{"role": "assistant", "content": note.strip() or "(no note this tick)"}]
            self.memory[aid] = (self.memory.get(aid, []) + [block])[-keep:]
        k.log_intent(aid, note)
        k.end_turn(aid, note)  # the closing note is the agent's notebook entry for this turn

    async def closing(self, k: Kernel, aid: str):
        """After the last tick: an agent may still use the rules action (propose, back, lesson), and nothing else. It
        sees the rules it ran under and what was proposed so far. The shared work is final by then, so nothing said
        here can move the run's score."""
        if getattr(k, "rulebook", None) is None or not k.agents[aid].active or k.exhausted():
            return
        tools = self.p.tools([closing_schema(t) for t in tool_schemas(k, aid) if t["name"] == "rules"])
        if not tools:
            return
        k.closing_now = True  # the ops that look back open now (alphateam/recipe.py Rulebook.act)
        k.begin_turn(aid)
        keep = int(getattr(k, "continuity", 0) or 0)
        kept = int(getattr(k.env, "memory_chars", 600))
        past = [shorten(m, kept) for block in self.memory.get(aid, [])[-keep:] for m in block] if keep else []
        shown = closing_text(k)
        k.trace.append({"t": k.tick, "agent": aid, "kind": "closing", "shown": shown})
        note = await self._loop(k, aid, past + self.p.first(shown), tools, CLOSING_USES, None, "closing")
        if note.strip():
            k.events.append({"t": k.tick, "agent": aid, "op": "closing.note", "ok": True, "target": None, "via": None,
                             "args": {}, "msg": clip(note, 300)})

    async def run_custom(self, k: Kernel, aid: str, ca: CustomAction, args: dict) -> str:
        task = (f"{k.observe(aid, consume=False)}\n\nYou are now carrying out the custom action x_{ca.name} "
                f"(v{ca.version}, defined by {ca.author}).\nProcedure:\n{ca.procedure}\n\nArguments: {json.dumps(args)}\n\n"
                f"Execute the procedure with at most {CUSTOM_STEPS} primitive actions, then reply with one line on what it accomplished.")
        tools = self.p.tools(tool_schemas(k, aid, custom=False))
        note = await self._loop(k, aid, self.p.first(task), tools, CUSTOM_STEPS, ca.name, "custom")
        return note or f"x_{ca.name} ran"

    async def summarize(self, k: Kernel, aid: str, text: str, focus: str) -> str:
        reply = await self._create(k, aid, "summarize", SUMMARY_SYSTEM, self.p.first(f"Focus: {focus or 'general'}\n\n{text}"), [], False)
        return reply.text.strip() or "(empty summary)"

    async def retrospective(self, k: Kernel, metrics: list[dict]) -> str:
        groups = sorted(k.active_groups(), key=lambda g: -len(k.agents_inside(g.id)))[:15]
        structure = [k.group_card(g.id) for g in groups]
        notes = [k.fmt(e, 400) for b in ("global", "library") for e in k.boards[b].entries
                 if e.kind in ("critique", "lesson", "decision", "proposal")][-40:]
        prompt = (f"Metrics by tick:\n{json.dumps(metrics)}\n\nLargest groups at the end:\n" + "\n".join(structure or ["none"]) +
                  "\n\nCritiques, proposals and lessons posted publicly:\n" + ("\n".join(notes) or "none") +
                  "\n\nWrite 5-8 transferable lessons about organizing: which structures, protocols and habits helped or hurt, "
                  "and under what conditions. Cite evidence from the run. End with one organizational experiment the next "
                  "society should try (a concrete change in structure, protocol or decision rights that this society did "
                  "not try) and the result that would show it helped.")
        reply = await self._create(k, "society", "retrospective", RETRO_SYSTEM, self.p.first(prompt), [], False)
        return reply.text.strip()

    async def generation_review(self, k: Kernel, societies: list[dict], n_experiments: int) -> dict:
        """Compare one generation's societies; return lessons for the next and n_experiments new experiments."""
        prompt = ("Societies of this generation (same mission instance, same agents' instructions except the assigned "
                  "experiment; the score is the mission's own):\n" + json.dumps(societies, indent=1) +
                  f"\n\nReturn JSON: {{\"lessons\": \"5-8 transferable lessons about organizing, citing this comparison\", "
                  f"\"experiments\": [{n_experiments} distinct organizational experiments for the next generation, each a "
                  "concrete change in structure, protocol or decision rights that the next society can enact with its "
                  "primitives (who forms which groups, who may act for them, who holds which decisions, how work is split "
                  "and checked), with the result that would show it helped; they should differ from each other and from "
                  "what the best society did]}")
        text = await self.complete(k, REVIEW_SYSTEM, prompt, max_tokens=3000, who="retrospective", json_mode=True)
        try:
            out = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except ValueError:
            out = {}
        lessons = out.get("lessons") or text
        exps = [str(e) if not isinstance(e, dict) else json.dumps(e) for e in (out.get("experiments") or [])]
        return {"lessons": lessons if isinstance(lessons, str) else json.dumps(lessons), "experiments": exps[:n_experiments]}


# DeepSeek's published prices per 1M tokens (peak; off-peak is half), docs read 2026-10-30. OpenRouter reports
# exact cost per call, so this table is only used for DeepSeek's own API.
DEEPSEEK_PRICES = {"deepseek-flash": (0.006, 0.3, 1.2), "deepseek-v4-pro": (0.044, 1.32, 3.96)}


def cost_usd(k: Kernel, preset: str, model: str) -> float | None:
    if k.usage.get("cost_microusd"):
        return k.usage["cost_microusd"] / 1e6
    if preset == "deepseek" and model in DEEPSEEK_PRICES:
        hit, miss, out = DEEPSEEK_PRICES[model]
        return (k.usage["cached"] * hit + k.usage["input"] * miss + k.usage["output"] * out) / 1e6
    return None
