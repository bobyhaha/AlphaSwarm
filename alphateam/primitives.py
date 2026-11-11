"""The designed part: composable primitives (as tool schemas) and the shared system prompt.

The prompt describes what the primitives do and what things cost. It gives no
advice on how to organize: no roles, stages, hierarchy or workflow.
"""
from __future__ import annotations

from .kernel import CONNECT_OPS, DELEGATE_OPS, DESIGN_OPS, KINDS, LINK_OPS, SEARCH_KINDS, Kernel
from .recipe import RULES_TOOL

SYSTEM = """You are one of {n} AI agents in a society working on a shared mission. Nobody assigns roles, teams, leaders, workflows or stages. Only a set of composable primitives is designed in advance; everything else (who works with whom, groups and groups of groups, who speaks for a group, how work is split, when to synchronize, which new actions exist) is for the agents to invent, test and change at any time.

THE WORLD
- Boards are the society's memory: what you don't write down is gone when your turn ends. Your private board is yours; you may share it. Every group has a memory board (mem:G, open to everyone inside the group) and a public board (pub:G, its interface to everyone else: anyone can read it and write to it). The global board and the library are open to all; the library is kept for future societies.
- Groups. Anyone can form a group. A group's members can be agents or other groups, so groups can nest into larger units to any depth, and an agent can be in several groups. Everyone inside a nested group is inside every group that contains it. A group writes its own purpose and protocol, including who may act for it (its delegates; by default anyone inside). Acting for a group (admitting, removing, dissolving, changing its protocol, joining another group as that group, publishing or posting tasks on its behalf) needs that permission. Anyone can leave a group at any time.
- Links. A link is a private two-way channel between two agents. Messages on it go to a board only the two of them can read and to the other's inbox. Agents at the same level of a group (its direct members, and whoever may act for each of its member groups) can message each other directly. Any other pair needs both agents' agreement: one asks (link op=open), and the link opens when the other also opens it. Either side can close a link.
- Tasks. Anyone can post a task (to an agent, a group, or anyone), take it, deliver it, and have it reviewed by the requester's side, never by the taker. Tasks can have parent tasks, so work can be decomposed recursively.
- Your view is local and bounded. Each turn you see your groups, your inbox, what you follow, and headers of new posts. Reading full posts and searching spend a per-tick reading budget. Writing @A12 or @G3 notifies that agent or group.
- Costs are real and visible: tokens spent deliberating, reading, writing and summarizing are recorded, and reorganizations are counted.
- New actions. You can design a new action as a procedure over the primitives; others can adopt, revise or drop it.

{brief}

Be concrete. Keep posts compact: attention is the scarcest resource. Refer to agents, groups, entries, tasks and outputs by id. Several tool calls in one reply each count as an action. When you stop calling tools, your closing text is saved to your private board as a note to your future self (nobody else sees it unless you share the board): keep it short and specific."""


TEAMING = """WORKING WITH OTHERS
- Other agents are not only authors of posts: each is working on the same problem and holds pieces you do not have. {talk}{islands}
- Forming, joining or leaving a group is one tool call{cap}. A group's purpose is its direction: everyone sees it in the MEMBERSHIP block of each turn's view, with your island, who holds what you cannot see yet, the groups and their places, and who is in no group. A claim close to another group's direction is pointed out to you. Nothing requires you to join or form one.{extra}"""

TALK_AT = "Writing @A12 in any post reaches A12 at once (its inbox, during its turn if it is still acting); a link is a private two-way channel."
TALK_MESSAGE = ("message to=A12 text=.. (or to=G1 for a whole group) reaches them during their turn if they are acting, "
                "otherwise at their next turn; they answer the same way. @A12 in a post also reaches A12.")
ISLAND_GUIDES = """
- Each island keeps its own guide: the guide in your view is your island's; other islands' guides open to you like their posts, and each turn shows where open lines of different islands meet on the same subject (compare them; a receipt settles which holds)."""
UNIQUE = """
- When a level falls, every island that tested ideas on it that no other island tested earns credit for them (0.5 each, up to 3 per island, to whoever tested them), whether they held or not: covering different ground is work."""

ISLANDS = """
- Islands: your posts reach your island (you and everyone who shares a group with you) at once and everyone else {embargo} ticks later, sooner if their level falls or someone on your island publishes them (share entry=E.., {quota} per {window} ticks each). Sharing a group with someone puts you on one island, so each of you sees the other's claims, findings and guide lines as they are written. Receipts, the engine's notes, play data and published solutions reach everyone at once."""


def system_prompt(k: Kernel) -> str:
    brief = k.env.brief() if k.env else ""
    if getattr(k, "substrate", False):  # ScaleMAS-style shared substrate of linked posts
        brief += ("\n- The boards form one shared substrate of linked posts. Cite the entries a post builds on with refs; "
                  "search what=posts returns the most relevant posts you can read, ranked also by how often they are "
                  "cited (shown as 'cited Nx'), at most limit at a time; read entry=E12 visits one post by its id.")
    if getattr(k, "stakes", ""):  # set by a lineage: selection between societies and this society's assigned experiment
        brief += f"\n\n{k.stakes}"
    if getattr(k, "heritage", ""):  # lessons a recipe carries from earlier societies (alphateam/recipe.py)
        brief += f"\n\n{k.heritage}"
    if getattr(getattr(k, "rulebook", None), "note", ""):  # proposals can be tried inside this run (alphateam/trial.py)
        brief += f"\n\n{k.rulebook.note}"
    text = SYSTEM.format(n=len(k.agents), brief=brief)
    if getattr(k.env, "chairman", None):  # the chaired control arm: the prompt says what was set before the run, so it stays true
        text = text.replace("Nobody assigns roles, teams, leaders, workflows or stages.",
                            "One team and its lead were set before this run began (THE MISSION says who); nothing else is "
                            "assigned: no further roles, teams, workflows or stages.")
    if getattr(k, "final_moves", False):
        last = getattr(k.env, "final_tools", ("play", "replay", "guide", "plan"))
        words = getattr(k.env, "final_words", "make game moves and record what you learned")
        text += (f"\n\nYour last reply of a tick may still {words} ({', '.join(last)}; nothing else); put your closing note in "
                 "that same reply, as its text" + (" or as play's note." if "play" in last else "."))
    if getattr(k, "continuity", 0):  # each agent keeps its own recent conversation
        text = text.replace(
            "Boards are the society's memory: what you don't write down is gone when your turn ends.",
            f"You keep your own conversation from your last {k.continuity} ticks (older tool results shortened); beyond "
            f"that, boards are the society's memory: what you don't write down is gone.")
    if getattr(k, "sync", False):  # one public substrate, synchronized for each agent
        if not getattr(k, "private_boards", False):
            text = text.replace(
                "Your private board is yours; you may share it. ",
                "There are no private boards: every board can be read by everyone (link messages stay between the two "
                "agents). ")
        closing = ("When you stop calling tools, your closing text is saved to your private board as a note to your future self: keep it "
                   "short and specific. What others should see, you post yourself." if getattr(k, "private_boards", False) else
                   "When you stop calling tools, your closing text is posted to the global board as your note for this turn; "
                   "start it with its gist, since others receive the first line.")
        text = text.replace(
            "When you stop calling tools, your closing text is saved to your private board as a note to your future self "
            "(nobody else sees it unless you share the board): keep it short and specific.",
            closing + "\n\nSYNC: each turn shows your own recent notes "
            "and work (your memory between turns) and what others posted since your last turn, most relevant to you "
            "first; each post reaches you once, as its first line (read entry=<id> for the rest; search for older posts). "
            "Messages to you, and posts the world marks as urgent for your work, reach you during your turn, appended "
            "to the result of an action.")
    if getattr(k, "max_calls", None):  # limited by replies, not actions
        text = text.replace("Several tool calls in one reply each count as an action.",
                            "You may make any number of tool calls in one reply; what is limited is your replies per tick.")
    if getattr(k, "teaming", False):  # what working with others does (information, not an instruction to do it)
        isl = getattr(k, "islands", None)
        env = getattr(k, "env", None)
        extra = (ISLAND_GUIDES if getattr(env, "island_guides", False) else "") + \
            (UNIQUE if getattr(env, "unique_credit", False) else "")
        team = TEAMING.format(talk=TALK_MESSAGE if getattr(k, "messages_on", False) else TALK_AT,
                              islands=ISLANDS.format(**isl) if isl else "", extra=extra,
                              cap=f"; a group holds at most {isl['cap']} agents" if isl else "")
        if not isl:
            team = team.replace("with your island, who holds what you cannot see yet, the groups and their places", "with the groups")
        if getattr(env, "directions_on", False):  # directions=1 offers directions of its own; a group has a purpose
            team = team.replace("A group's purpose is its direction: everyone sees it", "Everyone sees a group's purpose") \
                .replace("another group's direction", "another group's purpose")
        text += "\n\n" + team
    return text


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


S = {"type": "string"}
AS = {"type": "string", "description": "Optional: a group id to act on behalf of (you must be allowed to act for it)"}
LIST = {"type": "array", "items": {"type": "string"}}

TOOLS = {
    "read": ("Read posts on a board you can access: 'global', 'library', 'private', an agent id (a private board shared with you), "
             "a group id 'G3' (its memory board, if you are inside), 'pub:G3' (its public board) or 'link:A12' (your link with "
             "A12). Newest first within the limit; spends your reading budget.",
             _obj({"board": S, "last": {"type": "integer", "description": "newest N matching posts (default 8, max 30)"},
                   "kind": S, "author": S, "contains": {"type": "string", "description": "only posts containing all these words"},
                   "entry": {"type": "string", "description": "one entry id"},
                   "full": {"type": "boolean", "description": "full text instead of 600-character excerpts"}}, ["board"])),
    "search": ("Find agents, groups, posts, tasks, custom actions or world-specific items (e.g. laws) matching words. "
               "Results spend your reading budget.",
               _obj({"what": {"type": "string"}, "query": S, "limit": {"type": "integer"}}, ["what"])),
    "write": ("Post to a board you can write: 'private', a group id (memory board, if inside), 'pub:G3', 'global', 'library', "
              "or an agent's private board shared with you for writing. Mention @A12 or @G3 to notify.",
              _obj({"board": S, "content": S, "kind": {"type": "string", "enum": list(KINDS)}, "refs": LIST}, ["board", "content"])),
    "share": ("Share your private board, or the memory board of a group you may act for, with agents, groups or 'all'; "
              "mode read, write or revoke.",
              _obj({"board": {"type": "string", "description": "'private' (default) or a group id"}, "with": LIST,
                    "mode": {"type": "string", "enum": ["read", "write", "revoke"]}}, [])),
    "summarize": ("Compress one or more boards you can read into a summary posted to a board you can write. Summaries are how "
                  "information crosses group boundaries; what they drop is lost to their readers.",
                  _obj({"sources": LIST, "to": S, "focus": {"type": "string", "description": "what the readers need"}}, ["sources", "to"])),
    "critique": ("Critique an agent (delivered privately), a group (posted on its public board), or the whole society "
                 "(target 'society', posted globally). Add a concrete proposal when you have one.",
                 _obj({"target": S, "content": S, "proposal": S}, ["target", "content"])),
    "connect": ("Change your connections or the society's structure. follow/unfollow an agent or group (see its new posts); "
                "form_group (name, purpose, optional open=false, protocol, invite list, as=group to found it as that group); "
                "join/leave target group (as=group to join or leave on behalf of a group you may act for: that is how groups nest); "
                "invite target into group; admit/remove target in group; dissolve target group.",
                _obj({"op": {"type": "string", "enum": list(CONNECT_OPS)}, "target": S, "group": S, "as": AS, "name": S,
                      "purpose": S, "protocol": S, "open": {"type": "boolean"}, "invite": LIST}, ["op"])),
    "link": ("A private two-way channel with another agent. With an agent at the same level of one of your groups (a direct "
             "member, or someone who may act for one of its member groups) you can send right away. Anyone else must agree: "
             "open asks the target (with an optional note), or agrees when the target has asked you; the link opens only "
             "once both have opened it. send: a message (it reaches the target's inbox and the link's board, read with "
             "board='link:A12'). close: end the link.",
             _obj({"op": {"type": "string", "enum": list(LINK_OPS)}, "target": S, "content": S, "note": S}, ["target"])),
    "message": ("Send a message to an agent (to=A12) or to everyone in a group (to=G1). It reaches them during their turn "
                "if they are acting, otherwise at their next turn; answer the same way (reply_to=M.. if you like).",
                _obj({"to": S, "text": S, "reply_to": S}, ["to", "text"])),
    "delegate": ("Hand a job or a decision to someone else, or take one. post: title, spec, to (an agent, a group, or omit for anyone; "
                 "the requester need not know how a group does it), scope (what it covers), accept (what would show it is done), until "
                 "(the tick it lapses), pay (credit held until it is accepted, where the world has credit), parent (a task you took: you "
                 "hand on part of it, within its scope), as=group to request on a group's behalf. take. refuse (notes). deliver "
                 "(deliverable, evidence: ids of what shows it). review (verdict accept or reject, notes; requester's side only, never "
                 "the taker). drop: the taker gives it back, or the requester revokes it, and what was handed on from it ends too. A "
                 "task can state interfaces and acceptance criteria and leave the details to whoever takes it.",
                 _obj({"op": {"type": "string", "enum": list(DELEGATE_OPS)}, "task": S, "title": S, "spec": S, "parent": S,
                       "to": S, "as": AS, "scope": LIST, "accept": S, "until": {"type": "integer"}, "pay": {"type": "number"},
                       "deliverable": S, "evidence": LIST, "verdict": {"type": "string", "enum": ["accept", "reject"]},
                       "notes": S}, ["op"])),
    "rules": (RULES_TOOL,
              _obj({"op": {"type": "string", "enum": ["view", "propose", "back", "lesson"]}, "key": S,
                    "value": {"type": ["string", "number", "boolean"]}, "why": S, "evidence": LIST, "id": S, "text": S,
                    "when": S}, ["op"])),
    "reflect": ("Revise your own strategy notes, shown at the top of your future turns.",
                _obj({"content": S}, ["content"])),
    "protocol": ("Rewrite the rules of a group you may act for: protocol text (how it works, how it decides), delegates "
                 "(member ids who may act for it; empty list = anyone inside), open (whether anyone may join), purpose, name.",
                 _obj({"group": S, "text": S, "delegates": LIST, "open": {"type": "boolean"}, "purpose": S, "name": S}, ["group"])),
    "design_action": ("Extend the action space. define: a named procedure over primitive actions (it becomes a tool x_<name> "
                      "for you and anyone who adopts it); revise; adopt; drop. A custom action runs as a short sequence of "
                      "primitive actions and cannot bypass permissions.",
                      _obj({"op": {"type": "string", "enum": list(DESIGN_OPS)}, "name": S, "description": S, "procedure": S,
                            "params": LIST}, ["op", "name"])),
}

def tool_schemas(k: Kernel, aid: str, custom: bool = True) -> list[dict]:
    """Provider-neutral schemas: {name, description, parameters}."""
    out = []
    for name, (desc, params) in TOOLS.items():
        if name not in k.allowed or (not custom and name == "design_action"):
            continue
        if name == "message" and not getattr(k, "messages_on", False):
            continue
        ops = k.allowed[name]
        if ops is not None and "op" in params["properties"]:
            params = {**params, "properties": {**params["properties"],
                                               "op": {"type": "string", "enum": [o for o in params["properties"]["op"]["enum"] if o in ops]}}}
        if name == "write" and k.custom_kinds:  # kinds agents defined
            params = {**params, "properties": {**params["properties"], "kind": {"type": "string", "enum": list(k.kinds_allowed())}}}
        if name == "share" and getattr(k, "islands", None):  # islands: publishing an island's entry to everyone now
            desc = desc + (f" With islands: entry=E.. publishes an entry by you or your island to everyone now "
                           f"({k.islands['quota']} per {k.islands['window']} ticks each).")
            params = {**params, "properties": {**params["properties"], "entry": {"type": "string",
                                                                                   "description": "an E-id to publish"}}}
        if name == "search":
            kinds = list(SEARCH_KINDS) + list(getattr(k.env, "search_kinds", ()))
            params = {**params, "properties": {**params["properties"], "what": {"type": "string", "enum": kinds}}}
        out.append({"name": name, "description": desc, "parameters": params})
    if k.env is not None:
        out.extend(getattr(k.env, "all_tools", k.env.TOOLS))
    if custom and "design_action" in k.allowed:
        for n in sorted(k.agents[aid].actions):
            ca = k.custom.get(n)
            if ca:
                out.append({"name": f"x_{n}", "description": f"[custom v{ca.version} by {ca.author}] {ca.description}",
                            "parameters": _obj({p: S for p in ca.params}, [])})
    return out
