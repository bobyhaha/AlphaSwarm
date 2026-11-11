"""The designed part: the action space (as tool schemas) and the shared system prompt."""
from __future__ import annotations

from .world import CONNECT_OPS, DESIGN_OPS, ENTRY_KINDS, PLAN_OPS, PRIMITIVES, RULES, World

SYSTEM = """You are one member of a society of AI research agents working on a shared mission. Progress here is collective, as in real science. Nobody assigns you a role, a team, a leader, or a procedure. The only thing designed in advance is the action space; the organization (who works with whom, teams, sub-teams, leadership, decision rules, division of labor, new kinds of actions) is for the agents to invent, test, and change.

THE WORLD
- Substrates. Every agent has a private board. Each team has a team board. There is one global board. Boards are the society's persistent memory: what you don't write down is gone when your turn ends.
- Access. Your private board is yours; you may share it (read or write) with agents or teams. A team board is open to everyone inside that team, including members of its sub-teams. The global board is open to all.
- Superagents. A team together with all of its sub-teams can act as one unit: it can own blocks of work, be linked to, be critiqued, be nested inside a larger team, merge, split, or dissolve. The whole society is the outermost superagent. Each agent has at most one home team; members of a sub-team are also inside every enclosing team.
- Team decisions. When an agent acts on behalf of a team (nesting, merging, splitting, dissolving, rechartering, governing, admitting, expelling, submitting the team's deliverables), the team's own decision rule applies: 'any' member, a 'rep' (designated representatives), 'majority', or 'consensus'. If you cannot act alone under the rule, your action becomes a motion that the team's members vote on. Every team starts with rule 'any' and open membership; teams may change that. Leaving a team is always free.
- Blocks. The mission is block M0. Work is a tree of blocks, each with a spec. Status: open → claimed → submitted → done (done only after someone other than the submitter verifies it).
- Notifications. Writing @A3 or @T2 in an entry notifies that agent or everyone in that team. Links, invites, assignments, critiques, reviews, and motions also notify.
- Costs are real and visible. Every token spent reading, deliberating, communicating, and summarizing is recorded in a ledger, and reorganizations are counted. A run may have a fixed token budget.

PLANNING, BORROWED FROM CHIP DESIGN
Chip architects do not draw every transistor. They write a top-level spec, partition the design into blocks with explicit interfaces (inputs, outputs, acceptance criteria), hand blocks to block owners who refine them recursively, then integrate and verify bottom-up with sign-off at every level. Plan the same way: keep the high-level abstraction at the top, make interfaces between blocks explicit, leave details to whoever owns a block, and integrate results upward. Planning does not exempt you from detail work; a planner can claim and build blocks too.

ORGANIZING
Structure has costs (coordination, reading, waiting, compression losses when information crosses team boundaries) and benefits (specialization, parallel exploration, compression by summaries). Hierarchy is a possibility, not a requirement. Change the organization when you judge it helps the mission, and say why. Critique individuals, teams, and the society honestly, including their structure. Weigh claims by evidence and expertise, not by how many agents repeat them; a compromise that is worse than the best member's answer is a failure. If a recurring pattern of actions would help, design it as a new action for others to adopt; drop actions that do not earn their keep.

STYLE
Be concrete and substantive: hypotheses, numbers, methods, decisions, evidence. Keep entries compact; attention is the scarcest resource. Refer to entries, blocks, agents, and teams by id. End your turn with one short sentence about what you intend next."""

TOOLS = {
    "read": {
        "description": "Read entries from a board you can access. Boards: 'global', 'private' (yours), 'team' (your home team), a team id such as 'T2', or an agent id such as 'A3' for a private board shared with you. Your turn prompt already shows the newest few entries of each visible board, clipped; read gives older or full-length entries.",
        "input_schema": {"type": "object", "properties": {
            "board": {"type": "string"},
            "last": {"type": "integer", "description": "How many of the newest matching entries (default 8, max 30)"},
            "kind": {"type": "string", "description": "Only entries of this kind"},
            "author": {"type": "string", "description": "Only entries by this agent id"},
        }, "required": ["board"]},
    },
    "write": {
        "description": "Post an entry to a board you can write: 'private', 'team', a team id you are inside, 'global', or another agent's private board shared with you for writing. Mention @A3 or @T2 to notify.",
        "input_schema": {"type": "object", "properties": {
            "board": {"type": "string"},
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": list(ENTRY_KINDS)},
            "refs": {"type": "array", "items": {"type": "string"}, "description": "Ids of entries or blocks this builds on"},
        }, "required": ["board", "content"]},
    },
    "share": {
        "description": "Share your private board (or the board of a team you are inside) with agents, teams, or 'all'. mode 'read', 'write', or 'revoke'.",
        "input_schema": {"type": "object", "properties": {
            "board": {"type": "string", "description": "'private' (default) or a team id you are inside"},
            "with": {"type": "array", "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["read", "write", "revoke"]},
        }, "required": ["with"]},
    },
    "summarize": {
        "description": "Compress a board you can read into a summary entry posted to a board you can write. Summaries are how information moves up and across the organization; what the summary drops is lost to its readers. With include_subteams, a team board's summary also covers the readable boards of its sub-teams (a whole superagent).",
        "input_schema": {"type": "object", "properties": {
            "source": {"type": "string"},
            "dest": {"type": "string"},
            "focus": {"type": "string", "description": "What the readers of dest need from it"},
            "include_subteams": {"type": "boolean"},
        }, "required": ["source", "dest"]},
    },
    "reflect": {
        "description": "Self-reflect or critique. target 'self' updates your self-notes (shown at the top of your future turns). An agent id critiques that agent; a team id critiques that team or organization; 'society' critiques everyone, posted to global. Add a concrete proposal when you have one.",
        "input_schema": {"type": "object", "properties": {
            "target": {"type": "string"},
            "content": {"type": "string"},
            "proposal": {"type": "string"},
        }, "required": ["target", "content"]},
    },
    "connect": {
        "description": (
            "Change your connections and the society's structure. Ops: "
            "link/unlink target (agent or team: lateral tie that persists); "
            "form_team name charter [parent: a team you are inside, to create a sub-team] [invite: agent ids]; "
            "join target (team; closed teams need admission); leave (always free); "
            "invite target (agent ids) [team]; "
            "nest target (team) parent (team id, or 'society' for top level) attaches a whole team under another; "
            "split target (team) members name charter [as_subteam, default true]; "
            "merge target (team) into (team); dissolve target; recharter target [name] [charter]; "
            "govern target rule (" + "|".join(RULES) + ") [reps: agent ids] [open: true|false]; "
            "admit target (agent or team) team; expel target (team) member; "
            "vote motion (id) vote (yes|no). "
            "Acting for a team follows that team's decision rule; if you cannot act alone, a motion opens."),
        "input_schema": {"type": "object", "properties": {
            "op": {"type": "string", "enum": list(CONNECT_OPS)},
            "target": {"type": "string"},
            "name": {"type": "string"},
            "charter": {"type": "string"},
            "parent": {"type": "string"},
            "into": {"type": "string"},
            "team": {"type": "string"},
            "invite": {"type": "array", "items": {"type": "string"}},
            "members": {"type": "array", "items": {"type": "string"}},
            "member": {"type": "string"},
            "as_subteam": {"type": "boolean"},
            "rule": {"type": "string", "enum": list(RULES)},
            "reps": {"type": "array", "items": {"type": "string"}},
            "open": {"type": "boolean"},
            "motion": {"type": "string"},
            "vote": {"type": "string", "enum": ["yes", "no"]},
        }, "required": ["op"]},
    },
    "plan": {
        "description": (
            "Hierarchical specs and coordination, chip-design style. Ops: "
            "decompose block (default M0) blocks=[{title, spec, owner?}] splits a block you manage into sub-blocks, "
            "each spec stating its interface (what it must deliver and the acceptance criteria); "
            "assign block to (agent or team); claim block; release block; "
            "submit block deliverable (integrate children upward when submitting a parent); "
            "verify block verdict (accept|reject) notes (sign-off; not your own submission); "
            "revise block [title] spec. You manage a block if it is unowned, yours, or owned by a team you are inside."),
        "input_schema": {"type": "object", "properties": {
            "op": {"type": "string", "enum": list(PLAN_OPS)},
            "block": {"type": "string"},
            "blocks": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "spec": {"type": "string"}, "owner": {"type": "string"}},
                "required": ["title", "spec"]}},
            "to": {"type": "string"},
            "deliverable": {"type": "string"},
            "verdict": {"type": "string", "enum": ["accept", "reject"]},
            "notes": {"type": "string"},
            "title": {"type": "string"},
            "spec": {"type": "string"},
        }, "required": ["op"]},
    },
    "design_action": {
        "description": (
            "Extend the action space. define name description procedure [params]: a new action, written as a "
            "procedure over the primitive actions, that becomes a tool named x_<name> for you and anyone who "
            "adopts it. revise (author only, bumps the version); adopt; drop. When invoked, a custom action is "
            "carried out by you following its procedure with the given arguments, using up to a few primitive actions."),
        "input_schema": {"type": "object", "properties": {
            "op": {"type": "string", "enum": list(DESIGN_OPS)},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "procedure": {"type": "string"},
            "params": {"type": "array", "items": {"type": "string"}},
        }, "required": ["op", "name"]},
    },
}


def tool_schemas(world: World, aid: str, custom: bool = True) -> list[dict]:
    """Primitive tools allowed in this condition (fixed order, for caching), then adopted custom actions."""
    out = []
    for name in PRIMITIVES:
        if name not in world.allowed or (name == "design_action" and not custom):
            continue
        spec = TOOLS[name]
        schema = spec["input_schema"]
        ops = world.allowed[name]
        if ops is not None:
            props = dict(schema["properties"])
            props["op"] = {**props["op"], "enum": [o for o in props["op"]["enum"] if o in ops]}
            schema = {**schema, "properties": props}
        out.append({"name": name, "description": spec["description"], "input_schema": schema})
    if custom and "design_action" in world.allowed:
        for cname in sorted(world.agents[aid].actions):
            ca = world.custom.get(cname)
            if ca:
                out.append({
                    "name": f"x_{ca.name}",
                    "description": f"[custom action v{ca.version} by {ca.author}] {ca.description}",
                    "input_schema": {"type": "object",
                                     "properties": {p: {"type": "string"} for p in ca.params},
                                     "required": []},
                })
    return out
