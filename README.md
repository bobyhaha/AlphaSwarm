# Self-organizing agent society

> **The current system is AlphaTeam.** See [alphateam/README.md](alphateam/README.md) and run it with `run_alphateam.py`. AlphaTeam adds nested, overlapping groups that write their own rules, a science world with a hidden answer, attention budgets, organizational learning across episodes, OpenRouter/DeepSeek backends, and runs of up to 1000 agents. This page describes the first version (`society/`, `run.py`), kept for comparison.

A testbed for one hypothesis:

> Given a sufficiently expressive set of primitive actions, a population of autonomous agents can self-organize into adaptive, recursively compositional structures (teams, sub-teams, superagents) that outperform fixed organizations on complex, long-horizon tasks.

Only the **action space** and a small **kernel** are designed. Who works with whom, whether teams form, how deep the hierarchy goes, how a team decides, who leads, and which new actions exist are all left to the agents. This follows the bitter-lesson argument: provide general mechanisms and compute, and avoid building in assumptions about what a good organization looks like.

## Three layers

| Layer | Designed? | Contents |
|---|---|---|
| Kernel | fixed | substrates (boards) and permissions, the audit log of every action, the token ledger, the rules that make each action well defined |
| Primitive action space | fixed | `read` `write` `share` `summarize` `reflect` `connect` `plan` `design_action` |
| Learned action space | **by agents** | custom actions `x_*` defined, revised, adopted and dropped by agents |
| Organization | **by agents** | teams, nesting, decision rules, leadership, division of labor |

The society at time *t* is G_t = (V_t, E_t, H_t):

- **V**: agents and teams.
- **E**: the lateral graph of links, board shares, critiques, assignments and invites.
- **H**: containment. Each agent has at most one home team and each team at most one parent, so H is a forest. That makes every subtree a well-defined **superagent**. Lateral structure lives in E, so the society as a whole is a graph, not a tree.

## The action space

| Action | What it does |
|---|---|
| `read` | Read a board you can access: private, team, global, or a board shared with you. |
| `write` | Post to a board. `@A3` or `@T2` notifies that agent or every member of that team. |
| `share` | Share your private board, or a team board you are inside, for read or write; or revoke it. |
| `summarize` | Compress a board (optionally a whole superagent's boards) into a summary on another board. Summaries are how information crosses organizational boundaries, and what they drop is lost. |
| `reflect` | Critique yourself (updates the self-notes shown in every later turn), an agent, a team/organization, or the whole society. |
| `connect` | `link` `unlink` `form_team` `join` `leave` `invite` `nest` `split` `merge` `dissolve` `recharter` `govern` `admit` `expel` `vote` |
| `plan` | `decompose` `assign` `claim` `release` `submit` `verify` `revise` over a tree of blocks |
| `design_action` | `define` `revise` `adopt` `drop` custom actions |

### Superagents
A team together with its sub-teams acts as one unit. It can own blocks, be linked to, be critiqued, be nested inside another team, and merge, split or dissolve. Its outward interface is the blocks it owns and the summaries it posts upward. When an agent acts *on behalf of* a team (nest, merge, split, dissolve, recharter, govern, admit, expel, or submitting a team-owned block), the team's own decision rule applies:

- `any`: any member acts alone
- `rep`: designated representatives act alone
- `majority`
- `consensus`

If the acting agent can't act alone under that rule, the action becomes a **motion** that members vote on, and it executes when it passes. Teams start as `any` with open membership and may change both. Leaving is always free. The kernel enforces whichever rule a team has adopted; it does not prescribe one.

Leadership is **measured, not assigned**. A team's de facto lead is the member with the most coordination acts in that team (decomposing, assigning, verifying, summarizing upward, inviting, restructuring).

### Planning, borrowed from chip design
The mission is block `M0`. Agents decompose blocks into sub-blocks whose specs state interfaces and acceptance criteria. Blocks go to agents or whole teams, results are integrated upward by submitting the parent, and every block needs sign-off by someone other than the submitter. Planners keep the high-level abstraction and can still claim and build blocks themselves.

### Self-designed actions
`design_action define` registers a named procedure over the primitives, which then appears as a tool `x_<name>` for its author and for anyone who adopts it. When invoked, the agent carries out the procedure with up to 4 primitive actions, each logged with `via`. Custom actions cannot call other custom actions and cannot bypass the kernel; they are compositions, not new code. Adoption counts and use counts show which actions survive.

### Costs are visible
Every agent sees the society's token spend (deliberation, reading, communication, summarizing, critique, custom actions) and the number of reorganizations. `--token-budget` stops the run at a fixed number of tokens, so conditions can be compared at matched compute. Agents get no individual reward; the objective is the shared mission.

## Running

```bash
uv venv && uv pip install -r requirements.txt
```

Offline plumbing test. The mock backend makes scripted random choices, so its structure is not emergent:

```bash
.venv/bin/python run.py --backend mock --ticks 20
```

Claude agents. This needs `ANTHROPIC_API_KEY`, or `ant auth login`. The default is `claude-opus-5-5` at effort `medium`, with server-side refusal fallbacks enabled (`fallbacks: "default"`):

```bash
.venv/bin/python run.py --agents 6 --ticks 6 --token-budget 2000000
```

Start small: that first run shows the cost profile before you scale up. Useful flags:

| Flag | Meaning |
|---|---|
| `--condition` | `self_org` (default), `no_design`, `flat`, `fixed`, `single` |
| `--token-budget N` | stop when the society has used N tokens; use the same N across conditions |
| `--blank-personas` | identical agents with no specialty: does specialization emerge anyway? |
| `--inherit runs/X/run.json` | start with a previous society's custom actions and its organizational retrospective |
| `--actions-per-turn`, `--effort`, `--model`, `--concurrency`, `--seed` | the usual knobs |
| `--scenario` | a JSON file with `mission` and `people` (see `scenarios/`) |

Each run writes `runs/<timestamp>-<condition>-<backend>/`:

- `viewer.html`: a self-contained replay. It shows the nested superagent map over time, ties and this tick's interactions, an inspector for any team or agent, the event feed, the block tree with deliverables, the action space, and metrics. Open it in a browser (it loads d3 from cdnjs).
- `report.md`: the final organization, the block tree with deliverables and reviews, custom actions, the retrospective, and metrics by tick.
- `run.json`: everything (events, entries, snapshots, blocks, motions, ledger) for analysis.

`viewer/template.html` opened directly lets you load any `run.json`.

## Experiments this supports

| Condition | Question |
|---|---|
| `single` with matched `--token-budget` | Does a society beat one strong agent at equal compute? |
| `flat` | Is structure worth its coordination cost? |
| `fixed` (a director plus groups with representatives, structure frozen) | Does self-organization beat a prescribed hierarchy? |
| `self_org` vs `no_design` | What does an evolving action space add? |
| `--blank-personas` | Does specialization emerge without seeded expertise? |
| `--inherit` chains | Do organizational lessons and actions transfer across missions (a first step toward learning to organize)? |

Behaviors to look for, unprompted: independent exploration then selective synchronization, temporary teams that dissolve when done, verification groups others consult, splits when coordination overhead grows, merges when work becomes interdependent, and teams changing their decision rule.

## Layout

```
run.py                     CLI
society/world.py           kernel: boards, structure, governance, blocks, ledger, observation
society/actions.py         action space (tool schemas) and the shared system prompt
society/claude_backend.py  Claude agents: per-turn tool loop, summarizer, custom-action executor, retrospective
society/mock_backend.py    offline scripted backend for testing
society/runner.py          conditions, tick loop, outputs, report
viewer/template.html       replay viewer
scenarios/                 missions and populations
tests/test_kernel.py       kernel tests plus a stubbed Claude tool loop
```

## Known gaps

- **No quality evaluator yet.** Comparing conditions needs a task score: a rubric-based judge, or a task with a ground truth.
- **Containment is a forest.** An agent cannot belong to two teams at once (matrix organizations); cross-team work goes through links and shares.
- **An LLM choosing its structure is not learning the optimal structure.** Retrospectives and `--inherit` are the first hooks for organizational learning; reinforcement learning over organizational actions would be the stronger version.
- **Agents act in synchronous ticks.** Within a tick they act concurrently, and each agent's observation is taken at the start of its turn.
