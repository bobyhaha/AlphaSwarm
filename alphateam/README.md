# AlphaTeam: a self-organizing agent society

**Design the primitives of organization, not the organization.** Agents get a small set of composable primitives for information, coordination and adaptation. Nothing assigns roles, teams, stages, leaders, workflows or decision rules. Groups, groups of groups (superagents), protocols, division of labor and new actions are for the agents to build, use and abandon.

The question is not only whether agents form a hierarchy. It is whether they **discover when hierarchy is useful, build it, and drop it when it stops helping**. Three claims can be tested separately:

| Claim | What would show it here |
|---|---|
| Emergent organization | Groups, delegates and division of labor appear without being prescribed, and the organization mirrors the hidden structure of the task. |
| Adaptive organization | After a shock (laws change, agents leave), the structure changes and the score recovers. Societies facing a flat world stay flat; those facing a nested world nest. |
| Scalable organizational learning | Score per token improves with more agents, more compute, or more episodes of inherited culture. |

## What changed from v1 (`society/`)

| v1 | AlphaTeam |
|---|---|
| Teams form a forest: one home team per agent | Groups form a DAG. Agents can belong to several groups, and a group can join another group, so superagents nest at any depth. |
| Four decision rules plus motions and votes, built into the kernel | Each group writes its own protocol and names its delegates. Voting, if wanted, is something agents do by talking. |
| Chip-style block tree | Tasks with parents (`delegate`): post, take, deliver, and review by someone other than the taker. |
| Every agent saw the newest posts everywhere, silently truncated | A bounded local view plus `search`. Reading spends a per-tick attention budget, so communication overhead is real. |
| An open-ended mission with no score | A science world with a hidden answer and a checkable score. |
| Single runs | Episodes that carry culture forward, plus optional selection across a population of societies. |
| Claude only | OpenRouter (DeepSeek V4.1 by default), DeepSeek's own API, Claude, or an offline mock. |

## The kernel ([kernel.py](kernel.py))

| Family | Primitive | What it does |
|---|---|---|
| Information | `read` | Read a board you can access; spends attention. |
| | `search` | Search agents, groups, posts, tasks, actions, or laws. |
| | `write` | Post to a board; `@A12` or `@G3` notifies that agent or group. |
| | `share` | Share your private board, or a group's memory board. |
| | `summarize` | Compress boards into a summary posted elsewhere. |
| | `critique` | Critique an agent, a group, or the society. |
| Coordination | `connect` | Follow or unfollow; form, join, leave or dissolve a group; invite, admit or remove members. `as=G` acts on behalf of a group, which is how groups nest. |
| | `link` | A private two-way channel between two agents. Agents at the same level of a group can message each other directly; any other pair must both agree before messages flow. |
| | `delegate` | Tasks: post, take, deliver, review, drop. Tasks can decompose recursively. |
| Adaptation | `reflect` | Revise your own strategy notes. |
| | `protocol` | A group rewrites its own rules: protocol text, delegates, open or closed, purpose. |
| | `design_action` | Define, revise, adopt or drop new actions, which are procedures over the primitives. |
| World | `experiment`, `analyze`, `publish` | See below. |

The kernel enforces only what keeps actions well defined:
- permissions
- no cycles in nesting
- budgets for actions per tick, attention per tick and experiments per episode
- an audit log of every action
- a token ledger

### Boards
- Every agent has a private board.
- Every group has a memory board (`mem:G`, open to everyone inside it) and a public board (`pub:G`, its interface to everyone else).
- The global board is open to all.
- The library is open to all and is passed on to future societies.

### Links and levels
- **Same level:** a group's level holds its direct agent members plus whoever may act for each of its member groups (their delegates, or everyone inside if a group names none). Agents at the same level can message each other directly; joining a group counts as agreeing to talk.
- **Everyone else:** both agents must agree. One asks with `link op=open`, and the link opens when the other opens it too.
- **Why it matters for hierarchy:** a sub-team that names a delegate gets that delegate a direct line to the parent group's level, while the rest of the sub-team talks through it.
- **Where messages go:** to the other agent's inbox, and to a board only the two can read (`link:A12`).

### What each agent sees each turn
- its groups
- its inbox
- what it follows
- headers of new posts
- the lab status for its own outputs

Anything more costs attention.

The agent's closing text each turn is saved to its private board as a note to itself. The system prompt ([primitives.py](primitives.py)) describes what the primitives do and what they cost, and gives no advice on how to organize.

## The world: hidden laws ([hidden_laws.py](hidden_laws.py))

**The system.** An unknown system has inputs x1..xN (integers 0-10) and outputs y1..yM. Each output is a fixed formula of a few inputs: linear, quadratic, hinge and interaction terms, plus noise. Its structure has three levels:
- outputs in a module share inputs
- modules in the same super-module are coupled
- couplings across super-modules are rare

**Composite outputs** (in the default `hierarchical` world) sit on top of the base outputs:
- **Integration outputs** are formulas of base outputs from several modules of one super-module.
- **Top outputs** are formulas of integration outputs from several super-modules.

A law may name another output, such as `2*y12 - y40`, which then stands for the society's current law for it. So a composite's law is only as good as the laws it builds on, and it improves when they do, much like integrating blocks in chip design. This gives hierarchy a reason to exist without prescribing it: integrators depend on module teams. Laws that would refer to each other in a circle are rejected.

Input and output numbers are shuffled, so naming leaks nothing. Sizes grow with the society: 1000 agents face 310 outputs (273 base, 34 integration, 3 top), 276 inputs, and 69 modules in 9 super-modules. With `--structure modular` or `flat` there are no composites, which makes those worlds the controls.

**What agents can do.**
- **Measurement is limited.** Each agent measures only a few outputs (its instruments) and has a small experiment budget, so nobody can cover the system alone.
- **Private data.** Private boards start seeded with observations only that agent has.
- **Data follows readability.** Data lives on boards. `analyze` (correlation screen, least-squares fit with cross-validated R²) uses all data on boards the agent can read, so sharing and where results are posted matter.
- **One law per output.** `publish` sets the society's current law for an output, and the latest one wins.

**Feedback.**
- Every few ticks the lab validates every published law on separate conditions and posts the results publicly.
- Authors' track records are searchable, which gives the society a way to recognize expertise.

**Scoring.** The final score is mean test R² over all outputs on a hidden test set. Outputs without a law score 0.

**Shocks** (`--shock 10:shift`, `--shock 12:churn:0.3`):
- `shift` rewrites one super-module's laws without announcement.
- `churn` removes a fraction of the agents.

**Formulas are parsed safely.** They go through an AST whitelist of numbers, inputs, `+ - * / **` and a few functions. Nothing is passed to `eval`.

## Measurements ([metrics.py](metrics.py))

All of these are hidden from the agents:
- **Science:** test score, coverage, structure F1 (did the published laws find the true inputs?), and last validation.
- **Organization:** groups, groups with members, supergroups, nesting depth, largest group, share of agents in a group, memberships per agent, groups formed and dissolved, reorganizations, and protocol changes.
- **Mirroring:**
  - Each agent's focus is the module it recently screened, fitted, published or varied.
  - The index is P(share a group | same module) − P(share a group | different module).
  - 1 means groups follow the hidden modules exactly; 0 means they ignore them.
  - Level 1 uses each agent's smallest group against modules. Level 2 uses its outermost superagent against super-modules.
- **Hierarchy alignment:** for agents working on composite outputs, P(they share a group with someone working on the composite's parts) − P(they share a group with someone working on unrelated modules). A positive value means the organization's hierarchy follows the task's: integrators sit with what they integrate. The score is also reported separately for base and composite outputs.
- **Links:** open links and messages per tick.
- **Expertise:** the correlation between an agent's validated track record and the standing others give it (followers plus delegate roles). Is expertise recognized, or held back?
- **Communication:** tokens per tick by category (deliberation, reading, writing, summarizing, critique, custom actions).

## Experiments

| Run | Question |
|---|---|
| `--condition self_org` vs `fixed` vs `oracle` | Does self-organization beat a prescribed org chart? How close does it get to one designed with knowledge of the hidden modules? |
| `--condition flat` | Is structure worth its coordination cost? |
| `--condition single` (same total measurement budget) | Does a society beat one agent with every instrument? |
| `--condition no_design` | What does an evolving action space add? |
| `--agents 10,30,100,300,1000` (separate runs) | Does depth grow with size, as communication overhead predicts (roughly the log of the number of agents)? Does score per token scale? |
| `--structure hierarchical` vs `modular` vs `flat` | Do agents build hierarchy only when the task has nested structure? |
| `--shock T:shift`, `--shock T:churn:0.3` | Does the organization adapt and the score recover? |
| `--episodes 5 --structure hierarchical,flat` | Does inherited culture help, and does the society learn when to drop hierarchy? |
| `--episodes 5 --population 4` | Internal adaptation compared with external selection of the best society's culture. |

## Running

```bash
uv venv && uv pip install -r requirements.txt
```

Offline, with scripted agents. The structure is scripted, not emergent, so use this only to test plumbing and the viewer:

```bash
.venv/bin/python run_alphateam.py --backend mock --agents 1000 --ticks 12
```

Real agents through OpenRouter. The key is read from the environment and never written to disk:

```bash
export OPENROUTER_API_KEY=...
```

```bash
.venv/bin/python run_alphateam.py --backend openrouter --agents 30 --ticks 12
```

```bash
.venv/bin/python run_alphateam.py --backend openrouter --agents 1000 --ticks 20 --shock 12:shift --concurrency 256
```

**Cost.** A 10-agent test with `deepseek/deepseek-v4.1-flash` measured about $0.0013 per agent-turn. So 1000 agents × 20 ticks is roughly $26, and probably more, since observations grow with the society. OpenRouter reports the actual cost per call, and the run prints it every tick and saves it. Runs over 200 agent-turns ask before starting unless you pass `--yes`. `--cost-budget 100` stops a society once it has spent $100, keeping its results.

**Other backends:** `--backend deepseek` (`DEEPSEEK_API_KEY`, model `deepseek-flash`) or `--backend claude` (`ANTHROPIC_API_KEY`). `--reasoning` turns on model reasoning for OpenRouter and DeepSeek.

**Outputs.** Each run writes `runs_colony/<stamp>-<condition>-<backend>-<n>a/`:

| File | Contents |
|---|---|
| `viewer.html` | Replay: nested groups colored by the super-module each agent works on, charts over time, inspector, events, laws against the hidden truth. Opens in a browser. |
| `report.md` | Summary, metrics by tick, final organization, laws, retrospective. |
| `run.json` | State and snapshots, including the culture passed to the next episode. |
| `events.jsonl` | Every action. |
| `entries.jsonl` | Every post. |

**Multi-episode runs** also write `lineage.md`. `--viewer RUN_DIR` re-renders a run's viewer with the current template.

## Evolving the rules (recipes, 2026-11-09)

A recipe is the rules a society starts from: rights, information, incentives, procedures and advice. Recipes evolve across runs, and a revision is kept only when societies under it beat societies under its parent on the same task instances with the same budget. The design, what was taken from other systems and the check against GOAL.md are in [DESIGN.md](../DESIGN.md) section 20. So far the design world has a registry of 17 settings ([recipe.py](recipe.py)).

| File | What it holds |
|---|---|
| [recipe.py](recipe.py) | typed recipes and edits, the rules a recipe may set, the `rules` action agents use to propose changes and leave lessons |
| [archive.py](archive.py) | every recipe with its parent, runs, pairs, decisions and lessons, on disk; one archive belongs to one niche and frame |
| [paired.py](paired.py) | frames and arms (open, chaired, hierarchy, single, alone), paired runs, copying a running society, branches, the spending cap |
| [selection.py](selection.py) | the ladder a revision climbs: typed, dry run, probe, screen, confirm; the random-keep control |
| [code_runs.py](code_runs.py) | recipes on the code world (xsv): each run a process of its own, a stage's runs side by side, the hidden tests recorded and never read by selection |
| [stats.py](stats.py) | the lower bound of a mean, pairs needed, how far two mixes of actions differ |
| [generations.py](generations.py) | the generation loop, the menu-only control and the report on held-out instances |
| [trial.py](trial.py) | a society tries its own proposal on a copy of itself |

Everything below is offline with scripted agents and costs nothing. Scripted agents ignore most rules and never propose, so these show the machinery only.

```
.venv/bin/python tools/recipe_evolve.py --generations 3 --out runs_recipes/demo      # generations of recipes
.venv/bin/python tools/recipe_evolve.py --control --out runs_recipes/demo-control   # the same generations, a random revision kept
.venv/bin/python tools/recipe_evolve.py --menu-only --out runs_recipes/demo-menu    # the same ladder, no candidate from agents
.venv/bin/python tools/scaling_grid.py --control --pairs 20 --out runs_recipes/control   # the holder rule on and off, paired
.venv/bin/python tools/scaling_grid.py --plan --out runs_recipes/grid                # a bigger task for a bigger team, and its cost
.venv/bin/python tools/scaling_grid.py --fixed-blocks 16 --per-tick 192 --sizes 16 32 64 --plan --out runs_recipes/speed   # the same task, more agents
.venv/bin/python tools/selection_power.py                                            # spread of saved runs; what the ladder keeps
.venv/bin/python tools/code_evolve.py --fake --agents 4 --ticks 4 --generations 2 --out runs_recipes/rehearsal   # the xsv loop, rehearsed offline
.venv/bin/python tools/code_evolve.py --stage status --out runs_recipes/XSV1         # what an xsv archive holds
.venv/bin/python tools/code_evolve.py --fake --agents 4 --ticks 4 --screen-pairs 1 --confirm-pairs 0 --out runs_recipes/rehearsal1   # a first pass on single runs: the best candidate goes on as a lead
.venv/bin/python tools/keyed_runner.py                                               # holds the key in memory and starts queued paid stages (--key-pipe PATH takes the key once through a named pipe)
.venv/bin/python tools/record.py build --from runs_recipes/XSV1 runs_recipes/XSV2    # the record a proposer reads: every run's files, an index, timelines (DESIGN.md section 22)
.venv/bin/python tools/record.py show RUN; tools/record.py events RUN --round 6; tools/record.py trace RUN --agent A5 --kind reply   # a run round by round, its actions, its full trace
.venv/bin/python tools/code_rounds.py --out runs_recipes/XSV2 --stage table          # candidates judged on hidden tests (stages add, run, table, voices)
.venv/bin/python tools/regrade.py --out runs_recipes/XSV1 --write                    # grade saved programs again on the hidden tests (counts only)
```

A paid run needs `--backend openrouter --live`, the key in the environment and room in `spend.json`; the runner stops at its cap, and a run that stops early is not scored. Nothing here has run with models yet. The tests are scripts: `.venv/bin/python tests/test_recipe.py` (also `test_paired.py`, `test_evolution.py`, `test_live_path.py`).

## First results (not evidence yet)

**Live run.** A live 10-agent society on DeepSeek V4.1 Flash through OpenRouter, run for 6 ticks:
- It reached a test score of 0.58, with laws for 73% of outputs.
- It cost $0.08.
- It stayed flat, with no groups. At that size, this is what the communication-overhead argument predicts.

**1000-agent live baseline.** This run used DeepSeek V4.1 Flash through OpenRouter, ran for 20 ticks with a law shift at tick 12, and used the world before links and composite outputs were added.
- **Result:** a hidden-test score of 0.746, with laws for every output. It cost $25.38 for 70k model calls and took 19 minutes.
- **Organization:** it became a flat open data commons, not a hierarchy. About 40% of experiment results were posted to the global board, and nearly every fit drew on data from several boards. Only 7 small groups formed, holding 3% of agents, with no nesting.
- **Coordination failure:** latest-wins publishing caused churn. There were 5,910 publications for 273 outputs, and one output was overwritten 53 times. The agents diagnosed this themselves in their retrospective but built no structure to stop it.
- **Shock:** after the shift the score fell from 0.724 to 0.697, then recovered to 0.746, without any reorganization.
- **Interpretation:** that world gave hierarchy no job to do. Outputs decompose into independent problems, and a free global commons was enough to share data. Composite outputs, links with consent, and group track records were added in response.

**Mock run.** A 1000-agent mock run takes about 1 second of kernel time per tick, so the kernel will not be the bottleneck. Its structure is scripted.

The open questions are all in the experiments table, and none has been run at scale yet.

## Caveats

- **The world is designed by hand.** Difficulty, noise, module sizes and budgets are knobs that change what organization pays off. Report results across settings.
- **Selection is an outside optimizer.** `--population` selects on the hidden test score. The experience mechanism (`--episodes`) is the one where only the agents decide what carries forward.
- **One model, one society.** Every agent is the same model, so agents differ only in what they know: their instruments, data and history. A single run is one sample; compare conditions over several seeds.
- **Provider routing varies.** OpenRouter may route requests to different providers, which affects cost and possibly behavior.
