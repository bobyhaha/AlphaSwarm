# Exploration, credit and an evolving substrate (ARC world)

Written 2026-10-31. Code: `alphateam/explore.py`, `alphateam/rule_learning.py`, the kernel hooks, and tests in `tests/test_incentives.py`. Read with [GOAL.md](GOAL.md).

## 1. What agents can do

Every action is offered; none is required.

| Action | What it does |
|---|---|
| `play`, `look`, `replay` | Play your own copy of a game; `replay` reproduces a published solution |
| `focus` | Switch to another game |
| `plan` | `claim` a hypothesis on a problem (shown to everyone on it while open); `done` posts a finding with outcome works, dead end or unclear, plus the receipts that show it |
| `substrate` | `define_kind`: a new kind of entry anyone can write; `propose`: a change to the rules |
| `lead` | Found a team, join or leave one, assign a role and job to a direct child, report up, name a tracker, pin the team note, step down. Sub-teams nest |
| `contract` | Pay credit for a job: the pay is held when accepted and released when marked done |
| kernel primitives | Boards, groups, links, tasks, search, and `design_action` (new actions over all of the above, game tools included) |

### Inspired by Agensh's worker loop (adapted, not copied)

What carries the idea is how a plan lives on the substrate:
- **The claim** states a hypothesis and the result that would show it.
- **Notices.** Others with open plans on the problem are told of a new claim, so overlaps can be settled by direct messages.
- **Findings** along the way (`plan op=finding`) and the result (`plan op=done`: works, dead end or unclear) cite receipts.
- **Frontier notices.** Everyone with an open plan on a level is told when someone completes it.
- **The untried view** shows what no copy has tried at the frontier.

A short working cycle, in AlphaTeam's terms, is a view in the rule set (world option `cycle=1` shows it from the start). The rule loop can hide it or show it. It is offered, never required, and the score judges it.

**Royalties.** Shares are paid out of the earner's credit, never minted:
- a lead receives `lead_share` of what its children earn;
- an invented action's author receives `action_share` of what others earn with it;
- the earner keeps at least a tenth.

## 2. What the engine owns, and what can evolve

**Fixed, never evolved:**
- **The score:** levels completed out of levels to win, averaged over games.
- **The facts credit is computed from:**
  - level completions;
  - frame states reached, with cells that change on most steps masked;
  - which entries each agent read in full;
  - receipts, the engine's record of each play.

**The rule set, evolvable within bounds:**

| Group | Rules | Defaults |
|---|---|---|
| Credit | first completion of a level | 5 |
| | each unseen frontier state | 0.2, up to 2 per tick |
| | finding backed by receipts | 0.3 |
| | dead end backed by receipts | 0.3 |
| | each author the first solver read | 1 |
| | lead's share of its children's credit | 0.3 |
| | an invented action's author, of what others earn with it | 0.2 |
| | steps bought per credit point earned in the last two ticks | 2 |
| Substrate | how long plans stay open | 3 ticks |
| | which views each turn shows: problems, plans, findings, dead ends, what paid, unused actions, promoted kinds | all but promoted kinds |
| | items per view | 4 |
| | search ranking: weights on citations, author credit and recency | 0.25, 0, 0 |
| | which agent-defined kinds are promoted to every turn | none |

## 3. The loop that evolves them

**Within a run** (`--rule-learning W`). Every W ticks a reviser reads the evidence and returns a bounded rule set. The harness enforces it and posts it as RULES vN. The evidence is:
- the agents' proposals;
- which actions, combinations and kinds of entry have paid;
- what agents used, never used and reused;
- the progress in the window after each earlier revision.

There is no revert within a run: a run has only a few first completions, too few to judge a window. The reviser sees each earlier change's aftermath as evidence instead.

**Across runs.** A run exports its final rules, and the next run can start from them (world option `rules=<file>`). Runs are compared by the score's area under the curve (mean score over ticks). This rewards reaching levels sooner and is denser than the final score; Agensh reads time-to-threshold for the same reason.

## 4. Lessons taken from other work

**Queen v3:**
- Taken:
  - engine-owned receipts;
  - plans as intentions on a problem, visible to others;
  - findings carry their evidence;
  - a retrieval log gives reuse edges.
- Left out: signatures, version vectors and false-accept calibration. ARC's engine already checks completions.

**Evolving Agent Teams:** exploration of different paths is rewarded, and the organization itself is revised. Here the revisions are offered through credit and rules, not imposed by a supervisor.

**Agensh (arXiv 2609.26781):**
- Taken:
  - a dead-end outcome shown on each problem, the entry type it calls most valuable;
  - claims that steer peers to other angles;
  - judging by time as well as the final score.
- Not taken at first: pushing new entries mid-turn, and staggered activation. Delivery is now taken, adapted, as the `sync` option (section 7).
- Agensh had all workers equal, and roles emerged at 128 or more agents. Here delegation is offered, not required.
**VISTA (Han, Hu, Qiu, Wu, He; arXiv 2610.02200; read 2026-10-31).** One multimodal agent per game completes all 25 public ARC-AGI-3 games: Claude Opus 5.0 at extra-high effort scores 100 RHAE with 57% fewer actions than first-time humans. The same model scores 40.68 under the organizers' harness.
- **The harness, one step at a time** (GPT-5.6 Sol, RHAE):

  | Step | RHAE |
  |---|---|
  | Organizers' harness, text grids | 13.3 |
  | Images instead of text grids | 47.3 |
  | Larger action and time limits | 51.7 |
  | One continuous conversation, compacted when full | 65.8 |
  | Two notes, plus "build a compact, revisable model of the game" | 70.1 |
  | Every frame kept, with inspection of any past frame and region | 94.1 |
  | Exact pixel readout | 99.0 |

  The two notes: GUIDE.md holds durable understanding that carries across levels; WORKING.md is scratch, cleared when a level advances.
- **The prompt asks for prediction before action.** Before each move the agent states what it expects to see; afterwards, every visible change, expected or not.
- **The authors say the models' own capability remains a key factor.**
- **What it teaches AlphaTeam** (adapted, not copied):
  - Our agents start a fresh conversation every tick, the weakness of the organizers' harness that continuity fixed (+14). Sync's recent work (3 notes, 3 plays) is a thin version.
  - A durable guide per game, separate from per-level scratch, is the carry-forward split in another form. For a society it is a shared guide, the assembled picture in language, alongside or before the code model.
  - Every frame kept and inspectable is "preserve work" at the level of evidence. Our receipts keep moves and summaries, not frames; the transitions behind `model=1` keep frames that could be shared.
  - Prediction before each move, checked against the frame, gives the engine a signal of understanding rather than novelty.
  - The fair comparison for a society is a strong single-agent harness with the same model and compute.

- **Conflict resolution** (paper reread 2026-10-31; now part of GOAL.md):

  | Conflict | Agensh | AlphaTeam now | Gap |
  |---|---|---|---|
  | Overlapping scope | Claims are visible. A claim on what you are building is a collision, settled by direct message, which arrives mid-turn; other claims are ignored | A plan claim notifies those with open plans on the same level. Links need a handshake unless agents share a group. With sync, an @mention reaches the agent mid-turn | Overlap is judged only by level |
  | Concurrent changes to shared work | Git detects the conflict. The later merger takes in main, re-checks against the reference, then merges | ARC has no shared work to change. In the design world, knob changes overwrite each other; exclusive rights stopped the thrashing | No merge gate: a change is not required to start from the current state or to keep checks passing |
  | Disputed correctness | A counterexample withdraws an approval; a FAIL invalidates the author's latest patch; the author fixes it and peers review again | A challenge with receipts marks a finding contested until revised | Entries built on a contested finding are not flagged |
  | Who integrates | Integration is handed to a peer. At 1,024 agents, workers ask several integrators, take the first valid answer and cancel the rest. Workers agree on protocols and revise them | Contracts (offer, accept, escrow) and lead assignments | No way to ask several and take the first |
  | Stalled or failed work | Specialists take over after a failure; claims expire | Expired plans are shown as free to take over | Close |

## 5. Faithfulness audit against GOAL.md

| Goal | Implementation | Status |
|---|---|---|
| A set of actions agents can expand | Offered actions above. `design_action` can call game tools. `substrate define_kind` adds entry kinds | Faithful. **Fixed in this audit:** an inventor now earns a share of what others earn with its action, and the ledger now sees custom actions (`k.running_custom`) |
| Explore freely | Nothing is required. `focus` lets agents pick problems | **Fixed:** the brief no longer prescribes "claim before you test"; it describes the options. It still says the society gains most when members explore different things, the encouragement you asked for |
| Find which actions and combinations are useful, with credit as the means | Credit with consequences (steps) is paid to agents. A ledger by action, combination and kind read is shown to all | **Partial.** The ledger is an association (actions within two ticks of an award), not a causal estimate, and it is labeled so. Causal credit would need counterfactual reruns |
| Credit rules are not hand-fixed | The rule set plus the loop above, judged only by the score | Faithful in design, untested in a paid run |
| An emergent society: leads, teams, contracts, deep hierarchies | `lead` with nesting sub-teams; `contract` | **Fixed:** contracts existed only in the design world; now also in ARC |
| A lead like a CEO or PI: delegates jobs and roles, including keeping track, still works on details | assign, report, tracker, pin; leads keep playing | Faithful |
| Emergent, adaptive, scalable | Metrics for teams, roles, kinds, contracts and rule versions | **Gaps.** ARC runs have 16 agents (scalable untested at 50 or more; Agensh saw roles emerge only at 128 or more). Adaptation is only to level changes, with no shocks in ARC |
| Learning loop, not organization; institutions offered | The reviser changes credit and views only, never teams or roles | Faithful. One caveat: the reviser is a single model call. A decentralized alternative is agents voting on proposals |
| Continual learning: reflect, try, keep or revert | Agents keep notes; the society revises rules | **Partial.** Keep-or-revert happens across runs, not within, because of noise |
| Organizing pays when contributions are recognizable and members differ | Receipts and engine credit make contributions recognizable | **Gap.** X runs use one model for all. Mixed capability (`--strong N`) is available but untested on ARC |
| Coordination costs are real | Step, attention and reply budgets | Faithful |

**Resolved: the shares could be gamed when minted.** Wrapper actions, mutual use and passive leads could collect them. The shares are now royalties paid by the earner, so using an action or joining a team lasts only where it pays.

The running X1 and X2 use the version before this audit. They have the prescriptive brief, fixed rules, and no contracts, kinds or dead ends.

## 6. What the runs showed (2026-10-31)

Full results: paper Section 10, Table 6, and RESEARCH_LOG.md.

**Setup.** ARC-AGI-3 with reasoning: four games, 30 ticks.

| Condition | Runs | Mean score | Runs with 4 or more levels |
|---|---|---|---|
| Sharing baseline | 3 | 0.122 | 2 |
| Isolated | 3 | 0.134 | 2 |
| Stack, 16 agents (all versions) | 10 | 0.099 | 0 |
| Stack, 48 agents | 1 | 0.170 | 1 |

The 48-agent run also had the best AUC of any run (0.124).

**What agents used:**
- plans (32-114 per run), backed findings and dead ends, replay;
- one agreed interface (`action_sem`);
- link messages that split overlapping claims, at 48 agents.

**What went unused:**
- `lead`, `contract`, challenges, `focus`;
- agent-written runnable entries; only published solutions were run.

**The rule loop** found that plans and findings were reused by first solvers far more than notes. It raised their credit; agents responded modestly, and the score did not change.

**Bugs found and fixed:**
- the reviser's and organizer's reply limits were too small for reasoning;
- GAME_OVER frames report 0 levels to win.

## 7. Sync: one public substrate (world option `sync=1`, 2026-10-31)

**Why.** Three measurements from the ARC logs, after reading Agensh's code (`microsoft/Agensh`: prompts, shared-context service, mid-turn notice shim):
- **The private board was each agent's only memory.** Every tick starts a fresh conversation. Agents spent 34-55% of their reads re-reading their own notes, and useful knowledge stayed private: in X3b, 558 private notes against 48 public plans and findings.
- **Agents already read the public board.** In the new-design runs, about half of all reads (47-52%) went to the global board. What it held was the problem.
- **Nothing reached an agent mid-turn.** Agents in a tick run concurrently, but inbox messages showed only at the next tick, so a peer's dead end arrived after the steps were spent.

**What it does** (AlphaTeam's terms; the mechanisms are adapted, not copied):
1. **No private boards.** `write board=private` posts to the global board. Group memory boards become readable by all and stay writable by members. Link messages stay between the two agents.
2. **Memory is separate from privacy.** The closing note of each turn is posted publicly. Each turn starts with YOUR RECENT WORK: the agent's last 3 notes plus, in ARC, its last 3 plays with their receipts. Plays are no longer posted as data entries; receipts are the public record.
3. **Push once, most relevant first.** NEW SINCE YOUR LAST TURN lists the first line of what others posted on the global board since the agent's last turn. Each entry is delivered once, posts with the same text (ids and ticks aside) collapse into one line with a count, and the overflow is summarized. Default 8 lines (`sync_items`). Posts that mention the agent come as inbox messages instead.
4. **Relevance is the world's.** ARC ranks:
   - 0 urgent: a dead end on the level the agent's copy is on or the frontier, the published solution of the level its copy is on, a challenge to its finding;
   - 1: claims and findings on those levels;
   - 2: other posts about its game;
   - 3: other games.
5. **Mid-turn interrupts.** After each reply's actions, new inbox messages (link messages, mentions, overlap and solved notices) and new urgent entries are appended to the last tool result, under NEW WHILE YOU WORK.
6. **A smaller default view.** The `paid` and `unused` views are off unless a rules file turns them on. Every action stays offered.

**Code:** `Kernel.end_turn`, `gist`, `_push`, `interrupts`, `recent_work`, and the `sync` branches of `board_id`, `can_read`, `can_write` and `observe`; `ArcWorld.sync_rank`, `recent_work`, `entry_meta` (set by plan, challenge, finding and solution posts); the system prompt's SYNC paragraph; `LLMBackend._loop` appends interrupts. Tests: `test_arc_sync_public_substrate_push_once_and_interrupts` and `test_sync_notices_reach_an_llm_agent_during_its_turn`.

**Offline check (mock, 16 agents, 8 ticks).** No private posts; 130 public notes; 1,271 entries delivered; 267 collapsed as duplicates. The turn-start view grew from 2,781 to 3,619 characters on average (about 210 tokens), against reads and searches that took 57-68% of calls in paid runs. Mock turns run one after another, so mid-turn interrupts appear only in the LLM-loop test.

**Faithfulness to GOAL.md:**
- **Actions:** all still offered; only the private option of `write` and `share` is gone.
- **Diversity from information:** strengthened. Claims and dead ends on an agent's own level now reach it first, including during its turn.
- **Coordination costs:** delivered first lines cost no reading budget, as in Agensh; full reads and searches still do.
- **Credit (open choice):** reuse credit still flows only along full reads (`k.retrieved`). Delivered first lines do not count, so if agents act on first lines alone, reuse credit will understate what they used.

**Planned test** (needs budget; about $2.80 per 16-agent run). Compare the new design with and without `sync`, credit held fixed, 6 seeds each. Primary: runs reaching 4 or more levels, and AUC. Mechanism checks: share of calls spent on reads, overlap between copies at the frontier, and dead ends posted. Novelty credit (85% of all credit in the earlier runs) is a separate test.

## 8. Assembling the picture: carry-forward, explanations, and a shared model (`carry=1`, `model=1`, 2026-10-31)

**Why.** GOAL.md's jigsaw picture: agents made pieces but nothing assembled them, and the table was swept clean whenever a level fell.
- 44 of 60 plans that were open when someone else solved their level closed without a result.
- Findings were shown only for the frontier level.
- 71% of copies reached level 1 by replay. They arrived at level 2 with the position but not the understanding.

**Carry-forward (`carry=1`).**
- **Findings stay in view.** Backed findings from earlier levels of an agent's game appear under CARRIED FORWARD until a receipt at the new level contradicts them.
- **Challenges are scoped by level.** A challenge whose receipts come from a later level retires the finding there ("does not hold at level N") without calling it wrong. A counterexample at the finding's own level contests it as before.
- **Plans move up a level.** Plans open when their level falls go to the next level with their interim findings, and their holders are told.
- **Contests propagate.** A finding can name the findings it builds on (`builds_on`). When one of those is contested, the dependents are flagged and their authors told.

**Explanations with solutions (`carry=1`).**
- **The solution carries the solver's findings.** A published solution entry includes the solver's backed findings on the game, and the solver is asked to explain why the moves work.
- **`plan op=explain` attaches an explanation.** Anyone can add a finding with receipts to a solution, and it earns finding credit.
- **Replays teach.** A replay starts with WHY THESE MOVES WORK and ends with what has carried forward on the game.

**The shared model (`model=1`, `alphateam/game_model.py`, `alphateam/model_sandbox.py`).**
- **One program per game:** `step(grid, action, x, y) -> (new_grid, event)`. Version 0 predicts that nothing changes.
- **Every move is a test case.** Every move any copy makes is recorded as a transition. Repeats from the same frame are kept once and counted.
- **Merge rule:** a change is merged only if all of these hold:
  1. it was written on the current version; otherwise it is refused and the agent takes in the current version and tries again;
  2. it still explains every transition the current version explains;
  3. it explains at least one more.
  - The version is checked again after evaluation, so two agents proposing at once cannot both merge.
- **Credit is paid for predicting moves recorded after the merge.** It goes to moves the new version predicts and its parent did not, at 0.3 per move, up to 3 per merge per tick. Memorizing past moves earns nothing.
- **Agent code runs in a separate process.** The process runs with no imports, no names starting with `_`, no file or reflection functions, a whitelisted numpy, a CPU limit, no file writes and 0.5 s per call.
- **Tool ops:** `model view` shows the code and the transitions it gets wrong; `case` shows the frames before and after one transition; `test` checks without merging.
- **Each turn shows one line of model state:** version, author, and how much it explains overall and at the frontier.
- **A later society can start from these models** (`models=<run.json>`), so it learns from existing results.

**Tests.** Two new tests cover:
- carry-forward, explanations, level-scoped challenges and flagged dependents, on a real lp85 level;
- the merge rules, sandbox refusals and timeouts, credit for predictions, and inherited models.

All 95 tests pass.

**Offline mock (16 agents, 8 ticks, sync + carry + model).** Ran in 6 s. Version 0 explained 66% of recorded moves, 1,325 posts were delivered, and there were 22 mid-turn notices.

**Live smoke test (2 agents, 2 ticks, $0.02).** No tool errors. Agents viewed the model in their second turn, and closing notes now lead with a one-line summary.

**First test (launched ~17:12, $26 left).**
- Six seeds of the V configuration plus `sync=1 carry=1 model=1`; novelty credit is unchanged.
- Settings: 16 agents, reasoning, 4 games, 30 ticks, $4 cap each.
- Compared with V0-V2 and the other new-design runs (0 of 10 past 3 levels) and with the baselines (4 of 6).

## 9. Fixes from the traces (window 3, 2026-10-31 evening)

**What the traces showed** (A_assemble s0-s5, with G_guide_cont partway through):
1. **Turns without a game move.** 56% of agent-ticks made no game move (isolated baseline 50%, sharing 62%: this is long-standing). Agents spent their 2 replies on reads (2.8 per agent-tick) and looks (1.0), and 324 closing notes say "out of replies" with steps unused. With continuity (G runs) the share fell to 35-36% and reads to about 1.5.
2. **Insight lost between ticks.** An lp85 agent worked out level 2 at t24 ("the bands are combination-lock wheels to rotate onto a target"). By t29 it had replaced that with "fill the column-0 odometer", the move budget. Fresh conversations and a 3-note memory dropped the right idea.
3. **Evidence never became evidence.** A16 hit GAME_OVER by filling that budget (t20), but it was posted as a raw play record, not a dead end. No finding said it, so nothing marked it urgent, and A6 went on to fill the bar. In 6 runs, 5 plays ended in GAME_OVER and 2 dead-end findings mentioned one.
4. **Confusing level labels.** Copies showed "level 0/8" (levels completed) while problems said "level 1", and agents misread which level was solved.
5. **The model's exact-frame check was unfair.** vc33 and lp85 draw a move counter on the frame's edge (row 0 or column 0), so version 0 explained 0 of ~400 vc33 moves and every proposal had to model the counter first.
6. **The guide was written once and rarely revised.** Each run wrote one good v1 per game; few later versions appeared, since an edit rewrites the whole text on the current version.

**Fixes, each checked against GOAL.md:**

| Fix | Option | Why | GOAL.md |
|---|---|---|---|
| The last reply of a tick may still make game moves, with its closing note | `final_moves=1` | Half of agent-ticks had no move | Harness, not organization. Reading still has its costs ("coordination has real costs") |
| Copy shown as "on level K+1 of N, K completed"; the brief explains "level K/N" | with `sync` | Agents confused completed and current levels | Legible work |
| Edge lines (rows or columns 0, 1, 62, 63) that change on most of the first 20 moves are ignored when checking the model | `model=1` | Counters made every prediction fail | The fit check of the assembled piece must test the picture, not a counter |
| `guide op=add` appends one line without a base | `guide=1` | Rewrites were costly, so pieces were not added | "Compatible contributions are combined": an append cannot break anyone's lines |
| The engine posts a dead end with the receipt and the moves when a copy reaches GAME_OVER: urgent for that level, carried forward | `carry=1` | Losing conditions stayed in raw data | "Evidence decides"; the engine owns facts; "nothing is lost" |

Tests: `test_trace_fixes_counter_mask_guide_add_last_reply_moves_and_levels` and `test_engine_posts_game_over_as_a_dead_end`.

**Two more, from the lp85 pilot (P_fix_lp85_s0):**

| Fix | Why | GOAL.md |
|---|---|---|
| When the last reply makes moves without text, the note is the engine's summary of what they did | Notes had become "(made moves in my last reply)", losing memory and public knowledge | Make each agent's work legible; nothing is lost |
| The engine posts a dead end when a RESET sends a copy back a level ("RESET at the start of level L restarted the whole game") | One agent lost its level, and two disagreed about RESET; both were right in different states | Evidence decides, stated by the engine, which owns the fact |

Test: `test_engine_warns_when_reset_restarts_the_game`.

**And from the vc33 pilot (P_fix_vc33_s0):**
- **One engine dead end per kind per level.** Repeats are counted and shown as "happened Nx": the same GAME_OVER was posted 6 times.
- **A RESET that would restart the whole game is refused unless `restart_game=true`.** This applies at a level's start, but not after a GAME_OVER. The same loss happened three times, twice to one agent, after the warning had been delivered. GOAL.md: nothing is lost unless an agent means it; every action stays available.

**Pilots** (4 agents on one game, 30 ticks):
- lp85: L1 only; no game move in 26% of agent-ticks; 0 findings.
- vc33: L2 at t23; no game move in 14% of agent-ticks; no L3.

**Results so far (16 agents, 4 games, 30 ticks, reasoning):**

| Condition | Runs | Reached a level 2 | 4+ levels | Mean AUC |
|---|---|---|---|---|
| Earlier design | 10 | 0 | 0 | 0.066 |
| Assemble (sync + carry + model) | 6 | 0 | 3 | 0.078 |
| **+ guide + continuity 2** | 3 | **3** | 3 | **0.099** |
| Baselines (sharing, isolated) | 6 | 2 | 4 | ~0.074 |

Level 2 with the guide and continuity: 3 of 3 against 0 of 6 for assemble (p = 0.012). The traces credit continuity most: each solver kept a line of reasoning across ticks.

## 10. Faithfulness audit against GOAL.md (2026-10-31, 21:30)

The design as tested in F_full: sync + carry + model + guide + continuity 2 + final_moves + the section 9 fixes + reuse credit for pieces in view.

| GOAL.md | Implementation now | Status |
|---|---|---|
| **An expandable action set** | All primitives and world tools offered; `design_action`, `substrate define_kind` | Offered, still rarely used in ARC. Nothing removed: the guarded RESET still runs with `restart_game=true` |
| **Explore freely** | Nothing required; views inform; `focus` switches game | Faithful. The prompt's "say what you expect" is an invitation, not a rule |
| **Credit finds which actions and combinations pay** | Ledger by action and combination. First-solve credit; reuse credit now also for guide lines and findings in the solver's view. Credit for model predictions; engine-owned facts | **Partial.** Novelty credit (0.2) is still on and earlier took 85% of credit. Reuse credit now reaches the shared picture, so writing pays when it helps. Untested at scale |
| **The jigsaw: pieces, edges, one assembled region** | Pieces: findings, engine dead ends, solutions with explanations, model versions, guide lines. Assembled region: the code model (gated by fit) and the guide (language) | **Partial.** The model gates merges on fit, but agents rarely write code (2 merges in 9 runs). The guide checks base versions and needs evidence to remove lines, but nothing checks lines against receipts, and "progress = how much is assembled" is measured only for the model |
| **Learn from existing results, within a run** | Carry-forward; explanations with solutions; replays that teach; the guide shown every turn; continuity | Faithful. Traces show level-2 solvers reasoning over ticks with the guide in view |
| **Learn from existing results, across runs** | `inherit=<run.json>` starts a society from an earlier one's models and guides | **Fixed in this audit** (only models could be inherited); untested live |
| **Preserve and combine work** | Findings and plans carried across levels; engine dead ends; merges gated; nothing removed without evidence; RESET guarded | Faithful in mechanism. The pilots show agents writing less when they act more |
| **Unblocking work first** | Dead ends and solutions on an agent's level delivered first and mid-turn; reuse credit for pieces a solver used | Faithful |
| **Conflicts visible** | Overlap notices; challenges; contested flags on dependents; guide removals notify authors | Faithful |
| **Settled directly and fast** | Mid-turn interrupts deliver messages and urgent entries | Faithful |
| **Evidence decides** | Challenges and guide removals need receipts; the engine states GAME_OVER and RESET facts | Faithful. But delivered evidence is not always acted on: the RESET loss repeated after its warning |
| **A later contributor reconciles** | Model and guide edits name their base and are refused if it changed; guide appends combine | Faithful |
| **Responsibility moves explicitly** | Contracts, lead assignments | Offered, unused on ARC |
| **An emergent society: leads, teams, deep hierarchies** | `lead`, contracts, teams, offered | **Gap.** None formed on ARC. They formed only in the design world with take_lead and individual rewards |
| **Coordination has real costs** | Reading budget; replies per tick; step budget | Faithful. Delivered first lines cost no reading budget, as in Agensh |
| **Diversity from information, not assignment** | Claims, dead ends, untried regions, guide | Faithful. Duplication remains: two copies ran the identical mirrored sequence |

**Gaps to work on next, in GOAL.md's terms:**
1. Check guide lines against evidence, so the assembled picture in language also grows only when pieces fit. For example, each line cites receipts, and a counterexample flags it.
2. Duplication despite visible claims.
3. Delivered evidence not acted on.
4. No emergent organization on ARC.

## 11. What the full-scale test showed, and the audit updated (2026-10-31, 23:00)

Nine 16-agent runs of the section 10 design: F_full (6 seeds) and F_noguide (3 seeds, no guide).

| Condition | Runs | vc33 L2 | 5 levels | Mean AUC |
|---|---|---|---|---|
| A (sync + carry + model) | 6 | 0 | 0 | 0.078 |
| G (A + guide + continuity) | 3 | 3 | 1 | 0.099 |
| F_full (G + section 9 fixes + reuse credit in view) | 6 | 3 | 2 | 0.102 |
| F_noguide (F_full without the guide) | 3 | 1 | 0 | 0.096 |

F_full beats A on AUC (one-sided permutation p = 0.031). These runs cannot separate the guide from continuity (F_full vs F_noguide p = 0.39). No 16-agent run has reached a level 3, or level 2 of lp85, sb26 or ls20.

**What changed and what did not.**
- **Agents now act:** no game move in 6-17% of agent-ticks, against 37-56% before. They make about 8 moves per agent-tick and leave about 14 steps unused, so the step budget no longer binds.
- **The frontier levels need understanding, not moves.** lp85 level 2 got 600-980 moves per run. Agents decoded its controls (pivots rotate rings; a counter ends the game after about 64 moves), but never what a finished level looks like. vc33 level 3 and ls20 level 2 also end the game when probing exhausts a move budget.
- **The assembled picture stayed thin:**
  - Shared model: 0 merges in 9 runs; agents used it as a viewer of recorded frames.
  - Guide: 1-5 lines per game. 31 of 58 attempted lines were refused for length, and 23 were never resent.
  - Closing notes: 82% were engine move summaries.

**Audit rows that changed:**

| GOAL.md | Before | Now |
|---|---|---|
| **Credit finds what pays** | Reuse credit added for pieces in view | **Partial.** 87-93% of credit pays for new states, which rewards burning a level's move budget. Since steps no longer bind, credit has almost no consequence beyond information. Open |
| **The jigsaw: one assembled region** | Code model gated by fit; guide in language | **Gap for the model:** never assembled (0 merges in 9 runs). The guide is the region that grows. Its pieces were refused for length, which is fixed now: lines up to 600 characters and the guide up to 4,000, shown whole |
| **Preserve work** | Nothing removed without evidence | A refused piece is a piece swept off the table: 23 of 58 guide lines were lost that way. Fixed as above |
| **Make each agent's work legible** | Engine summaries when the last reply has no text | The record held moves, not reasoning. The last reply may now record too (guide, plan), and moves carry an optional `note` that becomes the public note |
| **Learn from existing results** | Explanations requested of solvers | About 1 explanation per run. The level-up frames are in the model's cases, but nothing yet shows a newcomer what a finished level looked like. A probe of lp85 L1 showed that the start-to-win change does not reveal the goal there, so this was not built |

**Still open, in GOAL.md's terms:**
1. Credit that rewards understanding rather than coverage.
2. The model as an assembled region. Smaller pieces that each claim only the moves they explain would match the jigsaw picture. They would not lower the barrier for the first merge, though: against version 0, a whole program also only has to leave inert moves unchanged. The likelier barriers are writing exact grid code at all, and that playing pays at once (novelty) while a model pays only on later moves. Untested, so not built.
3. Duplication.
4. Delivered evidence not acted on.
5. No emergent organization on ARC.

## 12. The guide became a log: combining is not removal (2026-11-01, 00:50)

**F2 = F_full plus the section 11 fixes.** 3 runs, 16 agents, $4 cap each. The runs hit the cap at t20-t29, so AUC is compared over the first 20 ticks.

| Condition | Runs | 5 levels (vc33 L2) | AUC@20 | vs F2 (one-sided exact) |
|---|---|---|---|---|
| F2 | 3 | 3 (t11, t14, t9) | **0.120** | |
| F_full | 6 | 3 (t30, t11, t14) | 0.084 | p = 0.024 |
| F_noguide | 3 | 1 | 0.083 | p = 0.05 |
| G | 3 | 3 | 0.079 | p = 0.05 |
| A | 6 | 0 | 0.064 | p = 0.012 |

- **Faster, same plateau.** F2 reaches the same 5-level plateau sooner, and sooner per dollar too: by about $2.10 spent, 3 of 3 F2 runs had it against 1 of 6 F_full runs.
- **No new levels.** No run reached a level 3, or level 2 of lp85, sb26 or ls20. The lp85 pilots (4 agents) also stop at L1.

**What the traces show:**
- **The shared picture now grows.** Each run added 20-33 guide lines, and only 107-137 closing notes were engine move summaries (F_full: 318-378).
- **It grows as a log.** There were 0 edits and 0 challenges. Two guides were full, mostly of level 1 lines, by t7 (s1 lp85) and t12 (s0 vc33). After that, 14 frontier lines were refused, including a vc33 win hypothesis and the vc33 L3 handle maps.
- **Combining was priced as conflict.** Editing meant rewriting the whole guide, and shortening someone else's line counted as removing it, which needs receipts showing it is wrong.
- **Understanding transfers through explanations.** An explanation of vc33 L2 named the win condition. Four agents carried it into L3, mapped the level and corrected each other with receipts. Then the dollar cap ended the run.

**Fix, and the audit rows it changes:**

| GOAL.md | Before | Now |
|---|---|---|
| **Preserve work** | Lines up to 600 characters, the guide up to 4,000 | **No piece is refused for space.** The guide stores 16,000 characters. Past 4,000, the view shortens the oldest lines (view shows them whole), so attention stays bounded and nothing is lost |
| **Combine work (the jigsaw)** | Combining required a whole-guide edit, and shortening others' lines counted as removal | **`guide op=merge`** turns numbered lines into one shorter line. Its authors become co-authors and share reuse credit, the merger included, since combining is work. The old lines stay in the history |
| **Resolve conflicts without an arbiter** | Challenge by receipts, unused; lines unnumbered in the brief | Lines are numbered in the brief, so `challenge line=N` is cheap. A merge needs no receipts, because it claims nothing is wrong. Authors and challengers are told and can add back or challenge anything lost. The engine cannot check that a merge keeps the content, so evidence is the check |
| **Institutions offered, never required** | | Merge is offered, never required. Whether agents use it is untested: they never used edit |

**Still open, with proposals not built (no budget to test them):**
1. **Credit for understanding.** Pay an explanation prequentially when a later level is completed by moves or a plan that cites it, as the model's credit pays for later predictions. The vc33 L2 explanation is the case it would reward. Novelty is still 80-84% of credit.
2. **The code model.** It had 0 merges in all 12 F runs. Agents use only its transition record, as a frame viewer, while its tool description costs tokens every call. An ablation that keeps the record and drops the step-function requirement would test whether it earns its place.
3. **Results read halfway.** On lp85 L2, "fill the bar to win" was tested again after it was refuted in the same run. The re-tester cited the refuter's interim result (bar at row 62, no win), not its end (GAME_OVER when the bar filled). Dead ends are in every brief, so this is not a visibility gap. It is the open item "delivered evidence not acted on", and no fix is proposed.
4. **The dollar horizon.** Richer records cost about 15-75% more per tick, so runs stop before t30. Comparisons should be spend-matched, as here.
5. Duplication, and no emergent organization on ARC (unchanged).

## 13. Ideas, credit for understanding, and lessons: depth from exploring more (2026-11-01)

**Why.** The user's view (2026-11-01): many agents explore different paths, reach different insights and share them, so a society should go deeper than one agent, not only faster. The F2 traces showed four things in the way:
- the same few ideas were tested again and again, one of them after it had been refuted;
- understanding travelled as moves (replays), not as explanations;
- partial findings from different agents were rarely combined;
- credit paid for new states (80-84%), not for ideas or understanding.

**What the F2 traces show** (all 14 run-game pairs at the level where they stuck; 958 frontier entries read, 5,167 frontier moves):
- **Few ideas, mostly echoes.** Each stuck level had 2-7 ideas about what completes it; 76% were voiced by two or more agents. Most echoed the previous level's rule, carried by a guide line or the solution replay. Independent ideas ranged from 0% (two vc33 runs) to 68% (one lp85 run). No win idea was supported, and on lp85 "the win is a target arrangement" was never tested. Correction (2026-11-01, rechecked): three claims did name a specific target (F2_s0 P26 at t15 and P37 at t26; F2_s1 P31 at t19, "sorted/rainbow order"). None was tested in its window: P26's author made one click, P37's author spent the window replaying L1 to get its copy back, and P31's author made two clicks after a RESET. Testing an arrangement takes many rotation clicks; the claims came late (t15-t26 of 29) with 3-tick windows.
- **About 21% of frontier moves re-tested refuted ideas** (about 1,080, an estimate). Re-testers mostly cited nothing or argued around the refutation. In P2_s0, three of the four agents had read the refutation of "filling the bar completes the level" and the four still spent about 178 more moves filling it.
- **Duplicate turns:** 277 moves (5%) repeated an earlier turn by another copy, mostly 12-15-click sweeps of one button.
- **Little combination:** 15 merges of findings from two or more other agents, against about 100 builds on a single agent. Merges that pinned down a mechanism held; merged win predictions failed; none completed a level. Receipt-backed partial findings were left uncombined (lp85 row rotation with wrap, read by three agents and cited by none).

**What is built** (world options `hypotheses=1` and `lessons=1`; both offered, nothing required):

| Piece | What it does | GOAL.md |
|---|---|---|
| IDEAS view | Every claim on the agent's level, with what came of it: testing, never tested, refuted, unclear, worked. Replaces the open and expired plan lists | Diversity from information, not assignment: agents see which ideas nobody has tested |
| Similar ideas pointed out | A claim whose words overlap an earlier one on the level (Jaccard >= 0.5) gets that idea and its outcome in the reply. It is never refused | Conflicts visible; institutions offered, never required |
| Credit for a new idea | The first `plan done` (works or dead end) of an idea on a level, with receipts made after its claim: +0.5. Testing an idea again earns none | Credit finds what pays; evidence decides (receipts after the claim, so an old receipt cannot earn it) |
| Combining pieces | `plan claim builds_on=[E-ids]` names others' findings, explanations or claims. Their authors are told; the parts are recorded | The jigsaw: pieces from different agents joined into one |
| Credit for understanding | When a level falls, each piece its solver cites (`play cites`, E-ids in the play's note, the pieces its plans build on) earns its author +2, and the pieces that one built on +1. Only pieces others posted before that tick; at most 6 per solve | Prequential: paid when a later result confirms the piece, as model credit pays for later predictions |
| Explanations read in replays | Count as used when the reader later completes a level (reuse credit), with no recency window | Learn from existing results; results carry their understanding |
| Repeats of refuted tests named | A play whose moves share a run of 6 with the receipt behind an agent's dead end on that level is told so in its result, with the dead end. Not for replays, nor for receipts that completed a level | Evidence decides, delivered where the moves are made; information, never a block |
| Lessons guide (`lessons=1`) | A guide shared by all games (`guide game=lessons`), shown every turn, ranked for everyone, inherited by a later society | Learn from existing results across games and across runs |

**Limits.**
- The engine judges sameness by word overlap. Rephrasing evades it, and two different ideas that share words can collide. It only decides the idea credit; the reply shows the match so the agent can judge.
- Credit for understanding depends on citations, which agents may not write. E-ids in notes are parsed to lower the cost.
- The IDEAS view adds about 1,500 characters per turn.
- The repeat notice catches exact runs of moves only. Re-testing an idea with different moves (filling a bar by other clicks) is not caught.
- Echoing the previous level's rule is not addressed directly; new-idea credit rewards departing from it.

**Tests.** `test_ideas_are_listed_new_ones_pay_and_cited_pieces_earn_understanding` and `test_a_play_that_repeats_a_refuted_test_is_told_so` (46 incentives tests pass; the other suites pass). Mock: 50 agents on vc33, clean.

**The run** (`tools/launch_vc33_depth.sh`): 50 agents of the full design on vc33 alone, 60 ticks, $45 cap: F2's configuration plus the guide merge (section 12) and both new options. The user asked for this run ($50 added) to see depth; there is no single-agent arm. If it does not get deep enough, the user offered $30 more. **Superseded:** this run was stopped after $0.30, because the IDEAS view then listed 8 rows (testing ideas first), so at 50 agents refuted ideas would never be seen; and a second start was stopped at t1 when the user asked for fewer agents. Replaced by section 14's runs.

## 14. The board as the substrate: everything delivered, search by use, every piece listed (`board=1`, 2026-11-01)

**Why.** The user asked why the board did not show everything, and pointed back to ScaleMAS (Qi et al., arXiv 2610.16120) and Agensh. The store never deleted anything, but each turn an agent received:
- the 8 most relevant new posts: about 30% of what others posted in F2 (16 agents, 23-29 posts a tick), under 10% at 50 agents;
- the last 4 findings and the last 4 dead ends on its level, and (with `hypotheses=1`) 8 rows of IDEAS, testing ideas first. At 50 agents more than 8 ideas are under test at once, so refuted ideas would never have been shown;
- a guide capped at 4,000 characters shown (16,000 stored), the shared text that filled up.

Search was rare and weak: in F2_s0, 43 searches against 683 reads of a named entry, and relevance was the share of query words present, so a query matched 72-115 posts.

**How the two boards work** (paper and code reread 2026-11-01):

| | Agensh (board service) | ScaleMAS | AlphaTeam before | AlphaTeam with `board=1` |
|---|---|---|---|---|
| Store | Append-only; every revision kept | Append-only; never edited | Append-only | Append-only |
| Entries | Typed gists of 100 characters (300 for patch summaries); detail unfolded on request | Sub-task, artifact, rationale, refs; must name each ref it used | Up to 3,000 characters; first line as gist | Same |
| What reaches an agent unasked | Every unread entry, once, grouped by type, duplicates counted; peers' entries also mid-turn | Nothing | 8 lines per turn, ranked | Every unread entry, once, grouped by type, duplicates counted |
| Retrieval | The latest 2,000 entries; boolean keyword search over the whole history | Top 8 by citation weight + BM25 relevance / its best value; each hit shows what it cites and up to 8 citers | Word-share relevance + 0.25 log(1 + citations) | ScaleMAS's ranking, adapted |
| A capped shared text | None | None | The guide | Guide lines are entries; none refused |

**What is built** (world option `board=1`, needs `sync=1`; everything offered, nothing required):

| Piece | What it does | GOAL.md |
|---|---|---|
| Everything delivered once | NEW SINCE YOUR LAST TURN lists every post by others since the agent's last turn, grouped by type: dead ends and challenges on its game, ideas claimed, findings and solutions, the guide and lessons, notes, plays, other games. Same-text posts are one line with a count. Pieces show 160 characters of their first line, notes 120, plays 90 | Diversity from information: what anyone learns reaches everyone; coordination costs stay real (full texts cost reading budget) |
| Use-weighted search | `search what=posts` ranks by citation weight + BM25 relevance divided by its best value, top 8 by default. Each hit shows what it cites and what cites it. The weight moves 0.3 of the way to 1 for each post or play by another agent that cites the entry (refs, builds_on, play cites, or its E-id in the text) and decays by 0.98 a tick | Credit and attention follow what the society uses; nothing ranks by authority |
| Every piece per level | IDEAS lists every claim on the level (now also without `board=1`, as `hypotheses=1` promised); FINDINGS, DEAD ENDS and CARRIED FORWARD list all of theirs, newest first, each with who used it. Past a size budget the later rows are shortened, down to an id and status; none is dropped | Preserve work; evidence stays in view |
| Pieces nobody has used | BACKED FINDINGS NOBODY ELSE HAS USED YET lists receipt-backed findings on the game that no other agent has cited | Combine partial insights (the user: "combine more should help") |
| Guide and lessons lines are entries | Each line shows the E-id of the entry that added it and can be cited. Adding is never refused for space. Past 4,000 characters the least-used lines are shortened first, none hidden. A merge is a new entry citing the lines it combines | The guide stops filling up; combining keeps its parts searchable |
| The whole inbox | All inbox items are shown, not the last 10 | Preserve messages |

**Not taken.** ScaleMAS rejects a post whose rationale does not name each ref; here citing stays optional, and E-ids named in the text count. Agensh caps entries at 100 characters; here entries keep their length and only the delivered line is short. Self-citations do not move the weight here (in ScaleMAS they do), because the engine adds refs to an agent's own plan.

**Size and cost.** A 50-agent synthetic load at F2's posting rates (no model calls): the turn-start view grows from about 2,100-3,400 tokens to 5,400-6,900, of which delivery is about 3,200 a turn. At F2's measured prices (input about $0.10 per million tokens; output, mostly reasoning, is most of the cost) that is about $1-2 more per 3,000 turns.

**Tests.** `test_board_delivers_everything_once_ranks_search_by_use_and_lists_every_piece`: 11 ideas listed, all claims delivered once and nothing on a second look, the unused finding listed and then shown as used, search ranks the used finding first with its citer, the weight fades, a guide past 16,000 characters is not refused and shows ids, a merge cites its parts. All suites pass (47 incentives + 61 others). Mock: 50 agents, 10 ticks, clean.

**The runs.** The user asked whether 50 agents was too many (10 or 20?). One 20-agent run replaces the 50-agent one (a second seed was started and stopped at t2 at the user's request, to save money), still 5 times F2's 4 agents per game. `AGENTS=20 SEED=0|1 CAP=20 tools/launch_vc33_depth.sh`: vc33 alone, 60 ticks, F2's configuration plus sections 12-14.

**What the run showed** (`runs_arc/H_vc33_20_s0`, finished 18:35, $17.19 for 1,200 agent-turns). One seed, and 20 agents on one game is 5 times the agents per game of every earlier run, so nothing here separates the board, idea credit and agent count.

| Level | First clear | All 20 copies | What did it |
|---|---|---|---|
| 1 | t3 | t5 | Published solution, replayed |
| 2 | t7 | t14 | Published solution, replayed |
| 3 | t28 (A7), after 21 ticks at 2 levels | t35 | A7's move cited 6 pieces by 3 others (A19 x3, A12 x2, A14), paid 2 credit each for understanding; the society's leading idea (a sideways shove, 8 claims) was wrong, and the solution was vertical pumping alone |
| 4 | not cleared in 30 ticks | | |

- **Depth.** 3 of 7 levels, the first run past 2 on vc33.
- **Level 4.** 69 new claims (25 new ideas) and 23 dead ends. The controls were mapped completely: three rail pairs, each with a raise and a sink handle, burst budgets of 1, 5 and 6 effective clicks, and ceilings. About ten goal guesses were refuted: a common row, a flat floor at r55 or r61, all bays down, the key at the floor, a hidden key, cross-pair re-arming, a ratchet. The level-3 analogue (move key1's colour-b letters to pipe 3's b band) came up at t47 and was still open at t60.
- **Where the last ten ticks went.** Two summaries of one deterministic measurement disagreed: A7's E1848 (a note: "six raises from pristine") against A8's E1900 and A12's E1962 ("one raise, then inert"). Agents settled such questions by rerunning, not by reading each other's receipts. RESET + 5x(16,62) + 6x(10,62) was played 20 times by 15 agents (t51-57), and RESET + 6x(10,62) 7 times by 7 agents (t48-60). Repeat notices went from 12 at t45 to 87 at t60. The game is deterministic, so one run would have settled each.
- **Explanations were not used.** The level 1, 2 and 3 explanations (E114, E247, E941) were cited by other agents 0, 1 and 0 times; E941 explains only the mechanism, because its author had not finished the replay. Level 4's goal search did not start from what made levels 1-3 complete.
- **Delivery carried the information; search was not needed.** Agents read 3,391 entries in full from their one-line deliveries (2.8 per agent-turn). `search what=posts` was called 6 times, none after t28. 65 of 78 agent findings (83%) were cited by another agent. The most cited entries were the two published solutions (66 and 36 citations) and guide lines (A5's E1110: 40 citations from 16 agents).
- **Guide and lessons.** The guide reached 90 lines and 43,000 characters over 95 edits, none refused, 0 merges. The lessons guide has 5 lines, all by t30: the step budget, clamped sliders, controls that consume a neighbour's room, the inert-click test, and that the exact cell matters.
- **Ideas and credit.** 36 new ideas; 111 of 133 claims built on pieces by two or more other agents. Understanding credit paid once (12 of 515 credit); 92% of credit still paid for new states. The shared model had 0 merges again.
- **Cost.** $0.0143 per agent-turn against F2's about $0.011 (+30%); 16.6M output tokens, 50M input (37M cached).

**Suggested next, not built.** (1) Let any agent ask the engine what a move sequence does from a level's start: the engine holds every receipt and the game is deterministic, so an answer from a past run costs no game steps, and a sequence nobody has run can be run once on a fresh copy for everyone. That would have replaced the 27 replications. (2) Ask for an explanation of what completed a level (the goal condition, not only the moves), and show the solved levels' explanations when a level is new.

**Why level 4 was not solved (from the game's source, 2026-11-01 evening).** Read from `ext/arcagi3_envs/vc33/5430563c/vc33.py` for diagnosis only; nothing from it goes into prompts. The black columns are tanks. Each rail handle pours 3 rows from one tank into its neighbour (pair 1: tanks A and B; pair 2: C and D; pair 3: D and E), and the key floats on its tank. The blue bars in the pipes are gates. A gate lights up (changes colour and shows a marker) only while the tanks on both sides are filled exactly to its bottom edge, and clicking it then swaps the keys across. Anywhere else, clicking it only uses a step. A level is won when each key's colour mark is level with the same-coloured target and on a tank next to the target's wall. On level 4 the key has to cross both gates to tank C and then rise until its b mark is at rows 29-30. A 21-click solution (`scratchpad/l4tools/l4_verify.py`) completes level 4 in the real game. Gates first appear on level 4 and recur on levels 5-7.
- **The society found the row but not the move.** The b-mark target row came up at t47. A17 then concluded that the key "can only move vertically", so the goal looked unreachable.
- **The gate was lit 65 times and noticed once.** Between t40 and t60 all 20 agents brought gate 1 to its lit state 65 times (RESET plus two pours, the "99, 150" cell counts that 34 posts quote; I found none asking why the second pour changed 50 more cells). 18 earlier gate clicks were in unlit states, and the lessons guide said a click that only drains row 0 "is not a control ... stop probing that neighbourhood".
- **The one real discovery was voted down.** At t57 A20 played RESET, 2 pours and then the gate: 57 cells changed and the key moved to tank B. A20 did not look at the frame. In t58, eight agents re-clicked the cell from other states (pristine, or after a raise), got 1-2 cells, and called the 57 "residue". A20 agreed, and A8's guide line E1921 recorded "(13,44) / row-44 is NOT a gate (settled, two receipts + one cross-check)". Nobody replayed A20's four-move receipt.
- **The limitation this shows.** Claims are stored without the state they were tested in, and they are weighed by how many agents agree. In a deterministic game one run from a known state settles what that state does; runs from other states say nothing about it. All 20 agents are one model with one prior (a cell either is a control or is not), so replication added agreement, not independent evidence. Agents did ask for frames (1,484 looks, 215 of them full; 731 of 1,502 plays with `show=true`; only 85 of 897 turns with moves saw no frame), but frames reach the model only as text, 64 lines of 64 colour digits. A20's t57 play asked for its frame, and reproducing the play shows the key's b mark at columns 18-23 instead of 3-8 in it; A20 still called the click inert. The engine's change summary ("57 cells changed in rows 0-54, columns 3-60") puts the key, the gate and the step bar in one box and does not say what moved.

**Read alongside: VISTA (Han, Hu, Qiu, Wu, He; arXiv 2610.02200, 2026-10-01).** One agent per game, no society, with frontier models at maximum effort (about 30.7M tokens per game). It reports 100.00 RHAE on the 25 public ARC-AGI-3 games with Opus 5.0, and 99.00 with GPT-5.6 Sol. The harness keeps every frame the game returns, including in-between animation frames, outside the model's context, indexed by turn and frame. The model receives only the final frame of each action and can call back any frame or region, enlarged, or its exact cell values. Its cumulative ladder (GPT-5.6 Sol, RHAE / games completed):

| Step | RHAE | Games completed |
|---|---|---|
| Official text-grid harness (external; different access route) | 13.33 | 4% |
| (i) Images instead of text | 47.32 | 36% |
| (ii) Higher action and time limits | 51.66 | 44% |
| (iii) One continuous conversation with compaction | 65.82 | 60% |
| (iv) Notes (GUIDE.md and WORKING.md) | 70.05 | 76% |
| (v) Frame archive + inspect | 94.10 | 96% |
| (vi) Exact pixel read-out | 99.00 | 100% |

- **What carried it was evidence the model can re-inspect.** Step (v) is the largest clean step, +24 points. With the weaker open GLM-5.3 Flash, removing only `inspect` and pixel read-out drops it from 66.93 to 13.27.
- **Images were not the main ingredient.** Text grids score 96.75 against 99.00 for images inside the full harness, but cost 2.3 times the tokens (71.9M against 30.7M). The 13 to 47 jump is confounded by the access change, and the authors attribute it to tokens: seven text frames are about 28K tokens per request, against about 2.2K as images.
- **Continuity +14; notes +4.** A larger context did not help: 780K tokens scored no better than 200K, at 3.4 times the tokens.
- **Unablated, but relevant here.** The prompt asks the agent, before each move, what it expects to see, and after each move, for "all visible changes, expected or not". That is the step A20 skipped at t57.
- **What AlphaTeam lacks by comparison.** It keeps only each copy's latest frame, drops in-between frames (`obs.frame[-1]`), and stores receipts as moves and change counts with no frames. A20's t57 frame could not be looked at again, by A20 or anyone else.
- **Adapted in AlphaTeam's terms (proposed, not built).** Each receipt keeps its frames, and any agent can look at a region of any receipt's frame or read its exact cells. In a society this makes the evidence itself shared: an agent can look at A20's frame instead of re-running the click from another state.

## 15. Evidence: what changed, every frame kept, and moves whose effect depends on the state (`evidence=1`, 2026-11-01)

**Why.** On vc33 level 4 (section 14), the one control that mattered works in a single state, and the society lost it in four ways:
- it was never seen changing: 65 lit-gate events, reported only as cell counts;
- A20's frame of the discovery was gone after its next move;
- eight runs from other states outvoted it;
- deterministic results were rerun instead of looked up.

VISTA (section 14) shows that evidence the model can return to is what carries a frame-based agent. Here that evidence is shared by the society, and the engine says what the records show; nothing tells agents what to do.

| Piece | What it does | GOAL.md |
|---|---|---|
| Object-level change reports (`alphateam/evidence.py`) | Each play step says what changed as objects: moved (same colour and shape elsewhere; touching parts that moved together are one object), recoloured, grew or shrank (with the edges that moved), appeared, gone. The count stays; the single bounding box goes. Counters are left out: edge rows or columns that change alone when nothing else does, learned per game (a corner cell on two edges teaches nothing). | Information, not instruction; general to frame worlds |
| Every frame kept | Each move's frames, in-between ones too (43 for the gate click), are kept packed in its receipt. `look receipt=R.. step=n frame=k region=[..]` opens any receipt's frame, anyone's; `step=0` is the frame before the play | Preserve work; evidence over its summary |
| Rare effects named | A move (or a click on the same small object) that did something in at most a quarter of its recorded runs on a level is named where it matters. To an agent whose click did nothing, it gives the run where it worked, with the moves since the level's start. To the agent who found it, it says the effect depends on the state and asks it to say which state. Each agent is told once per move per level | Evidence decides, not votes; credit and attention follow what happened |
| Engine records on posts | A finding, dead end or guide line that names a click with a rare effect carries the engine's record ("a click at (13,44) changed 56 cells in R313 step 3 (A20 t57, after click(16,62), click(16,62)) and nothing in 29 other runs"). Guide lines get it when shown, so it follows new evidence | A summary cannot outlive the record it contradicts |
| Recorded outcomes | States are keyed by the frame without counters, so a play whose every move repeats recorded runs from the same states is told so. `look outcome=[moves]` gives what those moves did from the copy's state, without steps, up to the first one not recorded | Preserve work; no paid repeat of a deterministic result |
| Expect, then observe | `play expect="..."`: the result opens with what the agent expected; the tool asks it to name every change afterwards | Our wording of VISTA's prompt line |
| `start=<file>` (test only) | Copies play the given moves before the run, unrecorded, so a test can begin at a later level. `tools/vc33_start_l4.json` holds the old run's published level 1-3 solutions | — |

**Not taken.** Images: OpenRouter lists `deepseek/deepseek-v4.1-flash` as accepting images, but VISTA's text grids scored 96.75 against 99.00 once the frames could be revisited, so text stays. Trust levels and "rests on" chains are built separately as `trust=1` (section 16). On ARC, ARC receipts carry their starting state (`start`) when `evidence=1`. With `trust=1`, ArcWorld also passes per-step results and reruns a receipt exactly from a level-start snapshot (`replay_receipt`, added by the fork), so `plan op=test` works on ARC; with `trust=1` off, `evidence=1` behaves as described here.

**Tests.** `test_evidence_reports_objects_keeps_frames_and_flags_moves_whose_effect_depends_on_the_state` uses vc33 level 4 from the start file and checks:
- two pours report the key sinking and the gate recolouring;
- the gate click reports "moved right 15";
- another agent opens that receipt's frame (key at c18-23) and its in-between frames;
- a click from another state is told where the gate worked;
- a guide line and a dead end naming (13,44) carry the record;
- a repeat is named;
- `look outcome` answers without steps.

All suites pass (incentives 48, bench_base 5, discoverybench 9, explorationbench 7, scigym 4, alphateam 18, design_world 8, kernel 10).

**Checked against the old trace, at no cost.** `scratchpad/l4tools/reenact_l4.py` re-plays H_vc33_20_s0's complete level-4 move log (t40-60, from each agent's first RESET): 2,858 moves, and every one changes exactly as many cells as the log recorded.
- The gate lighting is reported as "recoloured 1 -> c" 141 times, and the gate click as "moved right 15: a 6x6 object".
- A20 is told at t57 that its click did nothing in 7 runs from other states, and asked to say which state it needs.
- At t57-58, all eight agents who re-tried the gate from other states (A3, A4, A6, A8, A10, A14, A16, A17) are told it worked in R313 step 3 "after click(16,62), click(16,62)".
- A8's guide line E1921 would carry "a click at (13,44) changed 56 cells in R313 step 3 ... and nothing in 29 other run(s)".
- 206 plays (1,256 moves, 44% of the moves replayed) repeated recorded runs from the same states.
- Rare-effect notes total 23. The first version, without the rarity rule and the once-per-agent rule, sent 669, mostly handles that stop at their limits.

This shows what the agents would have been told, not what they would have done; only a run can show that.

**Cost.** A step line grows from about 90 to about 300 characters, roughly 0.6M more input tokens per 60-tick run, about $0.03 at $0.045 per million.

**Run (2026-11-02, `runs_arc/E_vc33_20_s0`).** The configuration is H_vc33_20_s0 (section 14) plus `evidence=1`, with every copy starting at level 4 (`start=`). It has 20 agents and was capped at $10, which stopped it at t28 ($10.12). **Level 4 fell at t11 ($4.61); the old run spent 30 ticks there without solving it.** All 20 copies were through by t13, by replaying A3's published moves. Level 5 was not solved in 17 ticks. Re-playing all 447 receipts through a fresh ArcWorld (`scratchpad/l4tools/reenact_e.py`) gives 0 id and 0 cell-count mismatches. It recovers the notes, which receipts do not store.

| t | What happened |
|---|---|
| 2-4 | The gate lights 7 times, a side effect of pours, reported as "recoloured 1 -> c: 36 cells at r43-54 c12-14". At t4, A7 posts E118: "the colour-1 gauge on wall c12-14 recoloured 1↔c (36 cells) — the gauge is live". In the old run's level-4 log, the same event happened 141 times as a cell count and was never named. |
| 6-7 | A20 reproduces it on purpose from the level start, citing E118. A5 finds that RESET restarts the level with a full move bar, which gives everyone the same starting state. A20 claims P19: the wall segments are gates, and the test is "the c12-14 segment's colour changes from 1 to c (36 cells)". |
| 8 | A4 and A19 click the lit gate: "moved right 15 / moved left 15: a 6x6 object (colours 4,b)". A20 posts its rule (the boat's row opens the gate), which is wrong. |
| 9 | Thirteen agents click the gate after pouring 2-3 times, most from the level start, citing E221/E224. |
| 10 | A3 reads the frame instead of the boat-row rule and finds that a gate opens when the two tanks it joins are level at its bottom (E272, from R177, where the second gate recoloured as C reached B's level). This is the rule in the source. Twelve agents read E272 the next tick. |
| 11 | A3 takes the key through both gates and raises it to the b mark (R202). It publishes the moves and an explanation; A5 replays them. |

What each piece did:
- **Object-level reports made the event seen.** It was the decisive piece. Agents' claims, tests and posts quote the report's own words ("recolour 1->c of the 36 cells", "moved right 15").
- **`expect`** was given in 378 of 447 plays. It shaped decisive tests ("Step 4 click(13,51) is the decisive test").
- **Rare-effect notes (71) carried the result but did not find it.**
  - The gate lit as a side effect of pours, and notes follow the clicked object, so no note pointed at it.
  - 13 of the first notes, at t1-2, were about the pour handle (10,62), whose effect depends on the water.
  - After the discovery, notes told 8 agents where the second gate click worked (R179, t10-11), and 15 more where the level-5 door clicks worked (R280, R309, R371).
  - A1's level-5 finding E498 cites R309 in the tick its note pointed there.
- **`look receipt` (18) and `look outcome` (27) were used little.** Copies' water levels drift apart, so outcomes were often "not recorded from this state".
- **152 exact-repeat notes**, many on the deliberate replays of the solution.
- **Attention concentrated on whoever held the evidence.** Up to t11, the top three authors (A20, A3, A5) drew 53% of the citations of others' posts, against 38% in the old run's level-4 phase. No groups, links, delegation or leads formed.

Level 5 has the gates turned 90 degrees, and copies carried the level-4 rule over ("transposed").
- By t28 they had mapped the pumps, confirmed the level lock, and found that a click on a lit door swaps two boats.
- At t28, A6 pointed at fixed coloured markers beside the doors, which fits the source's win check (a key matching a target's colour beside its tank). The budget ended there.

**What it does not show.** One seed against one seed, and the start file changes two things at once: all 20 agents met level 4 fresh at t1 with no guide from levels 1-3, while in the old run they arrived at t30-40 with long histories. The trace shows how the evidence was used. The same start with `evidence=0` (about 11 ticks, about $5) would show whether the fresh start alone explains it.

## 16. Test before trust, and DiG-bench (`trust=1`, world `dig`, 2026-11-01, built in a fork of the main session)

**Why.** On vc33 level 4 the society did test before it trusted: eight agents re-clicked A20's gate cell, saw nothing, and a guide line called the question settled. What was missing was whether a test reproduced the state the result came from, and a way to see what a conclusion rests on. The user asked to "test before we trust" and for "some sort of trust level so that one can trace back ... what assumptions are made". Traced on the old run (approximately, from the receipts each entry names), the most load-bearing unbacked piece is A17's t36 note "L4 KEY1 TRAVEL IS DOWNWARD ONLY": 133 later pieces rest on it within four citation steps.

| Piece | What it does | GOAL.md |
|---|---|---|
| Trust levels | The engine computes one for every finding, dead end, guide line, test and note from evidence, never votes: said (no engine record behind it), seen in k states (receipts from k distinct starting states), reproduced (an engine rerun matched, or someone else's run held), contested (a rerun failed, a test that repeated one of its runs failed, or a counterexample stands). Engine posts and published solutions are facts | Evidence decides |
| Rests on | Each piece lists what it cites, combines or names (claims it closes excluded) and shows its weakest link ("rests on unbacked E…"); a contested piece within three steps below it is flagged | Preserve and combine work, with its assumptions visible |
| `plan op=test` | With no evidence, the engine reruns the receipts behind a piece from the exact states they started in (worlds with `supports_replay`). With `result=holds|fails` and the tester's own receipts, the run is recorded as a test, marked as repeating one of the piece's runs (same starting state and the same moves) or not. Only a failure that repeats a run contests; failures under other conditions are counted and shown. (A first version compared starting states only. The ARC test showed why that is wrong: A20's play and a bare click both start at the level's start, but A20 poured twice first.) A test that moves a trust level earns credit (rule `test`, 0.3, evolvable within 0-2); the author is told | Test before trust; credit for useful work |
| Receipts know where they started | `start` (moves since the level began), `instance` (where copies differ) and per-step `results`; receipt lines read "from the start of level 4 + RESET click(16,62) click(16,62)" | A run is evidence about one state |
| Tags where pieces are seen | Findings, dead ends, carried pieces, unused findings, guide lines, everything delivered (kernel hook `entry_tag`) and search hits | — |
| `tools/trace_claims.py` | The tree under one entry with trust levels, or a run summary: load-bearing weak pieces, guide lines resting on weak pieces, every test. Older runs are traced approximately | Analysis after the run |

**DiG-bench world** (`--world dig`, `alphateam/dig_world.py`).
- **Games.** The 21 public games of arXiv 2608.12593, on api.digbench.ai with a token from an email sign-in (`DIGBENCH_API_TOKEN`). Names become `p1`…`p21`.
- **Instances.** Each agent plays its own session. The server seeds every session, so instances share rules but not details, and `supports_replay` is off: tests are runs on one's own instance, and "reproduced" means a claim held on another instance.
- **Screens and play.** Screens are short text, so there is no perception problem of the vc33 kind. Lives, step limits, creative mode and transitions are shown. RESET starts a new instance at level 1 and is refused while levels would be lost unless `restart_game=true`. A lost life and a game over become engine dead ends.
- **Solutions.** Published as "moves: …", replayable only from a level's start, and marked as found on another instance.
- **Concurrency.** play is async, with the HTTP steps off the event loop, so agents don't wait on each other's network calls.
- **`fake=1`.** Two local games with per-session seeds, for tests and rehearsals. T-1 is a cursor and a target. T-2 is a counter and a gate whose switch works only when the counter shows 3 (a state-dependent control), plus a creative mode.

**Comparing arms.** `tools/launch_dig.sh` runs either arm at about equal agent-turns, asking for both keys without echoing them. ARM=society runs PER_GAME agents per game, sharing everything. ARM=isolated runs PER_GAME one-agent runs per game with the same tools, sharing nothing. `tools/dig_compare.py` puts the society's best levels per game against the best of the isolated runs, with cost and agent-turns. Sync cannot run in the isolated condition, and the explore views would show others' findings, so the isolated arm is separate one-agent runs.

**Engine reruns on ARC.** With trust=1, each copy is snapshotted at every level start (`copy.deepcopy`, about 2 ms). Receipts carry `start` and per-step `results`, and `replay_receipt` reruns a receipt on a spare copy from its starting state, comparing every step's changed cells, level and state. They were added after section 15 was finished, with the main session's agreement, and are inactive without trust=1. On vc33 level 4, the test reproduces A20's three moves (99, 150 and 57 cells) by engine rerun. A click on the gate from the level's start is "not a repeat of its runs" and leaves the result reproduced. A doctored record is caught as "the record says 3 cells changed" and contested. A DiG failure on another instance is counted, not ruled on; since rules hold across instances it may be real, so it is shown next to the piece.

**Tests.** `tests/test_trust_dig.py`:
- the DiG world on the fake server: instances, receipts with start, level clears, published moves, life loss, RESET guard;
- trust levels: the engine rerun refused on DiG, a holds test from another instance reproducing, a fails test from another instance counted only, a said note and "rests on unbacked", a counterexample and "rests on contested";
- tags in deliveries and search, and the export;
- trust off changes nothing;
- a 4-tick mock society;
- on ARC, the gate case above.

The rehearsal of both arms with the mock model works end to end. All suites pass (incentives 48, alphateam 18, kernel 10, design_world 8, bench_base 5, discoverybench 9, explorationbench 7, scigym 4, trust_dig 5).

**Pilot (2026-11-02, `runs_dig/D_pilot_society_s0`, `runs_dig/D_pilot_isolated_s0`).** P-1 (14 levels), P-2 (5) and P-3 (8), 4 agents per game for 30 ticks, the vc33 design plus trust=1. The society is one 12-agent run ($4.31, 2.7 hours); the isolated arm is 12 one-agent runs ($5.26). Both used 360 agent-turns. The live API needed one change: the server's `levels_beaten` is now the level count (creative-mode levels do not count there).

| | P-1 /14 | P-2 /5 | P-3 /8 | mean score over 30 ticks | mean agent at t30 | cost |
|---|---|---|---|---|---|---|
| society, best agent | 13 (t25) | 3 (t28) | 5 (t23) | 0.443 | 0.706 | $4.31 |
| isolated, best of 4 runs | 13 (t25) | 3 (t30) | 5 (t30) | 0.418 | 0.552 | $5.26 |

- **The best agent ties: 21 levels in each arm.** The society got there sooner on P-2 and P-3 and at the same tick on P-1.
- **The typical agent does not tie.** In the society all four agents on P-2 ended at level 3 and all four on P-3 at level 5. The isolated runs ended at 2, 1, 3, 2 and 3, 5, 1, 2. On P-1 the arms are alike: 13, 13, 12, 12 in the society against 13, 12, 13, 13.
- **Rules moved between agents, not moves.** Each agent's instance has its own seed, so only 2 entries were ever run by another agent. Of the 21 first clears, 14 paid understanding credit, every time to entries by other agents. On P-3, A5 made all five first clears, and the other three reached level 5 on their own instances from A5's oracle rules (E338, E450).
- **Nobody used the test step:** 0 tests in all 13 runs. Of the society's 409 pieces, 333 were seen (backed by receipts), 73 said and 3 contested, all three through guide disputes. Agents asked each other in prose instead ("TEST THIS, everyone", E515). A test pays 0.3 credit only when it changes a piece's trust level, and on DiG it can never be an engine rerun.
- **Error correction worked on P-2 level 3 (t8 to t28).**
  - A10's guide line E255, from two runs, said "c plants a peg only where a hidden target has one".
  - A11 and A12 read other accepted columns on their own instances. Each concluded "the set is instance-dependent" from one state (E301, E309, tagged seen in 1 state).
  - A11's full sweep showed that planting both accepted columns does not clear (E333). A10 showed that acceptance depends on the state (E352). A12 then retired its own E308: "what looked like a hidden set was the 2-peg budget" (E393).
  - Phase 2 was found at t28.
  - Disagreeing instances were explained by state, not by instance. That is the reverse of vc33, where one refutation that ignored state killed a correct result. E255 was tagged contested.
- **A limit of the trace tool.** "Rests on" counts citations made to refute a piece: 43 pieces rest on E255, and most of them refute it. `trace_claims.py` should separate use from refutation (dead ends and refuting findings).
- **Speed.** A society tick took 87 s at first and about 5 minutes on average, up to 32 minutes (t8 to t9): 12 agents with large prompts, and each tick waits for the slowest. One-agent runs took about 3 minutes per tick.

**Next.** One seed, so the tie and the gaps carry no error bars. Both arms cost about $3.20 per game, so the other 18 public games would take about $57 of the roughly $64 left once the main session's ARC run ends. Cheaper designs:
- the society alone on the harder tiers (P-4..P-21), set against the paper's single-agent baselines;
- both arms on one game per tier above tier 1 (6 games, about $20).

**Shared copies (`copies=shared`, 2026-11-02, `runs_dig/D_shared_society_s0`).** The user asked why each agent plays its own copy, when teams could form, discuss and decide what to do. With own copies, a team has nothing to decide together: sharing findings is enough, so forming a team only adds cost. The one world where agents act on one object, the design world, is the one where coordination was essential and decision rights paid. `copies=shared` makes each game such an object:

| Piece | What it does | GOAL.md |
|---|---|---|
| One session per game | Everyone assigned to a game plays the same server session, and its steps per level and lives are spent together. A level solved once is solved for all of them | One assembled picture |
| Reconcile before acting | A play is not sent (no step spent) if the session moved since the agent last saw it, in its turn view, its last play or a look. It is shown who did what and the state now. Plays are applied one at a time | "Whoever contributes later reconciles with what is already there" |
| Nothing assigned | Who moves, and when, is for the agents to settle: board, links, groups, leads | Design the institutions, not the organization |
| RESET guard | A RESET that would lose levels is refused unless asked for, since it loses them for everyone | Nothing lost |

Code: `DigWorld._ck`, `_mirror`, `_moved_since`, `_moved`, the lock and check in `_play` and `_play_on`, the brief and tool texts, and one report line in `runner.py`. Tests: `tests/test_dig_shared.py` (3 here, and the lease test below). Without the option, own copies are unchanged.

**Run.** The own-copy society pilot's configuration (12 agents, 4 per game on P-1..P-3, 30 ticks, seed 0) with `copies=shared`: $3.05, 1.65 hours.

| | P-1 /14 | P-2 /5 | P-3 /8 | Score | AUC | Steps | Cost |
|---|---|---|---|---|---|---|---|
| Shared copies | **14 (t20)** | 2 (t26) | 3 (t18) | 0.592 | 0.386 | 1,928 | $3.05 |
| Own copies (pilot) | 13 (t25) | 3 (t28) | 5 (t23) | 0.718 | 0.443 | 4,944 | $4.31 |

- **P-1 was finished, the first time in any arm.** Every level fell at the same tick as with own copies or sooner (L10 at t13 against t17, L13 at t18 against t25), and L14 fell at t20. P-1 is arithmetic on exact quantities (Q, mQ, ...). On one session, each trace is evidence about the very state the others act on:
  - agents planned from each other's traces ("A4 used 9 steps on L3 ... banking mQ first is optimal", E47);
  - they steered each other's presses (A1's "STOP pressing z/x/c", E453; "press f only", E469).

  With own copies, each agent had re-solved every level on its own instance.
- **P-3 spent its lives.** Wrong answers cost a life, and 4 agents shared one session's 6 lives, where own copies had 6 each.
  - 4 lives were gone on level 1 by t7, and the last at t25. A RESET then took the session back to level 1.
  - The society re-cleared two levels on the new instance in five ticks.
- **P-2 stalled on level 2 from t3 to t26,** against t3 to t7 with own copies. Its rule checks presses in windows of three (E176). Several agents pressing one session make such a phase hard to track; one run cannot separate that from chance.
- **Half of all plays were refused.**
  - 327 of 650 plays were not sent because the session had moved.
  - In 80 agent-turns the agent made no move at all, and 86 times an agent was refused twice in a row.
  - Agents act at once within a tick, so whoever moves first makes everyone else's view stale.
- **No structure formed to settle who moves:** 0 groups and 0 links. Seven notes steered others with @-mentions. As in the design world, where agents never created authority over shared knobs, the need for coordination appeared and nobody built the structure to meet it.
- After P-1 was won, its four agents moved on: three to P-2 and one to P-3, adding to the contention there.

**Reading.** A shared object pays where state is exact and solutions do not carry over between instances (P-1). It costs where probing is risky and resources are shared (P-3), and where many hands disturb one state (P-2). The contention it created was paid in refused turns, not resolved by organization. One seed.

**Next (not built).** The design world's lesson is to offer decision rights over the shared object, never assign them. Two offered elements, which the rule loop can show or hide, at about $3 per run:
- a lease on a shared session, granted by a group's protocol or claimed for a tick, while others' plays wait ("A5 holds p3 until t12"). Built next, below;
- a scratch copy per agent for risky probes, with only the shared session scored. Not built.

**Holding the shared copy (`lease=1`, 2026-11-02, `runs_dig/D_lease_society_s0`).** The lease is an offered decision right over the shared session. Nothing in the world requires it, and the brief says so ("Holding is offered, never required").

| Piece | What it does | GOAL.md |
|---|---|---|
| `hold op=take` | Hold the game's session for this tick and up to 2 more (`ticks`), with a reason (`why`) that others see. While it is held, others' plays are not sent (no step spent), and they are told who holds it, until when and why | Offered decision rights, as in the design world |
| `op=request` | Join the game's queue. The holder hears of it. A holder cannot extend its hold while others wait | Nobody holds it forever |
| `op=release`, `op=pass to=` | Hand it on early, to the next in the queue or to a named agent, who is told | Agents settle the order |
| `for=<G-id>` | Hold it for a group of one's own, so that every member may move it | Structure has a use |
| Expiry | A hold ends with its last tick and passes to the next in the queue | No enforcement beyond the rule agents chose |

Code: `DigWorld._lease`, `_grant`, `_end_lease`, `_lease_text`, `_may_move`, `_hold`, and `on_tick_start`, plus the hold check in `_play`, the `hold` tool, and the brief, observe, score and export lines. Test: `tests/test_dig_shared.py::test_holding_the_shared_instance_is_offered_and_passes_in_turn`, and the mock run with `lease=1`.

**Run.** The same configuration with `copies=shared lease=1`: $3.49, 1.2 hours. One seed in each row.

| | P-1 /14 | P-2 /5 | P-3 /8 | Score | AUC | Steps | Levels per 1,000 steps | Cost |
|---|---|---|---|---|---|---|---|---|
| Shared, with the hold | 12 (t30) | **3 (t18)** | 5 (t29) | 0.694 | 0.399 | 1,134 | **17.6** | $3.49 |
| Shared copies | **14 (t20)** | 2 (t26) | 3 (t18) | 0.592 | 0.386 | 1,928 | 9.9 | $3.05 |
| Own copies (pilot) | 13 (t25) | 3 (t28) | 5 (t23) | **0.718** | **0.443** | 4,944 | 4.2 | $4.31 |

- **The offered hold became a turn order on every game.**
  - There were 4 takes and 69 requests. Agents joined the queue at position 2 or 3 and waited 3.1 ticks on average; 54 of 73 asked for 2 ticks.
  - Every game was held almost all the time: 94 holder-ticks over 90 game-ticks.
  - 50 holds ran their full length; 1 was released early and 2 were passed on.
  - No agent assigned the order, and none was assigned by the world. The queue in the offered rule became the society's schedule.
- **Collisions ended, but the urge to move did not.**
  - Plays refused because the session had moved fell from 328 to 9.
  - 150 plays were refused because someone held the session, about 5 per tick from t1 to t30, by every agent. Agents did not learn to wait.
  - Agent-turns with a sent play were 94 of 352, against 210 with shared copies and 343 with own copies.
- **Lives were kept.**
  - P-3 lost 2 lives in ticks 1–3 and none after, over four more levels. Without the hold it lost all 6 lives and the game by t25.
  - P-2 lost none, against 3 without the hold.
- **Waiting agents became advisers to the holder.** With shared copies alone there were 0 links.
  - There were 15 links and 85 link messages. 55 of the 100 delivered messages to a named agent went to the current holder of the sender's game. For example, A12 wrote to A9 at t19: "you hold p3 and the '!' chain screen is 'cabcc' ... a TIE. Answer n". A2 wrote to A1 at t18: "you hold p1 until end of t18 and your hold reason says 'finish L9', but ...".
  - The first group in any DiG run formed: G1 "p2-walk" (A10, t20, then A6, A8 and A11), to keep the walk mechanics and negative results in one place. It never held the session. No hold was taken for a group.
  - The board grew. Guide edits rose from 54 to 193, plan calls from 112 to 213, and global posts from 5 to 90. Guide merges rose from 1 to 12, and findings from 21 to 70.
  - There were the first 7 uses of the test step: 1 recorded and 6 refused, because DiG cannot rerun a run and no evidence was given.
- **One mover per game per tick capped throughput.**
  - From t2 on, exactly one agent moved each game in a tick. That held a game to the holder's allowance of 15 to 30 steps, against up to 67 steps a tick on P-1 without the hold.
  - P-1's later levels need finishers of 32 to 54 steps ("needs 30 steps this tick", A4 at t23). L10 fell at t21, against t13 without the hold, and P-1 was not finished.
  - A group hold would have let four members move, but none was taken.
- **Requests went stale in the queue.** Of the 44 granted requests that named a level, 21 named one already cleared by the time they were granted. For example, A2 asked at t3 to clear L2 with "exactly 2 q's" and was granted the hold at t4, on L3. The `why` was written for a state that had gone.
- **A failure unrelated to the hold:**
  - 409 plan claims were refused, 277 of them by A1 and 97 in a single reply at t7.
  - None had a `hypothesis`, and 408 carried `result` (holds or fails), the field for `op=test`. The same mistake caused every refused claim in the other two DiG runs (17 and 66).
  - With `--calls`, a reply may hold any number of actions, so a degenerate reply repeats a refused call. 10 `link` calls with content and a target but no `op` were refused too, among them advice to the holder.

**Reading.** The design world showed that agents do not create decision rights over a shared object. Here, when the right was offered, every game used it, and an order formed from it at once. The order kept shared lives and turned the agents who waited into advisers, links and a group. That changed what the society made more than any earlier DiG option. Its cost is serial execution: the throughput of one agent per game per tick. It also brought new problems: plans written for a state that had gone, and moves still tried against a held session. Against own copies, the score is about level (0.694 against 0.718) at a quarter of the steps, but behind on AUC. One seed per row, so differences in a single game are not separable from chance.

**Next (not built).**
- **Information for the holder; the holder still decides:**
  - When the level a hold named is cleared and others wait, tell the holder.
  - When a queued agent is granted the hold, show it what changed since it asked, as a refused play already does.
- **Remove the throughput limit for those who choose it:**
  - The waiting agents' unspent steps could go to the holder, offered as `hold op=request ... lend=1`.
  - Or a group hold could be easier to take: today it needs a group to exist first.
- **Outside this world: fixed by the main session the same evening** (`tests/test_fixes.py`; see §17):
  - in `llm.py`'s reply loop, an identical call refused earlier in the same reply is not run again;
  - `link` with no op sends when it has content, and opens otherwise;
  - a plan with `result` and neither op nor hypothesis is a test; a claim with `result` and no hypothesis is told that `result` belongs to `op=test`.

  The lease run's claims named `op=claim` explicitly, so they would now get the clearer refusal and would not become tests.

## 17. Islands: diversity by delay, groups that pay, nothing lost (`islands=1`, 2026-11-02)

**Why.** Every vc33 and DiG run so far had 0 groups, and the society's best stayed at one agent's best. The agents' game states were already varied: E_vc33_20_s0's 205 level-4 plays started from 173 move histories, and in H_vc33_20_s0 all 20 agents saw the gate light. What converged was belief. With `board=1` every agent gets the same inbox and the same guide, so all 20 hold the same theory, and one guide line ("settled: not a gate") stopped everyone. This is the effect the midterm report cites from Lazer and Friedman (a well-connected network converges too early on hard problems). It also matches the report's Commit-0 result: exchange every 3 rounds beat both every round and only at the end.

The user proposed private boards, free communication inside a group, and few posts to the common board per time window. They asked that nothing be lost and that diversity be measured, not just the score. The design gives a group a job: inside it, sharing is immediate, and outside it, sharing is delayed. That is the report's team condition (join when collaboration pays more than it costs) and Coase's (organize where coordinating inside is cheaper than across).

| Piece | What it does | GOAL.md |
|---|---|---|
| Islands | An agent's island is itself plus every agent that shares a group with it, directly or through nesting. A post reaches its island at once. | Primitives, not organization: agents choose groups |
| Opening | A post by another island opens to an agent `embargo` ticks after it was posted, earlier if the world says its level has fallen (`entry_open`: the entry's level ≤ the best level reached), or when someone in its island publishes it. Once open, it is delivered at the next turn start like any other post (each post once). | Nothing is lost: every post is open to everyone by tick + embargo |
| Publishing | `share entry=E..` publishes an entry by you or your island to everyone now: `quota` per `window` ticks per agent (`publish=1/3`) | Few posts to the common board |
| Evidence stays shared | World posts (engine dead ends, solutions, kinds, rules), play data, receipts and their frames, `look outcome`, and rare-effect notes reach everyone at once. The limits apply to interpretations: findings, claims, guide lines, explanations, critiques and closing notes. | Check others' evidence without adopting their beliefs |
| Views follow the same rule | The guide shows only open lines, keeping their numbers. The IDEAS list, findings, dead ends, carried and unused findings, open plans, search, read and following all hide posts that aren't open yet. Each turn says how many posts are held and when the next one opens. Mentions, claim-overlap notices, and merge, challenge and edit notices reach only agents who can see the post. | — |
| Guide edits keep what the editor can't see | `guide op=edit` re-appends lines the editor cannot see; merge and challenge accept only open lines | Preserve work |
| Directed messages | Links and posts to a group's public board reach their addressees at once | — |
| Group cap | A group, and every group containing it, holds at most `group_cap` agents (connect join/admit, team join). Without it, one group of 20 is just the common board again. | — |

**Options.** `islands=1 embargo=3 publish=1/3 group_cap=5`; needs `sync=1 board=1`. `tools/launch_vc33_depth.sh` takes `ISLANDS=1 EMBARGO PUBLISH GROUP_CAP` and `START=l5` (`tools/vc33_start_l5.json`: the level 1-3 moves plus A3's published level-4 solution, 91 moves; every copy starts on level 5).

**Measures** (`tools/diversity.py RUN [RUN ...]`, per level):
- claims and distinct ideas: word overlap ≥ 0.33. On E's level 5, claims stating the same door rule scored 0.36-0.40 and claims about different mechanisms at most 0.27. Word overlap is a crude proxy for meaning.
- distinct ideas tested;
- mean similarity of claims by different agents, and across and within islands;
- the top three authors' share of citations;
- distinct starting move histories of plays;
- the tick the level fell, and how many copies got past it;
- the best and mean copy at the end.

On the earlier runs: E level 5 had 37 claims, 29 ideas, 12 tested, similarity across agents 0.153, and 76% of citations to the top three. H level 4 had 85 claims, 77 ideas, 35 tested, 0.144 and 45%.

**Tests.** `tests/test_islands.py`:
- a claim and guide lines are hidden from another island (view, search, read, the IDEAS list, the guide);
- forming a group makes an island; the cap refuses a third member;
- an edit by an outsider keeps the lines it cannot see;
- one publish opens one line, and a second in the same window is refused;
- receipts are open at once;
- at the embargo tick, every post from t1 reaches the outsider exactly once, and all posts are visible to all;
- a post opens when its level falls;
- the export lists what was published and what was opened by a level;
- the cap holds for teams and nested groups;
- without islands, everything reaches everyone at once and publishing is refused.

A free 4-tick mock run of the full configuration works end to end (groups formed, joins past the cap refused). All suites pass (incentives 48, islands 3, alphateam 18, kernel 10, trust_dig 5, design_world 8, bench_base 5, discoverybench 9, explorationbench 7, scigym 4).

**Test run (2026-11-02).** vc33 level 5, which no run has solved, with both arms starting there. 20 agents, 20 ticks, $8 cap each, the E configuration (evidence=1):
- `S5_vc33_20_s0`: full sharing;
- `I5_vc33_20_s0`: `islands=1 embargo=3 publish=1/3 group_cap=5`.

One seed each.

**Why no team in I5 (trace, 2026-11-02).** In I5's 748 agent-written texts there were no messages, no @-mentions and no talk of groups. Agents tried 82 times to read held posts. After a refusal, 43 of 45 next looks were at their own frame, so they worked alone instead of reaching out. They named other agents in the third person 117 times and addressed them 0 times. The failure modes:
- nothing needs another agent: each has its own copy and 15 moves per tick;
- information arrives without asking;
- the held-post notice named nobody;
- whoever founds a group gains nothing until someone joins;
- a reply takes a tick;
- links need both sides to open them;
- reading takes 60-71% of actions, and 1 in 6 reads chased level 1-4 solutions that the start file played without recording;
- the prompt described the tools but tied them to no payoff.

**Teaming (`teaming=1`, for I6).**
- A prompt paragraph on what working with others does, as information, not instruction.
- A MEMBERSHIP block each turn: your island; who holds posts you can't see yet, by author, kind and level, without their text; groups and their places; the moves.
- With islands, @-mentions reach their addressee at once.
- A start file's levels are posted as earlier-society solution records.

**I7 additions:**
- `messages=1`: a `message` primitive to an agent or a group, delivered during the recipient's turn and kept in `messages_log`; 8 per tick.
- Founder help: a refusal for a full group names the others turned away and the move that starts another group. The membership block lists agents in no group and those turned away lately.
- Directions: a group's purpose is shown as its direction, and a claim close to another group's direction is pointed out.
- `island_guides=1`: each island keeps its own guide. Other islands' open lines are listed and viewable (`guide op=view island=..`), and pairs of open lines from different islands on the same subject are shown. An edit keeps other islands' lines.
- `unique_credit=1`: when a level falls, tested ideas no other island tested earn 0.5 each, up to 3 per island.
- Tests: `tests/test_islands.py`, 5 tests. All suites pass.

**Results (vc33 level 5, 20 agents, 20 ticks, one seed each; level 5 solved in none):**

| | S5 full sharing | I5 islands | I6 + teaming | I7 + the five additions |
|---|---|---|---|---|
| Claims / distinct ideas / tested | 54 / 39 / 17 | 45 / 36 / 11 | 68 / 54 / **25** | 59 / **53** / 13 |
| Agents claiming | 19 | 16 | 20 | 20 |
| Claim similarity across agents (lower is more diverse) | 0.153 | 0.136 | 0.141 | **0.131** |
| Top three authors' share of citations | 70% | 76% | 61% | **53%** |
| Groups (agents in them) / messages | 0 / 0 | 0 / 0 | 1 (5) / 0 | **2 (10) / 2** |
| Boat crossings through a gate | 1 | 0 | 1 | 0 |
| Cost | $6.29 | $5.89 | **$4.06** | $5.13 |

What worked:
- **Showing the payoff made groups form.**
  - I6: A6 founded G1 at t2 "so L5 findings don't wait 3 ticks". It was full at 5 by t4, and 4 more agents were refused. Each of them did nothing afterwards.
  - I7: after a refusal, A9 messaged the full group. A18 founded a second group, G2, "by splitting the work", and A8 joined because its direction "matches my button map".
- **The earlier-level records.** Searches for level 1-4 solutions fell from 12-15% of reads to 3-4%, and the records were read 154 times in I6.
- **Messages.** Rare (2), but real. At t17, A4 told A14 that its "E420 arm-gate ... does NOT reproduce with (62,36) on my copy": a replication report sent to the author.

What did not:
- **Publishing.** 0 uses in I5, I6 and I7.
- **Islands alone (I5).** Fewer claims and tests than full sharing, for a small gain in diversity.
- **Directions.** Mostly generic ("Solve vc33 level 5 ... share probe results instantly"); the near-direction note never fired.
- **Groups shared; they did not divide work.** 1 of G1's 117 member texts in I6 speaks of splitting work, and 0 in I7's G1. Splits happened implicitly, by reading.
- **Island guides.** The guide fragmented into 10 small notebooks (G1 had the most lines, 6), and other islands' guides were opened only 6 times. There were fewer guide lines (26 against 44) and fewer tested ideas (13 against 25) than I6: more distinct ideas, less depth.
- **Unique credit.** Untested, since no level fell.

**Reading.** Teaming without island guides (I6) gave the best mix: the most ideas tested, the least concentrated attention, and the lowest cost. The I7 additions raised breadth (distinct ideas, spread of attention, groups, messages) and lowered depth. No organizational setting cracked level 5. As at level 4, the bottleneck there is reading what wins the level, not organization.

**Next.**
- Remove the publish quota.
- Drop island guides, or show other islands' lines inline instead of in separate notebooks.
- Keep teaming, founder help, messages and the earlier-level records.
- Show each group's open claims instead of its free-text direction.
- For level 5 itself, a perception aid in the spirit of section 15 (for example, objects whose colour matches a target).
- The fork found three bugs in these files:
  - with `--calls`, a reply can repeat a refused call (one agent sent the same refused claim 97 times);
  - `link` without `op` is refused;
  - refused DiG claims put the hypothesis in `result`, and the refusal could say so.

**Fixes the fork found (2026-11-02, `tests/test_fixes.py`).**
- llm.py: an identical call refused earlier in the same reply is not run again (one agent had sent the same refused claim 97 times).
- kernel: `link` with content and no op is a send, and with no content an open.
- explore: with trust=1, `result=` without an op or a hypothesis is a test; a claim with `result` and no hypothesis is told that `result` belongs to op=test.

**What the findings were worth, graded against the game (level 5).** The source gives four pieces:
1. each right-edge button moves water 3 columns between neighbouring decks, and a boat rides its deck's water edge;
2. a band's door opens when the two decks it joins are level at its edge;
3. a click on an open door moves the boat(s) through, swapping them if both are there;
4. the goal: the e-boat must end beside the ee marker in the bottom wall (column 10) and the b-boat beside the bb marker in the top wall (column 16), so the boats swap decks and align with the marker of their own colour.

In S5, I5, I6 and I7:
- every run had piece 1 by t5, used by 15-19 agents;
- only S5 stated piece 2 correctly (E326, t12). The most-used door findings in S5 (E186, 137 uses) and I7 (E234, 126) credit the boat's position, which moves with the water, so the cause is confounded;
- boats crossed at most once per run;
- nobody stated piece 4.

The closest anyone came was E's A1 at t15 (E414): "L5 transposed ⇒ boat moves through doors vertically, then moves horizontally (pumping) to the b marker". It transposed A3's level-4 explanation (E319), which 17 of E's agents had used. The runs that started at level 5 had the level 1-4 moves but no explanations.

**I8 (I6's settings plus the earlier society's level-4 explanation, A3's E319 and E320, posted with the records; `tools/vc33_start_l5x.json`; $7.86; level 5 not solved).** The explanation was read 214 times, and agents reasoned from it from t1 ("L5 is L4 rotated 90°").
- **Piece 2** was stated correctly at t7 (A1) and refined at t13 (A10: "opens when BOTH tanks' surfaces reach the door's far (right) edge c39").
- **Piece 3:** boats crossed doors 74 times, starting at t6 (e-boat down 44, up 21; b-boat up 6, down 3), against at most 1 in any earlier level-5 run.
- **Piece 4:** A7 stated the right pairing at t5 (E118): "the two markers bb (wall r14-16, c20-21) and ee (wall r49-51, c14-15) are the candidate exits for boats 4,b and 4,e".
- **A decoy took the society.** 46 posts by 16 agents chased the "row-0 colour-4 hatch", which is the move counter: it grows with every click that changes something, so it moves with the boats. Change reports leave the counter out (0 report lines mention row 0), but the frames from `look` show it.
- **Diversity fell** (claim similarity 0.197; 24 distinct ideas, 10 tested) as everyone worked inside one framework. 2 groups formed.

What it shows:
- Understanding carried across levels as explanations is what transfers. Moves do not, and neither does organization.
- The recurring failure is a confounded correlation: a boat's position with its deck's water level at level 5, the boat's row with the gate at level 4, and here the counter with boat moves.

Next:
- Have `look` say which rows are counters. The engine already learns them.
- When an event recurs, have the engine contrast the states where it happened with those where it did not, so a confound is visible as evidence.

**I9 (2026-11-03; I8's settings plus the novelty fix of section 18; $6.67; level 5 not solved).** I9 is J1's control: the same code, without directions, trust or change_credit. The always-on notice naming a dropped claim was also on; it made no difference in J1 (16 claims dropped before the author's next claim, against 18 in I8).

- **The fix contains the decoy.** These counts cover agents' posts and the outcomes they predicted for their moves that read row 0 as a hatch or exit (pattern `hatch|exit|row 0|r0`):

  | Run | Fix | Decoy texts | Agents | Most in one tick | Novelty credit |
  |---|---|---|---|---|---|
  | S5 | no | 109 | 19 | 12 | 265 (96%) |
  | I5 | no | 31 | 13 | 11 | 281 (97%) |
  | I6 | no | 30 | 14 | 5 | 240 (95%) |
  | I7 | no | 43 | 16 | 10 | 325 (98%) |
  | I8 | no | 96 | 19 | 18 | 240 (97%) |
  | J1 | yes | 11 | 6 | 2 | 81 (74%; the solve paid the rest) |
  | I9 | yes | 10 | 8 | 2 | 94 (94%) |

- **How the fix acts** (the fork's point). Agents never see credit for a single move: none of I8's 376 play results mentions it. The route is steps. Credit earned in the last two ticks buys 2 extra steps per point, up to double the 15. So paying for counter states gave agents longer click streaks, and in those streaks the bar visibly grew while the boats moved. Moves executed per agent at level 5 (from receipts): I8 99, I9 78, J1 76 (two ticks lost). Counts from play arguments in events.jsonl undercount, because the log cuts long move lists short (see section 18's counting note).
- **Why I9 did not solve: the swap came late and nobody named it.** Swap and crossing counts come from the level-5 change reports in `receipts_full`: a swap is a step in which one boat or cargo moves 10 or more rows up and another moves down; a crossing is any move of 10 or more rows. "Goal" texts mention a wall token near words like goal, target, dock or finish.

  | Run | First swap | Swap steps / agents | First named | Crossings | Door lights | Tokens as the goal |
  |---|---|---|---|---|---|---|
  | I8 | t16 | 10 / 5 | t18 (A6) | 84 | 76 | 21 texts by 11 agents, from t3 |
  | J1 | t12 | 24 / 17 | t12-13 (A4, A17) | 106 | 93 | 26 by 12, from t4 |
  | I9 | t17 | 8 / 5 | never | 48 | 62 | 5 by 4, from t1 |

  I9 spent its middle ticks pinning down exactly when a door lights: 55 door-rule texts by 18 agents.
- 3 groups formed, and 75% of agents were in one by t18, the most of any run. 38 claims, 30 never tested; claim similarity 0.194; the three most-cited authors drew 79% of citations.

What it shows:
- **Keep the fix in every run.** It removed a false signal: the decoy spread in 5 of 5 unfixed level-5 runs and in neither fixed run.
- **It is not sufficient.** The solve needs the swap seen and named early enough to test the goal. J1 had it at t12; I8 and I9 had it at t16-17, with 3-4 ticks left. J1's added layer was unused, so its earlier swap was most likely chance (one seed each).

Next:
- A longer run of I9's settings: 30 ticks, about $10 of the $16.45 left.
- Or, in the spirit of section 15, an engine note when two objects exchange places, so a swap is named the moment it happens.

## 18. Directions: a parallel search kept visible (`directions=1`, 2026-11-02, fork session)

**Why.** The user asked why no hierarchy formed, then pushed back on one part of the answer: "insights could also be parallelized if diversity is increased". The traces bear that out. The search for an insight splits into draws, and twenty independent draws are worth about twenty chances, twenty correlated ones about one. In I8 the society held the pieces: A7 stated the goal pairing at t5 (E118), and A1 the door rule at t7. But 46 posts by 16 of the 20 agents chased the move counter, and the goal pairing was never tested; A7 itself moved on. Section 17's groups shared rather than divided work, and I7's directions were free-text group purposes ("Solve vc33 level 5 ... share probe results instantly"). A parallel search needs three things together: different directions, persistence on an unpopular one long enough to test it, and integration of pieces from different directions. That is also the job a research lab's PI does, so directions give organization something to do on a single frontier.

| Piece | What it does | GOAL.md |
|---|---|---|
| Direction | `direction op=open text=..` posts one line of attack on the agent's level (D-id); `op=join id=D..` and `op=leave`. An agent is on at most one at a time, in a group or not. | Offered, never required |
| Filing | Claims and findings an agent posts while on a direction are filed under it (a plan's later findings follow the plan's direction) and tagged `[D3]` in the IDEAS list and wherever entries are delivered or found | — |
| Headcount view | DIRECTIONS on your level, fewest agents first: members, opener, what its claims came to (worked, refuted, unclear, testing, never tested), findings; a note when one direction holds over half the game's agents; how many are on none | Information, not instruction |
| Persistence | YOUR DIRECTION at the top of a member's turn: ticks on it, how many share it, how its pieces stand | — |
| Credit | When a level falls, each direction on it whose pieces the solver cites (play cites, the pieces its plans build on, and what those combine), or that the solver is on, earns `direction` credit (default 3, bounds 0-10), split among the agents who were on it by their ticks on it plus three shares for each of their claims on it tested to a result (works or dead end), including those who left. Paid next to unique-idea credit at the start of `_first_solve`, and reported apart (`direction_credit`). All directions on the level then close. | Credit for being right early against the crowd |
| Different views | Within each delivery section, after identical posts are collapsed, posts of directions with fewer agents first (the agent's own first among equals), then posts filed under no direction. A crowd's posts come last for everyone, its own members too. Nothing is dropped and the pushed and collapsed counts are unchanged. The DIRECTIONS list uses the same order. | — |
| Integration | PIECES THAT HELD, BY DIRECTION: backed, uncontested findings that worked, listed side by side when two or more directions have some. A claim or finding building on pieces of two directions is marked as combining them, and their members are told. | Preserve and combine work |
| Islands | A direction is a post: its text and pieces open with the islands rule. Before then the view shows its headcount, members, opener and when it opens, not what it says, and it cannot be joined. | — |
| Naming | With `directions=1`, a group's purpose is called its purpose in the MEMBERSHIP block, the TEAMING text and the near-claim note, so "direction" names one thing | — |

**Options.** `directions=1` (needs `explore=1`; the views use `hypotheses=1`, `trust=1` and `islands=1` when they are on). `DIRECTIONS=1` in `tools/launch_vc33_depth.sh` and `tools/launch_dig.sh`.

**Code.** `alphateam/directions.py` (the `Directions` mixin; `Explore` inherits it). Hooks, each inactive without the option:
- explore.py: init and tool, the `direction` rule, filing in `_plan` (claims and findings; the trust_on default stays), the pot in `_first_solve`, the view after OPEN PROBLEMS, the tag in `ideas_view` and `entry_tag`, the brief, score, export and mock;
- kernel.py: an optional env `deliver_order` that orders collapsed groups within each section of `_deliver_all`, and "purpose" in the MEMBERSHIP block;
- primitives.py: "purpose" in the TEAMING text.

**Measures** (score): `directions`, `directions_joined`, `directions_paid`, `direction_credit`, `directions_tested` (with at least one plan done), `direction_peak_share` (the largest share of a game's agents ever on one direction: herding), `pieces_filed`, `plans_combining`. Export: `directions` (each with members, tenure, plans, pieces and what it was paid), `direction_log` (opens, joins, leaves), `piece_dir` (entry or plan to direction); each plan carries `direction` and, when it combines directions, `combines`.

**Tests.** `tests/test_directions.py`, 9 on the DiG fake server and ARC vc33 (the last three with the audit below):
- open, join, filing, the IDEAS tag, own direction first in the view, then fewest agents, and agents on none;
- a claim building on another direction's finding combines the two, and its members are told; leaving keeps one's ticks;
- deliveries put a smaller direction's posts first within a section, one's own first among equals, with nothing dropped;
- a level that falls pays the solver's direction 6:2 between its two members (3 ticks and one claim tested, against 2 ticks), pays a cited direction's only member 3, pays nothing to an uncited one, and closes all three;
- without the option nothing changes; a mock run works end to end;
- on vc33 level 5 with islands and teaming, another island sees the headcount but not the text until the embargo ends, and a group's purpose is called its purpose.

All 13 suites pass (bench_base 5, discoverybench 9, explorationbench 7, scigym 4, alphateam 18, design_world 8, dig_shared 4, directions 9, fixes 3, incentives 48, islands 6, kernel 10, trust_dig 5).

**Not built.** Search ranked by direction; directions carried to the next level; a direction's own board. No paid run yet. The proposed test is one vc33 level-5 arm with I8's settings plus `directions=1` (about $6-8), compared with I8 on `direction_peak_share`, directions tested to a conclusion, whether a minority goal direction is tested, and the tick level 5 falls.

### Credit and diversity audit of the level-5 runs (2026-11-02, fork session; the changes J1 tests)

The user asked for a deep analysis, following GOAL.md, of where to increase diversity and whether credit can be assigned better, then a run. GOAL.md makes credit the central means ("it tells agents which actions and combinations paid off") and says credit needs consequences. Here it has one: credit earned in the last two ticks buys steps. The audit covers S5, I6, I7 and I8, the four vc33 level-5 runs (20 agents, 20 ticks each). The replay tool is `novelty_sim.py` (it reruns every receipt from the level-5 snapshot and recomputes novelty), kept in the fork's scratchpad.

**Credit paid for spending the step budget.**
- In the game source, row 0 of vc33 is the level's step bar. Every click uses a step, and the colour-4 part grows as the budget runs out (level 5: 200 steps, 64/200 of a cell per click; at 0 steps the level is lost).
- Credit for new frame states was 95-98% of all credit in all four runs (I8: 240.2 of 248.5).
- Replaying every level-5 receipt from the snapshot: S5 56%, I6 65%, I7 64% and I8 56% of the states paid as new were new only in row 0. So in every run more than half of all credit paid for burning budget, and bought the payee more steps to burn.
- The novelty rule meant to leave counters out: it masks cells that change on more than 30% of the first 50 steps. At level 5 the bar moves on about a third of clicks and fewer than that early, so it masked 0 cells. `evidence.py`'s Counters had found row 0 before the first play, from the start file's moves.
- The "exit hatch" the society chased (section 17) was this bar: it grew as boats were walked because walking them took clicks.

**How the decoy spread (I8).**
- A16 called the bar "a progress meter" at t4 (E92). A19 made it a GUIDE line at t8 (E193: "L5 EXIT HATCH ... the likely exit for boat 4,e"), citing three receipts that do show the bar growing. The receipts back the observation; nothing backs the interpretation.
- With islands, the line opened to other islands at t11. At t11, six agents (A1, A3, A4, A8, A13, A15) took it up, each having read E193 or A19's note of the same idea (E194); A14, A2 and A18 followed at t12-t15. E193 was read by 14 agents and cited by 9.
- The right idea and the refutation were closing notes:
  - A7's goal pairing (E118, t5) was read by 12 agents and cited by 2, both A7's groupmates;
  - A14's refutation (E277, t12: walking the boat under the "hatch" does nothing; the bb and ee markers were "never tested by anyone") was read by 4 and cited by none.
  - Notes are not claims, so neither entered the IDEAS list. No challenge was filed against E193, and challenges pay nothing anyway.
- I8 ran without trust=1, so E193 showed no trust level.

**Persistence: half of all claims were dropped untested by their own author.**
- An agent has one open plan, and a new claim silently drops it. At level 5, S5 dropped 27 of 54 claims this way (23 with no finding), I6 39 of 68 (32), and I8 20 of 42 (18), on average 5.7-9.2 ticks after the claim.
- `tools/diversity.py`: claims never tested S5 34/54, I6 43/68, I8 32/42.

**Diversity measures (tools/diversity.py, level 5).**

| | S5 | I6 | I8 |
|---|---|---|---|
| Claims / ideas / tested | 54 / 39 / 17 | 68 / 54 / 25 | 42 / 24 / 10 |
| Similarity of claims across agents | 0.153 | 0.141 | 0.197 |
| Top three authors' share of citations | 70% | 61% | 64% |

I8's shared level-4 explanation (read 214 times) put everyone in one framework: "L5 is L4 rotated, find the exit". That transferred the door rule, and it narrowed the search to which thing is the exit. Both A7's right answer and the decoy came from it.

**Changes (all offered; J1 tests them together).**

| Change | Where | Addresses |
|---|---|---|
| New-state credit leaves out the counters the evidence engine found (a fix: always on; inert without evidence=1). The counter mask is ORed in where each state is keyed, not into the 50-step mask, which is frozen, so counters learned later count too. A counter learned mid-run re-keys states once, so a state seen before it can be credited once more; on vc33 the start file's moves teach row 0 before the first play. | explore.py `_novel` | Credit paid for the step bar |
| `change_credit=1`: a test that moves a piece's trust level pays the test credit × (1 + other agents who used the piece, up to 4). A counterexample with receipts to a finding (plan op=challenge) or to a guide line (guide op=challenge) pays the same, once per challenger and piece. "Used" counts citations and building on a piece (builds_on). | explore.py `_test`, `_plan`, `_guide`; directions.py | Refuting E193 (9 users) pays the most, at the time, with steps; today it pays nothing |
| `change_credit=1` with directions: the first test of a new idea filed under a direction pays the idea credit × (2 − the share of the game's agents on that direction) | explore.py `_plan`; directions.py | An idea on a line nobody else holds pays nearly double |
| A new claim says which of the agent's open plans it dropped untested (always on) | explore.py `_plan` | Persistence |
| The direction pot splits by ticks + 3 × claims tested; posts of smaller directions reach everyone first (the main session's points) | directions.py | Pays testing over presence; the view counterweighs the herd |
| trust=1 in the run | launch option | E193 would show "seen in 1 state" until someone else reproduced or contested it |

`tools/launch_vc33_depth.sh` takes `DIRECTIONS=1 TRUST=1 CHANGE_CREDIT=1`. Tests: directions 9 (with change credit on the DiG fake server, the counter mask on vc33, and the dropped-claim notice); all 13 suites pass. A mock rehearsal of J1's configuration on vc33 (6 agents, 4 ticks) ran end to end.

**Proposed, not built.**
- Ideas in closing notes enter IDEAS (A7's E118 never did).
- A guide line that interprets receipts is marked separately from what its receipts show.
- The step bar named in `look`: the main session's I9.
- Personas or mixed models for diversity of agents, which GOAL.md would count as assignment unless offered.

**The run.** J1 is I8's exact world options plus directions=1, trust=1, change_credit=1 and the fix: 20 agents, 20 ticks, start l5x, $8 cap. It changes several things at once. The main session's I9 is I8 plus the fix (and the step bar named in `look`). Comparing J1 with I9 separates the social and credit features from the fix.

### J1 results (2026-11-03): level 5 fell for the first time

`runs_arc/J1_vc33_20_s0` is I8's settings plus the novelty fix, directions=1, trust=1, change_credit=1 and the dropped-claim notice: 20 agents, 20 ticks, start l5x, $4.96. The machine slept during tick 9. Tick 10 lost 17 of 20 turns, and tick 11 lost 17 of 20 to connection errors, so J1 had about 18 working ticks to I8's 20.

| vc33 level 5 | I8 | J1 |
|---|---|---|
| Level 5 | not solved | **solved at t20 by A15** (score 0.714) |
| Posts reading row 0 as the exit or goal | 51 by 16 agents | 3 by 3 agents |
| Guide lines making row 0 the exit | 5 (E193 first, t8) | 0 |
| Credit for new states (share of all credit) | 97% (1,201 awards) | 74% (404) |
| States new only in row 0 that were paid | 56% of paid states | 0 (the fix; 56% of the states the old rule would count) |
| Level-5 claims / never tested | 42 / 32 | 42 / 30 (24 still on level 5 at the end, 18 open ones carried to level 6 at t20) |
| Claim similarity across agents | 0.197 | 0.201 |
| Moves played | 1,987 | 1,529 |
| Directions; agents joining; peak share | — | 1 (opened t1, generic: "L5 = L4's machine rotated"); 3; 0.2 |
| Change-credit payments; successful tests | — | 0; 0 |
| Claims refused for carrying a test result | 0 | 26 |
| Cost | $7.86 | $4.96 |

**How it fell: the pieces were assembled.** A15's winning play cited pieces by four agents, and understanding credit paid them:
- A1's pump map (E118) and door rule (E136, with E150 and E152), pieces 1 and 2 of the source's four;
- A17's discovery that a lit door swaps the boats' cargo (E231), piece 3;
- A5's layout (E57).

A15 supplied piece 4, the goal, and held it for six ticks:
- t15, P31: the bb and ee holes in the walls are "landing slots".
- t17: "those slots sit in regions no copy has ever clicked. Testing them first", from the untried-regions view and L4's explanation (E5/E6).
- t18, P37: the boat swap alone did not finish the level (R213), so the goal is a position.
- t19: each boat's hull right edge must equal the token's column, b at 21 and e at 15.
- t20: a pump into a full tank changes 0 cells (R235), so tank C had to be drained first. Then both boats docked.

This is the goal A7 stated at t5 in I8 and never tested. Here one agent tested it to a result. Direction D1's pot (3 credit) went to its four members by ticks, because a piece two steps behind the solver's citations was filed under it.

**What the new layer did and did not do.**
- Directions were barely used: one generic direction, 3 joins, 12% of claims filed under it. Agents organized in groups instead, two of five.
- `change_credit` paid nothing: no test succeeded (2 `plan op=test` calls, both refused) and nobody challenged a finding or a guide line, so its hook never had a call to pay.
- trust=1's `result` field produced 26 refused claims and no test. Agents sent `plan op=claim builds_on=[E5] result=holds` with no hypothesis, meaning "E5 holds", the same trap as the DiG runs. The trust tags were shown, but there was no decoy line for them to mark.
- The novelty fix worked as built: credited new states match the states new outside row 0.

**Reading (one seed).**
- J1 solved what I8 did not, at its last tick. The decoy cascade never started: no agent made row 0 the exit in the guide.
- Whether that is the fix (budget-burning was no longer paid, or bought extra steps), trust tags, the claim notice, or chance cannot be told from one run. Directions and change credit were inert, so they did not cause it.
- I9 (I8 plus the fix only) is the control the main session proposed.
- What did work is GOAL.md's jigsaw: four agents' pieces, kept and cited, assembled by one agent who stayed on the missing piece.

**Next.**
- Fix the test trap: a claim with result and builds_on but no hypothesis should become a test of what it names, or be answered with the exact call to make.
- Directions need a reason to be used. The group prompt (teaming) competes with them, and only one generic direction was opened.
- I9 as the control.

### I9: the control for J1 (2026-11-03)

The analysis is in section 17, "I9" (the main session's write-up, which includes the fork's points). In short:
- The novelty fix contained the decoy in both fixed runs (I9 and J1), against 5 of 5 unfixed level-5 runs. It most likely did so by removing the extra steps that bar-only clicks used to earn.
- On its own the fix did not solve level 5. I9 saw the cargo swap late (t17) and never named it. It treated the wall tokens as the goal rarely (5 texts by 4 agents) where J1 did so often (26 by 12).
- J1's added layer went unused, so its solve most likely came from the swap and the goal surfacing early, plus one agent persisting. One seed each.

**Counting note.** Event logs cut long move lists short: 99 of I8's 339 plays end in "…", and the parseable lists hold at most 7 moves. So count moves from receipts (I8 1,987 executed moves, 99 per agent; I9 1,554, 78; J1 1,529, 76), not from play arguments (I8 897 clicks, I9 710, J1 641).

**Next, for the $16.45 left (the user decides).**
- Make the test interface robust, then rerun J1's settings. J1's 26 refused claims tried to record "E5 holds"; a claim with a result and builds_on but no hypothesis should become that test.
- Second seeds of J1 and I9, about $13.
- A 30-tick run of I9's settings, about $10.
- The main session's engine note that names a swap when two objects exchange places (section 17).

## 19. A society that rebuilds a program (world `code`, `--reflect`, 2026-11-03)

The user's frame for this round: given enough primitives (agents can add their own) and incentives, an efficient collective and an emergent structure form. The only goal is accuracy, and any level of reflection and change is allowed, credit assignment included. There is a large action space with a guide, and credit assignment promotes the right behaviour. Three goals are worth trying: individual, team and society wealth.

### What was built

`alphateam/code_world.py` puts ten agents on ProgramBench xsv with te-next's evidence layer (reference behind a probe meter, regression gate, held-out set, hidden grader), so scores compare directly with the ten-agent baseline in `~/asp-runs` (hidden 736-845 of 1182, mean 780.7). It is an `Explore` world, so plans, trust, the guide, leads, contracts and substrate are the same code as on ARC and DiG.

| The user's list | In this world |
|---|---|
| Primitive tools | The kernel's and Explore's, plus shell, probe, publish, head, own, toolbox, pay, role, trace |
| Boards, public and private | `private=1`: private and group boards under the one public board, on which every post reaches everyone once |
| Trust per finding | said, seen in k inputs, reproduced, contested. The credit a piece earns scales with its level |
| Credit engine | Paid only for verified progress at HEAD. Shares are paid out of a merge's value, never minted |
| Hierarchy | Groups nest, leads get a royalty, a group can hold code, members read each other's copies. Measured, not built |
| Reflection | `alphateam/reflection.py`: judge, reflect, propose, enforce |

The credit engine. A merge is worth `gain` per logged check it newly passes plus `held_out` per held-out check HEAD newly passes (only a new best counts, and nobody sees those checks). Out of that value: the authors of the probes behind the checks, the pieces it used (cited, built on, or read in full in the last three ticks, weighted by trust, with a cut for whoever tested them), the authors of tools it used, and the owner who accepted it. Whoever landed it keeps the rest, at least a tenth. Findings, tests, claims and posts earn nothing when made: they are paid only when something verified rests on them. On ARC 75-98% of credit paid for novelty, which is activity; this engine has no such term.

The law. Everything a review may change is in one bounded dictionary (`LAW` in `code_world.py`): the weights, the shares, the trust weights, steps, ownership terms, the gate's order, the fine rate and the goal. A review may also change what each turn shows and the practices, which are offered text. `goal` decides whose wealth a payment is: the payee's, its team's (the widest group it is in, equally), or the society's (everyone, equally). The engine keeps the attribution either way (`attributed`), so a flat arm can still be read. What a review can never change is the accuracy score and the facts credit is computed from.

The review. Every W ticks: if the last review changed something and the held-out gain since is below `floor` times the gain before it, the change is reverted; each agent writes a short reflection from measured facts; an organizer (the same model, acting for the society) reads the reflections, agents' `substrate op=propose` posts and the record of earlier changes, and returns any change to the rules; the harness enforces it and posts the review. It never says who does what: it cannot put an agent in a team or a role, a practice has agent and group ids stripped, and what it sees about teams it can only post as a suggestion. Teams, roles and leads stay the agents' own choices, and credit is how a review makes them worth choosing. An unbuilt HEAD is no signal, never a zero.

Generations. `--episodes N` runs the lineage code that already existed: each society starts from no program code, with the law, practices, roles and tools of the one before, plus what it established (the guide and its dead ends), as GOAL.md asks. Program code is never carried. `heritage=org` carries how the society ran and nothing it learned about the program, so that a lineage's score measures organization alone; a run with `heritage=all` has more knowledge than the baseline and cannot be compared with it directly.

### What Avalon-M1 changed (General Reasoning, 2026, read in full)

Avalon trains one 120B model by policy gradient as organisations of up to nine agents, a CEO plus hired specialists, with one flat collective reward on every token. Its lessons, in AlphaTeam's terms. Ideas are adapted as offered, evolvable elements; no mechanism or text is taken over.

1. **Organizing is learned; it is not there by default.** The same harness lowered untrained models' scores (mean -2.4 and -6.6 points against their own single agent), and on ProgramBench took Nemotron from 13.3% to 4.0%. Lead Engineers were under 0.4% of hires for the first 80 training steps, then rose to 12.5%. So one run of an untrained model is the wrong place to expect a society. AlphaTeam has no gradient, so its learning has to live in the review and the lineage. This is why generations are in the plan.
2. **ProgramBench is the paper's worst case for organisations.** Even trained: single 29.4%, two hires 29.4%, four 25.2%, eight 25.7%. Scores flatten fast (91-95% of the gain in the first hour) and agents stop early. Code output grows as N^0.31 on fixed-specification tasks against N^0.73 on open-ended ones. xsv is a fixed specification, so it is a hard test of the thesis: losing to one agent at matched compute would repeat the paper, not refute the design.
3. **What an organisation buys is mostly latency.** At matched output tokens an eight-hire organisation and an extended single agent were comparable. The time limit is what gives coordination a route to reward. So runs are compared on rounds as well as on compute (`tools/xsv_compare.py`): ten agents have a tenth of the rounds a single agent uses for the same replies.
4. **A flat reward was enough, with a very large amount of training.** The paper presents it as a baseline and names the variance it causes (the "uneasy triangle" of long episodes, many agents and many environments). In AlphaTeam credit acts only through what agents are told and what it buys, so `goal=society` is the flat arm and attribution is a real alternative, a comparison the paper did not make.
5. **The structure that won was the surgical team, not the split by component.** Opening plans split by component fell from 71% to 31% while build/test/review pipelines rose from 16% to 54%. Leads were builders, with twice the production calls and 6.5% of their calls on coordination against 31% for others; specialists read a lead's trace 15 times as often as a peer's. A lead is the reference, not a summarizer. So `own_cap` lets one holder hold everything, proposals are the pipeline, `trace` is open to all, and the measures are top-builder share and who reads whom. The brief lists this among several ways to organize and recommends none.
6. **Review grew, and concrete reviews were the useful ones:** a counterexample, a recomputed number, a named bug. Credit reaches a tester or a diagnosis only when a fix lands on it (`piece_share`, `tester_share`).
7. **Coordination has an optimum:** about 5-15% of working time for small teams and 15-20% for large ones, with both extremes below average. AlphaTeam's ARC runs spent about 0.1% organizing. The review sees `organizing_share` as a fact; the paper's number is not given to it, so that it decides from its own evidence.
8. **Cooperation unravels without enforcement.** Public-goods contributions fell from 81% to 31-47% without punishment and held at 94-97% with it. Team and society goals invite free riding, so `pay op=fine` exists (it costs the one who fines) and contracts hold pay in escrow.
9. **Every visible channel leaks, and false zeros teach wrongly.** They merged 305 fixes across 114 environments. Lookup tables beat honest work 1,247 to 289, and constant answers banked 75% of a score. Here held-out checks pay more than logged ones (a lookup table of probes earns logged gains only), the review sees logged against verified gains, only counts leave the grader, and an unbuilt HEAD is no signal.
10. **Their open question is our gap.** They name the CEO as imposed structure and suggest a less opinionated start might let agents organize themselves. AlphaTeam has no CEO. Whether that works is what these runs test.

Not adopted, for now: subscription levels on channels (the board delivers every post once, by the user's rule), post receipts, reactions, a required opening plan, and the penalty they use to force hiring in training.

### The prompts checked against GOAL.md (2026-11-03, at the user's request)

The first version broke GOAL.md in several places. Each line below is what the text asks and what the prompts and the world now do.

| GOAL.md | Before the check | Now |
|---|---|---|
| Structure arises from agents' own choices; design the institutions and the learning loop, not the organization; nobody assigns roles | The organizer could place agents in teams and give them roles, and the system prompt said nobody assigns them | A review changes rules only. Practices have ids stripped, and teams come up only as suggestions |
| Work that unblocks others comes first | The gate went by credit | `offer_order=unblocks` by default: by how many modules at HEAD import what an offer changes, credit breaking ties. Each turn lists the connecting pieces; failing features that connect many others are starred |
| Credit tells agents which actions and combinations paid off; the society must see which practices worked | The views were removed from this world | WHAT HAS PAID and ACTIONS NOBODY HAS USED YET each turn; both are in the facts for reflection, and under shared goals the ledger follows the earner |
| Diversity from information: what nobody has tried yet | No view | NOT TRIED YET (options no logged observation has used), logged checks per feature, and what is in other copies |
| Nothing is lost; work on branches that did not win stays visible | Kept only in the agent's copy | Rejected and conflicting work is kept at `/public/unmerged`, for everyone |
| Conflicts are visible; the parties settle them directly | Only the later agent was told, without the other's name | Both parties are told, by name. Each turn shows overlapping files across copies |
| Evidence decides, not rank | An owner could turn a proposal down by rank | `own op=reject` needs receipts |
| Whoever contributes later reconciles; responsibility moves explicitly | Not said | In the brief; `own op=pass` |
| A result carries its understanding | `note` was "what this change does", 200 characters | Why it works and what was ruled out, 500 characters, delivered with the merge |
| The picture: progress is what is assembled | Not said | THE PICTURE in the brief, in GOAL.md's terms; the review counts files never assembled |
| Across runs a society starts from what earlier societies established | Only law, norms and tools | The guide and dead ends too, unless `heritage=org` |
| The working cycle is a view the rules can show or hide | Listed in the views, with no effect here | Shown by default in this world's words; a review can hide it |
| Every view lists everything (the user's rule) | Several `[:n]` cuts | Lists are whole and rows are shortened past a budget; gate notes reach their agent once |
| Papers are inspiration, not transplants | The organizer's prompt carried two of Avalon's findings as priors, one of them for a central builder | Removed: the organizer sees measured facts only |

Predictions, written before any society ran:
- Early on a single agent is well ahead. It already is: 43-53 of 179 held-out at 120 actions, against 0-19 for the ten-agent baseline.
- The first society will not show a surgical team unless its own practices lead there.
- If the review helps, it shows first as fewer overlapping merges and more of the probe budget used (the baseline used 25-36 of 480).
- `goal=society` will show less hoarding and less effort on credit, with more free riding once the skeleton exists.

### After a review of the design (2026-11-03, pasted by the user), and the first wave

The review's question goes further than Avalon's: can agents with equal rights discover which decisions are worth delegating, form layers, and reorganize when the task changes? Its checkable claims held against the code. What changed:

- **Organizational examples are an arm, not the default** (`ways=1`). Tool text explains permissions, costs and effects. `lead` no longer says "like a CEO or PI". The user asked for a guide to the action space, and that stays; the list of structures is what moved.
- **Delegation is one primitive that can carry a decision.** `delegate` now takes scope, acceptance criteria, an end tick and pay held until accepted. It goes to an agent or a group, can be refused, handed on within its scope (recursion through `parent`), or revoked, and what was handed on ends with it. In the code world a scope means the taker holds those files while the task lives; holders alone can hand over what they hold, and it returns when the task ends. `contract` is no longer offered, since delegate with pay does the same and can be handed on.
- **Every delegation is on the record:** `task_log`, a funnel (offered, seen, taken, delivered, accepted, chosen again, depth), and merges linked to the task they landed under. A zero now says at which step delegation stops. Open tasks are listed each turn; they used to be a count.
- **Position pays nothing by default** (`lead_share=0`), under every goal. Fees are for agents to negotiate.
- **A team's income follows a group its members chose** (`role team=G1`, or else their smallest group), so an umbrella group changes nobody's income.
- **A closing note stays private under `private=1`.** Publishing is a choice.
- **Credit and review are separate switches:** arms `fixed` (credit, no review) and `review` (review, no credit) join `plain`.

Not done yet: the three design-world variants with their five controls, interventions from a checkpoint, the three overlaid graphs, trust that separates reliability from scope of application.

First wave X1 (xsv, ten agents, 24 ticks, three replies a tick, one seed each): hidden 514 (individual), 416 (plain: no credit, no review), 404 (team), 367 (society), against c0's 781 and earlier single-agent 524. One seed cannot separate the arms: c0's own runs spread from 736 to 845. No group formed in any arm. The gap is the harness: te-next assigns nothing either, but it lands work continuously, rebases every copy after each merge and offers what is left at the end. AlphaTeam batched landing at tick end, so ten agents collided (team arm: 18 merges, 25 offers lost to conflicts, 428 lines at HEAD against 1,352). X2 uses 36 ticks of two replies; `stream=1` lands an offer at once, with copies rebased before their next command, and the last tick offers what is left in each copy.

FrontierScience pilot (`alphateam/science_world.py`). Paying for correct answers would make credit an oracle, so a random sample of questions, unknown to agents, is audited, and the score counts only the rest. Attempts stay hidden until an agent posts its own, so agreement between agents who worked blind is evidence; a copy reproduces nothing. The user asked that a task leave a single agent room: `tools/screen_science.py` tried every question three times in one shot, and 38 of 100 olympiad questions were solved at most once while 57 of 59 research questions never passed. The pilot uses 38 and 22 of those, with a single-agent budget sweep at one and three times the replies.


### Results and what they change (2026-11-04)

**Design world.** Every run through D3 had a broken post search, so those society numbers are withdrawn; DESIGN.md's earlier reading of them should not be cited. With search fixed the review's three worlds gave this. Independent: everyone reaches .9, nothing to organize. Coupled: a single agent wins, .81 and .92, against at best .07 for any 24-agent arm. Moving: flat rights are the only arm that is good on both seeds, .41 and .48. In all of them `free` built no structure: no groups, no delegated tasks, almost no messages.

Size changed everything. Eight agents with the same total replies and simulations, one agent per block, scored .74, .79, .67 under flat rights and -.15, .41, .73 free, in about a tenth of a single agent's time. Twenty-four agents had each spent a third of their replies on end-of-tick notes and two more on one block; eight agents lose a ninth and have room to think. So the unit that worked is a small group of deep agents, each alone on its piece, which is Avalon's surgical-team finding in another form.

Under flat rights the structure that matters was not in the rules. A holder is the only one who can move a knob, so everyone else must persuade it, and they do it with numbers: 85% of the messages went to holders, and holders added costs across blocks. That is information aggregation around a right, and nobody was told to do it. Free agents, with the same tool in hand, almost never claimed a knob, and a measured fact about which knobs keep moving did not change that. The rights came from the harness, so this is not yet the thesis. The open test is whether a society adopts the rule by itself: a review that can change rules, on a world where the review's evidence shows the fights.

**xsv.** Societies nearly level off near 700 (546, 656, 697 at 720, 1,440, 2,880 replies), below c0's 781 and level with a single agent's 644 at 1,440, in about a third of the wall clock. The plateau is not a coordination failure. Agents fix what their logged observations show, the logged mismatches run out, and nothing shows what has not been observed: coverage counts each option once, and 1,685 of 1,920 probes went unused. Credit pays for verified gains and nothing pays for finding a new mismatch. That is the next change to try: finer coverage, and credit for the probe that exposes one.

Delegation did not take on either world: 11 posted, 5 taken, 0 delivered across eight xsv runs. The two groups were one self-selected trio and one all-hands list. Refusals now say what went wrong, since agents kept reaching for `delegate` to claim plans.

On the bigger world (16 blocks, same 1,728 replies) a single agent stops saturating: .47 and slowing. Flat with 18 agents reached .44, still rising, in under 5 minutes against 71. On the moving world a single agent got 0 and .71, flat .63 and .45 with 8 agents. One or two seeds each, and seeds on this world can differ by more than a point.

Wall clock belongs in every comparison. It is what a society buys.

## 20. Recipes that evolve: the rules of the game as the thing selected (2026-11-09)

Proposal: [AlphaTeam: agent societies that evolve how they work together](https://claude.ai/artifact/AKJJWhguK5ANvPtuqShD2M). Code: `alphateam/recipe.py`, `archive.py`, `stats.py`, `paired.py`, `selection.py`, `generations.py`, `trial.py`; the optional `rules` action (`kernel.py`, `primitives.py`); the chaired arm (`institutions.py`, world option `chair=1`) and the hierarchy arm (the oracle condition with authority), both comparison arms; a run taken in parts (`runner.py`: `start`, `horizon`, `final`); backends that can be copied (`mock.py`, `llm.py`: `fork`). Tools: `tools/recipe_evolve.py`, `scaling_grid.py`, `selection_power.py`. Tests: `tests/test_recipe.py`, `test_paired.py`, `test_evolution.py`, `test_live_path.py` (42 tests, all offline).

### What was missing

`alphateam/evolve.py` passes the best society's culture to the next generation: one final score a society, no comparison of a change with what it changed, nothing typed. The review loops (section 19) change rules inside a run and judge by one noisy window. Neither says what a change was worth. The user's framing of 2026-11-09: be like AlphaEvolve, and evolve the collaboration strategies and substrates; give agents primitives and let them explore; keep evolving the recipe; aim at a recipe that is scalable and robust. This follows the Bitter Lesson: build in primitives and a score, and let search find the structure.

### The recipe

A recipe is the rules a society starts from. It has five parts. The system enforces four; the fifth is advice.

| Part | What it sets | Design-world settings (`recipe.DESIGN`) |
|---|---|---|
| Rights | who may do what | `holders` (none, shared, all), `claims` (off, agents, groups), `claim_price`, `claim_rent`, `rights_by_credit` |
| Information | who sees what | `messages`, `delivery` (top, all), `delivered`, `moving_facts`, `memory_ticks` |
| Incentives | who is paid for what | `shared_pay`, `lead_pay`, `group_pay`, `credit_every` |
| Procedures | which steps exist | `take_lead` (off, scope, team), `contracts`, `review_every`, designed actions carried over |
| Advice | what agents are told | lessons: text, when, evidence, status (untested, confirmed, refuted) |

Seventeen settings. (`evidence_share` was in the first version and was taken out after review: with `economy=1`, which every arm runs with, it changes nothing.)

- **Rules, not assignments.** A recipe never names an agent or a group. Lessons and designed actions have agent ids, group ids and the names of one task instance (knobs, blocks) taken out, and `validate` refuses text that still has one. So leads and teams form anew in every run, and one recipe runs at any team size. This is the rule of 2026-11-03 (a review changes rules only, never who does what) made a property of the data.
- **Typed edits.** A revision is a list of edits: `set` a setting within its bounds, `advise` or `unadvise` a lesson, `adopt` or `unadopt` a designed action. An edit the registry does not allow is refused with what is allowed. A recipe's id is a hash of its content, so the same rules reached by two paths are one recipe, and a dead end is not tried twice.
- **The frame never evolves.** The world, the budget (agents, ticks, replies, simulations), the model and the score are the frame (`paired.frame`). A recipe compiles to world options only (`recipe.options`), and a frame can lock a setting (`locked`).
- **Free checks.** `inert` catches a revision that changes nothing or only settings that have no effect (a claim price while claims are off).

### How a revision is tested (`selection.Ladder`)

| Stage | What it checks | Cost |
|---|---|---|
| typed | a valid edit of its parent, names nobody, not tried before | free |
| dry | changes something; scripted agents can run it for a few ticks | free |
| probe | a short live run: did the mix of actions move more than it moves between two runs of the parent? Required of a revision that changes only advice | cents |
| screen | a few pairs on selection instances; the progress score ranks | dollars |
| confirm | fresh pairs on new instances; kept when the lower bound of the mean paired gain is above zero; given up at a look when the mean is at or below zero | most of the cost |
| archive | kept with lineage and credit; a dropped revision stays on record | |

Defaults (`selection.Policy`): 4 screening pairs, 1 finalist, up to 30 confirming pairs with looks at 10, 20 and 30, a one-sided 90% bound, 3 runs before a recipe seeds a generation. At least two confirming pairs are required (one pair gives no bound). A revision that passed the screen while another went on in its place was not judged: it stays a `candidate` and can be taken up again (`Ladder._wait`); only a judged revision becomes a dead end.

Two controls use the same loop. `mode="random"`: one valid, runnable revision a generation is drawn at random and kept with a fixed chance, and nothing is scored; no lesson is marked by it. `generations.evolve(agents=False)` (`--menu-only`): the same ladder and budget with every candidate drawn from the menu of edits, which shows what agents' own proposals add over random typed edits.

Parents are drawn by standing (`archive.sample_parent`): the lower bound of what a recipe scored, so one lucky run does not make a leader; less for a recipe that already has children; sometimes at random. Selection instances and report instances come from disjoint seed ranges (`paired.instance`), so what a result is reported on was never used to select. A recipe's standing counts only its runs from scratch on selection instances (`archive.FRESH`: generation, confirm, retest): report runs, screens, probes and runs that tried changes on themselves do not move it.

An archive belongs to one niche and one frame (`Archive.bind`, else `ArchiveMismatch`): scores from another team size, budget or model would not compare. Its instance counters are on disk (`Archive.next`), so a later session never redraws an instance, and a runner never reuses a run directory.

### Pairs and branches (`paired.py`)

- `pair`: two fresh societies on one instance with one budget, one under each recipe.
- `branches`: one society under the parent to a tick, then copied (`fork`): one copy carries on, each other copy switches to a child (`recipe.apply_live`). Tested property: a copy that keeps its parent's rules reproduces the unbroken run exactly, so two branches differ only by the change. The parent's branch carries the cost of the shared ticks and each child its own, so the branches' costs add up to what was billed.
- What can change mid-run: every setting except `claims`, `take_lead` and `contracts`, which change the tools on offer and apply from the start of a run. Those are screened with fresh runs.
- Only a world whose state is all in memory can be copied (`forkable`; the design world is).
- The progress score is the mean score over a run's ticks (`stats.area`). On the saved runs it varies less than the last tick's score (below).
- Arms (`paired.frame`): `open`, `chaired`, `hierarchy`, `single`, `alone`. Budgets are per-agent replies and simulations a tick. Every arm runs with the same memory, review and credit options (`EVERY_ARM`), so arms differ in who is in charge and who can talk, not in how much an agent remembers or how often the score is posted. Blocks default to one an agent (the task grows with the team); fewer blocks than agents shares one task among more agents; more blocks than agents is refused.
- A run that stopped early (its budget ran out, or the provider refused) has no score: the runner raises `RunStopped` and nothing is recorded for it.
- The spending cap (`AlphaTeamRunner.cap_usd`) counts every run, the shared ticks of branches and the dropped copy of a trial included. No run starts at the cap; each run started under it gets an equal share of what is left as its own budget, checked before each agent's turn; a run whose provider reports no cost is counted at an estimate (`est_cost`).

### Trials inside one run (`trial.py`)

Every `every` ticks, the most backed proposal with at least `backers` backers that can change mid-run is tried on a copy of the society for `window` ticks. The society carries on as whichever copy scored higher, and the run ends `window` ticks sooner for each trial, so a society that tries changes pays for them out of the same budget. What the losing copy's agents did in the window is dropped: this is the price of a counterfactual, and it bends GOAL.md's "nothing is lost" for those ticks. Agents are told (the TRIALS line in the brief). A change a society kept this way becomes the first candidate of its generation and still has to be confirmed.

### The `rules` action and the generation loop

Under a recipe, agents have one more action (never without one, so earlier worlds are untouched): `rules op=view | propose | back | lesson`. A proposal names one setting and a value and says why, with entry ids as evidence; the same proposal made twice counts as backing. `generations.evolve` runs a society under the parent and makes candidates in this order: what the society kept after a trial; proposals pooled over every run under the parent (those that came up in most runs first, then the most backed); a lesson; an action its agents designed and at least two adopted (an `adopt` edit); and, if those are too few, random edits from the menu (marked `by: menu`). It gives them to the ladder. What agents propose and the lessons they leave are filed from every run under a recipe (`Archive.record_voices`), not only from generation runs. A lab note about a trial is a record, not advice. Scripted agents never propose, so offline every candidate comes from the menu.

### The comparison arms: chaired and hierarchy

**Chaired** (`chair=1`, frame arm `chaired`). Before the run begins, the harness founds one team with every agent in it and makes one agent, drawn with the task instance, its lead. The team holds every knob that several blocks share, and only the lead can set one or hand it over with a task; a member sets the knobs of its own block. The arm locks `take_lead=team`, `holders=none` and `rights_by_credit=False`, and the leave shock takes the chair. It is an imposed coordinator, to compare with societies where nobody is placed in charge, and it is labelled imposed in every measure.

**Hierarchy** (frame arm `hierarchy`). The oracle condition, teams over the task's hidden modules, with authority: each team's lead holds its knobs. This is what the saved "ceo" runs were: a prescribed hierarchy, not a chair.

In the chaired arm the system prompt says what was set (its line that nobody assigns leaders would be false).

### What other systems do, and what was taken (ideas, not text or mechanisms)

| Read (code, 2026-11-09) | What they do | AlphaTeam, in its own terms |
|---|---|---|
| OpenEvolve, ShinkaEvolve, Darwin Gödel Machine | one record a candidate with parent and generation; parents weighted by score and by how few children they have; a cascade of cheap tests first. A child enters the archive without its parent being re-run; evaluations are not repeated by default; DGM allows a fixed tolerance of 0.1 | `archive` records and lineage; `sample_parent` on the lower bound with a child count; the ladder; every comparison paired; leaders re-run; a failed revision never seeds |
| EvoTeam | a library of teams (rosters), rules per team, one forced decision with a validator that feeds errors back, copy-on-write versions, an append-only changelog, evolve then freeze then evaluate on a holdout; a fixed chairman | rules instead of rosters; edits refused with what is allowed; `decisions.jsonl`; selection and report instances kept apart; the chairman only as the control arm |
| EAT (Evolving Agent Teams) | a supervisor writes the team's roles and phases each iteration; the score comes from an eval the harness runs, one a round; a budget table | no supervisor: agents propose, backers rank, pairs decide; the engine's score; a spending cap in the runner |
| CORAL | attempts, notes and skills as files; notes with a claim, evidence, a status and what they supersede; reminders on an interval and on a plateau, with the plateau threshold at the noise floor | lessons with when, evidence and status; dead ends kept; trials on a beat |
| CrewAI (hierarchical) | a manager delegates with tools and every result returns to its context | the reason to measure the chaired arm against team size |

### The texts agents see, checked against GOAL.md

| Text | Where | Check |
|---|---|---|
| The `rules` tool description | `recipe.RULES_TOOL` | Says what each op does and how a proposal is judged. No advice to use it. Offered only under a recipe |
| The rules view | `Rulebook.view` | Every setting with its value, allowed values and one line of what it sets; proposals with backers; trials with both scores. Facts and options, no recommendation; nothing cut |
| Lessons in the brief | `recipe.heritage` | Earlier agents' own words, each with its status, and what the statuses mean. Within a size budget; the rest are posted on the global board, so each reaches every agent once |
| Proposal and lesson posts | `Rulebook.act` | The agent's own words under a fixed first line |
| The TRIALS line and a trial's result | `trial.py` | The rule and its cost, then the two measured scores and which copy carries on |
| The chaired arm's line | `institutions._institutions_brief`, `primitives.system_prompt` | A control that assigns a lead on purpose; the prompt says what was set, so it stays true |

| GOAL.md | Here |
|---|---|
| Institutions are offered, never required | settings switch offers on and off; nothing in a recipe can assign |
| The society tries, measures, keeps or reverts its practices | the ladder across runs; trials inside a run |
| A society starts from what earlier societies established | the archive: rules, lessons, designed actions, dead ends |
| Evidence decides, not rank or votes | backers only order what is tried first; the paired gain decides |
| Credit must have consequences | a revision's credit decides what later societies start from |
| Nothing an agent establishes is lost | dropped revisions stay on record. Open: a trial's losing copy loses the work of its window |
| Coordination has real costs | viewing the rules spends reading budget; a trial spends ticks |

### What the saved runs and a simulation say (no paid runs)

`tools/selection_power.py`, on the valid design runs (series D6 to D10):
- Runs that differ only by seed: the final score has a pooled standard deviation of 0.53 (14 degrees of freedom; 0.36 without the two CEO runs at -1.51 and 0.61), the progress score 0.30 (0.26).
- At a spread of 0.53 the default ladder keeps a useless revision about 5% of the time. It keeps a true gain of 0.4 in 89-97% of cases (about 61 runs), 0.3 in 72-89%, and 0.2 in 44-67% (about 49 runs), depending on how much of the noise a pair shares (0 to 0.6, which saved runs cannot tell). So at this spread selection finds effects the size of the holder rule (0.73 against 0.33 at eight agents) and misses many small ones.
- A generation of four useless candidates ends with one of them kept about 7% of the time (the best at the screen goes on to be confirmed).
- A positive control of 20 pairs detects a gain of 0.4 in 85-99% of cases, over the same range of shared noise.
- With scripted agents the probe separates the holder rule (the mix of actions moves by 0.06-0.09) from two runs of the same rules (under 0.005) in three ticks.

### Not built, and open

- An outside proposer that reads many runs' lessons. Candidates come from agents and from the menu.
- Registries for the code and science worlds. Their `law` already has bounds; a registry entry would map it.
- Tool changes mid-run (`claims`, `take_lead`, `contracts`).
- The reflect and consolidate cadence as settings; lesson pruning by ablation.
- Team sizes: by default the task grows with the team (one block an agent), because an agent simulates one block. `tools/scaling_grid.py --fixed-blocks` shares one design among more agents (several agents a block) at the same total replies and simulations a tick, which is the case the 8-to-24 failure raised. Untested with models.
- Carrying a trial's losing copy's posts over to the winner.
- Planted wrong results, as a robustness test.
- Nothing here has run with models. The comparison arms, the probe thresholds and the ladder's defaults are untested on live societies.

### The review of 2026-11-09 and what it changed

Two independent reviews, one of the proposal and one of the code, ran after the first build. The code review confirmed that worlds without a recipe behave as before (identical results on 22 runs), that a copied society shares no mutable state with its original, and that no path spends money without `--backend openrouter --live` and a key. What it found, all fixed and tested:

| Found | Now |
|---|---|
| A run the provider stopped was scored as a real run | `RunStopped`; nothing recorded |
| The single and alone arms ran with no memory and rarer reviews | every arm gets the same options (`EVERY_ARM`) |
| The cap missed the shared ticks of branches and the dropped copy of a trial, and was checked only between stages | all counted; a share of the cap for each run; an estimate when the provider reports no cost |
| Candidates could be stranded, and a second session reused directories and instances | candidates are retaken; counters on disk; directories never reused |
| One archive could mix niches and frames, and report runs moved a recipe's standing | `bind` and `ArchiveMismatch`; standing from runs from scratch only |
| `evidence_share` did nothing; the chaired arm's settings contradicted each other; the brief was silent or wrong on four settings | removed; locked; `_rules_brief` |
| The rules action was read only in generation runs | voices filed from every run |
| The chaired arm was weaker than described, and the saved "ceo" runs were a hierarchy | the chair holds every shared knob; a separate `hierarchy` arm |
| The positive control was not the flat arm of the November runs | holders=shared, claims off, take_lead off, 20 pairs |
| A revision that was never judged became a dead end | it stays a candidate |
| Nothing made `adopt` edits | an action two agents adopted becomes a candidate |

The proposal review's points went into the proposal: the budget is counted in replies and simulations (a single agent cost about three times the dollars and ten times the minutes at equal replies), the experiment has three arms and two ways of growing a team, every claim has a numeric test, and what is built is told apart from what is planned.

## 21. Recipes on the code world: evolving the rules of a society that rebuilds xsv (2026-11-09)

The user gave $50 to try the recipe loop of section 20 on ProgramBench xsv with ten agents, and asked that the
experiment be watched, stopped early when it does not look promising, fixed and restarted. Section 20's machinery was
built on the design world, where a society lives in memory, can be copied mid-run and takes minutes. A code-world
society keeps its state in Docker containers and git copies, takes about 35 minutes and $0.57-0.83 for 720 replies,
and cannot be copied. This section says what changed for it, and what was checked before money was spent.

### What a code-world recipe may set (`alphateam/recipe.py` CODE)

Fourteen settings, each an option or a law value the world already had. The defaults are the plain arm of
`tools/launch_xsv.sh` (run X1_plain): every action on offer, no credit, no review, work landing at the end of a tick.

| Part | Setting | Values | Default | What it sets |
|---|---|---|---|---|
| rights | `own` | on, off | on | whether the own action is offered (hold file patterns; a change to a held file waits for its holder) |
| rights | `own_ticks` | 2-12 | 6 | idle ticks after which a claim lapses |
| information | `messages` | on, off | on | whether the message action is offered |
| information | `private` | on, off | on | boards of one's own and of groups beside the public board |
| information | `teaming` | on, off | on | whether each turn shows what working in a group gives |
| information | `memory_ticks` | 0-4 | 2 | ticks of its own conversation an agent keeps |
| incentives | `credit` | on, off | off | whether verified progress at HEAD is paid; credit buys steps and goes first at the gate |
| incentives | `goal` | individual, team, society | individual | whose wealth a payment is (idle while credit is off) |
| incentives | `piece_share` | 0-0.6 | 0.25 | share of a merge's value paid to the pieces it used (idle while credit is off) |
| procedures | `stream` | on, off | off | work lands as soon as it is offered, or at the end of the tick |
| procedures | `offer_order` | unblocks, credit, open | unblocks | the order a tick's offers land in (idle while stream is on) |
| procedures | `delegation` | on, off | on | whether the delegate action is offered |
| procedures | `toolbox` | on, off | on | whether the toolbox action is offered |
| procedures | `review_every` | 0-12 | 0 | ticks between the society's reviews of itself (0: none) |

The frame, which no recipe may change: ten agents, 24 ticks of three replies, 12 steps a tick, 480 probes, three
containers at a time, the public board with everything delivered once (`board=1`), the model, and both scores. The
budget knobs of the law (`steps`) and the organizational examples (`ways`) are deliberately not settings: the first
would let a recipe buy compute, the second would put structure into the brief.

### The evaluator, checked on saved runs before any new run

A run has two scores. The held-out checks (179 observations of the reference that no agent sees) are measured at
every merge; the hidden tests (1,182, ProgramBench's own) are graded once at the end. Selection reads the held-out
checks only. The hidden tests are what a result is reported on.

On the 21 graded AlphaTeam runs of 2026-11-03/04 (`runs_xsv/`, read through `tools/xsv_compare.py`, counts only):
- final held-out against hidden: correlation 0.96 over all runs (budgets from 32 to 2,914 replies);
- within one configuration and budget (so only run-to-run luck differs): 0.72 over the three 24- and 48-tick groups,
  0.92 over the two groups with work landing at once. One hidden test moves with about a tenth of a held-out check;
- spread between runs at 24 ticks: 4.3 checks (landing at tick end, X1) and 9.0 checks (landing at once, X3);
- the one known rule effect: landing at once against landing at tick end is +20 held-out checks at tick 24 (2.9 times
  the spread) and +14 at tick 12 (1.5 times the spread). So a half-length run carries about half the information of a
  full one per dollar (1.5 squared over half the cost, against 2.9 squared over the full cost), and the within-run
  correlation of half-time held-out with the final hidden score is only 0.4. Screens are therefore full runs.

That effect is used as the experiment's positive control: the starting recipe has work landing at tick end, and the
first stage plants one candidate, `stream=on`, marked `by: control`. If the ladder does not keep it, the ladder or the
evaluator is at fault and the experiment stops there.

### What changed in the ladder for a world that cannot be copied (`alphateam/code_runs.py`)

- **A process per run** (`tools/code_job.py`). A run that dies takes nothing with it, is tried once more, and is
  counted at what a run is expected to cost.
- **Runs of one stage side by side**, at most `--jobs` at a time, their starts 15 seconds apart. The ladder gained two
  optional calls a runner may offer: `score_many` (retests together) and `confirm_many` (the pairs up to a look
  together). A runner without them is called one pair at a time, as before; the design-world tests are unchanged.
- **Every comparison is fresh full runs.** No branches (the world cannot be copied), and no short screens (above).
- **The parent's side of a screen is the mean of its full fresh runs so far** (its generation run, and its confirming
  and retest runs). There is one task, so a pair shares no instance luck that a fresh parent run would cancel, and a
  mean of several runs is a steadier mark than one. Confirming pairs are still fresh on both sides.
- **A full run ends with the hidden tests** (`Result.hidden`, recorded with every run in the archive). Nothing in
  selection reads it.
- **A planted control** (`evolve(..., planted=...)`): edits whose effect is known, taken first among the candidates.
- **The menu skips an edit that cannot act** (a setting idle under the parent's other settings), so a draw is not
  wasted on it.
- **`rules` in a tick's last reply.** The code world lists what the last reply of a tick may do; under a recipe that
  list also holds `rules`, so a lesson or a proposal can be left with the closing note. Without a recipe nothing
  changes.

The stages as run (`tools/code_evolve.py`): one full run of the parent, whose agents may propose rule changes
(`rules`); each candidate in 2 fresh runs; the best against the parent in up to 4 fresh pairs, looked at after 2 (given
up if the mean gain is not above zero) and after 4 (kept if the one-sided 90% lower bound of the mean paired gain is
above zero). At a spread of 9 checks a gain of 20 checks is kept about 95% of the time by 4 pairs, a gain of 10 about
64%, a useless revision at most 10% once it reaches the pairs. At most 17 runs a generation, about $11.70.

### The key and the budget (`tools/keyed_runner.py`)

The OpenRouter key is typed once at a hidden prompt in a Terminal tab and lives only in that process and the jobs it
starts. Stages are queued as files (`runs_recipes/queue/*.job.json`); only this project's Python tools can be started.
Every paid stage checks `spend.json` first, keeps $2 back, and gives each run a cap of three times its expected cost.

### Checked against GOAL.md before the first paid run

| Text agents see | Where | GOAL.md line it answers to | Verdict |
|---|---|---|---|
| The fourteen settings and their one-line meanings (`rules op=view`) | `recipe.py` CODE | "Design the actions, institutions and learning loop, not the organization. Institutions are offered, never required." | Each line says what a setting does. None says what an agent should do, and none names an agent or a group. |
| The `rules` action's description (unchanged from section 20) | `recipe.py` RULES_TOOL | "Continual learning ... the society tries, measures, keeps or reverts its practices." | It states how a proposal is judged (societies with and without it). Information. |
| "Your last reply of a tick may still act (..., rules; nothing else)" | `primitives.py`, from the world's list | "Institutions are offered, never required." | The list of what is possible grows by one name. No request to use it. |
| Lessons a recipe carries, in earlier agents' words, with their status | `recipe.py` heritage | "Across runs: a society starts from what earlier societies established." | Marked untested, confirmed or refuted. Ids of agents, groups and entries are stripped, so no lesson says who does what. |
| A recipe with `offer_order` other than unblocks | law text | "Work that unblocks others comes first." | A departure the search may try; only the held-out score decides whether it stays. The default keeps the principle. |

Nothing organizer-facing was added: the in-run review (`reflection.py`) is the code of section 19, and a recipe only
sets how often it runs.

### Tests

`tests/test_code_recipes.py` (11 tests, in-memory harness, scripted agents): the seed recipe equals the plain arm; an
idle setting is not a revision; a recipe sets the world and the review interval; `rules` in the last reply; a full run
is graded and a short one is not; runs as processes side by side; the screen's mark; a dead run is tried again and
counted; the planted control and the menu; the whole ladder on fresh runs with the hidden grade recorded and unread;
the keyed runner's allowlist. The other 19 test files pass unchanged (263 tests in all).

### A first pass on single runs (2026-11-10)

The user asked for a single seed first. `Policy(confirm_pairs=0, parent="tip")` stops the ladder after the screen
(`tools/code_evolve.py --screen-pairs 1 --confirm-pairs 0`). A generation is then one run of the parent and one run of
each candidate, five runs for four candidates. The best candidate that beat the parent's run is taken on as a *lead*:
archive status `lead`, its record says "not confirmed", and it seeds the next generation (`evolve(start_from=...)` and
`--parent ID` start a generation from another recipe). A candidate that beat the parent but was not the best stays a
candidate; one that did not is dropped. No lesson is marked by such a pass, because nothing in it was confirmed.

What single runs can show. Runs of one recipe differ by 4 to 9 held-out checks, so one run against one run resolves a
gap of about 20 checks (the planted control, work landing at once, is one) and not much less. Two more cautions are
built into how the numbers are read: the parent's side of every comparison is its own fresh generation run, which
nobody selected, while a lead's screen run was the best of up to four and so reads high; and all candidates of a
generation are compared with the same single parent run, so a low parent run makes all of them look good together.
The status view prints every run of every recipe (held-out checks and hidden tests), so a lead's screen run and its
later generation run can be read side by side. Repeated runs (the confirming pairs above, or `--stage report`) come
after the user has seen what the single runs show.

No text that agents see changed for this pass, so the GOAL.md table above stands.

### The key, as it was actually given (2026-11-10)

The user gave the key in the chat. It went once to the runner in the Terminal tab through a named pipe
(`keyed_runner.py --key-pipe PATH`): the pipe carries it through memory and is removed once read, so the key is in no
file, no log and no lasting command line. `runs_recipes/queue/runner.pid` lets one runner serve the queue at a time
(two would start every job twice). The ledger was re-based the same morning: the key's usage had risen by $178.08
since 2026-11-04 through work outside this project, so `cap_usd` is now what was spent at that moment plus the $50.

Tests: 264 in all; the new one is `test_a_first_pass_on_single_runs_takes_the_best_on_as_a_lead` (`tests/test_evolution.py`).

### What the first full run changed (2026-11-10, generation 1)

**The planted control is not a known effect on today's code.** The starting rules (work lands at tick end) scored 90
of 179 held-out checks in their first full run. The saved runs with tick-end landing scored 51 (X1_plain) and 47
(X2_plain), and the runs with landing at once 76 and 87 (X3_plain). But X1 and X2 ran in the evening of 2026-11-03,
before `society.py` and `code_world.py` were changed for the X3 wave (file dates: 21:22 and 00:13), and no saved run
has tick-end landing on the later code. So the "+20 checks for landing at once" compared two waves that differed in
more than that one rule. `stream=on` stays among the candidates as a planted revision, but what it shows is no longer
a test of the ladder against a known answer. The honest noise check in a single-run pass is a second run of an
unchanged recipe, which every later generation gives for its parent.

**The closing round: an opening for agents to propose.** In that run (720 replies) and in the smoke run no agent used
`rules`: no proposal, no lesson. GOAL.md records the same pattern from earlier worlds (organizational actions near
zero while a task presses). With no proposals, every candidate is a menu draw. `code_frame(closing=True)`
(`tools/code_evolve.py --closing`) adds one step after the last tick of a full run, when the program is final: each
agent is shown the rules it ran under and the proposals so far, and may use `rules` up to three times (propose, back,
lesson) or not at all. Nothing else is on offer. It cannot move the run's score (tested: the same scores with and
without it), so runs with and without it compare and it is not in the archive's fingerprint. A part of a run (a
probe) has none. It costs at most four model calls an agent.

| Text agents see | Where | GOAL.md line it answers to | Verdict |
|---|---|---|---|
| "The run is over ... One action is still open to you, rules, at most 3 times ... Later societies start from what is left here. A proposal is tried by running societies with and without the change, and it is kept only if those with it do better. Whether you use it is up to you; a reply without it ends your part." followed by the rules view | `llm.py` `closing_text` | "Institutions are offered, never required." "Continual learning. Every agent reflects, and the society tries, measures, keeps or reverts its practices." "Across runs: a society starts from what earlier societies established." | It says what is possible and what becomes of it. It asks for nothing, names no agent and no setting to change. A test checks the text holds no "you should", "you must" or "please". |

**The menu does not pay to change a rule nobody used** (`recipe.unused`). A setting may name the counts of a run that
show its mechanism in use (claims for `own` and `own_ticks`, messages, delegate actions, shared tools). A menu draw
that only switches off, or tunes, a mechanism with a count of zero in the parent's run is skipped: in the first full
run nobody claimed a file and nobody shared a tool, so four of the fourteen settings could not have shown anything.
Switching something on is never skipped, and what agents themselves propose is never filtered.

Tests: 267 in all. New: the closing round on the stubbed paid path and on the code world (a scripted proposal becomes
the first candidate, the run's score is unchanged), and the menu's skip of unused rules.

### One run does not rank recipes here: the single-run pass after generation 1 (2026-11-10)

Generation 1, one run each, in held-out checks of 179: the starting rules 90, `credit=on` 90, `stream=on` (the planted
revision) 80, `offer_order=credit` 78 (with credit off this is landing in agent order), `delegation=off` 65. Delegation
was used once in the run of the starting rules, so `delegation=off` is nearly a repeat of them, 25 checks lower. The
five runs have a mean of 81 and a standard deviation of 10, about what the saved runs gave for one recipe (9). So one
run against one run cannot tell these rules apart, and the first rule of the pass (taken on when ahead after one run,
dropped for good when behind) would have thrown away four recipes on one draw each.

The pass now works like a race (`Policy(confirm_pairs=0, lead_runs=2, retests=2)`, `tools/code_evolve.py --lead-runs 2
--retests 2`):

- **Every run counts.** A revision's gain is the mean of all its runs so far minus what its parent stands at now (the
  mean of the parent's own runs from scratch, one more each generation).
- **Two runs before a revision is taken on.** One that is ahead after one run is set aside "ahead"; the next
  generation gives up to two such revisions another run before anything new (`generations.ahead`). If the parent's
  mark falls because its second run was lower, revisions that were behind can be ahead and come back.
- **Nothing is dropped on one run.** A revision is dropped once it has two runs and is not ahead.
- **The menu only explores.** With re-runs handled this way, a menu draw is never a recipe already on file. What agents
  propose may be on file: then it gets another run.

It is still a first pass: a lead is "not confirmed", and no lesson is marked. `tools/reopen_single_runs.py` put
generation 1's archive in the state this rule would have left (four revisions back to candidates; each change is in
the archive's decision log). Tests: 268 in all (`test_in_a_single_run_pass_what_is_ahead_gets_a_second_run_before_it_is_taken_on`).

### What the closing round showed, and two changes that follow (2026-11-10, generation 2)

Given the opening, all ten agents of the second run of the starting rules used `rules`, and all ten left a lesson;
none proposed a value for a setting. So what agents add to the loop, on this task, is advice in their own words, and
the fourteen settings are the organizer's vocabulary, not theirs. The lessons agree with each other and with what the
run's own record shows (23 of 30 rejections were conflicts on shared files): check outputs byte for byte, replay
before publishing, rebase first, keep the shared CSV layer's interface stable, coordinate or briefly own
high-fan-in files.

- **Receipts are evidence.** Every agent's first call was refused because it cited run receipts (R-ids), which the
  code world's own actions tell agents to cite, and `rules` accepted board entries only. `recipe._cited` accepts an
  entry or a receipt of the world. An id that is neither is still refused, so nothing fabricated is filed.
- **Lessons together** (`generations.candidates(together=N)`, `tools/code_evolve.py --lessons-together N`). One lesson
  of ten is an arbitrary pick (the first to be filed). A candidate may instead carry the society's lessons as one body
  of advice, as a later society would start from them: those citing the most evidence first, as many as fit a
  recipe's advice budget (12 lessons, 2,400 characters; every agent is shown all of it). It is advice only, so it
  must pass the probe like any advice. `--single-lessons 0` leaves out the one-lesson candidate.

No text that agents see changed: a recipe's lessons are shown as before ("LESSONS FROM EARLIER SOCIETIES (their
words ...)", each marked untested, confirmed or refuted). Tests: 269 in all.

## 22. A proposer that reads the record: candidates from diagnosis, judged on an independent grade (2026-11-10)

The user's direction after the first ten runs: "go implement and revise and run experiments", following a comparison
with Meta-Harness (Lee et al., arXiv 2603.28052), AlphaEvolve and CORAL, a second assistant's review of the experiment
that the user passed on, and the user's own earlier design, Queen v3. On Meta-Harness's sentence that a proposer who
inspects execution traces "can often infer why a harness failed and which earlier design choices likely contributed to
the failure, not just that it failed", the user wrote: "I agree with this very much".

### Why the ladder of section 21 was set aside

- **One run cannot rank recipes here.** Ten full runs of 2026-11-10: held-out mean 79.5, standard deviation 11.6; three
  pairs of runs under the same rules differ by 23, 16 and 12 checks (within-rules standard deviation 12.4). On the
  hidden tests the same pairs differ by 51, 37 and 17 (within-rules standard deviation 27, on a mean of 590).
- **A menu of switches proposes changes smaller than that noise.** Nothing a menu drew was distinguishable.
- **The selection score was not independent of the rules it judged.** An agent whose work lands is told how many
  held-out checks newly pass, and credit is paid on them (`code_world._verify`). Across the ten runs the held-out
  checks and the hidden tests correlate 0.65; one run is fourth on held-out checks and last on hidden tests.
- **The archive could pool runs that do not compare.** The fingerprint ignored the reading budget, the reasoning
  effort and the code a run loads (found by the review; the code version is what confounded the control of 2026-11-03).
- **Societies left lessons, not rule changes.** 49 lessons, no proposal of a setting: the settings are the designer's
  vocabulary.

### What the three systems do that this now does too

| System | What it does | Here, in AlphaTeam's terms |
|---|---|---|
| Meta-Harness | a coding agent proposes each change after reading every earlier candidate's code, scores and raw traces; with scores and summaries alone it did no better than with scores (34.6, 34.9 against 50.0); cheap validation before a paid evaluation; the evaluator is outside the proposer | `alphateam/record.py`: every run's files as they are, with an index and timelines that only say where to look; `tools/proposer_brief.md`; scripted dry run before a candidate is filed; `tools/code_rounds.py` scores |
| AlphaEvolve | code scores a candidate, never a model; a stochastic candidate is run on several seeds inside the evaluator | the hidden tests judge; every candidate is run several times; nothing is taken on from one run |
| CORAL | a hidden grader, separate from the agents; a run reports its best attempt | the judge is the grader's count; agents never learn it |
| Queen v3 | the record is lossless and summaries are derived from it; the Queen links an observation to a suggested action; it is judged on paired runs, their number set from pilot variance | `trace.jsonl` keeps everything uncut; a proposal must cite events and name the rule behind what it saw; runs a candidate needs are set from the measured spread |

### The loop

1. **The record.** `tools/record.py build --from EXPERIMENT...` copies each finished run's files into
   `runs_recipes/record/runs/<id>/` (never a run's work folder) and derives `INDEX.md` (a row a run), `timeline.md`
   (a line a round), `RULES.md` (the settings) and `SHOWN.md` (the system prompt, every action's description, the
   first turn). From 2026-11-10 a run also writes `trace.jsonl`: what each agent was shown at each turn, each model
   reply, and each action with its whole arguments and whole result (`kernel.trace`; `events.jsonl` cuts arguments
   at 160 characters and results at 240). It is written and never read back, so it cannot change a run.
2. **Proposals from outside.** Fresh agents, each given only `tools/proposer_brief.md` and a neutral place to start,
   read the record and return up to two proposals: observation with citations, how often (n of N runs), the rule or
   shown text behind it with its place in the code, the change, a prediction that can fail, risks.
3. **Proposals from inside.** After the last tick the `rules` action gains two ops (`recipe.CLOSING_TOOL`): `record`
   (the run's timeline, one tick's actions, one agent's offers, a search) and `suggest` (a rule that is not a setting
   yet, in the agent's words, with evidence). During a run the action and its description are unchanged, and both
   ops are refused. Suggestions are filed (`archive/suggestions.jsonl`; `tools/code_rounds.py --stage voices`).
4. **Building.** Whoever maintains the rules (this session) builds a proposed rule as a setting whose default is
   today's behaviour, with a test, and checks every text an agent would see against GOAL.md. The same builder
   serves proposals from outside and from inside.
5. **An epoch.** The code is then frozen. The fingerprint (`paired.fingerprint`) now holds the reading budget, the
   effort, the reply limit, every fixed option of the frame and a hash of the modules a run loads
   (`paired.code_version`), so an archive refuses runs under other code. Starting rules and candidates run in the
   same batches, the same number side by side each time.
6. **Judging.** `tools/code_rounds.py --stage table`: hidden tests, mean and spread, and each candidate's distance
   from the starting rules with a one-sided 90% bound (Welch). Every candidate gets one run; those ahead get more;
   nothing is taken on with fewer than three runs. Held-out checks are printed beside it and decide nothing.
7. Every run joins the record, and the next round of proposers reads what came of the earlier proposals.

### What a change may be, and may not be

As in the brief: a value for a setting; a new rule the world enforces; a change to what agents are shown (what an
action returns, what a turn lists, what a refusal says). Never: an agent named, work or roles assigned, advice or
instruction added to what agents are shown, the budget, model, task or scores changed, anything of the hidden tests
or held-out checks shown, anything specific to xsv's commands.

### Texts checked against GOAL.md

| Text | Where | Check |
|---|---|---|
| The closing text | `llm.closing_text` | Says the run is over, what the one open action can still do and what becomes of it, and that using it is up to the agent. No request (a test refuses "you should", "you must", "please"). |
| The closing ops | `recipe.CLOSING_TOOL` | Describes `record` and `suggest` and who reads a suggestion. Offered only after the last tick. |
| The proposer's brief | `tools/proposer_brief.md` | Addressed to an outside proposer, not to a society. It carries GOAL.md's limits on what may change. |

### Round 1: what three readers of the record proposed, and what was built (2026-11-10)

Three outside proposers, each from a different place to start (the rounds without gain; work that did not land; the
first six rounds), each about 25 minutes and 33 to 35 queries of the record, none seeing another's work. Their
reports are kept as returned in `runs_recipes/record/candidates/round1/`.

**All three put the same diagnosis first or second, with the same change.** A file of a copy that once conflicts with
HEAD stays in the copy as the agent's version (`Harness.rebase_one` keeps it and the old base), so every later offer
carries it to the gate again: 108 of the 129 conflict refusals in ten runs are the same agent's same file refused
again in a later round. Meanwhile what the agent runs in its copy is not what the gate builds: its other files are
tested against its private version of the shared module and land, or are refused, against HEAD's. One agent had a
one-line fix refused eight times in a row with a local replay that passed every time. Agents say in their notes that
they are not offering the file, and it is offered anyway.

| Setting (default is today's behaviour) | Proposed by | What changes | A prediction that can fail |
|---|---|---|---|
| `on_conflict` = keep, take_head | all three | on a conflict the copy takes HEAD's version; the agent's is moved to /public/unmerged and the agent is told where | no agent has the same file refused for conflict in two rounds (31 pairs, 108 refusals today); offers by agents that had a conflict land as often as the rest (46% against 74% today) |
| `offer` = asked, named, every_tick | A (every_tick), B (named) | every_tick: the gate takes each copy's changes at the end of every tick, publish or not; named: only the files a publish names are offered | every_tick: publish calls under 15 a run (32 to 62 today), landings above 40 (mean 32), first landing by round 3. named: no landing in a tick whose note says not to land that file (at least 7 today) |
| `show_open` = off, on | C | each turn says which commands not at HEAD no copy has changed, and which some copy has, with whose | commands with no module at the end 3 or fewer (4.8 today); `join` gets a module in one run of three (none in ten today) |

What the proposers saw and did not propose, because GOAL.md forbids it: the world posting each missing command as an
open task (every run ends with 2 to 8 of 20 commands never started; in rounds 13 to 24 a landing that brings a new
command gains 4.1 unseen checks and one that changes existing files 0.77). Two of them wrote that this would be the
world splitting the work.

The new texts an agent can see, checked against GOAL.md:

| Text | Where | Check |
|---|---|---|
| The notice after a conflict under `take_head` | `CodeWorld._conflict` | What happened and where the agent's version is. No request. |
| The turn line "Moved aside by a conflict ..." | `CodeWorld.observe` | A fact, shown until the agent next changes that path. |
| The publish action under `named` and `every_tick` | `code_world.publish_tool` | What a publish offers under the rule in force. |
| The system prompt's lines on offers and on what stays in a copy | `CodeWorld.brief` | State the rule in force; under the starting rules the prompt is unchanged to the byte (checked against the record's SHOWN.md). |
| The turn line of `show_open` | `CodeWorld.open_pieces` | Which missing commands no copy has changed and which some copy has. It does not say what to do about it. |

### How round 1 is judged (written 2026-11-10 11:15, before any candidate had run)

- **Folders.** `runs_recipes/XSV2a`: five runs of the starting rules under the code before the new settings (closing
  round on). `runs_recipes/XSV2b`: the candidates and more runs of the starting rules, under the code with the new
  settings, frozen. The two codes show an agent the same texts under the starting rules, so the baseline is reported
  both ways: XSV2b's own runs, and those pooled with XSV2a's five.
- **Batches** of five runs side by side, each with one run of the starting rules in it, so that a difference between
  batches (time of day, load) shows in the baseline.
- **First pass:** every candidate runs once.
- **Second pass:** the three candidates with the highest hidden score in their first run get two more runs each
  (ties: the higher held-out count).
- **The estimate for a re-run candidate is the mean of its second-pass runs only.** Its first run picked it, so
  counting it would flatter it (the best of eight single runs is high by luck alone: with a spread of 27, about 38
  hidden tests above the mean).
- **Taken on:** a candidate whose second-pass mean is above the baseline mean with a one-sided 90% lower bound of the
  difference above zero (Welch). Anything less is reported as not shown, with its interval.
- **The predictions in the table above are checked on the record of every candidate run, whatever its score.** A rule
  can do what its proposer said it would and not raise the score, or the reverse; both are results.
- **Money:** about $0.75 a run; XSV2a about $4.50, XSV2b about 19 runs, $14. A later round and a confirmation of
  whatever is taken on come out of what is left.

**Amended 2026-11-10 11:58, after the five baseline runs and before any candidate had run.** One of the five
baseline runs collapsed: 10 hidden tests, against 505, 548, 555 and 641 in the others. Its first landing, in round 5,
replaced the launcher (`compile.sh`, `cli.py`) at a moment when HEAD passed no check, so the gate had nothing to
protect; after it no command answered, every later landing "broke nothing", and four agents then fought over
`compile.sh` to the end. That is one collapse in fifteen full runs of 2026-11-10. So:

- **A run has collapsed when it ends below 100 hidden tests.** Collapses are counted and reported for every recipe.
- **Means are compared over runs that did not collapse,** with the all-runs means printed beside them. One collapse
  moves a mean of five runs by 110 tests, four times the within-rules spread, and whether a rule makes collapses
  rarer or commoner is a question about a rate of about 1 in 15, which this budget cannot answer.
- A candidate with a collapse among its runs is flagged, whatever its mean.

**Made precise 2026-11-10 20:00, while batch 2 of the first pass was running and before any second-pass run.** The
rules above left three things open; they are settled here so that no result can settle them:

- **Which baseline decides.** The baseline is reported both ways (this round's own runs of the starting rules, and
  those pooled with XSV2a's). A candidate is taken on only if the lower bound is above zero against both. If the two
  disagree it is reported as not shown, with both intervals.
- **The second pass is two batches of five:** the three picked candidates once each, and two runs of the starting
  rules. The spare slots go to the baseline, which every comparison leans on, not to a favourite.
- **Second-pass runs have the closing round** (after the last round each agent may look at its run's record and
  suggest a rule). It comes after the program is graded and is outside the fingerprint, so it does not touch a score;
  it gives suggestions from societies that lived under a candidate rule, for the next round of proposals.

`tools/judge.py` applies these rules to the run files (`--pick 3` names the second-pass candidates); the table of
`tools/code_rounds.py` stays a plain listing of all runs.

### What the societies themselves suggested, and what was built from it (2026-11-10)

The closing rounds of the five baseline runs (`tools/code_rounds.py --out runs_recipes/XSV2a --stage voices`): 14
suggestions of rules that are not settings, from four of the five societies (the fifth left 11 lessons and no
suggestion), and 52 lessons. Read and grouped by me; the groups and their support:

| What they asked for | Suggestions, societies | Built |
|---|---|---|
| Replay the logged checks when work is offered and report the failure to its author | about 7, in 4 societies; 3 backed by a second agent | `publish_replays=on`: a publish also returns the logged checks replayed on the copy (run by the world, no step) |
| Do not offer an unchanged stale file again and again; make its agent reconcile and resubmit | 6, in 3 societies | `on_conflict=hold_back`: a file that conflicted stays as the agent's version and out of its copy's offers until the agent changes it |
| A view of who has what: per file, its owner at HEAD, who is changing it, earlier refusals and why | 5, in 3 societies | not built in this round (close to `show_open`, which a reader from outside proposed) |
| One more chance to reconcile after the last tick | 3, in 2 societies (and 2 in the live check) | not built: it changes the budget |

The second row is the same failure the three outside readers found, with a different remedy: the readers move the
agent's version out of its copy (`take_head`), the societies keep it in the copy and stop it at the door
(`hold_back`). Both are run.

The society that collapsed said why, in its own closing round: "a launcher landed despite replay still failing",
and asked for "a required smoke test from a separate temporary cwd with a relative input file before publishing
launcher/CLI changes". A rule that a landing must gain a logged check would have stopped that landing; it needs the
gate to judge an offer before it lands, which the harness does not offer, so it is not built.

A side finding while building: a notice to an agent is cut at 300 characters (`kernel.notify`), and the starting
rules' conflict notice names every path twice, so with several files in conflict the agent never reads the part that
says its version was kept and that it is the one to reconcile. The starting rules are left as they are; the notices
of the two new rules say what became of the file first and name each path once.

### The first pass of round 1: ten runs, and what they showed (2026-11-10)

Hidden tests of 1,182 (held-out checks of 179), one run each, in two batches of five with a run of the starting
rules in each. The record's `candidates/round1/RESULTS.md` has each run against what was expected of its rule.

| Candidate | From | Hidden (held-out) | Commands missing at the end | What its run showed |
|---|---|---|---|---|
| `memory_ticks=3` | the menu | 669 (78) | 2 | 45 landings, 24 files landed, fewest messages and looks at HEAD; 23 conflict refusals all the same |
| `on_conflict=take_head` | outside readers | 659 (88) | 4 | its predictions held: no repeat conflict, agents hit by a conflict adapted to HEAD's version and landed more than the rest |
| starting rules | | 571 (88), 543 (57) | 8, 8 | |
| `publish_replays=on` | societies | 537 (92) | 5 | the replay came back with every publish, and in 35 of 38 the publish was the agent's last reply of the round |
| `offer=named` | outside reader | 489 (78) | 7 | no repeat conflict; first landing late (round 5) |
| `show_open=on` | outside reader | 430 (64) | 12 | the line was shown and right, and ignored |
| `on_conflict=hold_back` | societies | 388 (64) | 11 | the file stayed out of offers, and eight files stayed private to the end; work tested against them was refused at the gate |
| `offer=every_tick` | outside reader | 364 (50) | 6 | unfinished files landed again and again; 19 regression refusals |
| +4 lessons | societies | 247 (33) | 16 | 271 shell calls against 343 to 408 elsewhere; messages, claims and looks at HEAD up |

Three things this pass showed that the plan had not assumed:

- **The hidden score is breadth.** Across the 14 traced runs that stood, hidden tests go with the number of commands
  that have no module at HEAD at the end (r = -0.84). The round-1 diagnosis, mine and the readers', was about
  conflicts; conflicts do not order the runs (the best run had 23 conflict refusals). A reader of round 1 had seen
  it ("a landing that brings a new command gains 4.1 unseen checks and one that changes existing files 0.77") and
  proposed only a line of text for it, because the direct rule (the world hands out the missing commands) is not
  allowed.
- **Replies are the budget.** With three replies a round, whatever a rule or a lesson makes an agent do with a reply
  is taken from building. The four lessons moved a quarter of the shell calls into messages, claims and looks at
  HEAD, and that run is the lowest that stood. A rule's cost in replies has to be counted with its effect.
- **Inside and outside, so far.** The societies' two rules did what the societies asked for and did not help in
  their one run; the outside readers' common rule did what they predicted and scored high in its one run; the best
  single run came from neither, a value on the menu. One run each: the second pass decides what can be said.

### Round 2: how it is run and judged (written 2026-11-10 20:50, before its proposals came back and before any second-pass run of round 1 had ended)

Round 1 spent a run on each of eight candidates and then had to discount the run that picked the best three. With a
within-rules spread of 45 hidden tests that is a poor use of runs. Round 2 has fewer candidates, more runs of each,
and no step that picks on a score:

- **At most three candidates,** chosen from the proposals before anything runs, on the evidence and the mechanism in
  the reports. Each is built as a setting whose default is today's behaviour, its texts checked against GOAL.md.
- **A base.** The starting rules plus every round-1 candidate that is taken on. If none is taken on, a round-1
  candidate whose second-pass mean is above the baseline's by more than one baseline spread (45) is carried as a
  provisional base, said to be provisional, so that its further runs come out of this round. If there is none, the
  base is the starting rules.
- **Three batches of five, each: the starting rules, the base, and the three candidates on top of the base.** So
  three runs of everything, side by side; all three runs of a candidate count. (If the base is the starting rules,
  the fifth slot is a second run of them.)
- **Judged on hidden tests, as before:** a collapse is a run below 100; means over runs that stood; one-sided 90%
  Welch bound. A candidate is an improvement on the base if its three runs are above the base's runs with the bound
  above zero; the base, and each candidate, is better than the starting rules by the same test against the starting
  rules' runs (this round's own, and pooled with the earlier ones). Anything less is reported as not shown, with its
  interval.
- **The code of round 2 goes into the project only when no run of round 1 is in flight** (a run that died would be
  tried again in a new process, which would import the changed code under the old fingerprint). It is built and
  tested in a copy of the code tree meanwhile.

### Round 1 judged, and round 2 amended before any of its runs (written 2026-11-10 22:45)

Round 1 ended at 22:33 (20 runs, $13.99). **No candidate is taken on.** Later runs only, hidden tests:

| Candidate | First run | Later runs | Against this round's starting rules (6 runs, mean 484, sd 108) | Against pooled (10 runs that stood, mean 515, sd 96) |
|---|---|---|---|---|
| a publish replays the checks | 537 | 546, 569 | +74 (bound +7) | +42 (bound -2) |
| three rounds of conversation | 669 | 439, 585 | +28 (bound -234) | -3 (bound -246) |
| take HEAD on a conflict | 659 | 629, 351 | +6 (bound -442) | -25 (bound -463) |

The spread of the starting rules is about 100 hidden tests, not 45: six runs in this round gave 571, 543, 577, 301,
411, 498. The first figure came from eight runs that happened to sit close together. Two consequences. The two
candidates picked on a high first run fell back, as a pick on one noisy run must. And three runs a candidate can show
only a difference of about 150, which no rule here is likely to have.

Amendments to round 2, made before the round-2 code was in the project and before any of its runs:

- **The base is the starting rules.** The written threshold was "more than one baseline spread (45)". The intent was
  one spread, the spread is now about 100, and no candidate's later mean is that far above either baseline.
- **The three chosen candidates stay, run as a two-by-two:** starting rules, `new_file=first_copy`,
  `instructions=off`, both. The effect of each rule is then estimated from every run: all runs with the rule against
  all runs without it (half the runs each), and the single cells are reported beside it.
- **Four batches of five, not three:** the four cells, and in the fifth slot `publish_replays=on`, the one round-1
  candidate close to its bar and the steadiest (537, 546, 569). Its four further runs are judged with its two later
  runs of round 1 by the round-1 test against both baselines (this round's starting rules, and those pooled with
  XSV2b and XSV2a).
- **The test for a rule in the two-by-two:** one-sided 90% Welch bound of (runs with the rule) minus (runs without)
  above zero, over runs that stood; collapses counted. With 8 runs against 8 and a spread of 100 the bound clears
  zero at a difference of about 65. A smaller true effect will be reported as not shown, and that is the honest limit
  of what the remaining budget can show.
- No batch is dropped or added on the scores. A batch is stopped only for a fault in the harness.

### Round 2 judged (written 2026-11-11 02:40)

Twenty runs, $14.04, no collapse, no fault. **Nothing is shown.** One copy at a time: +50 over eight runs against
eight (bound -18). Facts only: -47 (bound -114). A publish replays, six later runs: +28 on the pooled baseline (bound
-19), -7 on this round's. Details and each prediction: `runs_recipes/record/candidates/round2/RESULTS.md`.

What two rounds (45 runs of ten agents, 11 sets of rules) establish about the loop itself:

- **The evaluator is the limit.** Runs of one set of rules differ by 100 to 140 hidden tests (sd); the largest
  difference between rules is 50. To show +50 at this spread takes about 16 runs a side, about $23 a comparison.
  Single runs, and picks made on them, are noise: both round-1 picks fell back, and a halfway reading of round 2
  (+98, bound +7) shrank to +50.
- **The proposers are better at mechanism than at score.** Their predictions about counts mostly held (no repeated
  conflict under take HEAD, three runs of three; fewer commands missing under one copy at a time, three of four;
  fewer claims under facts only, eight of eight). None of these moved the score detectably.
- **What the score follows is breadth,** files landed and commands covered (r about 0.8 over 44 runs), and it is not
  settled early in a run. No rule tried so far changes breadth by more than about two commands.
- **A single agent with the same 720 replies scored 563** (one run), inside the ten agents' range and above their
  mean (529), at about the same cost and two to three times the clock time.

What follows for the next step, not yet done: the comparison needs either a less noisy judge of a rule (many short
runs, or the same early rounds replayed under two rules so that runs are paired) or rules with larger effects on
breadth; more single comparisons at this spread and budget will not decide anything.

### Known limits

- The ten runs of 2026-11-10 have no `trace.jsonl`: their record is the cut events and the world's own books.
- The hidden tests now select as well as report. A final claim needs fresh runs, and ideally programs no recipe was
  selected on.
- One task. A recipe that is better on xsv is a recipe that is better on xsv.
- Proposers from outside are a stronger model than the society's agents and see every run; a society sees its own.
  A difference between the two sources is not a clean comparison of inside against outside.
- The builder stands between a suggestion in words and a rule in code, for both sources.
