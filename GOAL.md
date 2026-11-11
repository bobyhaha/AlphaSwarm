# AlphaTeam: the project's main goal

Written 2026-10-31 from the user's statements and the proposal "Adaptive Agent Societies: Self-Organizing Hierarchies and Recursive Superagents".

## Goal

Build an agent society that learns to organize itself. Agents get a set of actions, which they can expand by defining new ones, and explore freely. The society discovers which actions, and which combinations of actions, are useful. Credit assignment is the central means: it tells agents which actions and combinations paid off.

**The picture: a hard problem is a large jigsaw puzzle** (the user's framing, 2026-10-31). Many agents can solve local pieces in parallel. Combining them into one coherent picture is just as important, and it is where societies fail.
- **Pieces** are findings, dead ends, solutions and fragments of a model of the problem.
- **Edges** are where pieces must agree: shared variables (the design world's shared knobs), overlapping claims, a mechanic that holds on two levels.
- **The assembled region** is one shared artifact. It grows only when a new piece fits everything already placed.
- What follows:
  - a placed piece is never swept off the table;
  - a piece joins only where its edges fit;
  - two pieces for one spot is a conflict, settled by which one fits the evidence;
  - pieces that connect many others go first (corners and edges; in the user's other project, import repairs);
  - clusters assembled apart are joined by integrators.
- **Progress is how much of the picture is assembled and coherent,** not how many pieces lie on the table.
- **So far AlphaTeam makes pieces but does not assemble them.** ARC agents posted findings and plans, but the table was swept clean whenever a level fell, and nothing joined the pieces. In the design world, coordination broke exactly at the edges (shared knobs).

The goals below follow from this picture.

**Learn from the results that already exist** (added 2026-10-31 at the user's request). Before placing new pieces, an agent studies the assembled region. A result should carry its understanding: why it works, which mechanics it shows, what was ruled out. Reproducing it is not enough.
- **Within a run:** an agent that replays a published solution should come away knowing what each part of it does, not just stand at the next level. In the 16-agent runs of the new design, 71% of copies reached level 1 by replay, and both level-2 completions in the baselines came from agents that had solved level 1 themselves. The moves travelled; the understanding did not.
- **Across runs:** a society starts from what earlier societies established (results, mechanics, dead ends), not from nothing.

**Preserve and combine useful work across agents** (added 2026-10-31 at the user's request; the main improvement in another of the user's projects).
- **Nothing an agent establishes is lost when someone else's result is published.** Work on branches that did not win survives and stays visible.
- **Compatible contributions are combined,** but only when everything that worked before still works. Previously passing checks are the gate for any merge.
- **Work that unblocks others comes first,** ahead of work that only adds more of the same.

**Resolve conflicts without a central arbiter** (added 2026-10-31 at the user's request, after rereading Agensh, arXiv 2609.26781).
- **The substrate makes conflicts visible:** overlapping claims, a contribution built on state that has since changed, a counterexample to a finding.
- **The parties settle them directly and fast.** A message reaches the other agent during its turn.
- **Evidence decides disputes about correctness,** not rank or votes. A concrete counterexample withdraws an approval until the author fixes the work.
- **Whoever contributes later reconciles with what is already there.** They take in the current state, re-check, then merge, instead of overwriting it.
- **Responsibility moves explicitly, to one agent at a time,** for example an integrator chosen from those who answer, with the other requests cancelled.

## Vision: an emergent society

Structure arises from agents' own choices, not from a prescribed org chart:
- leads, teams and contracts;
- hierarchies several levels deep (recursive superagents).

A lead works like a CEO or PI:
- it directs a team, delegating jobs and roles to its children, including keeping track of things;
- it still works on details itself;
- a child who leads a sub-team does the same, so structure recurses.

The claims to test: organization is **emergent** (not prescribed), **adaptive** (changes as the task changes) and **scalable** (keeps working with more agents).

## Where it should pay

Research-type tasks that need many explorations of one unknown system, where members build on each other's findings. ARC-AGI-3 is the test bed. Bags of independent tasks are a weaker test.

## Design principles

- **Design the actions, institutions and learning loop, not the organization.** Institutions are offered, never required. Examples: take_lead, contracts, delegation.
- **Continual learning.** Every agent reflects, and the society tries, measures, keeps or reverts its practices.
- **A shared substrate.** It has three parts:
  - private, local (group) and global boards;
  - linked posts with citations;
  - ranked search, as in ScaleMAS.

  The `sync` option (2026-10-31, after reading Agensh's code) makes it one public substrate. There are no private boards; an agent's memory is its own recent public notes and plays. What others post reaches each agent once, most relevant first, and messages and urgent entries arrive during its turn.
- **Coordination has real costs.** Views are bounded, and reading and actions have budgets.
- **Organizing pays only when contributions are recognizable and members differ.**
- **Make each agent's work legible to the others** (inspired by Agensh's worker loop, arXiv 2609.26781; adapted, not copied).
  - **What travels on the substrate:** what an agent intends, what would show it worked, its evidence and its result.
  - **What comes back:** others' progress returns to the agent.
  - **The working cycle** is a view the rules can show or hide. It is not a required procedure, and the score judges whether it helps.
- **Diversity comes from information, not assignment** (endorsed 2026-10-31). Agents see others' claims, dead ends and what nobody has tried yet; nobody is told what to explore.

## What we have learned so far (~200 runs)

- **Agents barely explore the action set.** Across 195 runs:
  - 61% of actions were world actions and 32% reads;
  - organizational actions were near 0;
  - 7 custom actions were defined.
- **Credit that is only information changes what agents say, not what they do.** Credit with consequences changes behavior, for example authority over decisions or more resources.
- **Offered institutions produced structure.** When agents were given take_lead, contracts and individual rewards:
  - 12 of 13 multi-block leads matched hidden modules;
  - hierarchies up to three levels deep formed;
  - capable agents took the leads.

- **On ARC-AGI-3, the new design changed behavior, not results** (16 runs, 2026-10-31). The design offered actions, engine credit, a substrate of plans, evidence and interfaces, and evolving rules.
  - **Agents used it.** They claimed plans, posted backed findings and dead ends, and replayed solutions. One run agreed an interface without prompting, and the rule loop correctly found which kinds of entry paid.
  - **No team, contract or self-written skill appeared.**
  - **None of 10 sixteen-agent runs got past 3 levels,** against 4 of 6 baseline runs (p ~ 0.008).
  - **48 agents did best** (AUC 0.124) and began splitting work among themselves.
  - **Why the 16-agent runs stalled (from the logs):** 85% of credit paid for reaching new states and 8% for solving. Most of the missing levels were ls20 level 1, where these runs made more moves than the baselines yet cleared it less often.

## Open problem

Credit for actions and combinations, not only for agents. On ARC the credit was computed but did not move the frontier; whether a society can *act* on what credit reveals (not only see it) is the open question. The society needs to see which practices worked, and that credit must have consequences.
