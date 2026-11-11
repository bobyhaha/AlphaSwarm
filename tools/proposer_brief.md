# Brief for a proposer that reads the record

You are asked to propose the next change to the rules of an agent society, from the full record of its earlier runs.
Nobody will follow up with questions, so your report has to stand on its own.

## The setting

Ten language-model agents rebuild a real command-line program (xsv, a CSV toolkit) from its behaviour, in 24 rounds.
Each agent has its own working copy; work reaches the one shared program (HEAD) only through a gate that merges,
builds and replays checks. Nobody is placed in charge: the society runs under a **recipe**, a set of rules the system
enforces (who may do what, what is visible, when work lands, how credit is paid) plus lessons carried from earlier
societies. The project's question is which rules let such a society do better at the same budget.

The score that decides is the number of **hidden tests** HEAD passes at the end (of 1,182). Agents never see them.
A second score, held-out checks (of 179), is also never shown, but an agent whose work lands is told how many of
them newly pass, so it is part of the run and not an independent judge.

## What you have

Everything is under `runs_recipes/record/` in this project (use paths relative to the project root):

- `INDEX.md`, `index.json`: one row a run, with the rules it ran under, both scores and counts. A summary for
  finding where to look, not evidence of why.
- `RULES.md`: the settings a recipe can set today, and what is fixed.
- `SHOWN.md`: what every agent is shown: the system prompt, every action with its description, the first turn.
- `runs/<id>/timeline.md`: a run round by round (what landed, what was turned away and why, holds, tasks).
- `runs/<id>/events.jsonl`: every action of every agent, with its arguments and what came back, each cut short
  (arguments at 160 characters, results at 240).
- `runs/<id>/trace.jsonl` (runs of XSV2a and later; the ten runs of XSV1 have none): the same run uncut. What each
  agent was shown at each turn (`kind` turn), every reply of the model with its text and calls (reply), every action
  with its whole arguments and whole result (action), the system prompt (system), the closing round (closing).
- `runs/<id>/run.json`: what the world kept (`world.merges`, `rejections`, `unmerged`, `claims`, `task_log`,
  `delegations`, `proposals`, `plans`, `findings`, `traces` of each agent's shell commands, `structure_measures`).
- `runs/<id>/entries.jsonl`: the boards. `metrics.json`: counts by round. `result.json`, `job.json`: scores, recipe.
- `candidates/`: changes proposed in earlier rounds, by readers like you and by the societies themselves after
  their runs, as they were written; what was built from them (`round1/README.md`); and what the runs of each built
  candidate showed against the prediction its proposer had written down (`round1/RESULTS.md`). The candidates' own
  runs are in the index like any other run, with full traces: a rule that was tried is evidence too, of what it led
  agents to do.

A query tool, run from the project root:

```
.venv/bin/python tools/record.py runs
.venv/bin/python tools/record.py show RUN
.venv/bin/python tools/record.py events RUN --round 6 --agent A5 --op publish --grep csvio --refused
.venv/bin/python tools/record.py trace RUN --round 6 --agent A5 --kind reply
.venv/bin/python tools/record.py diff RUN_A RUN_B
.venv/bin/python tools/record.py checks XSV2a XSV2b    # counts earlier proposals made predictions about, a row a run
```

Events are numbered by their line in `events.jsonl`; cite them as `RUN:line`. You may also read the project's code to
see what a rule does: `alphateam/code_world.py` (the world: gate, holds, tasks, credit, what a turn shows),
`alphateam/recipe.py` (the settings), `alphateam/primitives.py` (system prompt, actions).

## What is known, so you do not have to rediscover it

- Runs of the same rules differ by about 45 hidden tests (standard deviation over the eight runs of the starting
  rules that stood). One run of the starting rules in nine collapsed: it ended at 10 hidden tests after a landing
  replaced the launcher while HEAD passed no check. A score difference between single runs is not evidence of
  anything. What a run's events show did happen is evidence.
- The two scores agree loosely (correlation 0.80 across the 24 full runs that stood): the run with the most held-out
  checks among the traced runs is below the starting rules' mean on hidden tests.
- A first round of candidates was built and run once each, and three of them are being run again. Read
  `candidates/round1/RESULTS.md` first: every candidate's score, what its run showed against what its proposer
  expected, and two counts across all runs. A change that repeats one already tried needs a reason the record
  supports.

## What a change may be

1. A value for an existing setting.
2. A new rule the world enforces. Describe exactly what the world does differently and when. It will be built as a
   new setting whose default is today's behaviour.
3. A change to what agents are shown: what an action returns, what a turn lists, what a refusal says.

A change is tried on top of the starting rules, and on top of an earlier candidate if that one is taken on
(`candidates/round1/RESULTS.md` says where each stands). Say whether yours needs an earlier candidate's rule,
replaces it, or is indifferent to it.

## What a change may not be

- It may not name an agent, or assign work, roles, leaders, teams, stages or a procedure. Rules apply to whoever
  meets their condition.
- It may not add advice or instruction to what agents are shown ("you should", "first do"). Agents may be shown
  facts: what happened, what is in force, what an action does.
- The budget (agents, rounds, replies, shell steps, probes, reading budget), the model, the task and the scores
  are fixed. Nothing may show agents the hidden tests or the held-out checks.
- It may not be specific to xsv's commands. It has to make sense for any program a society might rebuild.

## How to work

How you investigate is up to you. Three requirements on the result:

- **Evidence.** Every claim cites runs and events. For each pattern say in how many of the full runs it appears and
  in how many it does not. A story from one run is the commonest way to be wrong here.
- **Mechanism.** Name the rule or the shown text that allows or causes what you saw, with the place in the code.
- **A prediction that can fail.** Say what the record of future runs would show if the change works (for example,
  a count in the index moving), and what it would show if it backfires.

Read only. Do not edit or create files in the project, do not start runs, and do not open any run's `work` folder,
the grader, the grades folder or anything of the hidden tests. Your report is your whole output.

## The report

Return up to two proposals, the one you would try first on top, each with:

- `title`
- `observation`: what happens, with citations (`RUN:line` or `RUN round N`)
- `how often`: n of N full runs, and the runs where it does not happen
- `mechanism`: the rule or text behind it, with the code location
- `change`: precise enough to build without asking you. For a new rule: the condition, what the world then does,
  what an agent is told when it applies. For shown text: where, and the words.
- `prediction`: what would be seen in the record if it works; your honest guess of the effect on the hidden score
- `risks`: how it could backfire, and what in the record would show that
- `checked and ruled out`: what you looked at that turned out not to matter, and what you did not check
