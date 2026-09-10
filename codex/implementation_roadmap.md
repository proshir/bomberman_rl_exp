# Implementation roadmap

## Latest benchmark pilot

Added solo `coin-heaven` completion rate, success count, and mean completion
steps (null when no game finishes). Fixed an unseeded Python random generator
used by `coin_collector_agent`; the earlier pilot did not fully control its randomness.
The initial `experiments/coin_supplied_{50,100,150,200}_steps/` runs are preserved
for diagnosis. Use the corresponding `_steps_seeded/` runs for this comparison.

Each corrected budget has 192 games: three agents, board seeds 10--17, action
seeds 0--1, and four corners. Mean coins at 50/100/150/200 steps:
`random_agent` 2.11/2.11/2.11/2.11; `peaceful_agent` 3.81/5.70/7.62/9.72;
`coin_collector_agent` 23.64/43.08/50/50. Only the collector completed games:
64/64 at both 150 and 200 steps, with mean completion time 127.17 steps.
Its completed game records match across those two budgets.

**Proposal, not yet selected:** use 100 steps for initial coin-count screening;
the collector has not reached the ceiling, while weaker agents still collect coins.
These are exploratory supplied-agent results, not learning-agent results.
Temporary summary checks, the existing benchmark smoke test, and checks of all
768 corrected game records passed. Shared training records remain the next
implementation task after choosing the screening budget.

Build and preserve a diverse collection of learning agents, discover which ideas
work best for Bomberman, and submit the strongest supported candidate. Develop at
least **two genuinely different learning models**, including one based on lecture
methods. The search is not limited to techniques suggested in the project brief.
See [the project brief](final_project.md), especially sections 4, 7, and 9.

## Goal and stages

The four tasks in the brief are progressive **benchmarks**, not limits on agent
design. Search broadly across learning algorithms, representations/features,
decision structures, rewards, and training methods. Possible families include
compact value-based agents, function approximation, spatial neural models, and
modular agents; none is selected yet.

1. **Coin navigation:** build the shared experiment pipeline and supplied-agent
   baselines, then cheaply screen several distinct agent ideas in `coin-heaven`.
   Compare coins collected, steps, and learning behaviour.
2. **Crates and survival:** extend promising and new branches with bomb, danger,
   escape, and crate information. Compare representations, rewards, curricula, and
   key hyperparameters; measure coins, crates, and self-deaths. Recheck Task 1.
3. **Hunting:** evaluate against `peaceful_agent`, then `coin_collector_agent`.
   Investigate opponent representation, attack, and risk; measure score, kills,
   deaths, and survival.
4. **Competition:** compare complete agents against `rule_based_agent` and fixed
   supplied-opponent lineups. Investigate any motivated game-playing idea, including
   planning and risk management; prioritize official score and track diagnostics.
5. **Final comparison:** give promising agents comparable tuning budgets, select
   on development data, evaluate frozen finalists on fresh games, and test the
   winner under official constraints.

For each idea: **hypothesis → cheap pilot → fair comparison → keep, revise, or
reject**. Diagnose behaviour to generate further ideas; use focused changes and
ablations to explain improvements. Preserve all implemented branches and results.
A seed or hyperparameter change is a variant, while a substantive change to what
or how the agent learns may define a different model.

**Accepted decision:** exclude self-play. Use the defined tasks and fixed supplied
opponents so effort stays on finding and comparing diverse agent ideas.

The user approves model choices, experimental strategy, and substantial runs;
Codex advises and implements approved work.

## Statistical experiment rules

1. **Define the question first.** Fix the baseline, changed components, primary
   metric, smallest meaningful effect, budgets, and stopping rule before the main
   comparison. Use one-component changes for attribution or a planned interaction
   experiment. Never add runs simply until a desired result appears.
2. **Compare fairly.** Use comparable training and tuning budgets, the same
   evaluation scenarios/opponents, balanced seats, and documented exploration.
   Control environment and agent randomness; matching seeds alone does not ensure
   matching trajectories.
3. **Separate tuning from testing.** Use distinct training, development, and final
   evaluation seeds. Confirm exploratory winners on fresh data. Report scenarios
   separately unless aggregation weights were fixed in advance.
4. **Repeat independently.** Learning-method comparisons need independent training
   runs; many games from one checkpoint are not independent training replicates.
   Use pilot variability and a precision/power target to choose runs and games.
5. **Report uncertainty and failures.** Report effects, 95% confidence intervals
   respecting training-run and game grouping, sample counts, and individual-run
   results. Preserve failed runs; avoid best-seed reporting. Unresolved differences
   are inconclusive. Plan multiple-comparison handling for multiple formal claims.
6. **Measure real performance.** Use task outcomes as primary metrics: initially
   coins within a fixed step budget; for competition, proposed mean official score.
   Track kills, self-deaths, survival, and runtime as diagnostics; record shaped
   reward separately. Generate figures from saved data.

These are practical rules for meeting the brief's scientific-experiment requirement;
exact models, budgets, and statistical procedures remain experiment-specific choices.

## Keep every tested variant

- `src/`: reusable experiment scripts and implementations of every developed model.
- `experiments/`: unique variant/run IDs, configurations, code versions, commands,
  dependencies, seeds, budgets, raw results, plots, and evaluated checkpoints or
  durable checkpoint references. Never overwrite another variant's artifacts.
- `codex/`: idea backlog, decisions, and experiment index. For each experiment:
  **ID/status → hypothesis → parent variant and change → protocol/artifacts →
  result with uncertainty → interpretation and next decision**.

Label proposals, approved work, exploratory observations, and confirmed results
clearly. Preserve unsuccessful variants as well as finalists.

## Where to start now

**Accepted implementation scope:** build the shared evaluation runner first,
using supplied agents to establish reference performance before developing learners.
The runner must also support `classic` with fixed opponent lineups.

Implemented `src/run_benchmark.py`. Run it from the project root, for example:

```bash
env/bin/python src/run_benchmark.py --agents peaceful_agent coin_collector_agent --seeds 10 11 12 --output experiments/coin_pilot
```

The first named agent is the baseline. Add `--opponents rule_based_agent` and
`--scenario classic` for later competitive evaluation. It records individual games,
world and action seeds, four corner rotations, and
per-agent outcomes in unique experiment directories. Supplied agents now accept
controlled randomness during benchmarking. Ordinary game commands retain unseeded
behaviour. Exploratory comparisons use paired differences from the first candidate
and 95% bootstrap intervals over board means, grouping seats and action seeds together.
This compares fixed policies; independent training replicates and their analysis
remain necessary when comparing learning methods.

An exploratory supplied-agent baseline is saved in
`experiments/coin_supplied_pilot/`: eight board seeds, two action seeds, and four
starting corners per agent. `coin_collector_agent` collected all 50 coins in all
64 games (mean completion time 127.83 steps, range 113--143); `peaceful_agent`
averaged 16.89 coins and `random_agent` 2.11. These are exploratory results, not a
fresh-seed confirmation comparison. No custom learner has been produced yet.
Small software checks are validation, not evidence that one agent is better.
Validation: the benchmark smoke test and existing game smoke test pass in
`src/tests/test_framework.py`. Tests are kept locally in the Git-ignored
`src/tests/` folder. Run from `src/` using the project's Python interpreter and
`-m unittest discover -s tests`.

**Accepted Stage 1 direction:** cheaply implement and screen all candidate families,
then give promising and inconclusive candidates fair confirmation budgets. The
detailed candidate list, implementation order, metrics, and exit criteria are in
[`stage1_coin_navigation_plan.md`](stage1_coin_navigation_plan.md).

**Accepted first learner comparison:** implement plain tabular Q-learning, observe
its learning and evaluation behaviour, then add symmetry canonicalization as one
controlled change. This supplies a clear step-by-step experiment for the report and
tests whether sharing equivalent states improves sample efficiency.

**Next implementation step:** add completion rate and completion-step summaries to
the benchmark, then measure supplied-agent performance at shorter fixed step budgets
to choose the primary screening budget. Then implement the plain tabular Q-learning
agent before its symmetry variant.
