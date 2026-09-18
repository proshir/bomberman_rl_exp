# Implementation roadmap

## Task 2 diagnostics and mixed curriculum pilot completed

Added benchmark diagnostics for repeated states, no-progress stretches,
progress events, visible coins, crates, bombs, invalid actions, and survival,
plus a mixed Coin Heaven/loot-crate training mode. The baseline crate audit
was safe but stagnant: 17.02 coins, 49.67 crates, 245.6 repeated states, and
a 227.2-step maximum no-progress stretch over 96 games.

The three-seed, 300-round mixed pilot recovered Coin Heaven navigation to 39.14
mean coins at round 300, compared with 19.70 for the crate-only regression, but
loot-crate performance was 10.41 coins versus 19.35 for the crate-only pilot.
Survival remained 100% and invalid actions zero. Raw results are in
`experiments/combat_fqi_history_antistag_mixed_pilot/`. This is a promising
Task 2 transition result, not yet an opponent-training gate: checkpoint
selection must require both scenario thresholds and reduce crate forgetting.

## History-aware anti-stagnation combat pilot completed

Added `combat_fqi_history_antistag_agent`, which retains the original combat
reward, fitted-Q settings, and deterministic safety layer while appending the
successful Coin Heaven history-8 and steps-since-progress signals. Eleven
focused agent tests plus two framework tests pass. The three-seed, 300-round,
400-step crate pilot reached 19.35 mean coins at round 300, compared with 6.54
for the original combat agent. On matched fresh crate boards it reached 16.68
versus 7.35, with 100% survival and no invalid actions or suicides.

The Coin Heaven regression failed: 19.70 mean coins and 1.0% completion versus
43.77 and 48.4% for the selected Stage-1 control. Opponent training was
therefore correctly stopped at the registered gate. The next experiment should
preserve Stage-1 navigation through initialization/distillation and a mixed
coin/crate curriculum before introducing opponents. See
[`combat_fqi_history_antistag_agent.md`](combat_fqi_history_antistag_agent.md).

## Stagnation feature with history 8/16, 600-round extension completed

The stagnation feature was retrained for 600 rounds with history windows 8 and
16. On development boards, stagnation+history-8 peaked at round 300 (45.25
coins, 54.2% completion), while stagnation+history-16 peaked at round 400--500
(43.69--44.25 coins, 41.7--45.8% completion). Both round-600 checkpoints
degraded sharply. All runs had zero invalid actions and deaths. Held-out
evaluation of the development-optimal checkpoints is required before selection.
See [`tree_fqi_history_stagnation_600round.md`](tree_fqi_history_stagnation_600round.md).

Held-out confirmation selected stagnation + history-8 at round 300: 43.27 mean
coins, 50.0% completion, and 162 repeated states at 400 steps. Stagnation +
history-16 reached 42.90/39.1% at round 400 and 40.70/32.8% at round 500, so
the longer history window did not generalize better.

## Loop-focused variant comparison completed

Held-out evaluation of five loop-focused variants found that the
steps-since-last-coin feature reached 50.0% full completion with 162 repeated
states per game, while a 0.05 revisit penalty reached 49.5% completion with
165 repeated states. The original history-8 baseline reached 25.5% completion
with 225 repeated states and had the highest mean coins (44.06 versus 43.27
for stagnation and 41.32 for revisit penalty). Longer history alone was weaker,
and the remaining-time feature hurt performance. See
[`tree_fqi_loop_variant_confirmation.md`](tree_fqi_loop_variant_confirmation.md).

## Compact DQN coin pilot completed

The compact DQN completed a three-seed, 300-round pilot with 400 training
steps per round and 100-step frozen evaluation. Its best mean was 29.71 coins
at round 200, falling slightly to 29.30 at round 300. Invalid actions were
zero, survival was 100%, and no game completed all coins. Under the current
evidence it trails the learned-history tree agent, although the protocols and
board sets are not identical. See [`dqn_coin_pilot.md`](dqn_coin_pilot.md).

## 600-round history-agent extension completed

The 400-step history agent was trained for 600 rounds across three seeds. On
the four-board development evaluation, mean coins peaked at 46.21 at round
300, while completion peaked at 43.8% at rounds 400 and 500. The round-600
checkpoint fell to 34.93 coins and 4.2% completion. Invalid actions and deaths
remained zero. Training longer is therefore useful for finding a better
checkpoint, but the final checkpoint is not automatically the best one. Fresh
16-board confirmation of rounds 300, 400, and 500 is the next step. See
[`tree_fqi_history_600round_pilot.md`](tree_fqi_history_600round_pilot.md).

Held-out confirmation selected round 400 as the balanced Stage 1 candidate: it
reached 41.65 coins and 49.0% completion at 400 steps. Round 300 had the
highest mean (44.06) but only 25.5% completion; round 500 reached 43.04 coins
and 47.9% completion. All held-out evaluations had zero invalid actions and
zero self-deaths.

## 400-step history-agent training pilot completed

The learned movement-history tree agent was retrained with the official
400-step episode horizon for 300 rounds across three independent seeds. On a
four-board frozen check with all four seats, the round-300 mean was 43.92 coins
(42.62, 44.81, and 44.31 by seed), versus 41.45 for the earlier 100-step-
trained agent on a different 16-board fresh set. Fifteen of 48 games completed
all coins; invalid actions and deaths were zero. This is promising but not a
definitive comparison because the board sets differ. The next step is matched
16-board evaluation on boards 22016--22031 at both 100 and 400 steps.

That matched confirmation is now complete. The 400-step-trained checkpoints
scored 39.80 versus 36.26 at a 100-step budget and 44.06 versus 41.45 at a
400-step budget. Invalid actions were zero throughout. However, 400-step full
completion was 25.5% for the new agent versus 33.3% for the old one, so the
longer-horizon training improves mean collection but not every success metric.

## Combat crate pilot completed

The approved three-seed, 300-round, 400-step crate pilot completed. Round-300
mean coins were 3.34, 6.41, and 9.88 (aggregate 6.54), below the matched
untrained mean of 11.56. All 96 round-300 frozen games had zero invalid actions
and survived, so the safety gates passed while the learning-improvement gate
failed. The initial combat tree/reward setup is therefore safe but not yet a
useful crate learner; opponent training is not justified from this result.
Raw records and checkpoints are in `experiments/combat_fqi_crate_pilot/`.
Details are in [`combat_fqi_agent.md`](combat_fqi_agent.md).

## Bomb-aware combat FQI implementation

The user approved implementing the first bomb-aware agent. Added a separate
`combat_fqi_agent` with six actions, blast and danger calculation, a time-aware
escape mask, combat features, fitted-Q training, focused safety tests, and a
new `run_combat_training.py` runner for crate and opponent curricula. Ten tests
pass, and tiny crate/classic smoke runs exercised bombing, tree fitting, and
checkpoint evaluation without suicides or invalid actions. These are software
checks only: no substantial training or scientific comparison has been run,
and no final model has been selected or copied into the agent directory. See
[`combat_fqi_agent.md`](combat_fqi_agent.md).

## Learned movement-history agent: promising, Stage 1 decision remains

Added a simple tree-FQI variant whose only new inputs are preceding action and
the capped count of recent visits to the present tile since coin progress. It
lets learning condition on movement history without prescribing an escape action.
A two-round smoke run and the matched three-seed, 300-round pilot passed. Its
round-300 development mean was 35.76 versus 23.52 for base tree FQI. On 16 fresh
boards, it scored 36.26 versus 25.13 at 100 steps and completed 64/192 games
within 400 steps, versus 1/192 for base FQI. This supports learned history as a
useful Stage 1 change, though most fresh games still did not complete. Agree on
the Stage 1 exit criterion before moving to crates and survival. See
[tree_fqi_history_agent.md](tree_fqi_history_agent.md).

## Tree-based fitted Q implementation

The user approved implementing tree-based fitted Q iteration in a simple style.
Added `tree_fqi_agent` with one regression tree per action, existing distance
features and legal-action masking, and round-end batch fitting. A two-round
smoke run and direct numerical checks passed. The approved three-seed, 300-round
pilot then reached a final mean of 23.52 coins within 100 steps, versus 20.00
for the masked-Q pilot on the same reused development schedule. This is promising
exploratory evidence. Fresh-board frozen-policy evaluation then reached 26.33
for tree FQI versus 22.18 for masked Q-learning at 100 steps (16 new boards,
three fixed checkpoints per method); this supports tree FQI as the current
stronger Stage 1 candidate, with the fixed-checkpoint and small-run limitations.
On another fresh 16-board set, the tree-FQI loop wrapper increased 100-step
collection from 24.34 to 31.79 and 400-step collection from 24.35 to 35.65.
It completed 13 of 192 400-step games, but is a history-based hybrid policy,
not improved learning. Horizon remains to be agreed before another training
variant. See
[tree_fqi_agent.md](tree_fqi_agent.md).

## SARSA loop follow-up and test removal

At the user's request, removed `src/test.py` and `src/test_loop_policy.py`;
earlier mentions of passing tests below are historical. Temporary recovery
copies are in `/tmp/bomberman-removed-tests-TiVPOB/`. No unittest files remain
in `src/`; this follow-up was checked directly and through saved game results.

Added the same frozen-policy loop intervention to linear SARSA. Across the
same 96 development games, mean coins rose from 12.36 to 22.33. All three
checkpoints improved, but their loop-variant means were 27.47, 31.28, and 8.25,
so a weak run remains. Repeated states fell from 72.58 to 48.45, with 11.44
interventions per game and zero invalid actions. Original weights and results
were preserved; no retraining or fresh-seed confirmation was performed.

Masked Q-learning with loop escape remains the strongest measured hybrid in
this comparison (30.09 mean coins). Details and reproduction commands are in
[loop_breaking_pilot.md](loop_breaking_pilot.md).

## Latest result: frozen-policy loop intervention

The user approved a cutoff audit and a loop-breaking comparison, without
retraining. The three final masked-Q checkpoints improved from 20.00 to 30.09
mean coins within 100 steps when a repeated-position detector occasionally
substituted another legal movement. Each checkpoint improved; repeated states
fell from 56.32 to 29.89 per game, with 6.17 interventions and zero invalid
actions on average. This is a promising hybrid policy on reused development
boards, not an improvement to the learned Q-tables or a confirmed final winner.

The audit confirmed both learners treat the 100-step training boundary as
terminal while omitting remaining time from their features. All 900 episodes
of each learner ended at that boundary. Whether to use a time-aware finite
horizon or treat 100 steps as truncation requires an explicit objective choice;
no training changes have been made. The issue is not a demonstrated cause of
the loops. Details and results: [loop_breaking_pilot.md](loop_breaking_pilot.md).

Proposed next decisions: confirm the frozen loop variant on fresh boards and
choose the intended training horizon before another learning experiment.

## Current position: second learning model pilot completed

Implemented a self-contained linear SARSA(lambda) agent with the masked
Q-table's state information and legal moves. The approved matched three-seed,
300-round pilot finished: final mean coins at 100 steps were 12.36 for SARSA
versus 20.00 for masked Q-learning. SARSA run means were 4.69, 26.38, and 6.03;
variation and repeated-state counts were substantially higher. This initial
configuration did not improve the baseline. It is not a rejection of the whole
model family or a fresh-seed confirmation result.

Details, protocol, validation, and artifacts: [linear_sarsa_pilot.md](linear_sarsa_pilot.md).
Recommendation: retain masked Q-learning as the working baseline and discuss
a focused failure diagnosis before more tuning, features, or model families.
Moving to bomb survival remains a separate stage decision. Earlier "next step"
paragraphs below are historical; benchmark, Q-table, masking, and this SARSA
pilot have been completed. Symmetry and fresh-seed confirmation remain pending.

## Feature representation pilot

Tested four feature modes with the same Q-learning settings, three training
seeds, 300 rounds per seed, and 100-step evaluation at rounds 0, 100, 200, and
300. The only changed component was `state_to_features`; all modes used the
same board and evaluation seeds.

| Feature mode | State contents | Round-300 mean coins | Final table sizes |
|---|---|---:|---:|
| `compact` | local walls, nearest-coin direction | 17.47 | 60, 60, 60 |
| `position` | absolute position, local walls, direction | 11.10 | 1102, 1097, 1102 |
| `distance` | local walls, direction, distance and remaining-coin buckets | **18.18** | 374, 389, 402 |
| `rich` | absolute position plus all distance features | 7.55 | 2884, 2821, 2790 |

The `distance` mode is the current feature choice because it had the highest
pilot mean and added useful distance/progress information without the large
table of the absolute-position modes. Its three final run means were 22.72,
8.56, and 23.25, so the result is unstable and needs a fresh confirmation
comparison. The `position` mode did not help despite distinguishing locations;
the table became sparse. The `rich` mode was worse still. These are exploratory
results, not evidence that distance features are universally best.

The default `q_table_agent` mode is now `distance`. All four experiment
directories are preserved under `experiments/q_table_features_*`; the original
compact baseline remains available. Next decision: confirm `distance` against
`compact` with a fixed independent-run budget, then investigate legal-action
masking or symmetry as separate changes.

## Latest training pilot

The first Q-table pilot is complete: three seeds, 300 rounds each, and 100-step
development evaluation at rounds 0/100/200/300. Mean coins increased from 4.91
untrained to 17.47 at round 300, with substantial variation between runs; the
matched supplied coin collector achieved 44.12. Training records, checkpoint
selection, validation, and interpretation are in [q_table_pilot.md](q_table_pilot.md).
`src/run_training.py` now records independent training runs and evaluates frozen
checkpoints. Next: discuss the observed instability before another experiment.

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
python src/run_benchmark.py --agents peaceful_agent coin_collector_agent --seeds 10 11 12 --output experiments/coin_pilot
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

## Legal-action masking variant

Created `src/agent_code/q_table_masked_agent/` as a separate learner. It keeps
the selected distance features, rewards, Q-learning settings, training budget,
and evaluation seeds from the `q_table_agent` pilot. Its only learning-policy
change is to remove blocked movement actions during exploration, greedy choice,
and next-state Q-value bootstrapping. The original `q_table_agent` remains the
feature baseline.

The matched three-seed pilot is in
`experiments/q_table_masked_pilot_confirm/`. Mean coins within 100 steps:

| Round | Distance features | Masked agent |
|---|---:|---:|
| 0 | 4.91 | 9.66 |
| 100 | 13.71 | 13.17 |
| 200 | 16.98 | 17.00 |
| 300 | 18.18 | **20.00** |

Final masked run means were 21.88, 18.56, and 19.56 coins. Its final Q-table
sizes were 405, 383, and 422 states, close to the unmasked distance branch
(374, 389, 402). Masking improved the final pilot mean by 1.82 coins without
substantially increasing the table. The first masked evaluation also started
higher because random tie-breaking could not choose blocked moves.

The masked agent remains far below the supplied `coin_collector_agent` (43.08
coins at 100 steps). This pilot supports masking as a useful improvement, not
as a complete solution. The partial directory `experiments/q_table_masked_pilot/`
was a failed launch with a feature-mode mismatch and is retained as a failed
run; it is excluded from the comparison above. Current branches are
`q_table_agent` as the feature baseline and `q_table_masked_agent` as the best
pilot result. Further work should diagnose legal movement cycles and test the
masked branch on fresh seeds before selecting a final learner.

Direct replay of the final seed-0 masked checkpoint on its 32 development games
recorded 21.875 mean coins and zero invalid actions. This verifies the mask is
active in the policy; it does not explain the remaining legal movement cycles.

## Master-branch navigation diagnosis incorporated

The master-branch replay diagnosis found that base policies ended mainly in
two-position cycles or WAIT traps, with feature aliasing between conflicting
nearest-coin route choices. The retained history agent also shows terminal
no-progress cycles and route-choice aliasing. These results keep the project in
Stage 1 and motivate focused representation experiments before advancing to
crates and bombs. Details are in
[`tree_navigation_diagnosis.md`](tree_navigation_diagnosis.md) and
[`tree_fqi_history_loop_diagnosis.md`](tree_fqi_history_loop_diagnosis.md).
