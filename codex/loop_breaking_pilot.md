# Frozen-policy loop-breaking pilot and cutoff audit

## Follow-up: SARSA loop variant

The user requested removal of unit tests and extending the loop intervention
to SARSA. Removed `src/test.py` and `src/test_loop_policy.py`; their temporary
recovery copies are in `/tmp/bomberman-removed-tests-TiVPOB/`. Earlier test
results below are historical; these test files are no longer in the project.
`test_linear_sarsa.py` was already absent. Validation for this follow-up uses
direct checks and frozen-policy game replays, with no new unittest files.

Approved comparison: reuse final SARSA checkpoints for seeds 0, 1, 2 from
`experiments/linear_sarsa_pilot/`. Evaluate baseline and loop variant on the
same eight development boards, action seed 0, four corners, and 100 steps
(192 games total). Copy the Q-table variant's exact detector and intervention:
eight-position history, threshold three, reset on coin progress, uniformly
choose another legal movement, and use separate intervention randomness.
No tuning or retraining. Keep each agent self-contained. Preserve baseline
coin counts and checkpoint hashes. The same board-grouped uncertainty and
development-data limitations apply.

### Measured SARSA follow-up results

Completed 192 games, with no retraining, in
`experiments/linear_sarsa_loop_pilot/`.

| Frozen checkpoint | Original mean coins | Loop variant | Difference |
| --- | ---: | ---: | ---: |
| Seed 0 | 4.6875 | 27.46875 | +22.78125 |
| Seed 1 | 26.375 | 31.28125 | +4.90625 |
| Seed 2 | 6.03125 | 8.25 | +2.21875 |
| Mean | 12.36458 | 22.33333 | +9.96875 |

Mean repeated states fell from 72.58 to 48.45. Mean interventions were 11.44
per game, and both policies had zero invalid actions. The paired board-bootstrap
95% interval for the gain was [8.77, 11.24], conditional on these frozen
checkpoints and with the development-data limitations already described.

All checkpoints improved, but the third remained weak (8.25 coins, 77.31
repeated states, 13.88 interventions per game). Loop escape helps but does not
fully resolve its policy failures. Under this matched development schedule,
the masked-Q loop variant remains stronger in mean coins (30.09 versus 22.33).
Neither hybrid result establishes improved learning or final generalization.

Direct checks confirmed identical intervention functions between the Q-table
and SARSA variants, identical SARSA feature files, baseline action parity before
intervention, legal alternative selection, and unchanged in-memory weights.
All 96 original-policy game coin counts matched their saved results; all three
checkpoint hashes remained unchanged. No unittest files were introduced.

The shared comparison runner now accepts `--loop-agent` and derives the original
agent name from the saved pilot configuration. Reproduce with a fresh output:

```bash
python src/run_loop_comparison.py --baseline experiments/linear_sarsa_pilot --loop-agent linear_sarsa_loop_agent --output experiments/linear_sarsa_loop_pilot
```

## Approved scope

The user approved auditing the 100-step cutoff and comparing the frozen masked
Q-table checkpoints with a loop-breaking evaluation variant. No retraining,
reward shaping, or change to terminal learning targets is included.

## Fixed protocol, before evaluation

Use final round-300 checkpoints for training seeds 0, 1, 2 from
`experiments/q_table_masked_pilot_confirm/`. Evaluate both policies on boards
10000--10007, action seed 0, four corners, and 100 steps: 96 games per policy.
These are reused development boards and fixed trained policies, not a new
learning-method comparison or independent confirmation.

The variant retains the original Q-values, feature functions, action mask,
and greedy tie-breaking. Maintain the last eight pre-action positions since
the latest change in remaining coins. If the current position occurs at least
three times in that history, replace the proposed action with a uniformly
random legal movement other than the proposed action. Exclude WAIT. Clear the
history after an intervention; if no alternative exists, retain the proposal.
Reset history and counters each round. Use a separate random generator seeded
with the action seed so intervention draws do not consume policy RNG draws.

This threshold and response are initial choices, not tuned on these results.
The wrapper uses position history beyond the baseline features. It is a hybrid
policy intervention, not an improvement to the learned representation or weights.

Primary outcome: mean coins and paired differences, reporting each checkpoint
separately. Diagnostics: repeated states (same position and remaining coins),
invalid actions, and intervention counts. Preserve all results regardless of
direction. Reproduce original baseline coin counts and verify checkpoint hashes.
Bootstrap board averages for uncertainty conditional on these three policies,
grouping corners and checkpoints sharing a board. This is not uncertainty over
new training runs. No additional runs based on the observed effect.

## Cutoff audit: observed implementation

`run_training.py` assigns the requested 100-step screening budget to the game's
`MAX_STEPS`, so it also limits training episodes. The framework runs the action,
updates the world, sends `game_events_occurred`, then checks `time_to_stop`.
At the maximum step it calls `end_round`, which calls each learner's final
callback. There is no separate truncation signal passed to the learner.

Masked Q-learning restores the pre-update value for this final action, then
performs a terminal update with target equal to the immediate reward. SARSA
discards its pending transition and applies the final update with next value
zero. Both omit remaining time from their features. Neither update currently
distinguishes completing collection from reaching the training time limit.

All 900 saved masked-Q training rounds and all 900 SARSA rounds lasted exactly
100 steps. Maximum coins in a training round were 36 and 41 respectively, out
of 50. Thus the time boundary affected every episode in both pilots.

A numerical check of the masked-Q callbacks starts with Q(s, WAIT)=2,
max legal next Q=10, reward=-0.01, alpha=0.1, gamma=0.95. The intermediate
update yields 2.749; the final callback restores the old value and produces
1.799, counting the reward once. Changing only the step from 1 to 100 leaves
the encoded features identical. This confirms terminal handling and missing
time information, not a double-update error.

## Cutoff interpretation and proposed direction

For a task genuinely defined as collecting coins within 100 steps, a terminal
target at step 100 is appropriate, but exact finite-horizon values generally
depend on remaining time. The current representation cannot distinguish that
time dependence. It already omits other state information too.

If 100 steps is only a convenient cutoff while learning a longer-horizon task,
it should instead be treated as truncation: retain the appropriate future-value
term, while still ending the sampled episode. Genuine terminal events must
continue to have zero future value. These are different objectives, so neither
is an automatic correction without deciding which task we want to learn.

This distinction follows [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
The original Stage 1 plan describes 100 steps as a screening metric, alongside
eventual completion within 400 steps. **Recommendation, not yet implemented:**
separate the training horizon from the evaluation budget and decide explicitly
whether to learn the official finite-horizon task or an artificial shorter one.
Then test the chosen treatment as a separate training variant with a fixed
interaction budget. The audit does not establish how much this explains the
observed loops or instability. No training behavior was changed in this task.

## Measured evaluation results

Completed all 192 games: 96 per policy. No model was retrained. Artifacts,
protocol, source hashes, checkpoint hashes, per-game records, and summaries
are in `experiments/q_table_loop_pilot/`.

| Frozen checkpoint | Original mean coins | Loop variant | Difference |
| --- | ---: | ---: | ---: |
| Seed 0 | 21.875 | 29.46875 | +7.59375 |
| Seed 1 | 18.5625 | 30.9375 | +12.375 |
| Seed 2 | 19.5625 | 29.875 | +10.3125 |
| Mean | 20.00 | 30.09375 | +10.09375 |

Mean repeated states fell from 56.32 to 29.89 per game. The wrapper intervened
6.17 times per game on average. Both policies had zero invalid actions.
The paired board-bootstrap 95% interval for the mean coin difference was
[7.02, 13.07], conditional on these three frozen checkpoints. It groups all
four corners and all three checkpoints sharing each of eight boards; it does
not establish uncertainty over new training runs or fresh test boards.

All three saved policies benefited on average. This supports loop escape as
a useful intervention under this development protocol, but does not establish
why the learned policies loop or that their underlying learning improved.
The modified policy adds history and random interventions; it is a hybrid
variant. It remains below the supplied coin collector's roughly 43--44 coins
at this budget. No direct matched collector rerun was part of this experiment.

**Recommendation:** preserve the loop variant as a promising evaluation policy,
while retaining the original learned baseline. Discuss fresh-board confirmation
and the explicit training-horizon decision before additional training. Do not
attribute this gain to solving the cutoff issue: training was unchanged.

## Validation and reproduction

Six tests passed: five checks in `src/test_loop_policy.py` and the existing
benchmark smoke test. They cover legal interventions, unchanged table and
checkpoint bytes, coin/round resets, no-alternative fallback, policy RNG parity
before intervention, and the terminal-target numerical audit. An initial test
command also named `test_linear_sarsa`, which is absent from the current
workspace; the focused available suite was then run successfully.

All 96 original-policy coin counts match the preserved pilot games exactly.
All three source checkpoints retained their SHA-256 hashes after evaluation;
the final source files match the recorded protocol hashes.

From the project root:

```bash
python src/run_loop_comparison.py --output experiments/q_table_loop_pilot
```

Use a new output directory when reproducing. The runner records the protocol
before evaluating and writes its aggregate results to `summary.json`.
