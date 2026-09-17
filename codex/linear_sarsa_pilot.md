# Linear SARSA pilot

## Approved scope and protocol

The user approved implementing a second learning model and comparing it with
masked tabular Q-learning, before further feature engineering. This is a Stage 1
development pilot, not a fresh-seed confirmation or a move to bomb survival.

Implement semi-gradient SARSA(lambda) with accumulating traces, following the
update described in Sutton and Barto, *Reinforcement Learning: An Introduction*,
Chapter 12: https://www.incompleteideas.net/book/bookdraft2018mar21.pdf.

Keep the masked baseline's eight categorical inputs: four blocked-neighbour
flags, nearest-coin direction signs, distance bucket, and remaining-coin bucket.
Use a separate one-hot encoding for each category and one bias (26 inputs,
nine active entries), divided by 3 to give unit norm. Each of five actions has
its own weights (130 parameters). No feature interactions or new information
are added. Feature and mask functions are copied unchanged into the new agent
so it remains self-contained for submission.

Learning rate 0.1, discount 0.95, epsilon 1.0 decaying by 0.995 per round to
0.05, coin reward 1.0, step cost 0.01. Initial, untuned trace decay: 0.8.
The learning rate matches numerically; its effect is not identical across a
normalized linear model and a Q-table. This compares complete methods, not
individual effects of the update rule, approximation, and traces.

Fixed budget: training seeds 0, 1, 2, 300 rounds each, 100 steps per round.
Use the baseline runner's board ranges starting at 1000, 1300, and 1600.
Evaluate rounds 0, 100, 200, 300 on development boards 10000--10007, action
seed 0, all four corners (32 games per checkpoint). Primary outcome: final
mean coins within 100 steps, with all three run means and learning curves.
Do not select the best seed or extend the run based on its results.

Compare with preserved `experiments/q_table_masked_pilot_confirm` results.
Replay its checkpoints with added diagnostics, checking that coin counts match.
Diagnostics: invalid actions and repeated pre-action states, where a state is
the position together with the remaining visible coins. A repeat measures
failure to make collection progress; it is not a count of distinct loops.
Development-board confidence intervals do not measure training-run uncertainty.

## Measured results

Completed the fixed budget: three runs, 300 rounds and 30,000 interactions per
run, with all four planned checkpoint evaluations. Artifacts are in
`experiments/linear_sarsa_pilot/`. The separate two-round validation run is in
`experiments/linear_sarsa_smoke/` and is excluded from learning results.

Final mean coins within 100 steps:

| Model | Seed 0 | Seed 1 | Seed 2 | Mean | SD across runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Masked Q-table | 21.875 | 18.5625 | 19.5625 | 20.00 | 1.70 |
| Linear SARSA(lambda) | 4.6875 | 26.375 | 6.03125 | 12.36 | 12.15 |

The final mean difference is -7.64 coins for SARSA. SARSA revisited an average
72.58 states per 100-step game, versus 56.32 for the masked Q-table. Both had
zero invalid actions in their 96 final evaluation games. SARSA's per-run repeat
means were 89.59, 42.69, and 85.47; the masked baseline's were 51.44, 60.63,
and 56.91. Repeated-state counts support continuing stagnation, not a claim
that a particular feature or algorithm component caused it.

SARSA seed 0 reached 23.0 coins at round 200 but fell to 4.69 at round 300.
Seed 1 reached 29.94 at round 100 and finished at 26.38. These intermediate
checkpoints were not substituted for the preselected final checkpoint.

This configuration did not improve the pilot baseline and was much less stable.
It does not establish that SARSA as a family is worse: only one encoding,
learning rate, trace setting, and small training budget were tested on reused
development boards. The additive representation shares weights across states
but cannot express every conjunction of features that a table can. This is a
possible limitation, not an experimentally isolated cause.

**Recommendation, not an approved new experiment:** retain masked Q-learning
as the working baseline. Keep this SARSA result as an unsuccessful initial
configuration. Before another model or feature expansion, discuss a focused
diagnosis of policy changes and movement cycles, and the remaining time for
bomb survival. No additional tuning or later-stage work was launched.

## Validation and reproduction

Four numerical tests cover feature/mask parity and normalized encoding,
eligibility-trace credit, bootstrapping with the executed next action, and
terminal handling plus frozen checkpoint loading. The existing benchmark
smoke test also passes. All five copied feature/mask functions match the
baseline's syntax trees exactly. All 900 SARSA round rewards agree with
coins minus the step cost, and all 12 checkpoints have finite 5-by-26 weights.

Final baseline checkpoints were replayed in
`experiments/q_table_masked_sarsa_diagnostics/`; all 96 coin counts exactly
match the original saved games. No baseline was retrained or overwritten.

The training runner now saves either linear weights or Q-tables and records
weight counts for linear agents. The benchmark additionally records invalid
actions and solo coin-navigation repeated states. Existing metrics are retained.

Run the pilot from the project root:

```bash
python src/run_training.py --agent linear_sarsa_agent --seeds 0 1 2 --rounds 300 --max-steps 100 --eval-every 100 --eval-seeds 10000 10001 10002 10003 10004 10005 10006 10007 --output experiments/linear_sarsa_pilot
```

Use a new output directory when reproducing. After the baseline diagnostic
replays, `python src/compare_sarsa_pilot.py` validates matched protocols,
rewards, finite checkpoints, and replay coin counts. It writes per-run results,
descriptive variability, and all learning curves to
`experiments/linear_sarsa_comparison/summary.json` and `learning_curves.csv`.
It also requires a new output directory for a rerun.
