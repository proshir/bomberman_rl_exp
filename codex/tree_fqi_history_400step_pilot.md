# History agent with a 400-step training horizon

## Configuration

This pilot retrained `tree_fqi_history_agent` with the official 400-step
episode limit instead of the earlier 100-step training cutoff. It used three
independent training seeds (0, 1, and 2), 300 rounds per seed, checkpoints at
rounds 0, 100, 200, and 300, and frozen evaluation on board seeds 22000--22003
with all four starting seats. The 400-step pilot was run as three parallel
CPU processes; no GPU was used.

Each checkpoint evaluation contains 16 games (four boards × four seats).

## Results

| Training checkpoint | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| 0 | 23.00 | 23.00 | 23.00 | 23.00 |
| 100 | 26.62 | 24.69 | 39.00 | 30.10 |
| 200 | 35.38 | 32.56 | 30.75 | 32.90 |
| 300 | 42.62 | 44.81 | 44.31 | **43.92** |

At round 300, 15 of 48 games completed all coins (31.25%). There were zero
invalid actions and zero deaths in every checkpoint evaluation. Mean repeated
states at round 300 were 225.50, 221.00, and 171.31 for seeds 0, 1, and 2.

The round-300 mean is higher than the earlier 100-step-trained history agent's
41.45 mean on its 16-board, 400-step fresh evaluation. The board sets are not
identical, however, so this is encouraging evidence rather than a definitive
comparison. A matched 16-board frozen comparison is still required before
claiming an improvement.

## Interpretation

Training with the full game horizon appears promising: performance continued
to improve through round 300 and the final three seeds were consistent. The
next validation should evaluate these three round-300 checkpoints on the same
16 fresh boards (22016--22031) used by the earlier confirmation, with both
100- and 400-step budgets. Preserve the 100-step-trained agent as the baseline.

## Matched fresh-board confirmation

The matched comparison used boards 22016--22031, all four seats, and the three
round-300 checkpoints from each training protocol. The 400-step-trained agent
improved mean coins at both horizons:

| Evaluation budget | Old 100-step training | New 400-step training | Difference |
|---|---:|---:|---:|
| 100 steps | 36.26 | **39.80** | +3.54 |
| 400 steps | 41.45 | **44.06** | +2.60 |

Invalid actions were zero for both protocols. At 400 steps, the old agent
completed 33.3% of games and the new agent completed 25.5%. The new agent's
400-step completion rate is therefore not yet an unconditional improvement,
despite its higher mean coin count. The board-paired 95% intervals for the
mean-coin differences were [2.21, 4.80] at 100 steps and [1.13, 3.94] at 400
steps, conditional on the six frozen checkpoints.

Artifacts are in
`experiments/tree_fqi_history_horizon_confirmation_history_vs_history_v2/`.
