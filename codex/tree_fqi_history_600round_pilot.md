# 600-round history-agent pilot

The 400-step-trained history agent was extended from 300 to 600 rounds for
three independent seeds. Checkpoints were evaluated on four boards (23000--
23003), all four seats, with a 400-step budget. Each checkpoint contains 48
games across the three seeds.

| Checkpoint | Mean coins | Completed games | Completion rate |
|---|---:|---:|---:|
| 0 | 21.88 | 0/48 | 0.0% |
| 100 | 35.54 | 6/48 | 12.5% |
| 200 | 37.31 | 16/48 | 33.3% |
| 300 | **46.21** | 11/48 | 22.9% |
| 400 | 43.60 | **21/48** | **43.8%** |
| 500 | 43.52 | **21/48** | **43.8%** |
| 600 | 34.93 | 2/48 | 4.2% |

All evaluations had zero invalid actions and zero deaths. The extra training did
not improve monotonically: the round-600 checkpoint degraded substantially,
while rounds 300--500 were the strongest region. This indicates checkpoint
selection and possible overfitting/instability are important; the final
checkpoint should not be selected automatically.

These are development-board results, not the final fresh-board comparison. The
next validation should evaluate the round-300, round-400, and round-500
checkpoints on the 16 held-out boards (22016--22031) before selecting a model.

## Held-out confirmation

The selected checkpoints were evaluated on boards 22016--22031, all four
seats, with 100- and 400-step budgets (64 games per training seed and budget).

| Checkpoint | 100-step coins | 400-step coins | 400-step completion |
|---:|---:|---:|---:|
| 300 | 39.80 | **44.06** | 25.5% |
| 400 | 36.93 | 41.65 | **49.0%** |
| 500 | 38.51 | 43.04 | 47.9% |

All held-out evaluations had zero invalid actions and zero self-deaths. Relative
to the old 100-step-trained history agent (41.45 coins and 33.3% completion at
400 steps), round 400 is the best balanced checkpoint: it raises completion
substantially while keeping mean collection slightly higher. Round 300 has the
highest mean coin count but a lower completion rate.

Artifacts are in
`experiments/tree_fqi_history_600round_heldout_confirmation/`.
