# Stagnation feature with longer movement history

The steps-since-last-coin feature was retrained for 600 rounds with 400-step
episodes using history windows of 8 and 16 positions. Each setting used three
independent seeds and four development evaluation boards.

| Variant/checkpoint | Mean coins | Completion | Repeated states |
|---|---:|---:|---:|
| Stagnation + history-8, round 300 | **45.25** | **54.2%** | **146.4** |
| Stagnation + history-8, round 400 | 40.85 | 25.0% | 233.8 |
| Stagnation + history-8, round 500 | 41.23 | 18.8% | 252.0 |
| Stagnation + history-8, round 600 | 37.15 | 16.7% | 266.0 |
| Stagnation + history-16, round 300 | 40.90 | 22.9% | 238.6 |
| Stagnation + history-16, round 400 | 43.69 | **45.8%** | 174.2 |
| Stagnation + history-16, round 500 | 44.25 | 41.7% | 188.3 |
| Stagnation + history-16, round 600 | 28.17 | 14.6% | 297.2 |

All evaluations had zero invalid actions and zero self-deaths. Longer training
again did not make the final checkpoint best. The development-board candidates
are history-8 at round 300 and history-16 at rounds 400--500. These must be
compared on the held-out boards before selection.

Artifacts are in
`experiments/tree_fqi_history_stagnation_600round/`.

## Held-out confirmation

The development-selected checkpoints were evaluated on boards 22016--22031,
all four seats, at both step budgets:

| Candidate | 100-step coins | 400-step coins | 400-step completion | Repeated states |
|---|---:|---:|---:|---:|
| Stagnation + history-8, round 300 | 38.26 | **43.27** | **50.0%** | **162.2** |
| Stagnation + history-16, round 400 | 37.89 | 42.90 | 39.1% | 194.4 |
| Stagnation + history-16, round 500 | 36.24 | 40.70 | 32.8% | 223.4 |

All held-out evaluations had zero invalid actions and zero self-deaths. The
history-8 candidate generalizes best and remains the current Stage 1 candidate;
the longer history window did not improve held-out completion.
