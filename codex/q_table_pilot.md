# First Q-table training pilot

## Scope and protocol

User approved a small training pilot and the shared training runner. The pilot
uses the proposed 100-step screening budget. This is exploratory development
evaluation, not a final model comparison or hyperparameter search.

Run from the project root:

```bash
env/bin/python src/run_training.py --seeds 0 1 2 --rounds 300 --max-steps 100 --eval-every 100 --eval-seeds 10000 10001 10002 10003 10004 10005 10006 10007 --output experiments/q_table_pilot
```

Three independent training runs use 300 rounds each, with board seeds
1000--1299, 1300--1599, and 1600--1899 respectively. Training corners rotate
each round. Each run's action RNG is initialized once using its training seed.
Learning rate 0.1, discount 0.95, epsilon 1.0 decaying by 0.995 each round to a
minimum of 0.05, coin reward 1.0, step cost 0.01. These are untuned settings.

Checkpoints at rounds 0, 100, 200, and 300 each have 32 evaluation games:
eight separate development board seeds, action seed 0, and all four corners.
Evaluation is greedy with random tie-breaking. The same development schedule
is reused to observe progress. Supplied-agent references use that schedule in
`experiments/q_table_pilot_references/`.

`src/run_training.py` saves configurations, source hashes and Git version,
per-round results, checkpoint files, evaluation games, and learning curves.
`src/run_benchmark.py --model-path PATH` evaluates a selected checkpoint.
No best-checkpoint selection was used for the final-round figures below.

## Measured results

Mean coins collected within 100 steps:

| Training rounds | Seed 0 | Seed 1 | Seed 2 | Mean across runs |
|---|---:|---:|---:|---:|
| 0 | 4.91 | 4.91 | 4.91 | 4.91 |
| 100 | 17.28 | 17.31 | 23.88 | 19.49 |
| 200 | 10.03 | 19.19 | 21.53 | 16.92 |
| 300 | 12.84 | 17.38 | 22.19 | 17.47 |

Matched supplied references: random 1.41, peaceful 5.50, coin collector 44.12.
All final Q-tables contain 60 visited feature states. Training used 90,000
total interactions. Evaluation intervals in each summary describe a fixed
checkpoint across sampled boards, not uncertainty across learning runs.

All three runs improve over the untrained policy on these development boards,
but learning is uneven and the final policies remain below the coin collector.
Three runs are a small pilot; no formal superiority or optimality claim is made.
**Proposed next discussion:** inspect looping and invalid moves, then decide
whether to proceed with the planned symmetry comparison or investigate the
current representation and settings. No further run is approved by this note.

## Implementation validation

The framework reports the final action through both training callbacks. The
agent now replaces the intermediate update with one terminal update and counts
that action once. A numerical check confirms terminal learning has no future
value term. A two-round smoke test verified train/save/load/evaluate behavior.
All 900 pilot round records were checked against the reward and epsilon formulas;
all 12 checkpoints have the planned evaluations, disjoint training/development
board seeds, and finite Q-values. No separate agent test file was added.

## Behaviour diagnosis

Replayed all 288 evaluation games for the nine trained checkpoints with
temporary instrumentation. Every coin count matched the saved result exactly.
A repeated state here means revisiting the same position with the same remaining
coins. No evaluation action encountered a feature tuple absent from its table.

| Seed / round | Mean coins | Invalid moves per game | Repeated states per game |
|---|---:|---:|---:|
| 0 / 100 | 17.28 | 0.12 | 63.16 |
| 0 / 200 | 10.03 | 52.50 | 79.00 |
| 0 / 300 | 12.84 | 0.12 | 72.47 |
| 1 / 300 | 17.38 | 0.06 | 61.72 |
| 2 / 300 | 22.19 | 0.03 | 52.09 |

The round-200 seed-0 table prefers RIGHT in feature state
`(0, 1, 0, 1, -1, -1)`, even though right is blocked. Its Q-value is 3.4869,
above UP 3.3345 and DOWN 3.4451. Since the action does not change the state,
greedy evaluation repeats it. On board 10006, seat 3, this gives 87 invalid
actions and 6 coins, compared with 45 coins at round 100. Only four of the
60 shared table states changed their greedy action sets between those checkpoints.

Movement cycles remain when invalid moves are rare. At round 300, seed 0,
board 10006, seat 1, the policy alternates UP from (1, 11) and DOWN from (1, 10),
with the same target coin at (3, 12). It has collected only one coin before
entering the cycle. Seed 2 has a similar two-position cycle on board 10003,
seat 3, after collecting two coins.

Training still uses random actions (epsilon approximately 0.37 down to 0.22
over rounds 201--300), whereas evaluation is greedy. Random actions can escape
these cycles, so improving training coin counts do not imply improving greedy
evaluation. The compact features merge distinct positions and coin layouts;
this and finite-sample Q-value errors are plausible contributors, not separately
established causes. More training, different features, and changed learning
settings have not been tested by this diagnosis.

No policy or training setting was changed. **Proposed next experiment:** discuss
a separate legal-action-mask variant for the wall-hitting failure. It would need
consistent masking in exploration, greedy selection, and bootstrap targets.
That change alone cannot eliminate cycles made entirely of legal movements;
those need a separate investigation before choosing a representation change.

## Feature variant prepared

The existing `q_table_agent` now uses a richer state representation for the next
pilot: absolute `(x, y)` position, four blocked-neighbour flags, nearest-coin
direction signs, a maze-distance bucket (`0`, `1`, `2`, `3--4`, `5--7`, `8+`),
and a remaining-coins bucket (`0`, `1--5`, `6--15`, `16--30`, `31+`). The
Q-learning rule, rewards, hyperparameters, and experiment protocol are unchanged.
The previous baseline results remain in `experiments/q_table_pilot/`.

A two-round training/save/load/evaluation smoke test passed with the new state
keys. No feature-variant learning result has been measured yet.

The subsequent feature pilot tested `compact`, `position`, `distance`, and
`rich` modes. `distance` was selected as the current default after the highest
round-300 mean (18.18 coins), ahead of compact (17.47), while position (11.10)
and rich (7.55) suffered from larger tables and sparse visits. The full table
and seed results are recorded in the roadmap and preserved in
`experiments/q_table_features_*`. The distance result is exploratory because
one of its three seeds reached only 8.56 coins.
