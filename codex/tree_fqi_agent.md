# Tree-based fitted Q iteration: implementation

## Accepted scope

The user approved implementing another Stage 1 approach, tree-based fitted Q
iteration, in a simple style consistent with the existing agents. A small
validation run is included; a comparative training pilot is not yet approved.

## Implementation

`src/agent_code/tree_fqi_agent/` contains callbacks, training, and a copy of the
existing distance-feature functions. The agent is self-contained and uses
scikit-learn's `DecisionTreeRegressor`, with one tree per movement/WAIT action.
There is no BOMB action or loop-breaking intervention. This is a solo coin-heaven
agent; its wall/crate mask is not a bomb-survival or opponent-aware mask.

State information, legal-action masking, coin reward, step cost, discount, and
epsilon schedule match the masked baseline. Unfitted actions have value zero.
Evaluation is greedy with seeded random tie-breaking.

Initial implementation defaults, not tuned settings:

- Keep the latest 30,000 transitions.
- At each round end, perform five fitted-Q sweeps, starting from the current trees.
- Each sweep computes reward plus discounted maximum legal next-action value
  using the previous trees, then refits each action tree on its observed samples.
- Use maximum tree depth 8 and minimum leaf size 5.

The final transition is stored once, with no bootstrap. This preserves existing
terminal-cutoff semantics; the training-horizon decision remains open. Trees
share values across feature combinations, but this comparison also changes the
learning procedure, so improvements cannot be attributed only to approximation.

The shared training runner now saves tree checkpoints, tree node counts, buffer
sizes, and the scikit-learn version. Checkpoints contain only inference trees;
they are not full resumable training state.

## Validation results

Completed two 10-step training rounds and frozen evaluation at rounds 0 and 2
(one board, all four corners). Artifacts: `experiments/tree_fqi_smoke/`.
The buffer held exactly 10 then 20 transitions, rewards matched coins minus
step cost, loaded predictions were finite, and final evaluation had zero
invalid actions. Feature code exactly matches the SARSA feature file.
A direct numerical check confirmed legal-only bootstrap maximization and
reward-only terminal targets. No new unit-test files were added.

These are implementation checks, not learning or performance evidence.
The installed Python and scikit-learn 1.9.1 were used; `env/` is absent.

Command (choose a fresh output directory for another run):

```bash
python src/run_training.py --agent tree_fqi_agent --seeds 0 --rounds 2 --max-steps 10 --eval-every 2 --eval-seeds 20000 --output experiments/tree_fqi_smoke
```

## Measured pilot results

The approved matched pilot completed: three independent training seeds, 300
coin-heaven rounds per seed, 100 steps per round, and frozen evaluation at
rounds 0, 100, 200, and 300. Each evaluation has eight development boards,
action seed 0, and four corners (32 games per checkpoint). Artifacts are in
`experiments/tree_fqi_pilot/`.

| Training rounds | Seed 0 | Seed 1 | Seed 2 | Mean across runs |
|---|---:|---:|---:|---:|
| 0 | 9.66 | 9.66 | 9.66 | 9.66 |
| 100 | 17.91 | 13.50 | 15.91 | 15.77 |
| 200 | 22.59 | 20.41 | 21.41 | 21.47 |
| 300 | 22.72 | 22.34 | 25.50 | **23.52** |

The final-run sample standard deviation was 1.72 coins. All final frozen
policies made zero invalid moves; repeated states averaged 51.91, 53.63, and
46.44 per game for seeds 0--2. The accumulated training loop took 38.4--39.1
seconds per run; this excludes frozen-policy evaluation subprocesses.

The final mean exceeds the prior masked-Q pilot's 20.00 coins under the same
development schedule, and it is more stable than the initial SARSA result
(12.36). This is exploratory development evidence only: both the function
approximator and learning procedure differ from the tabular baseline, the
evaluation boards have been reused during development, and three runs are too
few for a strong comparative claim. It does not yet identify a final agent or
show fresh-board generalization.

Validation checked all 900 round records: every round used 100 steps; the
buffer grew by exactly 100 transitions per round; rewards equal collected coins
minus 0.01 per step; tree sizes were positive; and all checkpoint predictions
were finite. The agent made no invalid actions in final evaluation.

## Proposed next decision

Freeze the current trees and confirm tree FQI, masked Q-learning, and any
selected loop variants on fresh boards before selecting a Stage 1 candidate.
The separate choice of training horizon remains needed before another training
variant or a move to crates and survival.
