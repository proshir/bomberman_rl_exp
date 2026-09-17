# Why coin navigation still fails

## Scope and evidence

User requested diagnosis, not another training experiment or policy change.
`src/diagnose_tree_navigation.py` replayed all 384 previously evaluated 400-step
games (192 base, 192 loop). Coin counts and repeated-state counts match the saved
400-step records exactly; the 100-step prefixes also match all 384 shorter games.
Checkpoint hashes were unchanged. No timeout warnings or errors occurred in the
diagnostic logs. Traces and summary are in `experiments/tree_navigation_diagnosis/`.
These boards are now diagnostic/development data, not a future untouched test set.

## Measured failure: persistent action cycles

All 192 base games end with a no-progress repeated pattern: 154 two-position
cycles and 38 stationary WAIT traps. Pattern classification compares the final
16 pre-action positions and remaining-coin counts against preceding positions.
191 games collected their last coin by step 100. Across 76,800 actions, 12,620
were WAIT (16.4%). Legal-action masking works, but legality does not imply progress.

Concrete example: checkpoint 0, board 20032, corner 0. At step 25, position
(7,5), 38 coins remain and a nearest coin is three moves away. RIGHT is the only
move reducing distance to any coin, but the tree prefers DOWN (6.711) over RIGHT
(6.587). At (7,6) it prefers UP (6.462) over DOWN (6.458), returning to (7,5).
The cycle persists to the end. The predicted values remain high despite the
realized no-coin cycle. An infinite sequence of -0.01 rewards discounted by 0.95
has return -0.20; this contrasts with the realized frozen-policy behavior, not
a claim that optimal full-state action values necessarily equal -0.20.

## Confirmed representation limitation

The features contain local blocked moves, the signs of displacement to one
nearest coin, a coarse distance bucket, and a remaining-count bucket. They omit
the exact displacement, surrounding coin layout, route geometry, and history.
They do compute maze distance, but discard most information about how the route
reaches that coin.

Among 727 observed feature tuples, 55 occur in situations with disjoint sets
of moves that reduce distance to the nearest available coin. A shared feature
tuple gives identical predictions, so no deterministic policy of these inputs
can choose a shortest-distance move in both situations. This is a demonstrated
loss of navigation information; nearest-coin progress is a diagnostic, not proof
of the action maximizing total 100-step score.

Example with the same checkpoint: input (0,0,0,0,1,1,3,4) occurs on board 20032,
corner 0, step 25 at (7,5), where only RIGHT reduces nearest-coin distance. It
also occurs on board 20033, corner 0, step 9 at (7,3), where DOWN or LEFT does.
The tree returns exactly the same action values in both situations. Increasing
tree depth alone cannot distinguish identical inputs.

## Why the wrapper only partly helps

Of 10,702 interventions with two subsequent recorded steps, 8,740 (81.7%) return
to the intervention's starting position without a coin. This is a descriptive
step-level count, not a set of independent statistical observations.

Example: loop checkpoint 0, board 20032, corner 0, three coins remain late in
the game. The wrapper moves UP from (15,15) to (15,14), decreasing nearest-coin
distance from 13 to 12. The learned policy immediately selects DOWN, even though
another UP would continue progress. The resulting six-step pattern repeatedly
escapes and returns. 129 of 192 loop games have a six-step no-progress pattern
at the end; other periodic patterns also occur. The intervention changes one
action but neither the following decisions nor the representation.

## Training factors: plausible contributors, not isolated causes

- Training still takes random exploratory actions about 22.3% of the time at
  round 300. Evaluation is greedy. Exploration can escape a learned greedy trap.
  This discrepancy is real, but its contribution has not been isolated.
- Trees share leaf predictions across inputs and fit bootstrapped targets.
  All 15 final trees reach the depth-8 cap. Approximation error and optimistic
  targets may reinforce bad rankings; hitting the cap does not prove that deeper
  trees would help. Identical-feature ambiguity remains regardless of capacity.
- The 100-step training boundary is treated as terminal while time is omitted.
  This is the already documented horizon issue. It may affect value estimates,
  but the cycles begin well before step 100, so it is not just a 400-step
  evaluation problem.

The supplied coin collector explicitly follows a path to a target. Our learner
receives a compressed description that can hide which route makes progress;
the collector's planning information is one relevant difference. No matched
collector rerun or causal training ablation was part of this diagnosis.

## Proposed next step, requiring agreement

Prioritize a focused representation experiment over another model family or
more random escape rules. Design inputs that preserve more coin/route geometry
(for example exact relative offsets and local coin layout), and check the
observed conflicting examples before training. Keep the model and other settings
fixed for attribution; agree on the horizon separately. The assignment requires
the action decision to remain learned, so avoid simply supplying a prescribed
best action. This is a proposal; no feature changes or training were performed.
