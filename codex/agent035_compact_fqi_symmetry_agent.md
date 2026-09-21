# Agent 035 corrected compact FQI with symmetry

Date: 21 September 2026. Status: implemented and integration-smoke-tested;
no substantial training or performance comparison.

Agent 035 is the corrected, symmetry-enabled successor to Agent 034. Agent 034
is intentionally retained unchanged as the historical schema-v2 baseline so
its existing checkpoints keep their original meaning. Agent 035 lives in
`src/agent_code/Agent_035_compact_fqi_symmetry_agent/` and is self-contained:
it does not import another agent at runtime.

## Representation and corrected contracts

The learned input remains 28 numeric values representing the design's 22
logical entries. Direction order is UP, RIGHT, DOWN, LEFT: coin route costs and
flags occupy indices 0--7, crate-bomb route costs and flags 8--15, bomb context
16--18, the seven-way previous-action one-hot 19--25, and recent visits plus
progress bucket 26--27. The feature schema is
`compact-routes-history-blast-range-v3`.

Unlike Agent 034, crate candidates include every free tile within the complete
bomb-power ray of a crate, subject to stone walls. Candidates must be in the
current static free-tile component, unoccupied at present, have positive crate
yield under the engine-aligned blast model, and admit a modeled escape after an
immediate hypothetical bomb. This is still a route proxy: it does not simulate
the future journey or promise the tile remains feasible on arrival.

The final engine event list replaces, rather than concatenates with, the
already observed event list for the last transition. Each terminal event is
therefore rewarded exactly once, and terminal FQI targets have no bootstrap.

## Symmetry and fitted updates

Training augments the selected fitted dataset with the eight unique elements
of the square's D4 symmetry group. Every transform consistently permutes:

- the four directional coin costs and four coin reachability flags;
- the four directional crate costs and four crate reachability flags;
- the directional previous-action categories;
- the executed action; and
- both current and next stored allowed-action masks.

WAIT, BOMB, START, bomb context, visits, and progress are invariant. The
augmentation is applied at fitting time, not stored eightfold in replay.
Agent 035 otherwise retains the declared six depth-8 action-specific trees,
minimum leaf size 5, gamma 0.95, five frozen-target passes every 20 rounds, and
masked epsilon-greedy collection.

Replay is bounded independently per complete `(scenario, lineup)` tag. Runner
weights are interpreted as scenario shares unless a complete
`scenario|lineup` weight is provided; scenario weight is divided across its
available lineups so merely adding lineups does not multiply that scenario's
share. The selected raw sample count and per-tag counts, allowed/executed action
support, tree fitting time, feature/safety/action time, maximum action latency,
and action count are recorded by the runner.

## Checkpoint and resume contract

Agent 035 checkpoints validate the feature, safety, reward, history, symmetry,
and training-state schemas. Frozen inference reads only the six trees. A full
training continuation also restores tagged replay, epsilon, completed rounds,
NumPy generator state, environment interactions, and cumulative timing/support
diagnostics. The runner supplies the latest immutable episode checkpoint and
continues the original board schedule. Agent 034 checkpoints are deliberately
incompatible with Agent 035.

## Validation

The following focused command passes 26 tests:

```text
python -m unittest -q test_agent_035 test_agent_034 test_combat_safety test_combat_route_features
```

Coverage includes the opposite-route witness, a two-cell crate-ray witness,
single-count terminal reward, all eight distinct direction permutations,
feature/action/current-mask/next-mask consistency, scenario-versus-lineup
weighting, full checkpoint round trip, masked nonterminal backups, safety, and
route contracts.

A real runner smoke completed 20 eight-step Coin Heaven rounds, crossed the
first augmented fit, wrote a 317-node six-tree checkpoint, and loaded it for
frozen evaluation with zero invalid actions. A separate one-round run resumed
in a new process for round 2; replay grew from 2 to 4 transitions and the
interaction counter continued from 2 to 4. On a dense synthetic crate board,
five feature calls took 0.026--0.029 seconds each on the test host. These checks
establish integration and budget plausibility only. No substantial learning
run, held-out comparison, exhaustive engine-equivalence proof, or official
single-thread latency distribution has been completed.

## 21 September 2026: final-checkpoint Classic gate

The episode-300 checkpoint from each of the three mixed-curriculum training
seeds was evaluated on the fixed Classic gate: eight fresh board seeds,
four seats, and separate peaceful, coin-collector, and rule-based opponents
(288 games total). Mean score across the three training seeds was 0.84 against
`peaceful_agent`, 2.96 against `coin_collector_agent`, and 2.64 against
`rule_based_agent`.

Survival was 100% / 91.7% / 70.8%, respectively, and mean invalid actions per
game were 0.00 / 0.19 / 0.94. The candidate averaged 0.08 kills and 0.14
suicides per game against the rule-based opponent. The result demonstrates
that the solo-trained checkpoint is safe in peaceful games but is not a strong
combat policy; the rule-based matchup in particular has substantial deaths and
invalid actions. Raw results are in
`/export/scratch/salitanl/agent035_compact_fqi_classic_300_20260921/`.
