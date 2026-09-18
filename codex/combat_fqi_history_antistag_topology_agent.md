# Combat FQI history/anti-stagnation topology variant

Sahand was here.

## Purpose

This is a controlled ablation of
`combat_fqi_history_antistag_agent`. It keeps the same reward, safety mask,
history features, tree settings, seeds, and mixed curriculum, then appends a
compact local-topology representation. The original 32-feature agent remains
unchanged as the control.

## Added representation

The topology block has 14 values:

- a flattened 3x3 patch of walls, free tiles, and crates around the agent;
- the number of free adjacent tiles;
- the number of adjacent crates;
- the number of free tiles within Manhattan radius two;
- a dead-end indicator;
- a straight-corridor indicator.

The resulting vector has 46 values. Dynamic bomb and explosion state continues
to be handled by the shared audited safety module and is not duplicated in the
topology block.

Focused topology, history, and safety checks pass (14 tests). A two-round
loot-crate smoke run also completed model fitting, checkpoint serialization,
and fresh-process evaluation.

## Mixed-curriculum pilot

The substantial run used three seeds, 300 rounds, 400 steps, alternating
Coin Heaven and loot-crate training episodes, and the same eight evaluation
boards and four seats as the 32-feature control. Results are archived in
`experiments/combat_fqi_history_antistag_topology_mixed_pilot/`.

| Checkpoint | Coin Heaven coins | Completion | Loot-crate coins | Crates | Repeated states |
|---:|---:|---:|---:|---:|---:|
| 0 | 21.59 | 0.0% | 11.56 | 44.66 | 257.3 / 202.1 |
| 100 | 36.82 | 1.0% | 2.62 | 10.59 | 308.5 / 368.0 |
| 200 | 36.52 | 3.1% | 11.33 | 34.14 | 306.0 / 295.6 |
| 300 | 37.79 | 1.0% | 8.62 | 26.23 | 308.4 / 318.8 |

The values in the repeated-states column are Coin Heaven / loot-crate. All
checkpoints had 100% survival and zero invalid actions.

Compared with the 32-feature mixed control at round 300 (39.14 Coin Heaven
coins and 10.41 loot-crate coins), the topology variant reached 37.79 and
8.62. It did not improve completion or reduce looping, and should not replace
the control. The result does not show that local topology is useless in
general; it shows that this particular shallow-tree topology block needs a
better action-conditioned or spatial model before it is worth retaining.
