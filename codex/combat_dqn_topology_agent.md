# Combat topology-feature vanilla DQN

Sahand was here.

## Purpose

This is a controlled neural ablation of `combat_dqn_agent`. It reuses the
14-feature local-topology block that was tested and rejected with shallow-tree
FQI. The 32-feature vanilla DQN remains unchanged as the control.

## Representation

The topology block appends exactly 14 values to the existing 32-dimensional
history/anti-stagnation vector:

- a flattened 3×3 local wall/free/crate patch (9 values);
- free adjacent tiles;
- adjacent crates;
- free tiles within Manhattan radius two;
- dead-end indicator;
- straight-corridor indicator.

The DQN therefore receives 46 inputs and still outputs six action values. The
reward, event handling, safety filter, callback interfaces, replay settings,
and vanilla target are unchanged. No topology-specific reward shaping or
Double-DQN logic is introduced.

## Why this ablation is useful

The same 14 values performed poorly with tree FQI. The 300-round mixed FQI
pilot reached 37.79 Coin Heaven coins and 8.62 loot-crate coins at round 300,
versus 39.14 and 10.41 for the 32-feature control. The 600-round follow-up
peaked around round 400 and ended at 28.07 and 3.06. All 576 FQI evaluation
games survived with zero invalid actions, but repeated-state counts remained
high. This indicates that the shallow tree representation did not exploit the
spatial block effectively; the neural ablation tests whether shared MLP
representations handle it better.

## Reproducibility

The combat training runner now seeds Python, NumPy, and PyTorch from the
experiment seed before model construction. This correction is required because
the earlier vanilla-DQN pilots had unseeded PyTorch initialization and showed
large seed-to-seed variation.

The topology DQN pilot is registered separately from the 32-feature control;
its result must be reported on the same held-out boards and seats before any
submission-model decision.
