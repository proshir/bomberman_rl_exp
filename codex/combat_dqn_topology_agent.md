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

## Corrected-seed 600-round pilot

The three-seed run used 600 rounds of 400 steps and the same held-out
evaluation protocol as the vanilla control. The best aggregate checkpoint was
round 100 at 16.53 mean coins (seed means 9.69, 34.03, and 5.88). Performance
then declined to 2.86 mean coins at round 600. The final 96-game evaluation
had 100% survival, zero invalid actions, and 14.5 mean crates. Seed variance
was large; seed 1 also ended with a much higher training loss (1.96 versus
0.08 and 0.18).

This is below the 32-feature vanilla-DQN 300-round result (18.48 mean coins)
and below matched history-FQI (19.35). The topology block therefore did not
improve the neural baseline and is rejected as the default representation.
The 32-feature agent remains the control for the next algorithmic progression.
