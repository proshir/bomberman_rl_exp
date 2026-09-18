# Repaired 46-feature topology DQN

Sahand was here.

`combat_dqn_r_topology_agent` is a separate 46-input agent. It keeps the
topology representation from the earlier topology ablation, but delegates all
algorithmic callbacks to the repaired `combat_dqn_agent` implementation:

- action-time safety masking in Bellman targets;
- exact action-time next-state features in replay;
- terminal empty masks and zero bootstrap values;
- the repaired replay buffer, checkpoint format, and optimizer logic.

The 46 inputs are the validated 32-feature history/anti-stagnation vector plus
the 14 local-topology values (3x3 patch, neighboring free/crate counts,
radius-two openness, dead-end flag, and straight-corridor flag).

The earlier 46-feature training launch using the old agent name was stopped
before completion. The new agent passed a 46-dimensional feature probe, the
19 relevant DQN/safety/topology tests, and a three-round end-to-end smoke run
with fresh-process checkpoint evaluation. No performance conclusion is drawn
from the smoke run; a matched multi-seed training run is still required.

Implementation is in the code repository under
`agent_code/combat_dqn_r_topology_agent/`, commit `0e7ac9e`.
