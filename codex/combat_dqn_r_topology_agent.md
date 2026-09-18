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

## Combat escape-safety repair (19 September 2026)

Classic evaluation of the 600-round Double-DQN checkpoints showed that most
deaths involved the agent's own bomb: 21 of 22 deaths against the coin
collector and 49 of 54 against the rule-based agent. Three saved failures were
replayed step by step. In each one, the static safety search approved a bomb
and an escape corridor, but the opponent later occupied the intended escape
tile. The framework executes simultaneous choices in randomized order, so the
move became invalid and the agent died in its own blast.

The shared combat safety search now computes the tiles that nearby opponents
could occupy during an escape. A bomb is unavailable if every escape route
depends on one of those tiles. In games with opponents it also requires the
agent to leave its own blast one step before the last possible movement. The
timing margin is disabled in solo games because an initial ablation reduced
Loot Crate performance from 47.76 to 30.48 coins by suppressing useful crate
bombs. Focused tests cover a contested single exit, an uncontested alternate
exit, the combat timing margin, and preservation of the original solo route.

The final implementation was evaluated without retraining on the same three
600-round checkpoints, boards, seats, and opponents as the baseline. Each row
contains 96 games:

| Scenario | Metric | Before | After |
| --- | --- | ---: | ---: |
| Solo Loot Crate | mean coins | 47.76 | 47.55 |
| Peaceful | mean score / survival | 7.15 / 92.7% | 5.99 / 94.8% |
| Coin collector | mean score / survival | 3.96 / 77.1% | 4.04 / 86.5% |
| Rule based | mean score / survival | 3.68 / 43.8% | 3.70 / 49.0% |

Self-death games fell from 21 to 10 against the coin collector and from 49 to
36 against the rule-based agent. Rule-based wins/ties/losses improved from
19/0/77 to 22/2/72. Peaceful score fell because the conservative mask placed
fewer bombs, and invalid actions against the rule-based agent increased from
92 to 112 in total. The repair therefore addresses a confirmed failure mode
but does not solve dynamic collision handling after a bomb has already been
placed or the agent's broader navigation and combat weaknesses.

Final evaluation artifacts are in
`eval_suite/results/ddqn_r_topology_600_combat_safety_precombat_20260919/`
and
`eval_suite/results/ddqn_r_topology_600_combat_safety_classic_20260919/`.

## Known remaining navigation issue

The agent can still become stuck near a corner and can settle into a repeated
left-right movement loop; the equivalent up-down loop is also possible. Coin
Heaven makes this especially visible: the weak seeds spend most of a 400-step
game without collecting another coin. The eight-position history and
stagnation bucket reduce some repetition but do not guarantee that a learned
policy will leave a corner or break a two-tile oscillation. This is a known,
unresolved issue and should be reported separately from the combat escape bug
fixed above.
