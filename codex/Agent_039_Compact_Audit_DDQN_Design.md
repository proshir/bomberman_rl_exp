# Agent 039: Compact Audit DDQN

Date: 21 September 2026  
Status: implemented and contract-tested; training not launched

## Decision

Agent 039 is the controlled compact successor to Agent 038. It keeps the
117-input feature calculations and the 128--128 Double-DQN, replay balancing,
time-aware action masks, and one-sampled D4 replay augmentation. It removes
only the semantically verified fields approved before implementation:

```text
117 Agent 038 inputs
- 1 scalar previous-action index
- 1 constant 3x3 center cell
- 6 action-conditioned armed-opponent copies
- 6 action-legality copies
+ 1 global armed-opponent flag
= 104 inputs
```

This is a representation test, not evidence that the compact model is better.
The matched three-seed training comparison remains pending.

## Compact layout

The Agent 039 extractor first computes the Agent 038 vector and then applies a
named, checked column mapping. The new global armed-opponent feature is at
index 63. The four retained opponent-response values occupy six directional
blocks of four values starting at index 64. The route suffix begins at index
88 and ends at index 103.

The authoritative current and next action masks remain outside the feature
vector and continue to control action selection and DDQN target calculation.
The response values in an illegal old action block remain zero, preserving the
existing gating behavior while removing the explicit legality input.

## Implementation checks

The contract tests cover:

- exact 104-input shape and finite values;
- source-to-compact column mapping;
- constant-center and legality-copy semantics on representative states;
- 104-column preallocated replay;
- 117-to-104 checkpoint migration;
- policy-only warm-start state reset;
- all eight D4 transforms and inverse closure;
- compact action-block and route alignment under symmetry.

## Evaluation plan

The primary comparison should be scratch-to-scratch with three seeds for the
full 117-input Agent038 control and Agent039. It should keep the Agent038
curriculum, maps, symmetry, optimizer, training length, and evaluation suite
unchanged. Evaluate at rounds 0, 150, 300, 600, 900, and 1200 using reward,
success/survival, self-destruction, and invalid-action metrics.

The checkpoint migration exists for diagnostics and optional warm-start tests;
it is not the primary causal comparison because six action-specific first-layer
columns are compressed into one global column.

## Episode-300 checkpoint result

The three-seed scratch training was intentionally stopped after episode 300.
The saved checkpoints were evaluated on the registered four development boards
and four seats. The one interrupted Deepkiller-only cell was rerun directly
from seed 2's saved checkpoint, without resuming training.

| Evaluation | Mean score | Survival | Invalid actions |
|---|---:|---:|---:|
| Rule-based x3 | 2.479 | 60.4% | 0.625 |
| Rule + collector + peaceful | 3.521 | 66.7% | 0.458 |
| Deepkiller + Arbiter + rule | 1.500 | 60.4% | 0.688 |
| Deepkiller + Arbiter + Agent037 | 1.458 | 64.6% | 1.958 |
| Deepkiller x3 | 1.167 | 85.4% | 0.396 |
| Coin Heaven | 49.646 | 100% | 0 |
| Loot Crate | 40.417 | 100% | 0 |

Against the matched Agent038 117-input scratch episode-300 checkpoints,
Agent039 is essentially tied on the two solo tasks, improves the mixed legacy
lineup, and is slightly lower on the focused learned-agent lineups. This is a
mixed early-foundation result, not evidence of a final feature decision or
tournament promotion.
