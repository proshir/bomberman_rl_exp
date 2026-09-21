# Agent 040: Optimized Compact DDQN

Date: 21 September 2026
Status: implemented and contract-tested; training not launched

## Decision

Agent 040 is a behavior-preserving implementation successor to Agent 039.
It keeps Agent 039's 104-input schema, 128--128 Double-DQN, safety masks,
population curriculum, replay weighting, and one-sampled D4 augmentation.
The change is computational: removed features are no longer computed and
repeated board searches are reused within a decision state.

## Implemented optimizations

- Directly assembles the 104 compact features instead of computing Agent 038's
  complete 117-value vector and slicing it afterward.
- Reuses immediate legality, safety candidates, danger schedules, opponent
  reachability, feasible-crate targets, route distance maps, and feature values
  through a per-state `StateContext`.
- Caches static crate-ray candidates for repeated board layouts.
- Shares the action-conditioned danger schedule across opponent-window response
  searches and caches blast rays used by those searches.
- Replaces repeated Python symmetry loops and offset lookups with precomputed
  D4 index maps while retaining the existing nearest-crate tie-break BFS.
- Replaces replay tag Python sets plus per-batch set-to-array conversion with
  dense index pools and reusable inherited batch arrays.
- Reuses cached action candidates for inference and next-state DDQN masks.
- Uses `torch.inference_mode()` for action-time inference.

No safety rule, action mask rule, reward rule, network architecture, training
schedule, or feature value was intentionally changed.

## Contract validation

The Agent 040 tests verify:

- direct 104-feature output is exactly equal to Agent 039's compact output on
  the representative Agent 032 state suite;
- all eight D4 transforms and masks match Agent 039 for the compact vector;
- circular replay replacement preserves tag membership and 104-wide batches;
- the Agent 040 package stages and imports in the normal training runtime;
- cached action candidates match Agent 039's safety candidate contract.

## Initial CPU benchmark

This is a small implementation benchmark, not a training result. It used the
three representative states from the existing optimized-feature contract,
single-threaded CPU execution, and the same fresh-state setup for both agents.

| Path | Agent 039 | Agent 040 | Relative result |
|---|---:|---:|---:|
| Fresh feature extraction, mean | 51.4 ms | 28.4 ms | 1.81x faster |
| Cached repeat of an identical state | not available | 0.0014 ms | state reuse enabled |
| Compact symmetry transform | 0.0673 ms | 0.0077 ms | 8.7x faster |
| Replay sample plus CPU batch handoff | 0.3527 ms | 0.0835 ms | 4.23x faster |

The feature benchmark is dominated by deterministic Python safety and board
searches, so the realized end-to-end training improvement must be measured in
the real engine. The 104-wide neural network and replay payload remain the same
size as Agent 039; Agent 040 is not claiming an additional input-width gain.

## Training and comparison protocol

The runner is
`eval_suite/run_agent040_optimized_training.py`. It intentionally matches the
Agent 039/038 population protocol: three seeds, the same 1,200-round default,
the same lineups, development boards, evaluation seats, checkpoint cadence,
and CPU-only environment. Agent 039 remains the control. A promotion decision
requires matched reward, survival, invalid-action, self-destruction, and
wall-clock measurements; the benchmark above alone is not sufficient.
