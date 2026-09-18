# Combat FQI history/anti-stagnation agent

Sahand was here.

## Purpose

`combat_fqi_history_antistag_agent` is a controlled response to the first
combat pilot.  That pilot was safe and learned to destroy crates, but its
round-300 frozen policy collected only 6.54 mean coins, below the untrained
policy's 11.56.  In the separate Coin Heaven experiments, an eight-position
history plus a steps-since-progress feature improved completion and generalized
better than a sixteen-position history.

The new agent asks one narrow question: can those successful navigation signals
help a bomb-aware tree policy retain coin collection while learning crates?  It
does not change the reward constants, tree settings, action mask, or safety
search, so a difference from `combat_fqi_agent` is attributable primarily to
the representation.

## Implementation

The agent lives in `src/agent_code/combat_fqi_history_antistag_agent/`.

- It keeps the original 30 combat features and appends the previous action,
  the capped number of visits to the current tile in the last eight positions,
  and a bucketed steps-since-progress value.
- Progress resets on a visible-coin-set change, crate destruction, opponent
  removal, or score change.  Movement history resets at the same point.
- Training uses cached features from the actual decision.  Bellman targets
  reconstruct the next observation's history, avoiding a mismatch between
  action-time and training-time features.
- The original deterministic bomb-safety module is imported rather than
  duplicated.  This keeps the tested safety behavior identical across the
  controlled comparison.
- All original combat reward values are unchanged.  No revisit penalty or
  hand-written loop-breaking action is used.

`src/run_combat_training.py` now accepts `--agent`, supports `coin-heaven` for
later regression checks, and has `--parallel-seeds`.  Parallel runs preserve
independent agent seeds and non-overlapping training-board ranges; thread counts
are limited externally so tree fitting does not oversubscribe the server.
Source provenance includes the shared combat feature and safety dependencies.

## Software validation

All 13 tests pass: 2 framework integration checks, 8 existing bomb-safety tests,
and 3 new history/progress tests.  The new tests check that the base combat
representation is unchanged, crate destruction resets both history and the
stagnation clock, and waiting without progress moves through the intended
stagnation buckets.

A two-round, 20-step `loot-crate` smoke run completed model fitting, checkpoint
serialization, fresh-process loading, and frozen evaluation.  Its scores are
not treated as performance evidence.  The retained smoke artifacts are under
`experiments/combat_fqi_history_antistag_smoke/`.

## Registered crate pilot

The substantial pilot uses the same protocol as the original combat agent:

- `loot-crate`, no opponents;
- seeds 0, 1, and 2, with distinct training boards starting at 4000, 4300, and
  4600;
- 300 rounds of at most 400 steps per seed;
- frozen evaluations at rounds 0, 100, 200, and 300;
- boards 30000--30007, all four seats, for 32 games per seed/checkpoint;
- mean coins as the primary outcome, with crates and bombs as diagnostics;
- zero invalid actions and at most 10% self-deaths as safety gates.

The run completed as a detached, three-seed CPU job on 18 September 2026.
Artifacts are written to
`experiments/combat_fqi_history_antistag_crate_pilot/`, and the combined log is
`experiments/combat_fqi_history_antistag_crate_pilot.log`.

## Results

On the registered development boards, the aggregate mean coins by checkpoint
were 11.56 at round 0, 8.47 at round 100, 16.89 at round 200, and **19.35 at
round 300**.  The selected round-300 result is +7.79 coins over the untrained
policy and +12.81 over the original combat agent's round-300 mean of 6.54.
Across all 96 selected-checkpoint games, survival was 100%, with no suicides or
invalid actions.

| Seed | Round-300 coins | Crates | Bombs |
|---:|---:|---:|---:|
| 0 | 17.16 | 46.56 | 19.44 |
| 1 | 24.25 | 66.13 | 26.91 |
| 2 | 16.66 | 50.75 | 20.22 |
| **Mean** | **19.35** | **54.48** | **22.19** |

A matched confirmation on fresh boards 31000--31015 used all four seats.  The
new agent reached **16.68 mean coins** over 192 games versus **7.35** for the
original combat agent's round-300 checkpoints on the same games.  The +9.33
matched improvement confirms that the history/progress representation helps
crate learning.  Both methods had 100% survival, zero suicides, and zero invalid
actions.  Results are under
`experiments/combat_fqi_history_antistag_fresh_crate/` and
`experiments/combat_fqi_original_fresh_crate/`.

The Coin Heaven regression on fresh boards 32000--32015 did **not** pass:

| Agent | Mean coins | Completed | Invalid | Survival |
|---|---:|---:|---:|---:|
| Combat history/anti-stagnation | 19.70 | 2/192 (1.0%) | 0 | 100% |
| Stage-1 stagnation/history-8 control | 43.77 | 93/192 (48.4%) | 0 | 100% |

Combat results varied strongly by seed: 32.66, 20.58, and 5.86 mean coins.
Thus the new representation fixes much of the crate-pilot failure but does not
retain the efficient navigation policy.  Raw results are in
`experiments/combat_fqi_history_antistag_coin_regression/` and
`experiments/tree_fqi_history_stagnation_coin_control/`.

The crate learning curves and rolling training diagnostics are visualized in
`experiments/combat_fqi_history_antistag_crate_pilot/plots/training_overview.png`.

## Pre-registered next gates

The sequence is evidence-gated rather than automatic:

1. The round-300 crate policy improved on the original and passed fresh-board
   and safety gates.
2. The selected checkpoint failed the Coin Heaven regression check; it is safe
   but has lost too much navigation performance.
3. Only after both crate and navigation checks pass should training against
   `peaceful_agent`, then `coin_collector_agent`, then `rule_based_agent`.
   Each stage retains frozen evaluation and self-death/invalid-action gates.

Opponent training is therefore not justified yet.  The next controlled change
should prevent catastrophic forgetting: initialize or distill navigation from
the selected Stage-1 policy, then train with a mixed Coin Heaven/loot-crate
curriculum and checkpoint selection that requires both metrics.  Reward-balance
changes should be tested separately, not combined with that curriculum change.
