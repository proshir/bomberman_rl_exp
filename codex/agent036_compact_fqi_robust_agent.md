# Agent 036: robust compact FQI

## 21 September 2026: implementation and validation

`Agent_036_compact_fqi_robust_agent` is a standalone successor to Agent 035.
It keeps the compact route/history representation and symmetry augmentation,
while repairing the decision-time and safety details found in the follow-up
audit:

- the next-state cache is keyed by the next actual decision, including the
  engine's callback convention;
- attributable progress resets the no-progress timer without erasing the
  personal history used by the feature vector;
- terminal events are counted once, including accumulated final event lists;
- bomb-escape accounting uses the original search timeline and retains the
  full set of surviving first follow-ups;
- useful-bomb filtering is explicit and configurable rather than being hidden
  inside generic survivability checks;
- replay balancing preserves rare scenario/lineup groups without shrinking
  the entire fit, and scenario weights are not multiplied by lineup count;
- checkpoint schema v2 restores trees, replay, epsilon, RNG state, completed
  rounds, interactions, and timing/support diagnostics.

The implementation also adds a dense-board uncached inference benchmark and
the training runner supports Agent 036 context and diagnostics. The focused
Agent 036 contract tests and the Agent 034/035 plus combat safety/route
regressions pass: 37 tests in total. On one CPU with thread counts limited to
one, 128 uncached dense-board decisions measured 150.1 ms median, 155.6 ms
p95, and 161.3 ms maximum.

A two-round mixed real-engine smoke run completed with zero invalid actions,
and evaluation checkpoints loaded successfully. A separate process-level
resume smoke ran one episode, resumed to episode two, preserved the replay and
interaction state (8 then 16 interactions), and created the episode-2
checkpoint. These are implementation and integration checks only; no
substantial Agent 036 learning comparison has been run.

## 21 September 2026: matched held-out solo evaluation

The three episode-300 checkpoints from the mixed run were evaluated together
with Agent 035 episode-300 checkpoints under the fixed pre-combat suite:
held-out boards 30000--30007, four seats, action seed 0, no opponents, and a
400-step horizon (96 games per candidate and scenario). Results are means over
the three training seeds:

| Candidate | Coin Heaven coins | Completion | Loot-crate coins | Crates | Survival | Invalid actions |
|---|---:|---:|---:|---:|---:|---:|
| Agent 035 | 21.22 | 2.1% | 15.87 | 41.55 | 100% | 0.00 |
| Agent 036 | 21.64 | 4.2% | 12.56 | 34.43 | 100% | 0.00 |

Relative to Agent 035, Agent 036 is essentially tied/slightly higher on Coin
Heaven (+0.42 coins and +2.1 percentage points completion), but lower on
Loot-Crate reward (-3.30 coins and -7.13 crates). The training-trajectory
values therefore overstated the apparent improvement: Agent 036 does not yet
replace Agent 035 as the best balanced compact-FQI checkpoint.

Raw results are in
`/export/scratch/salitanl/agent035_vs_agent036_round300_eval_20260921/`, with
the reproducible manifest in
`eval_suite/agent035_vs_agent036_round300_20260921.json`.
