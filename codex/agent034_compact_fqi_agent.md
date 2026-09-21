# Agent 034 compact FQI implementation

Date: 21 September 2026. Status: implemented and smoke-tested; no substantial
training or performance comparison.

The source agent is `src/agent_code/Agent_034_compact_fqi_agent/`. It follows
the compact candidate in `Bomberman_Li_Deepkiller_Compact_FQI_Design.md`.
The 22 logical values become 28 numeric model inputs:

| Numeric indices | Contract |
| --- | --- |
| 0–3 | Coin route cost, UP/RIGHT/DOWN/LEFT; one step for move plus exact remaining free-tile distance |
| 4–7 | Coin route reachable flags |
| 8–11 | Feasible crate-bomb tile route costs in the same order |
| 12–15 | Feasible crate-bomb tile reachable flags |
| 16–18 | Bomb available, crates hit here, viable first escape actions after a hypothetical bomb |
| 19–25 | One-hot previous action: UP/RIGHT/DOWN/LEFT/WAIT/BOMB/START |
| 26–27 | Visits in the previous eight decision positions; progress bucket 0–4 |

The crate target feasibility check tests a bomb **immediately at the candidate
tile** against the present bomb schedule and opponent positions. It does not
simulate travel there; actual BOMB legality and survival are checked again
at decision time. Permanent walls and crates block static routes. Current
bombs and opponents block the first movement; later static route cells treat
them as transient. An unreachable route has cost and flag zero. The feature
schema is `compact-routes-history-v2` after the 21 September transition repair.

The existing combat FQI time-indexed safety module supplies the action mask
and longest-survival emergency fallback. Six independent depth-8 regression
trees use gamma 0.95 and the audited combat FQI reward constants. After every
20 completed rounds, five FQI passes compute all targets from the frozen old
six-tree set before replacing any tree. Unfitted actions predict zero, but
excluded actions never enter a backup. Replay retains up to 40,000 records
per scenario/lineup tag and samples equal counts per tag for fitting.
Exploration starts at epsilon 1.0, decays by 0.995 each round to 0.05, and
samples only from the stored allowed set. Checkpoints contain the feature
schema and trees. The in-memory replay is not serialized, so process restart
does not preserve the fitted dataset; a restarted run must be treated as a
new collection phase rather than a seamless continuation.

Each replay record holds both decision-time vectors and masks, terminal flag,
reward, event list, scenario and complete lineup tags, board seed, round and
step, and feature/safety/reward/history schema IDs. History resets on round
start or an observable change to own score or the board's crate count. This
same rule runs during training and frozen inference. Previous action stays as the last decision until
round reset. The 22-entry candidate omits remaining time and is not claimed
to be a sufficient state representation.

Validation: `python -m unittest -q test_agent_034 test_combat_route_features`
passed seven tests. The known route witness has different 28-vectors and
opposite UP/DOWN costs. A one-round, eight-step Coin Heaven runner smoke
completed with eight transitions, zero invalid actions, and frozen checkpoint
evaluation; this is a runtime check, not learning evidence. A dense synthetic
board feature call took about 0.06 seconds in a local three-call check, below
the 0.5-second action limit but not a full latency distribution.

Dependencies: the implementation imports the project's existing
`combat_fqi_agent` safety and crate-target functions. Packaging Agent 034
alone would require including those modules or inlining the audited code.

## 21 September 2026: failed pilot and transition repair

The first three-seed mixed run was stopped around rounds 249–250 after its
round-100 and round-200 frozen policies stalled (roughly one Coin Heaven coin
and zero Loot Crate coins per game). The stopped artifacts are under
`/export/scratch/salitanl/agent034_compact_fqi_mixed_600_20260921_r3/`.
They use feature schema `compact-routes-history-v1` and must not be treated as
valid FQI learning results or continued from checkpoint.

Root cause: the engine gives the pre-action and post-action states the same
`(round, step)`. Agent 034 initially cached both under that key. During replay
collection, `observe(post_action_state)` returned the already cached
pre-action vector and mask. Every nonterminal FQI backup therefore used the
old decision as its next state. A round-200 frozen replay confirmed that this
produced a two-tile Coin Heaven loop and persistent waiting in Loot Crate.
Exploratory training scores concealed the greedy-policy failure.

The repaired implementation maps each post-action state to the next decision's
step (`step + 1`) before caching it and uses that cache entry for the replay
record. The following real `act` call uses the same entry. Training and frozen
inference now also reset history from the same observable score/crate progress
signature; the old version reset only in the training event callback. Feature
and history schema IDs were incremented so old checkpoints fail to load under
the repaired code. The focused tests now include a same-step collision witness
and a progress-history parity check; nine relevant tests passed. No new
training or performance comparison has been run on the repaired version.

## 21 September 2026: post-pilot audit and Agent 035 successor

A later requirement-by-requirement audit found that Agent 034 still duplicated
final-step events when the engine supplied the accumulated terminal list, only
considered crate-bomb positions immediately adjacent to crates, had no symmetry
implementation, ignored runner replay weights, emitted no action-support or
timing diagnostics, and could not restore replay, epsilon, RNG, or completed
rounds through the advertised resume workflow.

These contracts change checkpoint meaning and training behavior. Agent 034 is
therefore frozen as the historical schema-v2 baseline rather than silently
rewritten underneath its saved trees. The fixes are implemented and separately
versioned in `Agent_035_compact_fqi_symmetry_agent`; see
[`agent035_compact_fqi_symmetry_agent.md`](agent035_compact_fqi_symmetry_agent.md).
Existing Agent 034 checkpoints remain evaluation artifacts only and are not
valid initialization checkpoints for Agent 035.
