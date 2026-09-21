# Agent 042 — combat-progress DDQN and persistent anti-loop history

Status: implemented, contract-tested, and evaluated on 21 September 2026; the
300-round solo gate completed and shows partial anti-stagnation improvement,
but not a full solution.

## Why this successor exists

The Agent 040 audit (`Agent_040_Optimized_Compact_DDQN_Design.md`) showed two
related failure modes. In the exact episode-600 bad game, the agent spent 186
of 400 decisions on `WAIT`, remained in a two-cell oscillation after the arena
was empty, and selected `WAIT` over a legal step that reduced the opponent
BFS distance. It did place 21 bombs, including five at an opponent's current
tile, so the issue was not blanket bomb refusal: the observation did not make
the approach/attack value of individual actions sufficiently distinguishable.

The same audit found that Agent 040's progress signature included global crate
count and the number of opponents. Unrelated opponent actions therefore erased
the rolling action/position context. Its `combat_env_steps` counter also
advanced on solo transitions, so the nominal combat exploration schedule was
already near its floor when combat training began. Agent 041 added coin-route
inputs, but retained both the global reset signature and the single exploration
counter; its early matched run still showed long repeated-state stretches.

Agent 042 is a controlled successor: it retains Agent 041's 122-input
representation and Agent 040's reward, safety candidate mask, replay, DDQN
target masking, and D4 augmentation, then adds only the changes below. There
is no new reward term, hard-coded attack action, or action filter.

## Implemented fixes

### 1. Action-aligned combat progress (140 inputs)

The first 104 inputs remain Agent 040. Agent 041's 18 action-aligned coin
navigation inputs remain at 104–121. Agent 042 appends 18 combat inputs:

| Block | Width | Meaning |
|---|---:|---|
| `combat_progress` | 6 | clipped signed change in nearest-opponent BFS distance for each action |
| `combat_reachable` | 6 | whether the legal movement destination remains connected to an opponent |
| `combat_pressure` | 6 | inverse destination distance; for `BOMB`, normalized current opponent hits |

The block is zero when no opponent is present. It is transformed with the
same D4 action permutation as the action label and safety mask. Thus the DDQN
can learn “close, wait, or bomb now” without changing the policy's candidate
set. Agent 040/041 first-layer checkpoints can be warm-started with zero-padded
columns; old Q-values are unchanged when the new columns are zero.

### 2. Agent-local progress and persistent loop history

`progress_signature` now contains only the visible coin set and this agent's
own score. Global crate-count and opponent-count changes no longer clear the
position/action deques. On a progress change, Agent 042 resets only the
`steps_since_progress` clock while retaining the eight-step rolling context.
This makes repeated-state evidence survive unrelated opponents' actions. A
coin collected by another player remains an unavoidable partial-observation
ambiguity; it is explicitly left for re-analysis below.

### 3. Separate solo and combat exploration accounting

Solo transitions use the broad `1.0 → 0.05` linear schedule over 100,000 solo
environment steps. An episode is marked combat when an opponent is observed;
its dedicated `combat_env_steps` counter then uses `0.30 → 0.05` over 100,000
combat transitions. The counter is no longer consumed by solo foundation
training. Exploration mode is selected at action time, and the combat flag is
reset at round boundaries.

### 4. Provenance, compatibility, and tests

`run_combat_training.py` includes Agent 042 in the successor source-hash
groups. `test_agent042_combat_progress_ddqn.py` covers the 140-input contract,
solo-neutral combat columns, action-aligned combat progress, D4 action-label
transforms, replay width, DDQN setup, Agent 040/041 checkpoint migration, and
the local progress signature. The test suite passed (7 tests); Agent 041's
existing six-test contract suite also passed. CUDA was hidden and BLAS thread
counts were capped for these CPU checks.

## Ideas deliberately left for re-analysis on Agent 042

The 300-round solo run is an ablation of the observation/history/exploration
fixes, not proof of a combat solution. After its result is available, inspect
the saved traces and compare against Agent 040/041 under the same boards and
seats. The following ideas are intentionally not enabled yet:

1. **Learned novelty or a bounded loop intervention.** Test a small revisit
   penalty or a forced unexplored-candidate fallback only after measuring
   whether persistent history changes the loop rate; do not conflate the
   intervention with the representation result.
2. **Progress attribution.** Distinguish own coin collection from a coin
   disappearing because another agent collected it, and decide whether crate
   destruction or kills should advance the progress clock.
3. **Target persistence and opponent threat prediction.** The current combat
   block recomputes nearest-opponent BFS distances each state. Evaluate a short
   target lock, enemy bomb reachability, and bomb-timer-aware approach only if
   action-aligned distance still fails in four-player traces.
4. **Reward/credit assignment.** Re-test potential-based distance shaping,
   n-step returns, and a calibrated combat-hit/kill signal separately from the
   current shaped reward. No reward change is part of this Agent 042 ablation.
5. **Bomb decision gate.** Analyze whether bomb placement needs an explicit
   escape/target-value feature beyond Agent 040's safety and bomb-value block;
   do not hard-code “attack” before measuring it.
6. **Population and curriculum effects.** Solo stagnation can be fixed while
   four-player training still forgets navigation. Only after the solo gate,
   run the documented 60/20/20 or Final_finetune schedule with held-out
   learned opponents and compare survival, score, invalid actions, and loop
   diagnostics.

## Registered 300-round solo gate

The gate uses scratch Agent 042 training, three seeds, board stride 900, the
`coin-heaven` scenario only, 400-step games, and evaluations every 50 rounds
on boards 34000–34003 and seats 0–3 with diagnostics enabled. It is intended
to answer one narrow question: does the persistent-history plus broad-solo
exploration change the late-training stagnation pattern? It is not a combat
promotion run. Its command, artifact path, and measured comparison are
recorded below.

## Measured 300-round result (21 September 2026)

The run completed successfully as job `agent042-solo-300-20260921`. Artifacts
are in `/export/scratch/salitanl/agent042_solo_300_20260921`; each of three
seeds has 300 training rounds and a 16-game fixed evaluation at rounds 0, 50,
100, 150, 200, 250, and 300 (48 games per checkpoint pooled). The run used no
opponents, 400-step games, boards 34000–34003, all seats, and diagnostics. It
completed in about 8 minutes 53 seconds on the CPU worker setup.

| checkpoint | mean coins | completion | repeated states | max no-progress steps |
|---:|---:|---:|---:|---:|
| 0 | 3.10 | 0.0% | 386.0 | 377.1 |
| 50 | 41.23 | 8.3% | 280.9 | 284.0 |
| 100 | 48.46 | 54.2% | 129.6 | 136.9 |
| 150 | 49.81 | 93.8% | 17.8 | 27.3 |
| 200 | 49.77 | 93.8% | 18.3 | 27.4 |
| 250 | 49.77 | 85.4% | 42.3 | 50.9 |
| 300 | 49.38 | 60.4% | 109.8 | 116.6 |

All 48 final games survived and had zero invalid actions. The final Agent 042
checkpoint is better than the matched Agent 041 round-300 evaluation (48.73
coins, 45.8% completion, 153.1 repeated states), but it does not beat Agent
040's round-300 mean (49.29 coins, 64.6% completion, 101.9 repeated states).
Agent 040's later round-600/900 evaluations degraded to 166.3/193.2 repeated
states, so Agent 042's 300-round result is encouraging but not a long-horizon
promotion result.

**Conclusion: stagnation is not solved.** The fixes clearly delay and reduce
the loop during rounds 100–200 (about 18 repeated states and 27 no-progress
steps), but the final round-300 checkpoint regresses to 109.8 repeated states
and a 116.6-step no-progress stretch. Persistent history and broader solo
exploration are useful partial fixes, not a complete solution. The next
analysis should focus on the explicitly deferred novelty intervention,
progress attribution, and value/reward credit-assignment ideas above, with a
matched Agent 040 control and a longer Agent 042 continuation before any
combat curriculum is promoted.

Agent 043 implements the three immediate follow-ups from this conclusion in a
separate 146-input successor: bounded novelty intervention, event-attributed
progress timing, and three-step/potential-based credit assignment. Its design
and validation record is [`Agent_043_Novelty_Credit_DDQN_Design.md`](Agent_043_Novelty_Credit_DDQN_Design.md).
