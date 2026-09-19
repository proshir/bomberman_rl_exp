# Route-aware Double-DQN combat successor

Sahand was here.

## Status

**Implemented, software-validated, and trained for 600 rounds over three seeds
on 19 September 2026. The feature set is promising, but promotion and causal
feature claims remain pending the fixed evaluation suite and a matched
46-feature mixed-curriculum control.**

`Agent_022_combat_ddqn_route_agent` is the controlled successor to
`combat_dqn_r_topology_agent`. It preserves the repaired replay semantics,
reward, action mask, safety search, network architecture, optimizer, and
Double-DQN target. It changes only the state representation.

## Representation

The prior agent used 46 inputs: 32 combat/history values plus 14 local-topology
values. This successor removes five exact topology duplicates:

- the centre of the 3x3 patch, which is always the controlled agent's free tile;
- free-neighbour count and adjacent-crate count, both determined by the patch;
- dead-end and straight-corridor flags, both determined by cardinal patch cells.

It also replaces the coin and crate direction/distance triplets. For each of
UP, RIGHT, DOWN, and LEFT it provides two values for coins and two for
crate-approach tiles:

1. exact shortest-path distance after that movement, normalized only by board
   area; and
2. a reachability flag, so an unreachable target is distinct from a zero-step
   route.

The remaining-time scalar is `(400 - step) / 400`, clipped at zero. The final
vector has 52 inputs:

```text
14 retained base combat values
+ 8 coin route values
+ 8 crate-approach route values
+ 9 retained base combat values
+ 3 history/anti-stagnation values
+ 9 pruned topology values
+ 1 remaining-time value
= 52
```

The route maps are multi-source BFS maps over currently free field tiles. They
are calculated once per target type and then looked up for each candidate
movement, so this adds route information without eight independent searches.
Dynamic bombs and opponents remain handled by the shared action safety mask;
the route maps deliberately describe static board connectivity.

## Required aliasing regression

The DQN audit provides two normal-board states with an agent at `(7, 7)`, a
coin at `(9, 7)`, a blocking crate at `(8, 7)`, and a second crate at either
`(7, 5)` or `(7, 9)`. The complete prior 46-feature vectors are identical,
although the route-improving move is DOWN in the first board and UP in the
second.

`test_combat_route_features.py` verifies that the old vectors remain identical
for this witness while the successor's per-action route costs differ and rank
UP/DOWN in the required opposite order. It also verifies the 52-input network
probe and fixed `ddqn` target selection.

The full relevant automated suite passed: **35 tests**. This includes the
route tests, DQN-target and replay tests, topology/history/safety regression
tests, and the framework's one-round game and benchmark tests. A three-round,
20-step Coin Heaven smoke run also completed with checkpoints at rounds 0 and
3 and a resolved configuration containing all inherited hyperparameters. Its
60 transitions are intentionally below the 5,000-transition warm-up, so it
validates callback loading, feature dimensions, checkpointing, evaluation, and
provenance only—not learning quality. The ignored run artifact is
`experiments/combat_ddqn_route_smoke_20260919/`.

## Work completed through 19 September 2026

Sahand was here.

The work so far forms one traceable progression:

1. The repaired 46-feature topology DDQN was shown to specialize strongly on
   Loot Crate while remaining unreliable in Coin Heaven.
2. A normal-board counterexample proved that its complete feature vector could
   merge two states requiring opposite route-improving actions.
3. The 46 inputs were audited for redundancy. Five exact topology derivatives
   were removed, and action-conditioned exact route costs, reachability, and
   remaining time were selected as the first controlled successor.
4. `Agent_022_combat_ddqn_route_agent` was implemented as a separate agent. It
   reuses the repaired masked-DDQN replay, target, reward, safety, optimizer,
   and 128-128 MLP while changing the representation to 52 inputs.
5. The aliasing regression and the complete relevant software suite passed
   (35 tests), followed by a short end-to-end checkpoint smoke run.
6. A fresh three-seed, 600-round mixed Coin Heaven/Loot Crate run completed
   successfully. No checkpoint was resumed and no opponent was present.

The completed training artifact is
`experiments/combat_ddqn_route_mixed_600round_20260919/`. It is 27 MB and
contains the resolved configuration, source hashes, per-round records,
checkpoints every 50 rounds, and both-scenario evaluations. The launcher
finished with exit code 0 in 15 minutes 22 seconds of wall time.

## Completed 600-round protocol

The run used training seeds 0, 1, and 2, alternating Loot Crate on even rounds
and Coin Heaven on odd rounds. Every game used the official 400-step horizon.
Frozen evaluation ran every 50 rounds on board seeds 33000--33002 from seat 0,
giving three games per trained seed and scenario (nine games per aggregate
checkpoint). The three training seeds ran in parallel on CPU with one numeric
thread each and CUDA hidden.

Important resolved DQN settings were DDQN, gamma 0.99, learning rate `1e-4`,
batch size 128, replay capacity 100,000, warm-up 5,000 transitions, training
every four interactions, target synchronization every 1,000 optimizer steps,
and epsilon 1.0 to 0.05 over 100,000 interactions. At round 600 the seeds had
approximately 229k--233k interactions and 56k--57k optimizer updates.

The actual command was:

```bash
export CUDA_VISIBLE_DEVICES=""
export SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python src/run_combat_training.py \
  --agent Agent_022_combat_ddqn_route_agent \
  --curriculum mixed --rounds 600 --max-steps 400 --eval-every 50 \
  --seeds 0 1 2 --eval-seeds 33000 33001 33002 --eval-seats 0 \
  --diagnostics --parallel-seeds \
  --output experiments/combat_ddqn_route_mixed_600round_20260919
```

## Measured learning trajectory

The table aggregates the three independently trained seeds. Each cell contains
nine frozen games. `SD` is the sample standard deviation of the three per-seed
means, not a game-level confidence interval. Completion applies only to Coin
Heaven.

| Round | Coin Heaven coins (mean ± seed SD) | Completion | CH max no-progress | Loot Crate coins (mean ± seed SD) | Crates | LC max no-progress |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.89 ± 1.54 | 0.0% | 395.4 | 0.00 ± 0.00 | 0.00 | 399.0 |
| 100 | 27.78 ± 8.76 | 0.0% | 169.6 | 2.78 ± 3.02 | 16.22 | 317.0 |
| 200 | 31.44 ± 3.86 | 0.0% | 205.8 | 9.33 ± 1.53 | 44.11 | 155.6 |
| 300 | 23.11 ± 4.22 | 0.0% | 231.3 | 22.00 ± 6.69 | 82.78 | 74.2 |
| 400 | 33.78 ± 9.05 | 0.0% | 246.1 | 21.67 ± 2.91 | 76.33 | 120.6 |
| 450 | 40.22 ± 6.00 | 0.0% | 177.8 | 29.67 ± 9.49 | 94.11 | 80.2 |
| 500 | 44.22 ± 0.77 | 22.2% | 196.6 | 39.78 ± 5.19 | 108.00 | 46.0 |
| 550 | **49.67 ± 0.33** | **77.8%** | **72.8** | 30.11 ± 9.38 | 81.11 | 104.9 |
| 600 | 46.89 ± 2.69 | 55.6% | 121.6 | 36.78 ± 13.95 | 99.11 | 41.4 |

Survival was 100% throughout the reported final evaluations, and the final
training episodes had zero invalid actions and zero suicides. The round-600
seed details show why the aggregate alone is insufficient:

| Training seed | Coin Heaven coins | Completion | Repeated states | CH max no-progress | Loot Crate coins | Crates | Survival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 45.33 | 0.0% | 265.0 | 245.0 | 20.67 | 75.33 | 100% |
| 1 | **50.00** | **100%** | **2.0** | **9.0** | **45.00** | 111.33 | 100% |
| 2 | 45.33 | 66.7% | 105.67 | 110.67 | 44.67 | 110.67 | 100% |

## Feature analysis and current judgment

The new representation is **promising** for the following evidence-backed
reasons:

- The route-cost block passes the constructed information test: it gives
  different values and opposite UP/DOWN rankings for states that the complete
  old 46-vector represented identically. This establishes capacity, regardless
  of training noise.
- Coin Heaven rises from 0.89 coins before training to 49.67 at round 550,
  with 77.8% completion. This is the first current combat neural configuration
  to combine near-complete coin navigation with substantial crate ability.
- Loot Crate reaches 39.78 coins at round 500 and 36.78 at round 600 while
  preserving 100% solo survival. Seed 1 at round 600 achieves 50 Coin Heaven
  coins and 45 Loot Crate coins, showing that one shared 52-input policy can
  represent both behaviors.
- Removing the five exact topology duplicates did not prevent strong crate
  performance. The compact route summaries are therefore a better candidate
  than adding more derived local flags.

Feature-group judgment:

| Feature change | Current evidence | Promise |
| --- | --- | --- |
| Per-action coin route cost + reachability | Directly resolves the alias witness; Coin Heaven reaches 49.67 coins and 77.8% completion at round 550 | **High**, pending matched control |
| Per-action crate-approach cost + reachability | The shared policy reaches 39.78 aggregate Loot Crate coins at round 500 and individual seeds reach about 45 at round 600 | **Moderate to high**; below the old crate-only specialist and still confounded by curriculum |
| Remaining-time scalar | Removes a real finite-horizon alias between early and late states | **Theoretically justified but empirically unisolated** |
| Removal of five derived topology values | Exact redundancy is proven from the retained local patch; strong behavior survives their removal | **Low-risk simplification**, not an independent performance gain |
| Retained compressed history/stagnation values | Some seeds become nearly loop-free, but seed 0 still has a 245-step no-progress tail | **Useful but insufficient** |

The evidence does **not yet prove that the feature change alone caused the
gain**. The earlier repaired 46-feature checkpoints were trained only on Loot
Crate, whereas this run alternated both scenarios. Existing mixed controls use
tree FQI rather than the same DDQN. Representation and training distribution
therefore changed together. The current result validates the complete
route-aware mixed configuration, while a matched 46-feature DDQN mixed run is
needed for a causal ablation.

The features also remain incomplete:

- The route maps represent static field connectivity. Bomb timing, explosions,
  opponents, and contested escape tiles remain outside the route costs and are
  handled only by the separate safety mask.
- Multi-source distance gives the nearest reachable target after each action,
  but does not identify target diversity, downstream coverage, or whether a
  first action leads to many remaining coins.
- History still consists of compressed visit/stagnation summaries rather than
  an explicit short movement sequence. Seed 0's 265 repeated states and
  245-step no-progress tail show that route information reduces but does not
  eliminate loops.
- Remaining time, route costs, reachability, and topology pruning were added as
  one representation package. Their individual contributions are not isolated.
- Learning is non-monotonic. Round 550 is best for aggregate Coin Heaven,
  round 500 is stronger for aggregate Loot Crate, and round 600 has large
  Loot-Crate seed variation. A last-checkpoint rule is not justified.

Relative to the old 46-feature crate specialist, the trade-off is encouraging
but unresolved. The old fixed 96-game suite reported about 47.55 Loot Crate
coins but poor and highly variable Coin Heaven behavior (roughly 21.16 coins
across its three seed means and 0% completion). The route run's small internal
evaluation is much stronger in Coin Heaven but does not yet match that
specialist's validated crate mean. These numbers use different training
curricula and evaluation sample sizes and must not be presented as a final
head-to-head comparison.

## Proposed next work (not yet executed)

These are proposed validation steps, not authorization to launch them without
the user's decision:

1. Run the fixed pre-combat suite on all three seed checkpoints at rounds 500,
   550, and 600. This gives 96 held-out games per checkpoint generation and
   scenario under board seeds 30000--30007 and all four seats.
2. Apply a predeclared selection rule: require 100% solo survival, zero invalid
   actions and self-deaths, then use Coin Heaven completion/no-progress and
   Loot Crate coins jointly. Do not select by mean coins alone.
3. Confirm the selected frozen checkpoint on the suite's fresh-board protocol
   before treating it as the Task-2 candidate.
4. Run the selected checkpoint in classic mode against peaceful, coin
   collector, and rule-based opponents. This measures the starting Stage-3
   gap; it is not expected to prove combat learning because this run had no
   opponents.
5. For a scientific feature claim, train the repaired 46-feature topology DDQN
   with the identical mixed curriculum, seeds, horizon, and hyperparameters.
   That is the necessary representation ablation.
6. Inspect failed/looping replays from the fixed suite, particularly cases with
   high coin totals but incomplete Coin Heaven boards. Decide from those
   failures whether the next single change should be multi-target route
   coverage or explicit short-cycle history.
7. Only after the Task-2 gate is stable, introduce opponent experience and the
   already ranked opponent-reachability/trapping features. Preserve the
   Coin-Heaven and Loot-Crate suites as regression gates.

No CNN, reward change, replay change, or hyperparameter sweep should be mixed
into the route-feature conclusion before these validation steps; otherwise the
source of improvement would again be ambiguous.

## Four-cell feature/curriculum comparison

Sahand was here.

The completed 2-by-2 comparison is:

| Representation | Curriculum | Coin Heaven | Completion | CH max no-progress | Loot Crate coins | Crates | Protocol |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 46 features | mixed | 48.67 | 33.3% | about 191 | 48.89 | 118.67 | three internal boards per seed at round 600 |
| 52 route features | mixed | 46.89 | 55.6% | 121.6 | 36.78 | 99.11 | three internal boards per seed at round 600 |
| 46 features | Loot Crate-only | 21.16 | 0.0% | 324.0 | 47.55 | 118.00 | fixed 96-game suite, three checkpoints |
| 52 route features | Loot Crate-only | not measured | not measured | not measured | 47.89 | 118.44 | three internal boards per seed at round 600 |

The 46-feature non-mixed values are the older fixed-suite result from
`experiments/combat_ddqn_r_topology_300round_20260918/` and its retained
pre-combat manifest. The other three rows use the newer internal evaluation
protocol, so the table is directional rather than a single statistically
matched leaderboard.

Three conclusions are justified:

1. **Curriculum is the dominant navigation factor.** For the 46-feature
   representation, adding Coin Heaven episodes changes Coin Heaven from 21.16
   coins with 0% completion and a 324-step no-progress tail to 48.67 coins
   with 33.3% completion, while Loot Crate remains about 49 coins. The mixed
   curriculum teaches navigation without sacrificing crate score.
2. **The route features are promising for navigation quality, not raw crate
   score.** Under mixed training, 52 features reduce stagnation and raise
   completion (55.6% versus 33.3%) while mean Coin Heaven coins are similar.
   They reduce Loot Crate score in this run (36.78 versus 48.89), although the
   non-mixed route run recovers to 47.89. This indicates a representation and
   objective trade-off, not a universal improvement.
3. **There is no final winner yet.** The 46-feature mixed agent is currently
   the stronger balanced-score specialist; the 52-feature mixed agent is the
   stronger completion/anti-loop candidate. The route feature claim must be
   confirmed with the fixed 96-game suite, and the 52-feature row should be
   evaluated on the same held-out Coin Heaven boards before promotion.

The practical selection rule is therefore to evaluate both mixed candidates on
the fixed suite, select using completion and no-progress behavior jointly with
Loot Crate score, and reserve Loot-Crate-only checkpoints for crate-specialist
comparisons rather than treating them as general agents.

## Loot-Crate-only follow-up completed (19 September 2026)

Sahand was here.

To separate representation from curriculum, a fresh three-seed, 600-round
run completed with `Agent_022_combat_ddqn_route_agent`,
`curriculum=none`, and `scenario=loot-crate`. Its output is
`experiments/combat_ddqn_route_lootcrate_600round_20260919/`. It uses the same
DDQN recipe, 400-step horizon, evaluation cadence, and seeds as the mixed run;
only the training curriculum differs. The final internal evaluation averaged
47.89 coins and 118.44 crates across the three seeds, with 100% survival. This
is comparable to the mixed 46-feature topology run's 48.89 Loot Crate coins,
but it says nothing about Coin Heaven because that scenario was not included.

## Original training and evaluation plan

The original proposal required a fresh, matched multi-seed run and a documented
Coin Heaven/Loot Crate curriculum. It was intended to retain the 400-step
horizon, evaluation boards, seats, safety mask, reward, and all DQN
hyperparameters and report Coin Heaven completion and no-progress tails
separately from Loot Crate coins/survival, then use the classic suite only as a
subsequent opponent gate.

From the experiment repository, the intended training command is:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 src/run_combat_training.py \
  --agent Agent_022_combat_ddqn_route_agent \
  --curriculum mixed --rounds 300 --max-steps 400 --eval-every 50 \
  --seeds 0 1 2 --diagnostics \
  --output experiments/combat_ddqn_route_mixed_300round_20260919
```

The command intentionally uses CPU because these small MLP games are
environment/BFS-bound, not GPU-bound. This original 300-round proposal was
superseded by the completed 600-round experiment above.

## Fixed mixed-curriculum suite evaluation (19 September 2026)

Sahand was here.

Both mixed 600-round candidates were evaluated with the held-out pre-combat
suite: 8 fixed boards, 4 seats, 32 games per checkpoint, and the three seed-0,
-1, and -2 checkpoints (96 games per candidate and scenario). The manifest is
`eval_suite/combat_ddqn_mixed_features_600.json`; raw results are in
`eval_suite/results/ddqn_mixed_features_600_20260919/suite_summary.json`.

| Candidate | Scenario | Mean score across seeds | Completion | Mean crates | Repeated states | Max no-progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 46-feature mixed DDQN | Coin Heaven | 49.06 | 64.6% | — | 105.7 | 110.5 |
| 46-feature mixed DDQN | Loot Crate | 46.20 | — | 114.30 | 51.2 | 34.1 |
| 52-feature route mixed DDQN | Coin Heaven | 45.08 | 53.1% | — | 140.9 | 144.6 |
| 52-feature route mixed DDQN | Loot Crate | 37.20 | — | 96.97 | 101.2 | 71.7 |

All 384 games had 100% survival and zero invalid actions. The held-out suite
does not support promoting the 52-feature mixed checkpoint: it is worse on
both raw Loot Crate collection and Coin Heaven aggregate score, and its mean
stagnation is higher. Its seed-1 and seed-2 checkpoints are competitive, but
seed 0 is a severe failure (37.34 Coin Heaven coins, 9.4% completion and
64.34 crates in Loot Crate). The 46-feature mixed DDQN is the current
pre-combat candidate; checkpoint/seed selection remains important because its
Coin Heaven completion ranges from 56.3% to 78.1%.

## Matched four-cell feature/curriculum suite (19 September 2026)

All four completed 600-round training cells were subsequently evaluated under
the *same* held-out protocol: eight fixed boards, four seats, three trained
checkpoints, and both solo scenarios (96 games per cell and scenario; 768
games total). The manifest is
[`ddqn_feature_curriculum_2x2_600.json`](../eval_suite/ddqn_feature_curriculum_2x2_600.json)
and raw data are in
`eval_suite/results/ddqn_feature_curriculum_2x2_600_20260919/`.

| Features | Curriculum | Coin Heaven coins | Completion | CH max no-progress | Loot Crate coins | Crates |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 46 topology | Loot Crate-only | 21.16 | 0.0% | 324.0 | 47.55 | 118.00 |
| 52 route-aware | Loot Crate-only | 20.85 | 15.6% | 283.6 | 44.31 | 111.58 |
| 46 topology | mixed | **49.06** | **64.6%** | **110.5** | **46.20** | **114.30** |
| 52 route-aware | mixed | 45.08 | 53.1% | 144.6 | 37.20 | 96.97 |

All 768 games had 100% survival and zero invalid actions. Thus mixing is the
dominant, robust intervention: it turns the 46-feature agent from a
Loot-Crate specialist into a strong Coin-Heaven agent while retaining nearly
all of its crate performance. The route representation does not improve the
aggregate outcome in this controlled comparison. It is slightly less stagnant
than the 46-feature specialist when trained only on Loot Crate, but its mixed
variant is worse than the 46-feature mixed control on every aggregate task
metric and is unstable: its seed 0 gets 37.34 Coin-Heaven coins/9.4%
completion and only 21.53 Loot-Crate coins. The 46-feature mixed DDQN is the
selected balanced candidate; route features should not be promoted without a
separate change that addresses that instability.

## Sequential single-group feature pilots (19 September 2026)

The three proposed feature groups were then tested one at a time on the
repaired 46-feature topology DDQN. All branches retained the mixed curriculum,
400-step horizon, safety mask, reward, replay, network, optimizer, and three
training seeds. Each ran for 300 rounds on CPU, with seeds trained sequentially
inside its detached launch. The final evaluation used eight held-out boards and
four seats per trained checkpoint: 96 games per scenario across the three
checkpoints. CUDA was hidden explicitly.

| Agent | Added group | Input size | Coin Heaven coins | Completion | CH max no-progress | Loot Crate coins | Crates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Agent_024` | Per-action bomb consequences | 106 | 42.76 | 47.9% | 171.7 | **38.90** | **96.92** |
| `Agent_025` | Compact short-cycle history | 65 | **48.51** | **56.3%** | **133.2** | 37.17 | 93.18 |
| `Agent_026` | Global target coverage | 58 | 37.76 | 30.2% | 233.0 | 37.13 | 92.67 |

The values are means across the three final seed summaries; the no-progress
value is the corresponding mean diagnostic maximum. All 576 final evaluation
games were safe: every branch had 100% survival, zero invalid actions, and zero
suicides.

`Agent_025_combat_ddqn_short_cycle_agent` is the only one-group addition worth
promoting to a longer matched run. It reached 48.51 Coin-Heaven coins with
56.3% completion and reduced the mean no-progress tail to 133.2, while its
Coin-Heaven seed means were tightly grouped. It still trails the latest fixed
46-feature mixed reference on Loot Crate (37.17 versus 46.20 coins and 93.18
versus 114.30 crates), so this is a promising ablation, not a replacement
winner. The 46-feature base exposes a previous-action/revisit/stagnation
summary; Agent 025 is the first of these branches to expose explicit recent
actions and two-/four-cycle indicators, backed by an internal eight-step
window.

`Agent_024` is the better secondary branch when crate collection is the
priority: it led the three new agents on both Loot-Crate metrics, but its
navigation score and completion were below Agent 025. `Agent_026` is rejected
for now: target coverage did not reduce loops, had the largest seed variance,
and its final Coin-Heaven score fell below both the 46-feature reference and
the other new branches. No feature groups should be combined until Agent 025
has a matched 600-round/fixed-suite confirmation.

The raw CPU artifacts are retained under
`/export/scratch/salitanl/bomberman_feature_variants_20260919/`:
`agent024_action_safety_300_cpu`, `agent025_short_cycle_300_cpu`, and
`agent026_target_coverage_300_cpu`.

## Agent 025 classic evaluation (19 September 2026)

The three final 300-round Agent-025 checkpoints were evaluated in 288 fresh
classic games: eight boards (`32000`--`32007`), four seats, 400 steps, and one
supplied opponent per lineup. The CPU-only evaluation used the same classic
protocol as the 46-feature DDQN reference.

| Opponent | Mean score | Mean coins | Mean kills | Survival | Invalid actions/game | Suicides/game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `peaceful_agent` | 1.40 | 1.03 | 0.07 | 96.9% | 0.01 | 0.03 |
| `coin_collector_agent` | 2.58 | 2.11 | 0.09 | 87.5% | 0.10 | 0.13 |
| `rule_based_agent` | 2.25 | 2.09 | 0.03 | 54.2% | 0.60 | 0.39 |

This does not promote Agent 025 to combat. Against the current 46-feature
600-round combat-safety reference, it has lower score and coin collection in
every lineup (reference: 5.99/4.04/3.70 score and 4.48/3.47/3.28 coins), but
it commits fewer invalid actions against the coin collector and rule-based
opponents (0.10 versus 0.47 and 0.60 versus 1.17 respectively). Rule-based
survival is also slightly higher (54.2% versus 49.0%). This is an encouraging
safety signal, but the comparison is not a pure feature claim because Agent
025 has only 300 training rounds and no opponent experience. The next justified
step remains a longer solo confirmation of Agent 025 before any combat-training
change.

The manifest is
[`classic_agent025_short_cycle_300_20260919.json`](../eval_suite/classic_agent025_short_cycle_300_20260919.json)
and the raw suite output is
`/export/scratch/salitanl/bomberman_feature_variants_20260919/agent025_short_cycle_classic_300_cpu/`.

## Agent 027 staged replay curriculum (19 September 2026)

`Agent_027_combat_ddqn_short_cycle_staged_replay_agent` keeps Agent 025's
65-value observation exactly, but changes training: 70% Coin Heaven / 30% Loot
Crate for rounds 1--100, 25% / 75% for 101--300, then 50% Classic plus 25% of
each retained solo scenario for 301--600. Classic rounds rotate one opponent
at a time among peaceful, coin-collector, and rule-based. Scenario-tagged
replay samples the active target proportions, so earlier data remains present
during the later phase. Three seeds ran concurrently on CPU for 600 rounds;
CUDA was hidden. The fixed held-out solo checks used eight boards, four seats,
and 400 steps per checkpoint (96 games per scenario/checkpoint).

| Checkpoint | Coin Heaven coins | Completion | CH max no-progress | Loot Crate coins | Crates | LC max no-progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 47.72 | 83.3% | 65.3 | 42.54 | 103.64 | 47.0 |
| 400 | **49.83** | **89.6%** | **45.3** | 44.98 | 110.47 | 29.3 |
| 600 | 49.26 | 60.4% | 119.5 | **47.04** | **116.36** | **17.3** |

Every one of the 576 held-out solo games had 100% survival, zero invalid
actions, and zero suicides. The final checkpoint improves the matched
46-feature mixed reference's Loot Crate score/crates (47.04/116.36 versus
46.20/114.30) and preserves its Coin Heaven score (49.26 versus 49.06), but
its Coin Heaven completion is lower (60.4% versus 64.6%) and round 600
regresses from Agent 027's round-400 navigation peak. This is a useful
training-method result, but not a clean replacement claim: Agent 027 has 150
Classic training rounds and fewer solo rounds than the all-solo mixed control.

The in-training Classic episodes are diagnostic rather than a fair classic
gate: across its 450 episodes, mean score was 6.49, coins 5.42, kills 0.22,
survival 73.3%, invalid actions 0.35/game, and suicides 0.23/game. A fresh,
fixed classic-suite evaluation of a preselected checkpoint is required before
using those figures to claim combat performance. The final run artifacts are
`/export/scratch/salitanl/bomberman_feature_variants_20260919/agent027_short_cycle_staged_replay_600_cpu/`.
