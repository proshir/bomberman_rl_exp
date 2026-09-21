# Agent 038: Symmetric Population DDQN for Four-Player Tournament Play

Date: 21 September 2026  
Status: wide 117-input teacher implemented; not trained  
Proposed package: `Agent_038_symmetric_population_ddqn_agent`

## 0. Implementation amendment: wide teacher first

The implemented Agent 038 follows the later audit decision to retain every
Agent 037 feature before pruning. Its input is Agent 037's exact 101-value
vector followed by all 16 Agent 036 features that Agent 037 lacked:

```text
101 Agent 037 values
+ 4 coin-route costs
+ 4 coin-route reachability flags
+ 4 feasible-crate-route costs
+ 4 feasible-crate-route reachability flags
= 117 inputs
```

This supersedes the 84-input primary implementation proposed below. The
84-input contract remains the planned compact student/ablation candidate after
the wide model produces measured feature-importance evidence. Agent 038 can
initialize directly from an Agent 037 checkpoint: the 16 new first-layer
columns and matching Adam moments are zero-padded, exactly preserving the
teacher Q-values before training.

The implementation also samples one of the eight D4 symmetries for every
stored replay transition. It transforms the state features, selected action,
and next-action mask together, so replay capacity and optimizer frequency do
not increase eightfold. A fixed-order nearest-crate BFS has a directional
tie-break; Agent 038 recomputes that inexpensive three-value field under the
chosen orientation instead of assuming it is perfectly equivariant.

## 1. Decision

Agent 038 should be a self-contained, 84-input Double DQN initialized by
distilling the strongest Agent 037 checkpoints, then fine-tuned in a frozen
opponent population containing strong imported agents. Training should apply
one uniformly sampled, verified D4 board symmetry to each replay transition.

This combines the three useful proposals without treating any of them as a
guaranteed improvement:

1. replace the rule-based-only competition curriculum with a diverse frozen
   population that includes the strongest imported scorer and strongest
   combat-focused imported agent;
2. remove feature values that are exact derivatives, repeated copies, or
   irrelevant once the existing legal/safety mask is enforced; and
3. use the eight unique square symmetries to share directional experience and
   reduce seat/orientation bias.

The primary hypothesis is that Agent 037's main limitation is its competition
distribution, not insufficient network width. The compact input and symmetry
are supporting interventions: they should reduce nuisance degrees of freedom
and improve sample efficiency while the population curriculum supplies the
missing opponent behaviour.

Agent 038 should keep the 128--128 DDQN, audited time-aware safety mask, official
action order, and corrected decision-time replay semantics. It should not copy
Li Deepkiller's tabular learner, stochastic `target` label, `KILL!` label, or
large detector reward.

## 2. Evidence motivating the change

### 2.1 Agent 037 is competitive with supplied agents but does not generalize

The selected Agent 037 episode-1050 seed-0 checkpoint achieved, against three
rule-based opponents, mean score 3.5625, 78.1% survival, 37.5% outright first,
46.9% joint first, and mean rank 2.15625. It therefore learned useful Classic
play.

Against the strongest imported general scorer, `imp_li_deep_killer`, the gap
was clear:

| Lineup | Agent 037 score | Survival | Kills | Mean rank | Direct result |
| --- | ---: | ---: | ---: | ---: | --- |
| rule-based + Deepkiller + peaceful | 4.156 | 59.4% | 0.344 | 2.25 | 10 wins / 2 ties / 20 losses versus Deepkiller |
| three Deepkiller copies | 1.438 | 75.0% | 0.000 | 2.719 | 0 outright firsts in 32 games |

The second row is particularly diagnostic: survival remained respectable but
score and kills collapsed. This is consistent with a safe policy that does not
contest resources or create attacks effectively against stronger policies.

Agent 037 did train against rule-based agents. At episode 1200, 48% of its
effective replay sampling came from two Classic tags that both contained
`rule_based_agent`. It did not train against `imp_li_deep_killer`,
`imp_alii_arbiter`, or another strong learned policy. The issue is opponent
coverage, not absence of all competition training.

### 2.2 Imported agents define useful but distinct training pressures

The matched imported-agent audit identified:

- `imp_li_deep_killer` as the strongest general scorer: 5.22 score, 0.31 kills,
  and 81% survival against three rule-based opponents;
- `imp_alii_arbiter` as the strongest combat-focused candidate: 5.47 score and
  0.56 kills per game against three rule-based opponents; and
- `imp_li_sarsa_lambda` and `imp_li_double_q` as strong policies with different
  tabular update rules and behaviour.

Training against only one imported policy would invite exploitation of that
policy. Agent 038 therefore uses a small frozen population and reserves several
valid imported policies for unseen-opponent evaluation.

### 2.3 Bigger feature sets have not been reliable improvements

- The 113-input Agent 031 added 12 offensive values. It improved the
  single-rule-based score and survival but regressed the three-opponent and
  mixed-strong results. Those 12 values remain excluded.
- The 52-input route-aware branch underperformed the matched 46-input DDQN on
  both solo tasks. More planning summaries were not automatically useful.
- The 14-value topology block failed to improve the earlier 32-input DQN, and
  five values in that block were already proven exact derivatives.
- Agent 025's short-cycle history was the only one-group feature addition with
  enough evidence to retain, although it did not improve every task.

The Agent 037 first-layer weights at episode 1050 are nonzero across every
feature group. Weight magnitude is not causal feature importance, especially
with correlated inputs, so it does not justify deleting whole groups. The
safe first reduction is limited to exact or mask-conditional redundancy.

### 2.4 Symmetry is structurally valid but not yet proven for DDQN

Bomberman board geometry, bomb physics, and official score are equivariant
under the eight rotations/reflections of the square. Agent 035 already has a
tested implementation of all eight unique D4 action and mask permutations.
Its weak FQI result does not show that symmetry failed: learner, representation,
and training distribution differed substantially from Agent 037.

Symmetry is therefore useful as a controlled DDQN data augmentation. It must
transform the complete transition consistently. Li's reported loop of four
rotations and three flip settings should not be copied because it can contain
duplicate group elements and silently reweight samples.

## 3. The 84-input feature contract

Agent 037's 101 values are:

```text
29 base combat values
+ 3 base history values
+ 14 topology values
+ 19 short-cycle values
+ 36 action-conditioned opponent-window values
= 101
```

Agent 038 should use:

```text
29 base combat values
+ 2 base history values
+ 9 pruned topology values
+ 19 corrected short-cycle values
+ 1 global armed-opponent value
+ 24 action-conditioned opponent-response values
= 84
```

The action order remains `UP, RIGHT, DOWN, LEFT, WAIT, BOMB`.

### 3.1 Unconditional pruning

Remove the five topology derivatives already identified by the prior feature
audit:

- the centre of the 3x3 patch, which is always the controlled free tile;
- free-neighbour count;
- adjacent-crate count;
- dead-end flag; and
- straight-corridor flag.

The latter four are deterministic functions of the retained cardinal patch
cells. Keep the eight non-centre patch cells and radius-two free-space count.

### 3.2 Mask-conditional opponent-window pruning

Each of Agent 037's six action blocks currently stores:

```text
legal, any armed opponent, destination contested,
worst survival time, worst frontier size, new trap
```

Replace the 36 values with one global `any_armed_opponent` value and four values
per action:

```text
destination contested, worst survival time, worst frontier size, new trap
```

This removes eleven inputs:

- six legality values are redundant with the authoritative action mask. An
  illegal/unsafe action is neither executed nor admitted to the DDQN bootstrap;
- armed-opponent presence is one state-wide fact repeated in all six blocks.

For illegal actions, the four retained response values remain zero. The policy
must never infer legality from feature values; the stored current and next masks
remain authoritative.

This pruning reduces network input but does not remove the expensive opponent
window searches. Its expected inference speedup is therefore modest. The main
benefit is fewer correlated parameters and a cleaner symmetry contract.

### 3.3 Previous-action overlap

Remove the integer previous-action value from the three-value base-history
suffix. The 19-value short-cycle block already represents the recent actions.
Correct that block to right-align its two six-way action one-hots, so the final
slot always means the immediately preceding action; all-zero means no previous
action. Retain recent visits and stagnation bucket from the base history.

This is a contract correction as well as a one-input reduction. Tests must
cover round start, one prior action, and two or more prior actions.

### 3.4 Features deliberately retained

Retain coin, crate, opponent, bomb-value, danger, count, short-cycle, contested
destination, survival-time, frontier, and trap values. Existing results do not
support deleting these. In the checkpoint-weight diagnostic, coin/count values
and opponent contest/frontier values had among the largest first-layer column
norms; this is only supporting evidence, not an importance proof.

The remaining nine topology values are a named optional ablation, not part of
the primary pruning. A 75-input variant may remove all nine only after a matched
fixed-data comparison shows no meaningful loss. Do not combine that ablation
with a different curriculum or symmetry setting.

## 4. Network and learner

Use the established small learner:

```text
84 inputs -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(6)
algorithm                 Double DQN
discount                  0.99
batch size                128
learning rate             1e-4
replay capacity           100,000
train frequency           every 4 learner transitions
target synchronization    every 1,000 optimizer steps
loss                       smooth L1
gradient clipping         norm 10
```

Keep current and next legal/safety masks in replay and in target computation.
Keep the audited time-expanded safety fallback. Do not add a hard opponent-risk
veto: tournament wins sometimes require contested or aggressive actions.

Agent 038 must be self-contained. The submitted package may not import Agent
025, 030, 032, 037, `combat_dqn_agent`, or any imported opponent. Opponent
packages exist only in the training/evaluation staging runtime.

## 5. Fast initialization by policy distillation

Changing the input width prevents direct checkpoint continuation. Starting all
three seeds from random weights would discard the strongest learned navigation
and bombing behaviour and require another long warm-up.

Create a registered distillation dataset on training-only boards by rolling the
three Agent 037 episode-1050 checkpoints through solo and population Classic
lineups. Store:

- the 84-input Agent 038 vector;
- the original action mask;
- all six frozen Agent 037 Q-values;
- the teacher greedy action after masking;
- scenario, complete lineup, board, seat, and teacher checkpoint; and
- the D4 transform identifier when augmentation is applied.

Target 50,000--100,000 diverse decisions, balanced by scenario and complete
lineup. Distill the student with masked, centered-Q regression plus a small
greedy-action classification term. Validate on held-out distillation boards.

Required initialization gates:

- at least 95% agreement with the teacher's masked greedy action overall;
- no scenario/lineup below 90% agreement;
- finite Q-values and identical allowed-action support; and
- no worse than 5% score regression in a short supplied-agent smoke suite.

Distillation is initialization, not evidence that Agent 038 is stronger. The
subsequent population DDQN training must supply the improvement.

## 6. D4 symmetry for DDQN replay

For each sampled replay transition, independently sample one of the eight D4
transforms uniformly and transform:

- current and next feature vectors;
- executed action;
- current and next action masks;
- both stored recent-action one-hots;
- blocked/safe direction channels;
- local neighbour danger and topology positions;
- coin, crate, and opponent direction pairs; and
- the six action-conditioned opponent-response blocks.

`WAIT`, `BOMB`, scalar counts, danger at the current tile, history-success
flags, cycle flags, displacement, stagnation, and global armed-opponent status
are invariant. Rewards, terminal flags, scenario tags, and lineup tags are also
invariant.

Apply one transform at sample time rather than storing eight replay copies or
performing eight optimizer updates. This preserves replay capacity and nearly
preserves optimizer cost. The identity transform is one of the eight choices.

Required tests:

- eight distinct directional permutations;
- closure/inverse checks for feature, action, and mask transforms;
- transformed synthetic boards equal directly extracted transformed features;
- Q/action equivariance after inverse permutation on an intentionally
  symmetry-consistent test network; and
- current/next masks and `WAIT`/`BOMB` invariance.

An optional frozen-policy D4 ensemble may transform one already-computed feature
vector eight ways, evaluate one network batch, inverse-map Q-values, and average
them. Promote it only if it improves development rank and stays below the action
latency gate. Training augmentation does not require inference ensembling.

## 7. Frozen population curriculum

The tournament has three unknown opponents, so every Classic training episode
must contain a complete three-opponent lineup. Use frozen opponents to keep the
learning environment reproducible.

Primary training roster:

- `rule_based_agent` for the historical reference behaviour;
- `imp_li_deep_killer` for strong general scoring and survival;
- `imp_alii_arbiter` for aggressive combat pressure; and
- a frozen inference-only copy of the selected Agent 037 checkpoint for a
  familiar but learned policy.

Suggested generated-round shares:

| Share | Scenario / complete lineup |
| ---: | --- |
| 10% | Classic: rule-based ×3 |
| 25% | Classic: Deepkiller + Arbiter + rule-based |
| 20% | Classic: Deepkiller + Arbiter + frozen Agent 037 |
| 15% | Classic: seeded random three-agent sample from the four-policy roster |
| 15% | Coin Heaven |
| 15% | Loot Crate |

Use the same target shares in tagged replay before any bounded special-sequence
allocation. Rotate seats independently of board seed. Never identify a lineup
only as `classic`; preserve the complete ordered or canonicalized lineup tag.

Reserve these valid policies from all training and checkpoint selection:

- `imp_li_sarsa_lambda`;
- `imp_li_double_q`;
- corrected `imp_alii_sentinel`;
- `imp_alii_overlord`; and
- `harvy`.

They form an unseen-opponent generalization suite. If one is later introduced
into training, create a new experiment identity and replace it in the held-out
roster before launching the run.

Do not use three identical copies of one imported policy as the main training
distribution. That stress test is useful for evaluation but encourages a narrow
counter-policy.

## 8. Replay and reward adjustments

Agent 037 reserves up to 20% of sampled replay for post-bomb escape sequences.
Its strong survival but zero kills against three Deepkillers suggests that
escape-only emphasis should not grow further. Agent 038 should use:

- at most 10% post-bomb escape sequences;
- at most 10% attack-outcome sequences, beginning with the decision that led to
  a combat bomb and ending at detonation/death; and
- the remaining 80% from the registered scenario/lineup mixture.

The attack quota is filled only from real data; do not duplicate a handful of
rare kills without a minimum-support check. Log successful kills, opponent
deaths caused by others, failed pressure bombs, self-deaths, and safe escapes
separately.

Keep the current reward for the primary population comparison. It already uses
official values of +1 per coin and +5 per kill, with small crate/safety shaping.
Li's +500 predicted-kill reward must not be copied.

Run one later reward ablation only if population training remains crate-heavy
and kill-poor. That ablation may reduce Classic crate shaping from 0.2 to 0.05
while leaving solo Loot Crate reward unchanged. It must not be mixed silently
into the primary feature/symmetry experiment.

## 9. Controlled experiment sequence

The changes should be staged so failure remains interpretable.

### Stage A: feature and distillation audit

On one fixed collected dataset, compare:

1. the original 101-input teacher;
2. the 84-input distilled student; and
3. the optional 75-input no-topology student.

Measure action agreement, Q regression, per-lineup agreement, action latency,
parameter count, and memory. The 84-input model has 2,176 fewer first-layer
weights than the 101-input model, roughly a 7% reduction in the complete small
MLP. Do not promise a similar end-to-end speedup because opponent-window search
still dominates feature work.

Reject the 75-input variant unless it matches the 84-input variant within a
pre-registered tolerance. Promote 84 inputs if the distillation gates pass.

### Stage B: symmetry pilot

Using the same 84-input initialization and population schedule, run one seed
for 100--150 rounds with and without random D4 replay augmentation. Use only
development boards. Require no latency/safety regression and improvement in
at least two of: mean rank, score, seat variance, and direct imported-agent
non-loss rate.

If the pilot is tied, keep symmetry because it is cheap and removes directional
sampling bias only if confidence intervals and latency remain acceptable. If it
is materially worse, omit it from the full run rather than assuming more rounds
will repair it.

### Stage C: three-seed population training

Train seeds 0, 1, and 2 for 300--400 rounds from independently initialized
distilled students. Use a modest exploration restart, initially 0.10--0.15 and
decaying to 0.03, because the opponents are new. Compare that schedule with the
Agent 037 continuation value of 0.05 in the pilot; do not tune it on final
boards.

Save every 50 rounds. Select one global episode first, then one package seed,
using only development results and the gates below.

## 10. Evaluation designed for a four-team tournament

Boards 32000--32007 have already been inspected repeatedly and are no longer a
blind test for Agent 038 design decisions. Use new registered sets:

- training/distillation boards: existing training range plus explicitly logged
  new boards;
- development boards: 34000--34015;
- final blind boards: 35000--35015, opened once after checkpoint and seed are
  frozen.

Every Classic cell uses all four seats, action seed 0, and 400 steps. Evaluate
these development lineups:

1. rule-based ×3;
2. legacy rule-based + coin collector + peaceful;
3. Deepkiller + Arbiter + rule-based;
4. Deepkiller + Arbiter + frozen Agent 037; and
5. three Deepkiller copies as a stress test, not a training target.

The final blind suite repeats those lineups and adds unseen populations built
from SARSA-lambda, Double-Q, corrected Sentinel, Overlord, and Harvy. Keep at
least one complete held-out lineup untouched until the final run.

Primary tournament metrics are:

- mean rank and its full rank distribution;
- outright-first and joint-first rates;
- official score;
- direct wins/ties/losses versus each learned opponent;
- kills, coins, survival, suicides, and invalid actions; and
- performance by starting seat.

Checkpoint selection is lexicographic after safety gates:

1. lower pooled mean rank across development lineups;
2. higher outright-first rate;
3. higher pooled official score; and
4. lower suicide rate as the final tie-breaker.

Report board/seat paired differences against Agent 037, not just independent
means.

## 11. Promotion gates

Agent 038 replaces Agent 037 only if all mandatory gates pass.

### Correctness and deployment

- self-contained package with no sibling-agent or imported-agent runtime import;
- finite model/checkpoint values and deterministic frozen actions;
- exact 84-input schema and verified D4 transformations;
- clean-copy load and complete real games;
- one CPU thread, p95 action latency below 0.25 seconds and maximum below 0.45
  seconds; and
- no official-play multiprocessing or training artifact access.

### Safety and solo preservation

- zero invalid actions and zero self-deaths in the frozen solo suite;
- no more than 5% regression from the selected Agent 037 Coin Heaven and Loot
  Crate means; and
- no material increase in Classic suicides.

### Tournament improvement

- lower pooled mean rank than Agent 037 on development and final suites;
- higher pooled official score;
- improvement in at least two of three training seeds;
- no material regression in the three-rule-based historical cell;
- at least one outright first in the three-Deepkiller stress cell, improving on
  Agent 037's 0/32; and
- improve Agent 037's mixed Deepkiller direct record of 10/2/20, with a target
  of at least 50% non-losses on the matched final cell.

The last two values are useful concrete targets, not guarantees of statistical
significance.

## 12. Efficiency and cluster execution

The small network remains a CPU workload. Prior measurement found an RTX 2080
Ti slower than CPU for this training path. Use CUDA hidden and one math-library
thread per learner/evaluation worker. Store heavy run artifacts under
`/export/scratch/salitanl`, and launch long runs through `jobctl`.

Feature pruning should make the network slightly smaller, and sample-time D4
augmentation should be cheap. Population games may be slower than rule-based
games because Arbiter and frozen Agent 037 perform substantial search. Measure
rather than promise total speedup:

- feature and safety time;
- opponent decision time by package;
- environment seconds per round;
- replay transform/sample time;
- optimizer time;
- total wall time to each checkpoint; and
- evaluation games per second.

If Arbiter dominates wall time, preserve its replay share while generating its
rounds less often and retaining its transitions longer; do not replace combat
diversity with more rule-based games merely to claim faster training.

## 13. Implementation boundary

Create a new independent directory and leave Agent 037 unchanged:

```text
src/agent_code/Agent_038_symmetric_population_ddqn_agent/
    __init__.py
    callbacks.py
    config.py
    features.py
    model.py
    replay.py
    safety.py
    symmetry.py
    train.py
    tournament_checkpoint.pt
```

Also add focused tests for feature pruning, right-aligned history, symmetry,
distillation load, replay tags, masks, latency, and self-contained imports.
Imported opponents should be staged in an isolated scratch runtime using the
existing verified import machinery; they are not copied into the submission
agent.

## 14. What is deliberately excluded

Do not combine the first Agent 038 run with:

- Agent 031's rejected 12 offensive values;
- Li's stochastic target selector or `KILL!` feature/reward;
- a larger, dueling, recurrent, convolutional, or attention network;
- online-learning imported opponents;
- unrestricted self-play with a moving opponent policy;
- prioritized replay without bounded per-tag support;
- broad reward redesign;
- training or tuning on boards 32000--32007 or the new final blind boards; or
- removing the nine retained topology values without the named 84-versus-75
  ablation.

## 15. Expected outcome

The most likely source of improvement is population training against strong,
diverse frozen opponents. Exact feature pruning should reduce correlation and
slightly reduce model cost; D4 augmentation should improve directional sample
sharing and seat robustness. Neither supporting change is independently proven
to increase tournament score.

This design directly addresses the observed failure mode: Agent 037 survives
but scores poorly and produces no kills when all opponents are strong. Agent
038 is promoted only if it converts that safety into better rank, official
score, and direct results against both seen and unseen learned opponents.
