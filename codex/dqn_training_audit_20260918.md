# Why combat DQN training degrades: code and research audit

Sahand was here.

## Scope and conclusion

This is a diagnosis, not a new algorithm or training result. Inspected source
commit `71b0f5a4e9ff1bf06d0cf71ab84fb2902097b00b`, the algorithm knowledge base,
project specification, combat experiments, and earlier navigation/anti-loop
diagnoses. Checked upstream engine semantics and several public forks/projects.
No agent implementation or trained checkpoint was changed during the audit.

## Correctness repairs completed

Following this audit, the vanilla DQN was repaired on the combat branch.
Replay now stores a six-action next-state mask; terminal rows store an empty
mask. The vanilla target applies that mask before taking the target-network
maximum and gives terminal or empty-mask rows zero bootstrap. Transition
commitment now uses the feature cache from the exact following `act()` call,
so replay and inference share the same step and history context.

Regression coverage now includes masked high-value targets, terminal empty
masks, replay-mask preservation, and exact cached next-feature usage. The full
suite passes **29 tests**, and a fresh three-round, 30-step CPU smoke run
completed checkpoint saving and evaluation. The historical measurements below
remain pre-repair evidence; no performance claim is made until a corrected
matched pilot is run.

The strongest finding is a correctness error: DQN constrains actions during
play but bootstraps from unconstrained next actions during learning. A second
error makes replay's next-state stagnation feature disagree with action-time
features. The representation also demonstrably merges different route choices,
even with all 46 features. These issues should be addressed before attributing
failure to insufficient network capacity, training duration, or the 14 extra
features.

The earlier statement that the topology features *caused* the collapse was too
strong. Their measured run failed, but both DQN variants use the incorrect
target, and the 32-feature controls predate the PyTorch seeding fix. The
topology candidate should stay experimental; its independent effect remains
unresolved.

## 1. Confirmed: action constraints disappear from the learning target

Relevant source:

- `src/agent_code/combat_dqn_agent/callbacks.py:132`: action selection uses
  `safe_action_indices`, with `best_survival_action_indices` as fallback.
- `src/agent_code/combat_dqn_agent/replay.py`: transitions contain no next-action
  mask.
- `src/agent_code/combat_dqn_agent/model.py:33`: `vanilla_targets` takes the
  unrestricted maximum of all six target-network outputs.
- `src/agent_code/combat_fqi_history_antistag_agent/train.py`: the FQI baseline
  stores next-action masks and excludes disallowed values before maximizing.

This creates a mismatch between the behavior being learned and the behavior
that can be executed. An unavailable, unsafe, or useless bomb can receive a
large predicted value. The learner uses that prediction to raise the value of
preceding actions, even though it never executes that bomb in that state and
therefore never obtains evidence correcting its value there. Parameter sharing
can change these unobserved predictions further. A target network slows target
changes; it does not make an incorrect maximum correct.

The required target is still **vanilla DQN**:

`reward + gamma * max(target_Q(next_state, a) for a in next_candidates)`

The target network performs both selection and evaluation. `next_candidates`
must match the full action-time safe/useful filter and survival fallback;
masking only physically illegal moves is insufficient. Terminal transitions
must have zero bootstrap, with explicit handling to avoid `0 * -inf` when
terminal masks are empty. Double DQN is a separate subsequent ablation.

This agrees with the requirement to mask Q-learning updates as well as
exploration and exploitation in [Gymnasium's action-masking documentation](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/).
Extrapolation from insufficiently supported actions is also a central concern
in [Fujimoto et al., Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900).
Our learner is online with replay, not that paper's offline setting; the paper
supports the mechanism, not a numerical prediction for this experiment.

### Frozen-checkpoint evidence

Replayed 48 games: both 32- and 46-feature 600-round experiment families,
training seeds 0/1/2, checkpoints 100/600, board 30000, all four seats. Every
game reproduced its saved coin and crate totals. This is a targeted diagnostic
on one board, not a fresh generalization test or an estimate of the training
replay distribution.

For each visited state, measured whether the target network's unrestricted
argmax was outside the action-time candidate set:

| Model / checkpoint | Excluded target argmax | Physically illegal target argmax | WAIT actions | Mean final no-progress tail |
| --- | ---: | ---: | ---: | ---: |
| 32 features, round 100 | 91.02% | 6.17% | 60.83% | 326.8 steps |
| 32 features, round 600 | 89.77% | 2.10% | 83.13% | 350.3 steps |
| 46 features, round 100 | 85.77% | 16.48% | 63.69% | 268.7 steps |
| 46 features, round 600 | 98.25% | 16.94% | 85.98% | 367.3 steps |

Each row contains 12 games, 4,800 decisions. Most excluded maxima are not
physically illegal; they include actions rejected by the safe/useful gate.

A concrete failure is topology seed 1, round 600, board 30000, seat 0:

- One coin, seven crates, 384 WAIT actions in 400 steps.
- At step 200, position `(2, 1)`, one visible coin; permitted actions are
  RIGHT, LEFT, WAIT. There is no useful bomb at this tile.
- Policy Q-values: RIGHT 193.64, LEFT 193.14, WAIT 204.35, BOMB 217.78.
- Target Q-values: RIGHT 183.51, LEFT 183.47, WAIT 193.80, BOMB 206.32.
- For this continuing WAIT self-loop, reward is -0.01. The current formula
  gives approximately `-0.01 + .99 * 206.32 = 204.25`, almost the policy's
  204.35 prediction. Masking would instead give approximately 191.85.

Thus the network can fit its erroneous target closely while behaving badly.
In this solo task, even an extremely loose whole-episode upper bound on all
positive rewards is 135.7: at most 50 collected coins, 176 possible crates
times 0.2, 50 reveals times 0.2, 400 escape bonuses times 0.1, and survival
0.5. This ignores discounting, negative rewards, and already consumed resources.
The Q-values above exceed even that generous bound.

For topology seed 1, mean unrestricted target maximum on the four diagnostic
games rises from 2.33 at round 100 to 213.23 at round 600. The mean difference
between unrestricted and permitted maxima rises from 0.18 to 12.49. This is
strong evidence of inflated estimates and a concrete mechanism sustaining them.
A corrected training ablation is still needed to measure how much performance
the mask repair recovers.

## 2. Confirmed: next-state history does not always match the next decision

The engine increments `step` before acting. The old observation and post-action
observation delivered to `game_events_occurred` therefore both carry step `t`.
The next invocation of `act` carries step `t+1`. This is also the behavior of
the [upstream engine](https://raw.githubusercontent.com/ukoethe/bomberman_rl/master/environment.py).

`combat_dqn_agent/callbacks.py:118` reconstructs the future stagnation clock
using the post-action observation's step. When no progress occurs, this is one
step behind what the next decision sees, occasionally crossing a bucket
boundary. The same reconstruction pattern exists in the history-FQI baseline.

During the 48-game diagnostic, reconstructed next features disagreed with the
following actual action features 1,028 times out of 19,152 adjacent decision
pairs. All observed differences were feature index 31, the stagnation bucket.
This is a smaller defect than the unmasked target, but matters because the
history representation was introduced specifically to distinguish stagnation.

The repair should use the actual following decision's cached features, or
reconstruct its clock with the correct engine semantics and progress-reset
rules. Test unchanged progress, new coins/crates/score, bucket boundaries,
round resets, and terminal transitions using real callback order.

The existing 27-test suite passes. It tests action-time masking and a numerical
unrestricted target separately, but has no constrained-target regression test
or engine-level replay/action feature equality test. Passing it did not
establish these properties.

## 3. Confirmed representation limitation, including the 46-feature input

`combat_fqi_agent/features.py` finds a reachable coin using BFS, then retains
the signs of target displacement and a coarse path-distance bucket. It does
not preserve the route. Distances beyond seven merge into one value. Previous
action is a scalar action index; history provides a capped count of visits to
the current tile, not the actual eight-position sequence. Stagnation saturates
after 32 steps. The topology addition describes only a 3x3 patch and a few
aggregate counts.

Constructed an exact witness on the normal 17x17 stone-wall pattern. Agent at
`(7,7)`, coin at `(9,7)`, crate at `(8,7)`; a second crate is either `(7,5)` or
`(7,9)`. All other traversable cells are free. Use identical empty history and
no bombs/opponents.

| Distant crate | Current coin distance | Distance after UP | Distance after DOWN | Distance after LEFT |
| --- | ---: | ---: | ---: | ---: |
| `(7,5)` | 6 | 7 | 5 | 7 |
| `(7,9)` | 6 | 5 | 7 | 7 |

The complete 46-dimensional vectors are **exactly identical**, but the unique
coin-distance-reducing moves are opposite. No deterministic feed-forward
function of these inputs can select the route-improving move in both cases.
This is a route-information counterexample, not proof that nearest-coin
movement maximizes total shaped combat return; bombing is also available.

It extends the earlier measured aliasing evidence in
`tree_navigation_diagnosis.md` and `tree_fqi_history_loop_diagnosis.md` to the
current representation. More MLP units cannot recover information that the
feature extractor discarded. More history counts cannot reconstruct arbitrary
remote geometry either.

After correctness fixes, a representation experiment should preserve useful
route information across all candidate actions and distinguish this witness
before training. Exact/normalized route costs and reachable-target context are
possible compact inputs; a board tensor is a larger alternative. Earlier
coarsely bucketed route features and naive distance rewards already failed,
so repeating those changes without testing what information they preserve is
not justified. Action selection must remain learned, as required by the brief.

## 4. Why longer training can make these runs worse

Each 600-round model saw 240,000 environment transitions and performed 58,751
optimizer updates. Updates are running; this is not a skipped-warmup failure.
Epsilon reaches 0.05 at 100,000 transitions, approximately round 250. The
100,000-transition FIFO buffer holds about 250 full rounds. At round 600 it
contains approximately rounds 351–600, all at the exploration floor.

The plausible feedback mechanism is: inflated targets corrupt action rankings,
greedy behavior stalls, less exploration produces fewer useful trajectories,
and replay gradually replaces earlier diverse experiences with the later
behavior. We measured the target inflation and frozen loops; replay contents
were not saved, so the exact amount of loop data in training replay is not
measured. This is not proof that epsilon decay alone causes the collapse.

Exploratory training can also look much better than frozen play. For 32-feature
600-round seed 0, the final 100 training rounds averaged 36.97 coins; its final
greedy evaluation averaged 12.72. These use different boards and changing
versus frozen weights, so the gap does not isolate exploration. It does show
why training reward is insufficient for choosing a submission checkpoint.

Gamma is 0.99 for DQN versus 0.95 for FQI. The rough discount horizons are 100
versus 20 steps. Higher gamma increases dependence on bootstrap accuracy, but
is not intrinsically wrong. Target sync is every 1,000 optimizer steps, about
4,000 environment steps. Batch size 128 with an update every four steps gives
32 replay draws per new interaction after warmup. Those settings should be
tuned only after targets are valid.

## 5. Rewards, safety, and the actual project objective

The reward is coin +1, kill +5, crate +0.2, reveal +0.2, survive +0.5,
escape improvement +0.1, death -5, invalid -1, useless bomb -0.1, and step
-0.01. Survival is paid once at round end. Pure waiting for 400 steps therefore
earns -3.5 before discounting, not a profitable survival bonus every step.

Crate rewards give a necessary early signal, but the final objective pays for
coins and kills. Escape shaping is not implemented as a potential difference
and can reward moving out of danger without equally charging movement back
into it. Reward-component logging and a separate shaping ablation are useful;
the current evidence does not identify shaping as the cause of the stationary
WAIT collapse. If testing potential shaping, use the discounted form
`gamma * Phi(next_state) - Phi(state)` with appropriate terminal handling.
[Ng, Harada and Russell's paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
gives the policy-invariance conditions; it does not promise to repair aliased
inputs or bad Bellman targets.

Perfect solo survival mainly validates the deterministic safety shield on
these boards. It is not evidence of learned combat strength. The current DQN
pilots have no opponents, so they contain no opponent kills or opponent
behavior to learn from. They use loot-crate with 50 coins; classic has nine.
After stable Task 2 learning, retain a Coin Heaven regression test and evaluate
on classic before hunting and competition. The supplied project brief asks for
all four tasks, not just a high loot-crate coin count.

The 400-step task really terminates at its time limit, so zero terminal
bootstrap is appropriate. Remaining time is absent from the feature vector,
which aliases otherwise equal early and late states. This is a representation
limitation, not a reason to bootstrap beyond the game's actual end.
[Gymnasium's finite-horizon guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
distinguishes this from externally truncated continuing tasks.

## 6. Experiment comparisons need qualification

| Run | Round 100 | Round 200 | Round 300 | Round 400 | Round 500 | Round 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32-feature fresh 600-round DQN | 8.43 | 9.96 | 8.58 | 9.20 | 6.65 | 6.28 |
| 46-feature seeded 600-round DQN | 16.53 | 12.81 | 4.78 | 2.25 | 2.48 | 2.86 |

These are existing full evaluation means over three checkpoints/seeds and
eight boards times four seats per checkpoint, not the smaller audit sample.
The separate 300-round 32-feature pilot reached 18.48; matched FQI reached
19.35. These are descriptive comparisons, not statistically established
algorithm rankings.

- The 600-round runs were fresh starts, not resumes of the 300-round models.
- The earlier DQN runs did not seed PyTorch initialization. Fixing a seed makes
  runs repeatable; it does not remove variance between different seeds or fix
  unstable learning. A corrected 32-feature control has not been run.
- Training board ranges depend on requested duration: the 300-round seeds
  start at 4000/4300/4600; 600-round seeds start at 4000/4600/5200. Consequently
  the first 300 rounds of these experiments are not identical across all seeds.
- FQI and DQN differ in gamma, exploration schedule, replay capacity and update
  process. The comparison evaluates complete configurations, not just trees
  versus neural function approximation.
- The combat pilots initialize fresh models; they do not load the strong
  Stage-1 navigation checkpoint. A weaker Coin Heaven score from a separately
  trained combat model is not, by itself, evidence of catastrophic forgetting
  of pretrained navigation. Also, 300 alternating rounds provide about 150
  crate rounds, versus 300 for crate-only training; that comparison changes
  crate exposure as well as curriculum. Distinguish these effects in the report.
- Repeatedly used boards 30000–30007 are validation/development data once they
  inform model choices. Preserve another board set for final confirmation.
- Topology DQN's recorded `hyperparameters` is `{}` because its training module
  re-exports functions rather than constants. PyTorch version is also absent
  from the experiment configuration. Transitive source hashing omits some base
  reward/feature/safety dependencies. The checked source hashes that were
  recorded all match the present tree. Record resolved settings and all shared
  dependencies in future runs.

## 7. What upstream forks and other BombeRLe projects actually teach

Inspected the upstream GitHub fork list and selected public repositories with
substantive implementations. These projects use different years, budgets,
rewards and opponents; their reported scores are not directly comparable to
our pilot.

| Project | Verified approach | Relevant lesson |
| --- | --- | --- |
| [KunkelAlexander, direct fork](https://github.com/KunkelAlexander/bomberman_rl) | Tabular and CNN Q-learning; optional Double DQN and prioritized replay. Its [training source](https://raw.githubusercontent.com/KunkelAlexander/bomberman_rl/master/q_deep_agent.py) stores next legal actions and masks them for both vanilla and Double targets. | Target masking is independent of algorithm sophistication. Spatial input is richer than a 3x3 static patch. |
| [Maverick](https://github.com/nickstr15/bomberman) | [23-input, 60-hidden-unit, 6-output MLP](https://raw.githubusercontent.com/nickstr15/bomberman/master/agent_code/maverick/Model.py), engineered features, five-step updates and residual-based sample selection. [Training configuration](https://raw.githubusercontent.com/nickstr15/bomberman/master/agent_code/maverick/train.py) specifies 20,000 episodes. | A small network can be useful; representation, targets and training protocol matter. Configured episode count is not independently verified training provenance. |
| [NOBEL](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game) | CNN, dueling values, prioritized replay, symmetry handling, and separate coin/crate/full-game versions. | Successful deep approaches combine spatial information and staged training; their results do not isolate which addition helped. |
| [ivo-1, direct fork](https://github.com/ivo-1/bomberman_rl) | Tabular Q-learning and a Decision Transformer; README identifies the Q-learning agent as the tournament submission. | More sophisticated model families do not automatically become the chosen submission. |
| [Abhinand-p, direct fork](https://github.com/Abhinand-p/Reinforcement-Learning-Bomberman) | Q-table with pathfinding, blockage, bomb and crate features plus custom events. | Compact engineered approaches remain relevant. README success claims do not replace matched evaluation. |

The [Double DQN paper](https://arxiv.org/abs/1509.06461) motivates separating
action selection from evaluation to reduce maximization bias. That is a
reasonable later experiment, but an unmasked Double DQN still selects actions
outside our deployed policy's allowed set. It would not repair the primary
bug by itself. No external agent code was incorporated into our implementation.

## 8. Recommended progression, not yet implemented

1. Repair next-action masking in replay/targets and the history clock mismatch.
   Keep the vanilla target network, 128-128 MLP, rewards, and existing inputs.
   Add regression tests for excluded high Q-values, terminal rows, fallback
   masks and actual engine callback parity. Cover seeding at the model setup
   entry point if direct framework training is supported.
2. Run a paired corrected 32-/46-feature comparison with the same seed list,
   board sequences, interaction budget and evaluation schedule. Evaluate every
   25–50 rounds so a brief early peak is not missed. Track Q/target quantiles,
   masked versus unmasked maxima, WAIT/move/bomb frequencies, reward components,
   loops and no-progress tails alongside score/survival/loss. Start with a
   300-round gate; extend only when corrected curves justify it.
3. Choose a validation checkpoint, then confirm it on unused boards. Call the
   existing results pre-fix baselines; do not erase them or claim the new
   algorithm is better before measuring it.
4. If corrected learning is stable but still loops, test one representation
   change that resolves the route witness. Include remaining time. Check the
   previous failed route-feature experiments before choosing the encoding.
   Symmetry augmentation and categorical encoding are later controlled options.
5. Then test Double DQN on the retained representation. N-step returns may help
   delayed bombing credit after that. Keep each change attributable rather than
   combining network, replay, reward, exploration and representation changes.
6. Progress through classic crates, peaceful opponents and stronger opponents
   with a lightweight Coin Heaven regression test. There is no need to require
   perfect Coin Heaven performance before every combat experiment.

Before submission, package shared feature/reward/safety dependencies inside
the submitted agent; the current DQN imports sibling agent packages. Its setup
also refuses an existing model in training mode, so saved optimizer state does
not currently provide a resume workflow. Those are compatibility limitations,
not explanations for the completed fresh-run collapse.

## Verification and artifacts

The 27 existing tests passed in 6.134 seconds with CUDA hidden and one CPU
thread, following the IWR skill's checkpoint-evaluation guidance. Target
calculations are detached; warmup, selected-action gathering, optimizer updates
and target copying exist. The audit did not identify a reversed terminal mask.

Diagnostic script and results:

- `/export/scratch/salitanl/dqn-audit-ArjDXZ/audit.py`
- `/export/scratch/salitanl/dqn-audit-ArjDXZ/results.json`

The probe uses frozen models and reads the actual engine's action-time feature
cache. Results are descriptive per-decision measurements, with correlated
steps and only one diagnostic board. They are not confidence intervals or
post-fix performance estimates. PyTorch observed during audit: `2.7.1+cu118`.
