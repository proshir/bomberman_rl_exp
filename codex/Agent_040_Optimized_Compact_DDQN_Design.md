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
| Compact symmetry transform | 0.0673 ms | 0.0077 ms | about 8x faster |
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

## 21 September 2026: oscillation and combat-avoidance diagnosis

Status: measured frozen-policy diagnosis and proposed experiments. No agent,
reward, mask, checkpoint, or running training process was changed for this audit.
The initial status above describes implementation time; a separate 900-round
league-combat run has since completed.

### Evidence and reproducibility

Training artifacts: `/export/scratch/salitanl/agent040_regimen_900_20260921`.
Agent 041 comparison: `/export/scratch/salitanl/agent041_regimen_900_20260921`.
Comparison cutoff is episode 300 for Agent 041, whose training was ongoing.
Both runs start from scratch, use three seeds, 400-step games, board stride 900,
and development boards 34000--34003, all four seats. There are 48 games per
checkpoint/scenario. Input widths differ, so identical seed numbers do not
imply identical initial networks or subsequent experience.

The now-removed `src/diagnose_ddqn_behavior.py` instrument captured the actual
policy decision and computed diagnostics outside the timed callback. Outputs include
Q-values, full features, legal/safe candidates, bomb eligibility, own events,
geometry-based distances, history, and the hypothetical Agent 041 navigation
suffix for that same observation. It records checkpoint hashes. These suffixes
are representation probes, not Agent 041 counterfactual actions.

Artifacts: `/export/scratch/salitanl/agent040_behavior_audit_20260921/`:

- `classic_seat1/`: Agent 040 seed 0, board 34000, seat 1, episodes
  150/300/600/900, three rule-based opponents.
- `coin_seat3/`: Agent 040 seed 0, board 34003, seat 3, episodes
  200/300/600/900, solo Coin Heaven.
- `agent041_classic_seat1/`: Agent 041 seed 0, board 34000, seat 1,
  episode 300, three rule-based opponents.

All nine replays match saved evaluation score, coins, crates, kills, suicides,
bombs, learner steps and alive/dead status. All eight Agent 040 traces have
zero differences between requested and engine-executed actions. Timeouts in
the action callback therefore do not explain these examples.

### The exact episode-600 bad video

This is the game in
`/export/scratch/salitanl/agent040_episode600_classic_bad_seed0_board34000_seat1.mp4`.

- Final: score 0, coins 0, kills 0, crates 31, bombs 21, survived 400 steps.
- WAIT on 186 turns (46.5%). The last 50 decisions occupy just (13,3) and
  (13,4), including waits.
- From decision 272 onward, no crates or visible coins remain; two opponents
  remain. No personal productive event occurs after turn 271.
- On turns 301--400 a safe candidate reduces geometry-based BFS distance to
  an opponent on **100/100** turns, but it selects such a move on only 24.
  It waits on 54 turns, and no currently scheduled explosion threatens its
  current tile on any of those 100 turns. This does not prove every approach
  would succeed against future enemy bombs, but disproves a lack of permitted
  approach moves as the explanation.
- On turn 394 at (13,4), UP, DOWN and WAIT are all candidates. DOWN reduces
  opponent distance from 9 to 8. Q(WAIT)=3.37363, Q(UP)=3.35824,
  Q(DOWN)=3.34891; the learned ranking selects WAIT.
- Bomb is eligible on 23 turns in the full game, and selected on 21. Five
  placements have an opponent currently in the blast and zero crate hits.
  There is no blanket refusal to bomb when admitted by the filter.
- In the last 100 turns, hypothetical bombing passes the survival search on
  every turn but fails the usefulness test on every turn: no crate or current
  opponent is in its blast. Dropping distant bombs alone would not solve the
  missing approach behavior.
- 26 history resets follow changes without an own coin/crate/kill/discovery
  event on the preceding step. Global opponent activity erases personal
  repetition memory. This matters earlier, but is not the cause of the final
  empty-arena loop, where no reset occurs.
- Its total undiscounted training reward would be **+5.30**, despite zero
  official score: +6.20 crates, +0.20 discovery, +2.40 escape shaping,
  +0.50 survival, -4.00 step cost. This exposes an objective mismatch, not
  proof that any individual shaping term caused the policy.

Same Classic board/seat with Agent 040 checkpoints 150/300/600/900 yields
scores 0/2/0/12; the final checkpoint earns two kills. The specific game
recovers, but population-level navigation failures persist.

### Development across checkpoints

Pooled across all three training seeds and 16 development games per seed:

| Agent 040 episode | Coin Heaven coins | Completed /48 | Repeated states/game | Classic score | Classic kills/game | Classic survival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 49.917 | 44 | 23.08 | 2.083 | 0.083 | 50.0% |
| 200 | 49.979 | 47 | 7.56 | 3.271 | 0.208 | 37.5% |
| 300 | 49.292 | 31 | 101.85 | 2.958 | 0.188 | 50.0% |
| 600 | 48.833 | 20 | 166.27 | 3.854 | 0.208 | 60.4% |
| 900 | 48.833 | 15 | 193.21 | 3.792 | 0.250 | 77.1% |

Repeated states here mean identical position/remaining-coin-set observations
in solo evaluation, not a measured Classic oscillation rate. Standard Classic
evaluation had diagnostics disabled. All solo games survived, so the completion
collapse cannot be explained by dying early. Mean coins near 49/50 hides long
tails spent failing to collect the final coins. Degradation starts before
opponent training at episode 301, during the crate-heavy stage; later combat
training is therefore not the sole possible source of interference.

On the selected Coin Heaven board, the final collection occurs at turns
129/114/109/109 at checkpoints 200/300/600/900. Respectively, it then enters
a two-tile loop / WAIT loop / WAIT loop / two-tile loop. At checkpoint 600,
it waits at (3,1) while a coin is one safe LEFT away. Q(WAIT)=5.06603 versus
Q(LEFT)=5.06002. Its last 100 observations are **identical 104-feature vectors**.
At checkpoint 900 the last 100 observations alternate between two vectors.
History caps and greedy inference allow a permanent recurrent pattern.
There is no demonstrated collision between different full states requiring
opposite routes in this audit; do not call this a newly proved feature-aliasing
counterexample. What is directly measured is wrong ranking and saturated memory.

For the episode-600 Coin Heaven trace, Q(WAIT) at turn 301 is 5.066, whereas
the discounted realized return over the remaining 100 turns is -0.449 at
gamma=.99. This is calibration failure on the frozen greedy rollout, not a
claim about the optimal action value or the original minibatch TD residual.
The historical unrestricted-bootstrap and next-state-cache bugs are repaired
in this DDQN path and should not be reused as explanations without new evidence.

### Root causes and what Agent 041 addresses

| Mechanism | Evidence / status | Agent 041 coverage |
| --- | --- | --- |
| Wrong ranking of route-improving moves | Directly measured, including a coin one move away | Signed coin-distance changes make the decision easier to learn; no enforcement or guarantee |
| Returning to recently occupied destinations | Directly measured | Adds per-action destination counts, a useful refinement over aggregate history |
| History cleared by other players | Source and 26 resets in this game | Unchanged; new visit counts inherit the resets |
| Finite memory saturates into a recurring observation | Directly measured identical/alternating feature vectors | Still history length 8, count capped at 3, stagnation capped above 32 steps |
| No explicit route toward an attack position | Source; safe approach repeatedly ignored | Unchanged: new routes concern coins only |
| No coin/crate task remains | Empty arena from turn 272 | All 12 new progress/reachability values are zero for every remaining turn; only visits remain informative |
| Combat exploration spent before combat begins | Source and logs | Unchanged |
| Shaped returns reward crates/escape without official points | +5.30 reward at zero official score | Unchanged reward |
| Bomb gate excludes preparatory placements | Confirmed source; usefulness requires a present blast target | Unchanged |
| Conservative bomb-escape search | Opponent possible-occupancy union, spare escape step; 21 useful bombs rejected by survival in this trace | Unchanged; those rejections are not proven unnecessary |
| Navigation retention worsens over training | Development results | No new replay retention or distillation mechanism |

Agent 040 already contains coin/crate route distances and reachability. Agent
041's BFS is reused from that representation; it is not a new time-expanded
route planner. Beyond immediate legality it does not model future bomb/agent
occupancy along coin routes. Its important additions are explicit relative
coin progress and destination-specific visits. Bomb entries in all three new
blocks are zero. Its inherited opponent features primarily describe danger,
contested space and survival, plus coarse nearest-opponent direction/Manhattan
distance (all distances above seven share one bucket); they do not supply
action-specific paths to safe attack tiles or expected offensive pressure.

The combat exploration issue is concrete: `train.remember` increments
`combat_env_steps` on **every** transition, including solo games. On the first
combat episode (301), Agent 040 epsilons are .0500/.0552/.0500, and Agent 041
epsilons are .05088/.0500/.05280. The .30 initial exploration has already been
spent learning solo behavior. Calling the variable "combat" does not make
its schedule combat-specific.

At episode 300, Agent 041 collects 48.729 coins with 22/48 completions and
153.15 repeated states/game; Agent 040 at the same episode has 49.292,
31/48 and 101.85. In Classic, scores are 2.604 vs 2.958 and survival 37.5%
vs 50.0%. Neither has trained on opponents yet at this point. These early
development results establish that Agent 041 does not automatically eliminate
loops; they do not determine its eventual combat performance.

### Proposed fixes, in priority order

These are proposals, not implemented or proven improvements.

1. **Separate exploration by task.** Count genuine opponent transitions for
   combat decay and start the combat phase around epsilon .20--.30, decaying
   over actual combat experience. Keep solo exploration/retention separate.
   Start with this single-factor control because it changes no representation,
   reward or safety contract. Safe random actions still may not discover long
   approach/attack sequences, so this is necessary to test, not sufficient.

2. **Repair progress memory.** Keep rolling position/action history independent
   of global board changes. Track personal score/collection progress separately
   from map change and attacking/crate activity. Training may use own events;
   inference must reconstruct compatible signals from observations because
   evaluation agents do not receive training callbacks. Agent 036 already has
   an attributable-progress/independent-history implementation worth adapting.
   Do not simply lengthen the deque: longer history did not reliably win in
   earlier project pilots. Preserve exact action-time replay features.

3. **Teach attack positioning and late-game combat explicitly.** Add a compact
   action-aligned route to a position from which a bomb threatens a reachable
   opponent and retains an escape, together with opponent escape-space change
   or trap feasibility. Include goal/phase and target persistence if multi-step
   commitment is introduced. Train on sampled empty/low-crate arenas with one,
   two and three active opponents, then mix these into full Classic games and
   retain solo tasks. Training modifications are allowed by the assignment;
   final gates must use the unchanged official engine. Close spawns alone risk
   teaching bombing without teaching approach, so vary separation and obstacles.

4. **Improve credit assignment and replay support.** Add tagged late-game and
   successful approach--placement--escape sequences, while retaining failures
   and normal games. The existing 20% escape quota begins at bomb placement;
   it does not explicitly reserve preceding approach decisions or successful
   attacks. It is not evidence that escape is oversampled: actual prevalence
   must be measured. Test short n-step returns (e.g. 3--5) separately, with
   terminal handling, gamma^n bootstrap, compatible masks and documented
   off-policy assumptions. Consider balanced prioritization after this; raw
   high-TD sampling alone may miss well-established, small-residual loops.

5. **Test reward alignment after exploration/representation.** Preserve actual
   coin/kill outcomes; audit crate, discovery and escape shaping. The current
   escape bonus can reward repeatedly leaving self-created danger, but it did
   not reward the empty-arena tail here (99 rewards -.01, final reward .49).
   Test reducing redundant shaping, or a small bounded potential for useful
   approach with `gamma*Phi(next)-Phi(current)` and zero terminal potential.
   Use the same goal/phase definition in state and replay; do not pay an
   unconditional reward for each approach step or bomb. Correct potential
   shaping has an underlying-MDP invariance result, not a guarantee for this
   compressed neural learner. Existing project tests of naive distance rewards
   were unfavorable; a new shaping proposal must be a controlled experiment.

6. **Audit bomb eligibility before broadening it.** Distinguish physically
   illegal, no certified escape, and no current target. A measured offensive
   opportunity rejected solely by the usefulness heuristic could justify
   admitting safe preparatory bombs with predicted escape-space reduction.
   Preserve the escape check initially. Do not remove opponent-aware escape
   protection just to increase bomb count: attacking safely still matters.
   At the end of the supplied game, the immediate bottleneck is approach.

7. **Use a loop guard only as an explicit diagnostic/hybrid variant.** A blanket
   reversal ban is unsafe because bomb escape and corridor traversal require
   reversals. A bounded intervention should require persistent no-progress,
   repeated observations, a certified alternative and suitable danger context.
   If eventually deployed, train and evaluate the same policy/mask and count
   interventions. It can force departure, but cannot by itself teach combat.
   The project's learned-action goal makes a learned correction preferable.

The previous Agent 031 experiment already added 12 offensive inputs. Its
three-rule-based score was 2.990 vs 3.115 for the matched control; it was not
promoted. See `offensive_feature_comparison.md`. This is evidence against
assuming another feature expansion alone will solve four-player combat.

### Experiment and selection proposal

First compare Agent 040 unchanged against combat-counter repair only, then
history repair only, then the combination; use Agent 041 as the existing
coin-navigation-feature comparison. Subsequent offensive features/curriculum
and reward changes should be isolated rather than bundled. Use matched seeds,
interaction budgets and opponent schedules, plus untouched boards after
selection. The already inspected boards are diagnostic/development boards.

Record solo completion and repeated states, Classic win/score/kills/survival,
zero-score timeouts, personal no-progress tail, repeated 2/4-step patterns,
safe approach opportunities taken, eligible attack opportunities taken,
mask rejection reasons and post-clearance behavior. Do not reward more
movement or fewer waits when they simply cause earlier death. Keep a minimum
navigation completion criterion when choosing combat checkpoints: average
coins alone hides the observed late-game failures.

Reference for the potential-shaping proposal: Ng, Harada and Russell (1999),
[Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

### Agent 042 follow-up (21 September 2026)

The first, second, and representation portions of the proposal were implemented
as [`Agent_042_Combat_Progress_DDQN_Design.md`](Agent_042_Combat_Progress_DDQN_Design.md):
Agent 042 retains Agent 041's coin-navigation block, adds action-aligned
opponent progress/reachability/pressure features, preserves rolling history
through unrelated global changes, and separates solo from combat epsilon
accounting. Its three-seed 300-round solo gate reduced repeated states from
280.9 at round 50 to 17.8 at round 150, but the final round-300 checkpoint
returned to 109.8 repeated states and 116.6 maximum no-progress steps. Thus
the fixes are useful but do not close the loop; novelty intervention,
attribution, target persistence, reward/credit assignment, and combat
curriculum remain explicitly deferred re-analysis items.
