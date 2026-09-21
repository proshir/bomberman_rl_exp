# DDQN run audit and evidence-derived training regimen

**Audit date:** 21 September 2026

**Artifact cutoff:** 11:53 CEST

**Scope:** all durable notes in `codex/`, all result families under
`experiments/`, the recorded `jobctl` history, and completed Agent 032--037
scratch artifacts. Agent 038 is treated as an active experiment, not a result.

## Executive decision

The strongest defensible training recipe is a **small, masked, ordinary
Double DQN trained with a staged curriculum and lineup-aware replay**. Use the
101-input Agent 029/030/037 representation and the 128--128 MLP as the current
control. Do not make a larger/dueling network, an indiscriminate feature
expansion, a large opponent league, or a longer run the default.

The evidence says, in descending order of confidence:

1. Bellman-target legality and exact action-time transition semantics are hard
   correctness requirements. Results produced before those repairs cannot be
   used to infer which DQN variant is best.
2. Curriculum and replay composition have mattered more than the 46-versus-52
   feature change or the 128-versus-256 architecture change.
3. Solo competence must be learned and continuously rehearsed. A combat-only
   continuation can improve one matchup while silently losing navigation,
   crate efficiency, or performance against a different opponent.
4. Focused training on two or three complete four-player lineups is preferable
   to the 24-lineup league tried with Agent 032. The broad league did not beat
   its built-in-only control, whereas Agent 037's focused continuation improved
   its relevant pooled matchup score.
5. Training is non-monotonic. Select checkpoints by held-out board/seat
   evaluation across three seeds; never assume the final checkpoint is best.
6. For the present small vector MLP, CPU is the correct device. The measured
   GPU runs were slower per environment step.

This is a regimen for the next controlled DDQN family. It does not retroactively
change the two Agent 038 jobs already running.

## 1. What was audited

At the cutoff, the repository contained 82 top-level experiment directories
and 76 recorded `jobctl` jobs. The job records were:

| State | Count | Scientific interpretation |
| --- | ---: | --- |
| Succeeded | 43 | Candidate evidence only after checking protocol and artifact completeness |
| Failed | 15 | Usually launch, path, evaluation, or implementation failures; not negative learning results |
| Stopped | 15 | Mostly superseded/debug/device-timing runs; incomplete unless explicitly described otherwise |
| Blocked | 1 | Operational state, not model evidence |
| Running | 2 | Agent 038 warm and scratch branches; no completed training claim |

The directory and job counts are operational inventory, not 82 independent
scientific comparisons. Smokes, timing tests, duplicate launches, old-schema
checkpoints, and incomplete evaluations were separated from valid learning
runs.

### Evidence classes used in this audit

| Class | Meaning | Examples |
| --- | --- | --- |
| Valid comparison | Complete artifacts, compatible protocol, corrected implementation | Fixed 46/52 DDQN 2x2 suite; Agents 027--031; corrected Agent 037 evaluation |
| Qualified evidence | Useful, but protocol, seed count, horizon, or evaluation coverage differs | Stage-1 tree results; imported agents; Agent 033 partial population evaluation |
| Invalid for performance inference | Known semantic bug or only a smoke/timing run | Pre-fix DQN targets; Agent 034 schema-v1 FQI; Agent 030 callback-bypass continuations |
| Incomplete | Still running or lacks the required final evaluation | Agent 038; Agent 033 after global round 600 |

Means from different board sets, horizons, scenarios, or opponent lineups are
not treated as if they were a single leaderboard.

## 2. Evidence from the run history

### 2.1 Navigation and compact baselines

The early experiments established three enduring facts: legal actions matter,
state aliasing produces loops, and adding raw capacity does not automatically
fix either problem.

| Family | Representative result | What it established |
| --- | ---: | --- |
| Supplied coin collector | 43.08 coins/100 steps; 50 at longer horizons | Strong hand-built navigation reference |
| Plain/distance tabular Q | 17.47 / 18.18 coins | Sparse tables did not exploit added state well |
| Loop wrapper | 30.09 versus 20.0 masked-Q control | Loop context/intervention can matter materially |
| Linear SARSA(λ) | 12.36, high seed variance | Online traces were not competitive in this representation |
| Tree FQI | 23.52; 26.33 on fresh boards | Useful compact batch baseline |
| History tree FQI | 35.76; 41.45 on fresh 400-step boards | History/progress context relieved important aliasing |
| History-8/stagnation | 43.27 and 50% completion | Best compact navigation branch; longer history was not better |
| Coin DQN | best 29.71 at round 200; 29.30 final | A neural network alone did not beat the engineered tree state |

The 600-round tree extensions were non-monotonic: the balanced choice was
around round 400, not the final round. That checkpoint behavior later repeated
in DDQN.

### 2.2 FQI combat path

The FQI work is valuable as a control and as a record of semantic pitfalls,
but it is not the strongest route for the final tournament agent.

| Run | Coin Heaven | Loot Crate | Decision |
| --- | ---: | ---: | --- |
| Initial combat FQI | -- | 6.54 versus 11.56 untrained | Safe but degraded; reject |
| History/anti-stagnation FQI | 19.70 regression | 19.35 learning-curve result; 16.68 fresh | Representation helped crates but forgot navigation |
| Mixed FQI | 39.14 | 10.41 | Rehearsal restored navigation but diluted crate learning |
| 46-value topology FQI, round 600 | 28.07 | 3.06 | More topology and training hurt |
| Agent 035 corrected D4 FQI | 21.22 | 15.87 / 41.55 crates | Correct, safe, below DDQN |
| Agent 036 robust FQI | 21.64 / 4.2% completion | 12.56 / 34.43 crates | Safe, not promoted |

Agent 034 schema v1 is not a learning result: replay cached the pre-action
vector as the next state, so fitted targets did not describe the actual next
decision. The repaired v2 implementation has no matched learning run. Agent
035 showed that symmetry and balanced fitting could make the compact method
structurally sound; it did not show competitive performance.

### 2.3 The DQN audit changed what counts as evidence

The 18 September audit found three causal defects in the historical neural
runs:

- the Bellman maximization could bootstrap through prohibited actions; the
  target argmax was illegal in roughly 85--98% of inspected decisions;
- replay's `next_state` history could disagree with the state used by the
  following `act()` call (1,028 mismatches in 19,152 inspected pairs);
- the 46-value representation had exact aliases requiring opposite route
  decisions.

An observed prohibited `BOMB` Q-value of about 217 also exceeded the rough
positive-return scale of a whole good episode (about 135.7), making target
inflation visible. Therefore, pre-repair DQN curves are implementation
diagnostics, not evidence for or against vanilla DQN, DDQN, or a feature block.

The corrected contract is: store the feature vector and legal/safe action mask
from the actual following decision; use the online network to select the legal
next action; use the target network to evaluate it; and bootstrap zero at a
terminal transition.

### 2.4 Corrected DDQN progression

The comparable post-repair sequence gives the clearest training evidence.

| Candidate | Main change | Coin Heaven | Loot Crate | Combat conclusion |
| --- | --- | ---: | ---: | --- |
| Repaired 32-value vanilla DQN | Correct target/replay path | 7.04 | 45.90 | Safe solo crate specialist; weak navigation and Classic |
| 46-value DDQN, crate-only | DDQN/topology, same specialist curriculum | 21.16 | 47.55 | Safe; curriculum still narrow |
| 46-value DDQN, mixed | Alternate both solo tasks | **49.06**, 64.6% complete | **46.20**, 114.30 crates | Best balanced 46/52 control |
| 52-value route DDQN, mixed | Exact route costs and time | 45.08, 53.1% | 37.20, 96.97 crates | Fixed a known alias but regressed aggregate and seed stability |
| Agent 024 | Action/bomb consequences | 42.76, 47.9% | 38.90 | Safe, not better |
| Agent 025 | Short-cycle summaries | 48.51, 56.3% | 37.17 | Only one-feature branch worth extending |
| Agent 026 | Coverage summaries | 37.76, 30.2% | 37.13 | Reject |
| Agent 027 | 65 values plus staged, tagged replay | 49.26, 60.4% | 47.04, 116.36 crates | Strong solo balance; combat safety remained weak |
| Agent 028 | 256-wide dueling network | 48.30, 15.6% | 42.81 | Larger/dueling architecture regressed |

The decisive 2x2 comparison was representation (46/52) by curriculum
(Loot-Crate-only/mixed). Switching to mixed training moved the 46-value Coin
Heaven result from 21.16 to 49.06. Switching 46 to 52 values did not produce a
similar gain. Curriculum was the dominant intervention.

Agent 027 then demonstrated the benefit of an explicit staged curriculum and
tagged replay, reaching Classic scores of 10.02, 5.38, and 4.54 against the
peaceful, coin-collector, and rule-based lineups. Its rule-based survival was
only 43.8%; 54 of 96 games ended in death, including 36 self-deaths. Strong
solo safety was therefore not sufficient for adversarial safety.

### 2.5 Opponent-aware and continuation runs

| Candidate | Evidence | Judgment |
| --- | --- | --- |
| Agent 029, 101 values | Classic scores 10.552 / 4.646 / 4.521 against peaceful / collector / rule-based | Opponent-response context helped; retain the 101-value representation |
| Corrected Agent 030 at episode 900 | Three-rule score 3.115, mixed-strong 4.042; rule-based survival 79.2%, suicides 8.3% | Best safe starting point for focused continuation |
| Agent 031, 113 values | Single-rule score 4.792 and 88.5% survival, but three-rule 2.990 and mixed 4.010 | Extra 12 values overfit one pressure; do not promote |
| Agent 032 optimization | Exact 113-value equivalence; 2.05x feature and 3.19x CPU replay/tensor speedups | Keep implementation optimizations; speed does not establish policy quality |
| Agent 032 broad league | Best pooled relevant score 3.203 at episode 1000 versus 3.641 for built-in control | The 24-lineup imported league diluted useful training; reject as default |
| Agent 033 two-member population | Strong results through global episode 600, but each member had only about 300 active learning games and no final episode-1200 evaluation | Interesting, incomplete, not promotable |
| Agent 037 focused continuation | Episode-1200 pooled relevant score 3.943 versus corrected-control 3.579 | Focused complete-lineup continuation improved the target distribution |

Agent 032's broad league was a completed two-seed, 1,200-round comparison, not
a pending pilot. At episode 1,200 the imported-league branch had pooled score
3.031 and the built-in control 3.297; their survival was the same (0.641).
The reasonable inference is that 24 lineups spread the finite combat budget
too thinly. Because roster and distribution changed together, this is a
curriculum result, not proof that imported opponents are intrinsically bad.

Agent 033 alternated two learners. At global episode 600 its members had good
small-suite measurements, including three-rule scores of 4.000 and 4.125 and
one mixed Deep-Killer score of 5.000. But global episode is not per-member
experience, and evaluation stops halfway through the requested run. Treat it
as support for revisiting a small population, not as a winner.

Agent 037 made the cleanest tournament-focused test: retain the corrected
101-value model, use only two relevant complete lineups for 60% of new rounds,
and preserve 20% Coin Heaven plus 20% Loot Crate. Development pooled score
peaked at 4.099 around episode 1,000 and then moved non-monotonically. Frozen
episode 1,200 averaged 3.490 against three rule-based agents and 4.396 in the
mixed built-in lineup, for 3.943 pooled: about 10.2% above its corrected
episode-900 starting control. The seed-0 episode-1,050 package scored 3.5625
against three rule-based agents and 4.78125 in the mixed built-in lineup.

That package did not generalize equally to the strongest imported agents. It
scored 4.156 against a mixed Deep-Killer lineup but only 1.4375 against three
Deep Killers, with no outright first places. This is the clearest reason to
reserve unseen opponent rosters for holdout evaluation.

### 2.6 External references

The imported baselines show that strong compact agents can remain formidable
when their planning and safety priors are well engineered:

- Deep Killer: Coin Heaven 50/50 and Loot Crate 40.50, plus Classic score 7.12
  against one rule-based opponent and 5.22 against three;
- Arbiter: 5.47 against three rule-based opponents with 0.56 kills/game;
- Li SARSA: 4.97 against three rule-based opponents;
- Harvy: 42.97 Loot Crate and 4.16 against three rule-based opponents;
- Overlord: excellent Coin Heaven but only 4.72 Loot Crate.

These agents should be used as frozen training pressure or, preferably, unseen
holdouts. Their architecture labels are not causal explanations: much of their
strength comes from handcrafted planning, bombing, and escape logic.

## 3. Cross-run conclusions

### High-confidence conclusions

- Hard action masking belongs in both behavior and the Bellman target.
- Decision-time state, history, and mask consistency must be tested explicitly.
- The 128--128 ordinary DDQN is a better control than the 256-wide dueling
  ablation.
- Scenario/lineup-tagged replay and retained solo generation are necessary.
- Three independent seeds and checkpoint selection are necessary; one seed or
  the final round is not a reliable conclusion.
- CPU is faster for this model/environment combination.

### Medium-confidence conclusions

- The 101-value Agent 029/030/037 representation is the best current balance.
- A focused two- or three-lineup combat distribution is better than a broad
  24-lineup league at the available training budget.
- Warm-starting from a validated solo/combat checkpoint is more efficient than
  relearning all skills, provided optimizer migration is tested and replay is
  rebuilt under the new distribution.
- D4 augmentation is promising when applied as one random transform per stored
  transition; eight stored copies needlessly consume replay capacity.

### Unresolved questions

- Whether Agent 038's 16 route/reachability values improve generalization once
  curriculum and warm start are controlled.
- Whether a two-member population beats one learner at equal **per-learner
  environment steps**.
- Whether opponent-conditioned prioritized replay improves combat without
  overfitting or starving solo transitions.
- Whether a moderate exploration reset (`0.15--0.20`) is better than Agent
  038's current `0.30` reset. This needs a one-factor pilot.

## 4. Reference DDQN contract

Freeze this implementation while studying the regimen:

| Component | Reference setting |
| --- | --- |
| Input | Agent 029/030/037 101-value vector; a strict superset is a separate ablation |
| Actions | `UP, RIGHT, DOWN, LEFT, WAIT, BOMB` |
| Network | MLP 101--128--128--6, ReLU, ordinary Q head |
| Algorithm | Double DQN with online legal argmax and target-network evaluation |
| Loss | Smooth L1 |
| Optimizer | Adam, learning rate `1e-4` |
| Discount | `0.99` |
| Batch | 128 |
| Replay | 100,000 transitions, tagged by scenario and full ordered/canonical lineup |
| Warm-up | At least 5,000 new transitions before optimization after a fresh replay |
| Update cadence | One optimizer update every 4 environment steps |
| Target sync | Every 1,000 optimizer updates |
| Gradient clip | Global norm 10 |
| Safety | Same legal/time-expanded safe mask for acting and targets |
| Terminal target | No bootstrap |
| Device | CPU, CUDA hidden, one math-library thread per learner |

Any change to one row is an ablation. Do not simultaneously change the feature
set, network, curriculum, replay quotas, and exploration schedule.

## 5. Recommended regimen from scratch

Round numbers are global learner games. A round with several learning agents
must also report games and transitions **per learner**.

| Phase | Rounds | Generated games | Purpose and advance gate |
| --- | ---: | --- | --- |
| Contract smoke | 2--5 | One short game per scenario/lineup type | Check feature shape, action order, mask identity, terminal target, save/load, deterministic eval, and update occurrence |
| Navigation foundation | 1--100 | 70% Coin Heaven, 30% Loot Crate | Learn movement and coin routes; require zero solo invalid actions/self-deaths |
| Bomb/crate foundation | 101--300 | 25% Coin Heaven, 75% Loot Crate | Learn useful bombs and escape while rehearsing navigation |
| Built-in bridge | 301--600 | 50% Classic, 25% Coin Heaven, 25% Loot Crate | Introduce peaceful/collector/rule-based pressure without dropping the solo distribution |
| Focused population | 601--900 | 60% Classic over 2--3 complete lineups, 20% Coin Heaven, 20% Loot Crate | Target tournament behavior; use three-rule and one mixed strong lineup, plus at most one frozen learned-agent lineup |
| Conditional extension | 901--1,100 or 1,200 | Same frozen distribution | Continue only while held-out score improves and all retention gates pass |

Use a seeded shuffled cycle rather than independent random scenario choices so
each short window realizes the intended proportions. Keep the lineup set fixed
for a comparison. A useful focused set is:

1. three `rule_based_agent` opponents;
2. `rule_based_agent + coin_collector_agent + peaceful_agent`;
3. optionally `Deep Killer + Arbiter + frozen Agent 037` for a separate robust
   branch.

Do not put all available imported agents into the training roster. Reserve at
least Deep Killer, Li SARSA, Li Double-Q, Sentinel, Overlord, and Harvy lineups
that were not used by the branch as generalization tests.

### Exploration from scratch

Use `epsilon=1.0` initially, anneal to `0.10` over roughly the first
80,000--100,000 transitions, and then to `0.05` by the start of the focused
population phase. Exploration must choose only legal/safe actions. Log epsilon
against environment transitions, not just rounds, because episode lengths
differ by scenario.

## 6. Recommended warm continuation

Warm continuation is the default when the new model is an exact feature
superset or the only change is the opponent distribution.

1. Verify exact equality of the old feature prefix on a saved corpus.
2. For added features, zero-initialize only the new input columns so the first
   forward pass preserves every old Q-value. Verify this numerically.
3. Copy online and target weights. Copy Adam state only if tensor identity and
   shapes are preserved and a one-step optimizer test passes; otherwise record
   that it was reset.
4. Start a fresh replay buffer. Old replay lacks the new feature/lineup schema
   and can silently corrupt the comparison.
5. Collect at least 5,000 new transitions before the first update.
6. Run 20--50 adaptation rounds split equally across Coin Heaven and Loot
   Crate, then run 250--300 rounds at 60% focused Classic / 20% / 20% solo.
7. Reset epsilon to `0.15--0.20`, annealing to `0.05` over about 60,000 new
   transitions. Treat `0.30` as a separate exploration ablation, not a default.

Train a scratch arm through the short foundation checkpoint when testing a new
feature set. It is a causal/wiring control; it need not consume the entire
production budget once the warm migration has passed its predeclared gates.

## 7. Replay and augmentation regimen

Replay composition is part of the algorithm and must be serialized in the run
config.

- Tag every transition by scenario and complete opponent lineup. Record both
  the scheduled generation share and the effective sampled share.
- Generate fresh games for every retained tag. Sampling weights cannot recover
  tasks whose transitions were never generated or have been evicted.
- For the focused phase, sample 60% Classic, 20% Coin Heaven, and 20% Loot
  Crate, then split the Classic quota evenly across the registered complete
  lineups unless a predeclared weighting is under test.
- Cap escape/emergency oversampling at 10% of a batch. If attack-outcome
  prioritization is added, cap it independently at 10% and require a minimum
  support count before it is active.
- Keep the remaining 80% or more distributional, drawn through the registered
  task/lineup quotas. This limits feedback loops around rare shaped events.
- Apply one uniformly random D4 board symmetry to a sampled or newly stored
  transition. Transform state, action, next state, and next legal/safe mask
  together. Do not store eight copies by default.
- Log tag counts, ages, eviction rates, quota misses, priority ranges, and the
  fraction of batches containing each task.

## 8. Evaluation and checkpoint selection

### Board split

Boards 32000--32007 and 33000--33007 have already influenced design and
checkpoint decisions. They are development boards, not blind evidence.

- **Development screening:** boards 34000--34003, all four seats.
- **Development confirmation:** boards 34000--34015, all four seats.
- **Final blind audit:** boards 35000--35015, all four seats, opened once after
  the global episode and candidate definition are frozen.

If these ranges have already been inspected elsewhere, allocate new seeded
ranges before the next run and record them before training.

### Evaluation cadence

- Save a checkpoint every 50 rounds or 20,000 environment transitions,
  whichever comes first.
- Run the four-board screen every 50 rounds.
- Run the 16-board development suite at phase boundaries and for shortlisted
  checkpoints only.
- Select one **global episode across all three training seeds first**. Then
  select a package seed using the same frozen objective. Still report all three
  seeds at the selected episode.
- Compare candidates on paired board/seat games and report confidence
  intervals. Training reward is never the selection metric.

### Mandatory solo gates

- zero invalid actions and zero self-deaths in Coin Heaven and Loot Crate;
- each solo mean no more than 5% below the branch's frozen reference;
- no material loss of completion rate, crates, or progress diagnostics hidden
  by the mean coin score;
- 100% solo survival under the standard suite.

### Tournament objective

After the solo gates, rank candidates lexicographically by:

1. lower mean rank across the registered development lineups;
2. higher outright-first rate, then joint-first rate;
3. higher official score;
4. fewer self-deaths and invalid actions.

Require improvement in at least two of three training seeds and at least a 5%
pooled gain over the frozen reference, with no registered-lineup collapse,
before promotion. Also report coins, crates, bombs, kills, survival, episode
length, and direct head-to-head W/T/L when a reference occupies another seat.

## 9. Monitoring and early stopping

Log, by seed and replay tag:

- loss, gradient norm, update count, target-sync count, and epsilon;
- online and target Q median/p90/p99/max;
- fraction of unconstrained argmax actions excluded by the mask and the Q-gap
  to the best legal action;
- action frequencies, especially `WAIT` and `BOMB`;
- reward-component totals and terminal-event counts;
- replay population, age, sampled share, and quota misses;
- coins, crates, kills, rank, wins, survival, suicides, invalid actions;
- repeated states, maximum no-progress stretch, and progress events;
- feature, action, optimizer, and total decision latency.

Stop a branch and diagnose it when any of these occurs:

- NaN/Inf, a checkpoint load mismatch, mask inconsistency, or target bootstrap
  at a terminal state;
- Q p99 remains above roughly 150 and continues rising, or values above 200
  persist. These are warning thresholds derived from the reward scale, not a
  theorem;
- solo completion drops by more than 20 percentage points or either solo mean
  falls more than 10% for two consecutive evaluations;
- suicides or invalid actions worsen for two consecutive evaluations;
- a required replay tag is absent or persistently misses its quota;
- the full development objective has not improved for three checkpoint
  evaluations (about 150 rounds). Select the earliest statistically tied best
  checkpoint instead of automatically extending the run.

## 10. Compute and reproducibility

Agent 032's paired end-to-end diagnostic measured 14.35 ms/step on CPU and
16.40 ms/step on an RTX 2080 Ti: GPU was 14.3% slower after normalizing by
steps. Use CPU for the present vector MLP and environment loop. Hide CUDA and
set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to one for each learner. Reconsider GPU only for a
genuinely batch-heavy spatial model, after a paired end-to-end timing test.

Every run must preserve:

- source commit and dirty-tree/source hashes;
- exact command, job ID, hostname, thread/device environment, and wall clock;
- config, seeds, feature/checkpoint schema, action order, curriculum schedule,
  replay quotas, and opponent checkpoint hashes;
- per-seed learning curves and checkpoints;
- frozen manifests, per-game records, summaries, and selection decision;
- failed/stopped status and reason, without overwriting the artifacts.

Use a new output directory for every protocol or code change. Resume only an
identical run whose optimizer, replay, counters, RNG state, curriculum position,
and source provenance can be restored.

## 11. How to interpret the active Agent 038 experiment

The two active jobs are:

- `agent038-warm-population-1200-v2-20260921`;
- `agent038-scratch-population-1200-v2-20260921`.

They use a 117-value representation: the 101-value Agent 037 prefix plus 16
route/reachability values, consistent one-transform D4 replay, and a 14-slot
population schedule. The warm branch zero-pads Agent 037 seed-0 episode 1050;
the scratch branch is the control. Both reset exploration to 0.30. Their first
300 rounds are solo foundation; after that they generate 70% population and
15% of each solo task.

At the audit cutoff both v2 jobs were healthy and running. Their `episode=0`
evaluation confirms the warm migration at behavior level: warm seeds averaged
48.125--49.625 Coin Heaven coins and 45.125--47.250 Loot Crate coins, while
untrained scratch seeds were near-random.

The two v2 jobs were subsequently stopped by the user before population-phase
completion. No final Agent 038 population result exists; the round-150 values
below remain the last valid diagnostic checkpoint.

The three-seed round-150 evaluation also completed:

| Branch | Coin Heaven | Completion | Loot Crate | Crates | Mean over five Classic probes | Classic survival | Classic invalid/game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm | 47.71 | 10.4% | **44.04** | **114.67** | 1.442 | **79.6%** | **0.213** |
| Scratch | **49.85** | **89.6%** | 36.00 | 93.10 | **1.517** | 73.3% | 1.113 |

Both branches retained 100% survival, zero self-deaths, and zero invalid
actions in the two solo scenarios. Scratch learned the navigation phase
quickly; warm retained much stronger crate behavior and cleaner Classic
actions. These are foundation diagnostics only: neither branch had entered
population training, and averaging the five different Classic probes is not a
promotion objective. This checkpoint is not evidence that the new features or
population curriculum work.

Interpret future checkpoints as follows:

- round 150: solo foundation and retention only;
- round 300: pre-population foundation comparison;
- round 450: first checkpoint containing meaningful population exposure;
- rounds 600 onward: curriculum evidence, subject to per-seed and holdout
  gates;
- no promotion before a complete three-seed evaluation on unseen opponent
  rosters and a blind board range.

Do not alter or relabel the active run. If it succeeds, compare warm versus
scratch at equal rounds and transitions, and compare Agent 038 against frozen
Agent 037 on the exact same boards, seats, and lineups. Because Agent 038
changes features, symmetry, exploration, initialization, and curriculum, its
full result will be a system comparison; causal attribution requires the
one-factor follow-ups described above.

## 12. Agent 037 regimen test

On 21 September 2026 I ran the first bounded test of this regimen on the
validated Agent 037 checkpoint. The test intentionally used one seed and a
small evaluation slice; it is a contract/wiring test, not a learning claim.

### Contract tests

From the runtime source tree `/export/home/salitanl/projects/ml-project/bomberman_rl`,
with CUDA hidden and one thread per math library:

```text
python -m unittest -v test_agent037_tournament_fast_ddqn.py
```

Result: **6 tests passed**. This covered exact Agent 029 feature equivalence,
the 101-value preallocated replay contract, DDQN optimizer consumption, and
the inherited Agent 032 optimized-feature/replay parity tests.

### Continuation smoke

The `jobctl` run was:

```text
agent037-regimen-smoke-20260921
```

It initialized seed 0 from:

```text
/export/scratch/salitanl/agent037_tournament_fast_1200_20260921_v4/seed_0/checkpoints/episode_1050.pkl
```

and continued episodes 1051--1070 with `league-combat`, both Agent 037
lineups, two evaluation boards (`34000`, `34001`), and seats 0 and 1. The
complete artifact is:

```text
/export/scratch/salitanl/agent037_regimen_smoke_20260921/
```

The job exited with code 0 and wrote episode-1060 and episode-1070
checkpoints. `rounds.jsonl` confirms the intended generated cycle:

```text
classic, classic, classic, coin-heaven, loot-crate
```

and confirms both complete lineup tags at 30% each, with 20% Coin Heaven and
20% Loot Crate replay weights. The Agent 037 escape-replay overlay adds a
separate `combat_escape` tag at effective weight 20%; this is expected for the
inherited Agent 030 safety continuation and should remain explicit in future
comparisons.

The effective-config comparison found only intentional smoke differences in
seed count, round target, evaluation boards/seats/workers, output path,
checkpoint, and board stride, plus the current runtime runner's source hash.
Agent 037's feature, model, safety, replay, and inherited training hashes were
identical to the episode-1050 source run. Since later Agent 038/039 work changed
the shared runner source hash, this smoke is an integration test and not a
strict matched learning continuation.

| Evaluation | Episode 1050 | Episode 1070 | Change |
| --- | ---: | ---: | ---: |
| Coin Heaven coins | 48.50 | 48.50 | 0.00 |
| Loot Crate coins | 48.50 | 48.00 | -0.50 |
| Loot Crate crates | 123.50 | 122.75 | -0.75 |
| Three rule-based score | 2.00 | 3.25 | +1.25 |
| Mixed built-in score | 6.00 | 4.50 | -1.50 |

Solo survival remained 100% with zero invalid actions and zero self-deaths at
both checkpoints. Classic survival was unchanged at 50% against three
rule-based agents and 75% in the mixed lineup; the final mixed checkpoint had
0.25 invalid actions/game in this eight-game evaluation slice. The opposing
lineup delta is therefore noisy and mixed: it does not justify promotion or a
conclusion about learning quality. It does confirm that checkpoint loading,
training updates, replay tagging, schedule generation, evaluation, and
checkpoint saving all work on Agent 037.

The smoke's root `learning_curve.json` progress path was intentionally separate
from the runner's per-seed curve; the authoritative curve is
`training/seed_0/learning_curve.json`. Future job launches should point
`--progress-file` at the per-seed or aggregate file that the runner actually
writes.

## 13. Full Agent 037 regimen test in progress

The substantive test requested after the smoke run is registered as:

```text
agent037-regimen-continuation-1200-1500-20260921
```

It runs from the clean Agent 037 runtime worktree at commit `c81b579`,
initializes all three seeds from the previous Agent 037 episode-1200
checkpoints, and trains episodes 1201--1500 with:

- `league-combat`;
- two complete lineups: three rule-based agents and the mixed
  rule-based/collector/peaceful lineup;
- 60% Classic, 20% Coin Heaven, 20% Loot Crate generation/replay weighting;
- CPU-only execution with one thread per math library;
- checkpoint evaluation every 50 episodes on boards 34000--34007 and seats
  0--3, including both solo tasks and both Classic lineups.

The run output is:

```text
/export/scratch/salitanl/agent037_regimen_continuation_1200_1500_20260921/
```

The training runner evaluates the episode-1200 initializer and every scheduled
checkpoint on the same development split, giving a paired baseline and
continuation comparison. The separate queued evaluation was canceled before
starting because it would have duplicated those games. The final result will
be appended here with per-seed metrics, pooled deltas, solo retention, and the
promotion decision.

The durable manifest used for the intended paired comparison is
[`agent037_regimen_continuation_manifest.json`](agent037_regimen_continuation_manifest.json).

## 14. Recommended next decision sequence

1. Let the registered Agent 038 jobs finish or meet a predeclared stop rule;
   do not tune from round-150 noise.
2. Evaluate rounds 300, 450, 600, 750, 900, 1050, and 1200 with the same
   manifests, keeping the final blind boards sealed.
3. If warm Agent 038 wins without forgetting, run one confirmation branch
   changing only the route/reachability block. If it does not, return to the
   101-value Agent 037 control rather than adding capacity.
4. For the next new agent, use the regimen in Sections 4--10 and stop at the
   earliest held-out optimum. Budget extensions only after a measured plateau
   analysis, not because 1,200 is the nominal maximum.

## Source notes and artifacts

Primary interpretations and contracts are in:

- [`dqn_training_audit_20260918.md`](dqn_training_audit_20260918.md)
- [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md)
- [`offensive_feature_comparison.md`](offensive_feature_comparison.md)
- [`agent032_imported_league_pilot.md`](agent032_imported_league_pilot.md)
- [`Agent_037_Tournament_Fast_DDQN_Design.md`](Agent_037_Tournament_Fast_DDQN_Design.md)
- [`Agent_038_Symmetric_Population_DDQN_Design.md`](Agent_038_Symmetric_Population_DDQN_Design.md)
- [`tournament_training_protocol.md`](tournament_training_protocol.md)
- [`imported_stage2_results.md`](imported_stage2_results.md)
- [`imported_combat_results.md`](imported_combat_results.md)

The raw Agent 032--038 scratch paths and job IDs are intentionally retained in
their family notes and `jobctl` records; they should be copied to stable project
storage before scratch cleanup.
