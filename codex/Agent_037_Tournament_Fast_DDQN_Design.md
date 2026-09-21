# Agent 037: Tournament-Focused Fast DDQN Design

**Status:** design specification, not yet implemented or trained  
**Date:** 21 September 2026  
**Proposed agent directory:** `Agent_037_tournament_fast_ddqn_agent`

## 1. Decision

Agent 037 should be a **101-input, tournament-fine-tuned Double DQN**. It
should combine:

1. the corrected Agent 030 callback/training path;
2. Agent 029's 101-value opponent-response representation;
3. Agent 032's faster feature-search and preallocated replay techniques;
4. a continuation from the three corrected Agent 030 episode-900
   checkpoints; and
5. training directly against the two four-player lineups used by the frozen
   tournament evaluation, while retaining Coin Heaven and Loot Crate rounds.

This is the highest-confidence route to a stronger tournament agent under the
remaining time budget. It does not assume that a new algorithm is better just
because it is newer. Agent 036 is the cleanest compact-FQI implementation, but
its measured solo result is far below the strongest DDQN line and it has no
Classic tournament result. Agent 037 therefore returns to the experimentally
strongest branch.

The intended improvement is primarily **better training data**, not a larger
network or more features. The corrected 101-input control already has the best
pooled result over the two tournament-like lineups among Agents 029--031, but
it was not trained on both of those lineups. Agent 037 closes that mismatch.

There is no honest way to guarantee that Agent 037 will beat every previous
agent before evaluation. It earns promotion only if it passes the gates in
Section 11.

## 2. Evidence behind the choice

### 2.1 Best relevant measured results

The completed matched frozen comparison produced these mean Classic scores:

| Candidate | Three rule-based | Mixed strong | Equal-weight pooled score |
| --- | ---: | ---: | ---: |
| Agent 029, episode 600 | 2.781 | **4.229** | 3.505 |
| Corrected 101-input control, episode 900 | **3.115** | 4.042 | **3.579** |
| Agent 031, 113 inputs, episode 900 | 2.990 | 4.010 | 3.500 |

The corrected 101-input control is therefore the strongest single starting
point for the final evaluation lineups. It also improved rule-based survival
from 58.3% to 79.2% and reduced suicides from 22.9% to 8.3% relative to Agent
029. The 113-input branch reached 88.5% survival in the single-rule matchup,
but failed to improve either multi-opponent lineup and slightly regressed the
solo tasks. The extra 12 offensive inputs should not be included in Agent 037.

### 2.2 Components that did and did not work

| Finding | Agent 037 consequence |
| --- | --- |
| Agent 027/029/030's 128--128 DDQN is the strongest balanced family | Preserve the network and DDQN update exactly. |
| Agent 028's larger dueling network regressed | Do not enlarge or complicate the network. |
| Agent 029's 36 action-conditioned opponent-response inputs improved combat awareness | Keep the complete 101-input feature contract. |
| Agent 031's additional 12 offensive inputs helped one matchup but not broad competition | Omit them. |
| Correcting Agent 030's callback dispatch materially improved safety | Start from the corrected implementation and checkpoints. |
| Agent 032 cut feature-only time by 51.3% and replay/tensor preparation by 68.6% on CPU | Port those implementation optimizations to the 101-input contract. |
| The RTX 2080 Ti was slower than CPU in the paired end-to-end diagnostic | Train and evaluate on CPU. |
| Agent 036's held-out solo scores were 21.64 Coin Heaven and 12.56 Loot Crate | Do not use compact FQI as the tournament base. |
| Agent 033 population work has no completed promotion result | Do not make unvalidated population replay a dependency of the deadline path. |

## 3. Exact Agent 037 definition

### 3.1 Observation

The observation is exactly Agent 029/030's 101-value vector:

- the 65-value Agent 027 state, topology, history, and short-cycle prefix;
- six opponent-response values for each of the six candidate actions:
  immediate legality, armed-opponent presence, one-step destination contest,
  worst reachable survival time, worst safe-frontier size, and whether an
  immediately placeable opponent bomb creates a new trap.

The action order remains:

```text
UP, RIGHT, DOWN, LEFT, WAIT, BOMB
```

Agent 037 must reproduce Agent 030's full 101-value feature vector **exactly**
on the equivalence corpus. It should implement the shared opponent envelope,
cached legality checks, local hot-loop bindings, and allocation-light search
from Agent 032, but it must return after the 101-value adversarial prefix. It
must not compute Agent 031's discarded 12 offensive values and then slice
them off, because that would retain their runtime cost.

### 3.2 Network and learner

Keep the corrected control's learner unchanged:

```text
101 inputs -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(6)
algorithm              Double DQN
batch size             128
discount               0.99
learning rate          1e-4
replay capacity         100,000
train frequency         every 4 environment transitions
target update           every 1,000 optimizer steps
gradient clipping       norm 10
loss                    smooth L1
```

Preserve the existing legal/safety candidate mask and keep adversarial risk as
information for the learner. Do not add a broad hard opponent-risk veto in
the first Agent 037 run: Agent 027's audit showed that invalid moves were not
the main source of combat deaths, and an aggressive veto could prevent kills
or produce excessive waiting.

### 3.3 Replay

Use Agent 032's fixed NumPy circular arrays and reusable CPU tensor batches,
adapted to `state_dim=101`. Preserve:

- complete-lineup scenario tags;
- corrected callback dispatch;
- the existing combat-escape trajectory tag and bounded 20% sample quota;
- next-action masks and terminal handling; and
- deterministic replay RNG per training seed.

The escape quota is retained because it is part of the safer corrected
control. It must not be described as an isolated causal improvement; the
historical callback defect prevents that conclusion.

### 3.4 Runtime contract

The tournament package must contain only code available from the agent
directory plus normal installed dependencies and its selected checkpoint. It
must:

- have no runtime import from another custom `agent_code` directory;
- run on one CPU thread;
- use no multiprocessing during official play;
- stay below 8 GB memory;
- decide within 0.5 seconds per step on the official-class CPU;
- use deterministic evaluation actions except for explicitly seeded tie
  breaking; and
- load one model once in `setup()` rather than touching training artifacts at
  action time.

The engineering target is p95 action latency below 0.25 seconds and maximum
observed latency below 0.45 seconds, leaving margin around the official
0.5-second limit.

## 4. Initialization

Initialize one Agent 037 run per training seed from the corrected 101-input
episode-900 checkpoints:

```text
/export/scratch/salitanl/bomberman_feature_variants_20260920/
  offensive_feature_matched_v1/control101/seed_0/checkpoints/episode_0900.pkl
/export/scratch/salitanl/bomberman_feature_variants_20260920/
  offensive_feature_matched_v1/control101/seed_1/checkpoints/episode_0900.pkl
/export/scratch/salitanl/bomberman_feature_variants_20260920/
  offensive_feature_matched_v1/control101/seed_2/checkpoints/episode_0900.pkl
```

The migration must preserve policy weights, target weights, Adam state,
optimizer-step count, environment counters, and initial Q-values. Replay is
not serialized, so every continuation starts with a fresh buffer and the
normal 5,000-transition warmup. Retained solo generation is required; replay
weights alone cannot preserve absent experience in a fresh buffer.

Do not reset exploration in the primary run. Continue from the checkpoint's
recorded exploration state, which avoids destabilizing a competent policy.
If a later controlled pilot tests an exploration reset, that is a separate
experiment and must not be silently folded into the primary result.

## 5. Tournament-aligned continuation curriculum

Continue episodes 901--1200 with the existing `league-combat` scheduler and
only these two complete lineups:

```text
three_rule_based = rule_based_agent, rule_based_agent, rule_based_agent
mixed_strong     = rule_based_agent, coin_collector_agent, peaceful_agent
```

Each five-round cycle generates:

- three Classic rounds, reproducibly balanced across the two lineups;
- one Coin Heaven round; and
- one Loot Crate round.

Replay targets are:

- 30% three-rule-based Classic;
- 30% mixed-strong Classic;
- 20% Coin Heaven; and
- 20% Loot Crate,

before the existing bounded combat-escape sub-allocation is applied. This
schedule gives 60% of new rounds and replay weight to four-player combat while
continuing to generate the solo transitions needed to prevent catastrophic
forgetting.

This is preferable to the historical `tournament-combat` tail for Agent 037.
After episode 600 that schedule generates only three-rule-based games and
would not train the current weakest side of the corrected control: the
mixed-strong matchup.

Training boards remain separate per seed. Development boards are 33000--33007.
The frozen boards 32000--32007 must never be used for training, hyperparameter
selection, checkpoint selection, or debugging.

## 6. Fast implementation plan

Create the following independent directory without changing Agents 029--032:

```text
src/agent_code/Agent_037_tournament_fast_ddqn_agent/
    __init__.py
    callbacks.py
    config.py
    features.py
    model.py
    replay.py
    safety.py
    train.py
    test_contracts.py
    tournament_checkpoint.pt
```

Implementation sources:

- `callbacks.py`: a self-contained copy of the corrected action/callback path,
  with `MODEL_PATH` fixed to the local `tournament_checkpoint.pt`;
- `config.py` and `model.py`: local copies of the small DDQN definition,
  constants, and checkpoint helpers;
- `features.py`: Agent 032's optimized 101-value prefix only;
- `replay.py`: Agent 032's preallocated replay with a 101-value state width;
- `train.py`: Agent 032's allocation-light optimizer and corrected Agent 030
  terminal/callback behavior;
- `safety.py`: unchanged audited Agent 030/029 safety behavior; and
- `test_contracts.py`: package-local feature, checkpoint, latency, and import
  contract checks.

The implementation may be derived from the audited modules, but the final
Agent 037 files must use relative imports within Agent 037. In particular,
`features.py`, `callbacks.py`, `model.py`, and `safety.py` must not import
Agents 025, 027, 029, 030, 031, 032, or `combat_dqn_agent`. Those sibling
directories will not be present when the official framework plugs in the
submitted agent directory. Training-only helpers should also be local so the
archived project remains reproducible.

Also update `run_combat_training.py` provenance hashing so the Agent 037
configuration records every imported source file. No framework modification
may be necessary for the schedule because `league-combat`, repeated
`--classic-lineup`, and repeated `--league-eval-lineup` already support this
design.

The implementation order is deliberately short:

1. create the local 101-input optimized modules;
2. prove feature, mask, Q-value, and checkpoint equivalence;
3. run one real-game training smoke round;
4. benchmark CPU inference and one-step training;
5. run the three-seed continuation;
6. select on development boards; and
7. execute the frozen suite once.

## 7. Required validation before long training

### 7.1 Exactness tests

Tests must cover empty boards, crates, multiple opponents, armed opponents,
existing bombs, chain explosions, lingering flames, dead ends, contested
destinations, `WAIT`, and `BOMB`.

Require all of the following:

- Agent 037 feature shape is exactly `(101,)`;
- every feature is exactly equal to Agent 030 on the fixed corpus;
- legal and safety masks are identical;
- migrated policy and target Q-values agree within floating-point tolerance;
- greedy actions agree, including deterministic tie handling;
- one seeded optimizer update agrees within floating-point tolerance when fed
  the same sampled transition indices; and
- callback tests prove that ordinary, terminal, and combat-escape transitions
  enter the intended replay tags.

### 7.2 Performance tests

Benchmark on CPU with one math-library thread. Compare Agent 037 with Agent
030 and Agent 032 on the same stored states and transitions. Record:

- feature calls per second;
- mean and p95 `act()` latency;
- replay sample plus tensor-construction time;
- optimizer-update time; and
- end-to-end seconds per training round.

Agent 037 should be no slower than Agent 030 end to end and should approach
Agent 032's optimized input-pipeline speed. A microbenchmark speedup alone is
not sufficient evidence.

### 7.3 Smoke run

From seed 0's episode-900 checkpoint, train through episode 902 on fresh
boards with `--eval-every 1`. Require episode-901 and episode-902 checkpoints,
non-empty replay, finite Q-values, correct lineup tags, and a successful real
evaluation game. A two-round smoke run cannot reach the 5,000-transition
warmup, so it should not claim a real-game optimizer update. Exercise the
optimizer separately with a seeded replay buffer filled past warmup, then let
the full run prove the natural warmup-to-update transition.

## 8. Primary training command

After validation, the intended full run is:

```bash
AGENT=Agent_037_tournament_fast_ddqn_agent
OUT=/export/scratch/salitanl/agent037_tournament_fast_ddqn_1200

CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
bomberman_rl_exp/jobctl start \
  --id agent037-tournament-fast-train \
  --progress-file "$OUT/learning_curve.json" -- \
  python3 bomberman_rl_exp/src/run_combat_training.py \
    --agent "$AGENT" \
    --curriculum league-combat \
    --rounds 1200 \
    --seeds 0 1 2 \
    --parallel-seeds \
    --initial-checkpoints \
      /export/scratch/salitanl/bomberman_feature_variants_20260920/offensive_feature_matched_v1/control101/seed_0/checkpoints/episode_0900.pkl \
      /export/scratch/salitanl/bomberman_feature_variants_20260920/offensive_feature_matched_v1/control101/seed_1/checkpoints/episode_0900.pkl \
      /export/scratch/salitanl/bomberman_feature_variants_20260920/offensive_feature_matched_v1/control101/seed_2/checkpoints/episode_0900.pkl \
    --classic-lineup rule_based_agent rule_based_agent rule_based_agent \
    --classic-lineup rule_based_agent coin_collector_agent peaceful_agent \
    --league-eval-lineup rule_based_agent rule_based_agent rule_based_agent \
    --league-eval-lineup rule_based_agent coin_collector_agent peaceful_agent \
    --eval-every 50 \
    --eval-seeds 33000 33001 33002 33003 33004 33005 33006 33007 \
    --eval-seats 0 1 2 3 \
    --eval-workers 8 \
    --eval-scenario-workers 4 \
    --diagnostics \
    --output "$OUT"
```

If host contention makes three concurrent learners slower, reduce evaluation
workers first. Do not move the small network to GPU: the paired measurement
showed no end-to-end benefit.

## 9. Checkpoint selection without test leakage

Candidate episodes are 900, 950, 1000, 1050, 1100, 1150, and 1200. The
episode-900 checkpoints are the unchanged starting controls.

Use only development boards 33000--33007 to choose a global episode. For each
episode, aggregate all three training seeds, all four seats, and both lineups.
The primary development objective is:

```text
J_dev = 0.5 * mean_score(three_rule_based)
      + 0.5 * mean_score(mixed_strong)
```

Apply the safety and solo gates before comparing `J_dev`. Prefer the earliest
episode whose score is statistically indistinguishable from the maximum; this
limits unnecessary drift and gives the simpler training-duration claim.

On separate development boards, also audit the historical peaceful,
coin-collector, and single-rule lineups. They are secondary robustness checks,
not terms to optimize repeatedly. A large collapse in one of them is evidence
that the exact two-lineup continuation has overfit and blocks promotion until
it is understood.

After choosing one global episode, choose the single tournament-package seed
using the same development objective and gates. Record that choice before the
frozen evaluation. Do not choose a seed or episode after seeing frozen-board
results.

For scientific reporting, still evaluate all three checkpoints at the chosen
global episode and report both per-seed and pooled results. The final submitted
agent directory contains only the preselected single checkpoint. Copy it to
`Agent_037_tournament_fast_ddqn_agent/tournament_checkpoint.pt`; the `.pkl`
and `.pt` suffixes do not alter the existing PyTorch checkpoint payload. Then
load that fixed local path in a clean evaluation process before packaging.

## 10. Frozen evaluation

Use the existing Classic tournament protocol unchanged:

- board seeds 32000--32007;
- action seed 0;
- seats 0, 1, 2, and 3;
- 400-step Classic games;
- three rule-based opponents;
- mixed rule-based/collector/peaceful opponents;
- 32 games per lineup per checkpoint;
- three independently trained checkpoints, hence 96 games per lineup.

Also rerun the matched Coin Heaven and Loot Crate suite. Report mean score,
win rate, mean rank, coins, crates, bombs, kills, survival, suicides, invalid
actions, and episode length. Include bootstrap confidence intervals and
paired board/seat differences against Agent 029 and the corrected episode-900
control.

The frozen suite is a final audit, not a tuning loop. If Agent 037 fails, keep
the result and submit the best previously validated checkpoint rather than
modifying the design based on frozen boards.

## 11. Promotion gates

Agent 037 is promoted only if all mandatory gates pass.

### Gate A: correctness and deployability

- no feature/checkpoint-equivalence failure;
- no NaN or infinite values;
- no missing runtime import outside the final agent package;
- official-style one-thread inference stays within the time and memory limits;
- final zip loads and completes games in a clean copy of the framework.

### Gate B: solo preservation

- zero invalid actions and zero self-deaths in the solo audit;
- Coin Heaven and Loot Crate means are each no more than 5% below the
  corrected episode-900 source on the matched protocol;
- no material completion-rate or crate-destruction regression.

### Gate C: tournament improvement

Compare against both Agent 029 episode 600 and the corrected 101-input
episode-900 control. Promotion requires:

- higher equal-weight pooled score over the two frozen tournament lineups;
- improvement in at least two of the three training seeds;
- no material regression in either individual lineup;
- no increase in pooled suicide rate; and
- preferably either a paired 95% confidence interval above zero or at least a
  5% pooled-score gain when the available sample is too small for a decisive
  interval.

For orientation, the old frozen pooled means are 3.505 for Agent 029 and 3.579
for the corrected control. A convincing Agent 037 target is at least 3.76,
which is roughly 5% above the stronger control, while preserving both lineups.
This is a target, not a promised outcome.

## 12. Stop/go policy

Use the following decision sequence:

```text
exactness fails
  -> fix implementation; do not train

speed is not better than Agent 030
  -> profile once; retain correctness and continue only if deadline permits

development tournament score rises and solo gates pass
  -> freeze episode and seed choice, then run the frozen suite

frozen score beats both controls and all gates pass
  -> package Agent 037

frozen score fails or solo safety regresses
  -> do not promote; package the best previously validated DDQN checkpoint
```

## 13. Deliberately excluded changes

Do not combine these with the first Agent 037 run:

- the 113-input offensive extension;
- a larger, dueling, distributional, recurrent, or attention network;
- broad reward reshaping;
- prioritized replay beyond the already used bounded escape quota;
- population replay or unvalidated imported-opponent training;
- self-play league management;
- GPU training for this small model;
- a new hard adversarial safety veto; or
- tuning against boards 32000--32007.

Any one of these may become a later Agent 038 experiment. Combining them now
would make failure attribution impossible and would spend time on changes with
less evidence than the curriculum mismatch.

## 14. Expected outcome

Agent 037 is expected to train much faster than a new 1,000-round run because
it reuses three competent episode-900 checkpoints and adds only 300
tournament-aligned rounds. Its per-step feature and replay path should also be
faster than Agent 030 because it carries Agent 032's implementation
optimizations while omitting Agent 031's 12 costly and unpromoted features.

The key scientific hypothesis is:

> A corrected, fast 101-input DDQN already has sufficient opponent-response
> information; balanced replay and direct experience against both final
> four-player lineups will improve tournament score more reliably than adding
> model capacity or another feature family.

This design gives that hypothesis a clean test, preserves the strongest known
policy, and leaves enough time for a valid three-seed evaluation and a safe
fallback before submission.
