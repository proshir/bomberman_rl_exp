# Agent 043 — novelty, attributed progress, and credit assignment

Status: implemented, contract-tested, and promoted through the solo gate on
21 September 2026. Two matched three-seed 900-round league branches are active.

## Motivation

Agent 042 reduced Coin-Heaven stagnation strongly around rounds 150–200, but
its round-300 checkpoint returned to 109.8 repeated states and a 116.6-step
no-progress stretch. The three deferred fixes most directly aimed at that
failure were: a bounded novelty intervention, better attribution of progress,
and improved temporal credit assignment. Agent 043 implements those three in
isolation from the safety mask and retains Agent 042 as a warm-startable
control.

## Implemented changes

### 1. Action-aligned novelty and bounded loop guard

Agent 043 widens Agent 042's 140-input vector to 146 inputs with a six-action
`novelty` block. A legal movement receives `1/(1+visit_count(destination))`;
WAIT and BOMB remain neutral, and unsafe actions are still excluded by the
existing safety candidate mask. D4 replay transforms permute this block with
the action label.

When all of the following hold, the callback applies one bounded intervention:

- at least 24 steps have elapsed since attributed progress;
- the current tile has appeared at least twice in the rolling history; and
- at least eight steps have elapsed since the previous intervention.

The selected action is the most novel safe candidate, breaking ties by the
DDQN value. It is not a blanket reversal ban, does not permit unsafe actions,
and records `novelty_interventions` for later analysis. This is intentionally
a diagnostic hybrid guard: the learned novelty block remains available to the
network, while the bounded fallback prevents a proven late loop from
consuming the entire horizon.

### 2. Attributed progress clock

Training callbacks now advance the no-progress clock only on this agent's
`COIN_COLLECTED`, `COIN_FOUND`, `CRATE_DESTROYED`, and `KILLED_OPPONENT`
events. Evaluation, where events are unavailable, uses own score increases and
coin-set changes only when both consecutive observations contain no opponents
(the solo case). A coin disappearing during combat is therefore not falsely
credited to this agent. Event counts are retained in
`attributed_progress_events` for diagnostics, while rolling position/action
history remains persistent through unrelated global changes.

### 3. Reward and temporal credit assignment

The original Agent 042 reward is retained, with two controlled additions:

- three-step replay returns (`N_STEP_RETURN=3`) use
  `r_t + gamma*r_(t+1) + gamma^2*r_(t+2)` and a `gamma^3` DDQN bootstrap;
- bounded potential-based combat shaping uses scale `0.05` and
  `gamma*Phi(next)-Phi(current)`, where `Phi` is the clipped negative
  nearest-opponent BFS distance. Solo states have zero combat potential, so
  this shaping does not add a hidden solo target.

N-step transitions are transformed with one shared D4 transform at commit time,
which keeps the first state, action, future state, and future mask consistent.
Terminal partial returns are flushed without bootstrapping. The safety mask,
DDQN target masking, replay tags, and combat/solo exploration counters remain
unchanged.

## Compatibility and validation

Agent 040, Agent 041, and Agent 042 checkpoints are widened with zero-padded
first-layer columns; old Q-values are preserved when novelty columns are zero.
The runner hashes Agent 043 and its Agent 042 dependencies for provenance.

`test_agent043_novelty_credit_ddqn.py` passes six tests covering the 146-input
contract, novelty action transforms, replay/setup width, checkpoint migration,
event attribution, and the exact three-step return. Agent 042's seven-test
contract suite also passes. A real-engine smoke run completed three 80-step
solo rounds plus frozen evaluation without callback errors, invalid actions, or
checkpoint failures. Smoke artifacts:

`/export/scratch/salitanl/agent043_smoke_20260921_v2/`

## Measured 300-round Coin Heaven gate

The three-seed run at
`/export/scratch/salitanl/agent043_solo_300_20260921/` used 400-step Coin
Heaven rounds and frozen evaluations every 50 rounds on boards 34000--34003,
all four seats. At episode 300, each seed collected all 50 coins in every
evaluation game and survived every game.

| Training seed | Mean coins | Completion | Repeated states | Max no-progress steps | Mean steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50.000 | 100% | 23.375 | 23.813 | 151.500 |
| 1 | 50.000 | 100% | 13.188 | 18.688 | 141.688 |
| 2 | 50.000 | 100% | 15.750 | 21.438 | 143.750 |
| **Mean** | **50.000** | **100%** | **17.438** | **21.313** | **145.646** |

Against Agent 042's matched episode-300 result, completion improved from
60.42% to 100%, repeated states fell from 109.79 to 17.44, and the maximum
no-progress stretch fell from 116.60 to 21.31 steps. This passes the stated
solo anti-stagnation gate; it does not by itself prove combat quality.

## Approved staged league run

Job `agent043-staged-league-900-20260921` uses the dedicated
`agent043-staged-league` schedule:

- episodes 1--100: 100% Coin Heaven;
- episodes 101--300: 50% Coin Heaven and 50% Loot Crate;
- episodes 301--900: 100% complete four-player Classic games, split equally
  between three rule-based opponents and
  `imp_alii_arbiter` + `imp_li_deep_killer` + `rule_based_agent`.

The three training seeds run independently. Frozen evaluations occur every
150 rounds on boards 34000--34007, all four seats, both solo tasks, and both
Classic lineups. Evaluation uses eight game workers and eight scenario
workers; CUDA is hidden and math libraries are limited to one thread per
process. Imported opponents and their checkpoints are copied into a clean
scratch runtime and recorded by checksum. Artifacts are under
`/export/scratch/salitanl/agent043_staged_league_900_20260921/`.

The combat phase deliberately stops generating and sampling solo transitions,
matching the approved strict stage progression. Solo frozen evaluations remain
in the checkpoint gate so any navigation or crate forgetting is visible when
selecting among episodes 300, 450, 600, 750, and 900.

### Matched memory-retention branch

Job `agent043-staged-league-memory-retention-900-20260921` is a one-factor
comparison against the strict branch. Episodes 1--300, training seeds, boards,
lineups, evaluator, and worker counts are identical. Only episodes 301--900
change:

| Branch | Classic | Coin Heaven | Loot Crate |
| --- | ---: | ---: | ---: |
| strict stage | 100% | 0% | 0% |
| memory retention | 80% | 10% | 10% |

The retention scheduler deterministically shuffles every ten-round block per
seed. Each block contains eight complete Classic games, one Coin Heaven game,
and one Loot Crate game. The eight Classic games are split equally between the
three-rule-based and Arbiter/Deepkiller/rule-based lineups. Replay sampling
uses the same 80/10/10 weights, so rehearsal affects both generated experience
and DDQN updates. Artifacts are under
`/export/scratch/salitanl/agent043_staged_league_memory_retention_900_20260921/`.

The paired result should answer whether 20% solo rehearsal prevents
catastrophic forgetting without sacrificing too much combat exposure. Compare
the same episode checkpoints and all three seeds rather than only episode 900.

### Dataset warm-start continuation support

`eval_suite/run_agent043_staged_league.py` also accepts one
`episode_0650.pkl` checkpoint per training seed through
`--initial-checkpoints`. The launcher validates both the checkpoint count and
filename, forwards the resolved paths to `run_combat_training.py`, and records
them in `provenance.json`. `--no-eval` can suppress intermediate frozen
evaluations when the continuation is being used only to produce a later
checkpoint. Neither option changes the default fresh staged-league protocol.

## Still open after this run

Target persistence/enemy threat prediction and a bomb-specific offensive gate
remain separate experiments. The staged league must be judged on combat
score, rank/wins, kills, survival, suicides, invalid actions, and solo
retention across all three seeds. The final checkpoint is not automatically
preferred if an earlier held-out checkpoint is stronger.
