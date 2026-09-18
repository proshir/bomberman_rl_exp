# Combat FQI agent

Sahand was here.

## Status

The first bomb-aware agent implementation is complete and has passed short
software smoke checks. It has **not** had a real training run, model comparison,
or fresh-board confirmation. There is no final `trees.pkl` in the agent folder
yet, so this is not a submission candidate at this stage.

## Design

`src/agent_code/combat_fqi_agent/` is self-contained for tournament inference:

- `callbacks.py` loads six fitted action-value trees and chooses only from safe
  actions;
- `features.py` describes nearby obstacles and danger, visible coins, crate
  approaches, opponents, bomb value, and escape distance;
- `safety.py` calculates blast tiles, future danger, immediate legality, and a
  position/time escape search;
- `train.py` stores transitions and fits one regression tree for each of the
  actions `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, and `BOMB`.

The safety layer is intentionally deterministic. The trees decide which safe
action is valuable; the mask prevents movement into known obstacles and rejects
bombs for which it cannot find an escape. A bomb is offered to the learner only
when its blast includes a crate or opponent. If no action survives the complete
known danger schedule, the callback chooses among the actions with the longest
predicted survival.

The blast calculation mirrors this version of the framework. Stone walls stop
blast rays, while crates are destroyed but do not stop a ray in
`items.Bomb.get_blast_coords`. This differs from the wording in the brief and is
covered by a test so the choice is visible.

The reward uses official coins and kills, with small rewards for crates, revealed
coins, survival, and escaping immediate danger. Death, invalid actions, useless
bombs, and each elapsed step are penalized. These values are initial engineering
choices, not tuned hyperparameters or supported scientific findings.

## Training runner

`src/run_combat_training.py` supports `loot-crate` and `classic`, zero to three
fixed supplied opponents, independent training seeds, 400-step rounds by
default, frozen checkpoint evaluation, source hashes, and per-round records.
It records score, coins, crates, kills, suicides, bombs, invalid actions,
survival, reward, buffer size, and tree size.

Example crate pilot command, to be agreed before running:

```bash
python src/run_combat_training.py \
  --scenario loot-crate \
  --seeds 0 1 2 \
  --rounds 300 \
  --max-steps 400 \
  --eval-every 100 \
  --eval-seeds 30000 30001 30002 30003 30004 30005 30006 30007 \
  --output experiments/combat_fqi_crate_pilot
```

Later opponent stages use the same command with `--scenario classic` and, for
example, `--opponents peaceful_agent`.

## Validation completed

`python -m unittest -v test.py test_combat_safety.py` passes 10 tests: the two
existing framework checks and eight focused safety checks. The safety tests
cover walls, crates, current explosions, trapped and escapable bombs, occupied
tiles, bomb availability, and rejection of useless bombs.

Two deliberately tiny runs checked the complete training and evaluation path:

- two 20-step `loot-crate` rounds placed five bombs, destroyed nine crates,
  made no invalid actions, and survived both rounds;
- one 30-step `classic` round against `peaceful_agent` placed four bombs,
  destroyed five crates, made no invalid actions, and survived.

They also exercised tree fitting, checkpoint serialization, fresh-process model
loading, and frozen evaluation. Scores and coins were zero. These runs are not
retained as project experiments and must not be reported as performance results.

## Known limitations and next decision

The safety search treats future crate locations conservatively and does not
predict future opponent actions or chain reactions (the supplied framework does
not implement bomb chain detonation). The crate target feature searches free
tiles adjacent to crates rather than every possible distance-three bombing
position. Opponent targeting is still coarse.

Before substantial training, agree on the first fixed crate-pilot protocol and
its success criteria. Suggested primary outcomes are hidden coins collected and
self-death rate, with crates, bombs, invalid actions, survival, and runtime as
diagnostics. After that result, decide whether to revise safety/features or move
to the peaceful-opponent stage.

## Approved crate pilot protocol

The user approved the first substantial pilot on 18 September 2026. The fixed
question is whether the initial combat-FQI implementation learns to reveal and
collect hidden coins while retaining the safety layer's smoke-tested behaviour.

- Scenario: `loot-crate`, without opponents.
- Independent training seeds: 0, 1, and 2.
- Training budget: 300 rounds of at most 400 steps per seed.
- Frozen evaluations: rounds 0, 100, 200, and 300.
- Evaluation boards: 30000--30007, unused by training.
- Evaluation action seed: 0; all four starting corners.
- Primary outcome: mean coins per frozen-policy game at round 300, alongside
  change from the untrained round-0 policy.
- Safety gates: no invalid actions and no more than 10% self-deaths in the
  round-300 frozen evaluation.
- Diagnostics: crates, bombs, survival, training reward, interactions, tree
  size, and individual-run results.
- Stopping rule: complete all 300 rounds for all three seeds unless a software
  failure invalidates the run. Do not extend the run after observing results.

This is an exploratory pilot, not a confirmation comparison. A useful advance
is provisionally defined as at least five additional mean coins over the
untrained policy while passing both safety gates. Passing that gate would
support a separate fresh-board check before moving to opponent training; it
would not by itself establish a final agent.

## Crate pilot result

The fixed pilot completed for all three seeds. The untrained frozen policy
averaged 11.5625 coins on the eight evaluation boards. Round-300 means were:

| Training seed | Round-0 mean | Round-300 mean | Round-300 crates | Round-300 bombs |
|---:|---:|---:|---:|---:|
| 0 | 11.5625 | 3.34375 | 13.75 | 5.78125 |
| 1 | 11.5625 | 6.40625 | 22.4375 | 9.34375 |
| 2 | 11.5625 | 9.87500 | 32.03125 | 12.40625 |
| **mean** | **11.5625** | **6.54167** | **22.73958** | **9.17708** |

The learned policy therefore lost 5.02083 mean coins relative to the matched
untrained policy and did not meet the provisional improvement gate. The safety
gates did pass: all 96 round-300 evaluation games had zero invalid actions and
all agents survived. No opponents were present, so kills were necessarily zero.

This is evidence that the initial tree/reward setup is safe but learns a poorer
crate policy than its exploratory starting policy. The result does not justify
moving to opponent training or selecting this model for submission. The full
raw records, checkpoints, configurations, and learning curves are under
`experiments/combat_fqi_crate_pilot/`; the empty path-normalization launch that
preceded the corrected run is preserved separately as
`experiments/combat_fqi_crate_pilot_failed_launch/`.

The overview figure is generated by `src/plot_combat_training.py` and saved as
`experiments/combat_fqi_crate_pilot/plots/training_overview.png`. It shows why
training reward and per-round coin collection cannot substitute for frozen
policy evaluation: the former rises while the latter declines.
