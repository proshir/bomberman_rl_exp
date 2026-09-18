# Bomberman RL Algorithm Knowledge Base

Durable map of the algorithms, safety mechanisms, representations, training
methods, and experimental ideas in this repository. This is a knowledge base,
not a claim that every variant is submission-ready.

Status labels: **Implemented** means present in source; **Validated** means
covered by a focused test or documented experiment; **Pilot** means evaluated
under a development protocol; **Rejected/ablation** means not the preferred
default; **Proposal** means not yet implemented.

## 1. Environment and constraints

The game has one to four agents acting on a discrete board. The actions are
`UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, and `BOMB`. Observations include the
board (`-1` wall, `1` crate, `0` free), bombs and timers, explosion map, visible
coins, the player's score/bomb availability/position, and living opponents.

Bombs normally detonate after four steps, have power three, stop at stone walls,
destroy crates and agents, and remain dangerous for one extra step. Coins are
worth one point; opponent kills are worth five. Episodes normally last 400
steps. Official inference has a 0.5-second per-step budget on one CPU thread,
and an overrun affects the next step. This makes runtime and deterministic
safety semantics part of the algorithm.

The task ladder is coin navigation, crates plus survival, hunting, and finally
competition. The agent interface is `callbacks.py` (`setup`, `act`) plus
training-only `train.py` callbacks.

## 2. Shared ideas

### Legal-action masking

Movement is blocked by walls, crates, bombs, and other agents. `WAIT` is legal;
`BOMB` requires an available bomb. Masked agents restrict both exploration and
greedy selection to legal actions. Learned Bellman targets also bootstrap only
over legal/safe next actions:

```text
target = reward + gamma * max(Q(next_state, allowed_action))
```

### BFS navigation

Breadth-first search on free tiles is used for nearest coins, crate-approach
tiles, opponents, bomb destinations, and escape routes. BFS is preferred to
Manhattan distance when walls or crates make routes nontrivial.

### Time-expanded safety search

Combat safety searches `(position, time, has_left_origin)` against a future
danger schedule. It checks immediate legality, explosion on arrival, future
danger, and whether a newly placed bomb leaves an escape route. If no action
survives the complete schedule, the fallback chooses legal actions with the
longest predicted survival time.

The older combat solver searches until the latest known danger has passed. The
Harvy solver uses a bounded horizon and optimized flat buffers to meet the
action budget. Safety rejects actions; it is not itself a reward/value model.

### Reward shaping

Common training signals are coin `+1`, kill `+5`, small crate/reveal/survival/
escape rewards, death and invalid-action penalties, useless-bomb penalties, and
a per-step cost. These are training-only proxies for the official objective and
must be checked against held-out boards. The repository documents potential-
based shaping and novelty bonuses as research ideas, not established fixes.

### Reproducibility

Agents use seeded NumPy generators for tie-breaking and exploration. Experiments
record seeds, board ranges, seats, checkpoint/source hashes, effective settings,
per-game results, and timing. Paired comparisons reuse boards and corners;
uncertainty is usually a board-clustered bootstrap conditional on checkpoints.

## 3. Scripted baselines

- **Random agent:** random movement; low-performance opponent and smoke baseline.
- **Peaceful agent:** random movement and no bombs; easy hunting opponent.
- **Fail agent:** raises from `act`; failure-path diagnostic.
- **Rule-based agent:** hand-written danger, target, and bomb policy; curriculum and competition benchmark. Its simplistic safety motivated the improved time-aware solver.
- **Coin collector agent:** BFS to the closest reachable coin, with randomized tie order; strong navigation reference but no learned combat.
- **TPL agent:** framework example showing transition storage and training callbacks.

## 4. Coin-navigation algorithms

The compact navigation state uses categorical local obstacles, direction/distance
to the nearest coin, and remaining coins. The `distance` representation was
introduced because coarse features merged states with different useful actions.

### Plain tabular Q-learning — Implemented

A dictionary maps feature tuples to five action values. It uses epsilon-greedy
exploration and:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

The plain version can learn from illegal moves and is a baseline for masked
variants.

### Masked tabular Q-learning — Implemented/Pilot

Selection, exploration, and bootstrap targets are restricted to legal movement
actions plus `WAIT`. The loop-breaking development variant is a runtime wrapper,
not a change to the table.

### Linear SARSA(lambda) — Implemented/Pilot

Categorical state features become a one-hot vector with a bias; each action has
`Q(s,a) = w_a dot x(s)`. Semi-gradient SARSA with accumulating traces uses:

```text
delta = r + gamma * Q(s',a') - Q(s,a)
e <- gamma * lambda * e
e[action] <- e[action] + x(s)
w <- w + alpha * delta * e
```

Next actions are epsilon-greedy among legal actions. A frozen-policy loop
variant is evaluated separately.

### Tree fitted-Q iteration — Implemented/Pilot

Transitions are buffered, then one `DecisionTreeRegressor` is fit per action to
Bellman targets after each round. Typical settings: discount `0.95`, buffer
30,000, five fit iterations, depth eight, minimum leaf five. Trees generalize
better than a table across related features but still suffer representation
aliasing.

### History-aware tree FQI — Implemented/Validated

Adds previous action, visits to the current position in an eight-position
history, and bucketed steps since coin progress. History resets on a new round
or visible-coin change. Training reconstructs next-state history from cached
pre-action context so Bellman features match action-time features.

Optional revisit penalties and history lengths 16/32 were tested; they are not
the preferred defaults because they can penalize valid backtracking.

### Evaluation-time loop breaker — Implemented/Pilot

The frozen-policy wrapper stores the last eight positions since coin progress.
If the current position appears at least three times, it replaces the proposed
action with a uniformly sampled legal movement other than the proposal and
`WAIT`, then clears history. If no alternative exists, it keeps the proposal.

This reduced repeated states on development boards but is a hybrid intervention,
not learned policy improvement, and has not established fresh-board
generalization by itself.

### DQN — Implemented/Pilot

The coin DQN is a two-hidden-layer MLP (`features -> 128 ReLU -> 128 ReLU -> 5
Q-values`). Training uses a replay buffer (100,000), 2,000-transition warmup,
Adam at `1e-3`, batch size 128, Huber loss, a target network updated every 1,000
updates, legal-action masking, gradient clipping, and linearly decayed epsilon.
The online network selects the next action and the target network evaluates it
(Double-DQN-like target). Training may use CUDA; official inference is CPU.

## 5. Bomb-aware combat FQI

### Combat representation

The base combat feature vector adds legal/safe-action flags, local danger,
bomb availability, coin/crate/opponent directions and distances, blast yields,
escape distance, and crate/coin/opponent counts to navigation information.

### Combat policy

At each step it computes safe actions, falls back to longest-survival legal
actions if necessary, predicts six action values with action-specific trees, and
chooses the best candidate. Bombs are only considered useful if they hit a
crate/opponent and have a found escape route. There is no multi-step tactical
plan search in this family.

### Combat reward

The controlled combat variants use coin `+1`, kill `+5`, crate `+0.2`, reveal
`+0.2`, survive `+0.5`, escape improvement `+0.1`, death `-5`, invalid `-1`,
useless bomb `-0.1`, and step cost `-0.01`.

### History/anti-stagnation combat FQI — Implemented/Pilot

`combat_fqi_history_antistag_agent` appends previous action, capped current-tile
visit count, and bucketed steps since meaningful progress. Progress means a
change in visible coins, crate count, opponent count, or own score; it resets
history and the stagnation clock.

The important distinction is that it does not hard-code an escape action. The
trees learn how temporal context should change Q-values. It improved the crate
pilot with zero invalid actions/self-deaths but lost substantial Coin Heaven
navigation performance, showing that combat learning can cause catastrophic
forgetting without a mixed curriculum or navigation distillation.

### Combat vanilla DQN — Implemented/software validated

`combat_dqn_agent` is the deliberately minimal neural comparison. It uses the
existing 32-feature history-aware combat representation, reward/events, and
safety-constrained action selection, then replaces tree refitting with a
128-128 ReLU MLP, replay buffer, policy/target networks, epsilon-greedy
exploration, random mini-batch updates, and periodic target copies. Its target
is the standard vanilla form: the target network both selects and evaluates the
maximum next action. Double-DQN and other extensions are intentionally deferred.
A short fresh-process smoke run passed; no substantial performance pilot has
been registered yet.

The first combat DQN pilot reached 18.48 mean loot-crate coins at round 300,
slightly below matched history FQI at 19.35. A fresh 600-round run peaked at
9.96 around round 200 and declined to 6.28 at round 600. The earlier runner
did not seed PyTorch model initialization; this has been corrected before the
topology-feature neural ablation. That 600-round DQN ablation peaked at 16.53
mean loot-crate coins at round 100 and ended at 2.86, below the 32-feature
control; the 14-value block remains a rejected experimental representation, not
a change to the default.

## 6. Harvy / Arbiter advanced hybrid

This is not purely PPO, DQN, or hand-written planning. The structural split is:

- the neural policy ranks movement actions and provides the fallback;
- the safety solver supplies survival constraints;
- exact search owns bomb-plan decisions;
- the value head evaluates search leaves when enabled.

### Features and network

The engineered representation has 98 scalars covering directional context, BFS
targets, danger, escape, bomb yield, trap opportunities, opponent state, map
openness, score margins, and endgame signals.

The older model is a three-layer 256-unit MLP with policy logits `pi` and a
scalar `V`. The current Harvy/Arbiter-NG model uses a 12-channel 17x17 board
tensor plus scalar features, a CNN trunk, fusion, policy/value heads, and an
auxiliary margin head. Heads are zero-initialized so the safety skeleton is not
given arbitrary initial preferences.

### S0 policy fallback

1. Compute valid and safe actions.
2. Run the policy, optionally under eight rotations/reflections and average the mapped outputs.
3. If the current tile is about to be lethal, restrict to safe escape moves.
4. Otherwise rank valid moves by policy logits.
5. Apply only small survival tie-breaks such as loop/bomb-repeat handling.
6. Permit `BOMB` only through a safety gate.

The heuristic does not receive a large independent vote over root actions; an
earlier Q/heuristic blend double-counted information and hurt performance.

### P1 bounded exact search

Search generates legal one-step moves, optional coin-run plans, and bomb plans
of the form `BFS path to bomb tile -> BOMB`. Bomb candidates are ranked by crate
yield, nearby coins, opponent coverage, and distance, then pass an escape
certificate.

Each plan is rolled forward in a lightweight exact simulator until bomb effects
settle. Opponents use a seeded avoid-lethal-if-possible rollout policy. Common
random numbers can give all candidates the same opponent draws. The score is
approximately:

```text
crate reward + coin/reveal reward + 5 * certified kills
- 8 * own death + score-margin change + optional leaf V
```

Kills are credited only when the escape solver proves that an opponent cannot
survive; rollout luck is ignored. The best bomb must exceed the best movement
plan by a margin (default about `0.6`); otherwise search returns no action and
S0 selects a movement action.

### Tactical and optional arms

Inference-only proven-kill overlays, coin-take overlays, solo endgame bomb
approach, opponent plant guards, committed bomb approach, and backtrack
penalties exist as experiment switches. Rejected arms should not become
defaults without a new gate.

### Training pipeline

The documented pipeline is teacher demonstrations, feature/cache extraction,
policy cross-entropy, value regression to margin-to-go, symmetry augmentation,
and optional KL-anchored REINFORCE fine-tuning. The fine-tune uses exact event
rewards, round returns, and a KL penalty to the behavior-cloning prior. Search-
selected bomb steps are not ordinary policy steps because search owns them. The
current Arbiter training hook describes future online policy iteration; the
advanced checkpoint is primarily a pretrained/fine-tuned inference model.

## 7. Anti-loop and stagnation research

Observed navigation failures included terminal `WAIT` loops and two-step
reversals. Diagnostics found feature aliasing: distinct map locations and route
choices can share the same compact tuple. More samples, longer history alone,
or an unverified distance reward are not guaranteed fixes.

Tested ideas include previous-action features, recent-position counts,
steps-since-progress, explicit cycle detection, random legal alternatives,
separate intervention RNG, revisit penalties, history lengths 8/16/32, and
nearest-coin distance probes.

**Proposal:** episodic novelty FQI. Count visits to an identity containing board
geometry, player position, and visible coins; add a capped count feature and,
optionally, a small `beta / sqrt(N(identity))` training bonus. Keep FQI, masks,
trees, budget, and evaluation fixed. Measure both completion/coins and cycles;
novelty must not punish necessary backtracking.

## 8. Curriculum and experiment rules

Use the ladder: Coin Heaven -> loot-crate -> peaceful opponent -> coin collector
opponent -> rule-based/mixed competition. Freeze checkpoints and evaluate on
held-out boards, seats, and seeds; do not advance because training reward alone
increased.

Report coins/score, completion or wins, crates, kills, bombs, deaths, survival,
invalid actions, repeated states/interventions, and latency. Separate smoke
checks from performance evidence. Preserve rejected variants and ablations.

### Finite-horizon caution

A 100-step run can mean a genuinely terminal 100-step task or a truncated sample
of the official 400-step task. The former needs a zero future target at the
boundary; the latter retains continuation value. Remaining time should be in
the state for a truly finite-horizon objective. The repository's cutoff audit
found that these choices must not be silently mixed.

### Symmetry

Rotations and reflections can augment data, average predictions, and test
equivariance. Directional feature segments and movement actions must be
permuted consistently; `WAIT` and `BOMB` are invariant.

### Provenance and runtime

Record source/checkpoint hashes, effective environment settings, board and seed
protocol, device, and timing distributions. Any optimized safety or simulator
implementation needs parity tests against the engine. A timeout fallback should
return a safe/default action rather than an uncomputed action.

## 9. Comparative summary

| Family | Main strength | Main limitation |
| --- | --- | --- |
| BFS coin collector | Reliable navigation | No learned combat |
| Plain Q-table | Simple/interpretable | State explosion and illegal-action learning |
| Masked Q-table | Cheap legal control | Limited representation/generalization |
| Linear SARSA(lambda) | Online temporal credit | Linear approximation |
| Tree FQI | Compact nonlinear CPU model | Aliasing and fitted-value instability |
| History FQI | Learns temporal anti-stagnation context | More state/sample complexity; forgetting |
| DQN | Replay, target network, smooth approximation | Training sensitivity; limited combat representation |
| Combat FQI | Direct bomb-aware six-action values plus safety | No multi-step tactical search |
| Combat anti-stagnation | Combat safety plus temporal context | Navigation regression in tested pilot |
| Harvy/Arbiter | Exact tactical planning and certified kills | Complexity, latency, search assumptions, limited online learning |

## 10. Reusable blueprint

```text
raw game state
  ├─> compact/board features ─> learned policy/value
  ├─> legality + time-expanded safety mask
  └─> BFS targets + optional exact simulator
                         └─> candidate plans and certificates
                                      └─> action arbitration
```

Use this order of complexity:

1. engine-compatible legality;
2. BFS navigation;
3. masked learner;
4. time-aware escape safety;
5. combat features and bomb usefulness;
6. temporal history when loops are measured;
7. exact multi-step search only when latency and ablations justify it;
8. held-out gates after every complexity increase.

The key separation is: safety answers “can it survive?”, learning answers
“which safe action is valuable?”, search answers “does a committed plan beat
alternatives?”, and history answers “is the current observation hiding a loop?”.

## 11. Source map

- Environment/interface: `bomberman_rl/environment.py`, `agents.py`, `settings.py`, `events.py`
- Scripted baselines: `agent_code/coin_collector_agent`, `rule_based_agent`, `random_agent`, `peaceful_agent`
- Tabular Q: `agent_code/q_table_agent`, `q_table_masked_agent`, `q_table_loop_agent`
- Linear SARSA: `agent_code/linear_sarsa_agent`, `linear_sarsa_loop_agent`
- Tree FQI: `agent_code/tree_fqi_agent`, `tree_fqi_history_agent`, `tree_fqi_loop_agent`
- DQN: `agent_code/dqn_coin_agent`
- Combat FQI: `agent_code/combat_fqi_agent`
- Combat anti-stagnation: `agent_code/combat_fqi_history_antistag_agent`
- Harvy/Arbiter: `agent_code/harvy`
- Experiment runners: `run_training.py`, `run_combat_training.py`, and `run_*_confirmation.py`
- Existing durable notes: `bomberman_rl_exp/codex/`

## 12. Caveats

1. Development-board gains are not automatically generalization gains.
2. A safety mask can produce survival without good navigation.
3. A loop intervention can improve scores without improving learned weights.
4. Training reward can rise while frozen evaluation falls.
5. More history can reduce aliasing but increase sample complexity and punish backtracking.
6. More search can improve tactics but violate the action budget or overfit rollout assumptions.
7. Safety changes require parity tests against the engine.
8. Keep rejected variants and measured failures; they explain the retained design.
