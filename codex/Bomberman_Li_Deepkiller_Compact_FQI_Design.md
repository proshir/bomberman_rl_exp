# Agent 034: an independent compact FQI agent, informed by the Li review

Date: 20 September 2026  
Status: revised implementation proposal; Agent 034 is reserved, but this edit creates no agent, checkpoint, or training result.

## 1. Recommended direction

Build `Agent_034_compact_route_fqi_agent` from the team's existing route,
history, safety, and fitted-Q work. Its research question is whether small
fitted value models can learn balanced navigation and crate behaviour from
fewer independently defined inputs. Li's agent is an attributed external
reference, not the implementation template or a source of policy labels.

The proposed agent has three responsibilities:

1. **Board analysis and safety** compute legal movement, bomb timing, route distances, and surviving continuations under the project's engine model.
2. **A compact state representation** passes separate route, bomb, and short-history quantities to the learner.
3. **Six fitted action-value models**, initially small regression trees trained with fitted Q iteration (FQI), learn which allowed action is most valuable.

The implementation path is explicit:

| Order | Deliverable | Purpose |
| --- | --- | --- |
| 1 | A `route52` FQI reference using the existing 52-input feature contract | Establish correct masks, history, fitted targets, and serialization with known route information. |
| 2 | A `compact23` FQI candidate using the exact projection in Section 7 | Test removal of information with the same learner, reward, safety, and training protocol. |
| 3 | Frozen solo evaluation and evidence that training contributes beyond the shield | Select on navigation and crates together; diagnose failures before adding features. |
| 4 | Retained-data hunting and Classic experiments after the solo gate | Test opponent information and competition without losing the earlier skills. |

These are proposed configurations of one new agent, not four agent numbers.
The 52-input reference is a comparison bridge, not a claim that its broader
input is best for trees. The compact candidate's removed inputs are not
assumed redundant. A software-valid reference is sufficient to start the
compact comparison; the reference need not win a benchmark first.

Tree FQI is a justified next baseline, not a proven replacement for the strongest DDQN. The proposed compact state is also a hypothesis, not a validated sufficient representation of Bomberman.

### 1.1 Originality and the project brief

The [brief][project-brief] forbids copying an existing solution in whole or in
part and forbids a feature that deterministically returns the best action.
It also explicitly suggests pathfinding, lifesaving information, and board
symmetries. Those general methods are not evidence of copying Li. This review
does not establish that Li invented a unique method, or that using numeric
features automatically makes a design acceptable.

| Design element | Agent 034 decision |
| --- | --- |
| BFS distances, reachability, bomb timing, history, fitted Q iteration | Define from the team's own code and game requirements; record provenance. |
| Li's combined coin/crate/enemy/escape utility, its action ranking, `target`, or `KILL!` recommendation | Exclude from features, action selection, training labels, and reward shaping, including renamed or numerically encoded equivalents. |
| Li's symbolic `dead` encoding and safety implementation | Do not transplant. Independently computed danger and feasibility are useful information; the word `dead` itself is not the underlying issue. |
| Li's table, training code, checkpoints, or planner-generated imitation targets | Reference/testing artifacts only; do not use to initialize or train Agent 034. |
| Rotations/reflections and escape-continuation measurements | Later independent ablations; use the project's solver and eight tested symmetries. |

Credit inspiration in the report, but do not treat attribution as permission
to copy a prohibited solution. Changing SARSA to FQI, renaming labels, or
rewriting the same full action selector would not address the concern.
Conversely, a coin-direction feature is not automatically forbidden: the
brief expressly allows it. The relevant question is whether the combined
feature computation and masks already choose the strategic action.

The safety layer may leave only one surviving option in a dangerous state.
Record how often this occurs. When several actions are permitted, their
strategic ordering must come from learned Q-values. Masks must not discard
safe actions merely because Li's utility would prefer another one. Section 11
includes an untrained-value control and decision diagnostics to measure what
the learner contributes; those diagnostics are evidence, not an instructor's
approval of the design.

## 2. Evidence and attribution

This revision uses the local project files. It distinguishes three evidence levels:

- **Source findings:** Li's local imported files, and the team's route, history, and FQI implementations inspected for this revision.
- **Historical isolated checks:** the synthetic-board checks recorded by the original draft. They were not rerun during this documentation edit and establish only their stated behaviours.
- **Recorded project results:** findings in [Agent 034 design insights][project-note] and its experiment notes. This edit does not rerun their training or evaluation.

Li's repository is [Li-Jesse-Jiaze/MLE_project_bomberman][li-repo]. The original
draft reviewed uploaded files without a commit SHA. Their three recorded
source hashes match the current local imports. GitHub links below are
attribution references; the local files and hashes identify the reviewed
version. Local `callbacks.py`, `symmetry.py`, and `q_table.json` are now
available, so the earlier blanket statements that they were missing no
longer describe this workspace.

The [design insights][project-note] remain the evidence summary. This document
specifies the implementation sequence, resolves the earlier ambiguous compact
contract, and records the independent-design boundary. All new settings below
are proposals until implemented and logged. No Agent 033 work is changed.

## 3. What Li's Deepkiller actually does

### 3.1 Learning algorithm

Despite its name, the supplied `deep_learning_killer` uses a dictionary-backed Q-table, not a neural network. Each state key has six values for `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, and `BOMB`. Missing keys receive six zeros. See [table.py, lines 4–16][li-table].

The update in [train.py, lines 36–49][li-learn] has the sampled on-policy form:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
```

However, `a'` is selected by calling `choose_action()` inside `learn()`. The
local `callbacks.py` confirms a fresh epsilon-greedy sample with random
tie-breaking, not the next action actually executed in the environment.
Thus "SARSA-style sampled policy backup" is more precise than claiming an
exact observed SARSA transition. This distinction does not motivate copying
its learner or feature pipeline.

Verified training settings are alpha 0.1, gamma 0.6, initial epsilon 1.0, epsilon minimum 0.1, and round-wise decay 0.9995. The table is saved every 100 rounds. Reloading the Q-table does not restore the exploration schedule: setup initializes epsilon again. The decay condition can also take epsilon slightly below its nominal minimum. See [training setup][li-setup] and [round completion][li-end].

The schedule takes roughly 4,605 decays to reach epsilon 0.1. This is not
evidence of the actual training duration. The files do not establish original
training rounds, wall time, hardware, or the claimed tournament placing. The
project's import review records a 3,090-state table and separate solo/Classic
evaluations; those are reported in [the design insights][project-note], not
new measurements here.

### 3.2 State representation and planning

The feature extractor reads the full board internally, but returns six symbolic entries:

```text
[UP tile, RIGHT tile, DOWN tile, LEFT tile, WAIT tile, BOMB status]
```

The entries can contain labels such as `free`, `block`, `dead`, `enemy`, `coin`, `target`, `True`, `False`, and `KILL!`. Labels are overwritten during extraction, so they do not retain separate channels for legality, safety, and strategic desirability. See [final feature construction][li-final].

The planner performs substantial work before returning those labels:

- Builds a blast-impact matrix and a compressed danger map.
- Searches for surviving continuations for each possible first movement, including waiting.
- Computes route-distance estimates.
- Scores access to coins, opponents, and crate-bombing positions.
- Tests hypothetical bombing safety.
- Detects whether a proposed bomb appears to remove an opponent's escape possibilities.

The principal strategic scoring weights are:

| Component | Outside mapped danger | Inside mapped danger |
| --- | ---: | ---: |
| Coins | 300 | 300 |
| Main enemy | 30 | 0 |
| Other-enemy map | -5 | -300 |
| Crates | 1 | 1 |
| Escape count | 0.1 | 10 |

The main enemy is chosen by Manhattan distance. The enemy map also includes that enemy, so its contributions are combined. The best-scoring movement is converted to `target`; suitable waiting positions can become a bombing target. See [target scoring][li-target] and [enemy selection][li-enemy].

This explains its compact apparent state: substantial hand-written strategic
selection happens before learning. Agent 034 will not reproduce that combined
utility or the resulting recommendation labels.

### 3.3 Killing and rewards

`is_chance_to_kill()` compares escape possibilities around hypothetical bomb placement and can assign `KILL!`. This is a model-based trap prediction, not proof of an unavoidable kill. The learner still decides whether to choose `BOMB`. See [trap detection][li-kill].

The training reward mapping includes:

| Event | Reward |
| --- | ---: |
| Coin collected | +10 |
| Opponent actually killed | +50 |
| Action on a `target` label | +50 |
| Bombing a target | additional +50 |
| Bombing with an adjacent `enemy` label | +20 |
| Bombing on a `KILL!` prediction | +500 |
| Self-kill | -300 |
| Got killed | -100 |
| Action on a `dead` label | -100 |
| Ordinary movement | -1 |
| Waiting or bomb dropped | -5 |
| Invalid action | -20 |

These are rewards assigned by the training code, not necessarily official game scores. Events can stack: bombing a `target` can receive both target bonuses. Predicted trap quality receives much more immediate reward than an actual kill. That can reinforce detector mistakes. See [custom-event construction][li-events] and [reward mapping][li-rewards].

### 3.4 Symmetry

Training applies four rotations and three flip configurations, producing
twelve update calls per transition. A square has eight unique dihedral
transformations; conventional rotations and flips therefore duplicate some
transformations in that loop. The original upload review lacked `symmetry.py`;
it is now present locally. Agent 034 does not adopt the twelve-loop scheme.
A later symmetry ablation must enumerate and test eight unique transforms,
including actions, history, and masks. See [symmetry update loop][li-symmetry].

## 4. Source findings that change the improvement priorities

### 4.1 Recomputed features can describe a different state

Target ties are resolved with `np.random.choice()`. Meanwhile, training recomputes both old and new features. The same raw board can therefore receive different state keys on different calls. A transition can be updated under a representation different from the one used to select its action; shaping based on `target` can also change.

An isolated check extracted features sixteen times from the same empty standard board and obtained four distinct symbolic states. This is a verified behaviour of the uploaded extractor. See [random tie selection][li-ties] and [feature recomputation][li-events].

**Design response:** store the exact decision-time feature vector, history
snapshot, and mask. Feature computation is deterministic; exploration and
Q-value tie-breaking belong to the seeded policy. Agent 034 has no randomly
selected target label.

### 4.2 The danger map loses later overlapping explosions

`calculate_danger_map()` retains a single maximum urgency value at each tile. The search then interprets that scalar as one dangerous time interval. Multiple explosions at different times cannot all be represented. See [danger map][li-danger] and [safety search][li-safe].

An isolated example used bombs at `(1,3)` with timer 0 and `(3,1)` with timer 3. Their perpendicular blast lines overlap at `(3,3)` without the two bombs lying in one another's blast line. Using the code's own interval convention and explosion duration 2:

| Future step | Union of both dangerous intervals | Scalar danger-map test |
| --- | --- | --- |
| 1–2 | Dangerous | Dangerous |
| 3 | Clear | Clear |
| 4–5 | Dangerous | Clear |

**Design response:** represent occupancy and danger over time, such as `danger[t,x,y]`, rather than one urgency number per tile. Reuse the project's audited simulator and match the actual engine's update order.

### 4.3 Safety labels can be overwritten

When all safe continuation counts are zero, `look_for_target()` returns -1. The final extractor then labels the waiting position `target`. An isolated trapped-state check produced:

```text
safe counts: [0, 0, 0, 0, 0]
features:    [block, block, block, block, target, dead]
```

This can erase a danger label and influence shaping. The local callback
selects from table values without a separate safety mask, but this synthetic
example alone does not establish the saved table's chosen action. See
[target fallback][li-final].

**Design response:** keep legal-action masks, predicted safety, and strategic values separate. An emergency fallback must never be represented as guaranteed safe merely because it was selected.

### 4.4 Time-aware search already exists, but can be improved

`find_safe_position()` already queues `(x,y,steps,first_step)` and permits revisiting positions at different times. The separate `BFS()` also permits revisits before `max_wait_steps`; its position-only visited restriction is activated later. The earlier blanket claim that it always prunes all later arrivals was inaccurate. See [safety search][li-safe] and [route BFS][li-bfs].

The useful improvement is a consistent time-dependent model and efficient deduplication. For per-first-action safety counts, use `(x,y,t,first_action)` or an equivalent first-action bitmask. Deduplicating only `(x,y,t)` across all first actions would lose action-specific information.

On one empty-board five-step check, the existing search popped 1,346 queue entries containing only 302 distinct `(x,y,time,first_action)` tuples. This suggests avoidable repeated work; it is not an end-to-end speed benchmark. The search horizon should cover the relevant known explosion schedule, not rely blindly on a fixed five steps.

### 4.5 Further audit items and limits

- The trap detector advances the predicted state, then replaces its existing bomb list with original timers. Its no-bomb and bomb cases are not a clean comparison at the same simulated time. Compare synchronized scenarios before relying on marginal attack value. [Trap code][li-kill]
- Crates do not stop propagation in the supplied prediction loop. Whether that is correct requires the actual game engine; the earlier statement that crates necessarily stop these blasts should not be treated as established. The supplied predictor does not explicitly trigger bomb chain reactions. Their relevance is also engine-dependent. [Prediction code][li-predict]
- Hypothetical bombing at another position relocates the agent in a one-step prediction. This does not simulate the journey to that tile or prove bombing will be safe upon arrival. [Prediction code][li-predict]
- Route BFS initializes unreachable distances to a finite Manhattan-based fallback. Those values can still contribute to target scores. The proposed representation uses explicit reachability flags instead. [Route BFS][li-bfs]
- A small table stores values exactly for its encoded keys, but that does not make its abstraction Markov or eliminate aliasing. Distinct full boards can still share a key and require different decisions.

The historical isolated checks used synthetic settings `COLS=ROWS=17`, bomb
power 3, bomb timer 4, and explosion duration 2. Their original upload bundle
did not contain the engine, callbacks, symmetry implementation, or checkpoint.
Those limitations describe those checks, not today's local file availability.
Validate Agent 034 against the current project engine. No claim of full-game
improvement follows from the synthetic checks.

## 5. What the user's previous experiments contribute

The following are recorded results from the current [Agent 034 design insights][project-note], not new measurements made for this document.

| Reported finding | Consequence |
| --- | --- |
| Base navigation FQI: 23.52 mean coins after 300 rounds under the 100-step protocol; history FQI: 35.76 under that protocol | Short history is supported by evidence. |
| Fresh-board, 400-step navigation: history FQI 41.45 versus base FQI 25.17 | History benefits were not limited to the training boards. |
| History-8 stagnation FQI: 43.27 coins, 50.0% completion, but remaining repeated-state problems | Retain the compact history mechanism without assuming loops are solved. |
| Combat history FQI improved Loot Crate; mixed training restored navigation but weakened crate results | Training distribution and skill retention matter. |
| Adding fourteen local-topology values worsened mixed FQI, with further decline during longer training | More dimensions and more training do not automatically help. |
| Repaired 46-feature DDQN: 49.06 Coin Heaven coins and 46.20 Loot Crate coins, with 100% solo survival in the reported fixed suite | DDQN was not uniformly unsuccessful; this remains a strong balanced-solo control. |
| Agent 027 retained scenario-balanced experience and achieved strong solo results plus useful Classic scores | Preserve earlier scenario data during later stages. |
| The 101/113-input offensive expansion did not establish broad competition gains | Do not make that expansion the new FQI default. |
| Audited historical targets considered decision-time excluded actions in roughly 90–98% of audited decisions | Identical action-mask semantics in decisions and backups are essential. |
| Two boards had identical 46-feature vectors but opposite uniquely improving route actions | Fix demonstrated missing information rather than increasing dimensions generically. |
| Route-aware features resolved that witness but did not beat the DDQN control in aggregate | Correcting an information defect is necessary evidence, not proof of a better overall policy. |
| Feature extraction used 57.7% of measured CPU training time; optimization used 5.3% | Cheaper model fitting alone does not imply a dramatically faster system. |

These results support compact route information, short history, exact transition snapshots, balanced data, and controlled comparisons. They do not isolate feature count as the sole cause of DDQN's shortcomings.

## 6. Proposed architecture

### 6.1 Responsibilities

| Component | Responsibility | Rationale |
| --- | --- | --- |
| Board and timing model | Interpret actual engine rules and predict known hazards | Avoid learning deterministic physics through reward penalties. |
| Safety module | Produce legal actions, predicted-safe actions, and a defined emergency fallback | Keep constraints separate from strategic labels. |
| Route and feature module | Produce compact action-relevant distances, reachability, bomb context, and history | Preserve information that distinguishes useful choices. |
| Six regression trees | Estimate `Q_UP(z)` through `Q_BOMB(z)` from the same compact vector `z` | A reproducible, constrained baseline with one value model per action. |
| Policy | Choose the highest-valued allowed action, with masked exploration during training | Learning decides among feasible opportunities. |
| Dataset and fitter | Retain tagged transitions and periodically perform fitted updates | Reuse data while controlling scenario balance. |

The route and safety routines may read the full board. They return separate
distances and feasibility information, without combining them into a strategic
utility or selecting a target action. No imported Li module is a dependency.

The initial architecture uses six models receiving the same state vector. A shared model receiving action-conditioned inputs, `Q(z,a)=f(phi(z,a))`, was discussed as a later alternative; it is not the initial six-tree baseline and should be compared separately.

### 6.2 Safety contract

Pin the project's audited safety implementation and its transitive source
dependencies. Reject illegal moves and bombing without a verified escape
under the modeled conditions. Preserve the project's separate solo and
opponent-aware safety semantics; imposing combat conservatism on solo play
previously hurt useful bombing. The existing "useful bomb" restriction is
also a policy constraint: record it explicitly and keep it identical in both
Agent 034 representations and their matched controls.

Keep three masks distinct: immediate legality, predicted safety, and the
actual decision-time allowed set. If no predicted-safe action exists, the
allowed set consists of legal actions with the longest predicted survival.
Choose within it using the same recorded Q-value/tie policy used in replay
semantics. Cache the fallback flag and the actual allowed set for the
next-state backup. Never maximize over an empty set or silently restore all
six actions. An emergency option is not labelled safe merely because it was
selected.

Safety predictions are conditional on modeled opponents and engine dynamics. They cannot guarantee survival against every unmodeled simultaneous action.

## 7. Two explicit feature contracts

### 7.1 Reference: `route52-v1`

First reproduce the current [Agent 022 extractor][route-source] exactly,
including its action order, normalization, history, nine local-topology
values, and remaining-time scalar. Pin the dependency hashes. Do not load a
DDQN checkpoint into the trees; only the feature and safety contracts are
shared. Equivalence tests must compare actual values, not only vector length.

This provides an implementation reference and a representation shared with a
DDQN comparator. A learner-only performance claim still requires matched
reward, gamma, masks, data policy, budgets, and evaluation. The existing
historical DDQN scores alone cannot establish that causal claim.

### 7.2 Candidate: `compact23-v1`

The compact candidate is an exact projection of `route52-v1`, with 23 numeric
inputs. Direction order is `UP, RIGHT, DOWN, LEFT`; action order appends
`WAIT, BOMB`. Each route block interleaves `(cost, reachable)` pairs.

| Compact indices | Feature | Source indices in `route52-v1` | Count |
| --- | --- | --- | ---: |
| 0–7 | Coin route cost and reachability for each movement | 14–21 | 8 |
| 8–15 | Crate-approach route cost and reachability for each movement | 22–29 | 8 |
| 16 | Bomb available | 13 | 1 |
| 17 | Immediate crate yield, capped at 4 as in the reference | 33 | 1 |
| 18 | Existing bucketed escape distance after a hypothetical own bomb | 35 | 1 |
| 19 | Previous action index, with `WAIT` at round start | 39 | 1 |
| 20 | Recent visits to the current tile, capped at 3 | 40 | 1 |
| 21 | Existing six-level progress-clock bucket | 41 | 1 |
| 22 | Remaining-time scalar | 51 | 1 |
| | **Total numeric inputs** | | **23** |

The projection is unambiguous:

```text
compact23 = route52[[14, 15, 16, 17, 18, 19, 20, 21,
                     22, 23, 24, 25, 26, 27, 28, 29,
                     13, 33, 35, 39, 40, 41, 51]]
```

The previous-action index is a categorical value encoded as an integer, as
in the existing tree baseline. It does not imply a physical ordering of
actions. One-hot encoding is a later representation ablation. The original
draft's 22 logical entries / 28 numeric inputs, new `START` category, new
escape-count input, and different history bins are superseded by this exact
23-input projection. This removes several simultaneous semantic changes.

Legal, safe, and actual allowed masks remain policy/replay metadata. They are
not concatenated to `compact23`. The reference contains some local
safety/legality inputs that the projection removes; their information is not
assumed redundant merely because an external mask also exists.

### 7.3 Route and bomb semantics

Both contracts use the existing static BFS graph of `field == 0` tiles.
Walls and crates block that graph. Immediate action legality includes current
bomb/opponent occupancy; longer routes do not simulate those moving or timed
obstacles. Future hazards remain the safety solver's responsibility.

For movement `a`, cost is the shortest **remaining distance after the move**
to the nearest target, divided by board area. Do not add one for the proposed
move. An adjacent coin therefore gives `(0, 1)`. An illegal first move, empty
target set, or unreachable target gives `(0, 0)`. Preserve distinct reachable
distances without clipping, and use the reachability flag rather than an
invented finite fallback.

The current `crate_approach_tiles()` returns free cardinal neighbours of
crates. It does not enumerate every positive-yield bombing position and does
not prove future bombing safety. Keep that exact target definition for the
comparison. Expanding to all blast-range bombing positions or conditioning
targets on a simulated arrival changes the representation and is deferred.
Actual `BOMB` selection must always pass the current-state safety check.

Retain the reference's capped crate yield and bucketed escape-distance
semantics, including unavailable/no-escape cases; test those edge cases
against the donor implementation. An escape-distance bucket is not an escape
count or a general utility. A future continuation-count diagnostic must come
from the team's time-expanded solver, cover the known danger schedule, and
preserve each first action when deduplicating search states. Adding it as an
input or strategic tie-breaker is a separate experiment.

### 7.4 History and time semantics

Use the previous eight decision positions, excluding the current one when
counting visits. Cap the count at three. Match the existing observable
progress signature: visible coin set, crate count, living-opponent count,
and the agent's score. A signature change clears positional history and
resets the progress clock. This works in frozen games without training-only
events and includes newly revealed or opponent-collected coins.

Previous action persists across progress resets and starts as `WAIT` each
round. Reconcile it with the action reported by the framework; update memory
exactly once per transition. Use cached history snapshots so repeated feature
requests cannot append the same decision twice.

The existing clock bins are `0–2`, `3–4`, `5–8`, `9–16`, `17–32`, and `33+`,
encoded as 0–5. Remaining time is `max(0, 400 - step) / 400` under the official
horizon. Changing that horizon or denominator requires an explicit schema
change applied to both contracts. This compact observation is still not
claimed to be Markov.

### 7.5 What reduction may lose

The projection removes local geometry/danger inputs, opponent direction and
distance, immediate opponent blast yield, and global coin/crate/opponent
counts. These are not proven redundant. Test `route52` against `compact23`
with the learner fixed, and record the reduction as a feature-set ablation.
Do not attribute a gain to one individual removed input without another
controlled comparison.

Both contracts must pass the existing opposite-route witness. Passing that
test proves that one known alias is resolved; it does not prove sufficiency.
Retain counterexamples found in replays, especially cases with the same
compact observation but different useful safe actions. The first compact
candidate is a solo experiment. Before expecting it to hunt, test a small
explicit opponent-information addition against a matched no-addition control;
it currently has no direct opponent-location or attack-yield input.

Documentation check on 20 September 2026: a read-only calculation using the
existing extractor verified the 23-index projection, retained feature
meanings, opposite UP/DOWN route ordering, empty/adjacent coin targets, and
progress bins. This checks the proposed contract against existing code; it
does not validate a new Agent 034 implementation or establish learning
performance. The implementation must still pass the full checks in Section 11.

## 8. Training method: Tree FQI

### 8.1 Why it is a plausible fit

FQI learns action values through repeated supervised fits to Bellman targets. Regression trees can group similar structured feature vectors; ensembles are also established approximators within this method. See [Ernst, Geurts and Wehenkel (2005)][fqi-paper].

Our rationale is specific: the team's route and bomb quantities describe
useful choices; short history has helped earlier FQI; fitted passes reuse
experience; and small trees permit inexpensive value inference on CPU.
These motivate a comparison, not a claim that trees generally learn faster
than neural networks.

FQI still bootstraps from its own estimates. Limited coverage, poor representations, oversized trees, and distribution shifts can produce unstable or poor policies. FQI is not inherently more correct than DDQN, and a faster fit does not eliminate expensive feature computation.

### 8.2 Replay record

Each record should contain:

```text
decision-time features z
executed action a
reward r, with event accounting available for diagnosis
actual next-decision features z_next
terminal flag
current and next legal, predicted-safe, and actual allowed-action masks
emergency-fallback flags and exact history snapshots
scenario tag and complete opponent-lineup tag
episode/board provenance
feature, safety, reward, and history schema versions
```

Update memory exactly once per actual transition and cache the resulting
observation. Verify equality with the next `act()` observation when one
exists. Terminal records have no next-state bootstrap. Deduplicate the final
transition when both framework callbacks refer to it, and treat the official
400-step game ending as terminal. Diagnostic runner truncations must be
identified separately.

For representation comparisons, retain both exact feature views at collection
time with their schema IDs, or retain raw states plus exact immutable history
snapshots. Do not try to reconstruct removed `route52` information from a
23-value replay row. A same-data comparison is valid only for transitions
whose complete inputs and allowed masks are available to both variants.

Keep outcomes of executed allowed actions even when they end in death or failure; those outcomes are valuable learning data. Excluding forbidden actions from selection does not justify deleting failed permitted transitions.

### 8.3 Fitted update

For each fitted iteration, freeze all six old models. Compute targets before replacing any of them:

```text
if transition is terminal:
    y = r
else:
    y = r + gamma * max(Q_old[b](z_next)
                        for b in stored_allowed_actions_next)
```

For action `a`, fit its replacement regression tree on records whose executed action was `a`, with inputs `z` and targets `y`. Freeze the new six-model collection for the following iteration. Do not let sequential action fits mix old and new models in one target computation.

Initialize all six value predictors to zero. If an action has no records in
a fitting block, retain its previous predictor; a previously unseen action
therefore remains at zero. Record per-action counts and constant/unfitted
models. This default can be optimistic relative to negative fitted values,
so inspect coverage rather than interpreting it as learned competence.
Excluded actions receive `-inf` before the maximum, including unfitted ones.
Exploration samples only allowed actions; do not remove rare legal actions
from targets just to hide missing data.

### 8.4 Collection and fitting schedule

1. Start with masked exploration and a declared initialization.
2. Collect a registered block of games using the current policy.
3. Retain tagged experience and construct a scenario-balanced fitted dataset.
4. Perform several frozen-target fitting iterations between collection blocks.
5. Save fixed checkpoints and evaluate on development boards.
6. Resume collection with the updated policy.

The initial run collects its own experience from scratch. Reusing another
team-owned policy's data would be a separately declared experiment whose
interactions and provenance count toward the budget. Li's table, predicted
actions, and demonstrations are excluded from this plan.

### 8.5 Initial settings and ablations

| Setting | Proposed starting point |
| --- | --- |
| Models | Six individual regression trees, one per action |
| Maximum depth | 8 |
| Minimum leaf size | 5 |
| Fitted iterations per collection block | 5 |
| Collection block | 10 completed training games; refit between blocks, never inside `act()` |
| Gamma | 0.99, matching the audited route-DDQN recipe; the older combat FQI used 0.95 |
| Reward | Existing `reward_from_transition` from the team-owned combat/history FQI, also reused by the repaired DQN; pin source and event accounting |
| Initial tasks | Coin Heaven and Loot Crate |
| Episode mixture | Use the runner's existing alternating `mixed` curriculum |
| Replay retention | Separate FIFO partitions of up to 50,000 transitions per solo scenario; 100,000 total |
| Fitted sample | Equal scenario counts: `n = min(25000, count_CH, count_LC)` without replacement per scenario; use the same sampled rows for the five fitted iterations |
| Exploration | Masked epsilon-greedy, linear 1.0 to 0.05 over 100,000 actual training transitions |
| Randomness | Independent recorded streams for exploration, dataset sampling, and tree fitting |
| Evaluation policy | Frozen greedy Q-values; seeded random ties, evaluation decision seed 0, identical safety semantics |

These are proposed Agent 034 settings, not a claim to reproduce every older
FQI run. Depth 8 / five iterations match the navigation FQI; the inspected
combat-history implementation instead uses depth 10 / three iterations,
gamma 0.95, a 30,000-row buffer, round-wise epsilon decay, and round-end
fitting. Record this distinction when comparing historical scores.

Do not fit before both scenario partitions contain data. Log retained and
sampled counts by scenario and action at every fit; evaluate the realized
shares rather than assuming alternating games imply balanced transitions.
On entry to Classic, replace the two-tag sampler with a registered distribution
over both solo scenarios and complete opponent lineups. Preserve the solo
partitions and model state across that continuation.

Continue each fitting block from the preceding fitted models. Restarting from
zero every block and stopping after five passes would limit delayed-reward
propagation. Save resumable training state separately from the small inference
checkpoint: models, replay partitions, counters, schemas, and RNG states are
required for an exact continuation.

Only after the representation comparison, test a small Extra-Trees ensemble
if value instability or capacity warrants it. Keep features, sampled data,
masks, rewards, collection schedule, and evaluation fixed. Capacity, fitting
frequency, symmetry, and exploration changes each need separate identities.

### 8.6 Discount and credit assignment

Li's gamma of 0.6 substantially discounts delayed rewards:

| Gamma | Weight on a reward five steps ahead |
| --- | ---: |
| 0.60 | 0.07776 |
| 0.90 | 0.59049 |
| 0.95 | 0.77378 |

This motivates a later controlled sweep, especially for preparation followed by bombing and escape. It does not justify changing gamma, reward shaping, representation, and learner simultaneously.

Eligibility traces and discounting solve different problems: traces alter how experience updates earlier decisions, while gamma changes how delayed rewards are valued. Expected SARSA remains a useful alternative baseline if an online learner is desired. Double Q-learning is not an automatic fix for Li's sampled policy update, which does not use a maximizing bootstrap action. These alternatives are deferred experiments.

## 9. Reward and exploration policy

For the first FQI comparison, retain the audited project's reward accounting.
Do not use rewards for following Li's target, its trap prediction, or a
renamed planner recommendation. Later attack features must be independently
defined quantities tested against actual outcomes. Keep any reward change
separate from that representation experiment.

The earlier conversation sketched alternative reward numbers. They were illustrative and were never validated; this document does not adopt them as a new reward contract.

Potential-based shaping, `r_shaped = r + gamma*Phi(s_next) - Phi(s)`, is a possible later experiment. Policy-invariance claims require the usual appropriate state, discount, and terminal-boundary treatment; they should not be applied uncritically to aliased observations, nonstationary opponents, or arbitrary terminal potentials.

Visit-dependent exploration was also discussed. Keep a simple registered masked epsilon schedule initially. Defining reliable visit counts in a continuous or moderately rich feature space requires an explicit aggregation scheme and is a separate change.

## 10. Deferred strategic improvements

| Idea | Intended benefit | Why it is deferred / how to test it |
| --- | --- | --- |
| Restore a removed project-owned feature group | Repair a demonstrated compact-state alias | Start with the smallest relevant group from `route52`; keep learner, masks, and reward fixed. |
| Per-action escape continuation or bottleneck quantity | Distinguish fragile survival paths | Expose it from the team's safety solver; diagnose first, then test one input group. |
| Minimal opponent-location and blast-yield context | Give the solo compact learner information needed to hunt | Use independent measurements or the existing project features; compare against the same retained-data combat run without the addition. |
| Opponent-aware resource arrival | Avoid chasing coins an opponent is likely to collect first | Add arrival advantage, `min opponent arrival - own arrival`, with explicit route and opponent assumptions. |
| Bomb-location context | Describe potential attack locations before arriving | Keep travel cost, possible blast yield, and retreat feasibility separate; do not emit a chosen attack or bombing recommendation. |
| Escape-bottleneck analysis | Distinguish many reachable cells behind one exit from genuinely separate exits | Test one compact contention/bottleneck measure on diagnosed combat failures. |
| Counterfactual bomb value | Measure whether a proposed bomb improves the position relative to not bombing | Compare synchronized worlds using opponent policies that can react in each world; include openings created by destroyed crates. |
| Opponent response models | Improve attack and retreat predictions | Begin with a small declared set of plausible behaviours; model predictions are not guaranteed kills. |
| Persistent goals / options | Complete multi-step navigation or attack plans and reduce oscillation | Add remembered goals with safety interrupts; treat this as an architectural change, not a free feature. |
| Population opponents and self-play | Reduce exploitation of one fixed opponent's habits | Add only after solo competence; retain earlier scenario data and test unseen lineups. |
| Symmetry canonicalization or augmentation | Share directional experience | Verify eight unique transforms, masks, previous actions, and directional route channels. |

These are possible responses to measured failures, not an implementation
backlog to complete before the first comparison. Persistent goals and options
would change the architecture and are outside the initial Agent 034 scope.

## 11. Validation and evaluation plan

### 11.1 Before learning comparisons

- Verify engine-aligned blast timing, overlapping explosions, legal movement, crate rules, and any chain reactions.
- Verify separate legality, safety, and emergency masks; no combined strategic recommendation is produced by feature extraction.
- Verify exact `route52` parity, the 23-index projection, finite values, empty-target cases, history resets, and repeated-call idempotence.
- Pass `src/test_combat_route_features.py`'s opposite-route witness with both representations; add compact-state counterexamples when discovered.
- Verify terminal targets equal reward, prohibited actions never enter a maximum, empty safe sets use the stored fallback set, and all six targets use one frozen old model collection.
- Exercise progress changes, ordinary transitions, death, and round termination through real callback order; store each transition exactly once with the actual action-time context.
- Verify stratified dataset counts, no cross-scenario eviction, deterministic seeded sampling, and resumption of replay/RNG state.
- Run a short two-scenario smoke that actually reaches a fitting block, fits models, serializes them, reloads in a fresh process, and performs frozen inference. Smoke scores are not learning evidence.
- Load the inference bundle with sibling agents and `imported_agents` unavailable. Measure complete `act()` latency, including features and safety, on one CPU thread against the 0.5-second budget.

### 11.2 Controlled sequence

| Stage | Comparison | What it can establish |
| --- | --- | --- |
| A | `route52` FQI plus software/smoke checks | A working reference with tested inputs, safety, and fitted updates. |
| B | `route52` versus `compact23`, both using the Section 8 recipe | Whether the compact feature set helps under an otherwise matched collect/refit protocol. |
| C | Selected candidate versus old history FQI, repaired 46-feature mixed DDQN, and Agent 027 on the fixed suites | Practical performance relative to the strongest relevant project baselines; historical training differences remain disclosed. |
| D, if needed | One targeted feature, tree-capacity, or retained-batch change | Whether that particular change addresses a recorded failure. |
| E, after the solo gate | Hunting, then tournament lineups with retained solo data; minimal opponent input versus no-addition control | Combat value and skill retention. |

A smaller input plus FQI cannot be compared with the old DDQN and called a
pure learner improvement. Stage B fixes the learner; a separate fresh matched
DDQN/FQI run on `route52` would be needed for a learner-family claim. A
same-dataset Stage B diagnostic additionally controls transition composition,
but remains limited by that dataset's action coverage. Separate online
collection naturally produces different trajectories even with common seeds.

### 11.3 Solo pilot and selection

Proposed first budget: 300 rounds per representation and training seed, seeds
`0, 1, 2`, alternating solo scenarios, 400 steps per game. Save round 0 and
every 50 rounds; evaluate on development boards `33000–33002`, seat 0, decision
seed 0. Register all training-board ranges and ensure they exclude evaluation
boards. An extension to 600 rounds is a later experiment, not an automatic
response to poor results.

Before seeing results, register this common-round selection rule across all
three training seeds:

1. Exclude checkpoints with any solo invalid action or self-death, or survival
   below 100% on the development games.
2. Among the remaining checkpoints, maximize the smaller of the two
   scenario mean coin counts. Break ties by Coin Heaven completion, then
   smaller maximum no-progress duration, then the earlier checkpoint.
3. Freeze that round's three seed checkpoints. Do not choose a different
   best seed or round for each test scenario.

If no checkpoint clears safety, diagnose the failure before promotion. If
round 0 wins, report failure to improve on the untrained policy; it is not a
successful learned candidate merely because its shield survives.

Run [the fixed solo suite][solo-suite]: boards `30000–30007`, all four seats,
decision seed 0, 400 steps, 32 games per checkpoint/scenario and 96 per
candidate/scenario. These are established project comparison boards, not
newly untouched boards. Do not use their results to select another checkpoint.
For a fresh confirmation, reserve a disjoint board range after checking past
manifests, freeze the candidate, and record the range before running it.

Report coins, completion, crates, survival, suicides, invalid actions,
repeated states, and no-progress tails separately for both scenarios and each
training seed. Report seed variation and paired board/seat differences; an
aggregate gain must not hide a collapsed seed.

Promote the compact representation only if it meets the solo safety gates,
does not reduce either scenario's mean coin count relative to its matched
`route52` reference, and improves at least one scenario. Otherwise retain the
reference and diagnose the reduction. Also compare with the stronger DDQN
baselines: improving on a weak FQI reference alone is insufficient evidence
that Agent 034 should replace the current submission candidate.

### 11.4 Demonstrate the learner's contribution

Evaluate the frozen trained model against a zero-Q / random-tie policy with
the same masks and decision seeds. Log allowed-action counts, emergency
fallback frequency, fitted Q-values, and action changes attributable to the
trained models. Examine multi-option states; forced escape states cannot
show that a learner made a strategic choice.

Use this control to measure how much behaviour the shield already supplies.
A strong untrained shield is not itself proof of a rule violation, but a
learner with no useful measured contribution is not a successful learned
agent. Separately inspect feature and masking code for a hidden combined
action selector. Merely storing a learned model, changing feature names, or
showing that route perturbations change actions does not settle that question.

### 11.5 Combat and efficiency

Only introduce hunting/Classic after the selected representation passes the
solo gate. Record the opponent-feature decision from Section 7.5 before
training. Retain tagged Coin Heaven and Loot Crate data in every subsequent
fit. Start with the supplied peaceful/collector opponents, then apply
[the tournament protocol][tournament-protocol]: board seeds `32000–32007`,
all four seats, decision seed 0, 400 steps, `three_rule_based` and
`mixed_strong` lineups. Three training seeds give 96 games per lineup.
Keep single-opponent results distinct from this four-player gate.

Report score, wins/rank, kills, survival, suicides, invalid actions, and solo
regressions. A compact solo model with no attack context is not yet a
tournament candidate. Changes to opponent features, batch shares, safety
semantics, and reward must not be bundled into one claimed ablation.

Measure feature/safety time, fitting time, total wall time, environment interactions, dataset size and effective scenario shares, tree sizes, memory, and actual action latency. A fitting-time gain alone does not establish a faster training system.

## 12. Implementation checklist and decision record

Implement in this order when Agent 034 construction starts:

| Step | Concrete artifact | Completion condition |
| --- | --- | --- |
| 1. Reserve identity and pin sources | `src/agent_code/Agent_034_compact_route_fqi_agent/`; source/config manifest | Reconcile the registry with the reserved Agent 033 work; record exact donor versions without renaming existing agents. |
| 2. Feature and safety modules | `features.py`, `history.py`, `safety.py`, schema definitions | `route52` parity, `compact23` projection, route witness, history, and mask checks pass. |
| 3. Learner and data | `model.py`, `replay.py`, `train.py` | Six frozen-target fits, explicit tag balance, callback accounting, and resume checks pass. |
| 4. Inference interface | `callbacks.py`, versioned inference checkpoint | Correct `setup`/`act`, no training dependency at inference, standalone load and latency checks pass. |
| 5. Runner integration | Agent-specific config handling in `src/run_combat_training.py`; new manifests under `experiments/` | Runner accepts the agent and feature schema and records the resolved FQI recipe; no invented CLI option is assumed to exist. |
| 6. Smoke and record | Short two-scenario artifact and implementation notes | Actual fitting/reload verified; update `AGENTS.md` and `codex/implementation_roadmap.md` when the agent exists, and `codex/experiment_registry.md` for the registered run. |
| 7. Registered comparisons | Stage B/C artifacts and report-ready metrics | Three seeds, frozen selection rule, fixed suites, and learner-contribution control completed before promotion. |

Team-owned starting points are [route features][route-source],
[history features][history-source], [history callbacks][history-callbacks],
[combat FQI training][fqi-source], and [the shared safety solver][safety-source].
Use [the repaired DQN transition assembly][transition-source] for the actual
next-decision cache lifecycle, and the history modules for the specified
signature/encoding semantics. Record these dependencies as required by
[the design insights][project-note]. Review transitive imports:
several earlier agents share modules across directories. Package every
required inference helper inside Agent 034 with relative imports; its
tournament directory must not rely on sibling agents or the experiment tree.
Preserve provenance when adapting the team's code.

Use the existing `jobctl` workflow for substantial training/evaluation once
that stage is requested. Keep generated outputs in registered experiment
locations and use new run identities for ablations. The present request is a
design-document edit; implementation and training have not been executed.

The proposed first decision is therefore `route52` FQI as the tested reference
and `compact23` as its controlled reduction. Keep the project reward and
physics, exclude Li's strategic labels and utility, and advance according to
measured solo and combat results. This is the team's experimental design; it
does not claim a new RL invention or guaranteed superiority over DDQN.

## References and source provenance

### Primary code and methodological references

- Li-Jesse-Jiaze, [MLE_project_bomberman repository][li-repo], `agent_code/deep_learning_killer/`. Credit for that external planning and tabular-learning design belongs to its author(s); it is not Agent 034's code baseline.
- Local imported `features.py`, `train.py`, and `table.py` match the original upload hashes below. GitHub attribution: [features.py][li-features], [train.py][li-train], [table.py][li-table].
- Local import directory: `../imported_agents/Li-Jesse-Jiaze_MLE_project_bomberman__deep_learning_killer/`; `callbacks.py` was also inspected during this revision. Imports remain external reference artifacts.
- [Agent 034 design insights][project-note], current 365-line version at revision time: project evidence and experiment constraints. The original upload and current hashes are distinguished below.
- [Final project brief][project-brief], especially Sections 1, 7, and 9: copying restrictions, allowed feature examples, learning requirement, and systematic evaluation.
- [Route-aware DDQN design](combat_ddqn_route_agent.md), [combat history FQI](combat_fqi_history_antistag_agent.md), [solo protocol][solo-suite], and [tournament protocol][tournament-protocol]: local implementation and comparison contracts.
- Ernst, D., Geurts, P., and Wehenkel, L. (2005). [Tree-Based Batch Mode Reinforcement Learning][fqi-paper]. *Journal of Machine Learning Research*, 6, 503–556.

### Source hashes and review scope

The three Li hashes were rechecked against the local imported files on
20 September 2026. The synthetic audit examples in Section 4 were preserved
as historical observations and were not rerun. The project-insights hash was
updated separately rather than representing the older upload as current.

```text
features.py
f6e0c6dad0ceb0f7f3926eff4cebb4db6bb3b429d347ae12ca4da18c5016da21

train.py
626a350e7b295f8a56b649fea1ec39703fba97a9b9e06626e4dd2b006019eee8

table.py
63704908c3b9c43aa1f4fc97fcdd7c8ab8d5478150ab40d8c790303df977328b

agent034_fqi_design_insights.md — original uploaded version
eff7a9e5511a41d4b1771edb25abdf32747c5bea20a775036e719b60b16c8e54

agent034_fqi_design_insights.md — local version used for this revision
7ecee9a38fc81d880414ed4516a893af6237dc8ca5bf5c1ba9336f8b02012333
```

[li-repo]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman
[li-features]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py
[li-train]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py
[li-table]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/table.py#L4-L16
[li-learn]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py#L36-L49
[li-setup]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py#L15-L33
[li-end]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py#L105-L131
[li-symmetry]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py#L52-L62
[li-events]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py#L84-L101
[li-rewards]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/train.py#L134-L164
[li-danger]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L53-L67
[li-safe]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L69-L115
[li-predict]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L117-L178
[li-bfs]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L180-L234
[li-target]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L254-L351
[li-ties]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L334-L349
[li-kill]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L353-L363
[li-enemy]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L365-L381
[li-final]: https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman/blob/master/agent_code/deep_learning_killer/features.py#L373-L427
[project-note]: agent034_fqi_design_insights.md
[project-brief]: final_project.md
[route-source]: ../src/agent_code/Agent_022_combat_ddqn_route_agent/features.py
[history-source]: ../src/agent_code/combat_fqi_history_antistag_agent/features.py
[history-callbacks]: ../src/agent_code/combat_fqi_history_antistag_agent/callbacks.py
[fqi-source]: ../src/agent_code/combat_fqi_history_antistag_agent/train.py
[safety-source]: ../src/agent_code/combat_fqi_agent/safety.py
[transition-source]: ../src/agent_code/combat_dqn_agent/train.py
[solo-suite]: ../eval_suite/README.md
[tournament-protocol]: tournament_training_protocol.md
[fqi-paper]: https://www.jmlr.org/papers/v6/ernst05a.html
