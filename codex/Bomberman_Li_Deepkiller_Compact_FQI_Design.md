# Bomberman: lessons from Li's Deepkiller and a proposed compact FQI agent

Date: 20 September 2026  
Status: conversation synthesis and design proposal; no new agent or training run has been created.

## 1. Recommended direction

Build a hybrid agent with three responsibilities:

1. **A verified planner and safety module** interpret the full board, bomb timing, legal movement, routes, and escape possibilities.
2. **A compact state representation** passes action-relevant planning results and short history to the learner.
3. **Six fitted action-value models**, initially small regression trees trained with fitted Q iteration (FQI), learn which allowed action is most valuable.

The motivation is to preserve the useful planning structure in Li's `deep_learning_killer` while improving representation consistency, safety correctness, and learning from experience. Additional features must resolve a demonstrated decision problem. A larger feature vector or a different learner is not, by itself, an improvement.

Tree FQI is a justified next baseline, not a proven replacement for the strongest DDQN. The proposed compact state is also a hypothesis, not a validated sufficient representation of Bomberman.

## 2. Evidence and attribution

This note distinguishes three evidence levels:

- **Verified source findings:** inspection of the uploaded `features.py`, `train.py`, and `table.py` from Li's `deep_learning_killer`.
- **Isolated checks:** small synthetic-board checks executed against the uploaded feature extractor. These establish specific behaviours, not full-game performance.
- **Reported project results:** findings in the supplied `agent034_fqi_design_insights.md`. The underlying experiment logs and DDQN implementation were not independently audited in this conversation.

Li's repository is [Li-Jesse-Jiaze/MLE_project_bomberman][li-repo]. The linked GitHub `master` files are attribution references; direct retrieval failed during the initial discussion. The uploaded files subsequently became the authoritative code for this review. Line references below correspond to those uploads, and the source hashes are recorded at the end. No commit SHA was supplied.

The initial pasted conversations were useful context, but their explanations are superseded wherever direct source inspection established a correction.

## 3. What Li's Deepkiller actually does

### 3.1 Learning algorithm

Despite its name, the supplied `deep_learning_killer` uses a dictionary-backed Q-table, not a neural network. Each state key has six values for `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, and `BOMB`. Missing keys receive six zeros. See [table.py, lines 4–16][li-table].

The update in [train.py, lines 36–49][li-learn] has the sampled on-policy form:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
```

However, `a'` is selected by calling `choose_action()` inside `learn()`. The training function does not receive the next action actually executed in the environment. The earlier description of this as unquestionably trajectory SARSA was too strong. If `choose_action()` samples afresh, this is a sampled policy backup rather than the exact observed SARSA transition. That is not automatically invalid, but the missing `callbacks.py` is needed to establish the precise action-selection semantics.

Verified training settings are alpha 0.1, gamma 0.6, initial epsilon 1.0, epsilon minimum 0.1, and round-wise decay 0.9995. The table is saved every 100 rounds. Reloading the Q-table does not restore the exploration schedule: setup initializes epsilon again. The decay condition can also take epsilon slightly below its nominal minimum. See [training setup][li-setup] and [round completion][li-end].

The schedule takes roughly 4,605 decays to reach epsilon 0.1. This is not evidence of the actual training duration. The supplied files do not establish the original number of rounds, wall time, hardware, or a broad tournament benchmark. The previously quoted saved-table size was not verified because the checkpoint was not supplied.

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

This explains the appeal of Li's architecture: the learner makes decisions over substantial handcrafted planning results. The small output vector does not mean the underlying reasoning is small.

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

Training applies four rotations and three flip configurations, producing twelve update calls per transition. A square has eight unique dihedral transformations; with conventional transformation implementations, some of these twelve combinations duplicate reflections. The supplied `symmetry.py` was absent, so the exact transformation implementation remains unverified. A replacement should explicitly enumerate and test the eight unique symmetries, including action, history, and mask transformations. See [symmetry update loop][li-symmetry].

## 4. Source findings that change the improvement priorities

### 4.1 Recomputed features can describe a different state

Target ties are resolved with `np.random.choice()`. Meanwhile, training recomputes both old and new features. The same raw board can therefore receive different state keys on different calls. A transition can be updated under a representation different from the one used to select its action; shaping based on `target` can also change.

An isolated check extracted features sixteen times from the same empty standard board and obtained four distinct symbolic states. This is a verified behaviour of the uploaded extractor. See [random tie selection][li-ties] and [feature recomputation][li-events].

**Design response:** store the exact decision-time feature vector, history snapshot, and mask. Prefer deterministic numerical planning features; let exploration occur in the policy. If stochastic features are retained, record the realized observation instead of reconstructing it.

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

This can erase a danger label and influence shaping. It does not establish what the missing action-selection code would do. See [target fallback][li-final].

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

The isolated checks used synthetic settings `COLS=ROWS=17`, bomb power 3, bomb timer 4, and explosion duration 2, consistent with the earlier supplied descriptions. Full engine rules, `callbacks.py`, `settings.py`, `symmetry.py`, and original checkpoints were not supplied for execution. No claims of full-game improvement follow from these checks.

## 5. What the user's previous experiments contribute

The following are reported results from the supplied [Agent 034 design insights][project-note], not new measurements made for this document.

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

The model receives a summary, while the planner still reads the full board. A compact learned input does not require throwing away board information before planning.

The initial architecture uses six models receiving the same state vector. A shared model receiving action-conditioned inputs, `Q(z,a)=f(phi(z,a))`, was discussed as a later alternative; it is not the initial six-tree baseline and should be compared separately.

### 6.2 Safety contract

Use the project's audited safety implementation where available, rather than transplanting Li's predictor unchanged. Reject illegal moves and bombing without a verified escape under the modeled conditions. Preserve the project's separate solo and opponent-aware safety semantics; imposing combat conservatism on solo play previously hurt useful bombing.

If no predicted-safe action exists, select among legal actions with the longest predicted survival, with a deterministic tie rule. Store the actual allowed set, including this emergency fallback, for the next-state backup. Never maximize over an empty set or silently restore all six actions.

Safety predictions are conditional on modeled opponents and engine dynamics. They cannot guarantee survival against every unmodeled simultaneous action.

## 7. Proposed state: 22 logical entries

The proposal preserves route information, useful bomb context, and the short history supported by prior experiments. It avoids a large generic local patch.

### 7.1 Ordered feature contract

Direction order is always `UP, RIGHT, DOWN, LEFT`.

| Logical indices | Feature | Count | Rationale |
| --- | --- | ---: | --- |
| 0–3 | Coin route cost after each movement | 4 | Distinguish directions that differ in actual access to coins. |
| 4–7 | Coin reachable after each movement | 4 | Distinguish no route/no target from a long route. |
| 8–11 | Route cost to a useful crate-bombing tile after each movement | 4 | Learn where to move before bombing. |
| 12–15 | Such a crate-bombing tile reachable after each movement | 4 | Avoid attraction to inaccessible opportunities. |
| 16 | Bomb available | 1 | Distinguish immediate bombing from preparing for later use. |
| 17 | Predicted crates destroyed by bombing here | 1 | Describe immediate bombing utility. |
| 18 | Number of viable first escape actions after bombing here | 1 | Describe modeled escape flexibility. |
| 19 | Previous executed action, including a round-start category | 1 categorical | Give directional memory. |
| 20 | Visits to the current tile within the recent eight-position history | 1 | Identify repetition. |
| 21 | Bucketed steps since progress | 1 | Distinguish a productive route from stagnation. |
| | **Total logical entries** | **22** | |

For an explicit numeric encoding, represent previous action with seven one-hot categories: six actions plus `START`. That yields **28 numeric inputs**, not 22: 21 numeric values plus seven indicators. This distinction must be recorded in the implementation. Six allowed-action flags are separate policy/replay metadata, not additional learned inputs in this proposal.

### 7.2 Route semantics

Use exact static graph distances for the initial route channels, under one documented occupancy convention shared with the existing route-aware control. Permanent walls and current crates block movement; bomb occupancy must follow the chosen control's convention. Opponent motion and future explosions belong to the time-dependent safety module. A static shortest route is not a guarantee that its complete future traversal will be safe.

For candidate movement `a`, use one step for the proposed move plus the shortest remaining distance from its destination to the relevant target set. If the destination is statically invalid or no target is reachable, set the reachability flag to 0 and the cost placeholder to 0. A reachable adjacent target therefore has cost 1; the flag distinguishes a missing route. Do not use an invented finite fallback distance as if it were a real route.

A useful crate-bombing tile is an open position from which a bomb has positive crate yield and passes the project's documented bombing-position feasibility assessment. Record whether that assessment assumes bombing immediately at the tile or simulates arrival. The former is a planning proxy, not verified safety after travel. Recheck actual bombing safety when the agent reaches the tile.

No distance clipping is proposed initially. If normalization or clipping is later introduced, record it and verify that it preserves the known route witness. `WAIT` and `BOMB` still receive values from their own trees using the complete vector and their masks; they do not need separate movement-route entries in this first proposal.

### 7.3 Bomb and history semantics

Bomb yield and escape count describe a hypothetical bomb here under the verified simulator. Bomb availability and legality still determine whether bombing can actually be selected. Define unavailable/invalid hypothetical cases consistently, for example with zero utility values and an excluded bomb action.

Use the previous eight decision positions, excluding the current one, to count prior visits to the current tile. Reset positional history and the progress clock on a coin collection or crate destruction, and reset all history at a new round. Keep previous action as the last executed action until a round reset. The exact history conventions should match the existing audited baseline; changing them is a separate representation change.

A proposed progress-clock encoding is `0–3`, `4–7`, `8–15`, `16–31`, and `32+` steps, mapped to ordinal values 0–4. These thresholds are a specification proposal, not measured optimal bins; retain existing validated bins if they differ for the controlled baseline.

### 7.4 Limits and compatibility with Agent 034

The 22-entry proposal is a new compact candidate. It has not passed the route witness, full-game checks, or comparison with the project's 52-input route-aware contract. It must not silently replace required values in that contract on an assumption of redundancy.

The first implementation must list its actual dimensional contract. Test the known route-aliasing pair: the new vectors must differ and the route channels must identify the opposite improving actions. If this candidate fails, revise it before training. Other aliases can remain even after this witness passes.

Remaining episode time, richer escape margins, goal identity, and additional opponent information are potential additions if specific failures justify them. Because time is omitted initially, the representation is not claimed to be an exact Markov state for the finite-horizon game. The previous evidence for remaining time was mixed and representation-dependent.

## 8. Training method: Tree FQI

### 8.1 Why it is a plausible fit

FQI learns action values through repeated supervised fits to Bellman targets. Regression trees can group similar structured feature vectors; ensembles are also established approximators within this method. See [Ernst, Geurts and Wehenkel (2005)][fqi-paper].

Our rationale for testing it here is specific: the planner supplies meaningful route and bomb quantities; short history has already helped the user's FQI; the dataset can be reused across fitted passes; and small trees allow constrained CPU inference. These are reasons to run a comparison, not a general claim that trees are more sample-efficient than neural networks.

FQI still bootstraps from its own estimates. Limited coverage, poor representations, oversized trees, and distribution shifts can produce unstable or poor policies. FQI is not inherently more correct than DDQN, and a faster fit does not eliminate expensive feature computation.

### 8.2 Replay record

Each record should contain:

```text
decision-time features z
executed action a
reward r, with event accounting available for diagnosis
actual next-decision features z_next
terminal flag
current and next allowed-action masks
scenario tag and complete opponent-lineup tag
episode/board provenance
feature, safety, reward, and history schema versions
```

Update memory exactly once per actual transition and cache the resulting observation. Never recreate a different stagnation state or redraw a target label when replay is fitted. Terminal records have no next-state bootstrap.

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

Use a documented initialization and fallback for models with insufficient samples. Excluded actions must never enter the maximum through default or unfitted values. Exploration and support diagnostics are needed for allowed but under-sampled actions.

### 8.4 Collection and fitting schedule

1. Start with masked exploration and a declared initialization.
2. Collect a registered block of games using the current policy.
3. Retain tagged experience and construct a scenario-balanced fitted dataset.
4. Perform several frozen-target fitting iterations between collection blocks.
5. Save fixed checkpoints and evaluate on development boards.
6. Resume collection with the updated policy.

FQI can therefore be used in an ongoing collect–refit loop; batch fitting does not require one permanently fixed dataset. If existing competent agents are used to seed data, disclose and count that data and its provenance when comparing sample efficiency.

### 8.5 Initial settings and ablations

| Setting | Proposed starting point |
| --- | --- |
| Models | Six individual regression trees, one per action |
| Maximum depth | 8 |
| Minimum leaf size | 5 |
| Fitted iterations per collection block | 5 |
| Gamma and rewards for the controlled comparison | Preserve the audited control's settings initially |
| Initial tasks | Coin Heaven and Loot Crate |
| Dataset mixture | Explicitly registered, retained, scenario-balanced shares |
| Exploration | Masked epsilon-greedy with a recorded schedule |
| Evaluation policy | Frozen greedy policy with the same safety semantics |

Depth 8, leaf size 5, and five fitted iterations come from the reported reproducible baseline, not an optimization study. Five passes are not a universal adequate planning horizon. Specify whether each block continues from the last fitted models or restarts from zero; the proposed collect–refit design continues from the preceding fitted collection. Repeatedly resetting to zero and stopping after five iterations would limit how far delayed outcomes can propagate within that fitting cycle.

After the baseline works, replace each tree with a small Extra-Trees ensemble as a single-factor experiment. Keep the features, dataset, masks, rewards, collection schedule, and evaluation fixed. Then consider capacity, fitting frequency, or exploration changes individually.

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

For the first FQI comparison, retain the audited project's reward accounting. Do not copy Li's +500 detector reward automatically. Predicted attack quality is safer to treat initially as candidate information whose usefulness must be tested against actual game outcomes.

The earlier conversation sketched alternative reward numbers. They were illustrative and were never validated; this document does not adopt them as a new reward contract.

Potential-based shaping, `r_shaped = r + gamma*Phi(s_next) - Phi(s)`, is a possible later experiment. Policy-invariance claims require the usual appropriate state, discount, and terminal-boundary treatment; they should not be applied uncritically to aliased observations, nonstationary opponents, or arbitrary terminal potentials.

Visit-dependent exploration was also discussed. Keep a simple registered masked epsilon schedule initially. Defining reliable visit counts in a continuous or moderately rich feature space requires an explicit aggregation scheme and is a separate change.

## 10. Deferred strategic improvements

| Idea | Intended benefit | Why it is deferred / how to test it |
| --- | --- | --- |
| Learn Li's separate coin, crate, enemy, and escape score contributions | Learn trade-offs currently fixed by hand weights | Test a small linear scorer or a fixed-learner feature ablation; do not append all scores to the baseline at once. |
| Opponent-aware resource arrival | Avoid chasing coins an opponent is likely to collect first | Add arrival advantage, `min opponent arrival - own arrival`, with explicit route and opponent assumptions. |
| Bomb-location planning | Create attacks before reaching a `KILL!` square | Account for travel time, changing board state, and the agent's retreat. |
| Escape-bottleneck analysis | Distinguish many reachable cells behind one exit from genuinely separate exits | Test one compact contention/bottleneck measure on diagnosed combat failures. |
| Counterfactual bomb value | Measure whether a proposed bomb improves the position relative to not bombing | Compare synchronized worlds using opponent policies that can react in each world; include openings created by destroyed crates. |
| Opponent response models | Improve attack and retreat predictions | Begin with a small declared set of plausible behaviours; model predictions are not guaranteed kills. |
| Persistent goals / options | Complete multi-step navigation or attack plans and reduce oscillation | Add remembered goals with safety interrupts; treat this as an architectural change, not a free feature. |
| Population opponents and self-play | Reduce exploitation of one fixed opponent's habits | Add only after solo competence; retain earlier scenario data and test unseen lineups. |
| Symmetry canonicalization or augmentation | Share directional experience | Verify eight unique transforms, masks, previous actions, and directional route channels. |

Temporally extended goals have an established connection to option models; see [Khetarpal et al. (2021)][options-paper]. These proposals are extensions to the current design, not claims of new research inventions or demonstrated performance gains.

## 11. Validation and evaluation plan

### 11.1 Before learning comparisons

- Verify engine-aligned blast timing, overlapping explosions, legal movement, crate rules, and any chain reactions.
- Verify safety and strategic labels cannot overwrite each other.
- Verify fixed feature order, finite values, reachability conventions, history resets, and decision-time caching.
- Pass the existing opposite-route witness before declaring the compact representation adequate.
- Verify terminal targets equal reward and nonterminal targets use the stored allowed set, including emergency fallback semantics.
- Verify fitting, serialization, reload, and frozen inference in a short smoke run. Smoke performance is not learning evidence.
- Test action latency under the reported official 0.5-second, single-CPU-thread budget.

### 11.2 Controlled sequence

| Stage | Comparison | What it can establish |
| --- | --- | --- |
| 1 | Existing reproducible FQI control with audited safety, snapshots, and rewards | A trustworthy implementation baseline. |
| 2 | Compact candidate versus the control with the learner and protocol fixed | Whether representation changes help and preserve both solo skills. |
| 3 | Single trees versus Extra-Trees, holding representation and data fixed | Whether ensemble approximation helps. |
| 4 | Alternating mixture versus staged retained batches | Whether batch composition improves skill retention. |
| 5 | One targeted combat addition at a time | Whether the added information addresses the diagnosed failure. |

A new representation plus a new learner changes two factors. To separate their effects, include a bridge comparison using the same learner on both representations, or both learners on the same representation. A same-dataset FQI comparison controls data composition, but its conclusions remain limited by the support of that dataset.

Select checkpoints on development boards before testing on untouched held-out boards. Use three independent training seeds and the existing fixed 400-step solo suites. Report navigation and crate results separately: coins, completion, crates, survival, suicides, invalid actions, repeated states, and no-progress tails.

Only introduce Classic after preserving both solo skills. Retain tagged Coin Heaven and Loot Crate data in subsequent fits. Use the existing tournament-aligned protocol from the project note: board seeds 32000–32007, all four seats, 400 steps, and three-rule-based plus mixed-strong lineups. Report score, wins/rank, kills, survival, suicides, invalid actions, and solo regressions. Do not infer competition quality from a strong solo result.

Measure feature/safety time, fitting time, total wall time, environment interactions, dataset size and effective scenario shares, tree sizes, memory, and actual action latency. A fitting-time gain alone does not establish a faster training system.

## 12. Decision record

The conversation supports a **compact, route-aware, history-aware, safety-masked FQI experiment** inspired by Li's planning decomposition and constrained by the user's own experimental evidence.

It supports preserving useful planning calculations, fixing state/target inconsistencies, and adding strategic information incrementally. It does not establish that Li's implementation is safety-correct, that the 22-entry state is sufficient, that six trees will beat DDQN, or that every earlier proposed feature belongs in the first agent.

The immediate deliverable is this design note. Implementation, benchmark execution, and training results remain future work.

## References and source provenance

### Primary code and methodological references

- Li-Jesse-Jiaze, [MLE_project_bomberman repository][li-repo], `agent_code/deep_learning_killer/`. Credit for the baseline planning and tabular-learning design belongs to its author(s).
- Uploaded `features.py`: 427 lines; reviewed directly. GitHub attribution: [features.py][li-features].
- Uploaded `train.py`: 164 lines; reviewed directly. GitHub attribution: [train.py][li-train].
- Uploaded `table.py`: 16 lines; reviewed directly. GitHub attribution: [table.py][li-table].
- User-provided `agent034_fqi_design_insights.md`, 292 lines: reported project evidence and experiment constraints. The linked filename refers to the supplied project note; its hash below identifies the reviewed version.
- Ernst, D., Geurts, P., and Wehenkel, L. (2005). [Tree-Based Batch Mode Reinforcement Learning][fqi-paper]. *Journal of Machine Learning Research*, 6, 503–556.
- Khetarpal, K., Ahmed, Z., Comanici, G., and Precup, D. (2021). [Temporally Abstract Partial Models][options-paper]. arXiv:2108.03213.

### SHA-256 of the inspected uploads

```text
features.py
f6e0c6dad0ceb0f7f3926eff4cebb4db6bb3b429d347ae12ca4da18c5016da21

train.py
626a350e7b295f8a56b649fea1ec39703fba97a9b9e06626e4dd2b006019eee8

table.py
63704908c3b9c43aa1f4fc97fcdd7c8ab8d5478150ab40d8c790303df977328b

agent034_fqi_design_insights.md
eff7a9e5511a41d4b1771edb25abdf32747c5bea20a775036e719b60b16c8e54
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
[project-note]: sandbox:/workspace/scratch/8b71d0e23e2e/upload/agent034_fqi_design_insights.md
[fqi-paper]: https://www.jmlr.org/papers/v6/ernst05a.html
[options-paper]: https://arxiv.org/abs/2108.03213
