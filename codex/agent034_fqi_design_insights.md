# Agent 034 FQI design insights

## Status and scope

**Proposal only — no Agent 034 directory, code, checkpoint, or training result
exists yet.**  The user has reserved Agent 034 as the next FQI-family
candidate after the in-progress Agent 033 population prototypes.  This note is
a synthesis of completed, relevant project evidence.  It is not permission to
launch training or to declare FQI superior to DDQN.

The question for Agent 034 is deliberately narrower than “replace DDQN with a
tree”: **can a compact, route-aware, safety-correct fitted-Q learner recover
the sample efficiency of the successful navigation FQI while retaining the
balanced solo and combat behaviour learned in later DDQN experiments?**

The assignment needs a learned agent that handles coin navigation, crates and
survival, hunting, and competition.  Runtime is also part of the design: the
official action callback has a 0.5-second single-CPU-thread budget.  Training
may be faster or parallel, but final inference must remain self-contained in
the agent directory.

## What the successful runs actually established

| Evidence | Measured finding | Design consequence for Agent 034 |
| --- | --- | --- |
| Base tree FQI | The compact one-tree-per-action navigation FQI reached 23.52 mean coins after 300 rounds, ahead of masked Q-learning at the same 100-step protocol. | Trees can be a useful compact-value baseline; retain fitted batch updates and legal-action targets. |
| History FQI | Adding prior action, recent-position visits, and steps since progress raised the 100-step mean to 35.76.  On fresh boards it reached 41.45 coins at 400 steps versus 25.17 for base FQI. | A state-only local representation is insufficient; learned history/progress context is high-value information, not cosmetic tuning. |
| Stagnation/history-8 FQI | Held-out at 400 steps: 43.27 coins, 50.0% completion, and 162.2 repeated states.  Longer histories and remaining time were weaker in that *specific* navigation representation. | Start with the proven short history/progress mechanism.  Do not add long histories by default. |
| Combat history/anti-stagnation FQI | In Loot Crate it reached 19.35 coins with 100% survival and no suicides/invalid actions; fresh-board confirmation was 16.68 versus 7.35 for the original combat FQI. | The safety shield plus history/progress can make FQI learn useful crate behaviour. |
| Mixed combat FQI | Alternating Coin Heaven/Loot Crate restored navigation to 39.14 coins, but Loot Crate fell to 10.41 and completion remained 1.0%. | Scenario mixture prevents one-task specialization, but the old FQI did not retain both skills well enough.  Mixed data must be a first-class design component. |
| Local-topology FQI ablation | Adding a 14-value 3x3/topology block reduced the 300-round mixed result to 37.79 Coin Heaven / 8.62 Loot Crate.  At 600 rounds it degraded to 28.07 / 3.06, with high repeated-state counts. | More local dimensions are not a remedy for trees.  Avoid a broad local-topology expansion and do not assume longer training repairs tree fragmentation. |
| Repaired 46-feature mixed DDQN | The matched balanced control reached 49.06 Coin Heaven coins / 64.6% completion and 46.20 Loot Crate coins / 114.30 crates in the fixed 2x2 suite, with 100% solo survival. | This is the primary balanced-solo comparison, not the earlier FQI results.  Agent 034 must be evaluated against it under the same protocol. |
| Agent 027 staged replay | A 65-feature curriculum with retained scenario-balanced replay reached 49.26 Coin Heaven coins / 60.4% completion and 47.04 Loot Crate coins / 116.36 crates.  Its Classic scores were 10.02 / 5.38 / 4.54 versus peaceful / collector / rule-based. | Opponent experience and retained solo data matter.  FQI should consume explicitly tagged, balanced batches instead of allowing recent combat data to erase navigation. |
| Agent 029--031 combat follow-up | Agent 031 improved one-rule-based survival to 88.5% and score to 4.792, but did not improve broad competition or solo regression relative to 029. | Do not begin Agent 034 with the 101/113-input offensive feature expansion.  First establish compact solo/crate competence, then test combat information separately. |
| Agent 032 timing work | In a 30-round paired run, feature extraction consumed 57.7% of CPU training time while optimizer calls consumed 5.3%.  Agent 032 reduced feature-only time 51.3%. | Replacing neural optimization alone is unlikely to make training dramatically faster.  Reuse efficient feature/safety computation and measure complete wall time. |

## Non-negotiable lessons

### 1. Safety is a hard constraint, not a reward the learner may rediscover

The successful safety record across FQI and DDQN comes from deterministic
action filtering: reject blocked moves and bombs without a verified escape;
when no fully safe action exists, choose a legal action with the longest
predicted survival.  This must remain outside the FQI model.

Both action selection **and fitted Bellman targets** must use the same
next-state safe-action mask:

```text
y = r                                      terminal transition
y = r + gamma * max_{a in safe(s')} Q_target(s', a)  otherwise
```

The 18 September audit found that historical DQN targets maximized over
actions excluded at decision time in roughly 90--98% of audited decisions.
The result was inflated values and collapsing policies.  Agent 034 must cache
the actual decision-time feature/history snapshot and the resulting next-state
safe mask in every transition; it must not reconstruct a different stagnation
state later.

The opponent-aware escape-margin repair is important only for Classic games.
It reduced useful solo bombing when applied unconditionally, so Agent 034 must
retain the current distinction between solo and opponent safety semantics.

### 2. FQI needs information that determines the action, not more local detail

The decisive representation failure is the route-aliasing witness: two normal
boards produced the same 46-feature vector while UP and DOWN were opposite
unique route-improving actions.  No regressor, neural or tree-based, can learn
both correct actions from the same input.

The 52-input route-aware design demonstrates the right type of correction:
for each candidate movement, provide exact static-BFS route cost and a
reachability flag to coins and crate-approach tiles, plus remaining time.
It passed an explicit test that distinguishes and reverses the preferred
UP/DOWN action on the witness.  Agent 034 should inherit that *testable
property*, rather than blindly inherit either the old 32/46 feature vector or
all later offensive features.

This does **not** establish that the 52 features are automatically the right
tree input.  The route-aware DDQN did not beat the matched 46-feature mixed
control in the fixed suite.  The valid conclusion is only that action-relevant
route information is necessary for the known alias, while its effect must be
isolated for FQI.

### 3. Short learned history and progress context are the strongest proven
anti-loop signal

Previous action, visits to the present tile in an eight-position window, and a
bucketed progress clock consistently helped navigation.  The history must reset
when progress occurs and the next-state history must be derived from the actual
transition.  A small revisit penalty was also helpful in navigation, but it
changes the objective and should be an independent ablation, not part of
Agent 034's first implementation.

Agent 025's compact recent-action/cycle values were the strongest of three
later 65-feature DDQN ablations for Coin Heaven and no-progress tails.  They
are a reasonable *later* FQI ablation only after the shorter proven
history/progress basis is measured; combining both at first would make the
result uninterpretable.

### 4. Curriculum and batch composition are algorithmic choices

Crate-only learning produces a competent local bomb/open/collect routine but
can destroy long-horizon navigation.  Alternating scenarios restores
navigation but did not automatically retain crate learning for the old FQI.
Agent 027 shows that staged exposure plus retained replay can preserve both
solo skills better, and it is the only completed branch with clearly strong
Classic score evidence.

For FQI, the analogue of replay is the transition dataset and the sampling
distribution used during every fitted iteration.  Therefore Agent 034 must:

- tag each transition by scenario and, for Classic, complete opponent lineup;
- retain Coin Heaven and Loot Crate transitions throughout later stages;
- sample or weight those tags to an explicit target distribution when fitting;
- record actual retained counts and effective batch shares; and
- never interpret a Classic-only continuation with an empty or freshly reset
  buffer as retained-solo training.

The initial Agent 030 continuation made exactly that last mistake, and its
claimed escape-replay effect was invalidated by a later callback audit.

### 5. FQI checkpoints are non-monotonic; choose before looking at final tests

Both FQI and DDQN can peak well before the final round.  History-8 stagnation
FQI peaked around round 300 and worsened with longer training.  The topology
FQI continued to decline through round 600.  Agent 034 must save fixed
checkpoints, select only on development boards, and use untouched held-out
boards for confirmation.  Training reward is a diagnostic, not a promotion
metric.

### 6. Competition failure is primarily dynamic safety and data coverage

Agent 027's solo suite had zero deaths, invalid actions, and suicides, but its
rule-based survival was 43.8%.  Replay diagnosis showed many deaths came from
the agent's own bomb after an opponent contested the intended escape square;
the current opponent-aware safety repair helps but does not solve all
simultaneous-move interactions.  A solo-successful FQI agent must therefore
pass the solo gates before Classic training, then receive explicit opponent
coverage and opponent-aware escape filtering.  A high Loot Crate score is not
competition evidence.

## Recommended Agent 034 hypothesis and first implementation boundary

### Hypothesis

A compact FQI agent with (a) deterministic, target-masked safety, (b)
action-conditioned route information that resolves the known alias, (c)
history-8/progress context, and (d) scenario-balanced fitted batches can learn
a safer, faster-to-fit balanced solo policy than the earlier combat FQI.  It
may provide a credible second learned model and a stronger starting point for
later combat training.

### What this hypothesis does not claim

- It does not claim FQI will beat the best DDQN/Agent 027 competition score.
- It does not claim that tree fitting removes the dominant feature/safety/world
  simulation cost.
- It does not claim route features helped the DDQN aggregate; only that they
  fix a demonstrated information defect.
- It does not authorize mixing a new reward, new safety rule, new feature
  block, new curriculum, and new tree algorithm in one untraceable run.

### Minimal first agent

Agent 034 should be a separate self-contained directory, leaving all earlier
agents unchanged.  Its initial scope should be deliberately small:

1. Reuse the audited combat safety module, reward accounting, exact
   action-time replay context, and deterministic fallback behaviour.
2. Use one fitted action-value model per action, with target values computed
   from a frozen prior fitted iteration.  Fit only from legal/safe observed
   transitions and bootstrap only through the stored safe next-action mask.
3. Start with a **compact route-aware history representation**, selected from
   the 52-input contract by removing only values proven redundant and retaining
   route, reachability, history-8/progress, and task-relevant bomb context.
   The final dimensional contract must be listed and tested before training.
4. Keep model capacity conservative initially.  The prior shallow tree setup
   (depth eight, minimum leaf five, five fitted iterations) is a reproducible
   baseline, not a proven optimum.  Any switch to ensembles, deeper trees,
   fitted Double-Q, or uncertainty penalties must be a named one-factor
   ablation after that baseline works.
5. Begin with the two solo scenarios.  Do not introduce 101/113-input
   opponent-window/offensive features, escape-transition quotas, or population
   opponents until the selected solo checkpoint clears the fixed gates.

This scope deliberately differs from the failed topology FQI in one important
way: it adds global, action-conditioned route information while avoiding a
generic local-patch expansion.  It deliberately differs from the DDQN branches
in one way: the fitted learner is the changed component; safety, scenario
definitions, rewards, and evaluation remain comparable.

## Required pre-training checks

Before any multi-seed run, Agent 034 needs evidence in four categories:

1. **Feature contract:** fixed length/order; action-time history behavior;
   progress reset; empty target handling; and exact representation of all
   terminal values.
2. **Route witness:** the old 46-feature vectors collide on the known pair,
   while the new Agent 034 route values distinguish the pair and rank the
   route-improving action correctly.
3. **Safety/target correctness:** unsafe actions cannot be selected,
   exploration is masked, targets use only stored safe next actions, and
   terminal transitions have zero bootstrap.
4. **End-to-end smoke:** short solo Coin Heaven and Loot Crate runs complete
   fitting, serialization, reload, frozen evaluation, and provenance capture.
   These smoke scores must not be reported as learning evidence.

## Evidence-gated experimental sequence

The first comparison must preserve the existing scientific conventions rather
than optimize the protocol in favour of FQI.

| Gate | Fixed comparison | Decision rule |
| --- | --- | --- |
| A. Solo pilot | Three independent seeds; Coin Heaven and Loot Crate at the official 400-step horizon; checkpoint evaluations on a registered development board set. | Require zero invalid actions and zero self-deaths in frozen solo games.  Retain all checkpoints; do not promote on training reward. |
| B. Held-out solo suite | Preselect a checkpoint from Gate A, then use the fixed held-out Coin Heaven/Loot Crate suite and compare with the 46-feature mixed DDQN control and prior FQI controls. | Judge both tasks separately: coins, completion, crates, survival, suicides, invalid actions, repeated states, and no-progress tails. |
| C. Curriculum ablation, only if needed | Hold features, safety, reward, and FQI settings constant.  Compare an alternating mixed schedule with explicit staged/retained-batch scheduling. | Attribute any gain to batch distribution, not a feature or reward change. |
| D. Classic introduction | Only after Gate B preserves both solo skills.  Start with a declared opponent mix and retain solo data in every later fitted dataset. | Apply the standard frozen Classic suite; report score, wins/rank, kills, survival, suicides, invalids, and solo regressions. |

The tournament-aligned protocol remains the final competition gate: three
training seeds, fixed board seeds 32000--32007, all four seats, 400 steps, and
three-rule-based plus mixed-strong lineups.  No favorable single seed or
checkpoint should substitute for this suite.

## Efficiency expectations and measurements

FQI is likely to make **fitting** cheaper and more CPU-friendly than repeated
neural mini-batch updates, especially if fitting happens between rounds rather
than in `act()`.  It is not automatically a faster training system: each game
still performs feature extraction, time-expanded safety search, environment
stepping, and opponent actions.  Since those computations dominated the
measured Agent 031 CPU run, Agent 034 should reuse allocation-conscious code
where feature equivalence is verified.

Record separately:

- wall time per environment transition;
- feature/safety time;
- dataset size and tag balance;
- fitted-iteration time and tree sizes;
- peak memory; and
- policy action latency in a real 400-step game.

Only an equal-budget comparison can support “faster”: report both wall-clock
time and environment interactions to reach a pre-registered checkpoint, with
the same CPU limits and scenario schedule.

## Risks to actively avoid

- **Tree extrapolation / target feedback:** repeatedly fitting to its own
  maxima can lock in a poor `WAIT`, bomb, or loop value.  Inspect per-action
  target ranges, leaf sizes, and action frequencies at each checkpoint.
- **Data fragmentation:** a sparse high-dimensional representation will make
  individual leaves memorise board accidents.  Prefer compact, causal values
  over copied local statistics.
- **Safety-induced data bias:** unsafe actions are intentionally absent or
  masked; never let their default/unfitted value enter a target maximum.
- **Forgetting hidden by averages:** report Coin Heaven and Loot Crate
  separately.  An aggregate can conceal a collapsed navigation policy.
- **Protocol drift:** do not change boards, horizon, seats, reward, curriculum,
  feature set, and fitted method in the same experiment.
- **Premature combat features:** opponent trapping and offensive values are
  expensive, incomplete predictions.  The Agent 031 evidence supports them
  only as a narrow single-rule-based improvement, not as a default FQI input.

## Decision record

The evidence supports implementing Agent 034 as a **new, compact,
route-aware, history-aware, safety-masked FQI baseline**.  It does not support
reusing the old 46-feature topology FQI unchanged, adding more local topology,
or claiming that FQI is categorically better than DDQN.  Implementation should
be followed by the pre-training checks above; training remains a separate user
decision.

## Source notes

- `codex/tree_fqi_agent.md`
- `codex/tree_fqi_history_agent.md`
- `codex/tree_fqi_history_stagnation_600round.md`
- `codex/tree_fqi_loop_variant_confirmation.md`
- `codex/combat_fqi_agent.md`
- `codex/combat_fqi_history_antistag_agent.md`
- `codex/combat_fqi_history_antistag_topology_agent.md`
- `codex/dqn_training_audit_20260918.md`
- `codex/combat_dqn_r_topology_agent.md`
- `codex/combat_ddqn_route_agent.md`
- `codex/offensive_feature_comparison.md`
- `codex/experiment_registry.md`
- `codex/tournament_training_protocol.md`
