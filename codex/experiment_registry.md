# Experiment registry

Sahand was here.

This index records the durable experiment families in this project. Detailed
protocols and raw artifacts remain in the linked notes and `experiments/`
directories. Results are not compared across incompatible horizons, board
sets, or evaluation protocols without an explicit note.

## Stage 1: coin navigation

| Experiment | Algorithm / purpose | Result or status | Details |
|---|---|---|---|
| Q-table pilot | Tabular Q-learning baseline | Established the simple baseline and legal-action variants | [`q_table_pilot.md`](q_table_pilot.md) |
| Linear SARSA pilot | Linear SARSA(λ) | Online temporal-credit baseline; loop follow-up measured separately | [`linear_sarsa_pilot.md`](linear_sarsa_pilot.md) |
| Tree FQI pilot | One regression tree per action | Strong compact CPU navigation baseline | [`tree_fqi_agent.md`](tree_fqi_agent.md) |
| History FQI pilot | Previous action and recent-position context | Improved navigation and reduced feature aliasing | [`tree_fqi_history_agent.md`](tree_fqi_history_agent.md) |
| 400-step history pilot | Official 400-step horizon | Higher mean collection than the 100-step comparison, with completion tradeoffs | [`tree_fqi_history_400step_pilot.md`](tree_fqi_history_400step_pilot.md) |
| 600-round history extension | Longer training and held-out selection | Round 400 selected as the balanced Stage-1 candidate | [`tree_fqi_history_600round_pilot.md`](tree_fqi_history_600round_pilot.md) |
| Stagnation history 8/16 | Longer history and progress clock | History-8 selected; history-16 did not generalize better | [`tree_fqi_history_stagnation_600round.md`](tree_fqi_history_stagnation_600round.md) |
| Loop variants | Revisit penalties and interventions | Steps-since-coin and a small revisit penalty were useful; no universal fix | [`tree_fqi_loop_variant_confirmation.md`](tree_fqi_loop_variant_confirmation.md) |
| DQN coin pilot | Replay/target-network MLP | Best mean about 29.71; no full completion; below history tree under its protocol | [`dqn_coin_pilot.md`](dqn_coin_pilot.md) |

## Stage 2: crates and combat safety

| Experiment | Algorithm / purpose | Result or status | Details |
|---|---|---|---|
| Combat FQI pilot | Bomb-aware tree FQI with safety search | Safe, but round-300 mean 6.54 coins; crate gate failed | [`combat_fqi_agent.md`](combat_fqi_agent.md) |
| History/anti-stagnation combat | Combat FQI plus navigation history | Round-300 crate result 19.35 and fresh-board improvement, but Coin Heaven regression to 19.70 | [`combat_fqi_history_antistag_agent.md`](combat_fqi_history_antistag_agent.md) |
| Crate diagnostics | Repeated-state and no-progress instrumentation | 17.02 coins, 49.67 crates, 245.6 repeated states, 227.2-step maximum no-progress stretch | [`combat_fqi_history_antistag_agent.md`](combat_fqi_history_antistag_agent.md) |
| Mixed curriculum control | Coin Heaven and loot-crate training | Round-300: 39.14 Coin Heaven coins and 10.41 loot-crate coins; safe but stagnant | [`combat_fqi_history_antistag_agent.md`](combat_fqi_history_antistag_agent.md) |
| Local topology, 300 rounds | 3x3 patch and local openness | 37.79 Coin Heaven and 8.62 loot-crate coins; below control | [`combat_fqi_history_antistag_topology_agent.md`](combat_fqi_history_antistag_topology_agent.md) |
| Local topology, 600 rounds | Longer-training follow-up | Best balanced checkpoint around round 400; round 600 degraded to 28.07 / 3.06; rejected as default | [`combat_fqi_history_antistag_topology_agent.md`](combat_fqi_history_antistag_topology_agent.md) |
| Combat vanilla DQN implementation | Minimal PyTorch DQN comparison | Implemented; target masking and action-time replay consistency repaired; 29 tests and smoke test pass | [`combat_dqn_agent.md`](combat_dqn_agent.md) |
| Combat vanilla DQN pilot | 32-feature neural combat baseline | 300-round mean 18.48 versus 19.35 for matched FQI; fresh 600-round run peaked at 9.96 and degraded to 6.28 | [`combat_dqn_agent.md`](combat_dqn_agent.md) |
| Combat topology-feature DQN | Neural ablation of the failed 14-feature FQI topology block | 600-round best 16.53 at round 100, final 2.86; below 32-feature DQN and FQI; rejected | [`combat_dqn_topology_agent.md`](combat_dqn_topology_agent.md) |
| Route-aware combat DDQN | 52-input successor with exact action-conditioned coin/crate route costs and remaining time | Completed 600-round mixed run and matched fixed suite. It underperforms the 46-feature mixed control: 45.08 Coin Heaven / 53.1% completion and 37.20 Loot Crate coins, with a severe seed-0 failure. | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Spatial hybrid Rainbow | 5,912-value full-board, action-conditioned dueling quantile successor | Phase-0 implementation only; fixed contract and safety/feature tests exist, but no data, checkpoint, GPU run, or measured performance yet | [`rich_gpu.md`](rich_gpu.md) |
| 46-feature DDQN mixed follow-up | Matched topology representation under alternating Coin Heaven/Loot Crate training | Completed: fresh three-seed, 600-round run; final internal aggregate 48.67 Coin Heaven coins / 33.3% completion and 48.89 Loot Crate coins, all with 100% survival | [`combat_dqn_r_topology_agent.md`](combat_dqn_r_topology_agent.md) |
| 46-feature DDQN Loot Crate-only baseline | Topology representation under the old specialist curriculum | Completed previously as a resumed 300-to-600-round, three-seed run at `experiments/combat_ddqn_r_topology_300round_20260918/`; fixed-suite and classic results are retained | [`combat_dqn_r_topology_agent.md`](combat_dqn_r_topology_agent.md) |
| 52-feature DDQN Loot Crate-only follow-up | Route representation under the old specialist curriculum | Completed: fresh three-seed, 600-round run; final aggregate 47.89 Loot Crate coins and 118.44 crates, all with 100% survival; Coin Heaven was not trained/evaluated in this non-mixed protocol | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Fixed DDQN feature/curriculum 2×2 suite | Held-out comparison of 46/52 features × mixed/Loot-Crate-only training | Completed: 768 games, 96 per cell/scenario. Mixing is the dominant gain; 46-feature mixed is the selected balanced candidate (49.06 Coin Heaven / 64.6% completion; 46.20 Loot Crate / 114.30 crates). Route features do not improve the aggregate result. | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Agent 024 action-safety feature pilot | 46-feature mixed DDQN plus per-action bomb consequences | Completed CPU 300-round pilot: 42.76 Coin Heaven / 47.9% completion; 38.90 Loot Crate / 96.92 crates; safe but not the best balanced branch | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Agent 025 short-cycle feature pilot | 46-feature mixed DDQN plus compact recent-action/cycle summaries | Completed CPU 300-round pilot: 48.51 Coin Heaven / 56.3% completion; 37.17 Loot Crate / 93.18 crates. Classic 288-game gate is safer than the 46-feature reference in two lineups but lower-scoring everywhere; not combat-ready. | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Agent 026 target-coverage feature pilot | 46-feature mixed DDQN plus compact global coin/crate coverage summaries | Completed CPU 300-round pilot: 37.76 Coin Heaven / 30.2% completion; 37.13 Loot Crate / 92.67 crates; rejected for now due loops and seed variance | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Agent 027 staged replay curriculum | Agent 025's 65 features; 100 navigation / 200 crate / 300 retained-combat schedule with scenario-balanced replay | Completed CPU three-seed 600-round run and 288-game classic gate: final solo 49.26 Coin Heaven / 60.4% completion and 47.04 Loot Crate / 116.36 crates; classic scores 10.02 / 5.38 / 4.54 versus peaceful / coin collector / rule-based. Rule-based survival is 43.8%. | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Agent 028 dueling-256 architecture ablation | Agent 027 control with only a 65--256--256 dueling MLP replacing the 128-wide ordinary MLP | Completed CPU three-seed 600-round run and 288-game classic gate; rejected: final solo 48.30 Coin Heaven / 15.6% completion and 42.81 Loot Crate / 107.71 crates, below Agent 027; classic scores 9.70 / 4.45 / 4.05 | [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md) |
| Agent 029 adversarial-window feature branch | Agent 027 plus 36 opponent-response inputs (101 total) | Three-seed 600-round training and 288-game Classic evaluation completed: scores 10.552 / 4.646 / 4.521 against peaceful / collector / rule-based. Earlier untrained status was stale. | [`offensive_feature_comparison.md`](offensive_feature_comparison.md) |
| Agent 030 combat-escape replay continuation | Initial Classic-only pilot stopped for forgetting; retained-solo rerun completed with 60/20/20 episode mix | Callback audit invalidated the historical escape-replay claim. The corrected 101-input 600→900 control completed the 2,016-game matched suite: safety improved, but it did not exceed 029 overall. | [`offensive_feature_comparison.md`](offensive_feature_comparison.md) |
| Agent 031 offensive escape features | 12 additional inputs (113 total), zero-column warm start from 029; corrected 030 training as matched control | Completed matched three-seed 600→900 continuation and frozen suite. Best rule-based score (4.792) and survival (88.5%), but no general combat or solo improvement; not promoted. | [`offensive_feature_comparison.md`](offensive_feature_comparison.md) |
| Agent 032 optimized feature/replay implementation | Agent 031's unchanged 113-input representation with shared opponent envelopes, allocation-light searches, and preallocated replay/device batches | Implemented as a separate agent. Exact feature equivalence passed; feature-only benchmark was 2.05× faster (51.3% less time). Latest replay sampling plus tensor handoff was 3.19× faster on CPU and 2.79× faster on RTX 2080 Ti. The optimization result is valid; its downstream league learning result is recorded separately. | [`offensive_feature_comparison.md`](offensive_feature_comparison.md) |
| Agent 032 imported 24-lineup league | Two-seed, 1,200-round one-learner league versus a matched built-in-roster control | Completed. Best pooled relevant score was 3.203 for the imported league versus 3.641 for the control; final scores were 3.031 versus 3.297 with equal 64.1% survival. Broad league not promoted. | [`agent032_imported_league_pilot.md`](agent032_imported_league_pilot.md) |
| Agent 033 two-member population | Alternating two-learners with shared population curriculum | Incomplete for promotion. Evaluation exists through global episode 600 (about 300 active learning games/member), but no final episode-1200 evaluation exists. | [`agent_competition_audit.md`](agent_competition_audit.md) |
| Agent 034 compact route-aware FQI | Six depth-8 trees, 28 numeric inputs, cached decision-time masks, tagged balanced fitting every 20 rounds | Initial three-seed mixed run stopped around round 250: frozen policies stalled because replay stored the pre-action vector as the next state. The transition cache and train/evaluation history parity are repaired in schema v2; tests pass, but v2 has no learning result yet. Old v1 checkpoints are invalid for FQI comparison. | [`agent034_compact_fqi_agent.md`](agent034_compact_fqi_agent.md) |
| Agent 035 corrected symmetric compact FQI | Standalone Agent 034 successor with full-range crate targets, single-count terminal rewards, eight-way D4 replay augmentation, weighted tagged replay, resumable training state, and timing/support diagnostics | Three-seed mixed 300-round run completed; final solo evaluation averaged 21.22 Coin Heaven coins and 15.87 Loot Crate coins / 41.55 crates. The 288-game episode-300 Classic gate averaged 0.84 / 2.96 / 2.64 score against peaceful / coin-collector / rule-based opponents, with 100% / 91.7% / 70.8% survival. | [`agent035_compact_fqi_symmetry_agent.md`](agent035_compact_fqi_symmetry_agent.md) |
| Agent 036 robust compact FQI | Agent 035 successor with corrected decision-time callback keys, attributable-progress history, original-timeline bomb escape accounting, configurable useful-bomb filtering, robust tagged balancing, and checkpoint schema v2 | Completed matched held-out solo evaluation: 21.64 Coin Heaven coins / 4.2% completion and 12.56 Loot Crate coins / 34.43 crates, with 100% survival and zero invalid actions. Slightly improves Agent 035 on navigation but regresses its crate result; not promoted as the balanced winner. | [`agent036_compact_fqi_robust_agent.md`](agent036_compact_fqi_robust_agent.md) |
| Agent 037 focused tournament DDQN | Optimized 101-input DDQN continued for 300 rounds from corrected Agent 030 with two focused Classic lineups and 20%/20% solo retention | Completed three seeds. Episode 1200 scored 3.490 against three rule-based agents and 4.396 against the mixed built-in lineup (3.943 pooled), about 10.2% over the corrected source. Seed-0 episode 1050 selected; imported-opponent generalization remains weak. The six-test contract suite and a 20-round episode-1050-to-1070 regimen smoke both passed. | [`Agent_037_Tournament_Fast_DDQN_Design.md`](Agent_037_Tournament_Fast_DDQN_Design.md) |
| Agent 037 regimen continuation | Three-seed warm continuation from Agent 037 episode-1200 to episode-1500 under the documented 60/20/20 two-lineup `league-combat` schedule | Episode-1200→1300 segment preserved; active continuation from episode-1300 as `agent037-regimen-continuation-1300-1500-workers8-20260921` with eval every 150 and 8×8 evaluation workers. Paired evaluation uses boards 34000--34007, all seats, both solo tasks, and both Classic lineups. | [`ddqn_training_regimen_20260921.md`](ddqn_training_regimen_20260921.md) |
| Final_finetune curriculum | End-stage Agent 037 schedule with solo foundation/retention plus only complete four-player rule-based and verified strong imported lineups; excludes Coin Collector and Peaceful | Implemented and smoke-tested; no long run launched. Default imported opponents are `imp_li_deep_killer` and `imp_alii_arbiter`; held-out imported policies remain reserved for evaluation. | [`final_finetune_protocol.md`](final_finetune_protocol.md) |
| Agent 038 symmetric population DDQN | 117-input route/reachability extension, one-transform D4 replay, warm-versus-scratch comparison, and focused learned-agent population | Warm and scratch v2 jobs running at the 21 September audit cutoff. Round 150 is a solo-foundation diagnostic: scratch leads Coin Heaven (49.85), warm leads Loot Crate (44.04 versus 36.00); population training has not started, so there is no promotion result. | [`Agent_038_Symmetric_Population_DDQN_Design.md`](Agent_038_Symmetric_Population_DDQN_Design.md) |
| Agent 039 compact audit DDQN | 104-input controlled compression of Agent 038: remove previous-action scalar, constant topology center, legality copies, and action-conditioned armed-opponent copies; add one global armed-opponent flag | Three scratch seeds reached episode 300 and were evaluated before stopping. Solo performance was 49.646 Coin Heaven / 40.417 Loot Crate; classic scores were 2.479 / 3.521 / 1.500 / 1.458 / 1.167 across the registered lineups. Mixed early result versus Agent038; no promotion decision. | [`Agent_039_Compact_Audit_DDQN_Design.md`](Agent_039_Compact_Audit_DDQN_Design.md) |
| Agent 040 optimized compact DDQN | Behavior-preserving Agent 039 implementation with direct compact extraction, per-state search reuse, faster D4 transforms, dense replay tag indexes, and inference-mode action selection | Implemented and contract-tested. Fresh representative feature extraction measured 1.81x faster and compact symmetry about 8x faster on one single-thread CPU benchmark. Training and matched promotion evaluation are pending. | [`Agent_040_Optimized_Compact_DDQN_Design.md`](Agent_040_Optimized_Compact_DDQN_Design.md) |
| Agent 042 combat-progress DDQN | Agent 041's 122-input dynamic-navigation DDQN plus 18 action-aligned combat-progress inputs, agent-local persistent history, and separate solo/combat exploration counters | Three-seed 300-round solo gate completed. Stagnation is partially mitigated through round 200 but returns by round 300 (49.38 coins, 60.4% completion, 109.8 repeated states); combat promotion is not claimed. | [`Agent_042_Combat_Progress_DDQN_Design.md`](Agent_042_Combat_Progress_DDQN_Design.md) |
| Agent 043 novelty/credit DDQN | Agent 042 plus action-aligned novelty, bounded safe loop intervention, event-attributed progress, three-step returns, and bounded potential-based combat shaping | The three-seed 300-round Coin Heaven gate finished at 50.00 coins, 100% completion, 17.44 repeated states, and 21.31 maximum no-progress steps. Two matched 900-round league jobs are active: the strict branch uses 100% complete Classic games after round 300, while `agent043-staged-league-memory-retention-900-20260921` uses randomized 80% Classic / 10% Coin Heaven / 10% Loot Crate blocks. Both use the focused three-rule-based and Arbiter + Deepkiller + rule-based lineups. | [`Agent_043_Novelty_Credit_DDQN_Design.md`](Agent_043_Novelty_Credit_DDQN_Design.md) |
| Tournament-aligned curriculum protocol | Opt-in four-player Classic schedule with lineup-tagged replay and frozen tournament gate | Implemented in `run_combat_training.py`; no training result yet. Historical `staged-combat` behavior is unchanged. | [`tournament_training_protocol.md`](tournament_training_protocol.md) |
| Cross-run DDQN audit and training regimen | Evidence-weighted synthesis of navigation, FQI, repaired DQN/DDQN, league, population, imported-opponent, and Agent 037/038 artifacts | Completed 21 September 2026. Recommends a masked 101--128--128--6 DDQN, staged solo-to-focused-population training, tagged replay, three-seed checkpoint selection, new blind boards, and CPU execution. No training was launched for this analysis. | [`ddqn_training_regimen_20260921.md`](ddqn_training_regimen_20260921.md) |
| Pre-combat evaluation suite | Standardized gate before `classic` opponents | Completed comparison: repaired DQN 7.04 Coin Heaven / 45.90 Loot Crate versus FQI 18.14 / 19.35; fresh DQN boards confirmed 5.87 / 45.51 | [`../eval_suite/README.md`](../eval_suite/README.md) |
| Classic DQN opponent suite | First competition-oriented evaluation | 288 fresh games completed: score 3.01 vs peaceful, 2.72 vs coin collector, 2.50 vs rule-based; rule-based survival only 35.4% with 0.75 invalid actions/game | [`combat_dqn_agent.md`](combat_dqn_agent.md) |

## Current interpretation

Tree FQI is the strongest measured compact CPU baseline, but its feature-vector
representation and rolling buffer are bottlenecks for full combat. The
topology ablation did not solve this, even with 600 rounds. The current
evidence supports retaining the 32-feature history/anti-stagnation agent as a
control while testing a spatial neural or action-conditioned hybrid with the
same safety gates. No opponent-training result is promoted to a final
submission without held-out survival, invalid-action, and performance checks.
The 32-feature vanilla DQN is now a measured neural baseline, but it has not
beaten history FQI and longer training degraded in the fresh 600-round run.
The topology-feature DQN ablation also failed to improve the 32-feature
control: its best checkpoint was 16.53 and its final checkpoint 2.86. At that
stage, Double-DQN and other enhancements were intentionally deferred until the
vanilla baseline and checkpoint-selection protocol were stabilized. The later
route-aware experiment below records the subsequent DDQN progression.

The route-aware DDQN successor has now completed its first three-seed,
600-round mixed-curriculum run. Its small internal evaluation shows a material
balanced-task improvement, including near-complete Coin Heaven behavior and
strong Loot Crate behavior in seeds 1 and 2. It is promoted from
implementation-only to a promising Task-2 candidate, not to the submission
model. Results are non-monotonic and seed-dependent, and the run changed both
representation and curriculum relative to the repaired 46-feature specialist.
The fixed pre-combat suite and a matched 46-feature mixed DDQN control are
therefore required before claiming that the new features caused the gain.

The sequential one-group pilots keep the 46-feature mixed DDQN as the current
balanced reference. Agent 025's explicit short-cycle summaries are the only
new group promoted to a longer confirmation because they produced the best
Coin-Heaven score, completion, and anti-loop diagnostic among the three, but
they did not recover the reference's Loot-Crate score. Agent 024 is retained
as a crate-focused secondary ablation; Agent 026 is rejected. The next run
should extend Agent 025 under the fixed 600-round protocol before any feature
combination or opponent training.

The 18 September 2026 debugging session then audited the frozen DQN
checkpoints and found that action-time safety masks were absent from vanilla
DQN Bellman targets, replay next-state stagnation features could disagree with
the next action-time features, and the 46-feature representation still merged
states requiring opposite route choices. Target masking and action-time replay
consistency have now been repaired and tested. These findings change the
progression order: a corrected matched rerun comes before architecture or
Double-DQN comparisons. See
[`dqn_training_audit_20260918.md`](dqn_training_audit_20260918.md).
