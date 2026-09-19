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
