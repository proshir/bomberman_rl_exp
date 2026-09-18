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
| Combat vanilla DQN implementation | Minimal PyTorch DQN comparison | Implemented and smoke-tested; substantial performance pilot not yet run | [`combat_dqn_agent.md`](combat_dqn_agent.md) |
| Combat vanilla DQN pilot | 32-feature neural combat baseline | 300-round mean 18.48 versus 19.35 for matched FQI; fresh 600-round run peaked at 9.96 and degraded to 6.28 | [`combat_dqn_agent.md`](combat_dqn_agent.md) |
| Combat topology-feature DQN | Neural ablation of the failed 14-feature FQI topology block | 600-round best 16.53 at round 100, final 2.86; below 32-feature DQN and FQI; rejected | [`combat_dqn_topology_agent.md`](combat_dqn_topology_agent.md) |

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
control: its best checkpoint was 16.53 and its final checkpoint 2.86. Double-
DQN and other algorithmic enhancements remain intentionally deferred until the
vanilla baseline and checkpoint-selection protocol are stabilized.
