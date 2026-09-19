# Literature review for `combat_dqn_r_topology_agent`

**Review date:** 19 September 2026

## Scope

This review treats the current agent as a small-state, vector-input DQN with replay, a target network, safety/action masks, and handcrafted history/topology features. It focuses on successful game-playing methods relevant to the measured failures: route aliasing, long-horizon credit assignment, sparse combat experience, and poor transfer from Loot Crate to Coin Heaven.

This note contains literature-derived recommendations only. It does not claim a new training result or implementation change.

## Directly relevant evidence

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) achieved strong Atari performance with DQN, replay, target networks, and a convolutional spatial representation. The relevant lesson is representation: spatial structure should remain available to the network instead of being compressed into target displacement signs and local aggregates.

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) reduced DQN overestimation by separating action selection from target evaluation and improved Atari policies. This is especially relevant because the local DQN audit measured very large unrestricted target values. A masked Double-DQN target is a high-priority algorithmic ablation after the current target-mask repair.

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) replayed high-significance transitions more frequently and outperformed uniform replay on 41 of 49 Atari games. For Bomberman, rare self-deaths, successful escapes, kills, and route breakthroughs are likely more informative than repeated WAIT/loop transitions. PER should be tested only after target correctness and replay/action-feature alignment are established.

- [Rainbow](https://arxiv.org/abs/1710.02298) combined Double DQN, prioritized replay, dueling networks, multi-step returns, distributional value learning, and noisy exploration, obtaining state-of-the-art Atari results. The most plausible first subset here is Double DQN plus short n-step returns and PER. Dueling, distributional learning, and NoisyNet should remain separate later ablations rather than being introduced simultaneously.

- [Value Iteration Networks](https://arxiv.org/abs/1602.02867) embedded an approximate value-iteration planner in a neural network and generalized better to unseen grid layouts than reactive policies. This matches the current diagnosis: the remaining problem is not simply too few hidden units; the MLP lacks route-preserving information and explicit long-range planning.

- [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/abs/1912.01588) showed that diverse environment distributions are important for generalization. This supports the measured conclusion that training only on Loot Crate cannot be expected to transfer reliably to Coin Heaven.

## Bomberman-specific evidence

- [Learning How to Play Bomberman with Deep Reinforcement and Imitation Learning](https://dl.ifip.org/IFIP-LNCS-11863/hal-03652029) compared vector state representations and PPO/MLP/LSTM variants. Their strongest configuration used a hybrid state representation and imitation learning followed by PPO. The transferable idea is to use a teacher to provide navigation and escape trajectories before sparse-reward RL takes over; the final policy can still be learned and deployed by the project agent.

- [Accelerating Training in Pommerman with Imitation and Reinforcement Learning](https://arxiv.org/abs/1911.04947) used imitation learning followed by PPO, with reward shaping, heuristic action filters, and curriculum learning. It beat heuristic and pure-RL baselines with 100,000 games. For this project, the important combination is teacher initialization plus staged task difficulty, not switching blindly from DQN to PPO.

- [Exploration Methods for Connectionist Q-learning in Bomberman](https://research.rug.nl/en/publications/exploration-methods-for-connectionist-q-learning-in-bomberman/) compared several exploration methods with an MLP Q-learner. Max-Boltzmann exploration performed best overall; TD-error-driven exploration was initially strong but unstable. This motivates a masked Boltzmann or NoisyNet exploration ablation after the state representation is repaired.

- [Developing a Successful Bomberman Agent](https://arxiv.org/abs/2203.09608) reached first place among approximately 2,300 agents in a Bomberman-like arena using beam search, compact state encoding, opponent prediction, and aggressive pruning based on simulated survival. Although this is not a pure ML solution, it shows that explicit bomb/survival simulation is a powerful complement to a learned action scorer or teacher.

- [Action Space Shaping in Deep Reinforcement Learning](https://arxiv.org/abs/2004.00980) found that domain-specific removal of pointless actions can be important for successful game learning. The current action mask is therefore justified, but the hard safety mask and softer usefulness filter should be logged and ablated separately: over-filtering could remove strategically useful bomb or movement actions and create another target/behavior mismatch.

## Proposed successor direction

The highest-priority successor should preserve the current escape-safety repair while replacing the route-destroying representation. Two feasible alternatives are:

1. **Low-risk vector extension:** for each candidate action, compute action-conditioned shortest-path and survival features: distance to visible coins, reachable crate targets, reachable-region size, nearest safe tile after a bomb, blast timing, and remaining episode time. Add a unit test showing that the existing aliased-route witness produces different features for the two boards.

2. **Spatial network:** encode full-board channels for walls, crates, coins, bombs/timers, explosions, opponents, and the controlled agent with a small CNN, optionally concatenated with scalar history and remaining-time features.

Training should use a curriculum of Coin Heaven navigation, Loot Crate bombing/escape, mixed Coin Heaven/Loot Crate episodes, and finally opponent scenarios. Evaluation must retain separate gates for navigation, survival, kills, invalid actions, and score; aggregate scores can conceal the current Coin Heaven failure.

## Recommended ablation order

1. Corrected 32-feature versus 46-feature baseline, three seeds, both solo scenarios, and the existing opponent suite.
2. Route-preserving representation, with the route-aliasing unit test and the fixed evaluation protocol.
3. Mixed/curriculum training including Coin Heaven.
4. Masked Double DQN.
5. Three- to five-step returns.
6. Prioritized replay with importance correction.
7. Dueling head, then masked Boltzmann or NoisyNet exploration.
8. Optional imitation pretraining or a short-horizon survival-aware planner/teacher.

## Status

These are literature-derived recommendations. No model change or new training result is claimed by this note. The measured safety results and Coin Heaven diagnosis remain in [`combat_dqn_r_topology_agent.md`](combat_dqn_r_topology_agent.md).
