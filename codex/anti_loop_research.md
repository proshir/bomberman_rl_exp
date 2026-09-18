# Anti-loop methods for Coin Heaven RL

## Question and project evidence

This note reviews methods relevant to the measured failure of `tree_fqi_history_agent`: all 121 incomplete fresh-board 400-step games ended in a repeated state cycle, predominantly stationary WAIT or a two-step reversal. In the long no-progress tails, a legal move reduced distance to a visible coin at every step, but the learned FQI value ranking chose a different move 74.9% of the time. More importantly, 95 of 2,343 observed feature tuples merged states with disjoint route-improving action sets. See `tree_fqi_history_loop_diagnosis.md`.

The question is whether an RL-compatible anti-loop strategy exists. Yes, but the literature distinguishes preventing repeated *exploration* from repairing a policy whose state representation aliases different locations. Neither reward shaping nor more samples alone can guarantee that an approximate value function separates states represented by the same feature vector.

## Relevant research and projects

1. **Potential-based reward shaping (PBRS).** Ng, Harada, and Russell (1999) prove policy invariance when the additional transition reward is `F(s,a,s') = gamma * Phi(s') - Phi(s)`, for a bounded potential `Phi`. Their paper explicitly discusses distance- and subgoal-based potentials and warns that other shaping rewards can change the desired optimum. Source: [paper](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

   Our tested nearest-coin distance delta was a useful empirical probe, but it was not a complete PBRS design for the learner's discounted objective and it performed worse. Even a correct PBRS implementation is not expected to resolve the proved feature aliasing: policy invariance concerns the underlying MDP's optimum, while our trees approximate values after distinct states have been merged.

2. **Count-based and episodic novelty bonuses.** Tang et al. (NeurIPS 2017) describe the classical exploration bonus proportional to `1 / sqrt(N(s,a))`, including its pseudocount extension. Source: [paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3a20f62a0af1aa152670bab3c602feed-Paper.pdf). Savinov et al. (ICLR 2019) use an episodic memory and reward novelty determined by reachability from remembered observations; the authors report navigation improvements and publish [the method](https://research.google/pubs/episodic-curiosity-through-reachability/) and [reference implementation](https://github.com/google-research/episodic-curiosity).

   For Coin Heaven, this suggests rewarding a newly reached *full navigation state* during training, rather than prescribing a move. A terminal WAIT loop repeatedly reaches the same state, so its intrinsic return vanishes after the first visit. This stays RL: the FQI agent still selects the learned action with the highest Q value. The count must reset every episode and be included in the training state (or the reward is history-dependent from the learner's viewpoint). It can still penalize necessary backtracking, so it needs a small, separately tuned bonus and a fresh-board confirmation.

3. **Explicit cycle detection.** Ameloot and Van den Bussche, *Convergence in Navigational Reinforcement Learning* (2015), analyze a cycle-detection learning algorithm for reducible navigation tasks with acyclic solutions. Source: [paper record](https://www.researchgate.net/publication/285270591_Convergence_in_Navigational_Reinforcement_Learning). Tarbouriech et al. (ICML 2020) likewise emphasize that goal-oriented RL may contain policies that never reach the goal and that much prior theory assumes loop-free tasks. Source: [paper](https://proceedings.mlr.press/v119/tarbouriech20a.html).

   A runtime rule that rejects an action solely because it repeats a recent state would directly eliminate observed cycles, but it would be a hand-written planner/guard. It is useful as a diagnostic upper bound, not as the primary submitted Coin Heaven policy under the project's learned-action goal. It can also reject legitimate returns through a corridor.

4. **Bomberman implementations point to representation and exploration, not a ready-made loop cure.** Kormelink, Drugan, and Wiering's 2018 Bomberman Q-learning study compares exploration schemes and finds mixed exploration/exploitation methods stronger than purely greedy or random-walk behavior; Max-Boltzmann was their best overall method. Source: [paper record](https://research.rug.nl/en/publications/exploration-methods-for-connectionist-q-learning-in-bomberman/). A previous BombeRLe project used a dueling DQN, prioritized replay, and explicit board symmetries: [repository](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game). These are useful independent design references, not code to copy.

## Recommendation: one controlled RL candidate

Do **not** retest the same distance-progress reward or simply train the history agent longer. The measured effect was unfavorable and its state aliasing remains.

If another Coin Heaven variant is approved, test an **episodic novelty history-FQI** variant as one controlled design:

- Keep FQI, legal-action masking, action set, tree parameters, training budget, and evaluation protocol unchanged.
- Define a deterministic navigation-state identity from immutable board geometry, agent coordinate, and the exact visible-coin set. This identity is for counting and for a compact state feature; it does not select an action or run a planner.
- Reset `N(identity)` each episode. Append a capped bucket of `N(current identity)` to the existing history features so the learner can value first visits and revisits differently.
- During training only, add a small novelty bonus such as `beta / sqrt(N(next_identity))`, with `beta` selected by a small pre-registered sweep. Preserve the environmental coin reward and step cost. Avoid a hard action veto.
- Compare against the retained history agent on coins, completion, repeated-state count, and terminal cycle periods on held-out boards. Reject it if it reduces cycles but reduces completion because it discourages needed backtracking.

This candidate attacks both parts of the diagnosis: it gives repeated cycles progressively less training value and avoids forcing the tree to give identical values to all occurrences of the currently aliased feature tuple. It is still not guaranteed to win, because exact map/coin identity increases the effective state space. If it fails, the evidence would favor changing function approximation and map representation (for example, a small board-channel DQN) rather than further reward edits.
