# Stage 1 plan: coin navigation

## Objective

Develop and compare a broad set of learning agents in `coin-heaven`. Determine
which learning methods, state representations, and decision structures learn
efficient coin collection and which ideas may remain useful when crates, bombs,
and opponents are added.

The supplied `coin_collector_agent` is the reference policy. It is not a learning
model. The initial pilot evaluated each supplied agent on eight board seeds, two
action seeds, and four starting corners. The collector found all 50 coins in all
64 games and required 113--143 steps (mean 127.83). This creates a useful speed
target and shows that final coin count at 400 steps has a ceiling.

## Shared experimental method

All candidates use the same training and evaluation interfaces. Training,
development, and final evaluation seeds are disjoint. A learning-method comparison
uses independently trained instances, not only many games from one checkpoint.

For cheap screening, use a fixed interaction budget and a small number of training
seeds. Save learning curves and evaluate frozen checkpoints on matched board seeds,
action seeds, and starting corners. Use the pilot variation and measured runtime to
choose the larger confirmation budget before running it.

Primary screening outcome:

- coins collected within a fixed step budget short enough to avoid the 50-coin
  ceiling;

Secondary outcomes:

- fraction of games in which all 50 coins are collected within 400 steps;
- steps to collect all coins for successful games;
- invalid actions, evaluation runtime, training interactions, and training time;
- learning curves showing evaluation performance against training interactions.

Report individual training runs, mean differences, and 95% confidence intervals.
Keep failed and inconclusive candidates. Do not select checkpoints on final seeds.

Before the first learner pilot, extend the benchmark summary with completion rate
and completion steps. Evaluate the supplied agents at 50, 100, 150, and 200 steps
using the existing pilot schedule (board seeds 10--17, action seeds 0--1, and four
starting corners). Use that result to choose one primary screening budget that
distinguishes collection speed without making the outcome too sparse.

## Shared representation components

Implement reusable functions in project code for legal moves, relative coordinates,
maze distances, coin-density summaries, and rotations/reflections. These functions
may describe the state but must not directly prescribe the action. Keep the learned
decision inside each agent.

Candidate inputs may include:

- the agent position and legal adjacent moves;
- distances to the nearest few coins;
- numbers of coins in distance bands, directions, or local regions;
- local wall structure and corridor/dead-end information;
- the full wall, coin, and agent-position maps;
- remaining steps and remaining coin count.

Use the same basic reward definition initially: the environment coin reward plus a
small per-step cost. Treat additional reward shaping as a named variant so its effect
can be measured. Potential-based distance shaping is a later controlled variant if
the sparse signal prevents learning.

## Candidate families

Implement each family as a preserved agent directory. Begin with its smallest useful
version and stop expanding a family when the pilot shows a clear implementation
failure or no learning under the agreed budget.

1. **Tabular Q-learning.** First learn primitive movement values from a compact
   discrete state without symmetry handling. This is the first custom learner and
   establishes a simple lecture-method baseline. Then add canonical rotations and
   reflections as a separate variant of the same agent.
2. **Linear SARSA with eligibility traces.** Learn action values from maze-distance,
   coin-density, obstacle, and time features. Compare the base features with the
   addition of density features.
3. **Tree-based fitted Q iteration.** Learn nonlinear action values from stored
   transitions using regression trees. Use the same feature set as the linear agent
   so the function approximator is the main change.
4. **Compact-feature DQN.** Feed the shared numerical features to a small multilayer
   network. This tests nonlinear approximation independently of a spatial input.
5. **Spatial DQN.** Feed separate 17 x 17 maps for walls, remaining coins, and the
   agent position to a convolutional action-value network.
6. **Hierarchical target agent.** Learn which coin or region to pursue and use a
   target-conditioned learned navigator for primitive movement. Validate the
   navigator separately before training the selector.
7. **Attention target selector.** Represent remaining coins as a variable-size set
   with relative positions and maze distances. Learn the next destination, sharing
   the target-conditioned navigator with the hierarchical agent.
8. **Learned value with short lookahead.** Add bounded action-sequence search to the
   strongest suitable value learner. Compare the frozen learner with and without
   lookahead at the same checkpoint and record decision time.

Symmetry, curriculum, demonstration data, reward shaping, and lookahead are
experimental components rather than extra model families. Test them through focused
comparisons after the corresponding base agent works.

## Accepted first learner experiment

Implement plain tabular Q-learning first. Confirm that it can train, save, load, and
act without exploration during evaluation. Inspect its learning curve and behaviour
before changing the model.

The next variant adds symmetry canonicalization only. Keep the state features,
reward, update rule, hyperparameters, training budget, and seed schedules unchanged.
Transform both states and actions consistently under rotations and reflections.
Compare the variants across independent training runs using the same frozen-policy
evaluation boards.

The hypothesis is that symmetry sharing improves sample efficiency and reduces
variation between training runs because equivalent situations update the same table
entries. Compare learning curves throughout training as well as final performance;
symmetry may learn faster even if both variants eventually reach similar results.

## Implementation order

1. Complete the benchmark metrics and short-budget supplied-agent pilot.
2. Add shared training records: configuration, code version, seeds, interactions,
   checkpoint identity, per-episode outcomes, and evaluation links.
3. Implement plain tabular Q-learning and validate the full
   train/save/load/evaluate path.
4. Add symmetry canonicalization as one isolated change and compare it with plain
   Q-learning.
5. Implement linear SARSA, then tree-based fitted Q iteration, reusing the compact
   feature code.
6. Implement compact-feature and spatial DQN after confirming the installed neural
   library and expected training time.
7. Implement the target-conditioned navigator, hierarchical selector, and attention
   selector.
8. Add short lookahead to the most suitable learned value model.
9. Confirm promising and inconclusive candidates with comparable tuning budgets and
   fresh evaluation seeds. Record a Stage 1 decision for every family.

## Stage exit criteria

Stage 1 ends when every candidate family has a recorded keep, revise, or reject
decision supported by a pilot; at least two genuinely different learning models have
completed a fair confirmation comparison; frozen evaluations use fresh seeds; and
the selected branches, useful components, and observed failures are documented for
Stage 2.

Advancing to crates and survival remains a separate user-approved stage decision.
