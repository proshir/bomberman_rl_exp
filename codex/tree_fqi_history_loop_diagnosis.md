# Remaining loops in the learned-history coin agent

## Scope and replay validation

Replayed the 192 frozen 400-step fresh-board games for the retained
100-step-trained `tree_fqi_history_agent` checkpoints from
`experiments/tree_fqi_history_400step_confirmation/`. The replay diagnostic
reimplemented its greedy action selection around the same feature extraction,
legal mask, trees, and seeded tie-breaking. Coin totals and repeated-state
counts matched every saved game, and checkpoint hashes were unchanged.
Artifacts, including traces for terminal-cycle games, are in
`experiments/tree_fqi_history_loop_diagnosis/`.

The history agent remains the strongest retained Stage 1 candidate. This
diagnosis explains its remaining failures; it does not claim a new model result.

## Measured failure pattern

Of the 192 games, 71 completed all coins and 121 did not. Every incomplete game
ended in a repeated position-and-remaining-coins cycle. The final-cycle periods
were:

| Period | Failed games |
|---|---:|
| 1: stationary WAIT | 56 |
| 2: back-and-forth | 50 |
| 3 | 6 |
| 4 | 6 |
| 10 | 3 |

For failed games, the median no-progress tail after the last collected coin was
304 steps (mean 305.6). In that tail, there was at least one legal move reducing
the shortest free-tile distance to a visible coin at every decision. The greedy
policy selected an action outside that set 27,706 times out of 36,979 (74.9%),
and selected WAIT 18,516 times (50.1%). Moving toward the nearest coin is a
navigation diagnostic rather than a proof of the best full collection order,
but WAIT and immediate reversals in these persistent no-progress tails are
direct evidence of a value-ranking failure.

The selected wrong action's value exceeded the best distance-reducing legal
action by a median of 0.497 when averaging this margin within each failed game.
This is not mainly random tie-breaking.

Concrete witness: training seed 0, board 22063, corner 0 never collected a
coin. At position `(1, 1)`, DOWN was the only distance-reducing legal movement,
but the frozen trees assigned WAIT 8.0969 and DOWN 8.0623. The policy selected
WAIT for all 400 steps. The observed feature vector was unchanged throughout,
so the erroneous ranking repeated indefinitely.

## Remaining representation ambiguity

The history inputs reduce ambiguity but do not eliminate it. Of 2,343 observed
history-feature tuples with a route-improving legal move, 95 occurred in states
whose route-improving action sets were disjoint. For example, one identical
feature tuple required LEFT on board 22051 at `(11, 7)`, but UP or DOWN on that
same board at `(3, 5)`. A deterministic tree policy must give that tuple the
same action values in both situations.

This supports the earlier conclusion that map information is still compressed
too aggressively. The failed route-feature candidate showed that merely adding
coarse per-direction distance buckets did not solve the learning problem.

## Next controlled hypothesis

Keep the history features, tree architecture, legal mask, and training horizon
fixed. Test one separate history-FQI variant with a small dense **training-only
route-progress reward** derived from the change in shortest distance to a
visible coin, alongside the existing coin reward and step cost. The agent would
still choose actions from learned Q values; no path direction would be forced.

This targets the observed issue directly: the current trees can assign a high
value to an endless WAIT or reversal loop because rewards arrive only at coin
collection. A progress signal gives each route-reducing move evidence against
that erroneous value ranking. It must be evaluated against the unchanged
history agent on completion, repeated states, and coin score, because shaping
can favor a greedy nearest coin route rather than the best overall collection
order.
