# Agent 047 final-package evaluation (21 September 2026)

## Conclusion

The packaged Agent 047 is an excellent solo collector and a competitive but
lineup-sensitive four-player policy. Across five seat-rotated, 32-game
lineups it averaged 4.419 points, took 42 outright firsts plus 15 tied firsts,
and survived 69.4% of games. It won the cumulative-score comparison against
three rule-based agents and one mixed learned lineup, but placed last against
three Deep Learning Killers. The strongest dedicated result was against three
Arbiters, where it ranked first by cumulative score.

The evaluated checkpoint is the bundled `training.pkl`, SHA-256
`bec39a0c68e4e74d7af3382164b22eec6f89001ae62e8cb8eb7c0cc9d85f5aa0`.

## Protocol

- Original game rules, `classic` scenario, 400-step limit, CPU inference.
- Eight board seeds, 32000 through 32007, crossed with all four starting
  seats: 32 games per lineup.
- Policy RNG seed 0 and frozen checkpoints.
- One point per coin and five points per kill. Cumulative rank compares each
  fixed lineup slot's total over all 32 games.
- The isolation harness ran players in separate test processes to prevent
  package-name collisions; Agent 047 itself spawned no workers.
- These are familiar evaluation boards, not a new blind tournament set.

## Five-lineup four-player screen

| Opponents | Agent mean | Opponent means in slot order | Cumulative rank | First / tied first | Survival | Kills/game | Self-deaths | Invalid actions |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 3 rule-based | 4.56 | 3.12, 3.62, 2.81 | 1/4 | 9 / 4 | 53.1% | 0.344 | 8 | 7 |
| Rule-based, Coin Collector, Peaceful | 5.09 | 6.28, 3.75, 0.12 | 2/4 | 8 / 2 | 87.5% | 0.281 | 3 | 3 |
| Apex, Reaper, Arbiter | 4.62 | 2.59, 2.84, 5.59 | 2/4 | 7 / 2 | 43.8% | 0.313 | 2 | 12 |
| Overlord, Agent041 seed 0, Agent043 seed 1 | 4.91 | 2.12, 3.97, 0.62 | 1/4 | 14 / 3 | 75.0% | 0.219 | 3 | 8 |
| Deep Learning Killer, Harvy, Ruehl | 2.91 | 3.66, 3.09, 2.78 | 3/4 | 4 / 4 | 87.5% | 0.125 | 0 | 8 |

Aggregate over these 160 games: mean score 4.419, 42 outright firsts, 15
tied firsts, 69.4% survival, 0.256 kills/game, 16 self-deaths, and 38 invalid
actions.

## Dedicated homogeneous lineups

Against three Arbiters, Agent 047 scored 172 points (5.375/game) and ranked
first; the three opponent slots scored 113, 137, and 150. It recorded 7
outright firsts, 5 tied firsts, 62.5% survival, 18 kills, 3 self-deaths, and
11 invalid actions. Mean decision time was 12.03 ms, maximum 55.75 ms, with
no decisions above 500 ms.

Against three Deep Learning Killers, Agent 047 scored 68 points (2.125/game)
and ranked fourth; the opponent slots scored 84, 91, and 79. It recorded 4
outright firsts, 5 tied firsts, 84.4% survival, no kills, 1 self-death, and 11
invalid actions. Mean decision time was 10.63 ms, maximum 63.36 ms, with no
decisions above 500 ms.

The high survival but last-place cumulative result against Deep Learning
Killers is the clearest weakness: Agent 047 stayed alive but did not convert
combat opportunities into kills or enough coins.

## Additional checks and limitations

Solo evaluation over 32 games per scenario reached 50.00 Coin Heaven coins
and 47.34 Loot Crate coins, both with 100% survival. An unmodified native
32-round run against three rule-based agents scored 119 points, behind the
best opponent slot's 129; it was not seat-rotated and did not fix a separate
policy RNG seed, so it is a robustness check rather than a paired result.

Source inspection found that `last_novelty_intervention_step` is initialized
in `setup()` but is not reset at a round boundary. A late-round intervention
can therefore suppress the novelty guard early in the next round. The
evaluation did not alter the policy or isolate the causal effect of this
issue.

Machine-readable homogeneous-lineup summaries, selected replay files, and
rendered videos are committed under
[`../eval_suite/results/agent047_final_assessment_20260921/`](../eval_suite/results/agent047_final_assessment_20260921/README.md).
