# Agent 030 callback repair and Agent 031 comparison — 20 September 2026

Approved scope: repair 030 callback dispatch, add 12 offensive features, and
run a matched 101-input control versus 113-input candidate. No promotion is
assumed before results.

## Correction to earlier experiment interpretations

Agent 029 was trained for 600 rounds across seeds 0/1/2. Artifacts:
`/export/scratch/salitanl/bomberman_feature_variants_20260919/agent029_adversarial_window_staged_600_cpu_jobctl`.
Its frozen Classic suite is in the sibling
`agent029_adversarial_window_classic_600_jobctl_rerun` directory.

The first 030 continuation generated only Classic episodes into a fresh
buffer despite solo replay weights; it was stopped for observed forgetting.
The retained-solo rerun completed with 60% Classic, 20% Coin Heaven and 20%
Loot Crate episodes. Its results remain in
`/export/scratch/salitanl/bomberman_feature_variants_20260920/agent030_combat_escape_replay_retained_solo_classic_and_tournament_900`.

Both historical 030 runs have a second defect: imported game/end callbacks
resolved remember() in the base module, bypassing the custom tagging and decay.
Actual logged epsilon was 0.20 at episode 601 and 0.05 at episode 602. The claim
that 20%-to-5% decay occurred over 60,000 new transitions was false. The escape
quota was empty. Those runs do not establish an effect of escape replay or of
the intended exploration reset. Do not attribute their lower kill rate to
oversampling defensive transitions.

## Implemented repair and candidate

030 explicitly dispatches delayed and terminal transitions to its own remember.
Successful combat bomb placements and the following BOMB_TIMER decisions are
tagged through detonation, including a terminal transition and even if the last
opponent dies during the sequence. Exploration decays over new learner
transitions (including solo), and its counter is saved for later continuation.
As before, replay itself is not serialized, so a resumed buffer starts empty.

031 keeps the exact 101-feature prefix and adds four values for each of up to
three opponents: immediate blast exposure while armed, fraction of surviving
endpoints removed by our hypothetical bomb, a newly predicted trap, and
1/(1+path distance) to an attack tile (zero if unreachable). Slots are ordered
by Manhattan distance then position. Searches include known timers and
lingering flames; crates are static, future bombs and other players' future
occupancy are not predicted. A trap flag is conditional, not a guarantee.

Checkpoint migration extends policy and target input matrices from 101 to 113
with zero columns and likewise pads Adam moments. Existing parameters,
optimizer history and counters are preserved; initial Q-values agree within
floating-point tolerance. Reward and safety filtering do not change.

## Matched run and fixed evaluation

Artifact root:
`/export/scratch/salitanl/bomberman_feature_variants_20260920/offensive_feature_matched_v1`.

- `control101`: corrected 030, starting from original 029 episode-600 checkpoints.
- `candidate113`: 031, starting from zero-column versions of the same checkpoints.
- Seeds 0/1/2, episodes 601–900, 400 steps, seats rotated, same board schedule.
- Per five rounds: three 3x-rule-based games, one Coin Heaven, one Loot Crate.
- Base replay weights 60/20/20. With escape allocation: 48% ordinary Classic,
  16% Coin Heaven, 16% Loot Crate, 20% combat escape when available. These are
  target sample shares, not guaranteed oversampling relative to data frequency.
- Exploration .20 to .05 over 60,000 new transitions; same DDQN and reward.
- Development eval every 50 rounds on 33000–33002, seat 0. Initial imported
  checkpoint is now correctly labeled episode 600 rather than episode 0.
- Frozen evaluation includes original 029 and both final candidates, 8 boards
  (32000–32007), 4 seats, 3 checkpoints: 96 games per lineup per candidate.
- Five Classic lineups: peaceful, collector, rule-based, three-rule-based,
  and mixed rule-based/collector/peaceful. Both solo tasks use the same boards
  as a new matched regression confirmation, not the historical solo protocol.
- Total frozen games: 2,016. Report score, coins, kills, suicides, invalids,
  survival; for combat also outright wins, joint firsts, and midrank ties.
- No checkpoint selection based on the frozen suite; final episode 900 is fixed.

Entrypoint: `src/run_offensive_feature_comparison.py` prepares migrations and
manifests, runs either training command, and evaluates/aggregates after both
training jobs succeed. The job IDs are `agent030-callback-fixed-train`,
`agent031-offensive-train`, and `agent031-matched-eval`.

Validation covers callback/terminal dispatch, actual escape tags, epsilon
persistence, feature-prefix identity, opponent ordering, open versus trapped
geometry, lingering flames, preserved Q-values and padded optimizer updates.

## Completed frozen result

All three training jobs completed.  The frozen suite ran 2,016 games: 96
games for every candidate--lineup cell, with the specified eight boards, four
seats and three checkpoints.  The aggregation is
`offensive_feature_matched_v1/comparison.json` under the artifact root above.

| Classic lineup | 029 at 600 | corrected 101-input control at 900 | 113-input candidate at 900 |
| --- | ---: | ---: | ---: |
| Peaceful | **10.552** | 6.865 | 8.823 |
| Coin collector | **4.646** | 4.500 | 4.198 |
| Rule-based | 4.521 | 4.677 | **4.792** |
| Three rule-based | 2.781 | **3.115** | 2.990 |
| Mixed strong | **4.229** | 4.042 | 4.010 |

The repaired continuation substantially improved safety against the
rule-based lineup: survival changed from 58.3% (029) to 79.2% (control) and
88.5% (candidate); suicides changed from 22.9% to 8.3% and 4.2%, respectively.
The candidate also has the best single-rule-based score and win rate (52.1%),
but it does not improve the three-opponent or mixed-strong games and loses
solo regression performance relative to 029 (Coin Heaven 48.92 versus 49.27;
Loot Crate 45.07 versus 45.46).  Therefore 031 is not promoted as the overall
successor.  Keep the callback repair; treat the 12 offensive features as a
specific single-opponent improvement that needs a different combat-learning
intervention before another feature expansion.
