# Repaired 46-feature topology DQN

Sahand was here.

`combat_dqn_r_topology_agent` is a separate 46-input agent. It keeps the
topology representation from the earlier topology ablation, but delegates all
algorithmic callbacks to the repaired `combat_dqn_agent` implementation:

- action-time safety masking in Bellman targets;
- exact action-time next-state features in replay;
- terminal empty masks and zero bootstrap values;
- the repaired replay buffer, checkpoint format, and optimizer logic.

The 46 inputs are the validated 32-feature history/anti-stagnation vector plus
the 14 local-topology values (3x3 patch, neighboring free/crate counts,
radius-two openness, dead-end flag, and straight-corridor flag).

The earlier 46-feature training launch using the old agent name was stopped
before completion. The new agent passed a 46-dimensional feature probe, the
19 relevant DQN/safety/topology tests, and a three-round end-to-end smoke run
with fresh-process checkpoint evaluation. No performance conclusion is drawn
from the smoke run; a matched multi-seed training run is still required.

Implementation is in the code repository under
`agent_code/combat_dqn_r_topology_agent/`, commit `0e7ac9e`.

## Combat escape-safety repair (19 September 2026)

Classic evaluation of the 600-round Double-DQN checkpoints showed that most
deaths involved the agent's own bomb: 21 of 22 deaths against the coin
collector and 49 of 54 against the rule-based agent. Three saved failures were
replayed step by step. In each one, the static safety search approved a bomb
and an escape corridor, but the opponent later occupied the intended escape
tile. The framework executes simultaneous choices in randomized order, so the
move became invalid and the agent died in its own blast.

The shared combat safety search now computes the tiles that nearby opponents
could occupy during an escape. A bomb is unavailable if every escape route
depends on one of those tiles. In games with opponents it also requires the
agent to leave its own blast one step before the last possible movement. The
timing margin is disabled in solo games because an initial ablation reduced
Loot Crate performance from 47.76 to 30.48 coins by suppressing useful crate
bombs. Focused tests cover a contested single exit, an uncontested alternate
exit, the combat timing margin, and preservation of the original solo route.

The final implementation was evaluated without retraining on the same three
600-round checkpoints, boards, seats, and opponents as the baseline. Each row
contains 96 games:

| Scenario | Metric | Before | After |
| --- | --- | ---: | ---: |
| Solo Loot Crate | mean coins | 47.76 | 47.55 |
| Peaceful | mean score / survival | 7.15 / 92.7% | 5.99 / 94.8% |
| Coin collector | mean score / survival | 3.96 / 77.1% | 4.04 / 86.5% |
| Rule based | mean score / survival | 3.68 / 43.8% | 3.70 / 49.0% |

Self-death games fell from 21 to 10 against the coin collector and from 49 to
36 against the rule-based agent. Rule-based wins/ties/losses improved from
19/0/77 to 22/2/72. Peaceful score fell because the conservative mask placed
fewer bombs, and invalid actions against the rule-based agent increased from
92 to 112 in total. The repair therefore addresses a confirmed failure mode
but does not solve dynamic collision handling after a bomb has already been
placed or the agent's broader navigation and combat weaknesses.

Final evaluation artifacts are in
`eval_suite/results/ddqn_r_topology_600_combat_safety_precombat_20260919/`
and
`eval_suite/results/ddqn_r_topology_600_combat_safety_classic_20260919/`.

## Known remaining navigation issue

The agent can still become stuck near a corner and can settle into a repeated
left-right movement loop; the equivalent up-down loop is also possible. Coin
Heaven makes this especially visible: the weak seeds spend most of a 400-step
game without collecting another coin. The eight-position history and
stagnation bucket reduce some repetition but do not guarantee that a learned
policy will leave a corner or break a two-tile oscillation. This is a known,
unresolved issue and should be reported separately from the combat escape bug
fixed above.

## Explicit DDQN mixed-curriculum follow-up (19 September 2026)

Sahand was here.

The runtime callback now explicitly sets `self.dqn_algorithm = "ddqn"` and the
training module re-exports the DDQN recipe for provenance. A fresh three-seed,
600-round mixed Coin Heaven/Loot Crate run completed at
`experiments/combat_ddqn_r_topology_mixed_600round_20260919/`. This is the
matched 46-feature control for the route-aware successor; it does not reuse an
existing checkpoint. Its final internal evaluation averaged 48.67 Coin Heaven
coins with 33.3% completion and 48.89 Loot Crate coins, with 100% survival.

The fourth cell was already completed previously as the resumed
300-to-600-round run at `experiments/combat_ddqn_r_topology_300round_20260918/`
with `curriculum=none` and `scenario=loot-crate`. Its fixed-suite and classic
results are retained in `eval_suite/results/`. Together with the completed
46/52 mixed and 52 non-mixed runs, this completes the planned 2-by-2
feature/curriculum comparison. A redundant fresh launch was identified and
stopped before completion; its partial files are preserved separately under
the `_aborted_duplicate` suffix and are not a result.

## Stage-2 gap diagnosis (19 September 2026)

**Status: diagnostic conclusion; no new training or agent change.**  The
apparent conflict with the imported Stage-2 results is not a crate-learning
failure.  The best current Double-DQN checkpoints are exceptionally strong on
the distribution on which they were trained: all three 600-round checkpoints
trained **only** on solo `loot-crate`, and score 47.50, 47.06, and 48.09 mean
coins on the held-out Loot Crate suite (with 100% survival).  This exceeds the
42.97 local Harvy reference in the imported-stage table on that one metric.

The same saved checkpoints fail the equally matched Coin Heaven gate:

| Checkpoint seed | Coin Heaven coins | Completion | Repeated states | Longest no-progress tail |
| --- | ---: | ---: | ---: | ---: |
| 0 | 10.34 | 0% | 371.69 | 361.97 |
| 1 (best) | 41.38 | 0% | 260.19 | 241.44 |
| 2 | 11.75 | 0% | 366.38 | 368.59 |

Each number is the fixed 32-game (eight boards x four seats), 400-step solo
protocol.  Thus the checkpoint-selection-friendly best seed is still below
the 50-coin / 100%-completion imported references, and has an average of over
240 consecutive non-progressing decisions.  The ordinary vanilla-DQN variant
is worse (best 21.84 coins and 342-step mean no-progress tail).  This is a
systematic navigation/generalization problem with severe seed variance, not a
small score deficit.

The immediate cause is insufficient state information for global route
choice.  The 32 base inputs retain only the nearest coin's displacement signs
and a distance bucket (all paths longer than seven are merged); the added 14
topology values are only a 3x3 local patch plus local counts/flags.  They do
not encode which first move lies on a path to the target, remote crates that
block one of two corridors, the set of remaining coins, or remaining episode
time.  The DQN training audit constructs two normal-board states with
identical complete 46-feature vectors but opposite unique route-improving
actions.  A feed-forward policy cannot choose correctly in both.  The capped
visit count and capped stagnation bucket also do not reliably break a
two-tile/corner cycle once it begins.

The training objective amplifies that representation limitation.  The run has
no Coin-Heaven episodes or explicit path-planning supervision: `loot-crate`
provides dense local rewards for crate destruction (+0.2), coin revelation
(+0.2), and bomb escape (+0.1), while movement toward a distant already-visible
coin is rewarded only after collection.  It therefore learns a highly reliable
local bomb/open/collect routine, but has little pressure or data to learn the
long-horizon, bomb-free routing policy that Coin Heaven requires.  Once
epsilon reaches 0.05 (about round 250), later replay is predominantly this
specialist behavior, increasing the chance that a poor navigation policy is
reinforced rather than corrected.

The earlier fatal DQN target-mask and replay-history bugs were repaired before
these `r_topology` checkpoints were trained (the training provenance is commit
`aa94990`, after repairs `ed48a76` and `0e7ac9e`).  They are therefore not the
primary explanation for this measured Stage-2 gap.  The combat safety repair
also changes no solo Coin-Heaven result because it is disabled in solo games.

**Implication for a successor agent:** retain the proven bomb/escape action
shield and Loot-Crate policy as a baseline, but do not add more local scalar
features or merely train longer.  The next controlled change should give the
learner route-preserving global information (for example, board channels or
per-action shortest-path features) plus remaining time, verify that it
distinguishes the audit's aliased-route witness, and train/evaluate with a
documented mixture that includes Coin Heaven.  It must be compared separately
on both scenarios; averaging them would conceal this failure mode.

## Feature redundancy and successor candidates

**Status: diagnostic ranking and experiment plan; no new training result.**

Sahand was here.

The current 46 inputs should be ranked with grouped, retrained ablations rather
than neural saliency or pairwise correlation alone. Correlated inputs can make
an individual permutation test look unimportant even when the information is
useful through its feature group. The primary usefulness measure should be the
change in held-out Coin Heaven completion and coins, Loot Crate coins and
survival, and classic survival, kills, invalid actions, and win rate. The
route-aliasing witness is a prerequisite check: a proposed representation must
produce different action-relevant values for the two boards that currently
produce identical 46-vectors.

The clearest exact redundancies are:

| Features | Why they are redundant | Planned treatment |
| --- | --- | --- |
| 37, the centre cell of the 3x3 patch | The controlled agent occupies this cell, so it is always free | Remove |
| 42, free-neighbour count | It is determined by the four cardinal patch cells | Remove or group-ablate |
| 43, adjacent-crate count | It is determined by the same cardinal patch cells | Remove or group-ablate |
| 45, dead-end flag | It is a thresholded function of free-neighbour count | Remove or group-ablate |
| 46, straight-corridor flag | It is determined by the cardinal patch cells | Remove or group-ablate |

Removing these five values creates a 41-feature control without discarding the
raw local patch. The movement-safety flags (5--8) are also partly redundant
with blocked movement (1--4), local danger (9--13), and the external safety
mask. They should be tested as a group, after the exact redundancies above,
because they can still provide useful context about how constrained a state is.

The proposed successor features are ranked by route-aliasing reduction,
action relevance, information not already present in the 46-vector, compute
cost, and evidence from the imported Stage-2 agents:

1. **Per-action exact coin route costs.** Replace the nearest coin's direction
   signs and coarse distance with the shortest route cost after each candidate
   movement, plus an unreachable indicator. This directly resolves the
   constructed witness where UP and DOWN must be ranked differently.
2. **Per-action exact crate-approach route costs.** Use the same representation
   for reachable tiles from which a bomb can hit a crate. This preserves the
   useful Loot-Crate objective while removing the current direction/distance
   aliasing.
3. **Remaining episode time.** Add normalized steps remaining; early and late
   states otherwise look identical even though their optimal actions differ.
4. **Multi-target route coverage.** Add the number of coins reachable through
   each first action or a discounted sum of route values over the nearest few
   coins. This avoids committing solely to the nearest target.
5. **Per-action time-expanded survival margin.** Record survivable steps,
   earliest danger, reachable safe tiles, or distance to a safe tile after each
   action. This gives more information than the current binary safety flags.
6. **Bomb escape robustness.** Count viable first escape moves and independent
   escape routes, and record spare steps before detonation. This complements
   the existing escape distance and addresses opponent-blocked corridors.
7. **Opponent reachability and collision risk.** Estimate whether an opponent
   can claim each destination or escape tile within the next few steps. This is
   essential for classic training but cannot be learned from solo Loot-Crate
   episodes.
8. **Explicit short-cycle history.** Add the recent movement sequence,
   reversal indicators, two-cycle/four-cycle flags, consecutive WAIT count, and
   displacement. These are more direct anti-loop signals than the current
   capped visit count, although they do not provide a route by themselves.
9. **Opponent trapping value.** Estimate an opponent's remaining safe tiles and
   escape routes after a bomb. This is a later combat feature because it
   requires time-expanded simulation.
10. **Global target summaries.** Add coin/crate counts by direction, reachable
    targets by distance range, and unexplored regions. These are cheaper than a
    board tensor but remain compressed summaries.
11. **Full-board spatial channels.** Encode walls, crates, coins, the controlled
    agent, opponents, bombs/timers, explosions, and predicted danger for a small
    CNN, with scalar history and remaining-time inputs alongside it. This has
    the greatest information capacity but also the highest training cost; the
    imported results show that a spatial network without a matching curriculum
    and safety design can still fail.
12. **Absolute coordinates or border distances.** These can break some local
    aliases, but they risk memorizing fixed board layouts and are lower priority
    than route-preserving features.

The first controlled representation should therefore start from the current
agent, remove the five exact redundancies, replace the coin and crate
direction/distance triplets with action-conditioned route costs, and add
remaining time. It should pass the aliasing regression test before any long
training run. Only one later change should be introduced at a time:
multi-target routing, survival margins, opponent reachability, and finally a
spatial CNN. All feature extraction must use the same action-time state and
history in training, replay, and evaluation.

The first route-aware successor is now implemented as
[`Agent_022_combat_ddqn_route_agent`](../../bomberman_rl/agent_code/Agent_022_combat_ddqn_route_agent/).
Its 52-input representation, aliasing regression, and planned matched training
protocol are recorded in [`combat_ddqn_route_agent.md`](combat_ddqn_route_agent.md).
