# Tree FQI with learned movement history

## Accepted implementation scope

The user approved implementing a simple tree-FQI history agent after discussing
the distinction between learned history inputs and the hand-written loop wrapper.
`src/agent_code/tree_fqi_history_agent/` is a separate three-file agent.

It retains the distance-feature agent's local blocked-neighbour flags, nearest
coin direction signs, distance bucket, remaining-coin bucket, action set,
legal-action mask, fitted-Q updates, rewards, discount, exploration schedule,
buffer, tree settings, and terminal handling. It adds only:

- the action selected at the preceding decision; and
- a count from 0 to 3 of visits to the present tile in the previous eight
  positions since the last coin was collected.

The feature code does not choose an action. Fitted Q iteration still learns the
action values from transitions and rewards. During training, the agent stores a
feature snapshot before acting and constructs the next-state history inputs from
the executed action and resulting state, so Bellman updates use the same inputs
that the following decision observes.

This targets the observed return-to-loop failure: the tree can distinguish an
initial arrival at a tile from returning after UP, DOWN, LEFT, RIGHT, or WAIT.
It does not include the wrapper's forced action substitution.

## Implementation validation

Completed two 10-step training rounds and frozen evaluation in
`experiments/tree_fqi_history_smoke/`. The buffer held 10 then 20 transitions;
rewards matched coins minus the step cost; and the saved checkpoint contains
five fitted action trees. A direct state check confirms that revisiting a tile
after UP changes the final two inputs from `(WAIT, 0 visits)` to `(UP, 1 visit)`.
Training constants exactly match the original tree-FQI agent.

This validates the implementation only. No comparative learning pilot or fresh
evaluation has been run.

## Measured training pilot

Completed the matched three-seed, 300-round coin-heaven pilot. Evaluations at
rounds 0, 100, 200, and 300 use the same eight development boards, action seed
0, and all four corners as the original tree-FQI pilot. Artifacts:
`experiments/tree_fqi_history_pilot/`.

| Training rounds | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| 0 | 9.66 | 9.66 | 9.66 | 9.66 |
| 100 | 21.28 | 21.47 | 21.44 | 21.40 |
| 200 | 29.88 | 27.22 | 28.62 | 28.57 |
| 300 | 34.09 | 34.06 | 39.12 | **35.76** |

The original tree-FQI pilot reached 23.52 at round 300. All history-policy
seeds improved over that mean on the reused development boards. At final-round
evaluation, repeated states averaged 18.88 per game across seeds, compared
with 50.66 for base tree FQI; both variants had zero invalid actions. Neither
policy completed all coin collection within the 100-step training evaluation.

All 900 round records were checked: 100 steps each, correct transition-buffer
growth, rewards equal coins minus step cost, and finite frozen predictions. The
three saved checkpoints and four evaluation schedules per run are present.

## Fresh-board frozen comparison

Compared the three round-300 history checkpoints with their base tree-FQI
counterparts on 16 new boards (22016--22031), action seed 0, all four corners,
and 100- and 400-step limits. No policies were trained during this evaluation.
All 768 game records and frozen checkpoint hashes were verified. Artifacts:
`experiments/tree_fqi_history_fresh_confirmation/`.

| Budget | Base tree FQI | Learned history | Difference |
|---|---:|---:|---:|
| 100 steps | 25.13 coins | **36.26** | +11.13 |
| 400 steps | 25.17 coins | **41.45** | +16.28 |

At 100 steps, all three matched checkpoints improved (gains 13.19, 8.45,
11.75); the board-paired bootstrap 95% interval is [9.31, 12.95], conditional
on these six frozen checkpoints. At 400 steps, history completed 64/192 games
(33.3%), versus 1/192 for base tree FQI. Mean repeated states were 17.35 versus
45.71 at 100 steps and 214.97 versus 344.13 at 400 steps. Invalid actions were
zero for both.

The fresh evaluation supports a meaningful benefit from learning with recent
movement inputs. This is stronger evidence than the diagnostic wrapper because
the history-augmented values were learned through fitted Q iteration and the
gain persists on new boards. Limits: only three training runs, a 100-step
terminal training cutoff, the same underlying tree method, and no evidence yet
on crates, bombs, or opponents. The board bootstrap is conditional on fixed
checkpoints; it is not uncertainty over new trained models.

## Decision

Retain the learned-history tree agent as the leading Stage 1 navigation model.
Keep its frozen checkpoints and report both the positive development pilot and
fresh-board comparison. Before advancing stages, decide whether 41.45 mean
coins and a 33.3% all-coins completion rate within 400 steps are adequate for
the team's Stage 1 exit criterion. The policy has improved, but it still does
not finish most fresh games.

## Training-runner speed improvement

Before testing 400-step training, changed `src/run_training.py` so each frozen
checkpoint evaluation sends all board/corner tasks to one benchmark worker
batch. Previously the generic runner launched one Python process per game (32
processes for the usual eight boards and four corners). This changes only
evaluation orchestration; each task retains its board seed, agent seed, seat,
step budget, and frozen checkpoint.

On eight fresh 100-step games, the single-game and batched paths produced
identical per-game records after excluding elapsed-time measurements and
identical aggregate summaries. Evaluation wall time fell from 12.43 seconds to
2.27 seconds, a 5.5x speedup in that check. A two-round training-driver smoke
test also completed both checkpoint evaluations, reloaded the model, retained
20 transitions, and reported zero invalid actions. Tree fitting during training
is unchanged, preserving the current learning schedule.
