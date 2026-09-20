# Tournament-aligned combat protocol

This is the accepted protocol for the next combat experiment. Existing
`mixed` and `staged-combat` runs are historical results and remain unchanged.

## Training distribution

Use `run_combat_training.py --curriculum tournament-combat`. Episodes 1--100
are 70% Coin Heaven and 30% Loot Crate; episodes 101--300 are 25%/75%; from
episodes 301--600, half of the episodes are Classic and half retain the two
solo tasks. Classic episodes rotate these lineups:

1. one `peaceful_agent`;
2. one `coin_collector_agent`;
3. one `rule_based_agent`;
4. three `rule_based_agent` instances;

The 3-opponent lineups are the tournament-relevant part. From episode 601
onward, the default schedule consolidates on 3x `rule_based_agent` with a
small 25% retained-solo replay share, matching the final competition stage in
Ali Khaled's curriculum. Solo episodes are regression rehearsal, not the final
objective.

Replay is tagged by the complete lineup (`classic|rule_based_agent,...`) so
rule-based and multi-opponent experience cannot be diluted by easier Classic
games. The schedule and replay proportions are written to each run's
`config.json` and `rounds.jsonl`.

Example:

```bash
python3 src/run_combat_training.py \
  --agent Agent_029_combat_ddqn_adversarial_window_agent \
  --curriculum tournament-combat --rounds 1000 --seeds 0 1 2 \
  --parallel-seeds --output experiments/agent029_tournament_1000
```

The runner defaults to eight CPU game workers and four concurrent frozen
scenario evaluations per seed. Evaluation is read-only with respect to the
learner, so this parallel scheduling preserves the training trajectory and
metrics while reducing evaluation wall time. Use the explicit worker flags to
lower the process budget on smaller hosts.

Do not compare this run directly with the old 600-round Agent 027 result
without recording the changed lineup distribution and budget.

## Frozen gates

For every candidate checkpoint, the training runner now automatically runs the
solo gate on Coin Heaven and Loot Crate plus each configured Classic lineup.
The same frozen protocol is also available as
`eval_suite/classic_tournament_template.json` for independent checkpoint
comparisons. The primary combat metrics are
mean score, round-win rate, mean rank, survival, self-deaths, kills, and
invalid actions. A candidate is not promoted from a training curve alone.

Checkpoint selection is predeclared: first require zero invalid actions and
zero self-deaths in the solo gate; among survivors, choose the checkpoint with
the best pooled tournament score subject to non-regressing solo completion and
survival. Report all three training seeds and all four starting seats.

## Controlled comparisons

The first adversarial run should compare Agent 029 and an unchanged Agent 027
control under the identical tournament curriculum, seeds, and checkpoints.
Only after that comparison should opponent-response features, rewards, or
network architecture be changed. Training reward and solo coins are
diagnostic metrics, not ship criteria.
