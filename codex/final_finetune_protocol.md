# Final_finetune curriculum

`Final_finetune` is the focused end-stage curriculum for Agent 037. It keeps
the existing 300-round solo foundation, then uses a 3:1:1 schedule:

- three complete four-player Classic games;
- one Coin Heaven solo game; and
- one Loot Crate solo game.

Thus 60% of post-foundation experience is combat and every non-solo episode has
exactly three opponents. The runner rejects incomplete lineups and rejects
`coin_collector_agent` and `peaceful_agent` from Final_finetune training
lineups. Rule-based agents remain the historical control; the staging launcher
adds the strongest verified imported pair, `imp_li_deep_killer` and
`imp_alii_arbiter`, by default.

The implementation is in `src/run_combat_training.py`; the imported-opponent
launcher is `eval_suite/run_final_finetune.py`. Imported source and checkpoints
are staged into a fresh scratch runtime and are not edited in place.

Example warm-start launch:

```bash
python eval_suite/run_final_finetune.py \
  --output /export/scratch/salitanl/agent037_final_finetune_20260921 \
  --rounds 1500 --eval-every 150 \
  --eval-workers 8 --eval-scenario-workers 8 \
  --parallel-seeds \
  --initial-checkpoints \
    /export/scratch/salitanl/agent037_tournament_fast_1200_20260921_v4/seed_0/checkpoints/episode_1200.pkl \
    /export/scratch/salitanl/agent037_tournament_fast_1200_20260921_v4/seed_1/checkpoints/episode_1200.pkl \
    /export/scratch/salitanl/agent037_tournament_fast_1200_20260921_v4/seed_2/checkpoints/episode_1200.pkl
```

Use `--imported-opponents` with no values for a rule-based-only fine-tune.
The held-out imported policies (`imp_li_sarsa_lambda`, `imp_li_double_q`,
`imp_alii_sentinel`, `imp_alii_overlord`, and `harvy`) should remain reserved
for generalization evaluation unless a new experiment identity is created.

The schedule contract is covered by
`test_final_finetune_curriculum.py`; the staging and one-round imported-agent
smoke both passed on 21 September 2026. No long Final_finetune run is launched
by merely adding this curriculum.
