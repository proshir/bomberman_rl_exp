# Reusable combat-agent training and evaluation protocol

> **21 September 2026 update:** this file preserves the historical Agent 027
> workflow. Boards 32000--32007 and 33000--33007 have since influenced model
> and checkpoint decisions and are development boards, not a blind final
> test. For new DDQN runs, use the curriculum, board split, gates, and stopping
> rules in [`ddqn_training_regimen_20260921.md`](ddqn_training_regimen_20260921.md).

This is the copyable workflow for training a new combat agent, freezing its
three training-seed checkpoints, and evaluating them under the classic
tournament protocol used for Agent 027.

This document is the end-to-end runbook. The lower-level references are:

- [`../eval_suite/README.md`](../eval_suite/README.md) for the pre-combat
  Coin Heaven and Loot Crate suite;
- [`../tools/README.md`](../tools/README.md) for durable `jobctl` usage;
- [`experiment_registry.md`](experiment_registry.md) for experiment records.

Historical `mixed` and `staged-combat` results remain valid historical
artifacts. Do not silently relabel them as tournament-combat results.

## 1. Set the experiment identity

Run commands from the workspace root. Use a new scratch output directory for
each run; only use `--resume` when intentionally extending that exact run.

```bash
AGENT=Agent_029_combat_ddqn_adversarial_window_agent
TAG=agent029_tournament_1000
OUT=/export/scratch/$USER/$TAG
EPISODE=1000
TRAIN_JOB=$TAG-train
EVAL_JOB=$TAG-classic
MANIFEST=/export/scratch/$USER/${TAG}_classic_manifest.json
EVAL_OUT=/export/scratch/$USER/${TAG}_classic_eval
```

The implementation must contain:

```text
bomberman_rl_exp/src/agent_code/$AGENT/callbacks.py
```

Use training seeds `0 1 2`. In the frozen suite, these are three independent
candidate checkpoints, not three evaluation games.

## 2. Training distribution

Use `--curriculum tournament-combat`. Its schedule is intentionally
tournament-aligned:

- episodes 1--100: mostly Coin Heaven with some Loot Crate;
- episodes 101--300: mostly Loot Crate with some Coin Heaven;
- episodes 301--600: Classic rehearsal plus the two solo tasks;
- episodes 601 onward: primarily the three-rule-based-opponent Classic lineup,
  with a retained solo replay share.

Classic training lineups are one `peaceful_agent`, one
`coin_collector_agent`, one `rule_based_agent`, and three
`rule_based_agent` instances. Solo tasks are regression rehearsal; the
three-opponent Classic lineup is the tournament-relevant part.

The runner tags replay by complete lineup so multi-opponent experience is not
diluted by easier Classic games. The schedule is recorded in `config.json` and
`rounds.jsonl`.

## 3. Train all three seeds

The standard 1000-round command is:

```bash
bomberman_rl_exp/jobctl start --id "$TRAIN_JOB" \
  --progress-file "$OUT/learning_curve.json" -- \
  python3 bomberman_rl_exp/src/run_combat_training.py \
    --agent "$AGENT" \
    --curriculum tournament-combat \
    --rounds "$EPISODE" \
    --seeds 0 1 2 \
    --parallel-seeds \
    --eval-every 100 \
    --eval-workers 8 \
    --eval-scenario-workers 4 \
    --output "$OUT"
```

The training runner automatically evaluates saved checkpoints on the solo
tasks and configured Classic lineups. The worker flags affect wall-clock
scheduling only; they do not change the learner trajectory or checkpoint
values. Reduce them on a smaller host.

For a 600-round staged-combat control, use the same command with:

```text
--curriculum staged-combat --rounds 600
```

That is a different training distribution, not a pure 600-versus-1000
training-length comparison. Record code version and curriculum before making
that comparison.

Monitor the detached job with:

```bash
bomberman_rl_exp/jobctl status "$TRAIN_JOB" --tail 20
bomberman_rl_exp/jobctl tail -f "$TRAIN_JOB"
bomberman_rl_exp/jobctl ps "$TRAIN_JOB"
```

Before evaluation, require a successful job and verify all final checkpoints:

```bash
bomberman_rl_exp/jobctl status "$TRAIN_JOB"
for seed in 0 1 2; do
  test -s "$OUT/seed_${seed}/checkpoints/episode_$(printf '%04d' "$EPISODE").pkl"
done
```

Retain `$OUT/config.json`, which records agent, curriculum, hyperparameters,
code version, source hashes, seeds, and output location. Also retain
`learning_curve.json`, `rounds.jsonl`, and the per-seed directories.

## 4. Frozen classic-tournament protocol

Create one manifest per candidate. The fixed comparison uses:

- board seeds `32000` through `32007`;
- agent decision seed `0`;
- starting seats `0`, `1`, `2`, and `3`;
- Classic scenario with a 400-step horizon;
- `three_rule_based`: three `rule_based_agent` opponents;
- `mixed_strong`: `rule_based_agent`, `coin_collector_agent`, and
  `peaceful_agent`.

Each checkpoint plays 32 games per lineup. Three checkpoints therefore play
96 games per lineup per candidate.

Generate a manifest from the training output:

```bash
python3 - "$MANIFEST" "$OUT" "$AGENT" "$TAG" "$EPISODE" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, out_dir, agent, tag, episode = sys.argv[1:]
episode = int(episode)
out = Path(out_dir)

manifest = {
    "board_seeds": list(range(32000, 32008)),
    "agent_seeds": [0],
    "seats": [0, 1, 2, 3],
    "scenarios": ["classic"],
    "max_steps": 400,
    "diagnostics": True,
    "opponent_lineups": [
        {"name": "three_rule_based", "scenario": "classic",
         "opponents": ["rule_based_agent"] * 3},
        {"name": "mixed_strong", "scenario": "classic",
         "opponents": ["rule_based_agent", "coin_collector_agent",
                        "peaceful_agent"]},
    ],
    "candidates": [{
        "name": tag,
        "agent": agent,
        "checkpoints": [
            str(out / f"seed_{seed}" / "checkpoints" /
                f"episode_{episode:04d}.pkl")
            for seed in (0, 1, 2)
        ],
    }],
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
print(manifest_path)
PY
```

Do not silently change board seeds, seats, lineups, horizon, or diagnostics
when comparing agents. Record any protocol change in `experiment_registry.md`.

## 5. Run the frozen evaluation

Queue evaluation behind successful training:

```bash
bomberman_rl_exp/jobctl start --id "$EVAL_JOB" --after "$TRAIN_JOB" -- \
  python3 bomberman_rl_exp/eval_suite/run_suite.py \
    --manifest "$MANIFEST" \
    --output "$EVAL_OUT" \
    --parallel 3 \
    --game-workers 8
```

`--parallel` controls independent checkpoint/lineup entries and
`--game-workers` controls games inside each entry. Reduce either value if the
host has limited CPU or memory. Evaluation is read-only and does not retrain
the candidate.

For CPU-only evaluation on a shared host, hide CUDA and cap math-library
threads before launching the job:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The result directory contains:

```text
$EVAL_OUT/manifest.json
$EVAL_OUT/suite_summary.json
$EVAL_OUT/<candidate>_<lineup>_<checkpoint>/summary.json
```

The training job and evaluation job can be inspected with `jobctl status`,
`jobctl tail -f`, and `jobctl ps`. Keep their IDs with the experiment record.

## 6. Metrics and promotion decision

Read `suite_summary.json` and retain every checkpoint summary. Report all three
training seeds separately and as a mean for each lineup. Primary metrics are:

- mean score, round-win rate, and mean rank;
- coins, crates, bombs, and kills;
- survival rate and self-deaths (`mean_suicides`);
- invalid actions and steps.

The promotion gate is safety-first:

1. require zero invalid actions and zero self-deaths in the solo gates;
2. among candidates that pass, select by pooled tournament score;
3. reject a score winner that regresses solo completion or survival materially;
4. use confidence intervals and all three training seeds, not one favorable
   checkpoint.

Training reward and solo coin collection are diagnostic evidence, not ship
criteria. A classic tournament result does not replace the pre-combat
Coin Heaven and Loot Crate suite in `../eval_suite/README.md`.

## 7. Provenance checklist

Save or link all of the following in the experiment note or registry entry:

- exact training command and `jobctl` IDs;
- training `config.json`, code version, and source hashes;
- all three checkpoint paths;
- the exact frozen manifest;
- `suite_summary.json` and per-run summaries;
- learning curve and final evaluation metrics;
- every changed curriculum, opponent lineup, board seed, seat set, horizon,
  worker budget, or code version.

This prevents a faster or newer run from being mistaken for an isolated
algorithm change.
