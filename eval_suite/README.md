# Pre-combat evaluation suite

## Imported external-team agents

External team agents used for reference comparisons are kept separately in
`../imported_agents/` and excluded from the project repository via
`.gitignore`. This currently includes `harvy` and `ruehl_based_agent`. Their
checkpoints and evaluation results are reference artifacts only; they are not
project-owned submission agents.

Sahand was here.

This suite is the gate to run before introducing `classic` games with
opponents. It evaluates frozen checkpoints under one fixed protocol so that
the current FQI control and the repaired vanilla DQN can be compared fairly.

## Protocol

- Scenarios: `coin-heaven` and `loot-crate`
- Training horizon represented by the checkpoint: 400 steps per game
- Held-out board seeds: `30000`--`30007`
- Starting seats: `0`, `1`, `2`, and `3`
- Agent decision seed: `0`
- Three independently trained checkpoints per candidate (training seeds 0, 1,
  and 2)
- 32 games per trained checkpoint and scenario; 96 games per candidate and
  scenario in total
- No opponents
- Metrics: coins, score, crates, bombs, kills, survival, invalid actions, and
  coin-heaven completion

The suite intentionally keeps the two scenarios separate. Coin Heaven tests
the navigation regression, while Loot Crate tests whether the candidate still
handles crates and bombs safely. A model should not be promoted to opponent
training from loot-crate score alone.

Fixed, non-training agents such as Harvy may provide one checkpoint path; the
suite evaluates that checkpoint once per scenario rather than treating it as
three independent training seeds.

## Run

From the experiment repository:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 eval_suite/run_suite.py
```

The script calls the existing `src/run_benchmark.py`; it does not retrain any
agent. Results are written below `eval_suite/results/<timestamp>/`, with one
directory per candidate/checkpoint/scenario and a combined `suite_summary.json`.

To evaluate a different checkpoint, edit `eval_suite/suite.json` and retain the
same board seeds, seats, scenario, and horizon. Do not compare results from a
different protocol without recording that change in the experiment registry.

`latest_fixed_dqn.json` is a follow-up manifest using fresh board seeds
`31000`--`31007`; it is intended to check whether the repaired DQN result
generalizes beyond the comparison suite's boards.

`classic_latest_dqn.json` evaluates the same three repaired-DQN checkpoints
against the supplied `peaceful_agent`, `coin_collector_agent`, and
`rule_based_agent` on fresh classic boards (`32000`--`32007`).

## Promotion gate

Before moving to `classic` opponents, inspect both scenarios and all safety
metrics. In particular, require zero invalid actions and zero self-deaths in
the candidate evaluation, then select a checkpoint using both navigation and
crate performance. This suite is a pre-combat gate, not a claim that the
candidate is already competitive against opponents.
