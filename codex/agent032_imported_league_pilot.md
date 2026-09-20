# Agent 032 one-learner imported-opponent league pilot

Status: implementation and short end-to-end smoke passed on 2026-09-20;
paired 600-round, two-seed CPU jobs launched after a matched device timing
check. Results are **pending** until both jobs finish. Neither this note nor
the source change claims a performance gain.

## Design

`league-combat` retains the established 100-round navigation and 200-round
crate warm-up. From round 301 onward it schedules three Classic games, one
Coin Heaven game, and one Loot Crate game per five rounds. The Classic games
use a seeded shuffled cycle over 24 frozen opponent lineups: eight each with
one, two, and three opponents. There is exactly one learning Agent 032 in
every game; there is no simultaneous multi-learner self-play yet. Replay
weights are 60% Classic and 20% for each solo task, divided among lineup
tags. Solo rounds continue to be generated, preventing replay quotas from
referring only to old, eventually evicted transitions.

League roster: `peaceful_agent`, `coin_collector_agent`, `rule_based_agent`,
plus imported `imp_li_deep_killer`, `imp_li_sarsa_lambda`, `imp_li_double_q`,
`imp_alii_arbiter`, and `imp_alii_sentinel`. The control uses the same schedule
with only the three built-ins in its training roster. Both evaluate on the
same held-out boards 32000–32003, all four seats, Coin Heaven, Loot Crate,
three rule-based opponents, and a mixed lineup of rule-based, Deep Killer,
and Arbiter. Evaluation is checkpoint-frozen and CPU-isolated. Agent 032
training uses CUDA hidden and one math-library thread per process. Seeds 0 and
1 run concurrently, as do the league and control. During checkpoint evaluation,
each seed evaluates its four scenarios concurrently with two game workers per
scenario. This peaks near 32 game workers across the four learners on the
72-logical-CPU host.

Imported sources are staged under scratch, never edited in their original
directories. The one staged exception is Sentinel's callback, corrected to
unwrap the `q_net` weights in its frozen `best.pt`; the original source and
checkpoint are unchanged. Each run's `provenance.json` records checkpoint
hashes, roster, lineups, source commit, and the exact command.

## Runs and verification

- Parallel CPU league job: `agent032-league-cpu-parallel-20260920`, output
  `/export/scratch/salitanl/bomberman_agent032_league_cpu_parallel_20260920/`.
- Parallel CPU built-in control job: `agent032-builtin-cpu-parallel-20260920`,
  output
  `/export/scratch/salitanl/bomberman_agent032_builtin_cpu_parallel_20260920/`.
- The earlier sequential CPU jobs were stopped and replaced at approximately
  episode 100 of league seed 0. They are incomplete and must not be used as
  performance results; their scratch artifacts remain.
- Previous GPU league job `agent032-league-pilot-20260920` was stopped at
  approximately round 56 of seed 0. Its queued GPU control was also stopped;
  neither is a completed performance result. The old scratch artifacts remain.
- Schedule unit tests: `test_league_curriculum.py`, three checks passing.
- Staging check: `/export/scratch/salitanl/bomberman_league_stage_smoke_20260920/`.
- End-to-end runner check:
  `/export/scratch/salitanl/bomberman_league_runner_smoke_20260920/`.
  With a deliberately one-step horizon, it completed 301 rounds and logged
  the first Classic training game versus Deep Killer, Arbiter, and corrected
  Sentinel. This checks wiring, not learning or performance.

The implementation lives in `run_combat_training.py` and
`eval_suite/run_imported_league.py`. Earlier agent implementations and
checkpoints were not changed. After both jobs succeed, compare checkpoint
scores, kills, survival, suicides, solo retention, and per-lineup replay
counts at equal rounds and environment steps. Do not promote a new default
solely from training-round scores.

## CPU versus GPU decision

The timing driver `eval_suite/benchmark_league_device.py` ran the same 30
league rounds (301–330) from the same seed-0 episode-0 checkpoint, with
evaluation skipped but real games, imported opponents, replay, and optimizer
updates included. Its outputs are under
`/export/scratch/salitanl/bomberman_agent032_league_device_20260920/`.

| Device | Game/training time | Steps | Time per step | Wall time |
| --- | ---: | ---: | ---: | ---: |
| CPU | 160.93 s | 11,215 | **14.35 ms** | 167.68 s |
| RTX 2080 Ti | 189.63 s | 11,561 | 16.40 ms | 196.42 s |

The devices followed the same lineup sequence but differed slightly in
learner trajectories and step totals. Normalize by steps: the GPU run was
14.3% slower. This agrees with the earlier smaller non-league timing check.
The active GPU training job was therefore stopped and replaced by a fresh
CPU run; no mid-run replay-buffer migration was attempted.
