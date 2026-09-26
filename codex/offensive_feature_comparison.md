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

The historical `src/run_offensive_feature_comparison.py` entrypoint prepared
migrations and manifests, ran either training command, and evaluated and
aggregated after both training jobs succeeded. The driver is no longer in the
cleaned source checkout. The job IDs were `agent030-callback-fixed-train`,
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

## CPU versus GPU timing diagnostic — 20 September 2026

A direct device comparison of Agent 031 used the now-removed
`src/benchmark_combat_device.py` helper,
the original seed-0 113-input episode-600 checkpoint, and episodes 601–630 of
the same 60/20/20 curriculum. Both processes ran on compgpu5 with one math-library
thread; CUDA was hidden for the CPU baseline and physical GPU 0 (RTX 2080 Ti)
was isolated for CUDA. GPU availability and actual CUDA execution were checked
before the job. PyTorch was 2.14.0+cu130. No production checkpoint was changed.
Evaluation was skipped for this timing diagnostic; normal replay warmup,
batch size 128, update frequency, rewards, safety and per-round saves remained.

| Measurement | CPU | RTX 2080 Ti |
| --- | ---: | ---: |
| Training time, 30 rounds | 37.196 s | 38.783 s |
| Including learner/world setup | 40.030 s | 41.748 s |
| Learner transitions | 7,816 | 7,816 |
| Optimizer updates | 704 | 704 |
| Training milliseconds per transition | 4.759 | 4.962 |
| Feature extraction | 21.457 s | 21.483 s |
| Optimizer calls, including replay sampling | 1.956 s | 2.890 s |
| Median isolated update time | 2.114 ms | 3.325 ms |

All 30 paired rounds matched on board, scenario, steps, score, coins, crates,
kills, suicides, bombs, invalid moves, survival, replay counts and update counts.
This is an observed short-run match, not a general guarantee of CPU/CUDA
numerical equivalence. The isolated update diagnostic reset the original
checkpoint and used identical seeded synthetic replay, 30 warmup updates, then
three groups of 200 timed updates using the real optimizer implementation.
It measures replay sampling, transfers and synchronized loss retrieval too.

CUDA took 4.3% longer in this one paired timing run; the small elapsed-time
difference should not be treated as a precise general speed ratio. There is
no observed speed benefit from directly moving this implementation to GPU.
CPU feature extraction consumed 57.7% of training time, while optimizer calls
consumed 5.3% (this 30-round run includes the 5,000-transition replay warmup).
The tiny 113–128–128–6 network and serial decision/update path do not provide
enough work to offset CUDA overhead in this test. CPU feature/safety work is
the first performance target; larger batches or parallel actors would be a
separate training-system experiment, not a device-only switch.

Artifacts: `/export/scratch/salitanl/bomberman_device_timing_20260920/{cpu,cuda}/`
contains `timing.json`, config, round logs and disposable timing checkpoints.
Job: `agent031-device-timing` (succeeded). The GPU process exited after timing.
The JSON records the source commit, benchmark hash and initial checkpoint hash.

The commands below record the original procedure. The helper is no longer in
the cleaned source checkout, so these commands are historical rather than
runnable there. The saved timing artifacts remain available. The original run
used a fresh output path for each device and checked GPU ownership first:

```bash
cd /export/home/salitanl/projects/ml-project/bomberman_rl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SDL_AUDIODRIVER=dummy SDL_VIDEODRIVER=dummy
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_combat_device.py --device cuda --rounds 30 --config /export/scratch/salitanl/bomberman_feature_variants_20260920/offensive_feature_matched_v1/candidate113/seed_0/config.json --output /export/scratch/salitanl/bomberman_device_timing_repeat/cuda
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmark_combat_device.py --device cpu --rounds 30 --config /export/scratch/salitanl/bomberman_feature_variants_20260920/offensive_feature_matched_v1/candidate113/seed_0/config.json --output /export/scratch/salitanl/bomberman_device_timing_repeat/cpu
```

Use jobctl for detached execution; keep the timing artifacts under scratch,
not `/tmp`. This diagnostic does not launch a full training campaign.

## Agent 032 optimized feature implementation — 20 September 2026

Agent 032 is a separate agent directory. It keeps Agent 031's 113 inputs and
the corrected Agent 030 training callbacks. Earlier agent directories are not
modified. The optimization is confined to the feature module:

- the opponent reachability envelope is computed once and reused across all
  six action summaries;
- action legality is computed once per state instead of once per response
  search;
- the opponent-window search uses local board bounds and hot-loop bindings;
- the offensive survivor search uses the same state semantics with fewer
  helper calls, repeated indexing operations and temporary allocations.

The exact feature contract passed against Agent 031 on empty, armed,
multi-opponent, crate, bomb and lingering-explosion states. The test covered
the full 113-value vector and required exact equality. A feature-only benchmark
of 600 identical calls measured 4.501 seconds for Agent 031 and 2.192 seconds
for Agent 032: 2.05× throughput, or 51.3% less feature time. This is a
microbenchmark, so it does not predict the same end-to-end training speedup;
the environment, safety mask and rule-based opponents remain separate costs.

The training callback smoke test loaded a migrated episode-600 checkpoint,
ran one continuation round, wrote the episode-601 checkpoint and completed a
real evaluation game. Job `agent032-feature-smoke` succeeded. A matched
multi-seed training/evaluation run is still required before judging whether
the faster implementation preserves learning behavior over long runs.

## Agent 032 replay storage and transfer optimization — 20 September 2026

Agent 032 now keeps its own replay implementation; Agents 027–031 are
unchanged. The circular store uses fixed NumPy arrays for states, next states,
actions, rewards, terminal flags and next-action masks instead of allocating a
new tuple and several arrays for every transition. Scenario tags and the
Agent030 80/20 combat-escape sampling quota are preserved.

On sampling, reusable host staging arrays are filled with `np.take`. CUDA
staging is pinned, and reusable device tensors receive the six batch fields
with non-blocking copies. The optimizer consumes those tensors directly, so it
does not create six new `as_tensor(..., device=...)` objects on each update.
The public NumPy `sample()` API remains available for compatibility.

The isolated benchmark used 10,000 stored 113-input transitions and 300
128-transition batches. CPU sampling plus tensor construction fell from
1.124 ms to 0.353 ms per batch (3.19×) in the latest CPU pass. On an RTX
2080 Ti, CPU→GPU handoff fell from 1.161 ms to 0.417 ms (2.79×). These are replay/input-pipeline
measurements, not end-to-end training speedups; the small network and feature
extraction still dominate the complete run. Tests passed 12/12, including an
actual Agent032 optimizer step, and the real-game smoke job
`agent032-replay-smoke3` succeeded.

Artifacts and commands:

- source benchmark: the now-removed `src/benchmark_agent032_replay.py` helper;
- CPU result: run with `CUDA_VISIBLE_DEVICES= .venv/bin/python ... --device cpu`;
- GPU job: `agent032-replay-gpu-bench`, using physical GPU 0;
- smoke output: `/export/scratch/salitanl/bomberman_device_timing_20260920/agent032_replay_smoke3`.
