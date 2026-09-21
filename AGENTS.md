# Bomberman RL project instructions

## Repository layout and working rules

- `src/` contains the Bomberman reinforcement-learning implementation. Keep
  project code and training/evaluation scripts there.
- `experiments/` contains experiment configurations, outputs, metrics, plots,
  and checkpoint references. Keep reusable scripts in `src/`.
- `codex/` contains durable project context: the specification, plans,
  decisions, implementation notes, experiment results, and diagnoses. Read the
  relevant note before continuing earlier work, and update the existing note
  when its conclusions or status change.
- `eval_suite/` contains the fixed evaluation protocol and runner. Preserve its
  board seeds, scenarios, horizons, and metrics when making comparisons; record
  any protocol change in `codex/experiment_registry.md`.
- Consult `codex/final_project.md` for the assignment requirements before
  planning or implementing project work.
- Clearly distinguish proposed ideas, accepted decisions, implementation
  status, and measured results in saved notes. Keep claims tied to the stated
  protocol and artifacts.
- The user leads substantive project decisions. Treat the roadmap as context,
  not permission to implement the whole project or launch substantial training.
  Implement the requested or explicitly approved scope; ask before changing
  the model, experimental strategy, or implementation stage.

## Agent naming and numbering directive

- The canonical implementation-agent directory is `src/agent_code/`. In this
  experiment repository, `src/` is linked to the source repository; the
  top-level `agent_code/` directory contains run logs and is not the source
  agent registry.
- Do not rename or move any existing agent directory. The enumeration below is
  a reference ID only and is ordered by the first Git commit that introduced
  each currently present directory. When multiple agents were introduced in
  the same commit, use alphabetical order for the tie.
- Every new agent directory must put its number first, using the format
  `Agent_NNN_<descriptive_name>` (for example, `Agent_022_new_agent`). Consume
  the next number exactly once and preserve the number permanently.
- Keep this registry and the chronological account in
  `codex/implementation_roadmap.md` synchronized when a new agent is added.

Current agent enumeration, with existing directory names preserved:

| Number | Existing directory | First introduced |
| --- | --- | --- |
| `Agent_000` | `random_agent` | 2019-01-30, `8fc573a` |
| `Agent_001` | `user_agent` | 2019-01-30, `8fc573a` |
| `Agent_002` | `peaceful_agent` | 2021-02-12, `0c27846` |
| `Agent_003` | `rule_based_agent` | 2021-02-12, `0c27846` |
| `Agent_004` | `tpl_agent` | 2021-02-12, `0c27846` |
| `Agent_005` | `fail_agent` | 2021-03-16, `2d49d83` |
| `Agent_006` | `coin_collector_agent` | 2022-02-17, `918e9b3` |
| `Agent_007` | `q_table_agent` | 2026-09-10, `df39ed9` |
| `Agent_008` | `q_table_masked_agent` | 2026-09-11, `3d8ee79` |
| `Agent_009` | `linear_sarsa_agent` | 2026-09-11, `bf4dcbf` |
| `Agent_010` | `linear_sarsa_loop_agent` | 2026-09-11, `7df0ec1` |
| `Agent_011` | `q_table_loop_agent` | 2026-09-11, `7df0ec1` |
| `Agent_012` | `tree_fqi_agent` | 2026-09-17, `8153877` |
| `Agent_013` | `tree_fqi_loop_agent` | 2026-09-17, `219c530` |
| `Agent_014` | `tree_fqi_history_agent` | 2026-09-18, `fd63ea1` |
| `Agent_015` | `combat_fqi_agent` | 2026-09-18, `9d1c3bb` |
| `Agent_016` | `combat_fqi_history_antistag_agent` | 2026-09-18, `3798be9` |
| `Agent_017` | `dqn_coin_agent` | 2026-09-18, `3798be9` |
| `Agent_018` | `combat_fqi_history_antistag_topology_agent` | 2026-09-18, `8c367bb` |
| `Agent_019` | `combat_dqn_agent` | 2026-09-18, `3531cf5` |
| `Agent_020` | `combat_dqn_topology_agent` | 2026-09-18, `71b0f5a` |
| `Agent_021` | `combat_dqn_r_topology_agent` | 2026-09-18, `0e7ac9e` |
| `Agent_022` | `Agent_022_combat_ddqn_route_agent` | 2026-09-19, `cd0f205` |
| `Agent_023` | `Agent_023_spatial_hybrid_rainbow_agent` | 2026-09-19, `cd0f205` |
| `Agent_024` | `Agent_024_combat_ddqn_action_safety_agent` | 2026-09-19, `b19ba77` |
| `Agent_025` | `Agent_025_combat_ddqn_short_cycle_agent` | 2026-09-19, `b19ba77` |
| `Agent_026` | `Agent_026_combat_ddqn_target_coverage_agent` | 2026-09-19, `b19ba77` |
| `Agent_027` | `Agent_027_combat_ddqn_short_cycle_staged_replay_agent` | 2026-09-19, `1f65c49` |
| `Agent_028` | `Agent_028_combat_ddqn_dueling_256_agent` | 2026-09-19, `c867b7e` |
| `Agent_029` | `Agent_029_combat_ddqn_adversarial_window_agent` | 2026-09-19, uncommitted implementation |

| `Agent_030` | `Agent_030_combat_ddqn_escape_replay_agent` | 2026-09-20, `a3c42ec`; callback-repaired matched control completed |
| `Agent_031` | `Agent_031_combat_ddqn_offensive_escape_agent` | 2026-09-20, `a3c42ec`; matched evaluation completed; not promoted |
| `Agent_032` | `Agent_032_combat_ddqn_optimized_features_agent` | 2026-09-20, separate optimized 113-input feature implementation; smoke-tested |
| `Agent_033` | `Agent_033_population_replay_agent` and `Agent_033_population_frozen_agent` | 2026-09-21, pre-existing population variants; shared number requires later registry cleanup |
| `Agent_034` | `Agent_034_compact_fqi_agent` | 2026-09-21, compact route-aware tree-FQI; initial pilot invalidated by same-step replay cache collision, schema-v2 repair validated but not retrained |
| `Agent_035` | `Agent_035_compact_fqi_symmetry_agent` | 2026-09-21, corrected standalone Agent 034 successor with D4 replay augmentation and resumable training state; smoke-tested, not trained |
| `Agent_036` | `Agent_036_compact_fqi_robust_agent` | 2026-09-21, robust compact FQI successor with decision-time and safety repairs |
| `Agent_037` | `Agent_037_tournament_fast_ddqn_agent` | 2026-09-21, focused 101-input tournament DDQN continuation |
| `Agent_038` | `Agent_038_symmetric_population_ddqn_agent` | 2026-09-21, 117-input route extension with D4 replay and population curriculum |
| `Agent_039` | `Agent_039_compact_audit_ddqn_agent` | 2026-09-21, approved 104-input audit compression of Agent 038; implementation and contract tests complete |
| `Agent_040` | `Agent_040_optimized_compact_ddqn_agent` | 2026-09-21, direct/cached 104-input Agent 039 successor; contract-tested, training pending |
| `Agent_041` | `Agent_041_dynamic_nav_ddqn_agent` | 2026-09-21, Agent 040 successor with action-aligned coin navigation inputs; contract-tested |
| `Agent_042` | `Agent_042_combat_progress_ddqn_agent` | 2026-09-21, Agent 041 successor with action-aligned combat progress, agent-local persistent history, and separate solo/combat exploration; 300-round solo gate completed, stagnation partially mitigated but not solved |
| `Agent_043` | `Agent_043_novelty_credit_ddqn_agent` | 2026-09-21, Agent 042 successor with bounded novelty intervention, event-attributed progress, and three-step/potential-based credit assignment; 300-round solo gate passed and strict/20%-retention league branches active |

The next available number is `Agent_044`.

## Codex note index

Use the paths below to decide where to read or record information. Prefer
updating the most specific existing note instead of creating a duplicate.

### Project context and planning

- [`codex/final_project.md`](codex/final_project.md) — Transcription of the
  course assignment brief, requirements, deadline, and deliverables. Read this
  before making project-level plans or interpreting scope.
- [`codex/stage1_coin_navigation_plan.md`](codex/stage1_coin_navigation_plan.md)
  — Stage 1 coin-navigation objective, shared evaluation method, candidate
  families, and exit criteria. Update when the accepted Stage 1 protocol changes.
- [`codex/implementation_roadmap.md`](codex/implementation_roadmap.md) —
  Chronological algorithm progression, current findings, and proposed next
  steps across coin navigation and combat. Update accepted milestones and
  decisions, not every raw measurement.
- [`codex/experiment_registry.md`](codex/experiment_registry.md) — Index of
  experiment families, protocols, artifacts, and comparison status. Update when
  starting, completing, or qualifying an experiment.
- [`codex/algorithm_knowledge_base.md`](codex/algorithm_knowledge_base.md) —
  Durable reference for environment constraints, representations, safety
  mechanisms, algorithms, and their implementation/validation status. Update
  reusable technical knowledge or status labels here.

### Research and diagnosis

- [`codex/dqn_training_audit_20260918.md`](codex/dqn_training_audit_20260918.md)
  — Source-code audit explaining combat-DQN degradation, repaired correctness
  issues, and limits on interpreting earlier comparisons. Read before changing
  or evaluating combat DQN training.
- [`codex/anti_loop_research.md`](codex/anti_loop_research.md) — Research and
  project evidence about RL-compatible anti-loop methods, including the
  recommendation for a controlled novelty-history candidate.
- [`codex/tree_navigation_diagnosis.md`](codex/tree_navigation_diagnosis.md) —
  Replay-based diagnosis of coin-navigation cycles and feature aliasing in the
  original tree agent.
- [`codex/tree_fqi_history_loop_diagnosis.md`](codex/tree_fqi_history_loop_diagnosis.md)
  — Detailed replay diagnosis of remaining loops in the learned-history tree
  agent, including cycle patterns and action-ranking evidence.

### Coin-navigation implementations and pilots

- [`codex/q_table_pilot.md`](codex/q_table_pilot.md) — First masked tabular
  Q-learning pilot: protocol, measured results, validation, and behavior
  diagnosis.
- [`codex/linear_sarsa_pilot.md`](codex/linear_sarsa_pilot.md) — Approved
  linear-SARSA coin-navigation pilot, results, and reproduction checks.
- [`codex/dqn_coin_pilot.md`](codex/dqn_coin_pilot.md) — Compact neural DQN
  coin-navigation pilot, its training/evaluation protocol, and measured results.
- [`codex/tree_fqi_agent.md`](codex/tree_fqi_agent.md) — Implementation and
  validation record for the basic tree-based fitted-Q coin agent.
- [`codex/tree_fqi_history_agent.md`](codex/tree_fqi_history_agent.md) —
  Implementation and pilot record for tree FQI augmented with learned movement
  history.
- [`codex/tree_fqi_history_400step_pilot.md`](codex/tree_fqi_history_400step_pilot.md)
  — Three-seed history-agent pilot using the full 400-step training horizon and
  its matched held-out confirmation.
- [`codex/tree_fqi_history_600round_pilot.md`](codex/tree_fqi_history_600round_pilot.md)
  — 600-round extension of the history agent, checkpoint behavior, and held-out
  checkpoint selection evidence.
- [`codex/tree_fqi_loop_variant_confirmation.md`](codex/tree_fqi_loop_variant_confirmation.md)
  — Held-out comparison of one-factor loop-focused history, stagnation,
  remaining-time, and revisit-penalty variants.
- [`codex/tree_fqi_history_stagnation_600round.md`](codex/tree_fqi_history_stagnation_600round.md)
  — 600-round comparison of steps-since-coin with history windows 8 and 16,
  including held-out confirmation.
- [`codex/loop_breaking_pilot.md`](codex/loop_breaking_pilot.md) — Frozen-policy
  loop-breaking and SARSA follow-up audit, including cutoff interpretation,
  measured evaluation, and reproduction details.

### Combat agents and pilots

- [`codex/combat_fqi_agent.md`](codex/combat_fqi_agent.md) — Initial bomb-aware
  tree-FQI combat agent design, safety layer, reward, runner, and validation
  status.
- [`codex/combat_fqi_history_antistag_agent.md`](codex/combat_fqi_history_antistag_agent.md)
  — Combat FQI history/anti-stagnation representation, controlled pilot
  protocol, results, and next gates.
- [`codex/combat_fqi_history_antistag_topology_agent.md`](codex/combat_fqi_history_antistag_topology_agent.md)
  — Topology-feature ablation of the combat history/anti-stagnation FQI agent
  and its mixed-curriculum results.
- [`codex/combat_dqn_agent.md`](codex/combat_dqn_agent.md) — Vanilla DQN combat
  agent design, repaired training/evaluation behavior, and smoke-test status.
- [`codex/combat_dqn_topology_agent.md`](codex/combat_dqn_topology_agent.md) —
  Original 46-feature topology-DQN ablation, results, and interpretation
  caveats from the later training audit.
- [`codex/combat_dqn_r_topology_agent.md`](codex/combat_dqn_r_topology_agent.md)
  — Repaired 46-feature topology DQN, escape-safety repair, focused tests, and
  status of the required matched training comparison.
- [`codex/combat_ddqn_route_agent.md`](codex/combat_ddqn_route_agent.md)
  — Route-aware 52-feature Double-DQN successor, representation contract,
  aliasing regression, completed three-seed mixed run, feature judgment, and
  proposed fixed-suite/control evaluation gates.
- [`codex/ddqn_training_regimen_20260921.md`](codex/ddqn_training_regimen_20260921.md)
  — Cross-run evidence audit and the current recommended DDQN curriculum,
  replay, evaluation, checkpoint-selection, early-stop, and compute regimen.
- [`codex/agent032_imported_league_pilot.md`](codex/agent032_imported_league_pilot.md)
  — Completed broad imported-opponent league versus built-in-roster control,
  including the CPU/GPU decision and non-promotion result.
- [`codex/Agent_037_Tournament_Fast_DDQN_Design.md`](codex/Agent_037_Tournament_Fast_DDQN_Design.md)
  — Focused 101-input tournament continuation design and completed three-seed
  results, package selection, and imported-opponent generalization audit.
- [`codex/Agent_038_Symmetric_Population_DDQN_Design.md`](codex/Agent_038_Symmetric_Population_DDQN_Design.md)
  — Active 117-input warm-versus-scratch population experiment; design and
  run interpretation only until the registered jobs complete.
- [`codex/Agent_039_Compact_Audit_DDQN_Design.md`](codex/Agent_039_Compact_Audit_DDQN_Design.md)
  — Approved 104-input Agent 038 feature compression; implementation and
  contract-test status, with matched training comparison pending.
- [`codex/Agent_040_Optimized_Compact_DDQN_Design.md`](codex/Agent_040_Optimized_Compact_DDQN_Design.md)
  — Direct/cached Agent 039 feature computation, optimized D4 transforms and
  replay indexes, contract validation, and initial CPU benchmark.
- [`codex/Agent_042_Combat_Progress_DDQN_Design.md`](codex/Agent_042_Combat_Progress_DDQN_Design.md)
  — Agent 042 diagnosis, action-aligned combat inputs, persistent local
  history, separate exploration accounting, contract tests, and the 300-round
  solo stagnation gate.
- [`codex/Agent_043_Novelty_Credit_DDQN_Design.md`](codex/Agent_043_Novelty_Credit_DDQN_Design.md)
  — Agent 043 novelty guard, event-attributed progress clock, three-step
  returns, potential shaping, solo-gate result, and staged league protocol.
- [`codex/agent034_compact_fqi_agent.md`](codex/agent034_compact_fqi_agent.md)
  — Compact route-aware tree-FQI implementation, dimensional contract,
  validation status, and limits.
- [`codex/agent035_compact_fqi_symmetry_agent.md`](codex/agent035_compact_fqi_symmetry_agent.md)
  — Corrected compact FQI successor, eight-way symmetry contract, resumable
  checkpoint state, diagnostics, and validation status.
- [`codex/agent036_compact_fqi_robust_agent.md`](codex/agent036_compact_fqi_robust_agent.md)
  — Robust compact FQI successor, decision-time/safety repairs, benchmark,
  checkpoint-resume validation, and current evidence limits.

## Where to record new work

- Add a new implementation decision or durable algorithm fact to the most
  relevant note above and update `experiment_registry.md` if it creates or
  changes an experiment family.
- Add measured pilot results to that pilot's note, including seeds, boards,
  horizon, checkpoint, metrics, and artifact path. Do not overwrite earlier
  results; append a dated section when the protocol differs.
- Add a diagnosis to the relevant diagnosis note, and mark its evidence as
  diagnostic rather than a new result when no new training/evaluation was run.
- Keep `implementation_roadmap.md` as the concise project-level timeline and
  `final_project.md` as the source of assignment requirements.

## Long-running training and evaluation jobs

- Use [`jobctl`](jobctl) for detached training and evaluation commands that
  need to survive terminal, SSH, or chat-session disconnects.
- Give every job an explicit stable ID, for example
  `agent029-train`, `agent029-normal`, and `agent029-classic`.
- Queue dependent evaluations with `./jobctl start --after <training-id>`.
  A dependent job starts only when every dependency succeeds; failed
  dependencies block downstream jobs.
- Inspect jobs with `./jobctl list`, `./jobctl status <id>`,
  `./jobctl ps <id>`, and `./jobctl tail -f <id>`. Stop a specific job with
  `./jobctl stop <id>`; this targets its recorded process group and does not
  search for or kill unrelated Python processes.
- Keep generated job metadata and logs in `.jobctl/`; this directory is
  ignored by Git. Jobs not launched through `jobctl` do not have a safe
  controller ID and should not be stopped by broad process-name matching.
- For CPU-only evaluation, follow the cluster guidance: hide CUDA and limit
  math-library threads with `--env CUDA_VISIBLE_DEVICES=` and the appropriate
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` overrides.

## Jobctl command style

From the workspace root, use the direct path `bomberman_rl_exp/jobctl ...`
for training and evaluation job commands. Do not require a preceding `cd`
into `bomberman_rl_exp`; this keeps commands copyable from the project root.

## Reusable experiment runbook

- [`codex/tournament_training_protocol.md`](codex/tournament_training_protocol.md) — complete reusable workflow for training a three-seed combat agent, creating the fixed classic-tournament manifest, running frozen evaluation, judging metrics, and preserving provenance.
- [`codex/experiment_registry.md`](codex/experiment_registry.md) — record experiment identity, protocol changes, artifacts, and comparison status.
- [`eval_suite/README.md`](eval_suite/README.md) — pre-combat Coin Heaven and Loot Crate evaluation protocol.
