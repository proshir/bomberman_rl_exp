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
