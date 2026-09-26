# Agent code review and cleanup plan

**Status:** `q_table_agent`, `q_table_masked_agent`, `linear_sarsa_agent`, `linear_sarsa_loop_agent`, `q_table_loop_agent`, `tree_fqi_agent`, `tree_fqi_loop_agent`, `combat_fqi_agent`, `tree_fqi_history_agent`, `combat_fqi_history_antistag_agent`, `dqn_coin_agent`, and `combat_fqi_history_antistag_topology_agent` review passes complete; `combat_dqn_agent` is next.  
**Scope:** Review one package at a time under `agent_code/`.  
**Inventory:** [New agents since `0f55c1d`](repository_additions_review_since_0f55c1d.md). Earlier framework agents can also be reviewed if selected.

## Goal

Make the agents read like code the team understands, has deliberately shaped, and can maintain. Aim for ordinary, consistent project code: direct logic, meaningful names, and explanations of real design choices. A superficial pass that changes words or formatting cannot achieve this.

The [project brief](final_project.md#use-of-ai) permits AI assistance and calls for the team to refine drafts and make its contribution clear. Keep the report, Git history, and authorship claims accurate. There is no reliable test for whether code looks “AI written”; assess code quality and the team's understanding instead.

## Review sequence by introduction commit

Review the current packages in this order, based on the first commit that introduced each package after `0f55c1d`. Packages added in the same commit are listed alphabetically. The relationship column describes experimental lineage, not Python imports. Each package must remain self-contained.

| Order | First commit | Agent package | Relationship to earlier work |
| ---: | --- | --- | --- |
| 1 | 2026-09-10 `df39ed9` | `q_table_agent` | Starting tabular Q-learning agent in this sequence. |
| 2 | 2026-09-11 `3d8ee79` | `q_table_masked_agent` | Masked-action variant of `q_table_agent`. |
| 3 | 2026-09-11 `bf4dcbf` | `linear_sarsa_agent` | Separate on-policy SARSA approach; no direct predecessor in the sequence. |
| 4 | 2026-09-11 `7df0ec1` | `linear_sarsa_loop_agent` | Loop-aware variant of `linear_sarsa_agent`. |
| 5 | 2026-09-11 `7df0ec1` | `q_table_loop_agent` | Loop-aware branch of `q_table_agent`; parallel to the SARSA loop variant. |
| 6 | 2026-09-17 `8153877` | `tree_fqi_agent` | New fitted-Q-iteration approach. |
| 7 | 2026-09-17 `219c530` | `tree_fqi_loop_agent` | Loop-aware variant of `tree_fqi_agent`. |
| 8 | 2026-09-18 `9d1c3bb` | `combat_fqi_agent` | New bomb-aware combat FQI branch. |
| 9 | 2026-09-18 `b7a74b0` | `tree_fqi_history_agent` | History variant of `tree_fqi_agent`; not a continuation of the combat FQI branch. |
| 10 | 2026-09-18 `3798be9` | `combat_fqi_history_antistag_agent` | Adds navigation history and stagnation features to the combat FQI baseline. |
| 11 | 2026-09-18 `3798be9` | `dqn_coin_agent` | Separate neural coin-collection branch; no direct predecessor documented. |
| 12 | 2026-09-18 `8c367bb` | `combat_fqi_history_antistag_topology_agent` | Topology-feature ablation of the history/anti-stagnation combat FQI agent. |
| 13 | 2026-09-18 `3531cf5` | `combat_dqn_agent` | Neural combat branch using the validated 32-feature combat representation; separate from the tree-FQI learner. |
| 14 | 2026-09-18 `71b0f5a` | `combat_dqn_topology_agent` | Topology-feature ablation of `combat_dqn_agent`. |
| 15 | 2026-09-18 `0e7ac9e` | `combat_dqn_r_topology_agent` | Repaired topology-DQN branch based on `combat_dqn_topology_agent`. |
| 16 | 2026-09-19 `cd0f205` | `Agent_022_combat_ddqn_route_agent` | Controlled route-feature successor to `combat_dqn_r_topology_agent`. |
| 17 | 2026-09-19 `cd0f205` | `Agent_023_spatial_hybrid_rainbow_agent` | Separate spatial Rainbow branch; parallel to Agent 022. |
| 18 | 2026-09-19 `b19ba77` | `Agent_024_combat_ddqn_action_safety_agent` | Action-safety feature branch from the repaired 46-feature topology DDQN control. |
| 19 | 2026-09-19 `b19ba77` | `Agent_025_combat_ddqn_short_cycle_agent` | Short-cycle feature branch from the same repaired 46-feature control. |
| 20 | 2026-09-19 `b19ba77` | `Agent_026_combat_ddqn_target_coverage_agent` | Target-coverage feature branch from the same repaired 46-feature control. |
| 21 | 2026-09-19 `1f65c49` | `Agent_027_combat_ddqn_short_cycle_staged_replay_agent` | Keeps Agent 025's features and changes its curriculum and replay. |
| 22 | 2026-09-19 `c867b7e` | `Agent_028_combat_ddqn_dueling_256_agent` | Architecture ablation of Agent 027. |
| 23 | 2026-09-19 `0b03615` | `Agent_029_combat_ddqn_adversarial_window_agent` | Adds opponent-response inputs to Agent 027's base. |
| 24 | 2026-09-20 `a3c42ec` | `Agent_030_combat_ddqn_escape_replay_agent` | Corrected replay/callback continuation of Agent 029. |
| 25 | 2026-09-20 `a3c42ec` | `Agent_031_combat_ddqn_offensive_escape_agent` | Expands Agent 029's 101-input policy with offensive escape features; Agent 030 is its matched control. |
| 26 | 2026-09-20 `e5bc382` | `Agent_032_combat_ddqn_optimized_features_agent` | Behavior-preserving feature and replay optimization of Agent 031. |
| 27 | 2026-09-21 `b68269d` | `Agent_033_population_frozen_agent` | Frozen member of a paired population-training experiment; no direct predecessor documented. |
| 28 | 2026-09-21 `b68269d` | `Agent_033_population_replay_agent` | Learning/replay member of the same population experiment; paired with the frozen member. |
| 29 | 2026-09-21 `b68269d` | `Agent_034_compact_fqi_agent` | Separate compact FQI branch; not a continuation of the population agents. |
| 30 | 2026-09-21 `b68269d` | `Agent_035_compact_fqi_symmetry_agent` | Corrected symmetry-enabled successor to Agent 034. |
| 31 | 2026-09-21 `b68269d` | `Agent_036_compact_fqi_robust_agent` | Robustness and decision-time successor to Agent 035. |
| 32 | 2026-09-21 `c81b579` | `Agent_037_tournament_fast_ddqn_agent` | Tournament-focused continuation of corrected Agent 030; it also reuses implementation optimizations from Agent 032. |
| 33 | 2026-09-21 `04c8087` | `Agent_038_symmetric_population_ddqn_agent` | Extends Agent 037's representation; compares a warm start with scratch training. |
| 34 | 2026-09-21 `04c8087` | `Agent_039_compact_audit_ddqn_agent` | Controlled compact successor to Agent 038. |
| 35 | 2026-09-21 `66ac9d8` | `Agent_040_optimized_compact_ddqn_agent` | Behavior-preserving implementation successor to Agent 039. |
| 36 | 2026-09-21 `8cc8bef` | `Agent_041_dynamic_nav_ddqn_agent` | Adds action-aligned navigation inputs to Agent 040. |
| 37 | 2026-09-21 `8cc8bef` | `Agent_042_combat_progress_ddqn_agent` | Adds combat-progress inputs and agent-local history to Agent 041. |
| 38 | 2026-09-21 `8cc8bef` | `Agent_043_novelty_credit_ddqn_agent` | Adds bounded novelty and credit assignment to Agent 042. |
| 39 | 2026-09-21 `7aa10ce` | `Agent_041_dynamic_nav_ddqn_agent-edited` | Edited Agent 041 variant committed after Agents 041–043; a separate branch, not a successor to Agent 043. |
| 40 | 2026-09-23 `e3bff42` | `agent_047` | Standalone package and checkpoint continuation from Agent 043; no runtime imports from Agent 043. |

The sequence covers 40 packages present in the current source tree. `Agent_050_final_agent043` appears in an intermediate commit but is absent from the current tree. For each package, compare against its own baseline and actual predecessor where applicable; adjacent commit order alone does not prove dependence.

## Establish the project's style before editing

For each package, read its entry point and 3–5 relevant files from the same repository: the framework interface, its direct predecessor or closest related agent, and a few nearby files that the team considers representative. Identify existing conventions for naming, type hints, comments, file layout, error handling, and training code. If nearby files disagree, favor the clearest established convention for that family of agents. Do not import a generic outside style or force all agents into one template.

Trace the package's actual execution path before changing it: `setup`, `act`, state features, action filtering, policy inference, checkpoint loading, and training callbacks if present. Write down the input shapes and order, legal actions, default behavior, file paths, and any state carried between steps. Read relevant design notes, but do not inspect evaluation result files or run an evaluation. When results must remain unread, avoid opening agent-specific markdown unless it is known to contain no results; design notes can include evaluation tables.

## Edit for maintainability

- Keep the algorithm and its meaningful experimental differences visible. Express the key steps in the order they happen, with names drawn from game concepts and the algorithm rather than generic names such as `data`, `result`, or `helper`.
- Simplify needless indirection: one-use wrappers, duplicate aliases, pass-through functions, unused settings, unused imports, redundant branches, and defensive checks for impossible states. Retain a helper when it names a real concept, isolates a substantial operation, or is reused.
- Use straightforward Python control flow. Keep cohesive logic together; split a long function only when the extracted part has a clear responsibility. Avoid clever compact expressions and broad rewrites unrelated to the agent.
- Follow the user's comment preference for the current pass. When asked to remove all comments, remove every comment and docstring in the requested scope, including explanatory ones; do not keep exceptions or add replacement comments. Otherwise, remove narration, generic banners, stale claims, and docstrings that repeat the code.
- Use type hints and constants where they clarify contracts or prevent mistakes. Follow surrounding conventions; do not add boilerplate to every local variable or make routine values configurable without a use case.
- Preserve useful project-specific conventions even if imperfect. Do not introduce arbitrary naming variations, artificial imperfections, or cosmetic changes intended to imply different authorship.
- Keep runtime imports within the package and the permitted project environment. Resolve model files relative to the package. Do not add dependencies or change the package directory name without a concrete need.

When related agents perform the same operation, keep names, interfaces, and unchanged logic recognizably consistent. If a standalone successor needs the same implementation, duplicate that clear code within its package rather than adding a cross-agent runtime import. When the experiment changes behavior, make that difference easy to find and explain it in the design note. Do not standardize unrelated methods merely because their files look different.

## One-agent workflow

1. **Select and isolate.** Take the next package in the queue, or a different package the user explicitly selects. Limit code edits to it and record the starting commit and any existing unrelated changes. Do not create a per-agent review note; add a reusable process lesson to this plan only when one is useful.
2. **Understand.** Inspect the package, representative project files, relevant predecessor, checkpoint, training flow, Dockerfile dependencies, and available baseline. Record its feature order and shape, action rules, callback signatures, model loading, and expected outputs.
3. **Improve.** Make small, coherent changes to names, structure, comments, and dead code based on the issues actually found. Keep algorithm changes separate and propose them as a separate task. Do not rewrite unchanged logic simply to create a visual difference.
4. **Verify.** Review the diff for unintended behavior changes, broken local imports or paths, missing dependencies, and training regressions. Do not inspect evaluation results or run an evaluation. Explain expected behavior effects from the source changes; if the source alone cannot establish them, state the uncertainty and leave the behavior unchanged.
5. **Continue.** Mark the package review complete when the source review supports it, then ask the user before starting the next package. Pause when an unresolved behavioral change requires a decision.

## Completion criteria

- The agent remains independently runnable with the expected `setup`, `act`, and training callbacks, legal actions, model input order and shape, checkpoint format, and runtime paths.
- Existing policy behavior is preserved as reasoned from the source changes, or any behavior change and uncertainty are clearly identified. Do not use an evaluation run to claim preservation.
- A maintainer can follow the main decision path without navigating unnecessary wrappers or decoding vague names. Related agents have consistent interfaces where they share behavior.
- Training code remains usable where present. The package has no imports from other agents and uses only dependencies available in the project's Dockerfile environment.
- The diff is focused. Do not create per-agent review notes for a cleanup pass. Git history and the report accurately describe contributions and AI assistance under the course rules.

This is a code quality and comprehension pass. Do not rewrite history, delete genuine experiment evidence, fabricate human authorship, or claim an agent has been reviewed before the review occurs.

## Lessons from the `q_table_agent` pass

- Check repository roots before reading or editing: `bomberman_rl/` and `bomberman_rl_exp/` are separate Git repositories. Run `git status` and `git diff` from the relevant repository so unrelated experiment-note edits are not mixed with agent changes.
- Compare against the direct predecessor before aligning repeated code. For the Q-table family, preserve action order, blocked-direction order, feature tuple order, all supported feature modes, training callbacks, and checkpoint layout. Similar code should stay similar when it does the same job; keep actual experiment differences visible.
- Trace framework callbacks before simplifying training code. The terminal action can be reported through both callbacks, so the saved Q value and round counters need the existing correction logic. The short explanation belongs in the design note if the code comment is removed.
- Routine docstrings and comments can go when the code makes the operation clear. Keep rationale for unusual behavior in the design note, and avoid adding replacement comments that simply narrate the code.
- Separate a behavior-preserving cleanup from an algorithm change. For each changed function, compare inputs, outputs, tuple order, checkpoint reads/writes, and state updates against the starting version. The `q_table_agent` refactor grouped blocked-neighbour flags and feature components while retaining their order; moving progress-bucket work below the compact and position returns also avoids extra work in those modes.
- If a temporary comparison script cannot import the agent, first check how it is launched: a script run from outside the source repository may not have the repository root on `sys.path`. Run it from the source root or add that root to the script's import path. A failed import is not evidence of an agent failure.
- Do not read evaluation summaries or other evaluation result artifacts, and do not run evaluations. Keep the review focused on source behavior and existing design notes; state uncertainty rather than using evaluation results to resolve it.

## Lessons from the `linear_sarsa_agent` pass

- Keep SARSA's order visible: choose the next action using the current weights, then use that action's value to update the pending transition. Do not move the update ahead of action selection.
- The pending transition is completed in `act` because the next action is not known during `game_events_occurred`. The local import of `update` also avoids a startup import cycle because `train.py` imports from `callbacks.py`.
- Group the feature tuple by meaning before one-hot encoding it: four blocked directions, two coin-direction signs, and two progress values. Shift direction signs from `-1..1` into `0..2` before using them as array indices.
- Preserve `FEATURE_MODE = 'distance'` and its setup validation when they are part of the starting behavior. The runner can supply this setting, and validation determines which configurations are accepted even when the encoder only implements one mode.
- Keep the normalization rationale: the bias plus eight active categories produce nine nonzero entries, and scaling by `1/3` gives the encoded vector unit norm. This explains the numeric factor in a way the operation alone does not.

## Lessons from the `linear_sarsa_loop_agent` pass

- When a package rejects training in its framework `setup`, check copied training and exploration branches for reachability. Remove unreachable code only after tracing the supported callback path; keep the frozen policy's tie-breaking and checkpoint validation intact.
- Fold a one-use setup helper into `setup` when that makes checkpoint loading and per-round state initialization easier to follow. Preserve RNG initialization and call order when those generators affect action choices.
- Check the whole project for consumers before removing counters or state fields. `run_benchmark.py` reads `loop_interventions`, so this diagnostic must remain even though the agent package does not read it itself.
- Keep loop-breaking as a separate post-policy step: first get the learned greedy action, then apply the history rule using its own RNG. This keeps the experimental intervention distinct from the frozen learned policy.

## Lessons from the `q_table_loop_agent` pass

- In a frozen Q-table package, unseen states should keep the evaluation fallback of zero action values. Training-only insertion can be removed when the framework setup rejects training.
- Check whether a feature-mode setting is actually read by the feature function before keeping it as a configuration option. A setting that is assigned by a runner but ignored by the encoder does not select a representation.
- Keep external diagnostics such as `loop_interventions` when a project runner reads them, even if the agent itself never uses the value to choose an action.

## Lessons from the `tree_fqi_agent` pass

- Trace runner configuration before removing a setting. `run_benchmark.py` can set `FEATURE_MODE`, and this agent's `setup` validates that it remains `distance`.
- Preserve the fitted-Q iteration order: compute the full target batch from the current ensemble before replacing any per-action tree.
- Remove framework narration comments when requested, but review the pending-transition logic itself before changing it; callback ordering can affect which transitions reach the fitted dataset.

## Lessons from the `tree_fqi_loop_agent` pass

- Keep the feature tuple grouped and ordered like `tree_fqi_agent`: blocked directions, coin-direction signs, then progress buckets. The loop wrapper uses the same checkpoint representation.
- Keep the frozen learned policy and loop-breaking step separate. The first chooses among legal actions with the policy RNG; the later history rule uses its own RNG and increments the diagnostic consumed by `run_benchmark.py`.
- This package has no training module. Preserve its explicit training rejection and checkpoint loading in `setup` rather than retaining unreachable training code.

## Lessons from the `combat_fqi_agent` pass

- Keep immediate legality, future safety, and learned action values as separate steps. The safety mask and longest-survival fallback are part of this experiment's policy path.
- Group feature values by their meaning before assembling the vector, then compare the concatenation order with the original because the fitted trees depend on positional inputs.
- Keep safety behavior explicit in the code. If the user asks for no comments, remove all safety comments and docstrings, including those about blast behavior, bomb timing, escape margins, and opponent reachability.

## Lessons from the `tree_fqi_history_agent` pass

- Trace cached transition data across both callbacks and training before changing its shape. Store only values consumed by training or next-state feature construction, and use named fields instead of positional tuple indexes.
- Preserve the history timing: features for the current action use the prior action and visit history; training's next-state features append the old position and use the transition action as the next state's previous action.
- Keep `FEATURE_MODE` validation and environment-backed feature settings when the benchmark runner or training setup can supply those choices.

## Lessons from the `combat_fqi_history_antistag_agent` pass

- A standalone variant must keep its feature encoder and safety implementation inside its own package. Copy stable shared logic locally when runtime imports from another agent would make the package depend on that agent.
- Trace feature-cache consumers before changing the cache shape. Named fields make cached state features, position history, progress signature, and progress timestamp explicit across callbacks and training.
- Keep the combat feature order unchanged, then append the previous action, recent visit count, and stagnation bucket in that order. The history values are part of the fitted trees' input contract.
- Use this agent's own historical source as the behavior baseline when making it standalone. A neighboring agent may have since changed safety rules; copying its current implementation can change the target agent's policy even when both agents share a family name.
- Check runner configuration before removing setup validation or constants. `run_benchmark.py` can supply `FEATURE_MODE`; preserving behavior includes preserving which values `setup` accepts or rejects, and the order of setup checks.
- For a behavior-preserving cleanup, compare the changed package with its own starting version across setup, action selection, callbacks, feature order, cache timing, checkpoint I/O, defaults, and fallbacks. Keep experiment changes separate from readability changes, and do not infer equivalence just from lineage.
- When the user requests comment-free code, remove every comment and docstring in the requested scope, even if it explains a safety assumption. Do not replace removed comments with new ones.

## Lessons from the `dqn_coin_agent` pass

- Preserve the 21-value input order: blocked directions, nearest-coin direction and maze distance, coin count, position and step, local coin flags, quadrant counts, then free-neighbour count. The checkpoint shape does not describe what each position means.
- Preserve the breadth-first search neighbour order and the action order `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`; both can affect tie choices and the mapping between network outputs and actions.
- Follow the framework callback path through `agents.py` before changing transition handling. `end_of_round` can replace the pending final transition or create a terminal transition when the agent died; `Transition.done` controls bootstrapping. Remove unused flush arguments or tensors only after checking all call sites and consumers.

## Lessons from the `combat_fqi_history_antistag_topology_agent` pass

- Make a variant independent by copying the exact historical feature and safety implementations it used, then keeping those copies inside its package. Do not substitute a newer sibling implementation; it may contain behavior changes.
- Preserve the concatenation order of base combat features, history features, and topology features. Treat the full vector as the fitted trees' input contract.
- Use named records for cached action context and replay transitions when they replace positional tuples; first confirm every consumer and keep the stored values, history timing, and transition order the same.
