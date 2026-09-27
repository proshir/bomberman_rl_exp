# Agent code review and cleanup plan

**Status:** `q_table_agent`, `q_table_masked_agent`, `linear_sarsa_agent`, `linear_sarsa_loop_agent`, `q_table_loop_agent`, `tree_fqi_agent`, `tree_fqi_loop_agent`, `combat_fqi_agent`, `tree_fqi_history_agent`, `combat_fqi_history_antistag_agent`, `dqn_coin_agent`, `combat_fqi_history_antistag_topology_agent`, `combat_dqn_agent`, `combat_dqn_topology_agent`, `combat_dqn_r_topology_agent`, `Agent_022_combat_ddqn_route_agent`, `Agent_023_spatial_hybrid_rainbow_agent`, `Agent_024_combat_ddqn_action_safety_agent`, `Agent_025_combat_ddqn_short_cycle_agent`, `Agent_027_combat_ddqn_short_cycle_staged_replay_agent`, `Agent_028_combat_ddqn_dueling_256_agent`, `Agent_029_combat_ddqn_adversarial_window_agent`, `Agent_030_combat_ddqn_escape_replay_agent`, `Agent_031_combat_ddqn_offensive_escape_agent`, `Agent_032_combat_ddqn_optimized_features_agent`, `Agent_033_population_frozen_agent`, `Agent_034_compact_fqi_agent`, `Agent_035_compact_fqi_symmetry_agent`, `Agent_036_compact_fqi_robust_agent`, `Agent_037_tournament_fast_ddqn_agent`, `Agent_038_symmetric_population_ddqn_agent`, `Agent_039_compact_audit_ddqn_agent`, `Agent_040_optimized_compact_ddqn_agent`, `Agent_041_dynamic_nav_ddqn_agent`, `Agent_042_combat_progress_ddqn_agent`, `Agent_043_novelty_credit_ddqn_agent`, and `agent_047` review passes complete; `Agent_026_combat_ddqn_target_coverage_agent` is omitted from the review queue as requested; `Agent_041_dynamic_nav_ddqn_agent-edited` is next. Agents 028, 029, 030, 031, 034, 035, and 047 were selected out of sequence.

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
| 35 | 2026-09-21 `66ac9d8` | `Agent_040_optimized_compact_ddqn_agent` | Optimized compact successor to Agent 039; its starting source uses a 20% combat-escape replay share, compared with Agent 039's 10%. |
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

## Lessons from the `combat_dqn_agent` pass

- Before editing a base agent, search descendant packages for imports and attribute access. Preserve public functions, constants, callback exports, feature-cache tuple positions, replay-batch field order, and checkpoint keys that later agents use.
- Remove sibling-agent runtime imports by copying the behavior the package currently receives into its own modules. Keep its feature order, safety candidates, reward calculation, and helper names unchanged so the package runs independently and its descendants retain their interface.

## Lessons from the `combat_dqn_topology_agent` pass

- A standalone DQN variant needs its own callbacks, training, feature, safety, config, network, and replay modules. Keeping only a local feature encoder while forwarding the rest to a sibling still leaves a runtime dependency.
- Preserve the DQN input as the 32 combat/history values followed by the 14 topology values. Keep the topology helper and its ordering local to the package.
- When moving reward code into a local training module, check that every helper it calls is imported from the package. The copied reward function referenced `bomb_is_useful` and `earliest_danger`; add those local safety imports so the training path can execute.
- Keep checkpoint defaults relative to the standalone package so variants with different input dimensions do not point at a sibling's checkpoint.

## Lessons from the `combat_dqn_r_topology_agent` pass

- When a package becomes a base for later agents, check all descendants for imports before removing re-exported training constants or feature functions. Keep the base feature vector and callback exports stable.
- Preserve experiment settings that are fixed by the agent wrapper, such as the `ddqn` algorithm, and trace how the training runner supplies model and resume paths before making callbacks local.
- Keep each repaired variant's checkpoint filename and input dimension local to its package; descendants may share feature code while still using separate checkpoints.

## Lessons from the `Agent_022_combat_ddqn_route_agent` pass

- Trace callback and training imports transitively. A local feature file alone does not make a package independent when setup, optimization, replay, model loading, or safety still comes from another agent.
- Keep a successor's old feature blocks local, then replace the intended slices in place. For this route variant, the order remains the 14-value combat prefix, coin and crate routes, the nine-value combat suffix, three history values, nine topology values, and remaining time.
- When localizing a fixed-algorithm wrapper, preserve its algorithm choice, checkpoint filename, resume override, and runner-facing training names alongside the model input contract.

## Lessons from the `Agent_023_spatial_hybrid_rainbow_agent` pass

- Treat the spatial model input as a checkpoint contract: 20 ordered 17×17 channels, followed by 36 global values and 16 values for each of the six ordered actions. Keep repeated feature slots when they are present in the trained representation.
- Preserve the action-time feature-cache sequence in training callbacks. The pending transition is committed after the next decision state's features have been cached.
- Preserve module and parameter names, architecture values, and checkpoint metadata while reformatting model code; these determine whether existing weights can load.
- Search project source for consumers before removing config values. `N_STEP` had no reader in the package or Python source files in the project and was removed; keep that check away from evaluation artifacts.

## Lessons from the `Agent_024_combat_ddqn_action_safety_agent` pass

- Trace each safety helper to its caller before consolidating it. This package used the ordinary survival search for policy masks and training, but the opponent-aware search only for action-consequence features; keep the distinction explicit when making it standalone.
- When replacing a positional callback cache with named fields, update every local consumer, including training, and confirm no other package imports the cache implementation.
- Preserve the 46 topology inputs followed by ten values for each ordered action, along with the action-safety checkpoint filename and DDQN model/checkpoint structure.

## Lessons from the `Agent_025_combat_ddqn_short_cycle_agent` pass

- Keep Agent 025's 19 short-cycle values after the 46-value topology vector: two six-way action one-hots, two success flags, then reversal, wait run, two-step cycle, four-step cycle, and displacement.
- When removing a wrapper's sibling callback dependency, reproduce its setup locally without mutating the sibling module's checkpoint globals. Keep the short-cycle checkpoint path, DDQN choice, network state names, and checkpoint format intact.
- Descendant feature modules import Agent 025's action list, feature dimensions, short-cycle encoder, and main encoder. Keep those exports and their input order available while localizing the implementation.
- A named feature-cache record is useful for the expanded action and position history, but update the training consumer at the same time and preserve when the pending transition is committed.

## Lessons from the `Agent_027_combat_ddqn_short_cycle_staged_replay_agent` pass

- Preserve the 65-value short-cycle input and the six action order; descendants reuse the feature exports, callback names, training constants, and scenario replay type.
- Keep `ScenarioReplayBuffer.set_context`, `tags`, `indices_by_tag`, and `weights` stable. The training runner updates these each episode and reads the tag counts and active weights for its round records.
- Keep the tuple feature-cache layout because descendant training wrappers read the feature vector at index zero. When localizing training, bind its fallback feature builder to the agent's own callbacks; the old imported topology trainer was bound to a 46-value feature builder.
- Trace the active policy path before copying a re-exported safety module. This package's callbacks and training used the ordinary DQN safety search; its FQI safety re-export was not used by the policy or training path.
- When a package becomes self-contained, update the training runner's source-hash list to track its local modules and drop the now-unused upstream feature file.

## Lessons from the `Agent_028_combat_ddqn_dueling_256_agent` pass

- A network ablation that forwards most of its behavior to a predecessor still needs local callbacks, config, features, replay, safety, and training code if the package must run independently. Copy the behavior it currently receives and keep the experiment's model change explicit.
- Preserve the 65-value input and Agent 027's transition/cache/replay contracts. For the dueling network, retain the `trunk`, `value`, and `advantage` module names, 256-unit layers, mean-centered advantages, and checkpoint keys so existing weights and optimizer state keep their expected structure.
- Keep the `dueling_256_checkpoint.pt` default, explicit runner model paths, DDQN selection, and resume path behavior while localizing setup and checkpoint helpers.
- Once code is local, update the training runner's source-hash inputs to cover the local config, model, replay, and package entry file, and remove the predecessor paths that are no longer runtime dependencies.

## Lessons from the `Agent_029_combat_ddqn_adversarial_window_agent` pass

- Trace each safety import to its caller when a package uses more than one safety implementation. Agent 029's action mask and training use the ordinary DQN survival rules, while its 46-value base representation and opponent-window features use those shared movement/blast rules plus opponent reachability. Keep the opponent response as a learned feature rather than adding it to the action veto.
- Preserve the 101-value order: 46 combat/topology values, 19 short-cycle values, then six opponent-window values for each of the six ordered actions. Keep feature exports used by Agents 030, 031, and 032, including the action list, `state_to_features`, and `short_cycle_features`.
- When copying training locally, preserve the scenario replay interface and callback cache tuple because the training code and descendants consume them. Keep `MODEL_PATH`, `RESUME_PATH`, and callback exports available; Agent 030 sets those module values before delegating setup.
- Hash the new agent-local config, model, replay, entry point, and base feature module in its training manifest. Keep Agent 029 source hashes for descendant agents that still import its callbacks or features.

## Lessons from the `Agent_030_combat_ddqn_escape_replay_agent` pass

- When removing a predecessor import, copy the behavior Agent 030 actually received through its callbacks, feature modules, replay base, and trainer. Keep its 101-value feature order, action order, checkpoint filename, and DDQN network structure intact.
- Preserve the replay context API and the 20% `combat_escape` sample share. Set the transition tag before adding a transition and keep clearing it after each add so a later move does not inherit the previous tag.
- Keep the 8-slot feature-cache tuple and commit pending transitions only after the next decision state's features are cached. Bind fallback next-feature construction to Agent 030's own callback module so it uses the matching 101-value representation.
- Preserve the training constants and replay class exported from `train.py`; Agents 031, 032, and later descendants read those names. Check descendant source imports before narrowing a module's public surface.
- Once the package is standalone, hash its local callbacks, feature modules, safety, config, model, replay, training, and package entry files in the runner. Descendants that import Agent 030 need those local paths in their own manifests.

## Lessons from the `Agent_031_combat_ddqn_offensive_escape_agent` pass

- Keep the 101 adversarial-window values in their established order and append the 12 offensive escape values. The checkpoint migration pads the first network layer from 101 to 113 inputs, so feature order and dimension remain a weight compatibility contract.
- Localize the callback, safety, model, replay, and training code together. Agent 031 used a six-field feature-cache tuple from Agent 029, so its copied trainer must use Agent 031's own `next_features` when it needs to build a fallback state.
- Preserve the `offensive_escape_checkpoint.pt` filename, Agent 030's combat escape replay tag and 20% sample share, and the trainer exports used by the runner. Keep the dedicated checkpoint preparation script in the run manifest, and hash Agent 031's local modules after removing upstream package imports.

## Lessons from the `Agent_032_combat_ddqn_optimized_features_agent` pass

- Preserve the 113-value order: the 46 base values, 19 short-cycle values, 36 opponent-window values, then 12 offensive summaries. Keep the optimized shared opponent envelope and local survivor search as implementation changes only.
- Keep `ScenarioReplayBuffer`'s preallocated arrays, tag membership, compatibility views, `TransitionBatch` fields, and reusable `sample_torch` path. Later agents import these replay and feature APIs, so retain their names and behavior while localizing dependencies.
- Bind the copied trainer to Agent 032's own callbacks and replay module. Its pending-transition cache uses the six-field Agent 029 layout, and the trainer fallback must call the matching local `next_features` function.
- Update Agent 032's run manifest to hash its local modules and remove its upstream hashes. Keep Agent 032's feature and replay hashes in descendant manifests where those agents still import them.

## Lessons from the `Agent_033_population_frozen_agent` pass

- In a thin callback wrapper, inspect the delegated `setup` function: it may overwrite fields the wrapper assigned first. Bind local feature modules in the setup implementation that actually runs.
- A frozen policy needs its inference dependency closure locally—callbacks, features, safety, model, and config—but does not need replay or training modules. Keep its checkpoint path and serialization format aligned with the paired learner.

## Lessons from the `Agent_034_compact_fqi_agent` pass

- Localize the complete safety dependency path: Agent 034 used combat FQI safety through both its policy mask and its reward calculation, and its crate route features also imported a combat FQI helper.
- Preserve the 28-value feature order, `compact-routes-history-v2` schema, six-tree checkpoint, and the decision snapshot lookup at `new_state["step"] + 1`. The engine reports the post-action state with the old decision's step number.

## Lessons from the `Agent_035_compact_fqi_symmetry_agent` pass

- The D4 transform depends on Agent 035's 28-value layout: four action-aligned route groups, the directional part of the previous-action one-hot, and invariant bomb/history values. Keep the transform offsets aligned with the feature encoder.
- Resume checkpoints include replay records, RNG state, counters, and fit diagnostics as well as trees. Keep the schema tuple and serialized field names stable; load the training schema lazily in callbacks to avoid the callbacks/training import cycle.

## Lessons from the `Agent_036_compact_fqi_robust_agent` pass

- Preserve the 28-value order: eight coin-route values, eight feasible-crate-route values, bomb availability/yield/escape count, seven previous-action flags, recent visits, and progress bucket. The symmetry transform depends on those directional offsets.
- Agent 038 and Agent 040's original source imported Agent 036's route helpers. Agent 040 now keeps only the `candidate_crate_tiles` helper it uses; keep Agent 036's `movement_routes`, `feasible_crate_tiles`, and `candidate_crate_tiles` interfaces stable for current consumers.
- The bomb-escape feature counts safe immediate follow-up actions from the same timed safety search used by the policy. Keep that shared timeline and the useful-bomb filter intact; remove an obsolete safety helper only after checking all package consumers.
- Keep Agent 036's full checkpoint schema and saved training-state keys stable. It stores replay, RNG state, progress counters, fit diagnostics, and replay weights alongside the trees.

## Lessons from the `Agent_037_tournament_fast_ddqn_agent` pass

- Preserve Agent 037's 101-input layout as 46 base values, 19 short-cycle values, then 36 opponent-window values; later packages import its feature exports.
- Keep the callback's six-field feature-cache tuple, 8-step histories, pending-transition timing, combat escape tag, and 20% replay share stable while localizing its callbacks and trainer.
- Copy the safety and feature helpers Agent 037 actually used into its package. Keep DQN model parameter names, checkpoint fields, tournament checkpoint name, and DDQN selection intact.
- Once standalone, hash Agent 037's local package entry point, base features, config, model, and replay in the runner. Retain Agent 037 feature hashes in descendant manifests that still import its encoder.

## Lessons from the `Agent_038_symmetric_population_ddqn_agent` pass

- Preserve the 117-value order: Agent 037's 101-value prefix followed by eight coin-route values and eight feasible-crate-route values. The route targets depend on Agent 036's opponent-aware bomb-survival search.
- Keep the D4 action and feature permutations aligned with the vector offsets. Recompute the nearest-crate direction triplet after transformation because its breadth-first search uses direction order to break ties.
- When warm-starting from Agent 037, pad the first layer's 16 new columns and matching Adam moments with zeros. Keep policy-only warm starts resetting the target, optimizer state, counters, and exploration schedule.
- Preserve the 10% combat-escape replay share and transition tagging. Keep the standard DDQN action mask and the route-specific survival search as separate local implementations.
- Hash Agent 038's local modules in its runner manifest. Descendants that import its wide feature interface should hash the local feature and safety modules they execute.

## Lessons from the `Agent_039_compact_audit_ddqn_agent` pass

- Preserve the 104-value mapping from Agent 038: keep the selected combat, topology, history, opponent-response, and route columns in order, then insert the state-wide armed-opponent flag at the expected position. Keep the 117-to-104 first-layer and optimizer-state migration aligned with that mapping.
- For a standalone compact successor, keep the wide source encoder and its route-specific survival search local, then apply the compact mapping in the package's feature module. Keep ordinary action masking and route feasibility as separate safety paths.
- Preserve the 10% combat-escape replay share, scenario replay API, six-action order, callback cache layout, and checkpoint conversion exports used by Agent 040.
- Hash Agent 039's local modules in its runner manifest. Agent 040 originally forwarded checkpoint conversion to Agent 039; after localizing that helper, hash the Agent 040 checkpoint module and retain the Agent 039 hash only for agents that still import it.

## Lessons from the `Agent_040_optimized_compact_ddqn_agent` pass

- Keep Agent 040's direct 104-value encoder and per-state `StateContext` cache together. Descendants use its `StateContext`, `ACTIONS`, `MOVE_DELTAS`, feature size, action-mask callback, symmetry transforms, replay batch, and safety exports.
- Copy the helper implementations into Agent 040 and use package-relative imports so its feature, safety, symmetry, checkpoint, model, and replay paths do not depend on another agent at runtime.
- Preserve Agent 040's dense replay index pool and 20% combat-escape sample share. The 20% value comes from Agent 040's starting replay import; Agent 039 uses 10%, despite the stated implementation lineage.
- Keep the 117-to-104 checkpoint conversion and Agent 040's migration and warm-start labels local. Normal setup continues to load the configured checkpoint directly and does not invoke conversion.
- When localizing a base used by later agents, update the runner's source hashes for that package and include each local Agent 040 module its descendants import in their manifests. Remove old dependency hashes only when no remaining descendant import path uses them.
- Static source checks can establish syntax, package-relative imports, and the public names consumed by descendants. They do not establish evaluation parity; do not run evaluation unless requested.

## Lessons from the `Agent_041_dynamic_nav_ddqn_agent` pass

- Preserve the 122-value order: Agent 040's 104 values, then six signed coin-route progress values, six reachability flags, and six normalized destination-visit counts. Each six-value block follows `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB`; symmetry transforms must permute only the four directions and leave `WAIT` and `BOMB` in place.
- Keep navigation as observation data. The existing safety candidate set still controls final action selection, and the optional 104-to-122 checkpoint warm start zero-pads the added input columns and matching optimizer state.
- For a self-contained successor, move the base feature/context, replay, symmetry, safety, callback, model, and training implementations into the package. Keep descendant-facing names such as `StateContext`, `MOVE_DELTAS`, and `_navigation_features` available from the local feature module.
- A thin callback wrapper can mutate another package's model-path and checkpoint-loader globals. Copy the setup path locally, preserve its feature-size and checkpoint checks, and apply the optional checkpoint expansion inside the local setup.
- When localizing training, compare imported constants, reward terms, pending-transition timing, replay tagging, epsilon counters, and checkpoint fields. Remove setup work whose object is immediately replaced only after confirming it does not advance a shared RNG or alter retained state.
- Hash every local Agent 041 runtime module in its runner manifest. Keep the Agent 041 feature hash in descendant manifests because Agents 042 and 043 import that feature interface; remove its former upstream hashes only from Agent 041's own manifest.
- Static checks confirmed the feature API, local imports, and source structure only. No tests or evaluation were run, so evaluation parity remains unverified.

## Lessons from the `Agent_042_combat_progress_ddqn_agent` pass

- Preserve the 140-value input order: Agent 040's 104-value base, Agent 041's 18 navigation values, then 18 combat-progress values. Keep all six-action blocks in `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB` order, and transform those blocks consistently under symmetry augmentation.
- When localizing a descendant, copy its actual transitive runtime dependencies into the package and use package-relative imports. Keep Agent 043's expected feature, callback, replay, symmetry, checkpoint, and training exports available from Agent 042.
- Retain Agent 042's solo and combat epsilon schedules, coin-and-score progress signature, position/action history, and combat-episode accounting when replacing inherited callback or training setup.
- Compare copied shared helpers with the specific source version Agent 042 used. Similar names across agents can hide changed behavior; the bomb-value helper and replay settings need to match this package's prior behavior.
- Hash every local Agent 042 runtime module in its own runner manifest. For descendants that import Agent 042 at runtime, include the local modules in their source-hash closure and remove upstream hashes only after tracing all remaining imports.
- Static checks can verify feature order, exports, imports, and source structure. No tests or evaluation were run for this pass, so evaluation parity remains unverified.

## Lessons from the `Agent_043_novelty_credit_ddqn_agent` pass

- Keep the 146-value input as Agent 042's 140 values followed by six action-aligned novelty values. The novelty values use the existing candidate mask and rolling position history; symmetry augmentation must permute this final block with the earlier action blocks.
- Preserve Agent 043's event-attributed progress clock, solo coin and score fallback, bounded loop guard, separate solo and combat exploration schedules, and three-step return queue. Its pending transition is committed after the next decision state's features are cached.
- Replace inherited setup and checkpoint globals with local setup, model, and checkpoint code. Warm starts from 104, 122, or 140 inputs must pad the first network layer and matching optimizer state to 146 inputs; a 146-input checkpoint remains unchanged.
- Trace inherited reward functions to their own imports. Agent 043's old combat-DQN reward reference used safety helpers that were absent from that module, so its local trainer now uses the same reward terms with its own safety imports before applying combat potential shaping. This repairs that training path; source review alone cannot establish prior training parity.
- Keep the replay buffer's 20% combat-escape share and runner-facing training exports. Because Agent 043 intentionally retains its Agent 042 and combat-DQN dependencies, keep those upstream modules in its runner source-hash closure.
- Static source checks confirmed local imports, retained policy functions, feature layout, and syntax. No evaluation was run, so evaluation parity remains unverified.
- If the submitted Agent 047 copy is the standalone version, Agent 043 can keep its original Agent 042 and combat-DQN dependencies. A follow-up readability pass only needs to remove comments and docstrings; preserving those imports keeps the historical experimental lineage visible.

## Lessons from the `agent_047` pass

- The packaged continuation already keeps its runtime imports inside `agent_047`. Keep the bundled `training.pkl` path, 146-value feature order, feature schema, checkpoint migration labels, safety rules, symmetry maps, and three-step replay behavior as source contracts.
- Remove stale predecessor prose and use the current package's names for internal state. A local context cache does not need an Agent 043 name, and a one-use alias for `QNetwork` adds no useful choice.
- Define solo and combat exploration schedules once in callbacks, then import the same constants into training while preserving its exported names. This prevents the policy and training metadata from drifting apart.
- Compare edited function bodies with the package's own starting source after excluding docstrings. This pass changed the local context name, removed the model-class alias, and corrected one feature error message; the remaining function bodies stayed the same. Static review does not establish evaluation parity.

## Lessons for requested before/after evaluation checks

- Run an evaluation comparison only when requested. Use the pre-edit source commit as the baseline in an isolated checkout, and run it separately from the edited working tree.
- Use the same trained checkpoint for both versions whenever it is available. Match the runner, scenario, board seeds, action seeds, seats, opponents, step limit, and relevant environment settings; keep outputs in separate temporary directories.
- Compare outcomes per game and per seed, including policy metrics and safety statistics. Ignore elapsed-time fields when checking behavioral equality because runtime measurements vary between runs.
- If the trained checkpoint named by a prior manifest is unavailable, verify the path before substituting anything. A deterministic initialized checkpoint shared by both versions can provide a quick runtime comparison, but label it as a source-behavior check; it does not establish that the trained policy retains its prior evaluation result. Ask for the missing checkpoint to confirm trained-policy parity.
- Report the baseline commit, evaluation settings, matched metrics, output locations, and any limitation that changes what the comparison can establish.
