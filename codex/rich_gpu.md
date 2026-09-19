# Rich GPU/CPU agent plan

## Status and scope

**Implementation decision, 19 September 2026.** The approved Phase-0 package
now exists at `src/agent_code/Agent_023_spatial_hybrid_rainbow_agent/`.
It includes the fixed observation contract, self-contained deployment safety
mask, spatial dueling quantile network, framework training hooks, and three
focused feature/safety contract tests. No teacher dataset, checkpoint, GPU job,
or substantial training has been run, so this document contains no measured
Agent-023 result.

### CPU smoke check, 19 September 2026

One 400-step `coin-heaven` framework round in training mode (seed 23001) and
a separate 400-step checkpoint-loading evaluation round (seed 23002) both
completed without callback, feature, model, safety, or timeout errors. CUDA
was explicitly hidden and BLAS/OpenMP threads were capped at one. Logged
action times were 20.8 ms mean / 200 ms maximum during the first training-mode
round (the first action includes cold initialization), and 20.5 ms mean / 30
ms maximum in the separate evaluation round. The checkpoint was random
initialization with fewer than the replay warm-up transitions, so its movement
behavior and score are **not** a performance result. Logs are retained under
the source repository's `experiments/agent_023_smoke_20260919/`.

### Aborted 600-round mixed-run diagnostic, 19 September 2026

The requested one-seed, 600-round mixed Coin-Heaven/Loot-Crate run was started
on an otherwise free A100 GPU with one isolated CUDA device and four CPU BLAS
threads. It was stopped at the user's request; it produced **no usable training
or performance result**. The initial 32-game-per-scenario episode-0 evaluation
did complete, confirming only random-initialization behavior: 0.125 mean Coin
Heaven coins, zero Loot-Crate coins, 100% survival, and zero invalid actions.

Two implementation defects were found and fixed during this diagnostic:

1. the training callback accidentally shadowed its runner checkpoint wrapper
   with the model serializer, causing the first attempt to fail at end of
   round 1;
2. the first learner update (after 4,096 replay entries) built quantile-loss
   tensors and prioritized-replay weights on CPU while the model was on CUDA.

The replacement attempt completed ten 400-step rounds at roughly 3.6 seconds
per round with zero invalid actions or deaths, then exposed the second defect.
A direct A100 forward/backward/priority-update test passed after the repair.
The attempted resume then failed before training because the generic runner
does not allow its episode-0 evaluator to reuse an existing output directory.
All detached processes were confirmed stopped and GPU 0 was released. Scratch
artifacts are retained at
`/export/scratch/salitanl/agent_023_full_mixed_600_seed0_20260919/` and
`/export/scratch/salitanl/agent_023_full_mixed_600_seed0_retry_20260919/` for
diagnosis only; do not treat them as experiment results.

`Agent_022_combat_ddqn_route_agent` is finished and remains the compact,
route-aware Double-DQN reference. Its completed three-seed mixed run reached
49.67 mean Coin-Heaven coins and 77.8% completion at round 550, and 39.78
mean Loot-Crate coins at round 500 on its small internal evaluation. The best
individual seed reached 50.00 Coin-Heaven coins and 45.00 Loot-Crate coins,
but round-600 Loot-Crate performance ranged from 20.67 to 45.00 across seeds.
It proves that one route-aware learned policy can perform both tasks, while
also proving that seed stability and long-horizon crate planning remain the
gap. The next proposed implementation is a new
agent, tentatively named `Agent_023_spatial_hybrid_rainbow_agent`. The number
is not consumed until the agent directory is created.

The goal is to build the strongest model that can be trained and selected
reproducibly before the submission deadline. It should use all *useful* free
CPU/GPU capacity: wide parallel data generation, large GPU batches, many
independent candidates, and repeated seed confirmation. It must not waste
shared compute on duplicated CPU-bound evaluations or multiple jobs on the
same GPU. The primary initial target is Stage 2 (`coin-heaven` plus
`loot-crate`); the representation and interface must also support later
opponent training.

No imported student implementation or checkpoint will be copied into the
submission agent. Imported agents are behavioral references and sources of
high-level design hypotheses only. Teacher data will come from project-owned
planning code and supplied framework agents.

## Measured performance targets

The fixed Stage-2 protocol uses board seeds `30000`--`30007`, all four seats,
400 steps, and 32 games per scenario. These boards are now development data
because their results have influenced model design.

| Reference | Coin Heaven | Loot Crate | Interpretation |
| --- | ---: | ---: | --- |
| Imported `imp_li_deep_killer` | 50.00 coins, 100% completion, 144.78 completion steps | 40.50 coins, 97.03 crates, 100% survival | Strongest verified imported all-round Stage-2 agent |
| Local external reference `harvy` | 50.00 coins, 100% completion, 194.09 completion steps | 42.97 coins, 109.19 crates, 100% survival | Strongest measured Stage-2 reference |
| Imported `imp_alii_overlord` | 50.00 coins, 100% completion, 133.75 completion steps | 4.72 coins, 12.47 crates, 100% survival | Evidence that a large spatial CNN alone is insufficient |

The new agent's development gate is:

- 100% survival, zero self-kills, and zero invalid actions;
- at least 50.00 mean Coin-Heaven coins with 100% completion;
- more than 42.97 mean Loot-Crate coins and more than 109.19 crates;
- lower no-progress and repeated-state diagnostics than the retained local
  baseline;
- single-thread CPU inference comfortably below the official 0.5-second
  action limit, with a provisional engineering target of p99 below 0.10
  seconds.

These are development thresholds, not a final performance claim. A selected
configuration must be frozen and confirmed on unused board seeds before it is
called better than a reference. Learning-method comparisons require multiple
independently trained seeds; many games from one checkpoint are not multiple
training replicates.

## Central hypothesis

The remaining gap is caused by the interaction of four limitations rather
than insufficient MLP width alone:

1. compact feature vectors still discard global spatial and temporal detail;
2. sparse delayed rewards make bomb placement and escape difficult to learn;
3. serial environment interaction cannot keep a GPU busy;
4. uniform one-step replay spends most updates on repetitive, low-information
   movement transitions.

The proposed agent therefore combines a full-board neural representation,
action-conditioned engineered information, exact safety and bounded bomb
planning, demonstration pretraining, and sample-efficient off-policy RL. Each
part must be logged and later ablated so that a strong result remains
scientifically interpretable.

## Proposed agent

### Exact first feature contract: rich but auditable

The initial rich model has **5,912 input numbers per decision** before neural
encoding:

```text
20 spatial channels x 17 x 17 tiles = 5,780 values
36 global / temporal scalars             =    36 values
16 action-conditioned values x 6 actions =    96 values
                                         ---------
total                                    = 5,912 values
```

This 5,912-value contract is fixed for the first rich-model implementation and
must be recorded in its resolved configuration. The important point is that it
is a full-board state plus all six action consequences, not a large
unstructured handcrafted vector.

All channels and scalars are restricted to data available in `game_state` at
the time of the decision plus deterministic calculations from that state.
Hidden coin locations, future opponent actions, rewards, event callbacks, and
the simulator's unseen future state are forbidden inputs.

The feature bank is deliberately broad at the start. We will not try to guess
the perfect small set before training. Instead, the raw board preserves nearly
all observable information and grouped held-out ablations determine which
derived information actually earns its compute cost.

### 1. Spatial observation

Use a `17 x 17` board tensor with exactly 20 channels. Store categorical
channels as compact integer or bit-packed data in replay and convert to
floating-point tensors only when sampling a GPU batch.

Planned channels are:

1. stone walls;
2. crates;
3. visible coins;
4. controlled-agent position;
5. opponent positions;
6. current explosion lifetime;
7. bombs with timer 0;
8. bombs with timer 1;
9. bombs with timer 2;
10. bombs with timer 3;
11. earliest predicted lethal time, normalized;
12. tiles that become lethal within the next action horizon;
13. time-expanded safe-reachable tiles;
14. static free-space connected component from the agent;
15. normalized multi-source coin route-distance field;
16. normalized multi-source crate-approach route-distance field;
17. per-tile immediate crate yield if a bomb is planted there;
18. per-tile certified escape feasibility after a bomb is planted there;
19. capped decayed recent-visit heatmap;
20. decayed recent-position trail.

Separate bomb-timer and predicted-danger channels are intentional. A single
binary bomb map cannot express when a corridor becomes lethal. Static route
fields complement, rather than replace, the time-expanded safety search.

All coordinate conventions and channel semantics require deterministic unit
tests, including rotations/reflections, blast blocking at walls/crates,
chained explosions, and the route-aliasing witness from the DQN audit.

### 2. Exact scalar and action-conditioned bank

The 36 global values are:

- own bomb availability, remaining-time fraction, own score, best opponent
  score, score margin, living-opponent count, visible-coin count, and crate
  count (8);
- one-hot previous action including a round-start value (7);
- short action history encoded as the two actions before the immediately
  previous action, each as a one-hot plus
  whether each succeeded (14);
- steps since any progress, recent-position repeat count, current reachable
  area, current safe-reachable area, nearest coin distance, nearest
  crate-approach distance, and nearest opponent distance (7).

The 16 values for **each** of UP, RIGHT, DOWN, LEFT, WAIT, and BOMB are:

1. legal;
2. safe under the same time-expanded mask used at execution;
3. earliest lethal time after the action;
4. safe-reachable region size after the action;
5. shortest escape-path length;
6. coin-route distance after the action;
7. coin-route reachability;
8. crate-route distance after the action;
9. crate-route reachability;
10. immediate coin opportunity;
11. immediate crate yield;
12. crate-reveal opportunity (number of crates affected, without using hidden
    coin locations);
13. certified bomb-escape feasibility;
14. opponent blast coverage;
15. revisit/short-cycle penalty;
16. bounded plan value.

Some action values are defined as zero plus a validity flag when an action is
illegal. The network must never infer that an invalid action has a favorable
route or plan value.

The final Q head should score each action using both a shared state embedding
and that action's own consequence features. This avoids forcing a global
embedding to rediscover which route or escape value belongs to which action.

### 3. Neural architecture

The first implementation should be rich enough to use GPU batches and exploit
the full spatial feature bank, while still satisfying official CPU inference:

- residual CNN trunk with 96 channels and six residual blocks as the default
  training model; profile 128 channels/eight blocks as a high-capacity arm;
- GroupNorm or LayerNorm rather than BatchNorm, because observations are
  strongly correlated and inference uses batch size one;
- spatial pooling plus a scalar/action-feature fusion MLP;
- dueling value and advantage decomposition;
- distributional action values, preferably quantile regression or a fixed
  categorical support;
- approximately 3--8 million parameters, subject to measured CPU latency;
- a small deployment-distillation head is permitted only if the stronger model
  misses the official CPU latency gate.

A recurrent GRU is a gated follow-up, not part of the minimum viable model.
It should be added only if opponent experiments demonstrate temporal aliasing
that the current state and short history cannot resolve. Recurrent replay,
burn-in, and hidden-state resets add substantial correctness risk without a
clear Stage-2 benefit.

### 4. Safety and bounded planning

Retain a deterministic time-expanded safety layer. The learned policy must
never be trained or evaluated against Bellman targets containing actions that
the deployed policy cannot execute.

At action time:

1. enumerate legal actions;
2. calculate the same safe-action mask used by behavior and Bellman targets;
3. let the network rank safe movement and wait actions;
4. generate a small set of promising bomb plans of the form
   `route to tile -> place bomb -> escape`;
5. roll those plans through a lightweight exact simulator until the relevant
   explosions settle;
6. choose a bomb only when its certified value exceeds the best movement by a
   tuned margin.

Search should be bounded by explicit node, depth, and wall-clock budgets.
Log whether each final action came from the network, safety fallback, or bomb
planner. This is required to detect a planner that silently dominates the
learned policy.

## Learning algorithm

The initial complete candidate will use a conservative Rainbow-style subset:

- masked Double-DQN selection/evaluation;
- three- or five-step returns;
- prioritized replay with importance-sampling correction;
- dueling distributional value heads;
- gradient clipping and a slowly updated target network;
- dihedral rotation/reflection augmentation with action remapping;
- mixed offline demonstration and online replay batches;
- auxiliary losses for danger time, route distance, safe-region size, and
  immediate progress.

NoisyNet exploration is deferred until a masked epsilon/temperature baseline
is stable. Components must not be added merely because they appear in
Rainbow. The local evidence says representation, safety, and curriculum can
dominate the algorithm label.

### Demonstration pretraining

Create a project-owned teacher using BFS routing, time-expanded escape search,
and bounded bomb-plan evaluation. Generate trajectories with randomized
tie-breaking so that the dataset covers more than one route through equivalent
states.

Pretraining objectives:

- cross-entropy on the teacher's safe action;
- value or return regression;
- auxiliary spatial predictions;
- optional margin loss between the teacher action and unsafe/unproductive
  alternatives.

The first target dataset is **10 million** transitions covering Coin Heaven
and Loot Crate. It should be generated by CPU workers under `/export/scratch`,
sharded by scenario and seed, and recorded with source/configuration hashes.
Replay storage must avoid duplicating large float32 current/next tensors;
compact board state plus metadata should be reconstructed during sampling.

Feature completeness and feature importance are separate questions. The model
starts with the entire bank above. After a strong trained candidate exists,
run these predeclared grouped ablations on unused development boards:

1. remove static route-distance maps;
2. remove time-expanded danger and escape maps;
3. remove bomb-yield/plan maps and bomb action features;
4. remove temporal visit/history information;
5. remove opponent context;
6. remove all engineered scalar/action values while retaining the raw board.

Use score, crates, survival, completion, no-progress tails, and action latency
to judge groups. Do not select individual features from the training reward or
from a single favorable seed. Permutation importance and gradient saliency may
diagnose a trained network, but the held-out removal experiment determines
whether a group is retained.

### RL curriculum

Use transitions, not requested rounds, as the primary training budget.
Maintain separate evaluation curves for every scenario.

1. **Navigation:** Coin Heaven with varied boards and seats.
2. **Bomb mechanics:** crate layouts emphasizing safe placement and escape.
3. **Stage-2 mixture:** Coin Heaven plus Loot Crate, with balanced replay so
   crate transitions do not erase navigation.
4. **Hunting:** peaceful and coin-collector opponents after Stage 2 passes.
5. **Competition:** rule-based and mixed supplied opponents after hunting
   passes.

Do not use self-play unless the existing accepted project decision is changed.
Do not expose rewards, future events, hidden coins, or latent simulator state
in observations available to the deployed policy.

## Training system that actually uses the hardware

The environment is CPU-heavy while neural optimization is GPU-heavy. The
training architecture should therefore separate actors from the learner:

```text
CPU rollout workers -> compact replay shards/queue -> GPU learner
       ^                                           |
       +------------- periodic weights ------------+
```

### CPU actors

- Profile 16, 32, 64, and 96 workers; select the highest count that raises
  useful transitions/second without queue, CPU, or NFS collapse.
- Give each worker one environment process and one CPU thread.
- Generate exploration, teacher, and later opponent trajectories in parallel.
- Use deterministic seed allocation and record each worker's board, agent,
  opponent, and exploration seeds.
- Batch or shard writes under `/export/scratch/$USER`; do not stream millions
  of tiny files to NFS.

### GPU learners

- Use one GPU per learner/configuration; this model is too small to justify
  data-parallel training across several GPUs.
- Use large sampled batches, initially 512--2048 depending on memory and
  measured update quality.
- Use automatic mixed precision only after an fp32 correctness run and loss/Q
  diagnostics are stable.
- Run different seeds or hyperparameter candidates on every genuinely free
  GPU; no GPU receives two unrelated training jobs at once.
- Prefer RTX 6000/2080 Ti-class hardware for pilots; use A100/H100-class GPUs
  only if profiling proves that the smaller GPU is the bottleneck.

Before every GPU launch, check `hostname`, `whoami`, `nvidia-smi`, `gpustat`,
CPU load, memory, and scratch capacity. Never use a GPU occupied by another
user. Isolate the selected device with `CUDA_VISIBLE_DEVICES`, cap BLAS/OpenMP
threads, put heavy outputs under `/export/scratch/$USER`, and launch long jobs
with a detached `setsid` wrapper that records PID, command, source provenance,
logs, exit status, and `/usr/bin/time -v` output.

Evaluation and artifact generation are CPU-bound. Run them with
`CUDA_VISIBLE_DEVICES=""` and one BLAS/OpenMP thread per worker so that they do
not create useless CUDA contexts.

## Experiment sequence

### Phase 0: contracts and throughput

- Implement observation, augmentation, action-feature, network, replay, and
  target unit tests.
- Prove that behavior and Bellman masks are identical.
- Verify n-step terminal handling and prioritized-replay weights on small
  deterministic examples.
- Benchmark environment steps/second, replay throughput, GPU utilization,
  memory, and batch-update time at 1/16/32/64 actors.
- Benchmark batch-one inference and bounded planning on one CPU thread.

Exit only when the GPU receives real work, no CPU worker oversubscription or
NFS bottleneck is present, and the final-action latency budget is safe.

### Phase 1: imitation gate

- Generate the teacher dataset on independent training boards.
- Pretrain three initialization seeds for a fixed **2 million GPU updates** or
  until validation plateaus under a predeclared patience rule.
- Evaluate frozen policies without RL on development Coin Heaven/Loot Crate.
- Inspect disagreement between policy, teacher, safety shield, and planner.

The imitation model need not beat Harvy, but it must demonstrate competent
navigation, safe bombing, and materially better initial replay quality than a
random network.

### Phase 2: controlled algorithm build-up

Use matched transitions, training/evaluation seeds, and checkpoints:

1. spatial masked DDQN with uniform one-step replay;
2. add multi-step returns;
3. add prioritized replay;
4. add dueling distributional heads;
5. add imitation initialization/mixed demonstration batches;
6. add bounded bomb planning at inference.

The complete rich candidate may be trained early to find a strong policy, but
these controlled removals are required afterward to explain which components
matter. Preserve Agent 022 as the compact vector reference throughout.

### Phase 3: resource-scaled search

Use successive halving rather than fully training every combination:

- screen **24 configurations** at 500,000 online environment transitions each
  after shared imitation pretraining;
- promote the best 8 using both Coin Heaven and Loot Crate, not their
  unregistered average;
- train the best 8 for 2 million transitions each;
- train the best 3 for 10 million transitions with three independent seeds;
- train the selected recipe for 20 million transitions with five independent
  seeds if time permits.

These are upper budgets, not a promise to burn compute after a decisive
failure. The scheduler should immediately fill newly free GPUs with the next
independent candidate, seed, or confirmation run. It must never reuse a
development evaluation result as a reason to keep a run alive indefinitely.

Primary tunable dimensions are learning rate, n-step horizon, replay priority
strength, target-update rate, demonstration fraction, distributional support,
CNN width/depth, augmentation probability, and bomb-plan margin. Safety
semantics and the evaluation protocol are not hyperparameters.

Stop weak runs early for catastrophic Q growth, persistent zero crate yield,
navigation collapse, self-death, invalid actions, or sustained failure to use
the GPU. Do not stop or extend runs based on favorable noise in the fixed
evaluation boards.

### Phase 4: frozen Stage-2 comparison

For every finalist:

- choose checkpoints using development boards only;
- run 32 games per scenario for each independent training seed under the fixed
  protocol;
- report per-seed and pooled descriptive results, while treating training
  seeds as the independent units;
- freeze the selected recipe/checkpoint rule;
- confirm it on unused boards and record confidence intervals and board-level
  results.

The final Stage-2 target is to beat the Harvy reference's 42.97 Loot-Crate
coins / 109.19 crates / 100% survival while matching perfect Coin-Heaven
completion. Only after this gate should the agent advance to the fixed hunting
and competition suites.

## Diagnostics and provenance

Every run must save:

- resolved configuration, source commit, dirty status, and source hashes;
- hardware, CUDA/PyTorch versions, device, actor/thread counts, and seeds;
- transitions and updates per second, learner duty cycle, queue occupancy,
  replay age, GPU utilization, peak CPU/GPU memory, and wall time;
- loss, gradient norm, Q/target/return quantiles, TD-error distribution, and
  replay priorities/importance weights;
- reward components, action frequencies, mask sizes, planner/shield takeover
  rates, bombs, crates, coins, kills, deaths, invalid actions, survival,
  repeated states, and no-progress tails;
- separate Coin Heaven, Loot Crate, and opponent evaluation curves;
- periodic checkpoints including optimizer, scheduler, replay metadata, and
  random-number states sufficient for a documented resume.

Training reward alone is not a checkpoint-selection metric. Frozen greedy
evaluation and safety diagnostics determine promotion.

## Risks and fallback decisions

| Risk | Detection | Response |
| --- | --- | --- |
| GPU mostly idle | low utilization and learner queue starvation | increase/profile actors and batch transfer; do not move pure environment work onto GPU nodes blindly |
| CNN navigates but cannot bomb | strong Coin Heaven, weak crate yield | strengthen teacher coverage, action-conditioned bomb features, and multi-step credit rather than only widening CNN |
| Planner dominates policy | high planner takeover rate | tighten planner scope, train on planner decisions, and report learned versus planned performance separately |
| Q-value instability | exploding Q/targets, TD errors, or seed collapse | lower update-to-data ratio/LR, inspect masks and n-step boundaries, then tune target updates |
| Navigation forgetting | Loot Crate improves while Coin Heaven falls | stratified replay, retained demonstrations, and fixed mixed-curriculum proportions |
| Development-board overfit | large drop on fresh confirmation | widen training distribution and restart selection without touching final boards |
| CPU inference too slow | high p95/p99 latency on one thread | reduce CNN width/blocks, cache static maps, cap search, or distill into a smaller deployment network |
| Storage/NFS pressure | high I/O wait or many small writes | compact replay, larger shards, local scratch staging, and periodic consolidated artifacts |

If the full spatial Rainbow candidate cannot beat the Stage-2 references in
the available time, the fallback is not an arbitrarily larger model. Retain
the best validated Agent 022 checkpoint or implement a smaller learned
action-ranker distilled from the project-owned planner, whichever wins the
fresh fixed comparison and deployment-latency gate.

## Deliverables after approval

1. `src/agent_code/Agent_023_spatial_hybrid_rainbow_agent/` with a
   self-contained deployment package;
2. reusable parallel rollout, teacher-data, GPU training, resume, and profiling
   scripts under `src/`;
3. focused observation, symmetry, replay, target, safety, planning, and
   inference-latency tests;
4. experiment configurations and checkpoint references under `experiments/`;
5. updates to `AGENTS.md`, `codex/implementation_roadmap.md`,
   `codex/experiment_registry.md`, and the algorithm knowledge base when the
   agent is actually created or measured;
6. a final packaging check proving that callbacks import only files contained
   in the submitted agent directory and run within the official CPU/RAM/time
   constraints.

Substantial implementation and training begin only after the user approves
this proposed architecture and experiment strategy.
