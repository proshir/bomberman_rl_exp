# Shared project plan and experiment record

## How we work

You choose the direction; Codex explains options, recommends approaches, and helps implement the agreed step. This file records our next work and evidence for the report. Proposed experiments are not approved runs.

The [project brief](final_project.md), especially sections 4, 7, and 9, asks for manageable subgoals, meaningful comparisons, hyperparameter optimization, and explanations of improvements and failures. We must develop at least two learning models, with one based on lecture techniques. Keep detailed assignment requirements in the brief rather than repeating them here.

## What we will do together

1. Understand the game and establish a reproducible supplied-agent baseline.
2. Choose and implement a first learning model, progressing through coins → crates and escape → peaceful opponents → competitive play.
3. Use observed problems to propose and test improvements. Discuss the second model early enough to compare both properly.
4. Compare the models, select a submission candidate, and check it in the original tournament framework.
5. Build the report from saved experiments and our explanations as we work.

For each improvement: **question → options → your decision → implementation → experiment → interpretation → next decision**. Recheck earlier capabilities after major changes.

## Proposed experiment method — awaiting agreement

Use small exploratory runs to debug and tune, then a separately planned comparison of selected versions. Before that comparison, fix the question, baseline, primary metric, meaningful improvement size, training budget, evaluation conditions, checkpoint-selection rule, and sample size. The research supports controlled comparisons and uncertainty estimates; the choices below adapt that guidance to our project. [Empirical Design in RL](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf)

- **Measure the right outcome.** For coin collection, propose coins collected within a fixed step budget, with completion rate/time as supporting measures. For competitive play, propose mean official score per game, with kills, coins, self-deaths, survival, and runtime as diagnostics. Report shaped training reward separately. Keep different scenarios/opponent lineups separate unless we agree on aggregation weights beforehand.
- **Distinguish two questions.** Testing a fixed checkpoint asks how that player performs. Comparing learning methods also requires independent training runs. For example, 10 separately trained agents evaluated over 200 games each give 10 training replicates, not 2,000. Save both training-run and game identities.
- **Make comparisons fair.** Evaluate frozen policies with recorded exploration behavior, comparable training interaction and tuning budgets, the same scenario/opponent schedule, and balanced starting positions. Record training steps as well as episodes and elapsed time. Change one component for a focused experiment; use a small combined comparison if two changes may interact.
- **Control randomness and selection.** Separate training, development, and final evaluation seed schedules. Account for environment and agent/opponent randomness. Use matched evaluation conditions where feasible; identical seed numbers alone do not guarantee identical game trajectories. Select checkpoints on development results, then evaluate them on fresh final games.
- **Report differences and uncertainty.** Propose the mean score difference with a 95% bootstrap confidence interval, preserving training runs as groups and any genuine pairing. Show individual run results too. For a fixed benchmark suite, a run-level interval describes training variability conditional on that suite; claims about new boards need uncertainty over sampled boards as well. Reused boards/seats must retain their grouping. Bootstrap intervals do not make a tiny sample reliable. [Reliable RL evaluation methods](https://github.com/google-research/rliable)
- **Choose sample sizes after a pilot.** There is no guaranteed “five seeds is enough” rule. Use pilot variability, a practically meaningful effect, and compute cost to agree on the main sample size through power or precision analysis. Freeze the main budget before inspecting its result; do not keep adding runs until significance appears. If resources cannot resolve a small difference, report it as inconclusive. [How Many Random Seeds?](https://arxiv.org/html/1806.08295v2)
- **Avoid selective conclusions.** Preserve unsuccessful runs and explain crashes. Do not report only the best seed, checkpoint, opponent, or metric. Confirm exploratory winners on fresh data; if testing many formal claims, agree on multiple-comparison handling. An interval containing zero does not establish equality.

Exact models, metrics, run counts, and statistical implementation remain undecided. We will approve a concrete protocol for the first experiment after a small baseline/pilot.

## What we save for each main experiment

Raw configurations and outputs go in root-level `experiments/`; code stays in `src/`. Append a short entry here using this template:

- **ID / date / status / contributors:**
- **Question and reason:** What did previous evidence suggest?
- **Agreed change and comparison:** What changed, and what stayed fixed?
- **Reproduction:** Command, code version, configuration, dependencies, seeds, budgets, opponent setup, and checkpoint/run links.
- **Results:** Primary outcome, uncertainty, sample counts, diagnostic metrics, and plot/raw-data links.
- **Interpretation and next decision:** What is supported, what is uncertain, and what did we decide?

Generate report figures from saved data. Record sources and assistance honestly. These entries support Methods, Training, and Experiments and Results; contributions and decisions also support Project planning. The team reviews and refines the report and keeps it out of the public code repository.

## Done so far

- Agreed the roles of `src/`, `experiments/`, `codex/`, and `env/`, and the user-led advisory workflow.
- Agreed to keep two Git repositories: the project root for instructions, notes, and selected experiment files, and the existing `src/` repository for code. Root ignore rules exclude `src/` and `env/`; related changes need separate commits. Root initialization and the first commit are pending because `.git/` is read-only, including with elevated execution.
- Created the condensed Markdown brief and repository instructions.
- Reviewed the brief's experimental requirements and the research linked above; proposed the method in this file. It is not yet an accepted protocol.
- Read-only framework inspection found existing `--save-stats` support. Its per-round output currently aggregates coins/kills/suicides across agents, so we must check what additional per-agent episode evidence is needed.
- Found separate world and agent randomness: `src/environment.py` uses a seeded generator, while `src/agent_code/rule_based_agent/callbacks.py` calls `np.random.seed()` without a fixed seed and uses Python's random shuffle.
- No custom learning agent, baseline experiment, or training result has been produced in our work so far.

**Next decision:** agree on the scope of the baseline/pilot and its measurements before implementation.
