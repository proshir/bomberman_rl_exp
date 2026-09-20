# Experiment job control

`../jobctl` gives every detached training/evaluation command a stable ID. The
ID is the handle to use in a later chat message: `status`, `tail`, `ps`, and
`stop` all operate on the exact process group started for that ID.

The registry is stored in `.jobctl/` by default. Set
`BOMBERMAN_JOBCTL_DIR=/path/to/private/job-registry` if the experiment tree is
shared or if logs/metadata should live elsewhere.

## Start training and queue evaluations

Use explicit IDs that are easy to quote in chat. For example:

```bash
cd /export/home/salitanl/projects/ml-project/bomberman_rl_exp

./jobctl start --id agent029-train \
  --progress-file /export/scratch/salitanl/agent029_staged_600/learning_curve.json -- \
  python3 src/run_combat_training.py \
    --agent Agent_029_combat_ddqn_adversarial_window_agent \
    --curriculum staged-combat --rounds 600 --seeds 0 1 2 \
    --parallel-seeds --output /export/scratch/salitanl/agent029_staged_600

For tournament-aligned training, use the opt-in four-player schedule. It
rotates single-opponent rehearsal and true three-opponent lineups while
retaining Coin Heaven and Loot Crate replay:

```bash
python3 src/run_combat_training.py \
  --agent Agent_029_combat_ddqn_adversarial_window_agent \
  --curriculum tournament-combat --rounds 1000 --seeds 0 1 2 \
  --parallel-seeds --output /export/scratch/salitanl/agent029_tournament_1000
```

The training runner automatically evaluates every saved checkpoint on the
solo gates and all configured tournament lineups. Use
`eval_suite/classic_tournament_template.json` for an independent frozen
comparison after replacing its checkpoint paths.

./jobctl start --id agent029-normal --after agent029-train -- \
  python3 eval_suite/run_suite.py \
    --manifest /path/to/agent029-normal.json --parallel 3 --game-workers 8

./jobctl start --id agent029-classic --after agent029-train -- \
  python3 eval_suite/run_suite.py \
    --manifest /path/to/agent029-classic.json --parallel 3 --game-workers 8
```

The two evaluations above are queued immediately and begin only after the
training job exits successfully. They can run concurrently with each other.
The manifests should point to the checkpoints produced by the training job.
Use separate output directories in the manifests so the evaluations do not
write into the same result directory.

`jobctl` creates the detached session and process group, so do not wrap the
command in another `setsid`: keeping the command in the job's group ensures
that `stop` also catches children during startup.

## Inspect or control jobs

```bash
./jobctl list                  # active jobs
./jobctl list --all            # active and finished jobs
./jobctl status agent029-train --tail 20
./jobctl status --all
./jobctl ps agent029-train
./jobctl tail -f agent029-train
./jobctl stop agent029-classic
./jobctl stop agent029-train --force
```

`stop` sends a graceful signal first, waits up to ten seconds, then kills any
remaining process in the job's process group. It does not search by a broad
`python`/agent name, so unrelated jobs are not targeted. A dependent job whose
parent fails is marked `blocked` and is never started.

Each job directory contains `job.json` and `stdout.log`; the `status` command
also reports the number of descendant processes and their aggregate CPU and
memory percentages. `--progress-file` adds a small JSON progress summary when
the training/evaluation program writes a learning curve or summary during its
run.
