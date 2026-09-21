"""Stage and train the Agent 039 compact audit DDQN.

The scratch branch is the primary matched comparison against the Agent 038
117-input control. The optional warm mode migrates Agent 038 checkpoints into
the 104-input schema, but is not the causal feature-ablation comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_agent038_dual_training import (
    CANDIDATES,
    DEFAULT_AGENT037_ROOT,
    FROZEN_AGENT037,
    IMPORTED_ROSTER,
    SOURCE,
    checksum,
    cpu_environment,
    evaluation_lineups,
    stage_package,
    training_lineups,
    copy_runtime,
)


AGENT = "Agent_039_compact_audit_ddqn_agent"
DEFAULT_AGENT038_ROOT = Path(
    "/export/scratch/salitanl/agent038_scratch_population_1200_v2_20260921"
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("scratch", "warm"), default="scratch")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--eval-scenario-workers", type=int, default=2)
    parser.add_argument("--agent038-root", type=Path, default=DEFAULT_AGENT038_ROOT)
    parser.add_argument("--agent038-episode", type=int, default=300)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds <= 300:
        parser.error("Agent 039 population training needs more than 300 rounds")

    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    runtime = run / "stage" / "runtime"
    copy_runtime(runtime)
    shutil.copytree(
        SOURCE / "agent_code", runtime / "agent_code", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    imported = {meta["package"]: meta for meta in CANDIDATES}
    imported_records = {}
    for name in IMPORTED_ROSTER:
        checkpoint = stage_package(imported[name], runtime)
        imported_records[name] = {
            "source": str((Path("/export/home/salitanl/projects/ml-project/bomberman_rl_exp") / "imported_agents" / imported[name]["source"]).resolve()),
            "checkpoint": str(checkpoint),
            "sha256": checksum(checkpoint),
        }

    frozen_source = (
        DEFAULT_AGENT037_ROOT / "seed_0" / "checkpoints" /
        f"episode_{1050:04d}.pkl"
    ).resolve()
    if not frozen_source.is_file():
        raise FileNotFoundError(frozen_source)
    frozen_destination = (
        runtime / "agent_code" / FROZEN_AGENT037 / "tournament_checkpoint.pt"
    )
    shutil.copy2(frozen_source, frozen_destination)

    initial_checkpoints = []
    warm_records = []
    if args.mode == "warm":
        initial_root = run / "initial_checkpoints"
        for seed in args.seeds:
            source = (
                args.agent038_root / "training" / f"seed_{seed}" /
                "checkpoints" / f"episode_{args.agent038_episode:04d}.pkl"
            ).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = initial_root / f"seed_{seed}" / "episode_0000.pkl"
            subprocess.run(
                [sys.executable, "-m", f"agent_code.{AGENT}.checkpoint",
                 str(source), str(destination)],
                cwd=runtime, env=cpu_environment(), check=True,
            )
            initial_checkpoints.append(destination)
            warm_records.append({
                "seed": seed, "source": str(source),
                "source_sha256": checksum(source),
                "destination": str(destination),
                "destination_sha256": checksum(destination),
            })

    train_lineups = training_lineups()
    eval_lineups = evaluation_lineups()
    command = [
        sys.executable, "run_combat_training.py", "--agent", AGENT,
        "--curriculum", "agent038-population", "--diagnostics",
        "--rounds", str(args.rounds), "--board-stride", str(args.rounds),
        "--max-steps", "400", "--eval-every", str(args.eval_every),
        "--eval-workers", str(args.eval_workers),
        "--eval-scenario-workers", str(args.eval_scenario_workers),
        "--eval-seeds", "34000", "34001", "34002", "34003",
        "--eval-seats", "0", "1", "2", "3", "--seeds",
        *map(str, args.seeds), "--parallel-seeds", "--output", str(run / "training"),
    ]
    for lineup in train_lineups:
        command.extend(["--classic-lineup", *lineup])
    for lineup in eval_lineups:
        command.extend(["--league-eval-lineup", *lineup])
    if initial_checkpoints:
        command.extend(["--initial-checkpoints", *map(str, initial_checkpoints)])

    os.symlink(SOURCE / ".git", runtime / ".git")
    provenance = {
        "experiment": "agent039_compact_audit",
        "mode": args.mode,
        "agent": AGENT,
        "source": str(SOURCE.resolve()),
        "source_head": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "source_dirty": subprocess.run(
            ["git", "-C", str(SOURCE), "status", "--short"], text=True,
            stdout=subprocess.PIPE, check=True,
        ).stdout.splitlines(),
        "cuda_visible_devices": "",
        "seeds": args.seeds,
        "rounds": args.rounds,
        "curriculum": "agent038-population",
        "input_dim": 104,
        "control_input_dim": 117,
        "imported_training_roster": imported_records,
        "frozen_agent037": {"source": str(frozen_source),
                            "sha256": checksum(frozen_source)},
        "warm_initializers": warm_records,
        "training_lineups": train_lineups,
        "evaluation_lineups": eval_lineups,
        "development_boards": [34000, 34001, 34002, 34003],
        "command": command,
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Staged Agent 039 {args.mode} branch at {runtime}", flush=True)
    if args.stage_only:
        return
    subprocess.run(command, cwd=runtime, env=cpu_environment(), check=True)


if __name__ == "__main__":
    main()
