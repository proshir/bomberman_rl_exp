"""Stage and train matched warm-start and scratch Agent 038 branches.

Run this driver once with --mode warm and once with --mode scratch. Both modes
use identical boards, seeds, population lineups, evaluation tasks, and Agent
038 source. The warm mode converts the corresponding Agent 037 checkpoints
into policy-only episode-0000 initializers with fresh training state.
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

from run_imported_stage2 import (
    CANDIDATES,
    IMPORTED,
    SOURCE,
    copy_runtime,
    stage_package,
)


AGENT = "Agent_038_symmetric_population_ddqn_agent"
FROZEN_AGENT037 = "Agent_037_tournament_fast_ddqn_agent"
IMPORTED_ROSTER = ("imp_li_deep_killer", "imp_alii_arbiter")
DEFAULT_AGENT037_ROOT = Path(
    "/export/scratch/salitanl/agent037_tournament_fast_1200_20260921_v4"
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_lineups() -> list[list[str]]:
    """Fourteen slots implement the registered Classic mixture."""
    rule = "rule_based_agent"
    deep = "imp_li_deep_killer"
    arbiter = "imp_alii_arbiter"
    frozen = FROZEN_AGENT037
    return (
        [[rule, rule, rule]] * 2
        + [[deep, arbiter, rule]] * 5
        + [[deep, arbiter, frozen]] * 4
        + [
            [rule, deep, frozen],
            [rule, arbiter, frozen],
            [deep, rule, arbiter],
        ]
    )


def evaluation_lineups() -> list[list[str]]:
    return [
        ["rule_based_agent"] * 3,
        ["rule_based_agent", "coin_collector_agent", "peaceful_agent"],
        ["imp_li_deep_killer", "imp_alii_arbiter", "rule_based_agent"],
        ["imp_li_deep_killer", "imp_alii_arbiter", FROZEN_AGENT037],
        ["imp_li_deep_killer"] * 3,
    ]


def cpu_environment() -> dict[str, str]:
    result = os.environ.copy()
    result.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
    })
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("warm", "scratch"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--eval-scenario-workers", type=int, default=2)
    parser.add_argument(
        "--agent037-root", type=Path, default=DEFAULT_AGENT037_ROOT
    )
    parser.add_argument("--agent037-episode", type=int, default=1050)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds <= 300:
        parser.error("Agent 038 population training needs more than 300 rounds")

    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    runtime = run / "stage" / "runtime"
    copy_runtime(runtime)
    shutil.copytree(
        SOURCE / "agent_code",
        runtime / "agent_code",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    imported = {meta["package"]: meta for meta in CANDIDATES}
    imported_records = {}
    for name in IMPORTED_ROSTER:
        checkpoint = stage_package(imported[name], runtime)
        imported_records[name] = {
            "source": str((IMPORTED / imported[name]["source"]).resolve()),
            "checkpoint": str(checkpoint),
            "sha256": checksum(checkpoint),
        }

    episode_name = f"episode_{args.agent037_episode:04d}.pkl"
    frozen_source = (
        args.agent037_root / "seed_0" / "checkpoints" / episode_name
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
                args.agent037_root / f"seed_{seed}" / "checkpoints" / episode_name
            ).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = initial_root / f"seed_{seed}" / "episode_0000.pkl"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    f"agent_code.{AGENT}.checkpoint",
                    str(source),
                    str(destination),
                ],
                cwd=runtime,
                env=cpu_environment(),
                check=True,
            )
            initial_checkpoints.append(destination)
            warm_records.append({
                "seed": seed,
                "source": str(source),
                "source_sha256": checksum(source),
                "destination": str(destination),
                "destination_sha256": checksum(destination),
            })

    train_lineups = training_lineups()
    eval_lineups = evaluation_lineups()
    command = [
        sys.executable,
        "run_combat_training.py",
        "--agent", AGENT,
        "--curriculum", "agent038-population",
        "--diagnostics",
        "--rounds", str(args.rounds),
        "--board-stride", str(args.rounds),
        "--max-steps", "400",
        "--eval-every", str(args.eval_every),
        "--eval-workers", str(args.eval_workers),
        "--eval-scenario-workers", str(args.eval_scenario_workers),
        "--eval-seeds", "34000", "34001", "34002", "34003",
        "--eval-seats", "0", "1", "2", "3",
        "--seeds", *map(str, args.seeds),
        "--parallel-seeds",
        "--output", str(run / "training"),
    ]
    for lineup in train_lineups:
        command.extend(["--classic-lineup", *lineup])
    for lineup in eval_lineups:
        command.extend(["--league-eval-lineup", *lineup])
    if initial_checkpoints:
        command.extend([
            "--initial-checkpoints",
            *(str(path) for path in initial_checkpoints),
        ])

    os.symlink(SOURCE / ".git", runtime / ".git")
    provenance = {
        "experiment": "agent038_warm_vs_scratch",
        "mode": args.mode,
        "agent": AGENT,
        "source": str(SOURCE.resolve()),
        "source_head": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": subprocess.run(
            ["git", "-C", str(SOURCE), "status", "--short"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.splitlines(),
        "cuda_visible_devices": "",
        "seeds": args.seeds,
        "rounds": args.rounds,
        "board_stride": args.rounds,
        "curriculum": "agent038-population",
        "generated_round_shares_after_round_300": {
            "classic": 0.70,
            "coin-heaven": 0.15,
            "loot-crate": 0.15,
        },
        "imported_training_roster": imported_records,
        "held_out_imported_agents": [
            "imp_li_sarsa_lambda",
            "imp_li_double_q",
            "imp_alii_sentinel",
            "imp_alii_overlord",
            "harvy",
        ],
        "frozen_agent037": {
            "source": str(frozen_source),
            "sha256": checksum(frozen_source),
            "runtime_destination": str(frozen_destination),
        },
        "warm_initializers": warm_records,
        "training_lineups": train_lineups,
        "evaluation_lineups": eval_lineups,
        "development_boards": [34000, 34001, 34002, 34003],
        "command": command,
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Staged Agent 038 {args.mode} branch at {runtime}", flush=True)
    if args.stage_only:
        return
    subprocess.run(
        command, cwd=runtime, env=cpu_environment(), check=True
    )


if __name__ == "__main__":
    main()

