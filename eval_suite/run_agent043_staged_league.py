#!/usr/bin/env python3
"""Stage strong frozen opponents and launch Agent 043's 900-round league."""

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


AGENT = "Agent_043_novelty_credit_ddqn_agent"
IMPORTED_ROSTER = ("imp_alii_arbiter", "imp_li_deep_killer")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def training_lineups() -> list[list[str]]:
    return [
        ["rule_based_agent"] * 3,
        ["imp_alii_arbiter", "imp_li_deep_killer", "rule_based_agent"],
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=900)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-scenario-workers", type=int, default=8)
    parser.add_argument(
        "--initial-checkpoints", nargs="+", type=Path,
        help="One Agent043 episode-0650 checkpoint per training seed.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Suppress frozen evaluations during online continuation.",
    )
    parser.add_argument(
        "--memory-retention", action="store_true",
        help=("After episode 300, retain 10 percent Coin Heaven and 10 percent "
              "Loot Crate alongside 80 percent complete four-player Classic "
              "games."),
    )
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds <= 300:
        parser.error("the staged league requires combat rounds after episode 300")
    if args.initial_checkpoints is not None:
        if len(args.initial_checkpoints) != len(args.seeds):
            parser.error(
                "--initial-checkpoints requires one path per training seed"
            )
        for checkpoint in args.initial_checkpoints:
            if not checkpoint.is_file():
                parser.error(f"Initial checkpoint does not exist: {checkpoint}")
            if checkpoint.stem != "episode_0650":
                parser.error(
                    "Dataset warm-start checkpoints must be named episode_0650.pkl"
                )

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
            "source": str((IMPORTED / imported[name]["source"]).resolve()),
            "checkpoint": str(checkpoint),
            "sha256": checksum(checkpoint),
        }

    os.symlink(SOURCE / ".git", runtime / ".git")
    lineups = training_lineups()
    curriculum = (
        "agent043-staged-league-memory-retention"
        if args.memory_retention else "agent043-staged-league"
    )
    command = [
        sys.executable,
        "run_combat_training.py",
        "--agent", AGENT,
        "--curriculum", curriculum,
        "--rounds", str(args.rounds),
        "--board-stride", str(args.rounds),
        "--max-steps", "400",
        "--eval-every", str(args.eval_every),
        "--eval-workers", str(args.eval_workers),
        "--eval-scenario-workers", str(args.eval_scenario_workers),
        "--eval-seeds", "34000", "34001", "34002", "34003",
        "34004", "34005", "34006", "34007",
        "--eval-seats", "0", "1", "2", "3",
        "--seeds", *(str(seed) for seed in args.seeds),
        "--parallel-seeds",
        "--diagnostics",
        "--output", str(run / "training"),
    ]
    if args.initial_checkpoints is not None:
        command.extend([
            "--initial-checkpoints",
            *(str(path.resolve()) for path in args.initial_checkpoints),
        ])
    if args.no_eval:
        command.append("--no-eval")
    for lineup in lineups:
        command.extend(["--classic-lineup", *lineup])
        command.extend(["--league-eval-lineup", *lineup])

    provenance = {
        "experiment": (
            "agent043_staged_league_memory_retention_900"
            if args.memory_retention else "agent043_staged_league_900"
        ),
        "agent": AGENT,
        "curriculum": curriculum,
        "source": str(SOURCE.resolve()),
        "source_head": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "source_dirty": subprocess.run(
            ["git", "-C", str(SOURCE), "status", "--short"],
            text=True, stdout=subprocess.PIPE, check=True,
        ).stdout.splitlines(),
        "phases": [
            {"episodes": [1, 100], "distribution": {"coin-heaven": 1.0}},
            {"episodes": [101, 300], "distribution": {
                "coin-heaven": 0.5, "loot-crate": 0.5,
            }},
            {"episodes": [301, args.rounds], "distribution": (
                {"classic": 0.8, "coin-heaven": 0.1, "loot-crate": 0.1}
                if args.memory_retention else {"classic": 1.0}
            )},
        ],
        "training_lineups": lineups,
        "evaluation_lineups": lineups,
        "imported_opponents": imported_records,
        "held_out_imported_agents": [
            "imp_li_sarsa_lambda", "imp_li_double_q",
            "imp_alii_sentinel", "imp_alii_overlord", "harvy",
        ],
        "seeds": args.seeds,
        "initial_checkpoints": (
            [str(path.resolve()) for path in args.initial_checkpoints]
            if args.initial_checkpoints is not None else None
        ),
        "no_eval": args.no_eval,
        "development_boards": list(range(34000, 34008)),
        "command": command,
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Staged Agent 043 league runtime at {runtime}", flush=True)
    if args.stage_only:
        return 0
    return subprocess.run(command, cwd=runtime, env=cpu_environment()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
