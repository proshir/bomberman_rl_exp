#!/usr/bin/env python3
"""Stage strong frozen opponents and launch Agent 037 Final_finetune.

Final_finetune keeps the runner's solo warm-up/retention schedule, but every
non-solo episode is a complete four-player Classic game.  Imported opponents
are staged outside the repository runtime and are never modified in place.
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

from run_imported_stage2 import CANDIDATES, IMPORTED, SOURCE, copy_runtime, stage_package


AGENT = "Agent_037_tournament_fast_ddqn_agent"
DEPENDENCIES = (
    AGENT,
    "Agent_025_combat_ddqn_short_cycle_agent",
    "Agent_027_combat_ddqn_short_cycle_staged_replay_agent",
    "Agent_029_combat_ddqn_adversarial_window_agent",
    "Agent_030_combat_ddqn_escape_replay_agent",
    "Agent_032_combat_ddqn_optimized_features_agent",
    "combat_dqn_agent",
    "combat_dqn_r_topology_agent",
    "combat_fqi_agent",
    "combat_fqi_history_antistag_agent",
    "combat_fqi_history_antistag_topology_agent",
)
IMPORTED_CHOICES = {
    "imp_li_deep_killer",
    "imp_alii_arbiter",
}
FORBIDDEN = {"coin_collector_agent", "peaceful_agent"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env() -> dict[str, str]:
    result = os.environ.copy()
    result.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    })
    return result


def build_lineups(imported: list[str]) -> list[list[str]]:
    """Return complete three-opponent lineups, excluding weak built-ins."""
    lineups = [["rule_based_agent"] * 3]
    if imported:
        mixed = list(imported) + ["rule_based_agent"] * (3 - len(imported))
        lineups.append(mixed[:3])
    return lineups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-scenario-workers", type=int, default=8)
    parser.add_argument("--parallel-seeds", action="store_true")
    parser.add_argument("--initial-checkpoints", nargs="+", type=Path,
                        required=True)
    parser.add_argument("--imported-opponents", nargs="*",
                        choices=sorted(IMPORTED_CHOICES),
                        default=["imp_li_deep_killer", "imp_alii_arbiter"],
                        help="Strong frozen imported opponents; omit values for rule-only.")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    if len(args.initial_checkpoints) != len(args.seeds):
        parser.error("supply one initial checkpoint per training seed")
    if any(not path.is_file() for path in args.initial_checkpoints):
        parser.error("all initial checkpoints must exist")
    if any(name in FORBIDDEN for name in args.imported_opponents):
        parser.error("Final_finetune excludes Coin Collector and Peaceful")

    lineups = build_lineups(args.imported_opponents)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    stage = run / "stage"
    runtime = stage / "runtime"
    runtime.mkdir(parents=True)
    copy_runtime(runtime)
    for name in DEPENDENCIES + ("rule_based_agent",):
        shutil.copytree(SOURCE / "agent_code" / name,
                        runtime / "agent_code" / name)

    imported_meta = {item["package"]: item for item in CANDIDATES}
    staged = {}
    for name in args.imported_opponents:
        checkpoint = stage_package(imported_meta[name], runtime)
        staged[name] = {
            "source": str((IMPORTED / imported_meta[name]["source"]).resolve()),
            "checkpoint": str(checkpoint),
            "sha256": checksum(checkpoint),
        }

    # run_combat_training records its source commit and hashes; expose the
    # repository metadata from the staged runtime as well.
    os.symlink(SOURCE / ".git", runtime / ".git")
    command = [
        sys.executable, "run_combat_training.py",
        "--agent", AGENT,
        "--curriculum", "Final_finetune",
        "--rounds", str(args.rounds),
        "--board-stride", str(args.rounds),
        "--max-steps", "400",
        "--eval-every", str(args.eval_every),
        "--eval-workers", str(args.eval_workers),
        "--eval-scenario-workers", str(args.eval_scenario_workers),
        "--eval-seeds", "34000", "34001", "34002", "34003",
        "--eval-seats", "0", "1", "2", "3",
        "--seeds", *(str(seed) for seed in args.seeds),
        "--initial-checkpoints",
        *(str(path.resolve()) for path in args.initial_checkpoints),
        "--diagnostics",
        "--output", str(run / "training"),
    ]
    if args.parallel_seeds:
        command.append("--parallel-seeds")
    for lineup in lineups:
        command.extend(["--classic-lineup", *lineup])
        command.extend(["--league-eval-lineup", *lineup])

    (run / "provenance.json").write_text(json.dumps({
        "mode": "Final_finetune",
        "source": str(SOURCE.resolve()),
        "source_head": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "agent": AGENT,
        "lineups": lineups,
        "imported_opponents": staged,
        "initial_checkpoints": [str(path.resolve())
                                for path in args.initial_checkpoints],
        "command": command,
    }, indent=2) + "\n")
    print(f"Staged Final_finetune runtime at {runtime}", flush=True)
    if args.stage_only:
        return 0
    return subprocess.run(command, cwd=runtime, env=env()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
