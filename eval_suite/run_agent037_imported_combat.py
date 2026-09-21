#!/usr/bin/env python3
"""Evaluate Agent037 against the strongest imported combat policy.

The staged runtime keeps the repository runtime untouched while making Agent037
and its local feature/model dependencies available alongside the imported
policy.  The benchmark protocol matches ``run_imported_combat.py``: Classic
boards 32000--32007, all four seats, a 400-step limit, and CPU-only workers.
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
    PROJECT,
    SOURCE,
    copy_runtime,
    stage_package,
)


AGENT = "Agent_037_tournament_fast_ddqn_agent"
IMPORTED_AGENT = "imp_li_deep_killer"
CHECKPOINT = Path(
    "/export/scratch/salitanl/agent037_tournament_fast_1200_20260921_v4/"
    "seed_0/checkpoints/episode_1050.pkl"
)
DEPENDENCIES = (
    AGENT,
    "Agent_025_combat_ddqn_short_cycle_agent",
    "Agent_030_combat_ddqn_escape_replay_agent",
    "Agent_032_combat_ddqn_optimized_features_agent",
    "combat_dqn_agent",
    "combat_dqn_r_topology_agent",
    "combat_fqi_agent",
    "combat_fqi_history_antistag_agent",
    "combat_fqi_history_antistag_topology_agent",
)


def sha256(path: Path) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--game-workers", type=int, default=4)
    args = parser.parse_args()

    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    stage = run / "stage"
    runtime = stage / "runtime"
    suite = stage / "eval_suite"
    runtime.mkdir(parents=True)
    suite.mkdir(parents=True)
    copy_runtime(runtime)

    for name in ("rule_based_agent", "peaceful_agent"):
        shutil.copytree(SOURCE / "agent_code" / name,
                        runtime / "agent_code" / name)
    for name in DEPENDENCIES:
        shutil.copytree(SOURCE / "agent_code" / name,
                        runtime / "agent_code" / name)

    meta = next(item for item in CANDIDATES
                if item["package"] == IMPORTED_AGENT)
    imported_checkpoint = stage_package(meta, runtime)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    os.symlink(runtime, stage / "src")
    shutil.copy2(PROJECT / "eval_suite" / "run_suite.py",
                 suite / "run_suite.py")

    lineups = [
        {
            "name": "mixed_top_imported",
            "scenario": "classic",
            "opponents": ["rule_based_agent", IMPORTED_AGENT,
                           "peaceful_agent"],
        },
        {
            "name": "three_top_imported",
            "scenario": "classic",
            "opponents": [IMPORTED_AGENT] * 3,
        },
    ]
    manifest = {
        "board_seeds": list(range(32000, 32008)),
        "agent_seeds": [0],
        "seats": [0, 1, 2, 3],
        "scenarios": ["classic"],
        "max_steps": 400,
        "diagnostics": True,
        "opponent_lineups": lineups,
        "candidates": [{
            "name": AGENT,
            "agent": AGENT,
            "checkpoints": [str(CHECKPOINT)],
        }],
    }
    manifest_path = suite / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "provenance.json").write_text(json.dumps({
        "python": sys.executable,
        "source": str(SOURCE.resolve()),
        "cuda_visible_devices": "",
        "agent037_checkpoint": str(CHECKPOINT),
        "agent037_checkpoint_sha256": sha256(CHECKPOINT),
        "imported_checkpoint": str(imported_checkpoint),
        "imported_checkpoint_sha256": sha256(imported_checkpoint),
        "manifest": manifest,
    }, indent=2) + "\n")

    smoke = run / "smoke"
    smoke.mkdir()
    command = [
        sys.executable, str(runtime / "run_benchmark.py"),
        "--agents", AGENT, "--opponents", "rule_based_agent", IMPORTED_AGENT,
        "peaceful_agent", "--model-path", str(CHECKPOINT),
        "--scenario", "classic", "--seeds", "32000", "--agent-seeds", "0",
        "--seats", "0", "--max-steps", "30", "--metric", "score",
        "--diagnostics", "--output", str(smoke / AGENT),
    ]
    smoke_result = subprocess.run(command, cwd=stage, env=env(), text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
    (smoke / "smoke.log").write_text(smoke_result.stdout)
    if smoke_result.returncode != 0:
        raise RuntimeError("Agent037/imported smoke test failed; see smoke.log")
    (run / "smoke_passed.json").write_text(json.dumps({
        "passed": True,
        "output": str(smoke / AGENT),
    }, indent=2) + "\n")

    command = [
        sys.executable, str(suite / "run_suite.py"),
        "--manifest", str(manifest_path), "--output", str(run / "full"),
        "--python", sys.executable, "--parallel", str(args.parallel),
        "--game-workers", str(args.game_workers),
    ]
    (run / "full_command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, cwd=suite, env=env(), text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (run / "full.log").write_text(result.stdout)
    (run / "exit_code.txt").write_text(str(result.returncode) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
