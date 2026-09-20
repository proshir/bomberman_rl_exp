"""Stage and launch the Agent 033 two-member population experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from run_imported_stage2 import CANDIDATES, IMPORTED, SOURCE, copy_runtime, stage_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()

    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    runtime = run / "stage" / "runtime"
    copy_runtime(runtime)
    shutil.copytree(
        SOURCE / "agent_code", runtime / "agent_code", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    imported = {meta["package"]: meta for meta in CANDIDATES}
    deep_killer = imported["imp_li_deep_killer"]
    checkpoint = stage_package(deep_killer, runtime)
    command = [
        sys.executable, "run_population_training.py",
        "--output", str(run / "training"),
        "--rounds", str(args.rounds),
        "--max-steps", str(args.max_steps),
    ]
    (run / "provenance.json").write_text(json.dumps({
        "agent": "Agent_033_population_replay_agent",
        "frozen_agent": "Agent_033_population_frozen_agent",
        "external_opponents": [
            "imp_li_deep_killer", "peaceful_agent", "coin_collector_agent",
            "rule_based_agent",
        ],
        "deep_killer_source": str((IMPORTED / deep_killer["source"]).resolve()),
        "deep_killer_checkpoint": str(checkpoint.resolve()),
        "rounds": args.rounds,
        "max_steps": args.max_steps,
        "command": command,
    }, indent=2) + "\n")
    print(f"Staged population runtime at {runtime}", flush=True)
    subprocess.run(command, cwd=runtime, check=True)


if __name__ == "__main__":
    main()
