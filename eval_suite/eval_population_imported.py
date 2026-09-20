"""Evaluate Agent 033 checkpoints against imported-agent lineups."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


IMPORTED = (
    "imp_li_deep_killer", "imp_li_sarsa_lambda", "imp_li_double_q",
    "imp_alii_arbiter", "imp_alii_sentinel",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint0", type=Path, required=True)
    parser.add_argument("--checkpoint1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for member, checkpoint in enumerate((args.checkpoint0, args.checkpoint1)):
        for opponent in IMPORTED:
            name = f"classic__{opponent}__rule_based_agent__peaceful_agent"
            output = args.output / f"member_{member}" / name
            command = [
                sys.executable, str(args.runtime / "run_benchmark.py"),
                "--agents", "Agent_033_population_replay_agent",
                "--scenario", "classic", "--model-path", str(checkpoint),
                "--max-steps", "400", "--seeds", "32000", "32001",
                "32002", "32003", "--agent-seeds", "0", "--seats",
                "0", "1", "2", "3", "--batch-size", "1", "--parallel", "2",
                "--metric", "score", "--output", str(output),
                "--opponents", opponent, "rule_based_agent", "peaceful_agent",
            ]
            jobs.append((member, opponent, command))

    def run(job):
        member, opponent, command = job
        subprocess.run(command, cwd=args.runtime, check=True)
        name = f"classic__{opponent}__rule_based_agent__peaceful_agent"
        summary = args.output / f"member_{member}" / name / "summary.json"
        return {"member": member, "opponent": opponent,
                "summary": json.loads(summary.read_text())}

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(run, jobs))
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
