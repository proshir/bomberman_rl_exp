"""Evaluate both Agent 033 population members from one checkpoint round."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


SCENARIOS = [
    ("coin-heaven", "coins", []),
    ("loot-crate", "coins", []),
    ("classic__rule_based_agent__rule_based_agent__rule_based_agent", "score",
     ["rule_based_agent"] * 3),
    ("classic__rule_based_agent__imp_li_deep_killer__peaceful_agent", "score",
     ["rule_based_agent", "imp_li_deep_killer", "peaceful_agent"]),
]


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
        for name, metric, opponents in SCENARIOS:
            output = args.output / f"member_{member}" / name
            command = [
                sys.executable, str(args.runtime / "run_benchmark.py"),
                "--agents", "Agent_033_population_replay_agent",
                "--scenario", "coin-heaven" if name == "coin-heaven" else
                ("loot-crate" if name == "loot-crate" else "classic"),
                "--model-path", str(checkpoint), "--max-steps", "400",
                "--seeds", "32000", "32001", "32002", "32003",
                "--agent-seeds", "0", "--seats", "0", "1", "2", "3",
                "--batch-size", "1", "--parallel", "2", "--metric", metric,
                "--output", str(output),
            ]
            if opponents:
                command.extend(["--opponents", *opponents])
            jobs.append((member, name, command))

    def run(job):
        member, name, command = job
        subprocess.run(command, cwd=args.runtime, check=True)
        summary = args.output / f"member_{member}" / name / "summary.json"
        return {"member": member, "scenario": name,
                "summary": json.loads(summary.read_text())}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, jobs))
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
