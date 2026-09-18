"""Run the standardized pre-combat checkpoint evaluation suite.

Sahand was here.

The benchmark runner already handles deterministic board/seat construction and
per-game diagnostics. This orchestration layer only expands the manifest into
one benchmark invocation per candidate checkpoint and scenario, then combines
the resulting summaries without changing their values.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "src" / "run_benchmark.py"
DEFAULT_MANIFEST = Path(__file__).with_name("suite.json")


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def run_one(spec):
    """Run one independent benchmark subprocess."""
    index, command, run_name, run_output, checkpoint_index, checkpoint = spec
    print(f'[{index}] {run_name}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return {
        "run_name": run_name,
        "checkpoint_index": checkpoint_index,
        "checkpoint": checkpoint,
        "summary": load_json(run_output / "summary.json"),
    }


def run_suite(manifest_path: Path, output: Path, python: str,
              parallel: int = 1) -> None:
    manifest = load_json(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    specs = []
    scenarios = manifest.get("scenarios", [])
    lineups = manifest.get("opponent_lineups")
    if lineups is None:
        lineups = [{"name": "solo", "scenario": scenario, "opponents": []}
                   for scenario in scenarios]
    for candidate in manifest["candidates"]:
        checkpoints = candidate["checkpoints"]
        if len(checkpoints) != 3:
            raise ValueError(f'{candidate["name"]} must provide three checkpoints')
        for lineup in lineups:
            scenario = lineup.get("scenario", "classic")
            opponents = lineup.get("opponents", [])
            lineup_name = lineup["name"]
            metric = "score" if opponents else "coins"
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                checkpoint_path = (ROOT / checkpoint).resolve()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
                run_name = (f'{candidate["name"]}_{lineup_name}_'
                            f'{checkpoint_index}')
                run_output = output / run_name
                command = [
                    python, str(BENCHMARK),
                    "--agents", candidate["agent"],
                    "--opponents", *opponents,
                    "--model-path", str(checkpoint_path),
                    "--scenario", scenario,
                    "--seeds", *map(str, manifest["board_seeds"]),
                    "--agent-seeds", *map(str, manifest["agent_seeds"]),
                    "--seats", *map(str, manifest["seats"]),
                    "--max-steps", str(manifest["max_steps"]),
                    "--metric", metric,
                    "--diagnostics" if manifest.get("diagnostics") else "",
                    "--output", str(run_output),
                ]
                command = [part for part in command if part]
                specs.append((len(specs) + 1, command, run_name, run_output,
                              checkpoint_index, checkpoint))
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            records = list(executor.map(run_one, specs))
    else:
        records = [run_one(spec) for spec in specs]

    summaries = []
    record_index = 0
    for candidate in manifest["candidates"]:
        checkpoints = candidate["checkpoints"]
        for lineup in lineups:
            scenario = lineup.get("scenario", "classic")
            opponents = lineup.get("opponents", [])
            lineup_name = lineup["name"]
            candidate_summary = records[record_index:record_index + len(checkpoints)]
            record_index += len(checkpoints)
            summaries.append({
                "candidate": candidate["name"],
                "agent": candidate["agent"],
                "scenario": scenario,
                "lineup": lineup_name,
                "opponents": opponents,
                "checkpoints": [
                    {key: record[key] for key in
                     ("checkpoint_index", "checkpoint", "summary")}
                    for record in candidate_summary
                ],
            })
    result = {
        "manifest": str(manifest_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": summaries,
    }
    (output / "suite_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f'Suite summary saved to {output / "suite_summary.json"}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Run independent checkpoint evaluations concurrently.")
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(__file__).with_name("results") / stamp
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    run_suite(args.manifest.resolve(), args.output.resolve(), args.python,
              args.parallel)


if __name__ == "__main__":
    main()
