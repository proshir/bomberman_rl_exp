"""Stage frozen imported opponents and train the one-learner league pilot.

The stage and training output live under a fresh scratch directory. Earlier
agents and imported source packages are never modified. Use --mode builtin
for a matched schedule/control with only the three built-in opponents.
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


BUILTINS = ("peaceful_agent", "coin_collector_agent", "rule_based_agent")
IMPORTS = ("imp_li_deep_killer", "imp_li_sarsa_lambda", "imp_li_double_q",
           "imp_alii_arbiter", "imp_alii_sentinel")
AGENT = "Agent_032_combat_ddqn_optimized_features_agent"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineups(roster: tuple[str, ...]) -> list[list[str]]:
    """Eight reproducible slots each for one, two, and three opponents."""
    result = []
    count = len(roster)
    for size in (1, 2, 3):
        for index in range(8):
            if count == 8:
                offsets = (0, 3, 5)[:size]
            else:
                offsets = tuple(range(size))
            result.append([roster[(index + offset) % count]
                           for offset in offsets])
    return result


def fix_sentinel_loader(callbacks: Path) -> None:
    old = ("if isinstance(sd, dict) and 'state_dict' in sd:\n"
           "                    sd = sd['state_dict']")
    new = ("if isinstance(sd, dict):\n"
           "                    sd = sd.get('state_dict', sd.get('q_net', sd))")
    source = callbacks.read_text()
    if source.count(old) != 1:
        raise RuntimeError("Sentinel checkpoint loader changed upstream")
    callbacks.write_text(source.replace(old, new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("league", "builtin"), default="league")
    parser.add_argument("--agent", default=AGENT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--eval-scenario-workers", type=int, default=4)
    parser.add_argument("--parallel-seeds", action="store_true")
    parser.add_argument("--initial-checkpoints", nargs="+", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 300:
        parser.error("A league pilot needs more than 300 rounds to reach combat")
    if args.initial_checkpoints and len(args.initial_checkpoints) != len(args.seeds):
        parser.error("Supply one initial checkpoint per seed")
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    stage = run / "stage"
    runtime = stage / "runtime"
    copy_runtime(runtime)
    shutil.copytree(SOURCE / "agent_code", runtime / "agent_code",
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    imported = {meta["package"]: meta for meta in CANDIDATES}
    checkpoints = {}
    for name in IMPORTS:
        checkpoint = stage_package(imported[name], runtime)
        checkpoints[name] = {
            "source": str((IMPORTED / imported[name]["source"]).resolve()),
            "checkpoint": str(checkpoint),
            "sha256": checksum(checkpoint),
        }
    fix_sentinel_loader(runtime / "agent_code" / "imp_alii_sentinel" / "callbacks.py")
    # The runner records the original commit and hashes its staged source.
    os.symlink(SOURCE / ".git", runtime / ".git")

    roster = BUILTINS + IMPORTS if args.mode == "league" else BUILTINS
    training_lineups = lineups(roster)
    evaluation_lineups = [
        ["rule_based_agent"] * 3,
        ["rule_based_agent", "imp_li_deep_killer", "imp_alii_arbiter"],
    ]
    command = [sys.executable, "run_combat_training.py",
               "--agent", args.agent, "--curriculum", "league-combat",
               "--rounds", str(args.rounds), "--max-steps", "400",
               "--eval-every", str(args.eval_every),
               "--eval-workers", str(args.eval_workers),
               "--eval-scenario-workers", str(args.eval_scenario_workers),
               "--eval-seeds", "32000", "32001", "32002", "32003",
               "--eval-seats", "0", "1", "2", "3",
               "--seeds", *map(str, args.seeds),
               "--output", str(run / "training")]
    if args.parallel_seeds:
        command.append("--parallel-seeds")
    for lineup in training_lineups:
        command.extend(["--classic-lineup", *lineup])
    for lineup in evaluation_lineups:
        command.extend(["--league-eval-lineup", *lineup])
    if args.initial_checkpoints:
        command.extend(["--initial-checkpoints",
                        *(str(path.resolve()) for path in args.initial_checkpoints)])
    (run / "provenance.json").write_text(json.dumps({
        "mode": args.mode, "source": str(SOURCE.resolve()),
        "source_head": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "imports": checkpoints,
        "staging_changes": {
            "imp_alii_sentinel": "unwrap q_net from best.pt in staged callback"
        },
        "roster": roster, "training_lineups": training_lineups,
        "evaluation_lineups": evaluation_lineups,
        "command": command,
    }, indent=2) + "\n")
    print(f"Staged {args.mode} roster at {runtime}", flush=True)
    if args.stage_only:
        return
    subprocess.run(command, cwd=runtime, check=True)


if __name__ == "__main__":
    main()
