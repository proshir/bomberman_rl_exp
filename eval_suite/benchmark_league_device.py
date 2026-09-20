"""Time the same short imported-opponent league segment on CPU or CUDA.

Run from a staged runtime directory with the same seed config/checkpoint for
both devices. Checkpoint evaluation is skipped, but actual games, features,
replay sampling, and optimizer updates are included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path.cwd()))

import run_combat_training as runner
from agent_code.combat_dqn_agent.model import DEVICE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--start-episode", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    if DEVICE.type != args.device:
        raise RuntimeError(f"Requested {args.device}, detected {DEVICE}")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    config = json.loads(args.config.read_text())
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    config.update(output=str(run), start_episode=args.start_episode,
                  rounds=args.start_episode + args.rounds,
                  resume_checkpoint=str(args.checkpoint.resolve()),
                  eval_every=args.rounds + args.start_episode + 1)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    runner.evaluate_all = lambda *unused: []
    started = perf_counter()
    runner.train(config)
    wall = perf_counter() - started
    records = [json.loads(line) for line in
               (run / "rounds.jsonl").read_text().splitlines()]
    steps = sum(row["steps"] for row in records)
    updates = records[-1]["optimizer_steps"] - records[0]["optimizer_steps"]
    result = {
        "device": args.device, "rounds": len(records), "steps": steps,
        "optimizer_updates_after_first_round": updates,
        "wall_seconds": wall,
        "training_seconds": records[-1]["training_seconds"],
        "training_ms_per_step": 1000 * records[-1]["training_seconds"] / steps,
        "first_episode": records[0]["episode"],
        "last_episode": records[-1]["episode"],
        "opponent_lineups": [row["opponent"] for row in records
                             if row["scenario"] == "classic"],
    }
    (run / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
