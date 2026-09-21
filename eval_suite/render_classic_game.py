"""Render one four-player classic game to an MP4.

The benchmark normally runs headless and records only statistics.  This
utility uses the same ``BenchmarkWorld`` and callback loading path as the
benchmark, but keeps rendering enabled so a selected game can be inspected
visually.  The candidate occupies ``--seat`` and plays against the opponent
names supplied with ``--opponents``.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import settings as s  # noqa: E402
from environment import GUI  # noqa: E402
from run_benchmark import BenchmarkWorld  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True,
                        help="candidate agent package name")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", required=True,
                        help="opponent agent package names")
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=range(4), required=True)
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.opponents) != 3:
        parser.error("classic rendering requires exactly three opponents")

    # Each candidate package exposes the same module-level MODEL_PATH hook
    # used by run_benchmark.py.  Agent 040's callbacks then forwards this path
    # to the shared DDQN loader during setup.
    callbacks = importlib.import_module(f"agent_code.{args.agent}.callbacks")
    callbacks.MODEL_PATH = args.model_path.resolve()

    # Match run_benchmark.play_game: rule-based policies consume Python's
    # global RNG, while the board itself uses the explicit world seed.
    random.seed(args.agent_seed)
    s.MAX_STEPS = args.max_steps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / f"{args.output.stem}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    world_args = SimpleNamespace(
        no_gui=False,
        fps=10,
        turn_based=False,
        update_interval=0.01,
        save_replay=False,
        replay=None,
        make_video=str(args.output),
        continue_without_training=True,
        log_dir=str(log_dir),
        save_stats=False,
        match_name=f"render_{args.output.stem}",
        seed=args.seed,
        agent_seed=args.agent_seed,
        silence_errors=False,
        scenario=args.scenario,
        seat=args.seat,
    )
    agents = [(args.agent, False)] + [(name, False)
                                      for name in args.opponents]
    world = BenchmarkWorld(world_args, agents)
    gui = GUI(world)
    world.new_round()
    while world.running:
        gui.render()
        world.do_step()
    gui.make_video()
    world.end()
    print(args.output)


if __name__ == "__main__":
    main()
