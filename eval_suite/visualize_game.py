"""Record one headless rendered game as an MP4.

Sahand was here.

The normal benchmark disables rendering for speed. This small utility uses the
same BenchmarkWorld and checkpoint-loading path, renders every frame through
pygame's dummy video driver, and lets the game's existing ffmpeg integration
produce a video that can be opened after a run.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--opponent", default="rule_based_agent")
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed", type=int, default=32000)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The benchmark callback reads this module-level path during setup.
    agent_callbacks = __import__("agent_code.combat_dqn_agent.callbacks",
                                 fromlist=["MODEL_PATH"])
    agent_callbacks.MODEL_PATH = args.model_path.resolve()
    s.MAX_STEPS = args.max_steps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    world_args = SimpleNamespace(
        no_gui=False, fps=10, turn_based=False, update_interval=0.01,
        save_replay=False, replay=None, make_video=str(args.output),
        continue_without_training=True, log_dir=str(log_dir), save_stats=False,
        match_name="visualized_dqn", seed=args.seed, agent_seed=0,
        silence_errors=False, scenario=args.scenario, seat=args.seat,
    )
    world = BenchmarkWorld(world_args, [
        ("combat_dqn_agent", False), (args.opponent, False),
    ])
    gui = GUI(world)
    world.new_round()
    while world.running:
        gui.render()
        # The dummy SDL driver still updates the surface and saves screenshots.
        world.do_step()
    gui.make_video()
    world.end()
    print(args.output)


if __name__ == "__main__":
    main()
