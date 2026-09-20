"""Stage frozen imported agents and run the fixed Classic comparison.

The stage is isolated under scratch. Imported callbacks and checkpoints are
copied; Sentinel's staged callback gets a narrowly scoped checkpoint-wrapper
fix, recorded in provenance. The same opponents and boards are used for every
candidate. Smoke failures are recorded and excluded.
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
    CANDIDATES, IMPORTED, PROJECT, SOURCE, copy_runtime, stage_package,
)


NAMES = (
    "imp_li_deep_killer",
    "imp_li_sarsa_lambda",
    "imp_li_double_q",
    "imp_alii_overlord",
    "imp_alii_sentinel",
    "imp_alii_arbiter",
    "imp_piscih_bomb_voyage",
)
LINEUPS = (
    {"name": "rule_based", "scenario": "classic",
     "opponents": ["rule_based_agent"]},
    {"name": "three_rule_based", "scenario": "classic",
     "opponents": ["rule_based_agent"] * 3},
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--game-workers", type=int, default=2)
    parser.add_argument("--candidates", nargs="+", default=[*NAMES, "harvy"])
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    stage = run / "stage"
    runtime = stage / "runtime"
    suite = stage / "eval_suite"
    runtime.mkdir(parents=True)
    suite.mkdir(parents=True)
    copy_runtime(runtime)
    for name in ("rule_based_agent", "coin_collector_agent",
                 "peaceful_agent"):
        shutil.copytree(SOURCE / "agent_code" / name,
                        runtime / "agent_code" / name)

    requested = set(args.candidates)
    unknown = requested - set(NAMES) - {"harvy"}
    if unknown:
        parser.error(f"Unknown candidates: {sorted(unknown)}")
    selected = {meta["package"]: meta for meta in CANDIDATES
                if meta["package"] in requested}
    rows = []
    for name in NAMES:
        if name not in requested:
            continue
        meta = selected[name]
        checkpoint = stage_package(meta, runtime)
        if name == "imp_alii_sentinel":
            # The committed best.pt wraps the policy weights under q_net.
            # The upstream callback only unwraps state_dict and otherwise
            # silently leaves all eight model parameters at initialization.
            callbacks = runtime / "agent_code" / name / "callbacks.py"
            old = ("if isinstance(sd, dict) and 'state_dict' in sd:\n"
                   "                    sd = sd['state_dict']")
            new = ("if isinstance(sd, dict):\n"
                   "                    sd = sd.get('state_dict', sd.get('q_net', sd))")
            source = callbacks.read_text()
            if source.count(old) != 1:
                raise RuntimeError("Sentinel checkpoint loader changed upstream")
            callbacks.write_text(source.replace(old, new))
        rows.append((name, checkpoint,
                     runtime / "agent_code" / name / "callbacks.py"))
    if "harvy" in requested:
        harvy_source = IMPORTED / "harvy"
        harvy_dest = runtime / "agent_code" / "harvy"
        shutil.copytree(harvy_source, harvy_dest)
        rows.append(("harvy", harvy_source / "my-saved-model.pt",
                     harvy_dest / "callbacks.py"))

    os.symlink(runtime, stage / "src")
    shutil.copy2(PROJECT / "eval_suite" / "run_suite.py",
                 suite / "run_suite.py")
    manifest = {
        "board_seeds": list(range(32000, 32008)),
        "agent_seeds": [0],
        "seats": [0, 1, 2, 3],
        "scenarios": ["classic"],
        "max_steps": 400,
        "diagnostics": True,
        "opponent_lineups": list(LINEUPS),
        "candidates": [{"name": name, "agent": name,
                        "checkpoints": [str(checkpoint)]}
                       for name, checkpoint, _ in rows],
    }
    manifest_path = suite / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    provenance = {
        "python": sys.executable,
        "source": str(SOURCE.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "candidate_files": {
            name: {"checkpoint": str(checkpoint),
                   "checkpoint_sha256": sha256(checkpoint),
                   "callbacks_sha256": sha256(callbacks)}
            for name, checkpoint, callbacks in rows
        },
        "staging_changes": {
            "imp_alii_sentinel": "unwrap q_net from the frozen best.pt checkpoint"
        } if "imp_alii_sentinel" in requested else {},
        "manifest": manifest,
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    # One short combat game verifies imports and catches hard setup failures.
    smoke = run / "smoke"
    smoke.mkdir()
    passed = []
    for name, checkpoint, _ in rows:
        output = smoke / name
        cmd = [sys.executable, str(runtime / "run_benchmark.py"),
               "--agents", name, "--opponents", "rule_based_agent",
               "--model-path", str(checkpoint), "--scenario", "classic",
               "--seeds", "32000", "--agent-seeds", "0", "--seats", "0",
               "--max-steps", "30", "--metric", "score", "--output",
               str(output)]
        result = subprocess.run(cmd, cwd=stage, text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (smoke / f"{name}.log").write_text(result.stdout)
        if result.returncode == 0 and (output / "summary.json").is_file():
            passed.append(name)
        else:
            print(f"smoke failed: {name}; see {smoke / f'{name}.log'}",
                  flush=True)
    (run / "smoke_passed.json").write_text(json.dumps(passed, indent=2) + "\n")
    manifest["candidates"] = [candidate for candidate in manifest["candidates"]
                              if candidate["name"] in passed]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not passed:
        raise RuntimeError("No candidate passed combat smoke evaluation")
    print(f"Combat smoke passed: {', '.join(passed)}", flush=True)

    command = [sys.executable, str(suite / "run_suite.py"),
               "--manifest", str(manifest_path), "--output", str(run / "full"),
               "--python", sys.executable, "--parallel", str(args.parallel),
               "--game-workers", str(args.game_workers)]
    (run / "full_command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, cwd=suite, check=True)


if __name__ == "__main__":
    main()
