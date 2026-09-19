#!/usr/bin/env python3
"""Stage and run the frozen imported-agent Stage-2 evaluation suite.

The staging tree is deliberately outside the repository runtime.  Imported
callbacks are copied unchanged; only checkpoint filenames/config paths are
redirected where the original callback hard-codes them.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src"
IMPORTED = PROJECT / "imported_agents"
PYTHON = Path(sys.executable)


def candidate(source: str, package: str, checkpoint: str, *,
              override: str | None = None, config_model: str | None = None,
              q_table: str | None = None) -> dict:
    return {
        "source": source,
        "package": package,
        "checkpoint": checkpoint,
        "override": override,
        "config_model": config_model,
        "q_table": q_table,
    }


CANDIDATES = [
    candidate("3j14__bomberman_rl__strong_agent", "imp_3j14_strong",
              "multimulti_best_yet.pt", override="multimulti.pt"),
    candidate("Alii-Khaled__BomberMan__arbiter", "imp_alii_arbiter",
              "my-saved-model.pt"),
    candidate("Alii-Khaled__BomberMan__sentinel", "imp_alii_sentinel",
              "checkpoints/best.pt", override="my-saved-model.pt"),
    candidate("Alii-Khaled__BomberMan__overlord", "imp_alii_overlord",
              "my-saved-model.pt"),
    candidate("FFFROZEN090__bomberman_rl__Yu_policy_agent", "imp_yu_policy",
              "best_model/III-III_SFF_seq_1_layer_2_alpha_0_hidden_32_55000.pt",
              override="checkpoints/III-III_SFF_seq_1_layer_2_alpha_0_hidden_32_55000.pt"),
    candidate("Li-Jesse-Jiaze_MLE_project_bomberman__dqn", "imp_li_dqn",
              "my-saved-model.pt"),
    candidate("cciao_bomberman_rl__expert_rl_w", "imp_cciao_expert",
              "models/model-1000"),
    candidate("ericgoldclub_bomberman_rl__kill_agent", "imp_eric_kill",
              "kill_agent_saved_model_best.pt", override="kill_agent_saved_model.pt"),
    candidate("ericgoldclub_bomberman_rl__loot_crate_agent", "imp_eric_loot",
              "loot_crate_agent_saved_model.pt"),
    candidate("itisacloud_bomberman_rl__GlasHoch_Rangers", "GlasHoch_Rangers",
              "models/GlasHoch_long_runer_9th_batch_32_lr_e5_1_sync_500_no_imitation_classic_default_1050000.pth",
              config_model="GlasHoch_long_runer_9th_batch_32_lr_e5_1_sync_500_no_imitation_classic_default_1050000.pth"),
    candidate("ivo-1_bomberman_rl__coli_agent", "imp_ivo_coli",
              "q_tables/q_table-2022-04-02T02:55:49.npy",
              q_table="q_table-2022-04-02T02:55:49.npy"),
    candidate("piscih__bomberman_rl__Bomb_Voyage", "imp_piscih_bomb_voyage",
              "dqn_model.pt"),
    candidate("Li-Jesse-Jiaze_MLE_project_bomberman__deep_learning_killer",
              "imp_li_deep_killer", "q_table.json"),
    candidate("Li-Jesse-Jiaze_MLE_project_bomberman__double_q",
              "imp_li_double_q", "q_table_1.json"),
    candidate("Li-Jesse-Jiaze_MLE_project_bomberman__sarsa_lambda",
              "imp_li_sarsa_lambda", "q_table.json"),
    candidate("nilskre__bomberman_rl__big_bertha_v1", "big_bertha_v1",
              "models/26_03_2021_21_38_41"),
]


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


def copy_runtime(runtime: Path) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    for path in SOURCE.iterdir():
        if path.name == "agent_code":
            continue
        if path.is_file() and path.suffix == ".py":
            shutil.copy2(path, runtime / path.name)
        elif path.name in {"assets", "replays", "screenshots"}:
            target = runtime / path.name
            if path.is_dir() and not target.exists():
                os.symlink(path, target)
    (runtime / "agent_code").mkdir()
    (runtime / "agent_code" / "__init__.py").write_text("")


def replace_with_link(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target.resolve(), path)


def stage_package(meta: dict, runtime: Path) -> Path:
    source = IMPORTED / meta["source"]
    destination = runtime / "agent_code" / meta["package"]
    shutil.copytree(source, destination, symlinks=True)
    checkpoint = (source / meta["checkpoint"]).resolve()

    if meta["override"]:
        replace_with_link(destination / meta["override"], checkpoint)

    if meta["package"] == "imp_yu_policy":
        # The original callback loads MODEL_PATH and BEST_MODEL_PATH.  Make
        # both resolve to the selected frozen checkpoint without changing code.
        replace_with_link(
            destination / "checkpoints" / Path(meta["override"]).name,
            checkpoint,
        )
        config = destination / "config.py"
        config.write_text(config.read_text().replace("WANDB = True", "WANDB = False"))

    if meta["q_table"]:
        qdir = destination / "q_tables"
        if qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir()
        shutil.copy2(checkpoint, qdir / meta["q_table"])

    if meta["config_model"]:
        config = destination / "configs" / "default.yaml"
        text = config.read_text()
        lines = []
        for line in text.splitlines(True):
            if "load_path:" in line:
                indent = line[: len(line) - len(line.lstrip())]
                line = f'{indent}load_path: models/{meta["config_model"]}\n'
            lines.append(line)
        config.write_text("".join(lines))
        model_src = source / meta["checkpoint"]
        model_dst = destination / "models" / Path(meta["config_model"]).name
        replace_with_link(model_dst, model_src)

    return checkpoint


def write_manifest(path: Path, rows: list[tuple[dict, Path]]) -> None:
    candidates = []
    for meta, checkpoint in rows:
        candidates.append({
            "name": meta["package"],
            "agent": meta["package"],
            "checkpoints": [str(checkpoint)],
        })
    manifest = {
        "board_seeds": list(range(30000, 30008)),
        "agent_seeds": [0],
        "seats": [0, 1, 2, 3],
        "scenarios": ["coin-heaven", "loot-crate"],
        "max_steps": 400,
        "diagnostics": True,
        "candidates": candidates,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def smoke_one(args: tuple[dict, Path, Path, Path]) -> tuple[dict, bool, str]:
    meta, checkpoint, runtime, output = args
    out = output / meta["package"]
    command = [
        str(PYTHON), str(runtime / "run_benchmark.py"),
        "--agents", meta["package"], "--model-path", str(checkpoint),
        "--scenario", "coin-heaven", "--seeds", "30000", "--seats", "0",
        "--agent-seeds", "0", "--max-steps", "20", "--metric", "coins",
        "--diagnostics", "--output", str(out),
    ]
    result = subprocess.run(command, cwd=runtime.parent, env=env(),
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out.parent / f"{meta['package']}.log").write_text(result.stdout)
    return meta, result.returncode == 0, result.stdout[-4000:]


def main() -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(os.environ.get("IMPORTED_STAGE2_RUN_DIR",
                                 f"/export/scratch/{os.environ.get('USER', 'unknown')}/bomberman_imported_stage2_{stamp}"))
    stage = run_dir / "stage"
    runtime = stage / "runtime"
    eval_root = stage / "eval_suite"
    smoke_root = run_dir / "smoke"
    full_root = run_dir / "full"
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    smoke_root.mkdir(parents=True)
    copy_runtime(runtime)

    rows = []
    for meta in CANDIDATES:
        try:
            checkpoint = stage_package(meta, runtime)
            rows.append((meta, checkpoint))
        except Exception as exc:
            (run_dir / "staging_failures.jsonl").open("a").write(
                json.dumps({"candidate": meta, "error": repr(exc)}) + "\n"
            )

    aliases = {
        "strong_agent": "imp_3j14_strong",
        "dqn": "imp_li_dqn",
        "expert_rl_w": "imp_cciao_expert",
    }
    for alias, target in aliases.items():
        target_path = runtime / "agent_code" / target
        alias_path = runtime / "agent_code" / alias
        if target_path.exists() and not alias_path.exists():
            os.symlink(target_path, alias_path)

    smoke_args = [(meta, checkpoint, runtime, smoke_root)
                  for meta, checkpoint in rows]
    smoke_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(smoke_args) or 1)) as pool:
        for result in pool.map(smoke_one, smoke_args):
            smoke_results.append(result)
    (run_dir / "smoke_results.json").write_text(json.dumps([
        {"candidate": meta["package"], "passed": passed, "tail": tail}
        for meta, passed, tail in smoke_results
    ], indent=2) + "\n")

    passed_names = {meta["package"] for meta, passed, _ in smoke_results if passed}
    passed_rows = [(meta, checkpoint) for meta, checkpoint in rows
                   if meta["package"] in passed_names]
    write_manifest(eval_root / "manifest.json", passed_rows)
    shutil.copy2(PROJECT / "eval_suite" / "run_suite.py", eval_root / "run_suite.py")
    os.symlink(runtime, stage / "src")
    (run_dir / "provenance.json").write_text(json.dumps({
        "project": str(PROJECT),
        "source": str(SOURCE),
        "python": str(PYTHON),
        "cuda_visible_devices": "",
        "parallel": 16,
        "protocol": json.loads((eval_root / "manifest.json").read_text()),
        "staged_candidates": [meta for meta, _ in passed_rows],
    }, indent=2) + "\n")

    command = [
        str(PYTHON), str(eval_root / "run_suite.py"),
        "--manifest", str(eval_root / "manifest.json"),
        "--output", str(full_root), "--python", str(PYTHON),
        "--parallel", "16",
    ]
    with (run_dir / "full_command.txt").open("w") as handle:
        handle.write(" ".join(command) + "\n")
    result = subprocess.run(command, cwd=eval_root, env=env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    (run_dir / "full.log").write_text(result.stdout)
    (run_dir / "exit_code.txt").write_text(str(result.returncode) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
