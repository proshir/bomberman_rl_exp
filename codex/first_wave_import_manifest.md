# First-wave imported agents

Imported on 2026-09-19 from shallow clones. Only the principal learned agents were copied into `bomberman_rl_exp/imported_agents`; repository history, template agents, support/pretraining agents, duplicates, logs, and unrelated files were excluded.

| Imported directory prefix | Source | Commit |
|---|---|---|
| `cciao_bomberman_rl__` | https://github.com/cciao/bomberman_rl | `7e7a8af376d5ab37176ec7a6d237291e52adb4a7` |
| `ericgoldclub_bomberman_rl__` | https://github.com/ericgoldclub/bomberman_rl | `fda8d42c0fb77b576b04340293d123c70f4e9c07` |
| `ivo-1_bomberman_rl__` | https://github.com/ivo-1/bomberman_rl | `d4efc460bcbe54a3303e69e1e0d7ec731ca907f2` |
| `Li-Jesse-Jiaze_MLE_project_bomberman__` | https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman | `a7fe5041b02548ce4438e502ca3adb11576bae75` |
| `itisacloud_bomberman_rl__` | https://github.com/itisacloud/bomberman_rl | `26ac1cf55a66b999e09921a9dd74eb9189f84f0c` |
| `KunkelAlexander_bomberman_rl__` | https://github.com/KunkelAlexander/bomberman_rl | `38eccb6f0fbfa954da3b916f6c1a9cf0b3f91adb` |
| `antonH22_cnn-based-dql__` | https://github.com/antonH22/cnn-based-dql | `06f245eb6500249e5b15210b6c9c2dfd50c9a926` |
| `Alii-Khaled_BomberMan__` | https://github.com/Alii-Khaled/BomberMan | `8e1afb88f689b1b674598744c3bdbbfb4a5ba171` |
| `cs224_bomberman_rl__` | https://github.com/cs224/bomberman_rl | `86b5f78710f4b7b1f77f165c38f027c4a26c1e43` |
| `3j14_bomberman_rl__` | https://github.com/3j14/bomberman_rl | `f6dd4db162bcb122562ba1161b96dd2eb0a920bf` |
| `FFFROZEN090_bomberman_rl__` | https://github.com/FFFROZEN090/bomberman_rl | `31a390cd58b5abb5c3d4beba903a952f0d174ed8` |
| `piscih_bomberman_rl__` | https://github.com/piscih/bomberman_rl | `61a1fe818560689659fdf85fd2539bdaf9706517` |
| `nilskre_bomberman_rl__` | https://github.com/nilskre/bomberman_rl | `7de7ae6fa327fee2333e6788370ecf6a5a030308` |

Imported agent directories:

- `cciao`: `expert_rl_w`
- `ericgoldclub`: `kill_agent`, `loot_crate_agent`
- `ivo-1`: `coli_agent`, `coli_agent_offline`
- `Li-Jesse-Jiaze`: `deep_learning_killer`, `double_q`, `dqn`, `sarsa_lambda`
- `itisacloud`: `GlasHoch_Rangers`, `SchmerzGebierge_Aua`
- `KunkelAlexander`: `dqn_allstar_duel` plus its required DQN/support modules (`q_deep_agent.py`, `q_helpers.py`, prioritized replay, settings, and environment event helpers)
- `antonH22`: `agent_a`
- `Alii-Khaled`: `arbiter`, `sentinel`, `overlord`; the pre-existing `harvy` symlink is the repository's current ship and is documented separately below
- `cs224`: `agent_011_shred`
- `3j14`: `strong_agent`
- `FFFROZEN090`: `Yu_policy_agent`
- `piscih`: `Bomb_Voyage` (ResNet-style DQN with spatial attention and `dqn_model.pt`)
- `nilskre`: `big_bertha_v1` (dueling DQN with committed TensorFlow SavedModel)

The removed first-pass directories were retained temporarily outside the workspace at `/tmp/bomberman_excluded_agents.yFBkXj` and are not part of the project import.

The pre-existing `harvy` symlink points to `/export/scratch/salitanl/BomberMan_alii/agent_code/Harvy`. The Alii-Khaled README identifies Harvy E130 as its current ship; it was not replaced because it is active local work.

The existing `harvy` and `ruehl_based_agent` entries were preserved; they are symlinks to pre-existing external work and were not overwritten.
