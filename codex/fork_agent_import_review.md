# Bomberman-RL fork and agent import review

Snapshot checked: 2026-09-19. Source: the 78 repositories returned by the GitHub forks API for [`ukoethe/bomberman_rl`](https://github.com/ukoethe/bomberman_rl/forks).

## Decision

Do not clone all forks into `imported_agents`. They are predominantly student submissions or untouched copies of the upstream template. Clone the following five first:

| Priority | Repository | Agent(s) worth importing | Why |
|---|---|---|---|
| High | [`cciao/bomberman_rl`](https://github.com/cciao/bomberman_rl) | `base_rl`, `expert_rl_w` | Multi-step off-policy Q-learning with handcrafted features, prioritized replay, and imitation-learning pretraining. This is the clearest reference for a compact engineered learner and a staged training pipeline. |
| High | [`ericgoldclub/bomberman_rl`](https://github.com/ericgoldclub/bomberman_rl) | `RUEHL_BASED_AGENT`, `Ultra_Network_agent`, `kill_agent`, `loot_crate_agent`, `mlp_agent`, `q_agent` | Several trained PyTorch agents and saved checkpoints, including different objectives (kill, loot, general play). Useful for checkpoint behavior comparisons and reward-design ideas. |
| High | [`ivo-1/bomberman_rl`](https://github.com/ivo-1/bomberman_rl) | `coli_agent`, `coli_agent_offline` | Tournament Q-learning agent plus a Decision Transformer implementation, training continuation support, q-tables, plots, and offline trajectories. Best source for comparing tabular online learning with offline sequence modeling. |
| High | [`Li-Jesse-Jiaze/MLE_project_bomberman`](https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman) | `dqn`, `dqn_5tile`, `double_q`, `deep_learning_killer`, `feature_is_everything`, `sarsa`, `sarsa_lambda` | Broad controlled algorithm collection with saved tables/checkpoints, symmetry handling, and a reported third-place result. Good for ablations, though the code should be imported selectively. |
| Medium | [`itisacloud/bomberman_rl`](https://github.com/itisacloud/bomberman_rl) | `GlasHoch_Rangers`, `SchmerzGebierge_Aua` | PPO-style and feature-heavy agents with A* / expert utilities, multiple `.pth` checkpoints, configuration files, and imitation/no-imitation variants. Worth cloning for model and feature inspection. |

These are optional second-wave imports:

| Repository | Agent(s) | Assessment |
|---|---|---|
| [`Abhinand-p/Reinforcement-Learning-Bomberman`](https://github.com/Abhinand-p/Reinforcement-Learning-Bomberman) | `classic_1`, `novoice_agent` | **Clone if compact Q-learning is needed.** Handcrafted wall, bomb, crate, blockage, and shortest-path features; custom events; Q-tables and saved models. Small and easy to mine, but claims are not matched evaluations. |
| [`nilskre/bomberman_rl`](https://github.com/nilskre/bomberman_rl) | `big_bertha_expert`, `big_bertha_v1` | **Clone if imitation + DQN is needed.** Includes prioritized replay, TensorFlow models, TensorBoard logs, and an expert-to-DQN training sequence. Old TensorFlow/CUDA setup is a substantial compatibility cost. |
| [`bGuenes/bomberman_MLE_homework`](https://github.com/bGuenes/bomberman_MLE_homework) | `agent_berkay`, `agent_coin`, `agent_coin_II`, `agent_the_destroyer_of_universes` | **Optional.** Many models and reward/batch-normalization variants, but the repository is very large and artifact-heavy. Import individual agents, not the whole repository. |
| [`karthikaf/Bomberman-RL`](https://github.com/karthikaf/Bomberman-RL) | `hitman`, `Bond` | **Optional.** Has a trained model and five `hitman` checkpoints; useful as a small DQN checkpoint source. Documentation and reproducibility appear limited. |
| [`Fjallripa/bomberman`](https://github.com/Fjallripa/bomberman) | `agent_h1`, `agent_h2`, `agent_h2b`, `agent_m1`, `agent_sq10`, `agent_swq12`, `BombCoin-Miner` | **Optional for historical experiments.** Many variants, logs, Q-data, and checkpoints, but naming and layout are inconsistent and the repository is noisy. |
| [`KunkelAlexander/bomberman_rl`](https://github.com/KunkelAlexander/bomberman_rl) | `dqn_allstar_duel` / `q_deep_agent.py` | **Imported.** This is a CNN-based DQN with configurable Double DQN, prioritized replay, dueling-style output, legal-action masking, and saved online/target models. It is the strongest direct fork for the DDQN line. The inspected code does not show a ResNet backbone. |
| [`Anze-/bomberman_rl`](https://github.com/Anze-/bomberman_rl) | `genetic_agent`, `coin_hunter_agent`, `survival_agent`, `wall_breaker` | **Optional, not a primary RL import.** Useful evolutionary-policy and rule-based baselines, including serialized genetic-agent networks; no central learned DQN/PPO pipeline. |
| [`antonH22/cnn-based-dql`](https://github.com/antonH22/cnn-based-dql) | `agent_a` | **Optional.** Focused CNN deep Q-learning implementation; useful as a minimal CNN reference, but no checkpoint was visible in the tree. |
| [`xn-peng/bomberman_rl`](https://github.com/xn-peng/bomberman_rl) | `my_agent`, `second_agent` | **Optional.** Multiple saved models and custom feature utilities; worth inspecting if model snapshots or feature engineering are needed. |
| [`rahul2227/bomberman_rl_RBN`](https://github.com/rahul2227/bomberman_rl_RBN) | `Q_learner`, `Servus` | **Optional only.** Contains a Q-learning agent and a Keras model, but little evidence of a reusable or documented training setup. |
| [`Michael-ale000/SMURFER`](https://github.com/Michael-ale000/SMURFER) | `smurfer` | **Optional only.** A small standalone agent with training code; useful for a quick behavioral comparison, not worth a full import by itself. |

## Direct inspection of the remaining forks

The following forks were checked at the repository-tree level. They are **not worth cloning into `imported_agents`** for the current project: they contain only the upstream template agents, a minimal user agent, or no distinct reusable learned agent was visible.

`narame7/bomberman_rl`, `cs224/bomberman_rl`, `MansurDaschaew/bomberman_rl`, `GwydionJon/bomberman_rl`, `FreWill9/bomberman_rl`, `maxiherzog/bomberman_rl`, `yingchanchu/bomberman_rl`, `OoJJBoO/bomberman_rl`, `YannEbling/bomberman_rl`, `neelkanthrawat/Bomberman-RL-Koethe`, `FFFROZEN090/bomberman_rl`, `Lennox-Elaphurus/bomberman_rl`, `Janik7777/bomberman_rl`, `chunjuanjuan0451/bomberman_rl`, `lukaswenzl/bomberman_rl`, `Adnan-Asif/Bomberman-Agents`, `jonaskleine/bomberman_rl`, `Enig7Ma/bomberman_rl`, `Alii-Khaled/BomberMan`, `ylchin/bomberman_rl`, `binyamolango/final_project_bomberman_rl`, `ukdewan123-ui/bomberman_rl`, `smitha13798/bomberman_rl`, `druzsan/bomberman_rl`, `Myst-pix/bomberman_rl`, `cmxin24/bomberman_rl`, `aanhlongg/bomberman_rl`, `Ravanoid3/bomberman_rl`, `piscih/bomberman_rl`, `baumange/bomberman_rl`, `leonbegiristain/RL_Bomberman`, `eliaserland/bomberman_rl`, `irisakohler/bomberman_rl`, `hericks/bomberman_rl`, `Viveal/bomberman_rl`, `dibstan/bomberman_rl`, `thekaharis/bomberman_rl`, `PrimeF/bomberman_rl`, `theGindar/bomberman_rl`, `pkollenz/bomberman_rl`, `andquintero/bomberman_rl`, `Philipp-g/bomberman_rl`, `mdawas/bomberman_rl`, `aruna-ram/bomberman_rl`, `yiwen-lu/bomberman_rl`, `ywzcode/bomberman_rl`, `jonasw247/bomberman_rl`, `Akatuoro/bomberman_rl`, `georggrab/bomberman_rl`, `Sthuthi98/bomberman_rl`, `ChristopherRotter/bomberman_rl`, `sashakowa/bomberman_rl`, `AntonEberhardt/bomberman_rl`, `SeZven/bomberman_rl`, `juso40/bomberman_rl`, `simaesm/bomberman_rl`, `cemdaloglu/bomberman_rl`, `nilsfriess/bomberman_rl`, `FatManWalking/bomberman_rl`, `PrzemyslawWozniakowski/bomberman_rl`, `3j14/bomberman_rl`, `momodaidh/bomberman_rl`.

## Import policy

Do not merge foreign agents directly into `agent_code`. Preserve provenance and prevent accidental behavior changes by cloning selected repositories under a separate, clearly named area such as:

```text
imported_agents/
  cciao_bomberman_rl/
  ericgoldclub_bomberman_rl/
  ivo1_bomberman_rl/
  li_jesse_jiaze_mle_project_bomberman/
  itisacloud_bomberman_rl/
```

For each imported agent, record the upstream URL, commit SHA, original agent directory, framework/dependency versions, checkpoint files, action-mask behavior, feature representation, and reward/event definitions. Copy only the agent source and the minimum required model/config files; do not copy `.git`, replay logs, TensorBoard runs, avatars, or large duplicate artifacts unless they are needed for a reproducibility experiment.

The recommended order is `cciao` → `ericgoldclub` → `ivo-1` → `Li-Jesse-Jiaze` → `itisacloud`. Run each through the existing evaluation suite before drawing conclusions: a saved checkpoint or README ranking is evidence that an agent exists, not evidence that it beats the current local baseline under matched seeds, opponents, and episode limits.
