# Imported-agent competition audit

Audit date: 2026-09-19. The direct-fork list contained 78 repositories. I checked repository trees, agent documentation, checkpoint presence, model/training code, and reported evaluation evidence. The import set now contains the strongest candidates found, but no fork result is directly comparable to our local evaluation until all agents are run under the same seeds and opponents.

## Strongest candidates

| Tier | Agent | Evidence | Competition use |
|---|---|---|---|
| A | `harvy` (Alii-Khaled) | Existing local symlink to the current Alii-Khaled ship. Its README reports a CNN + scalar fused policy, behavior-cloning warm start, RL fine-tuning, lookahead, and a reported 0.681 round-win rate in its own battery. | First snapshot contender; preserve as the current external best candidate. |
| A | `Alii-Khaled_BomberMan__arbiter` | Committed PyTorch checkpoint and a documented P0 inference agent. | Immediate checkpoint evaluation. |
| A | `Alii-Khaled_BomberMan__sentinel` | Committed `best.pt` and `best_stage1_nav.pt`; documented as an MLP Dueling-DQN curriculum agent. | Immediate DDQN-style checkpoint evaluation. |
| A | `Alii-Khaled_BomberMan__overlord` | Committed CNN checkpoint; README calls it the backup CNN agent. | Immediate CNN checkpoint evaluation. |
| A | `cs224_bomberman_rl__agent_011_shred` | Committed VGG-style CNN model, detailed state representation, augmentation, and a report that the deeper/batch-normalized variant improved behavior before later NaN instability. | Immediate CNN checkpoint evaluation; treat training reproducibility as weak. |
| A | `3j14_bomberman_rl__strong_agent` | Explicitly identified as the final agent, with `multimulti_best_yet.pt` and performance/score traces. | Immediate trained MLP/ensemble evaluation. |
| A | `FFFROZEN090_bomberman_rl__Yu_policy_agent` | Committed `best_model` checkpoint and documented coin-heaven, loot-crate, and classic scenarios. | Immediate policy-checkpoint evaluation. |
| B | `Li-Jesse-Jiaze_MLE_project_bomberman__dqn` | Saved PyTorch DQN checkpoint with a compact, easier-to-adapt implementation. | CNN/DQN baseline. |
| B | `KunkelAlexander_bomberman_rl__dqn_allstar_duel` | CNN DQN source with Double DQN, prioritized replay, dueling output, and legal-action masking. No committed trained checkpoint was found in the fork tree. | Train/evaluate as the main DDQN research candidate, not as a ready-made winner. |
| B | `antonH22_cnn-based-dql__agent_a` | Small PyTorch CNN-DQN implementation. No committed checkpoint was found. | Train/evaluate as a clean CNN baseline. |
| A | `piscih_bomberman_rl__Bomb_Voyage` | PyTorch `DQNResNet` with residual blocks, spatial attention, 10-channel spatial input, and a committed `dqn_model.pt` checkpoint. | Immediate ResNet checkpoint contender. |
| B | `nilskre_bomberman_rl__big_bertha_v1` | TensorFlow dueling DQN with committed SavedModel, prioritized replay support, and an expert/imitation-learning lineage. The environment is old TensorFlow/CUDA. | Immediate dueling-DQN checkpoint contender, after dependency isolation. |

The existing imported `ivo-1/coli_agent`, `cciao/expert_rl_w`, `ericgoldclub/kill_agent`, `ericgoldclub/loot_crate_agent`, `itisacloud/GlasHoch_Rangers`, and `itisacloud/SchmerzGebierge_Aua` remain useful comparison agents, but their repositories provide weaker matched-performance evidence than the Tier A set above.

## Heuristic ranking before local competition

This is a **guess**, not an observed leaderboard. It ranks expected competitive value for immediate evaluation, weighting a usable checkpoint and explicit performance evidence more heavily than the algorithm name. Positions may change substantially once action interfaces, feature preprocessing, and checkpoints are tested in the local environment.

1. `harvy` — Alii-Khaled current ship; strongest explicit self-reported result.
2. `Alii-Khaled_BomberMan__arbiter` — committed checkpoint and same project's tested inference stack.
3. `piscih_bomberman_rl__Bomb_Voyage` — pretrained ResNet-style DQN with spatial attention.
4. `cs224_bomberman_rl__agent_011_shred` — pretrained deeper VGG-style CNN and reported improvement over its predecessor.
5. `3j14_bomberman_rl__strong_agent` — final MLP ensemble with best checkpoint and performance traces.
6. `Alii-Khaled_BomberMan__sentinel` — committed dueling-DQN curriculum checkpoint.
7. `Alii-Khaled_BomberMan__overlord` — committed CNN backup checkpoint.
8. `FFFROZEN090_bomberman_rl__Yu_policy_agent` — committed policy checkpoint with multiple documented scenarios.
9. `Li-Jesse-Jiaze_MLE_project_bomberman__dqn` — compact saved PyTorch DQN.
10. `nilskre_bomberman_rl__big_bertha_v1` — dueling DQN and imitation lineage, but older TensorFlow dependencies.
11. `cciao_bomberman_rl__expert_rl_w` — prioritized multi-step Q-learning with imitation pretraining.
12. `ericgoldclub_bomberman_rl__kill_agent` / `loot_crate_agent` — trained objective-specific checkpoints.
13. `ivo-1_bomberman_rl__coli_agent` — tournament Q-learning agent with q-tables and plots.
14. `itisacloud_bomberman_rl__GlasHoch_Rangers` / `SchmerzGebierge_Aua` — trained PPO-style candidates, but higher integration cost.

Research-only until trained locally: `KunkelAlexander_bomberman_rl__dqn_allstar_duel` and `antonH22_cnn-based-dql__agent_a`. Their implementations are interesting, but no committed trained checkpoint was found, so they should not outrank the ready-to-run checkpoint agents yet.

## What the scan did not establish

- CNN, DDQN, dueling, or prioritized replay is not proof of a better policy. Representation quality, action masking, reward shaping, checkpoint selection, and opponent distribution can dominate the algorithm label.
- The Alii-Khaled README reports the strongest explicit recent result, but it is self-reported under that repository's own battery. It must be treated as a hypothesis until reproduced locally.
- `KunkelAlexander` is the strongest direct DDQN implementation found, but it is a training candidate rather than a verified pretrained competitor because no saved model appeared in its fork tree.
- The scan found no direct `ukoethe/bomberman_rl` fork with a clearly identified ResNet backbone. `cs224/agent_011_shred` uses a VGG-style CNN; Alii-Khaled uses CNN-based agents; neither is ResNet.
- The non-fork project [`TatjanaChernenko/reinforcement_learning_agent_Bomberman_game`](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game) is worth keeping as external research context because it combines CNN, dueling values, prioritized replay, and symmetry handling, but it should not be mixed into the direct-fork import set without a separate compatibility review.

## Competition protocol

Use two tracks:

1. **Checkpoint track:** `harvy`, Alii `arbiter`/`sentinel`/`overlord`, `cs224 agent_011_shred`, `3j14 strong_agent`, `FFFROZEN Yu_policy_agent`, and Li-Jesse-Jiaze `dqn`.
2. **Training track:** Kunkel `dqn_allstar_duel` and AntonH22 `agent_a`, trained or fine-tuned under the same budget as the local DQN/DDQN agents.

For every track, normalize the interface and record source commit, checkpoint filename, framework version, action masking, feature shape, opponent set, seeds, rounds, mean score, win rate, survival rate, kills, self-kills, coins, and invalid actions. Do not rank an agent from its README result alone; rank the matched evaluation outputs.
