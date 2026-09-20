# Imported-agent Classic combat comparison (2026-09-20)

## Protocol and provenance

Driver: [`eval_suite/run_imported_combat.py`](../eval_suite/run_imported_combat.py).
Frozen imported checkpoints were tested on Classic boards 32000–32007,
agent seed 0, all four starting seats, and a 400-step limit. Each candidate
played 32 games against one `rule_based_agent` and 32 games against three
`rule_based_agent` opponents. No candidate played another candidate. The
opponents, boards, and seats were matched; game score includes collection and
kills, so kills and survival are reported separately. `harvy` is a local
external reference, not an imported entry.

Main artifacts: `/export/scratch/salitanl/bomberman_imported_combat_arch_20260920/`.
Corrected Sentinel artifacts:
`/export/scratch/salitanl/bomberman_imported_combat_sentinel_fixed_20260920/`.
Both jobs completed successfully. Each root includes `provenance.json`,
`smoke_passed.json`, the effective staged `manifest.json`, game-level JSONL,
per-block summaries, and `full/suite_summary.json`. CPU-only workers used
CUDA hidden and one math-library thread each.

The first Sentinel block is **invalid**: its imported callback did not unwrap
the `q_net` weights from `best.pt`; `strict=False` silently left its network
randomly initialized. A separate, otherwise identical staged rerun changed
only that loader to unwrap `q_net`; the imported source and checkpoint remain
untouched. All Sentinel numbers below use the corrected rerun. The other
Alii/Harvy checkpoint loads were checked against the saved state dictionaries.

## Results

Each matchup column is mean score / mean kills / survival / outright score
leads. A score lead excludes ties. Values are over 32 games; kills and score
are per game. Rank within each lineup, not by a single pooled average.

| Candidate | One rule-based opponent | Three rule-based opponents | Deployment architecture |
| --- | --- | --- | --- |
| `imp_li_deep_killer` | **7.12 / 0.25 / 88% / 28** | 5.22 / 0.31 / **81%** / 15 | Discrete-feature tabular Q-learning, JSON Q-table; despite its name, no deep net. |
| `imp_li_sarsa_lambda` | 6.56 / 0.16 / **94%** / 28 | 4.97 / 0.28 / 75% / **18** | Discrete-feature tabular SARSA(λ), JSON Q-table. |
| `imp_li_double_q` | 6.81 / 0.22 / 81% / 26 | 4.38 / 0.22 / 69% / 12 | Discrete-feature tabular Double Q-learning, two JSON Q-tables. |
| `imp_alii_arbiter` | 5.62 / **0.28** / 91% / 24 | **5.47 / 0.56** / 62% / 14 | 98-feature 3×256 policy/value MLP plus bounded search and explicit safety logic. |
| `harvy` (local reference) | 5.91 / 0.16 / 69% / 21 | 4.16 / 0.25 / 59% / 11 | 12-channel board CNN plus 98-scalar fusion, policy/value heads, time-expanded safety/search. |
| `imp_alii_sentinel` (corrected load) | 3.09 / **0.28** / 72% / 11 | 3.84 / 0.50 / 53% / 10 | 46 engineered inputs, 2×256 dueling Q MLP, safety helpers. |
| `imp_alii_overlord` | 3.06 / 0.25 / 78% / 12 | 3.75 / 0.41 / 41% / 9 | 12-channel residual CNN, 8-scalar fusion, dueling Q and danger heads, safety filter. |
| `imp_piscih_bomb_voyage` | 0.09 / 0.00 / 9% / 0 | 0.03 / 0.00 / 3% / 0 | 10-channel two-block residual CNN with spatial attention and DQN head. |

`imp_li_deep_killer` is the strongest general scorer in these matched games,
and its 81% survival against three opponents is also the best in that
lineup. `imp_alii_arbiter` is the strongest **combat-focused** candidate:
against three opponents it has the highest score (5.47) and most kills
(0.56/game). This is a hybrid policy, so the result cannot be credited to its
MLP alone. `imp_alii_sentinel` also gets 0.50 kills/game after the checkpoint
fix, but has lower score and 53% survival. Harvy's excellent prior solo
collection results do not carry over to this combat matchup.

This is a fixed-opponent comparison, not a tournament or training-algorithm
ablation. Eight board seeds and 32 games per matchup give a useful ranking
signal but do not establish a universal best agent. Scores from the solo
coin-heaven/loot-crate suite are not directly comparable.
