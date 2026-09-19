# Imported-agent Stage-2 evaluation: results and reference notes

**Status:** measured snapshot, 2026-09-19.  This note is the comparison
baseline for later evaluation of locally trained agents.  It records only what
the frozen imported-agent suite actually ran; it is not a claim that every
import is portable or that every included checkpoint loaded correctly.

## Protocol and artifact provenance

The driver is [`eval_suite/run_imported_stage2.py`](../eval_suite/run_imported_stage2.py).
It stages imported callbacks unchanged (apart from the listed checkpoint-path
redirects), first performs a one-game smoke test, and then uses the fixed
suite protocol:

| Item | Value |
| --- | --- |
| Boards | seeds `30000`–`30007` |
| Agent seed | `0` |
| Seats | `0`–`3` |
| Games per agent/scenario | 32 |
| Scenarios | `coin-heaven`, `loot-crate` |
| Horizon | 400 steps |
| Main score | coins collected |
| Diagnostics | repeated states, no-progress streak, progress events, invalid actions |
| Completed-result root | `/export/scratch/salitanl/bomberman_imported_stage2_20260919_114422/full/` |
| Harvy matched-result root | `eval_suite/results/harvy_20260918_232105/` |
| Ruehl matched-result source | `/export/scratch/salitanl/bomberman_imported_stage2_20260919_114422/ruehl_full_v2/suite_summary.json` |

`coin-heaven` measures pure navigation/collection with 50 initially visible
coins.  `loot-crate` is the more complete bombing-and-collection test: coins
must first be exposed by destroying crates.  The evaluations are **solo**
(`--opponents` empty), so these numbers do not establish combat strength.

Results already written for the listed agents are complete (32 games per
scenario).  A separate successful 32-game `ruehl_based_agent` run is
recorded at `.../ruehl_full_v2/suite_summary.json`; it is a useful existing
local reference, not an imported-agent rank entry.

## Measured results

All values below are means over the 32 seat/board games.  `Complete` means all
50 coin-heaven coins were collected; it does not apply to loot-crate because
the generated number of accessible coins varies.  A dash means that the full
result was not available, not a zero.

| Agent / imported source | Coin heaven: coins; complete; steps | Loot-crate: coins; crates; survival | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `imp_li_deep_killer` (Li-Jesse-Jiaze) | **50.00; 100%; 144.78** | **40.50; 97.03; 100%** | Best end-to-end collector/explorer. |
| `imp_li_sarsa_lambda` (Li-Jesse-Jiaze) | 7.88; 3.1%; 391.84 | 36.13; 87.59; 96.9% | Excellent crate/coin behaviour, poor open-board navigation. |
| `imp_li_double_q` (Li-Jesse-Jiaze) | 16.69; 21.9%; 285.84 | 26.56; 67.47; 100% | Strong all-round tabular baseline, but below the two specialists. |
| `imp_alii_arbiter` (Alii-Khaled) | 7.38; 0%; 400.00 | 9.44; 25.22; 100% | Safest neural/search hybrid after the leader; some useful crate play. |
| `imp_alii_overlord` (Alii-Khaled) | **50.00; 100%; 133.75** | 4.72; 12.47; 100% | Fastest perfect open-board navigator; weak transfer to crates. |
| `imp_alii_sentinel` (Alii-Khaled) | **50.00; 100%; 135.53** | 3.88; 10.56; 100% | Near-fastest perfect navigator; same crate-transfer gap. |
| `imp_eric_kill` (ericgoldclub) | 19.91; 0%; 400.00 | 0.53; 6.56; 0% | Learns/encodes navigation better than its name suggests, but bombs itself in crate play. |
| `imp_3j14_strong` (3j14) | 8.34; 0%; 400.00 | 0.38; 6.16; 9.4% | Some coin movement; unstable bombing and early self-death. |
| `imp_eric_loot` (ericgoldclub) | 2.50; 0%; 400.00 | 0.00; 0.81; 90.6% | **Do not treat as a checkpoint score:** the callback expects `loot_crate_saved_model.pt`, while the staged artifact is named `loot_crate_agent_saved_model.pt`; it initialized a new DQN instead. |
| `imp_piscih_bomb_voyage` (piscih) | 1.50; 0%; 400.00 | 0.06; 0.16; 100% | Legal and safe but almost entirely stalled. |
| `harvy` (local reference) | **50.00; 100%; 194.09** | **42.97; 109.19; 100%** | Strongest measured local reference; best loot-crate result in this table. |
| `ruehl_based_agent` (local reference) | 1.75; 0%; 400.00 | 13.38; 44.63; 90.6% | Existing local DQN reference; useful control for later trained-agent comparisons. |

The two local rows use the same 30000–30007 seeds, four seats, 400-step
horizon, solo lineup, and coin metric as the imported results.  Harvy is a
full 32-game result in each scenario.  Ruehl is also a full 32-game result in
the `ruehl_full_v2` run; older Ruehl runs in `runs/` use different seed sets
and should not be mixed into this table.

### Scenario rankings

Rank separately rather than hiding the navigation-versus-crate trade-off in a
single arbitrary average.

The main ranking includes every measured policy, including the two local
references.  This makes the performance targets visible in one place; the
implementation discussion below still distinguishes imported and local code.

| Rank | Coin-heaven (coins; completion / mean completion step) | Loot-crate (coins; crates; survival) |
| --- | --- | --- |
| 1 | `imp_alii_overlord` (50.00; 100% / **133.75**) | `harvy` (**42.97; 109.19; 100%**) |
| 2 | `imp_alii_sentinel` (50.00; 100% / 135.53) | `imp_li_deep_killer` (40.50; 97.03; 100%) |
| 3 | `imp_li_deep_killer` (50.00; 100% / 144.78) | `imp_li_sarsa_lambda` (36.13; 87.59; 96.9%) |
| 4 | `harvy` (50.00; 100% / 194.09) | `imp_li_double_q` (26.56; 67.47; 100%) |
| 5 | `imp_eric_kill` (19.91; 0%) | `ruehl_based_agent` (13.38; 44.63; 90.6%) |
| 6 | `imp_li_double_q` (16.69; 21.9%) | `imp_alii_arbiter` (9.44; 25.22; 100%) |
| 7 | `imp_3j14_strong` (8.34; 0%) | `imp_alii_overlord` (4.72; 12.47; 100%) |
| 8 | `imp_li_sarsa_lambda` (7.88; 3.1%) | `imp_alii_sentinel` (3.88; 10.56; 100%) |
| 9 | `imp_alii_arbiter` (7.38; 0%) | `imp_eric_kill` (0.53; 6.56; 0%) |
| 10 | `ruehl_based_agent` (1.75; 0%) | `imp_3j14_strong` (0.38; 6.16; 9.4%) |
| 11 | `imp_eric_loot` (2.50; 0%; unverified load) | `imp_piscih_bomb_voyage` (0.06; 0.16; 100%) |
| 12 | `imp_piscih_bomb_voyage` (1.50; 0%) | `imp_eric_loot` (0.00; unverified load) |

The coin-heaven ordering breaks the 50-coin tie by completion speed.  The
loot-crate ordering is by mean collected coins, with crates and survival shown
to explain the result.  `ruehl_based_agent` is included as a local control,
not as an imported checkpoint.

Practical imported-agent ranking for a local-agent target is therefore:

1. `imp_li_deep_killer`: the only verified imported policy that is elite on both fixed scenarios.
2. `imp_li_sarsa_lambda` and `imp_li_double_q`: strongest verified crate specialists; Double-Q is the safer all-round comparison, SARSA is the better loot-crate target.
3. `imp_alii_overlord` and `imp_alii_sentinel`: best navigation-speed references, but not good general Stage-2 targets until crate transfer improves.
4. `imp_alii_arbiter`: useful safe hybrid reference, but substantially lower reward/collection than the table-based leaders here.

Including the local references, Harvy is the strongest measured policy overall:
it ties the 50-coin navigation ceiling and exceeds `imp_li_deep_killer` on the
loot-crate score (42.97 versus 40.50) while maintaining 100% survival.  It is
therefore the primary local target for future trained-agent comparisons;
`imp_li_deep_killer` remains the strongest imported implementation target.

## What each implementation contributes

| Family / agent | Implementation found in import | Feature or design lesson supported by this evaluation |
| --- | --- | --- |
| Li `deep_learning_killer` | Despite the name, deployment is a JSON **tabular Q-policy**, not a neural net.  It discretizes local directional tile/safety/target/bomb information, chooses an argmax action, and has heavily shaped combat/coin rewards. | Compact state abstraction plus explicit safety/target features can outperform much larger networks on these fixed boards.  It is the clearest benchmark to reproduce or ablate first. |
| Li `sarsa_lambda` | Same categorical local feature generator and JSON table; on-policy SARSA(λ) training. | Eligibility-trace/on-policy training produced strong crate opening and safe collection but did not generalize to sparse open-board pathing.  Test history/coverage features if borrowing this direction. |
| Li `double_q` | Two JSON Q-tables; inference sums them before legal action selection. | Double estimation gives the best compromise among the Li tabular variants, including nonzero completion on coin-heaven.  A good low-compute baseline. |
| Alii `overlord` | 12-channel 17×17 residual CNN with global/max pooling, an 8-scalar state head, dueling Q head, auxiliary danger predictor, and safety filter. | Rich spatial representation is enough for extremely fast open-board collection, but did not itself solve long-horizon crate planning.  Preserve explicit bomb-escape logic when trying CNNs. |
| Alii `sentinel` | 46 engineered features and a two-layer, 256-unit dueling MLP, with safety helpers. | Engineered local/topological features can match the CNN exactly on pure navigation.  Its failure mode is objective transfer, not basic legality/survival. |
| Alii `arbiter` | 98 engineered features; 3×256 shared policy/value MLP; safe-action mask, loop/bomb-repeat tie-breakers, test-time dihedral averaging, and bounded search/value-leaf evaluation. | Strong safety (no suicide/invalid actions) and the best Alii crate score.  Search/safety should be treated as a complementary action shield, not evidence that the learned policy alone is better. |
| Eric `kill_agent` | Hybrid DQN: 14 spatial board channels plus a 22-dimensional vector branch, target network, and valid-action mask. | Its 19.91 open-board coins show the representation contains useful navigation signal.  The 100% loot-crate suicide rate makes bomb timing/escape the critical missing feature. |
| 3j14 `strong_agent` | `sklearn.neural_network.MLPRegressor` fitted-Q style learner over a 7×7 reduced map plus 17 engineered features, including escape-path cues. | Local map + escape cues give modest coin collection, but only 9.4% crate survival: local features/targets need stronger long-horizon bomb planning and anti-loop control. |
| piscih `Bomb_Voyage` | 10-channel full-board `DQNResNet` and legal-action mask; reward strongly prioritizes kills. | A spatial deep model without compatible objective coverage can be safe yet inert in solo collection.  Do not adopt architecture without matching reward/curriculum. |
| Local `harvy` | Harvy/Arbiter-NG hybrid: 12-channel 17×17 spatial representation plus 98 engineered scalar features, policy/value network, explicit time-expanded safety solver, and bounded exact bomb/escape search. | Best measured crate result and perfect navigation completion.  The important combination is learned ranking plus a hard safety/search layer; architecture alone is not the explanation. |
| Local `ruehl_based_agent` | Compact dueling DQN with seven board maps plus scalar features, legal-action masking, and a saved `z_best-model.pt` checkpoint. | A useful conservative local neural control: it survives well and opens crates, but its open-board navigation is weak and it trails the imported tabular leaders. |

## Failure and comparability register

The following smoke-tested imports had no full score and must remain marked
**not evaluated**, rather than placed at the bottom of a performance ranking.

| Agent | Outcome | Reason observed |
| --- | --- | --- |
| `imp_yu_policy` | Not evaluated | Missing `wandb` dependency. |
| `imp_li_dqn` | Not evaluated | Checkpoint deserializes onto CUDA without a CPU `map_location`. |
| `imp_cciao_expert` | Not evaluated | Binary/NumPy import incompatibility (`numpy.core.multiarray failed import`). |
| `imp_ivo_coli` | Not evaluated | Full rerun indexed beyond the Q-table’s 1536 states (`IndexError`); smoke pass was insufficient. |
| `big_bertha_v1` | Not evaluated | TensorFlow-era import did not complete its smoke test. |

### How to compare future local agents

Run the same board seeds, seats, agent seed, scenarios, horizon, and
diagnostics before adding a row to this note.  Report both scenario columns,
not just their average, and identify checkpoint-loading or compatibility
changes explicitly.  The immediate targets are: beat **40.50 coins / 97.03
crates / 100% survival** on loot-crate (`imp_li_deep_killer`), and beat
**133.75 completion steps at 100% completion** on coin-heaven
(`imp_alii_overlord`); the stronger local Harvy anchor is **42.97 coins /
109.19 crates / 100% survival** on loot-crate.

Do not compare combat claims with this table: every full entry used empty
opponents.  Add a separate, fixed-opponent ranking when a combat suite is run.
