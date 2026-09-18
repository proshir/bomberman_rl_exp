# Loop-focused history-agent variants

Five one-factor variants were trained for 300 rounds with 400-step episodes,
three seeds each, and evaluated on held-out boards 22016--22031, all four
seats, at 100 and 400 steps. The baseline is the original history window of 8
positions. Each variant changes only one feature or reward component.

| Variant | 400-step coins | Completion | Repeated states |
|---|---:|---:|---:|
| Baseline history-8 | **44.06** | 25.5% | 225.1 |
| History-16 | 42.56 | 31.8% | 211.7 |
| History-32 | 41.64 | 28.1% | 225.0 |
| Steps-since-coin | 43.27 | **50.0%** | **162.2** |
| Remaining-time feature | 38.85 | 26.6% | 245.8 |
| Revisit penalty (0.05) | 41.32 | **49.5%** | 164.9 |

All variants had zero invalid actions and zero self-deaths. The stagnation
feature and revisit penalty are the clearest loop-breaking improvements: they
roughly halve the completion failures while reducing repeated states. They
slightly reduce mean coins, so the choice depends on whether the Stage 1
objective prioritizes finishing all coins or maximizing average collection.

The longer history windows alone were not sufficient; window 16 helped
completion modestly, while window 32 returned to baseline loop frequency. The
remaining-time feature hurt both collection and loop diagnostics in this
configuration.

Artifacts are in
`experiments/tree_fqi_history_loop_variant_heldout_confirmation/`.
