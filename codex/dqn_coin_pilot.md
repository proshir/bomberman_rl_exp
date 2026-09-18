# Compact DQN coin-navigation pilot

The compact DQN was trained for 300 rounds with 400 environment steps per
training round, across seeds 0, 1, and 2. Frozen evaluations used 100 steps,
board seeds 10000--10007, and all four seats. The run used CUDA-enabled
PyTorch, but the environment and feature extraction remained CPU-bound; the
launcher completed successfully in 1:02:21 with exit code 0.

| Training checkpoint | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| 0 | 1.69 | 0.31 | 0.22 | 0.74 |
| 100 | 25.69 | 17.97 | 20.84 | 21.50 |
| 200 | 32.66 | 29.06 | 27.41 | **29.71** |
| 300 | 28.66 | 29.81 | 29.44 | 29.30 |

All frozen evaluations had zero invalid actions, 100% survival, and zero full
coin-completion games. The final checkpoint therefore did not improve over the
round-200 mean. This pilot is not a matched fresh-board comparison with the
history agent, but it is weaker than the history agent's 39.80 mean on the
100-step held-out evaluation under the current evidence.

Artifacts are in `experiments/dqn_coin_pilot/`.
