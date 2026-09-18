# Combat Double-DQN agent

Sahand was here.

## Motivation

The 46-feature local-topology tree ablation was trained for 600 rounds but
remained below the 32-feature tree control and continued to loop. The likely
issue is representation capacity: shallow per-action trees fragment a finite
replay sample when spatial features are added. This note records the controlled
algorithm transition rather than deleting the tree-FQI baseline.

## Implementation

`combat_dqn_agent` preserves the existing combat history/anti-stagnation
features, reward calculation, custom events, safety mask, and callback
interfaces. It replaces per-action tree refitting with:

- a PyTorch MLP with two 256-unit ReLU layers;
- separate policy and target networks;
- legal-action-restricted epsilon-greedy selection;
- a bounded 100,000-transition replay buffer;
- random mini-batches and Double-DQN targets;
- Huber loss, Adam, gradient clipping, and periodic target synchronization;
- terminal transitions with zero bootstrap value;
- checkpoints containing both network states, optimizer state, epsilon, and
  training counters.

Defaults are gamma 0.99, learning rate 1e-4, batch size 128, 5,000-transition
warmup, training every four environment steps, and target synchronization every
2,500 optimizer steps. The input dimension is inferred from the existing
feature extractor (32 at present), rather than hard-coded in the network.

The original FQI agents remain available under their existing directories and
the combat runner supports both checkpoint formats. DQN round records expose
replay size, optimizer steps, epsilon, and latest loss alongside the existing
reward, score, survival, coin, kill, and timing fields.

## Validation

The new DQN tests cover replay capacity/sampling, legal-action masking,
terminal Double-DQN targets, tensor shapes, checkpoint round trips, and a
successful optimizer update. Together with the existing framework, safety,
history, topology, and compact-DQN tests, **27 tests pass**.

A two-round, 20-step CPU smoke run completed fresh-process evaluation and saved
a PyTorch checkpoint containing the requested state. It did not reach the
5,000-transition warmup threshold, so its score is only a software check and
not performance evidence. A substantial combat-DQN pilot is the next measured
experiment.
