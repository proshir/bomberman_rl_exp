# Combat vanilla-DQN agent

Sahand was here.

## Motivation

The local-topology tree-FQI variant was extended to 600 rounds but remained
below the 32-feature control and continued to loop. To keep the algorithm
progression controlled, the next comparison is a simple neural DQN using the
original validated feature vector. The FQI agent remains available as the
experimental baseline.

## Minimal implementation

`combat_dqn_agent` preserves:

- the existing 32-feature history/anti-stagnation extractor;
- the existing combat reward and custom event handling;
- the existing six actions and safety-constrained candidate actions;
- the existing training callbacks and fresh-process evaluation runner.

It adds only a 128-128 ReLU MLP, a bounded replay buffer, policy and target
networks, epsilon-greedy exploration, random mini-batch updates, periodic
target-network copying, and model save/load. The target is deliberately
vanilla DQN:

```python
with torch.no_grad():
    next_q_values = target_net(next_states)
    next_values = next_q_values.max(dim=1).values
    targets = rewards + gamma * (1.0 - dones) * next_values
```

No Double-DQN, dueling network, prioritized replay, n-step returns, new
features, or new reward shaping is included.

## Validation

Nineteen focused history, safety, topology, and vanilla-DQN tests pass. They
cover replay capacity, vanilla terminal targets, tensor shapes, checkpoint
contents, and one optimizer update. A two-round, 20-step CPU smoke run also
completed fresh-process evaluation and checkpoint loading. It is software
validation only and did not reach replay warmup, so no performance conclusion
is drawn from it.
