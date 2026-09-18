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

Twenty focused history, safety, topology, and vanilla-DQN tests pass. They
cover replay capacity, vanilla terminal targets, tensor shapes, checkpoint
contents, and one optimizer update. A two-round, 20-step CPU smoke run also
completed fresh-process evaluation and checkpoint loading. It is software
validation only and did not reach replay warmup, so no performance conclusion
is drawn from it.

## Combat DQN pilot results

The fresh 300-round pilot used three seeds, 400 training steps per round, and
the same eight held-out boards and four seats for each checkpoint. Its final
round-300 mean was 18.48 coins (seed results 32.16, 10.03, and 13.25). The
matched history/anti-stagnation FQI control reached 19.35 coins on the same
96-game evaluation suite. Both policies had 100% survival and zero invalid
actions.

A fresh 600-round extension did not improve the result. Its aggregate curve
was 8.43, 9.96, 8.58, 9.20, 6.65, and 6.28 coins at rounds 100, 200, 300,
400, 500, and 600 respectively. The best checkpoint was round 200; the final
checkpoint degraded. All final evaluation games remained safe, with zero
invalid actions. The run completed in 11:34 and performed 58,751 optimizer
updates per seed.

The first runner seeded Python and NumPy but not PyTorch, so these fresh DQN
runs were not exactly reproducible despite matching nominal seeds. The combat
runner now calls `torch.manual_seed(seed)` (and CUDA seed initialization when
available) before constructing the model. A later frozen-checkpoint audit found
that the target calculation did not apply the action-time safety mask and that
replay could reconstruct a mismatched stagnation bucket. Therefore these pilot
scores are retained as pre-fix baselines; the corrected target and replay
comparison must be run before selecting a submission checkpoint. See
[`dqn_training_audit_20260918.md`](dqn_training_audit_20260918.md).
