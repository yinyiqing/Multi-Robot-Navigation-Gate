# D4 Critic Gradient Guard Pilot From 5A

Status: rejected diagnostic pilot.

## Protocol

- Actor-only warm-start from 5A; Critic starts fresh.
- Critic batches contain `75%` interaction transitions.
- Strong-Actor exploration noise is `0.30` during Critic warmup.
- Actor unlock requires at least 32 dangerous samples, linear positive-gradient share `<=0.90`, and angular one-sided share `<=0.90`.
- Reward uses the same visible active-neighbor set as the local Critic.

## Result

- The gradient gate was evaluated 226 times and never passed; Actor parameters remained exactly equal to 5A.
- Angular one-sided gradient share improved from about `0.96` to `0.50-0.63`.
- Linear positive-gradient share remained about `0.96-1.00` through the end of training.
- Epoch 1/2 full success was `0.450/0.421`, but the Actor parameters are identical; this difference is evaluation nondeterminism, not Actor regression.
- Epoch 1 deep/close/margin full success was `0.200/0.500/0.775`.
- Epoch 2 deep/close/margin full success was `0.117/0.450/0.850`.

## Decision

Interaction oversampling fixes the arbitrary left/right Critic bias but does not constrain counterfactual linear-action extrapolation. Do not unlock the Actor or add epochs to this configuration. The next controlled pilot adds a local Critic action-ranking loss only for close, approaching interactions while retaining the gradient guard.

Large replay checkpoints remain local and are not included in Git.
