# D4 Dense Safety Reward Pilot From 5A

Status: rejected controlled pilot.

## Protocol

- Actor-only warm-start from frozen 5A; epoch 1 keeps the original 24-dimensional TD3 Actor frozen.
- Epoch 2 updates the Actor from interaction replay only.
- Critic input is the 24-dimensional Actor observation plus ego-frame relative position and relative velocity for up to nine visible active neighbors.
- Cooperative reward remains distance-weighted `0.8 self + 0.2 neighbor`.
- Relative to the preceding ego-motion context pilot, this run only strengthens the `1.2 m` speed-dependent proximity penalty and positive clearance reward.

## Result

| metric | epoch 1 frozen 5A | epoch 2 trained | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.827 | 0.827 | 0.000 |
| collision | 0.173 | 0.173 | 0.000 |
| full success | 0.457 | 0.429 | -0.029 |
| deep full success | 0.150 | 0.100 | -0.050 |
| close full success | 0.525 | 0.525 | 0.000 |
| margin full success | 0.850 | 0.825 | -0.025 |
| average episode steps | 30.9 | 37.1 | +6.2 |

The stronger safety reward did not produce a usable strong-interaction Actor. Do not add epochs to this configuration.

## Diagnosis

- Dangerous transitions at nearest visible-neighbor distance `<=0.8 m` are only about `2.6%` of the uniformly sampled replay.
- After epoch 1 Critic training, both linear and angular action gradients are positive on essentially `100%` of sampled dangerous states.
- The Actor therefore follows a random-initialization Critic bias: it increases dangerous-state linear action by about `+0.075` and develops a global angular offset.
- The previous safety reward also used neighbors outside the Critic's visible active-neighbor set.

The next controlled run aligns the reward neighbor set with the Critic, draws `75%` of every Critic batch from interaction replay, raises strong-Actor exploration only during Critic warmup, and refuses to unlock the Actor while dangerous-state Critic gradients remain nearly one-sided.

## Files

- `epoch001_frozen_5a_actor.pth`, `epoch001_critic.pth`: frozen baseline snapshot.
- `epoch002_actor.pth`, `epoch002_critic.pth`: rejected trained candidate.
- `evaluations.npy`: two validation summaries.
- `train.log`: complete training and validation log.
- `tensorboard.tfevents`: training curves.

Large replay checkpoints remain local and are not included in Git.
