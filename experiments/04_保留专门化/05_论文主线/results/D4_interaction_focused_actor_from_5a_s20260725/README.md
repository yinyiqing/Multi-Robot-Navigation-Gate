# D4 Safety-Focused Actor Pilot From 5A

Status: promising candidate; repeat validation required.

## Protocol

- Original 24-dimensional TD3 Actor, actor-only warm-start from 5A.
- Fresh ego-motion local Critic with `75%` interaction sampling.
- Critic counterfactual ranking only at distance `<=1.0 m` and closing speed `>=0.1 m/s`.
- Actor unlock remains protected by the Critic gradient guard.
- Actor updates use only the same close-approaching subset, drawn from a 256-transition candidate batch with at least 16 eligible samples.
- Angular output is anchored to 5A with weight `2.0`; linear output remains free to learn deceleration.
- Epoch 1 is the frozen 5A baseline; epoch 2 updates the strong-interaction Actor.

## Result

| metric | epoch 1 frozen 5A | epoch 2 trained | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.820 | 0.840 | +0.020 |
| collision | 0.179 | 0.160 | -0.019 |
| unresolved | 0.001 | 0.000 | -0.001 |
| full success | 0.436 | 0.500 | +0.064 |
| deep full success | 0.150 | 0.217 | +0.067 |
| close full success | 0.500 | 0.600 | +0.100 |
| margin full success | 0.800 | 0.825 | +0.025 |
| average episode steps | 35.1 | 34.6 | -0.5 |

This is the first controlled strong-Actor pilot to improve full success and every interaction stratum within the same run. Epoch 2 is the saved best checkpoint.

## Behavior Audit

- Raw linear-action delta at distance `<=1.2 m`: `-0.049`.
- Raw linear-action delta at distance `<=0.8 m`: `-0.079`.
- Raw linear-action delta on close-approaching replay states: `-0.070`.
- Global angular-action delta: `-0.055`, smaller than the preceding pilot's magnitude `0.093` but not eliminated.
- Final dangerous-state linear positive-gradient share is roughly `0.10-0.20`.

The intended deceleration behavior was learned and the previous dangerous acceleration failure was removed. The remaining angular drift and Gazebo evaluation variance mean this is not yet a final accepted expert. Repeat the fixed 140-scenario validation and then run an independent training seed before D5 gate admission.

## Files

- `epoch001_frozen_5a_actor.pth`, `epoch001_critic.pth`: frozen baseline snapshot.
- `epoch002_actor.pth`, `epoch002_critic.pth`: promising trained candidate.
- `evaluations.npy`: both fixed-validation summaries.
- `train.log`: complete training and validation log.
- `tensorboard.tfevents`: training curves.

Large replay checkpoints remain local and are not included in Git.
