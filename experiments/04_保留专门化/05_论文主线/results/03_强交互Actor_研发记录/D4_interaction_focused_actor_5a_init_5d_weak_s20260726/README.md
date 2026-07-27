# D4 5A-Initialized Strong Actor With Frozen 5D Weak Actor

Status: rejected after epoch 2; do not continue the trained checkpoint.

## Purpose

This experiment decouples the two Actor roles in the oracle rollout:

- the trainable strong Actor is initialized from 5A;
- the frozen weak Actor outside the `2.0 m` interaction region is 5D;
- TD3, reward, fixed manifests, Critic safety ranking and focused Actor updates match the preceding successful all-5A pilot;
- seed is `20260724`.

The first process was intentionally stopped before validation while the experiment order was discussed. The second process resumed its Replay, Critic, counters and optimizer state and completed both epochs normally.

## Result

| metric | epoch 1 frozen hybrid | epoch 2 trained | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.839 | 0.811 | -0.028 |
| collision | 0.160 | 0.187 | +0.027 |
| unresolved | 0.001 | 0.001 | 0.000 |
| full success | 0.514 | 0.407 | -0.107 |
| deep full success | 0.217 | 0.100 | -0.117 |
| close full success | 0.650 | 0.475 | -0.175 |
| margin full success | 0.825 | 0.800 | -0.025 |
| average episode steps | 36.8 | 36.2 | -0.6 |

Actor training loses 15 full-success scenarios out of 140, including seven deep and seven close scenarios. Epoch 1 remains the saved best checkpoint. The gradient guard accepted most updates, so it did not prevent this behavior-level regression.

## Decision

Do not extend epoch 2. Resume the previously successful all-5A training trajectory without changing its rollout distribution. After selecting its best strong Actor, pair it with frozen 5D in a separate oracle evaluation before training any deployable gate.

The unexpectedly strong frozen hybrid result (`0.514` full success) is not treated as established improvement until repeated fixed-manifest evaluation confirms it.

## Files

- `epoch001_frozen_hybrid_actor.pth`, `epoch001_critic.pth`: frozen hybrid baseline.
- `epoch002_actor.pth`, `epoch002_critic.pth`: rejected trained checkpoint.
- `evaluations.npy`: both fixed-validation summaries.
- `train_part1_interrupted.log`, `train_part2_completed.log`: complete process history.
- `tensorboard_part1.tfevents`, `tensorboard_part2.tfevents`: training curves across the resume boundary.

Large Replay checkpoints remain local and are not included in Git.
