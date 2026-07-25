# D4 Safety-Focused Actor Pilot From 5D

Status: necessary comparison complete; do not continue this checkpoint.

## Protocol

- Same seed `20260724`, fixed train/validation manifests, reward, Critic and focused Actor update rules as the successful 5A pilot.
- Original 24-dimensional TD3 Actor, actor-only warm-start from 5D.
- Fresh ego-motion local Critic with `75%` interaction sampling and counterfactual safety ranking.
- Actor updates use only close-approaching samples; angular output is anchored to the warm-start Actor.
- The launcher used 5D both as the strong Actor initialization and as the frozen weak Actor outside the `2.0 m` oracle region.
- Epoch 1 is the frozen 5D baseline; epoch 2 updates the strong-interaction Actor.

## Result

| metric | epoch 1 frozen 5D | epoch 2 trained | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.824 | 0.830 | +0.006 |
| collision | 0.173 | 0.169 | -0.004 |
| unresolved | 0.003 | 0.001 | -0.002 |
| full success | 0.471 | 0.479 | +0.008 |
| deep full success | 0.150 | 0.183 | +0.033 |
| close full success | 0.550 | 0.600 | +0.050 |
| margin full success | 0.875 | 0.800 | -0.075 |
| average episode steps | 36.0 | 35.1 | -0.9 |

Epoch 2 improves full success by only one scenario out of 140 and loses three margin successes. Under this protocol, the all-5D configuration provides substantially less learning signal than the all-5A pilot, which improved full success by nine scenarios out of 140.

## Interpretation

This comparison is necessary but not a pure initialization ablation: changing `BASE_MODEL` also changed the frozen oracle weak Actor. It supports rejecting continued training of this all-5D checkpoint. The next configuration decouples the roles: initialize the strong Actor from 5A while keeping frozen 5D outside the interaction region, matching the intended two-expert system.

The shutdown monitor stopped ROS immediately after all epoch 2 artifacts were saved. The final ROS interruption traceback is expected and does not invalidate the completed evaluation or checkpoints.

## Files

- `epoch001_frozen_5d_actor.pth`, `epoch001_critic.pth`: frozen baseline snapshot.
- `epoch002_actor.pth`, `epoch002_critic.pth`: trained comparison checkpoint.
- `evaluations.npy`: both fixed-validation summaries.
- `train.log`: complete training, validation and controlled-shutdown log.
- `tensorboard.tfevents`: training curves.

Large replay checkpoints remain local and are not included in Git.
