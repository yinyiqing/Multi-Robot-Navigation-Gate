# D4 Formal Balanced Strong-Interaction Actor Training

Status: candidate; repeat fixed validation and standalone paired evaluation pending.

## Protocol

- Run-start commit: `7307602`.
- Seed: `20260724`.
- Original 24-dimensional TD3 Actor; Actor initialized from 5A and Critic initialized from scratch.
- Training-time oracle rollout uses the trainable Actor at neighbor distance `<=2.0 m` and frozen 5A elsewhere. The validation values below therefore measure this oracle combination, not the strong Actor running alone and not a learned Gate.
- Cooperative reward remains `0.8 self + 0.2 neighbors` with distance weighting.
- Critic uses ego-frame neighbor motion context, `75%` interaction replay, and the previously selected counterfactual safety ranking weight `5.0`.
- Actor updates use only close-approaching states (`distance <=1.0 m`, closing speed `>=0.1 m/s`); angular output is anchored to 5A with weight `2.0`.
- Epoch 1 is the frozen 5A baseline. The Actor unlock delay is `21,000` agent samples.
- Train: `2560` fixed scenarios from `full_train.json.gz`, sampled with `balanced_cycle`.
- Validation: the same fixed `60 deep + 40 close + 40 margin` scenarios after every `20,000` agent samples.
- Budget: `16` epochs and `320,000` agent samples. All `2560/2560` training scenarios were visited; test was not read.

## Result

| metric | epoch 1 frozen 5A | epoch 16 best | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.826 | 0.917 | +0.091 |
| collision | 0.174 | 0.079 | -0.095 |
| full success | 0.436 (61/140) | 0.707 (99/140) | +0.271 |
| deep full success | 0.183 (11/60) | 0.617 (37/60) | +0.434 |
| close full success | 0.475 (19/40) | 0.675 (27/40) | +0.200 |
| margin full success | 0.775 (31/40) | 0.875 (35/40) | +0.100 |
| average episode steps | 33.1 | 54.5 | +21.4 |

Epoch 16 is the validation-selected best checkpoint. The large success and collision improvements show that this training configuration produced a useful candidate, including on the target deep stratum. The longer completion time shows that the learned behavior is also substantially slower or more conservative.

This is one validation realization used repeatedly for checkpoint selection. It is not yet an independent paper result. Before test or Gate training, repeat the fixed validation and then compare standalone 5D and standalone epoch-16 Actor on identical scenario IDs. Only that paired evaluation can establish expert complementarity and decide whether a Gate is justified.

## Integrity

| artifact | SHA-256 |
| --- | --- |
| `epoch001_frozen_5a_actor.pth` | `72dd1fb05a095b4f7ed85d52f7995ebe6e7013d9c9705bcff8b4e1a304e3bb0e` |
| `epoch001_critic.pth` | `18d6da9a2bd5531bb6097643b52421685c950fc1e048441efa50bc42fde648c0` |
| `epoch016_best_actor.pth` | `d0f349507bf6d427d30af15020373ab3cee33b4bfc19a57cd21230f237f95673` |
| `epoch016_best_critic.pth` | `6c6f5807fb183bee0fc16e7ec8d07f6db1be0ee8c66fe576a6a0aaf21a2f85d9` |
| train manifest | `d5b9b1fb968c8752e54e66f1ea3f25e7c2bf45eae3f012a686008704964da142` |
| validation manifest | `3b2646a842b777f8c60dca4c452cb78eb3a223ffe59139b8501797aa1d23d583` |

## Files

- `train.log`: complete training and validation log.
- `evaluations.npy`: epoch-level validation summaries.
- `tensorboard.tfevents`: training curves.
- `epoch001_*`: frozen baseline snapshots.
- `epoch016_best_*`: validation-selected candidate snapshots.

Large replay/recovery checkpoints remain local under `TD3/checkpoints/` and are not publication artifacts.
