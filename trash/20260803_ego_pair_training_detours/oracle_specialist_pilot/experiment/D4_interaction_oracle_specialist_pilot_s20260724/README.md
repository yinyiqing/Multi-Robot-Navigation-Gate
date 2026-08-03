# D4 Five-Agent Oracle Specialist Pilot

Status: rejected pilot.

## Protocol

- Frozen weak Actor: original 5D.
- Strong Actor and local-geometry Critic: complete 5D warm-start, original TD3 and `24 -> 800 -> 600 -> 2` Actor.
- Rollout oracle: strong Actor at nearest active robot distance `<=2.0 m`; frozen 5D otherwise.
- Critic target follows the same hybrid policy. Critic uses the full replay; Actor updates use only oracle-active transitions.
- Train: fixed five-agent `256 deep + 256 close + 128 margin` scenarios.
- Validation: fixed `60 deep + 40 close + 40 margin` scenarios.
- Epoch 1: Actor frozen through 20000 agent samples. Epoch 2: Actor unlocked after 21000 samples and evaluated near 40000 samples.
- Reward: distance-weighted `0.8 self + 0.2 neighbor`; forward, stagnation, wall-clearance and local-navigation additions disabled.

## Result

| metric | epoch 1 frozen baseline | epoch 2 oracle specialist | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.8271 | 0.8343 | +0.0071 |
| collision | 0.1729 | 0.1657 | -0.0071 |
| full success | 0.5000 | 0.4714 | -0.0286 |
| deep full success | 0.2333 | 0.2000 | -0.0333 |
| close full success | 0.5750 | 0.5750 | 0.0000 |
| margin full success | 0.8250 | 0.7750 | -0.0500 |

The strong Actor activation rate was `0.4347/0.4404` before/after training. The interaction replay contained `17595/40036 = 43.95%` of transitions.

Two additional paired evaluations were run with the same ordered 140-scenario manifest and independent seeds. Across all three evaluations, 5D versus epoch 2 achieved overall full success `192/420 = 0.4571` versus `187/420 = 0.4452`; agent success was `1730/2100 = 0.8238` versus `1728/2100 = 0.8229`. The first evaluation's four-episode decline was therefore partly simulation variance, but the target deep stratum declined in every run: `14 -> 12`, `8 -> 7`, and `13 -> 8`. Aggregate deep full success was `35/180 = 0.1944` for 5D and `27/180 = 0.1500` for epoch 2. See `repeat_evaluation/README.md`.

## Diagnosis

Replay audit shows that the candidate Actor learned an almost global positive angular shift rather than state-dependent yielding. Mean raw-action angular deltas were `+0.146` at `1.2--2.0 m`, `+0.154` at `0.8--1.2 m`, and `+0.137` at `<=0.8 m`. In the critical band it also increased raw linear action by `+0.090`. For critical states whose action changed by more than `0.05`, the trained Critic preferred the candidate action in `79.38%` of cases.

The oracle replay split fixed the earlier contamination by non-interaction Actor updates, but the current checkpoint did not learn a deep-interaction advantage. This epoch 2 checkpoint and training configuration fail the predeclared admission rule; this does not reject the two-Actor and Gate research direction. Do not add epochs to the unchanged configuration. Before another training run, address the one-direction turn and dangerous acceleration signals and then define a new controlled pilot.

## Files

- `epoch001_frozen_5d_actor.pth`, `epoch001_critic.pth`: frozen baseline snapshot.
- `epoch002_actor.pth`, `epoch002_critic.pth`: trained candidate snapshot.
- `evaluations.npy`: paired aggregate evaluations.
- `replay_interaction_audit.json`: replay distribution and Actor/Critic action audit.
- `train.log`: complete console log.
- `tensorboard.tfevents`: training diagnostics.
