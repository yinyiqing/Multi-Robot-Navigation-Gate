# D4 Ego-Motion Critic Context Pilot From 5A

Status: rejected controlled pilot.

## Protocol

- Actor-only warm-start from frozen 5A; epoch 1 keeps the Actor frozen.
- Epoch 2 updates the original 24-dimensional TD3 Actor from interaction replay only.
- The new Critic uses an 87-dimensional state: 24-dimensional Actor observation plus ego-frame relative position and relative velocity for up to nine neighbors.
- Train and validation manifests, seed, reward, optimizer settings and sample budget match the paired legacy-context 5A run.
- Reward remains distance-weighted `0.8 self + 0.2 neighbor`, with `robot_safe_distance=1.0` and the legacy action-independent proximity penalty.

## Result

| metric | epoch 1 frozen 5A | epoch 2 trained | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.844 | 0.823 | -0.021 |
| collision | 0.154 | 0.177 | +0.023 |
| full success | 0.500 | 0.421 | -0.079 |
| deep full success | 0.200 | 0.133 | -0.067 |
| close full success | 0.575 | 0.400 | -0.175 |
| margin full success | 0.875 | 0.875 | 0.000 |

The new context did not prevent post-unlock regression. Its epoch 2 overall full-success rate is the same `0.421` as the paired legacy-context 5A run, while deep and close are lower.

## Diagnosis

Replay contains `40047` transitions, but only `1139` (`2.84%`) are at nearest-neighbor distance `<=0.8 m`. In that critical band, the trained Actor increases raw linear action by `+0.128` instead of slowing down. For critical states whose action changes by more than `0.05`, the trained Critic prefers the candidate action in `76.77%` of cases.

The coordinate-frame and relative-motion context correction was necessary for a coherent Critic input, but it is not sufficient. The remaining failure is consistent with sparse, delayed collision supervision and a weak action-independent proximity penalty. Do not add epochs to this configuration. The next controlled pilot keeps the same 5A warm-start and ego-motion Critic, and changes only the proximity reward to provide dense speed-dependent supervision from `1.2 m` inward.

## Files

- `epoch001_frozen_5a_actor.pth`, `epoch001_critic.pth`: frozen baseline snapshot.
- `epoch002_actor.pth`, `epoch002_critic.pth`: rejected trained candidate.
- `replay_interaction_audit.json`: distance-stratified replay and Actor/Critic audit.
- `train.log`: complete training and validation log.
