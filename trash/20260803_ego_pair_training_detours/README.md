# Ego/Pair Training Detours Archive

Archived on 2026-08-03 to keep normal repository searches focused on the formal training line:

> 5A warm start + corrected edge-1 scenarios + one complete shared Actor with conventional TD3 training.

## Archived items

### Controlled-ego Critic-only v2

- `scripts/start_training_full_actor_edge1_simple_critic_v2.sh`
- `scripts/stop_training_full_actor_edge1_simple_critic_v2.sh`
- `scripts/audit_local_td3_critic_checkpoint.py`
- Four dedicated Critic replay/calibration JSON files from `experiments/03_保留专门化/02_论文主线/10_纯单冲突完整Actor_pilot/local_data/`

Reason: this branch froze the Actor and trained a Critic from controlled-ego replay. It was useful for diagnosis, but it is not the formal full shared-Actor training protocol.

### Pair-interaction pilot

- `scripts/start_training_pair_interaction_pilot.sh`
- `scripts/stop_training_pair_interaction_pilot.sh`
- `experiments/03_保留专门化/02_论文主线/results/03_强交互Actor_研发记录/D4_pair_interaction_pilot_s20260724/`

Reason: this branch trained a two-agent/pair Actor and does not match the five-agent, full-episode shared-Actor line.

### Oracle-specialist pilot

- `scripts/start_training_interaction_oracle_specialist_pilot.sh`
- `scripts/stop_training_interaction_oracle_specialist_pilot.sh`
- `scripts/start_test_interaction_oracle_repeat.sh`
- `experiments/03_保留专门化/02_论文主线/results/03_强交互Actor_研发记录/D4_interaction_oracle_specialist_pilot_s20260724/`

Reason: this branch depended on oracle interaction routing/specialist behavior and is not part of conventional single shared-Actor TD3 training.

### Complex edge-1 v9 pilot

- `scripts/start_training_full_actor_edge1_pilot.sh`
- `scripts/stop_training_full_actor_edge1_pilot.sh`
- The previous long-form experiment README

Reason: this branch mixed the edge-1 question with a local Critic, cooperative reward, Actor anchors, acceleration caps and recovery objectives. It was replaced by the conventional 5A-style training protocol.

## Still valid outside this archive

Pair metadata remains valid for offline selection of corrected edge-1 scenarios. It must not enter Actor/Critic training as pair identity or routing information.

The edge-1 manifests, their builder, 5A models and launchers, generic TD3/replay/Critic/calibration code, and dataset provenance remain in their original locations.
