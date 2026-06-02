# Stage 1 Single Local Cases Warm-Start Smoke Test

## Run

- model: `TD3_velodyne_multi_v4`
- scenario: `curriculum`
- cases: `../../cases/stage1_single_local_cases.json`
- agents: 1
- episodes: 6
- sampling: `cycle`
- purpose: verify the curriculum reset/test path and confirm the targeted local cases expose known single-agent weaknesses.

## Result

| metric | value |
| --- | ---: |
| success_rate | 0.500 |
| collision_rate | 0.333 |
| unresolved_rate | 0.167 |
| full_success_rate | 0.500 |
| timeout_episode_rate | 0.167 |

## Per-Case Outcomes

| episode | case order | success | collision | timeout | final_distance | steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | near_goal_capture_front | 1 | 0 | 0 | 0.264 | 17 |
| 2 | near_goal_capture_side | 1 | 0 | 0 | 0.286 | 27 |
| 3 | wall_separated_south | 0 | 1 | 0 | 1.638 | 251 |
| 4 | wall_separated_north | 0 | 1 | 0 | 2.645 | 17 |
| 5 | near_obstacle_recovery | 0 | 0 | 1 | 2.595 | 300 |
| 6 | offset_goal_approach | 1 | 0 | 0 | 0.264 | 15 |

## Interpretation

The curriculum smoke test is valid: the reset path runs, case cycling works, and the existing warm-start model fails on the targeted local-defect cases. The two wall-separated cases fail by collision, and the near-obstacle recovery case times out. This supports Stage 1 as a useful first course before returning to dense multi-agent interaction.

## Files

- `test_multi_curriculum_stage1_single_TD3_velodyne_multi_v4_6episodes.raw.log.gz`
- `test_multi_curriculum_stage1_single_TD3_velodyne_multi_v4_6episodes.npy`
