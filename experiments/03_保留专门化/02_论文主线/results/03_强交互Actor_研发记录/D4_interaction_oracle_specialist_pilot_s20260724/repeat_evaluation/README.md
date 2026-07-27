# Repeated Paired Evaluation

This directory compares the frozen 5D Actor with the epoch 2 oracle combination on the same ordered 140-scenario validation manifest. Repeat 1 uses seed `20260725`; repeat 2 uses seed `20260726`. The original training-time validation is treated as repeat 0.

## Per Run

| run | policy | agent success | collision | full success | unresolved | timeout |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5D | 579/700 | 121/700 | 70/140 | 0 | 0 |
| 0 | epoch 2 | 584/700 | 116/700 | 66/140 | 0 | 0 |
| 1 | 5D | 572/700 | 128/700 | 58/140 | 0 | 0 |
| 1 | epoch 2 | 574/700 | 125/700 | 60/140 | 1 | 1 |
| 2 | 5D | 579/700 | 121/700 | 64/140 | 0 | 0 |
| 2 | epoch 2 | 570/700 | 130/700 | 61/140 | 0 | 0 |

The direction of the overall full-success difference changes across runs, confirming substantial simulator variance. Aggregating all three runs gives 5D/epoch 2 agent success `0.8238/0.8229`, collision `0.1762/0.1767`, and full success `0.4571/0.4452`.

## Interaction Strata

| stratum | policy | agent success | collision | full success |
| --- | --- | ---: | ---: | ---: |
| deep, 180 episodes | 5D | 647/900 | 253/900 | 35/180 |
| deep, 180 episodes | epoch 2 | 640/900 | 260/900 | 27/180 |
| close, 120 episodes | 5D | 518/600 | 82/600 | 65/120 |
| close, 120 episodes | epoch 2 | 521/600 | 79/600 | 66/120 |
| margin, 120 episodes | 5D | 565/600 | 35/600 | 92/120 |
| margin, 120 episodes | epoch 2 | 567/600 | 32/600 | 94/120 |

Deep full success declined in every run: `14 -> 12`, `8 -> 7`, and `13 -> 8`. The checkpoint is therefore not a strong-interaction specialist even though close and margin are roughly preserved or slightly improved. Do not continue the unchanged training configuration to epoch 3.

## Files

- `5d_r1.npy`, `epoch2_r1.npy`, `5d_r2.npy`, `epoch2_r2.npy`: per-episode records in manifest order.
- Corresponding `.log` files: complete console and Gazebo output.
- `summary.json`: machine-readable aggregate.
