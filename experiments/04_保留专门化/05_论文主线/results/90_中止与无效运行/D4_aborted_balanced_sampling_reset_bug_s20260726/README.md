# Aborted Balanced-Sampling Run

Status: invalid training-coverage pilot; excluded from model selection.

The run used the intended `deep/close/margin = 40/40/20` balanced-cycle
selection, but switching to the validation manifest reset the per-band training
cursor after every epoch. After roughly 900 training episodes, only 212 of the
2560 unique train scenarios had been visited. The run was stopped during epoch
5 and must not be reported as the full-pool result.

Validation results before stopping were:

| epoch | full success | deep | close | margin |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.579 | 0.400 | 0.575 | 0.850 |
| 2 | 0.543 | 0.333 | 0.600 | 0.800 |
| 3 | 0.571 | 0.367 | 0.575 | 0.875 |
| 4 | 0.529 | 0.350 | 0.500 | 0.825 |

The replacement preserves and checkpoints the training-manifest sampling state
across validation. It starts again from the selected epoch-7 Actor with a fresh
Critic and replay buffer.
