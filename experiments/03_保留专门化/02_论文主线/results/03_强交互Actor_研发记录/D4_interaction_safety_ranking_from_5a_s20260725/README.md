# D4 Critic Safety Ranking Pilot From 5A

Status: rejected controlled pilot.

## Result

| metric | epoch 1 frozen 5A | epoch 2 trained | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.839 | 0.830 | -0.009 |
| collision | 0.161 | 0.170 | +0.009 |
| full success | 0.500 | 0.464 | -0.036 |
| deep full success | 0.217 | 0.133 | -0.084 |
| close full success | 0.550 | 0.575 | +0.025 |
| margin full success | 0.875 | 0.850 | -0.025 |

## Diagnosis

The close-approaching Critic ranking loss corrected dangerous-state linear positive-gradient share from the previous `0.96-1.00` to roughly `0.20-0.35`, and the Actor gradient gate passed consistently late in epoch 2. However, the Actor still sampled every interaction within `2.0 m`, while the ranking constraint covered only close, approaching states. After training, Actor raw linear action increased by `+0.012` at `<=1.2 m`, and angular action acquired a global `+0.093` offset.

The Critic correction is retained, but this Actor update distribution is rejected. The next controlled pilot updates the Actor only from close-approaching samples and anchors only its angular output to 5A. Offline replay gives dangerous-state linear delta about `-0.027`; angular-anchor weight `2` reduces the global angular shift from about `+0.048` to `+0.009` without weakening deceleration.

Large replay checkpoints remain local and are not included in Git.
