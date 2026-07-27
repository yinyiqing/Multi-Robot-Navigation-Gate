# D4 Matched Validation of the Interaction Actor

Status: complete. The training-time 5A + interaction-Actor oracle combination reproduced; standalone full-episode use is outside the Actor's training contract.

## Matched Validation

Both policies used seed `20260727` and the same ordered 140-scenario strong-interaction validation manifest.

- Baseline: 5A throughout the episode.
- Candidate: frozen 5A outside the `2.0 m` oracle interaction region and epoch-16 interaction Actor inside it, exactly matching formal training and validation execution.
- No test scenario was read.

| metric | 5A | 5A + interaction Actor | delta |
| --- | ---: | ---: | ---: |
| agent success | 0.8186 | 0.9157 | +0.0971 |
| collision | 0.1814 | 0.0800 | -0.1014 |
| full success | 59/140 = 0.4214 | 98/140 = 0.7000 | +0.2786 |
| timeout | 0 | 2 | +2 |
| mean steps | 35.2 | 54.3 | +19.1 |

Stratified full success:

| band | 5A | 5A + interaction Actor |
| --- | ---: | ---: |
| deep (60) | 0.1833 | 0.5333 |
| close (40) | 0.4000 | 0.8250 |
| margin (40) | 0.8000 | 0.8250 |

Full-success pairing is: both `51`, 5A-only `8`, candidate-only `47`, neither `34`; McNemar exact `p=8.07e-8`. The improvement is concentrated in deep and close, while mean completion time and timeout show a real conservatism cost.

The repeat closely matches the validation-selected formal epoch 16: agent success `0.917 -> 0.916`, collision `0.079 -> 0.080`, full success `0.707 -> 0.700`, and mean steps `54.5 -> 54.3`. The training effect is reproducible under its actual execution contract.

## Standalone Diagnostic

An initially mis-scoped diagnostic forced the interaction Actor to control every state. It was stopped after `83/140` scenarios because the Actor had only been trained and updated as a conditional interaction subpolicy; consecutive 300-step timeouts made the mismatch explicit. This partial run is retained for provenance and is not mixed into the matched result.

The complete 5D standalone run from the same diagnostic is also retained but used seed `20260728`; it is not a paired baseline for the matched 5A comparison.

## Files

- `5a_baseline.*`: complete matched 5A baseline.
- `5a_plus_strong_oracle.*`: complete matched candidate.
- `5d_standalone_diagnostic.*`: complete 5D diagnostic.
- `strong_standalone_partial83.*`: stopped standalone interaction-Actor diagnostic.
