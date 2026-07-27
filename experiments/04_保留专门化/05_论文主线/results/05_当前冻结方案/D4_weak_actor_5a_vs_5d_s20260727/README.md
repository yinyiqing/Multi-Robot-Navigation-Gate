# D4 Weak-Interaction Actor Comparison: 5A vs 5D

Status: complete. Select 5A as the weak-interaction Actor because performance is equivalent to 5D and the validated interaction Actor was trained with 5A outside the interaction region.

## Protocol

- Fixed `weak-interaction-validation-v1`: 206 standard + 42 dense scenarios, all with policy-independent `conflict_edge_count = 0`.
- Seed `20260719`; five agents; single Actor throughout each episode.
- 5A was newly evaluated on the 248-scenario view.
- 5D values are extracted by identical scenario ID from its existing complete standard-500 and dense-1000 validation runs with the same seed. Test was not read.

## Result

| pool | model | agent success | collision | full success | timeout | mean steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| standard (206) | 5D | 0.9680 | 0.0291 | 176/206 = 0.8544 | 3 | 40.3 |
| standard (206) | 5A | 0.9699 | 0.0282 | 177/206 = 0.8592 | 2 | 41.8 |
| dense (42) | 5D | 0.9905 | 0.0048 | 40/42 = 0.9524 | 1 | 30.3 |
| dense (42) | 5A | 0.9857 | 0.0143 | 40/42 = 0.9524 | 0 | 26.6 |
| overall (248) | 5D | 0.9718 | 0.0250 | 216/248 = 0.8710 | 4 | 38.6 |
| overall (248) | 5A | 0.9726 | 0.0258 | 217/248 = 0.8750 | 2 | 39.3 |

Full-success pairing over all 248 scenario IDs is: both `205`, 5D-only `11`, 5A-only `12`, neither `20`; McNemar exact `p=1.0`. There is no evidence that either model is better on weak interaction.

The weak-interaction result alone would permit either model. The tie is resolved in favor of 5A because the formal interaction Actor was initialized from 5A and its matched validation uses frozen 5A outside the interaction region. Selecting 5A therefore keeps the final system to two parameter-independent Actors and one deployable Gate without adding a separate 5D-to-5A compatibility problem. 5D remains a required historical/generalist baseline.

## Files

- `5a_weak_248.log`, `5a_weak_248.npy`, `5a_weak_state.pt`: complete 5A evaluation.
- The 5D source artifacts remain in `D4_standard_expert_timeoutfix_v3_validation500` and `D3_fixed_v1_generalist_validation`; they are referenced rather than duplicated.

SHA-256:

```text
0c2400c71d306dfbb116502e522bd053f510e20945634a9735977d2fc479dc6f  5a_weak_248.log
0feaa493b6e6dede31c00627294d51f13d6243e8552bb213d2b3513e5a100f05  5a_weak_248.npy
05f58a8e3d893fd144f6973169e9c9433146757edc09e51c4efcc56a530ea753  5a_weak_state.pt
142e5a2316bdc572038fd5316d007869cd365b8f0fa5637e08cf22133f2e521e  weak-interaction validation manifest
```
