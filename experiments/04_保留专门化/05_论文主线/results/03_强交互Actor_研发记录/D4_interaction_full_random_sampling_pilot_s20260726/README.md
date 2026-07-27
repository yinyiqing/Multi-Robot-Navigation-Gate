# Full Strong-Interaction Random-Sampling Pilot

Status: rejected diagnostic; superseded by `balanced_cycle`.

The pilot used the complete 2560-scene strong-interaction train pool but sampled
scenarios uniformly at random. Because the pool contains 951 deep, 437 close,
and 1172 margin scenarios, this protocol under-sampled close interactions and
was not used as a paper result.

Two startup variants are retained under `logs/`:

- the original-5A run stopped before its first validation;
- the old epoch-7 run completed one validation (`0.564` full success) and was
  then stopped.

Neither value is a formal result: the sampler was uniform random with
replacement, and the epoch-7 Actor already came from the earlier 640-scenario
protocol. The formal replacement restarts from original 5A and uses the fixed
`balanced_cycle` sampler documented in the curriculum view README.
