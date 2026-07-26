# Full Strong-Interaction Random-Sampling Pilot

Status: stopped before formal evaluation.

The pilot used the complete 2560-scene strong-interaction train pool but sampled
scenarios uniformly at random. Because the pool contains 951 deep, 437 close,
and 1172 margin scenarios, this protocol under-sampled close interactions and
was not used as a paper result.

The run was stopped before the first validation cycle and retained only as a
diagnostic record. The formal replacement uses the fixed `balanced_cycle`
sampler documented in the curriculum view README.
