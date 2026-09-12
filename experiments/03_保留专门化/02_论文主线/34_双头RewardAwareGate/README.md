# G34 Dual-Encoder Reward-aware Gate

> Historical note: the former independent evaluation on `[640:896]` is retained for audit only.
> The current paper evaluation uses the G25-slice matched protocol in
> `../36_G34_G25slice_matched/` with `[0:256]`; do not use the former result for current tables or claims.

Status: offline admission and checkpoint audit passed; Dense32 development pilot authorized.

## Objective

Implement the advisor's replacement method directly. Each deployable Gate feature history is mapped to:

```text
p_y       = predicted 2 m interaction phase
A_hat     = predicted bounded one-step reward difference, r_I - r_N
```

The two predictions jointly form the routing score:

```text
score = sigmoid(logit_y + 0.25 * A_hat)
```

The existing `0.43/0.33` hysteresis thresholds, three Router-update minimum hold, and routing stride two
are retained. The navigation Actor, interaction Actor, G0/G1 perception, manifests, and reward function
remain frozen.

## Architecture and training

The two heads use separate 64-dimensional GRUs. The phase branch is trained from scratch on the complete
G11 A1 plus student-visited Gate frames with the privileged 2 m label. The reward branch is trained on all
audited G27/G33 counterfactual records, with per-scene weighting so repeated environment seeds do not count
as independent scenes. No B2 parameters initialize either branch.

Terminal rewards are retained because a one-step collision or goal is meaningful policy evidence. To prevent
the approximately 100-point terminal reward from dominating regression, the target is transformed using the
training-only P95 absolute reward difference:

```text
A = tanh(clip(Delta r, -s, s) / s),  s = P95_train(|Delta r|)
```

## Offline admission

Before any closed-loop rollout:

1. all input hashes and train/validation scene separation must pass;
2. reward validation Huber loss must beat a train-median constant predictor;
3. reward prediction Pearson correlation must be positive;
4. adding the reward prediction may reduce phase-label validation F1 by at most 0.02 and increase FPR by at most 0.02;
5. all saved model outputs and normalization values must be finite.

If any condition fails, G34 stops offline. If all pass, run one 32-episode Dense development pilot first.
No 256-episode run is authorized by this protocol.

## Offline result

Training completed on seed `20260912`. The phase branch used `70,981` training frames and `4,785`
validation frames. The reward branch used `548` audited records from `356` unique training anchors in
`336` scenes, with `132` validation records from `65` scenes. Repeated measurements are scene-weighted.

The reward validation weighted Huber loss is `0.01841`, below the train-median constant baseline
`0.04817`; Pearson correlation is `0.76267`, and reliable reward-sign accuracy is `0.88`. Adding the
predicted reward advantage changes phase F1/FPR from `0.83675/0.22792` to `0.83603/0.22548`.
All preregistered offline criteria passed.

The original trained checkpoint is `local_data/training/seed20260912/best.pt`, SHA-256
`f3f70ffd822eb07449a05cfc0ee5ecc21c34f3b8991d9315873fc28d1f38128a`. Its first saved model ID
incorrectly retained the G33 data-stage name. No weights were changed: checkpoint audit creates
`best_runtime.pt` with only the model ID corrected to `G34-dual-encoder-reward-aware`, verifies exact
state-dict equality, input hashes, finite normalization and finite outputs, and records the result in
`checkpoint_audit.json`.

## Dense32 development pilot

The first 32 scenes of the frozen Dense validation manifest are evaluated with seed `20260912`, first
for G34 and then for the frozen 5A control. B2 is not an adoption baseline. This pilot is diagnostic and
does not authorize an inferential claim.

The decision rule was fixed before reading the pilot outcomes:

1. scenario order and all `32 x 5` robot terminal outcomes must audit correctly;
2. G34 must select both Actor modes (`0.05 <=` mean interaction share `<= 0.95`) and switch at least once;
3. pass for a fixed 64-episode confirmation only if full-success count is no lower than same-seed 5A,
   robot-collision count is no higher, and at least one is strictly better;
4. a deficit of at most one full-success episode and at most two additional robot collisions is
   inconclusive; anything worse is a failed pilot;
5. an inconclusive or failed pilot stops before expansion. No 256-episode run is launched automatically.

Entry point: `scripts/start_g34_pilot.sh`. Live log: `logs/pilot32/queue.log`.

### Dense32 result

The pilot completed without retries. G34 achieved `11/32` full successes (`34.38%`) versus `7/32`
(`21.88%`) for same-seed 5A, a directional difference of `+12.50 pp`. Robot collision was `42/160`
(`26.25%`) versus `51/160` (`31.88%`), a difference of `-5.63 pp`. There were no unresolved robots or
timeout episodes. G34 selected the interaction Actor on `68.19%` of frames and made 243 switches.

All preregistered pilot checks passed. These 32 scenes only authorize a disjoint 64-episode Dense
development confirmation; they are not an inferential result and do not authorize a 256-episode run.

## Dense64 development confirmation

Before running, the next 64 validation scenes were frozen as source slice `[32:96]` in
`local_data/protocol/dense_confirmation64.json.gz`, SHA-256
`4f70d1f7536f9619c5f687ea98c24d3a68b985e9a1d3673066d5df8c1f5ed4f7`. The overlap with the
Dense32 pilot is zero. G34 and 5A use the same seed `20260913` and identical scene order.

The confirmation passes only if terminal accounting and scenario order are valid, both Actor modes are
selected, G34 full-success count is no lower than 5A, robot-collision count is no higher than 5A, and at
least one of the two outcomes is strictly better. A failure stops the method before independent evaluation;
a pass only authorizes freezing G34 and registering a separate independent evaluation. It does not launch
or authorize a 256-episode run automatically.

Entry point: `scripts/start_g34_confirmation64.sh`. Live log: `logs/confirmation64/queue.log`.

### Dense64 result

The disjoint confirmation completed without retries. G34 achieved `27/64` full successes (`42.19%`)
versus `20/64` (`31.25%`) for same-seed 5A, a directional difference of `+10.94 pp`. Robot collision
was `68/320` (`21.25%`) versus `84/320` (`26.25%`), a difference of `-5.00 pp`. Neither method had an
unresolved robot or timeout episode. G34 interaction selection share was `73.60%`, with 458 switches.
All confirmation checks passed.

## Independent Dense test

G34 is now frozen for independent evaluation. The preregistered test uses the previously unused original
Dense test slice `[640:896]`, 256 scenes, seeds `20260914/20260915/20260916`, and only G34 plus same-scene
5A (`1536` episodes total). G25 used `[0:256]`, G26 used `[256:384]`, and G27 used `[384:640]`; the new
manifest must also pass scene-ID and complete-geometry overlap audits against train, validation, and all
three earlier test slices before launch.

The primary endpoint is episode-level full success. Its G34-minus-5A difference uses a scene-cluster BCa
95% interval and a two-sided scene-level sign-flip test at `alpha=0.05`. Robot-level collision is the
secondary safety endpoint with the same scene clustering. Success, collision, paired-success steps,
interaction selection share and switches are all reported. B2 is neither run nor used as the primary
baseline. The full frozen protocol is `local_data/independent_test/protocol.json`.

No performance early stopping or post-result Router tuning is allowed. The result is retained even if a
claim check fails.

The queue started at `2026-09-12 11:19 CST`. The PID file is `.g34_independent_test.pid`, the queue log is
`logs/independent_test/runner.log`, and per-run logs follow
`logs/independent_test/g34_p3_<method>_s<seed>_attempt<n>.log`. Frozen statistics are generated
automatically at `local_data/independent_test/statistics.json` after all six runs finish.
