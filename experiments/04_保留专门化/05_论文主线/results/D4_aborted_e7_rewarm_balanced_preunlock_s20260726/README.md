# D4 Aborted Epoch-7 Rewarm Run

Status: stopped before Actor unlock; diagnostic only.

This run loaded the Actor from
`interaction_focused_actor_from_5a_s20260725_epoch_007`, initialized a fresh
Critic and Replay, and switched training to the complete 2560-scenario
`balanced_cycle` protocol.

It was stopped at `17762` agent samples and `166` training episodes, before the
configured Actor unlock at `21000` agent samples. Therefore it contains no new
Actor-learning result and must not be treated as a failed candidate.

The setup was rejected because the old epoch-7 Actor had already been trained
with random sampling from the 640-scenario stage-2 pool. Rewarming a new Critic
from that Actor would mix two sampling protocols and would not provide a clean
formal training curve. The replacement restarts the previously successful
all-5A focused-Actor configuration and changes only the training manifest and
sampler.

The complete stopped-process log is retained under `logs/`.
