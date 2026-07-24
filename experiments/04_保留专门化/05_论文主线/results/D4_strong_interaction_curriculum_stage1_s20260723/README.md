# D4 Strong-Interaction Curriculum Stage 1

状态：`rejected candidate`。Stage 1 没有通过 close 提升、margin 不明显退化的准入条件，因此不进入 Stage 2。

## 协议

- base Actor/Critic：完整复制 5D warm-start；原始 5D 不修改。
- train：`256 close + 128 margin`，固定场景随机采样，不包含 deep。
- validation：固定 `60 deep + 40 close + 40 margin`，两轮完全相同。
- epoch 1：Actor 锁定到 `21000` agent samples，作为同协议 5D 基线。
- epoch 2：完整 `24 -> 800 -> 600 -> 2` Actor 参与训练。
- reward：距离加权 `0.8 self + 0.2 visible-neighbor`。
- Actor/Critic learning rate：`1e-6 / 8e-5`；没有 GRU、额外 reward 或结构改动。

## 结果

| Metric | Epoch 1: frozen 5D | Epoch 2: trained Actor | Delta |
| --- | ---: | ---: | ---: |
| Agent success | 0.8386 | 0.7771 | -0.0614 |
| Collision | 0.1614 | 0.2100 | +0.0486 |
| Unresolved | 0.0000 | 0.0129 | +0.0129 |
| Full success | 0.4929 | 0.3357 | -0.1571 |
| Timeout episodes | 0.0000 | 0.0643 | +0.0643 |
| Mean steps | 32.49 | 59.44 | +26.94 |
| Close full success | 0.6250 | 0.2750 | -0.3500 |
| Deep full success | 0.2167 | 0.1333 | -0.0833 |
| Margin full success | 0.7750 | 0.7000 | -0.0750 |

`best` 位于 epoch 1，但 epoch 1 Actor 与原始 5D 参数逐位相同，因此它不是训练得到的强交互专家。

## 失败机制

使用 `latest` checkpoint 中全部 `40001` 个 replay state，对原始 5D、epoch 2 Actor 和 epoch 2 Critic 做离线审计：

- epoch 2 的线速度 action 相对 5D 平均增加 `0.1080`；mean absolute delta 也是 `0.1080`，即所有 replay state 上只增不减；
- `26.15%` 的 state 至少一个动作维度变化超过 `0.05`；
- 在这些真正变化的 state 上，epoch 2 Critic 有 `76.80%` 判断新动作 Q 更高，平均高 `2.622`；
- 但真实 validation 的碰撞、超时和三个风险分层全部变差；
- Actor 各层参数 relative L2 只变化约 `0.16%--0.74%`，仍足以造成 `5.38%/3.84%` 的线速度/角速度符号翻转。

因此当前 Actor 没有学到“强交互时让行”，而是利用 Critic 的速度偏好形成普遍加速。Critic 的当前几何输入和稀疏终局反馈不能可靠区分“快速通过”与“即将碰撞”，继续增加 epoch、seed 或直接进入 deep 只会重复已经出现过的价值误导。

可复现命令：

```bash
source env.python.sh
python3 scripts/analyze_actor_critic_drift.py \
  --base-actor TD3/pytorch_models/TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best_actor.pth \
  --candidate-actor TD3/pytorch_models/strong_interaction_curriculum_stage1_s20260723_epoch_002_actor.pth \
  --critic TD3/pytorch_models/strong_interaction_curriculum_stage1_s20260723_epoch_002_critic.pth \
  --checkpoint TD3/checkpoints/strong_interaction_curriculum_stage1_s20260723_latest.pt
```

大型 replay checkpoint 不纳入 Git。归档保留两轮 Actor/Critic、evaluation 数组、完整日志和 TensorBoard event；哈希见 `summary.json`。
