# 单冲突完整 Actor 主线

## 问题

从 5A warm start，在完整路径只有一条冲突边的五车场景中继续训练一个共享 Actor，验证降低冲突复杂度后完整策略能否学会避让。

## 当前协议

- Actor：加载 5A best，五台车共享同一个完整 Actor；
- Critic：fresh 原始 24 维 Critic；
- 场景：`edge1_full_horizon_v1/train.json.gz`，511 个 corrected edge-1 场景；
- replay：保存全部 active agent transition；
- reward：与 5A 一致的 individual navigation reward；
- exploration：普通 TD3 高斯噪声；
- Actor warm-up：前 20000 agent samples 冻结 Actor；
- budget：`6 x 5000` agent samples；
- monitor：固定 50 个 edge-1 validation 场景。

训练过程不读取冲突 pair 身份，不选择受控 ego，不做专家路由，也不使用 Gate 或 local Critic。Pair 元数据只在训练前用于离线筛选 edge-1 场景。

## 启停

```bash
scripts/start_training_full_actor_edge1_from_5a.sh
scripts/stop_training_full_actor_edge1_from_5a.sh
```

默认模型名：

```text
full_actor_edge1_from_5a_s20260803
```

## 判断标准

前四次评估用于确认冻结的 5A 基线和 fresh Critic 训练稳定。Actor 解冻后比较 epoch 5/6 与冻结基线：

- collision 下降且 agent/full success 不下降：继续训练并做完整 validation；
- 指标无变化：调整普通 TD3 的探索噪声或 Actor 学习率；
- collision、timeout 或 unresolved 明显上升：停止，保留解冻前 checkpoint。

旧 v9 pilot、controlled-ego Critic-only v2 和对应审计已归档到 `trash/20260803_ego_pair_training_detours/`，默认不再查看。
