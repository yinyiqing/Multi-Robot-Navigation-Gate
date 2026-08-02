# epoch-16完整训练状态分叉pilot

状态：`运行中`。

## 假设

epoch-16失败于独立导航，不等于它的训练状态不可继续。原完整checkpoint包含：

- 与冻结epoch-16逐张量一致的24维Actor；
- 已经支持有效局部避障的87维ego-motion Critic；
- `320000`条replay，其中`147696`条为交互样本；
- 冻结5A负责普通状态、epoch-16负责交互状态时采集的完整轨迹。

因此直接分叉这套训练状态，比从随机Critic重新学习更符合“局部专家全程化”的目标。

## 单变量协议

相对epoch-16原训练只改变执行和Actor更新范围：

- 训练和验证中Actor B全程独立执行，关闭Oracle rollout；
- Actor从interaction-only更新改为全部状态更新；
- 训练期Bellman target继续使用原教师契约：`>2.0 m`由冻结5A产生target action，`<=2.0 m`由Actor B产生；
- Oracle target只进入训练Critic，不参与环境执行或验证；
- Critic batch按`50% interaction + 50% normal`采样；
- 保持原reward、Actor/Critic学习率、安全ranking和gradient guard；
- 使用完整dense train，validation为固定50场ultrafast monitor。

源checkpoint：

```text
TD3/checkpoints/interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_latest.pt
```

训练分叉不会修改源checkpoint。

## 两轮预算

- epoch 17：从`320000`累计到约`325000` agent samples，Actor保持冻结，获得同协议独立运行基线；
- epoch 18：Actor在`326500` samples后解冻，运行到约`330000` samples，检查第一段全状态更新趋势；
- 每轮验证50个固定scenario；任一明显碰撞、timeout或全局动作偏置退化立即停止。

该pilot只回答“复用专家完整训练状态后，第一步全程化是否有正向趋势”，不直接作为论文最终结果。

## 脚本

- 启动：`scripts/start_training_actor_b_from_epoch16.sh`
- 停止：`scripts/stop_training_actor_b_from_epoch16.sh`
