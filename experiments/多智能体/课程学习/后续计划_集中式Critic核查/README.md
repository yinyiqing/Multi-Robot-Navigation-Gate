# 后续计划：集中式 Critic 核查

## 背景

当前课程学习主线已经确认：

- 前面较简单课程是有效的
- 更复杂的五车密集阶段里，warm-start actor 本身可用
- 但 actor 一旦继续更新，就容易退化

导师给出的新建议，不是继续盲调解冻时机，而是先确认：

- 我们当前的集中式 critic / 分布式 actor 这条线，是否已经按 MADDPG / MAPPO 一类方法的关键细节改对

## 当前目标

这一轮的目标不是继续扩新实验，而是先完成一次实现核查，回答：

1. critic 现在到底“看到了什么”
2. replay buffer 现在到底“存了什么”
3. actor 更新时，其他智能体相关动作和梯度是否处理正确
4. 当前实现和标准 CTDE 思路相比，还差哪些关键细节

## 核查清单

### 1. Buffer

- 当前 replay buffer 是否保存了足够的多机信息：
  - 当前时刻多机状态
  - 当前时刻多机动作
  - 下一时刻多机状态
  - done / active mask
- 如果没有，当前 critic 实际上只能算“部分多机 context critic”，而不是更标准的 centralized critic

### 2. Critic 输入

- critic 输入到底是：
  - 本机 observation + 本机 action + 邻居几何
  - 还是 joint observation + joint action
- 当前 local / geometry critic 的信息范围，与 MADDPG 风格 centralized critic 有多大差距

### 3. Actor 更新

- 以当前智能体为中心更新 actor 时：
  - 其他智能体动作是否参与 critic 计算
  - 如果参与，是否应该 detach
- 共享 actor 条件下，这里的梯度传播是否会产生不合理串扰

### 4. Target 计算与 mask

- target critic 的 next action 是如何构造的
- done / active agent / finished agent 的 mask 是否正确进入 target 和 loss
- 是否存在“已完成或非活跃 agent 仍污染 centralized critic 训练”的问题

## 完成标准

这一轮结束时，需要产出：

1. 一份“当前实现与 MADDPG / MAPPO 常见做法对照表”
2. 一份“可能影响 critic 是否真正管用的问题清单”
3. 明确下一步到底是：
   - 先修集中式 critic 实现细节
   - 还是可以转去 BC / 保守 critic / 更强 ensemble

## 暂定顺序

1. 先读我们当前训练代码和 replay buffer
2. 再对照 MADDPG / MAPPO 论文和常见实现
3. 最后整理成问题清单和修改建议
