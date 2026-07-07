# 01 冲突验证

这一步先不急着做切换。

先回答一个最关键的问题：

**单一 actor 继续往密集场景训练时，会不会破坏它原来已经学好的普通导航能力。**

## 现在要验证什么

我们想验证的不是“dense actor 更强”这么简单，而是下面这件事：

1. 普通 actor 在普通五车场景里更稳
2. 往密集场景继续训练后，模型会更偏向密集交互
3. 但这种继续训练，可能会损伤它在普通场景里的原有能力

如果这三点成立，就可以把问题正式定义成：

**单一共享 actor 在普通导航能力和密集协同能力之间存在冲突。**

## 第一轮先用现成模型

先不训练新模型，先拿已经有的模型做交叉测试。

### A. 普通 actor

- `TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`

这是现在最自然的普通五车 actor。

### B. 密集 actor

- `TD3_velodyne_multi_v4_curriculum_stage3_asym_pair_5_from_5a_cleanstart_v2_best`

这是现在最自然的密集 actor 候选。

### C. 继续训练后的单一 actor

先选一版已经继续往 `stage3_asym_three_5` 训练过的模型，作为 overwrite 路线代表。

第一轮先看：

- `TD3_velodyne_multi_v4_curriculum_stage3_asym_three_5_joint_action_critic_midcheck_from_stage3_pair_cleanstart_v2_best_best`

后面如果需要，再补别的 `stage3_asym_three_5` 版本做对照。

## 第一轮测试矩阵

### 普通场景

先测标准五车普通场景：

- 普通 actor -> 普通场景
- 密集 actor -> 普通场景
- overwrite actor -> 普通场景

这里主要看：

- success_rate
- collision_rate
- full_success_rate

### 密集场景

再测 `stage3_asym_three_5`：

- 普通 actor -> 密集场景
- 密集 actor -> 密集场景
- overwrite actor -> 密集场景

这里主要看：

- success_rate
- collision_rate
- full_success_rate
- timeout_episode_rate

## 怎样算“冲突成立”

如果出现下面这种格局，就说明这个问题基本成立：

1. 普通 actor 在普通场景最好或最稳
2. 密集 actor / overwrite actor 在密集场景更强
3. 但 overwrite actor 在普通场景明显差于普通 actor

这就说明：

- 往 dense 继续训练不是“纯增益”
- 它会带来能力偏移
- 也就是我们想说的 preserve-and-specialize 动机

## 当前记录

- 已确定普通 actor
- 已确定密集 actor
- 已确定第一版 overwrite actor
- 下一步直接启动第一批交叉测试
