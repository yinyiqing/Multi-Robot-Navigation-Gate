# 第二课程：多车主线回接

## 这门课解决什么

第二课程不是密集交互训练。它的作用是把第一课程学到的单车局部导航能力，接回原来的多车主线：

`stage1 -> 2A -> 2D -> 3A -> 3D -> 3D2`

这里的 A/D/D2 对应原来的实验组：

- A：共享 policy，不用 local critic。
- D：原始局部邻域 critic，critic 看邻居几何和邻居动作。
- D2：几何邻域 critic，critic 只看邻居相对几何。

命名规则：

- `旧3D2`：旧三车主线里的 D2，warm start 来自旧统一 baseline。
- `课程3D2`：第二课程链路里的 D2，warm start 来自课程 3A guarded actor。
- 两者方法同属 D2，但初始化路径和训练保护条件不同，不能混写成一个结果。

## 当前主线节点

| 节点 | 目录 | 结论 |
| --- | --- | --- |
| 2A 共享 Policy | `01_2A_共享Policy/` | 第一课程 best 接回 2A，普通测试优于旧 2A。 |
| 2D 局部 Critic | `02_2D_局部Critic/` | gentle 版可作为当前 2D 节点；best 在早期，后续 actor 更新会退化。 |
| 3A 共享 Policy | `03_3A_共享Policy/` | guarded best 300 集 full success `0.827`，作为当前 3A 节点。 |
| 3D 原始局部 Critic | `04_3D_原始局部Critic/` | 300 集 full success `0.813`，好于旧 3D，但 best 在 actor 解冻前。 |
| 3D2 几何邻域 Critic | `05_3D2_几何邻域Critic/` | 40 集 eval epoch 3 full success `0.925`，后续 latest 明显退化；正在测 best 的 300 集。 |
| 诊断与旁路 | `诊断与旁路/` | pairwise warmup、失败启动、过早 repair 等，不作为主线节点。 |

## 关键教训

- 课程学习主要继承 actor/policy，不应把不同输入结构的 critic 硬继承。
- 2D、3A、3D、3D2 多次出现“早期 best 好，继续更新 actor 后退化”。这说明问题不是 warm start 不行，而是新 critic 或新阶段数据还没稳定时，actor 更新容易把已会的目标收敛能力带偏。
- 当前的 best checkpoint 可以作为节点测试，但不能把“解冻前 best”说成“该阶段 actor 更新已经学成”。
- 如果后续要证明 D/D2 阶段真的提升了 policy，需要单独设计更保守的 actor 更新策略，并记录“解冻后 best”。

## 当前进行中

正在测试课程 3D2 best：

- model: `TD3_velodyne_multi_v4_curriculum_stage2_to_3d2_geo_critic_from_3a_guarded_best`
- active log: root `logs/test_multi_stage2_three_geo_critic_best_...20260608_084901.log`
- 跑完后归档到 `05_3D2_几何邻域Critic/logs/test/`

## 细节档案

完整逐轮解释和旧路径记录保留在：

- `README_旧版长记录.md`

