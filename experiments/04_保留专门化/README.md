# 保留专门化

这里放从“保留旧能力”一路收敛到“局部交互增强”的主线。

完整故事稿见：

- `README_故事版.md`

## 这条线回答什么

一句话：

**如何把已经学会普通导航的多机器人策略扩展到密集交互场景，同时不破坏它原来已经学好的能力。**

## 为什么会转到这里

前面的结论已经比较清楚：

1. 课程学习前两段有效：
   - 单车补稳有效
   - 普通多车回接有效
2. 到更复杂的密集多车阶段后：
   - warm start 模型本身还能跑
   - 但 actor 一继续更新，性能容易退化
3. `PAIR(from_5d) -> THREE_5` 验证后发现：
   - warm start 本身可用
   - 继续更新 actor 仍会逐轮退化

所以现在的判断是：

**主问题不再只是 critic，而是单一 actor 在普通导航和密集协同之间可能存在能力冲突。**

补充一个容易混的点：

- 我们不是没做过“纯 dense 五车”主线
- 旧线里已经直接试过 `stage2_dense_gentle` 和 `stage2_dense_bridge`
- 结果都不理想：同步五车中心强交互太难，课程起点不顺，actor 继续更新后还会退化
- 现在的 `PAIR / THREE_5` 不是换题，而是把原来那条纯 dense 死路拆成更可学的渐进交互版本

## 现在的主线

一句话：

**先把第二个 dense 专家训出来，再考虑用 gate / attention 去保留普通能力并增强密集交互。**

当前顺序很重要：

1. `5D` 先作为普通到 dense 的最强 bridge baseline。
2. 直接 `stage4_asym_dense_5` 太硬，先作为压力测试。
3. 新增 `stage4_asym_dense_5_bridge` 作为 dense 专家训练入口。
4. 等 dense 专家真的比 `5D` 更专，再训练 gate / attention；否则 gate 只会学成“多数时候选 5D”。

## 这一目录后面准备怎么放

- `01_冲突验证/`
  - 先证明单一 actor 继续往 dense 训练时，会不会破坏原有普通场景能力
- `02_双actor切换/`
  - 作为中间验证：能力偏向存在，但粗切换不够
- `03_门控注意力增强/`
  - 当前主线：先训练 dense 专家，再做 gate / attention 组合
- `04_安全兜底/`
  - 如果主线有效，再把传统规划或安全控制并进来

现在目录已经先建好，后面所有新实验都优先往这 4 个子目录里放。

## 当前一句话记录

- 旧主线卡点：actor 解冻后退化
- `5A + 5D` 互补不足，不能直接训练 gate
- 新主线：先补一个真正更适合 dense 的第二专家

## 当前故事

1. 直接继续训练单一 actor，容易退化。
2. `5A` 更像普通导航 actor。
3. `5D` 更像普通到 dense 的桥接 actor。
4. `PAIR(from_5d)` 是当前 dense 专门化基线，但正式 dense 测试没有超过 `5D`。
5. `5A + 5D` 的 hard switch 和 oracle 显示互补不足，不能直接进入 learned gate 训练。
6. 当前先回到专家训练：如果没有更专的 dense actor，gate / attention 的故事站不稳。

## 当前测试口径

- `standard_5`
  - 普通场景主测试
- `stage3_asym_three_5`
  - dense 场景主测试
- `stage2_dense`
  - 只作为压力测试，不作为当前主 benchmark

### 当前模型角色

- 冻结基础 Actor：`5D`
- 新训练模型：`stage4_asym_dense_5_bridge` 上的 dense expert
- `5A`、`PAIR(from_5d)`、`THREE_5`：历史对照，不进入新训练图

额外约定：

- `5D` 不是被淘汰，而是当前最强的 bridge baseline
  - 它在 `stage3_asym_three_5` 测试里已经达到 `0.902 / 0.097 / 0.650`
  - 说明它本身已经具备较强 dense 泛化
- `PAIR(from_5d)` 是当前 dense specialized baseline
  - 它代表“在 bridge actor 基础上继续做密集专门化”后的版本
  - 训练内 eval 已达到 `0.921 / 0.079 / 0.750`
  - 正式测试为：`standard_5` 的 `0.891 / 0.085 / 0.573`，`stage3_asym_three_5` 的 `0.880 / 0.122 / 0.575`
  - 正式 dense 测试没有超过 `5D`，因此暂不能直接定为 dense expert
- `THREE_5` 是更难一级的扩展场景，不作为当前 baseline
- 旧 `stage2_dense` / `stage2_dense_gentle` 保留为失败前史和压力边界，不再当主线训练入口

补充：

- `2026-07-13`：`5D -> stage3_asym_pair_5` 已完成 3 epoch，第三轮达到 `0.921 / 0.079 / 0.750`，但正式测试未超过 `5D`。

## 当前简单记录

- `2026-07-12`：进入 `02_双actor切换`，先做最小推理版：默认 `5A`，交互变强时切到 `PAIR`。
- `2026-07-12`：粗双 actor 切换两边都没赢，主线开始从“策略硬切换”收敛到“局部交互感知的门控注意力增强”。
- `2026-07-13`：`5A + 5D` case-level oracle 三个 case 全选 `5D`，确认二者互补不足；learned gate 暂缓，先寻找更专门化的专家组合。
- `2026-07-13`：停止双 Actor gate 路线，建立冻结 `5D` 的本地时空 Attention 残差主线。
- `2026-07-13`：旧 Attention run 在约 670 episode 停止。best 在 episode 300，后续 gate/residual 退化为近常量且角速度残差翻转；该结果只说明旧目标不稳定，不能证明时空 Attention 有效。
- `2026-07-13`：建立 balanced v2：三组分层回放、reward scale、非饱和 gate/residual 约束、固定评估、Actor 衰减和无提升早停。
- `2026-07-15`：新仓库 `Multi-Robot-Navigation-Gate` 独立出来，gate / dense-expert 工作不再污染原主线仓库。
- `2026-07-15`：确认 hard `stage4_asym_dense_5` 对 `5D` 太硬，固定策略 40 集 success `0.355`、collision `0.680`、full success `0.025`，只作为压力测试。
- `2026-07-15`：新增 `stage4_asym_dense_5_bridge`，几何约束为最小可能起点间距约 `1.04m`、目标间距约 `0.72m`；`5D` 固定策略 40 集 success `0.540`、collision `0.475`、full success `0.250`，适合作为 dense 专家训练入口。

## Dense5 排查结论

`2026-07-14` 的 `stage4_asym_dense_5` 排查后，当前先记住这几件事：

1. 刚才那次 `dense5-from-5d` 失败 run 不能拿来判断 `warm start` 本身好坏。
2. 当时真正的问题首先是配置不一致，不是单纯“5D actor 不适合继续训”。
3. 现在 `stage4_asym_dense_5` 已先改回更接近原始 `5D` 的配方，但还没有重新开跑。

### 三版对照

| 项目 | 原始 5D 成功版 | 刚才错误 dense5 版 | 修正后 dense5 版 |
| --- | --- | --- | --- |
| 模型入口 | `stage2_to_5d_geo_critic_from_5a_guarded` | `stage4_asym_dense_5` | `stage4_asym_dense_5` |
| warm start 源 | `5A` actor | `5D best` actor | `5D best` full model 优先 |
| load 方式 | actor-only，critic 改结构后重建 | actor-only，critic 重建 | 默认先整套加载，只有 critic shape 不匹配才退回 actor-only |
| local critic | `True` | `False` | `True` |
| geometry only | `True` | `False` | `True` |
| reward mode | `average` | `average_plus_interaction` | `average` |
| reward self weight | `0.8` | `0.85` | `0.8` |
| local navigation reward | `False` | `True` | `False` |
| eval episodes | `40` | `60` | `40` |
| expl noise / min | `0.025 / 0.012` | `0.018 / 0.006` | `0.025 / 0.012` |
| actor lr | `1e-6` | `1e-6` | `1e-6` |
| critic lr | `8e-5` | `2e-5` | `8e-5` |
| actor update delay | `20000` | `18000` | `20000` |
| resume latest checkpoint | 默认会续 | 默认会续 | `dense5` 启动脚本里默认关掉 |

### 当前判断

- 原始 `5D` 的成功，不能简单理解成“5D actor 很万能”，它其实也是在一套特定 critic 和 reward 配方下训出来的。
- 刚才那次失败，更像是“换了训练问题定义后，又只继承了 actor，没有公平继承原始 5D 的训练结构”。
- 所以在重新开跑之前，不能先下结论说“拟合较强的 actor 不适合 warm start”。

### 接下来怎么看

- 如果后面用“修正后 dense5 版”继续训，还是明显差于原始 `5D`，那时再认真讨论：
  - 是不是 `5D` actor 本身过拟合原分布；
  - 还是从 `5D` 直接跳到 `dense5` 的分布跨度太大；
  - 还是 dense5 课程本身就不够顺。
