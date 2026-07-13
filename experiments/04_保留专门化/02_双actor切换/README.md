# 02 双actor切换

这一步现在定位成一个中间验证，不再当最终主线。

- 普通 actor
- 密集 actor
- 简单切换规则

第一版先追求清楚可跑，不追求花哨。

后面这里主要放：

- 选用的两个 actor
- 切换指标
- 切换规则
- 和单一 actor 的对比结果

一句话：

先验证“能力偏向存在”之后，直接做粗粒度策略切换够不够。

## 第一轮最小版本

- 普通 actor：
  - `5A`
- 密集 actor：
  - `PAIR`
- 切换规则：
  - 默认用 `5A`
  - 当前方可见邻居足够近时，切到 `PAIR`
  - 邻居重新变远后，再切回 `5A`

## 当前简单记录

- `2026-07-12`：已在测试脚本里补上最小双 actor 切换骨架，先做推理阶段切换，不动训练。
- `2026-07-12`：`DUAL -> 标准五车` 跑完，能跑通，但暂时没有超过 `5A`，说明切换规则还需要继续调。

## 第一轮结果

### `DUAL -> 标准五车`

- 120 episodes
- `success_rate=0.882`
- `collision_rate=0.102`
- `unresolved_rate=0.018`
- `full_success_rate=0.558`
- `timeout_episode_rate=0.092`
- `mean_dense_action_share=0.124`

### 当前一句话判断

- 这版最小双 actor 没有崩
- 但在标准五车上还没有超过 `5A`
- 需要结合 `DUAL -> dense` 一起判断

### `DUAL -> stage3_asym_three_5`

- 120 episodes
- `success_rate=0.867`
- `collision_rate=0.132`
- `unresolved_rate=0.002`
- `full_success_rate=0.550`
- `timeout_episode_rate=0.008`
- `mean_dense_action_share=0.392`

### 第一轮总判断

- 这版最小双 actor 能跑通
- 但无论在标准五车还是 dense，都没有超过当前单 actor 最优基线
- 所以现在不能说“双 actor 已经成功”
- 更准确地说：
  - 能力偏向确实存在
  - 但“整策略硬切换”太粗，还没有把两个 actor 的能力真正用好

## 现在的第二轮主线

- 第一轮 `5A + PAIR` 先不继续细调
- 第二轮先切到：
  - 普通 actor：`5A`
  - dense actor：`5D`
- 原因：
  - `5D` 目前是正式 dense test 最稳的 actor
  - `PAIR(from_5d)` 训练内更顺，但正式 `stage3_asym_three_5` test 没超过 `5D`

这一轮要先回答两个问题：

1. 两个 actor 到底有没有足够互补性
2. 如果有，最小门控能不能先把这种互补性用出来

## 脚本现状

`TD3/test_velodyne_td3_multi.py` 现在支持三种模式：

- `single`
  - 单 actor 测试
- `hard_switch`
  - 基于最近邻距离和可见邻居数的硬切换
- `case_oracle`
  - 按 case 名直接指定用 `standard` 或 `dense`
  - 这是验证上界用的，不是最终方法

对应环境变量：

- `DRL_MULTI_STANDARD_ACTOR_FILE`
- `DRL_MULTI_DENSE_ACTOR_FILE`
- `DRL_MULTI_ACTOR_SELECTION_MODE`
- `DRL_MULTI_CASE_ORACLE_MAP`（只在 `case_oracle` 下需要）

辅助工具：

- `build_case_oracle_map.py`
  - 输入两份 test 日志
  - 自动比较各 case 的 `full_success / success / collision`
  - 输出 `case_oracle` JSON

## 第二轮准备怎么做

顺序固定：

1. `5A + 5D` 跑一版 `hard_switch`
   - `standard_5`
   - `stage3_asym_three_5`
2. 统计 `5A` 和 `5D` 在 `stage3_asym_three_5` 的 case 级优劣
3. 按 case 做一版 `case_oracle`
   - 先看理论上界有没有明显提升空间
4. 只有 oracle 确认“两个 actor 确实互补”后，再继续做 learned gate

## 当前一句话

- 这一步不再证明“两个 actor 能不能切”
- 这一步要证明的是：
  - `5A` 和 `5D` 有没有可利用的互补性
  - 如果有，简单门控能不能先把它转成实际收益
