# D4 Edge-1 Conservative Residual V2

状态：`rejected candidate`。Q 归一化与 base-action anchor 成功阻止 residual 饱和，但没有带来可靠的 edge-1 成功率提升。

## 协议

- 复用 v1 epoch 1 的 residual Actor 与已预热 Critic。
- train/validation 与 v1 完全相同：512/423 个 edge-1 场景。
- residual scale `0.10`，Actor Q normalization alpha `1.0`。
- 冻结 5D 动作 anchor weight `2.5`。
- 训练 40139 agent samples 后运行完整 423 场 validation。

## 结果

| Model | Agent success | Collision | Unresolved | Full success | Timeout | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 epoch 1 frozen baseline | `1785/2115 = 0.8440` | `327/2115 = 0.1546` | `3/2115 = 0.0014` | `217/423 = 0.5130` | `3/423 = 0.0071` | 36.835 |
| conservative v2 | `1779/2115 = 0.8411` | `334/2115 = 0.1579` | `2/2115 = 0.0009` | `219/423 = 0.5177` | `2/423 = 0.0047` | 34.882 |
| historical 5D paired-pool baseline | `1794/2115 = 0.8482` | `319/2115 = 0.1508` | `2/2115 = 0.0009` | `222/423 = 0.5248` | - | - |

v2 相比现场 baseline 只多 2 个 full-success episode，同时少成功 6 个 agent、增加 7 次碰撞；相对历史 5D 仍全面更差，且远低于预先规定的 `0.60` full-success 门槛。因此不启动更多 seed 或 epoch。

## 机制检查

40139 个 replay state 上：

- residual 均值 `[+0.0209, -0.0107]`；
- residual 标准差 `[0.0049, 0.0042]`；
- 最大绝对值 `[0.0422, 0.0332]`；
- `abs(residual) > 0.095` 的边界饱和比例为 `0`。

说明 conservative objective 解决了 v1 的恒定边界饱和，但学到的仍主要是小幅全局偏置，而非足以改善交互决策的状态相关行为。剩余瓶颈应转向交互观测和监督信号：当前 Actor 缺少邻居相对速度/TTC/优先级，`0.8/0.2` reward 也没有明确提供让行目标。继续调整 residual 容量、anchor 或 epoch 缺乏证据。

归档保留 best Actor/Critic、完整日志、evaluation 数组和 TensorBoard event；大型 replay checkpoint 不纳入 Git。哈希见 `summary.json`。
