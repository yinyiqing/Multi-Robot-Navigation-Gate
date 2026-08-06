# G12-R2-S1固定困难Case诊断结果

状态：`completed / repair required`。日期：`2026-08-06`。

## 1. 运行完整性

本次有效运行使用冻结S0 epoch 2 best，仅评测、不更新网络。四个stage共42个固定case，
每case重复3次，总计`126/126`个episode；分析器复算结果与运行时`summary.json`逐字节一致。

```text
Actor SHA-256: 7cb61925a4188e638859f88d38288e0431e5f05be489fa6107a77c7efaed3822
evaluation seed: 20260812
summary SHA-256: ac0c1c1d31893044bf9831e49800e0a43b3b0c34861bf92f9cf29202884e1413
process integrity: passed
```

阶段间ROS/Gazebo均先清理再重启，最终进程与`14621/14721`端口全部释放。有效日志位于
`logs/archive/validation/g12_r2_s1_diagnostic/`。首次launcher清理失败的运行仍单独保存在
`logs/archive/diagnostic/g12_r2_s1_launcher_cleanup_failed_20260806/`，不得混用。

## 2. 汇总结果

| stage | episodes | full success | collision | timeout | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| `stage1_single` | `18` | `16/18 = 0.8889` | `2` | `1` | repair |
| `stage1e_single_rescue` | `36` | `22/36 = 0.6111` | `11` | `3` | repair |
| `stage1f_wall_parallel_rescue` | `36` | `19/36 = 0.5278` | `8` | `9` | repair |
| `stage1g_collision_guard` | `36` | `15/36 = 0.4167` | `11` | `10` | repair |
| **合计** | **126** | **72/126 = 0.5714** | **32** | **23** | **repair** |

42个stage-case项中，`22`项为`3/3 pass`，`0`项为`2/3 borderline`，`20`项为
`0-1/3 repair`。总成功率只作困难case诊断，不是broad validation或论文性能指标。

## 3. 缺口结构

基础目标捕获、近目标切向接近、偏置目标接近和隔墙导航大多为`3/3`。失败集中于：

- `near_obstacle_recovery`：`1/3`，两次碰撞；
- 贴墙平行直行与朝墙转向：跨stage反复为`0-1/3`，主要碰撞；
- 离墙转向和保持安全间距继续推进：跨stage反复为`0-1/3`，碰撞与timeout并存；
- `wall_parallel_reverse_clear`：跨stage反复为`0-1/3`，主要碰撞。

几何去重审计表明，20个repair stage-case项对应8个独立困难几何；跨stage同名case的
起点、目标和朝向一致，主要差别是历史训练权重。累计结果为：

| 独立case | success / evaluations |
| --- | ---: |
| `near_obstacle_recovery` | `1/3` |
| `wall_parallel_north_clear_straight` | `1/9` |
| `wall_parallel_north_clear_yaw_in` | `1/9` |
| `wall_parallel_north_clear_yaw_out` | `1/9` |
| `wall_parallel_north_safe_straight` | `1/9` |
| `wall_parallel_north_safe_yaw_in` | `0/6` |
| `wall_parallel_north_safe_yaw_out` | `0/6` |
| `wall_parallel_reverse_clear` | `1/9` |

S1训练清单应保留这8个唯一几何及其来源记录，再按“近障恢复、贴墙姿态、离墙推进、
反向脱离”分组平衡采样，不能把同一几何复制三份后隐式放大权重。

## 4. 决策

诊断满足预注册的S1启动条件，但不授权直接消耗全部`80k`预算。下一步先冻结repair-only
manifest、S0 Actor/Critic完整warm start、分段预算和broad n1回归门槛。每段更新后必须：

1. 重测对应repair case；
2. 重测S0 broad n1 validation；
3. 若broad能力遗忘则回滚，不以targeted峰值替代普通导航准入。
