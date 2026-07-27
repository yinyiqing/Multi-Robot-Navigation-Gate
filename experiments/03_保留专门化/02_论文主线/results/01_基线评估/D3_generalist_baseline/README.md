# D3 Generalist Baseline

状态：`standard-5 subset complete`。D3 整体尚未完成；procedural low/medium/high 与 canonical held-out baseline 必须在 D1-D2 后补齐。

## eval-5d-standard / seed 1000

| 项目 | 值 |
| --- | --- |
| 模型 ID | `generalist-5d` |
| 场景 | `standard-5` |
| Actor mode | `full` |
| 机器人 | 5 |
| test seed | `1000` |
| episodes | `1000` |
| metric | collision-priority mutually exclusive outcomes |

## 结果

| 指标 | 计数 | 比例 | episode bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| agent success | `4408 / 5000` | `0.8816` | `[0.8718, 0.8912]` |
| collision | `495 / 5000` | `0.0990` | `[0.0898, 0.1084]` |
| unresolved | `97 / 5000` | `0.0194` | `[0.0156, 0.0234]` |
| full success | `569 / 1000` | `0.5690` | `[0.5380, 0.5990]` |
| timeout episode | `90 / 1000` | `0.0900` | `[0.0730, 0.1080]` |

一致性检查：

```text
4408 + 495 + 97 = 5000 = 5 robots * 1000 episodes
```

辅助统计：

- mean episode steps: `70.156`
- mean final distance: `0.3950 m`
- mean episode reward: `104.6508`
- success histogram for `0..5` successful robots: `[0, 4, 15, 119, 293, 569]`
- collision histogram for `0..5` collided robots: `[637, 249, 99, 12, 3, 0]`

Bootstrap 使用 20,000 次 episode-level resampling，统计 seed 为 `20260717`。agent 指标先在每个 episode 内除以 5，再对 episode 重采样。

## 文件

- `eval-5d-standard_seed1000_1000ep.log`: 完整运行日志。
- `eval-5d-standard_seed1000_1000ep.npy`: 1000 行逐 episode 数据。
- `summary.json`: 配置、统计、列定义和校验哈希。

`.npy` 列定义：

```text
0 episode_num
1 cumulative_env_steps
2 cumulative_agent_samples
3 episode_env_steps
4 episode_agent_samples
5 mean_reward
6 success_count
7 collision_count
8 full_success
9 mean_final_distance
10 unresolved_count
11 timeout_episode
12 scenario_name
```

## 复现说明

运行于 2026-07-17，启动时 Git HEAD 为 `033ae75`，工作区包含尚未提交的 outcome metric 修正；相关评测和环境实现随后固化在 `73c905c`。运行期间后续文档和 residual 元数据修改不影响已启动的 full Actor 测试进程。

因此该结果可作为当前 D3 standard baseline，但后续 paired 正式实验必须从干净提交启动，并在日志中直接记录 commit。模型权重 SHA-256 已保存在 `summary.json`。

与旧 300-episode 结果相比，full success 从约 `0.590` 变为 `0.569`，仍落在本次 95% CI 内。旧结果使用 success/collision 可重叠口径，只能进行趋势核对，不能做严格绝对比较。
