# R2-10k 与 B2 的 D2 预比较结果

状态：`complete / preliminary only`。评测已完成 `200/200` 场；未读取 sealed test。

## 结果

两种方法使用完全相同的 G11-D2 manifest 和场景顺序：`0-edge` 100 场、`edge-1` 100
场。B2 使用已归档结果，R2-10k 是本轮重新评测的冻结大 Actor。

| 方法 | agent success | collision | unresolved | full success | timeout | 平均步数 |
|---|---:|---:|---:|---:|---:|---:|
| B2 Gate | 0.932 | 0.063 | 0.005 | 0.745 | 0.025 | 45.24 |
| R2-10k 单 Actor | 0.940 | 0.057 | 0.003 | 0.785 | 0.015 | 21.02 |

逐场 full success 配对结果为：R2-10k 改善 25 场，B2 改善 17 场，持平 158 场，
McNemar exact `p=0.280`。因此当前结果不能写成 R2-10k 显著优于 B2。

分层结果：

| 冲突边 | 场数 | B2 full success | R2-10k full success |
|---|---:|---:|---:|
| 0 | 100 | 0.850 | 0.920 |
| 1 | 100 | 0.640 | 0.650 |

## 解释边界

R2-10k 在这份简单准入集上表现更快，主要因为它是单 Actor，始终使用普通导航策略，
不存在 B2 约 `0.776` 的 interaction Actor action share 和由此带来的保守等待。该结果
反过来说明 B2 的 Gate 代价需要在复杂场景中验证：只有当 Gate 在 multi-edge/dense
场景减少冲突的收益超过切换和保守行为的代价时，双 Actor 方法才有成立的证据。

本结果不回答：

- R2-10k 是否能处理多冲突场景；
- B2 是否在 dense 场景中比单 Actor 更有优势；
- 可部署 Gate 是否优于 `2.0 m` 真值 Oracle。

下一步是在同一份冻结 dense manifest 上重新运行 R2-10k、B2、5A、epoch-16 和 Oracle，
先做小规模 dense pilot，再做完整 dense validation。

原始结果：

- `local_data/r2_vs_b2_d2_prelim/summary.json`
- `local_data/r2_vs_b2_d2_prelim/results/g12_r2_10k_d2_r1_s20260809.npy`
- `../../../../logs/archive/validation/g12_r2_vs_b2_d2_prelim/`
