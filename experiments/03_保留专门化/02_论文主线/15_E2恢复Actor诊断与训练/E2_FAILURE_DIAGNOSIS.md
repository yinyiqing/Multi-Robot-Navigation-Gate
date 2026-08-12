# E2失败窗口诊断

状态：`completed / first-pass log diagnosis`.

输入：

- manifest：`g12_r2_curriculum_v1/n5/validation.json.gz`
- E2单独日志：
  `logs/archive/training/current_generalist_r2style/n5_efficiency_e2_admission/current_generalist_n5_efficiency_e2_admission_s20260817.log`
- E2 + 旧epoch-16 oracle日志：
  `logs/archive/training/current_generalist_r2style/e2_oracle_epoch16_admission/current_generalist_e2_oracle_epoch16_admission_s20260817.log`

输出：

- `local_data/e2_failure_diagnosis/diagnosis_summary.json`
- `local_data/e2_failure_diagnosis/paired_cases.csv`

## 总体结果

| 方法 | full success | agent success | collision | unresolved | timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2单独 | `0.7500` | `0.9267` | `0.0567` | `0.0167` | `0.0667` | `49.28` |
| E2 + 旧epoch-16 2m oracle | `0.7250` | `0.9050` | `0.0800` | `0.0150` | `0.0667` | `50.56` |

结论：旧 `epoch-16` 的2m oracle接入不能作为新E2组合的上界。它不是没有作用，而是
同时带来明显误切。

## E2失败构成

E2共有`30/120`个非full-success case。

按失败类型：

| 类型 | case数 |
| --- | ---: |
| collision | `22` |
| timeout+unresolved | `6` |
| collision+timeout+unresolved | `2` |

按场景池：

| pool | case数 |
| --- | ---: |
| dense | `22` |
| standard | `8` |

按冲突拓扑：

| topology | case数 |
| --- | ---: |
| multi | `20` |
| edge1 | `9` |
| zero | `1` |

这说明E2的主要剩余短板仍与交互有关，尤其是dense multi-edge；但它不是单纯
“近车就不行”，因为edge1和zero上总体已经很强。

## 分层性能

按pool：

| pool | full success | collision | timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| standard | `0.8667` | `0.0200` | `0.0500` | `40.37` |
| dense | `0.6333` | `0.0933` | `0.0833` | `58.20` |

按topology：

| topology | full success | collision | unresolved | timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | `0.9750` | `0.0000` | `0.0050` | `0.0250` | `26.65` |
| edge1 | `0.7750` | `0.0600` | `0.0050` | `0.0250` | `34.90` |
| multi | `0.5000` | `0.1100` | `0.0400` | `0.1500` | `86.30` |

真正需要Actor I介入的是multi-edge恢复，不是所有2m近车状态。

## 旧epoch-16 oracle的作用

逐场比较：

| relation | case数 |
| --- | ---: |
| oracle_improved | `11` |
| same | `95` |
| oracle_degraded | `14` |

`oracle_improved` 全部来自E2失败case，说明局部接管确实存在潜在收益。

`oracle_degraded` 全部来自E2本来成功的case，说明当前2m规则误切会破坏E2已经能处理的
状态。

典型改善：

| episode | case | topology | E2失败 | oracle结果 |
| ---: | --- | --- | --- | --- |
| 1 | `standard-20260717006453-310e3af54445` | multi | timeout+unresolved, 300步 | success, 84步 |
| 8 | `dense-20260718009166-dc896a5fcf95` | multi | collision, 258步 | success, 58步 |
| 48 | `dense-20260718008373-f0de0a37adec` | multi | timeout+unresolved, 300步 | success, 75步 |
| 81 | `dense-20260718009349-6b75be3931d6` | multi | timeout+unresolved, 300步 | success, 90步 |
| 117 | `dense-20260718009317-db62e21fe167` | multi | collision, 100步 | success, 42步 |

典型退化：

| episode | case | topology | E2结果 | oracle退化 |
| ---: | --- | --- | --- | --- |
| 5 | `dense-20260718008477-2166c0b8d4ec` | multi | success, 28步 | collision |
| 44 | `dense-20260718009006-9ad8097a40ac` | multi | success, 28步 | collision+timeout+unresolved |
| 71 | `dense-20260718008956-23b63793de30` | edge1 | success, 29步 | collision |
| 104 | `dense-20260718009342-e413bb6628e2` | multi | success, 31步 | collision |

## 对训练路线的含义

1. `2.0 m`不能再作为充分切换条件。
2. `2.0 m`仍可作为候选窗口筛选条件，因为旧oracle确实救回`11`个case。
3. 新Actor I的训练目标应聚焦E2失败窗口，尤其是：
   - dense multi-edge；
   - timeout/unresolved恢复；
   - 长步数collision或碰撞前纠偏；
   - E2低进度但尚未失败的近失败窗口。
4. 需要负样本：E2本来成功但旧oracle会误切退化的`14`个case，必须用于训练Gate或
   oracle规则的抑制项。
5. 动态reward和邻域critic应只放在Actor I训练侧，并且必须包含恢复推进项；只惩罚近车
   会复现保守等待或误切问题。

## 下一步实验建议

先做小规模 `recovery-oracle`，不直接训练可部署Gate：

```text
候选进入：
  near_robot_or_dense_geometry
  AND low_progress_or_stagnation
  AND not_near_goal_success

候选退出：
  progress_recovered OR nearest_robot_risk_reduced OR max_hold_reached
```

准入：

- recovery-oracle必须超过E2单独；
- collision不得高于E2；
- timeout不得高于E2；
- 改善case必须多于退化case；
- 如果oracle不成立，不启动Gate训练。
