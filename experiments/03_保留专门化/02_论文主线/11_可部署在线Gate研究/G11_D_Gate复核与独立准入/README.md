# G11-D Gate复核与独立准入

状态：`D1 registered / not started`。登记日期：`2026-08-05`。

G11-C只证明B2的student-rollout聚合值得保留，没有证明训练稳定性或最终闭环准入。
本阶段保持两个Actor、detector、训练数据、模型结构和主seed checkpoint全部冻结。

## D1：B2训练seed复核

- 数据仍为A1冻结5A轨迹640场与B1 student轨迹640场；
- validation仍为A1内部120场，不读取导航test或sealed test；
- 主seed固定为`20260804`，复核seed预注册为`20260805-20260808`；
- 只训练8帧T1 Gate，CPU、2线程、`nice -n 15`，不启动Gazebo、不使用GPU；
- 不从复核seed中挑选新的主模型，也不根据复核结果修改阈值策略。

D1通过条件：4个复核seed中至少3个同时满足冻结overall/weak FPR上限，并且相对A1主seed
的F1下降不超过`0.03`、positive interval IoU下降不超过`0.04`。失败则先检查训练不稳定，
不得直接扩大闭环实验。通过只说明聚合训练可复现，不是导航成绩。

运行入口：

```bash
bash scripts/run_g11_d_seed_replication.sh
```

活动日志写入`logs/active/g11_d/`，完整结束后归档到
`logs/archive/diagnostic/g11_d/`。汇总写入本目录`local_data/seed_replication_summary.json`。

## D2：独立闭环准入

D1通过后才启动。D2必须先构建并冻结一套来自导航validation、与A1内部120场和G11-C
完全互斥的0-edge/edge-1 manifest。不得复用G11-C调参，不读取导航test。

至少比较：

| 方法 | 作用 |
| --- | --- |
| 5A always-on | 普通导航基线 |
| epoch-16 always-on | 条件Actor全程使用边界 |
| min-LiDAR/TTC规则Gate | 非学习可部署下界 |
| A1 Gate | 无student聚合对照 |
| B2主seed Gate | 当前候选 |
| 2.0 m oracle | 不可部署上界 |

主指标为full success，并报告agent success、collision、unresolved、timeout、平均步数、
interaction Actor占比、切换次数、同场配对检验和oracle收益恢复率。准入同时要求：

1. 0-edge full success相对5A下降不超过3个百分点；
2. edge-1上B2的改善明确多于退化，并显著超过5A；
3. timeout不系统增加；
4. B2明确超过规则Gate；
5. full success达到`0.45`或恢复至少60%的同场oracle收益；
6. 不能以更高full success掩盖interaction Actor长期占用和明显效率退化。

D2协议、manifest哈希、运行seed、重复次数和停止条件必须在启动Gazebo前补齐并提交。
通过D2后才评估multi-edge；若multi-edge不足，可在读取sealed test前登记训练数据修订，
但必须取消single-to-multi零样本泛化表述。
