# G20 同场跨协议强基线审计

状态：`stopped / out of paper scope`。登记并停止日期：`2026-08-17`。

## 停止原因

论文范围已冻结为“不重新训练Actor的已有策略在线复用”。N5与R2-10k使用新的完整课程
和连续Critic，回答的是完整策略重训问题，不是F-A1与R2B之间的同流程容量消融。继续本
实验会再次把训练协议变化与路由结构变化混为一个比较，因此在首个N5 repeat未完成时停止。

已有5A、epoch17完整结果和N5部分结果只作内部历史证据，不进入论文方法表；本实验不得
恢复，也不得据此启动N5专用Gate。

## 唯一问题

在新课程训练出的N5/R2-10k出现后，它们与旧F-A1在完全同场评测中的真实差距是多少；
把epoch-17接到N5后，策略互补性是增加还是被N5自身的保守行为覆盖。

本实验是跨训练协议审计，不是纯参数量实验。主容量结论使用同5A流程的R2B。

## 固定方法

| ID | 定义 | 用途 |
| --- | --- | --- |
| `n5` | 原宽N5全程执行 | 训练充分单Actor基线 |
| `r2` | R2-S4 10k加宽Actor全程执行 | 同新课程容量对照 |
| `f_a1` | 旧5A + epoch-17 + 冻结F-A1 | 已有双Actor方法 |
| `n5_recovery` | N5 + epoch-17特权恢复规则 | 新Actor组合的可路由上界诊断 |

`n5_recovery`使用其他机器人真值距离和低进展状态，只是训练可见的privileged oracle，
不是可部署方法，也不进入最终方法排名。

## 固定评测

- manifest：`g12_full_scene_selection_v1/validation.json.gz`；
- SHA-256：`52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635`；
- 120场：standard/dense各60，0-edge/edge-1/multi-edge各40；
- seeds：`20260830/20260831`；
- 每种方法240 episodes，总计960 episodes；
- 单Gazebo顺序执行，场景顺序、物理步长、终止条件完全一致；第二个repeat反转方法顺序；
- 不读取sealed test，不根据运行结果换checkpoint、seed或规则阈值。

目录中早先未完成的`5a`和`epoch17`单独运行不属于本次四方裁决，不进入分析。

## 报告规则

1. 报告两个repeat合并后的overall及0/1/multi分层结果和逐场配对检验。
2. `F-A1`是否低于N5/R2-10k必须如实报告；不得将跨协议差异写成纯参数容量效应。
3. `n5_recovery`只诊断N5与epoch-17的互补性，不授权训练N5专用Gate。
4. 主表容量对照继续使用同5A起点、数据、预算和优化流程的R2B，并披露其Actor解冻后
   坍塌、best停在冻结阶段这一限制。

## 产物

- 实时日志：`logs/active/g20-overnight-final-unified/runner.log`；
- 逐方法日志：同一目录下`g20_<policy>_s<seed>_attempt<N>.log`；
- 结果：`local_data/results/`；
- 完成后日志归档到`logs/archive/validation/g20_overnight_final_unified/`。
