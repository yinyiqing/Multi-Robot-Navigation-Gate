# G20 同场主线裁决

状态：`registered / running`。登记日期：`2026-08-17`。

## 唯一问题

在训练充分的原宽N5出现后，旧F-A1相对旧5A的收益是否仍能支持双Actor路线；若不能，
N5与冻结epoch-17之间是否至少存在可被新Router利用的特权上界。

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

## 决策规则

1. 先报告两个repeat合并后的overall及0/1/multi分层结果和逐场配对检验。
2. 若`n5_recovery`相对`n5`没有正的full-success配对趋势，或收益完全由timeout/效率恶化
   换取，则停止双Actor性能主线，不训练新Gate。
3. 若`n5_recovery`稳定优于`n5`，则冻结N5和epoch-17，下一步只训练N5专用时序Router；
   F-A1保留为旧5A路由对照，R2保留为容量对照。
4. `F-A1`是否低于N5必须如实报告，不再通过选择旧基线或训练较差大Actor规避。

## 产物

- 实时日志：`logs/active/g20-overnight-final-unified/runner.log`；
- 逐方法日志：同一目录下`g20_<policy>_s<seed>_attempt<N>.log`；
- 结果：`local_data/results/`；
- 完成后日志归档到`logs/archive/validation/g20_overnight_final_unified/`。
