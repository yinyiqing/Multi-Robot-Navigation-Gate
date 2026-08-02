# 局部专家全程化：Actor B工作线

状态：`epoch-16完整状态分叉pilot呈正向趋势，等待200场复核；离线蒸馏与D2b fresh Critic已拒绝`。

## 1. 目标

当前方法固定为：

```text
Actor A：冻结5A，负责普通/弱交互导航
Actor B：从epoch-16局部避障专家继续训练，恢复全程导航能力
Gate：仅在A、B都能独立导航且存在互补性后训练
```

Actor B必须从起点独立运行到终点。`5A + epoch-16`的`2.0 m`真值切换只作为教师和不可部署上界，不作为最终方法。

## 2. 已有证据

- 固定140场单冲突验证：5A与oracle组合full success为`0.421/0.700`。
- 完整1000场dense validation：5A与oracle组合full success为`0.309/0.545`。
- epoch-16独立运行256场：agent success `0.707`、full success `0.231`、timeout episode rate `0.516`。
- epoch-16完整状态分叉pilot的固定50场：冻结基线full success/timeout为`0.300/0.380`，短程全状态更新后为`0.420/0.340`；collision从`0.144`降至`0.124`。
- 因此epoch-16已经具有真实局部避碰能力，缺失的是正常区推进和交互后重新启动能力。
- 历史独立Dense Actor全部从5A初始化；`epoch-16 -> 全程Actor B`此前没有被直接训练和验证。

## 3. 训练原则

第一候选保持原24维Actor和基础TD3，不增加新网络模块：

1. 从冻结epoch-16 Actor warm-start，训练期间由Actor B全程控制。
2. 使用dense train轨迹恢复目标推进、交互后重启和连续多冲突处理。
3. 正常状态以冻结5A动作作为能力恢复参考；交互状态以初始epoch-16动作作为能力保持参考。
4. 两类参考只进入训练loss，不参与执行；验证时禁止oracle接管。
5. fresh Critic必须通过同状态反事实校准；复用原专家完整训练状态时，先做冻结Actor基线，并使用gradient guard限制退化更新。
6. 先跑短pilot，不直接长跑或扫大量超参数。

这不是训练第三个“折中Actor”。验收要求Actor B在完整episode中同时保留避碰和恢复推进能力。

## 4. 准入与停止线

短pilot使用固定50场dense monitor观察趋势，并使用固定200场做候选复核。

必须同时满足：

- 相对epoch-16独立运行，timeout显著下降，不能只把停车变成碰撞；
- 相对5A，dense full success有正向趋势，collision不增加；
- 单冲突strong validation不丢失epoch-16相对5A的主要收益；
- 0-edge/standard能力没有不可接受退化；
- Actor动作变化不是全局恒定加速、减速或单向转弯偏置。

出现以下任一情况立即停止当前配置：连续两次验证full success下降超过`0.10`、timeout超过`0.30`、危险状态系统性加速、或Critic反事实排序再次与真实N-step回报相反。

当前epoch 18候选虽然趋势为正，但`timeout=0.340`仍未过线。因此冻结候选并扩大复核，不直接增加训练轮数。

## 5. Gate进入条件

只有Actor B独立通过后才进入Gate：

1. 在同一validation上冻结A/B，记录逐场`A-only/B-only/both/neither`。
2. 若B全面优于A，直接使用B，不训练Gate。
3. 若`A-only`和`B-only`都稳定存在，再用可部署本机感知训练Gate。
4. Gate学习的是“当前状态下哪位完整策略的预期闭环结果更好”，不是复现`2.0 m`距离阈值，也不是识别standard/dense场景名。

## 6. 两周执行顺序

- 第1-2天：已完成Critic校准和epoch-16完整状态分叉短pilot。
- 第3-5天：对epoch 17/18做固定200场同场复核，并检查strong-interaction能力保持。
- 第6天：A/B互补性审计，决定是否进入Gate。
- 第7-9天：训练Gate或在B全面占优时转为单Actor方法。
- 第10-12天：冲突拓扑`edge/max_degree/simultaneous` OOD评估与必要消融。
- 第13-14天：统计、图表和论文写作；sealed test只在模型与阈值冻结后运行一次。

## 7. 论文主张

双Actor与Gate本身不是创新点。保留的论文贡献候选是：从局部交互专家中保留避碰能力，同时恢复完整导航能力的策略整合方法，以及按冲突拓扑而非仅按车数/空间密度验证多车组合泛化。若Actor B或拓扑泛化证据不成立，应收缩主张，不能只靠架构名称投稿。

离线双教师可行性筛选见 [01_双教师离线蒸馏pilot](01_双教师离线蒸馏pilot/README.md)。

D2b Critic校准见 [02_D2b_Critic同状态校准](02_D2b_Critic同状态校准/README.md)，当前在线实验见 [03_epoch16完整训练状态分叉pilot](03_epoch16完整训练状态分叉pilot/README.md)。
