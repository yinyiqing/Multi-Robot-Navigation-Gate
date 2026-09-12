# G33 双头联合监督预检

更新时间：2026-09-12

状态：补采集与质量审计已完成；数量闸门未通过，未启动训练或闭环长跑。

## 目的

把导师的要求落实为可检查的定义：每个 Gate 样本同时具有交互标签 `y` 和同状态双 Actor
一步 reward 差 `Delta r = r_I - r_N`，训练时由二者共同监督，部署时由本机观测预测二者，
再用预先固定的联合规则选择 Actor。

## 预检结果

预检脚本：`scripts/preflight_reward_aware_joint_gate.py`

输入为 G27-P1 已审计的反事实数据：

- train：251 个 reward anchors，对应约 70,981 个 Gate 训练帧，覆盖率约 0.35%；
- validation：57 个 reward anchors，对应约 4,785 个 Gate 验证帧，覆盖率约 1.19%；
- 可靠阈值 `|Delta r| > 0.005` 时，train/validation 中 `Delta r` 与 `y` 的方向一致率均约 45%；
- train 中 48 个样本满足 `|Delta r| > 1`，5 个样本满足 `|Delta r| > 10`；validation 中分别为 9 和 2 个。

完整机器可读报告位于 `local_data/preflight_report.json`。

## 2026-09-12 补采集结果

按八个距离×动作差异分层重新选择并采集了 128 个训练候选和 64 个验证候选。fresh-process
采集和四分支同状态对齐均正常；最终得到：

- train：`62` 个 usable anchors，八个分层均至少 6 个；
- validation：`37` 个 usable anchors，八个分层均至少 2 个；
- train/validation 的 usable scene ID 无交集；
- 每个 usable anchor 都保留 `r_N`、`r_I`、`Delta r`、reward components 和终止信息。

质量审计报告位于 `local_data/counterfactual/audit/summary.json`，结果集位于同目录的
`train.npz` 和 `validation.npz`。

## 结论和停止线

当前数据质量已经通过首轮验收，但数量仍不能支持“每一个 Gate 样本都有 reward-aware 监督”的联合训练，也不能把 reward 符号
直接当成交互标签替代物。此前 B3/B4 等版本的闭环退化与这一数据/目标不匹配相符。

在补齐同状态双 Actor 一步 reward、明确 reward 终止项处理并冻结唯一联合决策规则前：

- 不启动 G29 或任何新的 256 场闭环队列；
- 不根据旧 B2 的结果反复调 reward 权重或阈值；
- 不把当前 G29 写成已验证的导师方法。

本批次没有自动进入训练。若继续补采集，应先把目标有效样本数和终止 reward 的处理规则写入
新的协议，再启动下一批；不能仅凭本批次的分层覆盖就宣布可以跑 256 场。
