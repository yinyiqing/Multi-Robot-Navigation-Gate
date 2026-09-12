# G32 / B8：冻结 B2 的独立 Reward preference head

状态：离线训练完成，准备 32 场 development pilot。

B8 完全冻结 B2 的 GRU 与 interaction head，只训练一个独立 reward preference head。部署时保留
B2 原始 interaction probability；只有概率处于 `[0.40, 0.60]` 且 reward head 置信度达到 `0.25`
时，才用 reward preference 做 tie-break。两个 Actor、G0/G1 和 B2 均冻结。

离线检查：251 个 reward 锚点中 110 个超过 `|Delta r| > 0.005` 噪声阈值；验证可靠样本准确率约
90%；冻结主 phase 输出与 B2 最大差异为 0。

先运行固定 32 场 pilot；只有成功/碰撞方向和 Interaction 选择比例没有明显退化，才扩展到完整
256 场 development。pilot 不用于最终显著性结论。
