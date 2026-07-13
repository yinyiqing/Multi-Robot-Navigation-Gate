# 03 门控与局部增强

这一步研究如何保留已有 actor，并只在局部强交互状态下引入额外能力。

## 当前状态

- learned gate 尚未实现。
- Attention 和联合动作 Critic 的通用代码能力仍保留，但当前没有有效的主线实验。
- `5A + 5D` 的 hard switch 和 case oracle 已证明二者互补性不足。
- `PAIR(from_5d)` 已完成正式测试，但 dense 表现没有超过 `5D`。

## 进入门控训练前的条件

1. 找到在 `standard_5` 和 `stage3_asym_three_5` 上有明确分工的两个专家。
2. 用相同测试集完成单模型、hard switch 和 oracle 对照。
3. Oracle 必须显示组合上界高于两个单模型，才值得训练 learned gate。
4. 两个专家保持冻结，只训练门控网络。

Attention 只作为门控有效后的可选特征增强，不作为当前成败点。
