# 03 门控注意力增强

本目录暂停。

原因很简单：现在还没有一个真正优于 `5D` 的 dense 专家 actor。没有第二个可靠专家，gate / attention 的故事会变成“保护 5D”，而不是“在普通和 dense 能力之间学习选择”。

## 已完成的检查

- `5A + 5D` hard switch：没有超过单独 `5D`。
- case-level oracle：互补不足，多数仍选 `5D`。
- frozen `5D` + attention residual：没有崩，但不是 dense actor；residual 基本没有打开。

## 保留结论

attention residual 的失败不能证明 attention 没用，只说明这条实现没有解决当前目标：

- best 出现在 actor residual 真正训练前。
- gate 和 residual 的有效动作修正接近 0。
- bridge 最难的是 `center_weave` 和 `offset_cross` 两个 case。

这些结论已经合并进上级 README。后续除非已经训练出更好的 dense actor，否则本目录不继续新增实验。
