# 05 3D2：几何邻域 Critic

作用：从课程 3A guarded best 接到 3车 D2 组，即几何邻域 critic。

D2 的含义：

- D2 是方法名，表示几何邻域 Critic。
- 旧3D2 和课程3D2 都是 D2。
- 区别在初始化路径：旧3D2 从旧统一 baseline 来，课程3D2 从课程 3A guarded actor 来。

当前结论：

- 训练 epoch 3 的 40 集 eval full success 为 `0.925`。
- 后续 latest 明显退化，epoch 17 full success 降到 `0.075`。
- 当前正在对 epoch 3 best 做 300 集正式测试。

命名：

- 方法组：3D2
- 中文名：课程 3车几何邻域 Critic

