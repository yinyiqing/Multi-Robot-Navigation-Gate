# 保留与专门化

状态：`current research branch`。唯一当前协议见[论文主线](02_论文主线/README.md)。

## 当前问题

普通导航Actor已经形成，但直接在dense任务上继续更新会覆盖已有能力。当前采用状态级
职责分工：

```text
Actor N = 冻结5A，普通导航
Actor I = 冻结epoch-16，局部机器人避让
Gate    = 本机观测驱动的在线进入/退出选择器
```

Actor I不是完整Dense Actor，不要求全程独立导航。`2.0 m`真值距离只定义训练分工和
oracle上界；最终Gate必须只使用本机传感器。

## 目录

| 目录 | 状态 | 内容 |
| --- | --- | --- |
| [`01_历史诊断/`](01_历史诊断/README.md) | historical / rejected | 覆盖训练、旧双Actor和旧dense expert失败原因 |
| [`02_论文主线/`](02_论文主线/README.md) | current | 冻结双Actor、Gate、数据边界和准入 |
| [`90_未启用方案/`](90_未启用方案/README.md) | inactive | 未启用安全兜底，不与主线并行 |

## 已关闭

- 完整Dense Actor与继续扫描reward/Critic；
- epoch-16整网全程续训；
- 24维单帧Residual；
- pair、controlled-ego和复杂Critic支线；
- corrected edge-1完整Actor继续训练。

失败证据仍可引用，但后续不得从历史README的旧“下一步”恢复这些路线。完整状态见
[实验注册表](../EXPERIMENT_REGISTRY.md)。
