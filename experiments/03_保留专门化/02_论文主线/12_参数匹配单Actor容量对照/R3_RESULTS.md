# G12-R3完整五车混合训练结果

状态：`completed / failed after Actor unlock / R2-10k remains frozen reference`。
日期：`2026-08-08`。

## 运行

R3按预注册的`standard/strong/dense/strong`四槽循环完成`40,014 agent samples`。
R2-10k只加载Actor，Critic重新初始化；Actor在`21,000` samples后解冻。训练清单和
validation均未读取G11-D2、G11-E或sealed test。

日志归档于：
`logs/archive/training/g12_r3/40k_pilot/`。

## 固定validation结果

| checkpoint | agent success | full success | collision | unresolved | timeout | avg env steps |
|---|---:|---:|---:|---:|---:|---:|
| R3-20k，Actor冻结 | `0.892` | `0.667` | `0.103` | `0.005` | `0.025` | `26.3` |
| R3-40k，Actor已更新 | `0.842` | `0.575` | `0.157` | `0.002` | `0.008` | `24.4` |
| R2-10k冻结参考 | `0.900` | `0.700` | `0.100` | `0` | `0` | `20.39` |
| 5A同场基线 | `0.810` | `0.5583` | `0.190` | `0` | `0` | `20.08` |

## 结论

1. R3-20k仍保留了R2大Actor的能力；R3-40k在Actor解冻后full success下降
   `0.092`，agent success下降`0.050`，collision上升`0.054`。
2. R3-40k高于5A约`0.017`个百分点，但没有达到R3预注册的“总体不低于R2参考”条件，
   不能作为最终大Actor性能模型。
3. 本次没有NaN或进程异常，退化发生在正常Actor更新阶段。因此结果继续支持：主要问题
   是fresh-Critic上的Actor更新稳定性和分布覆盖，而不是简单的参数容量不足。
4. 不启动R4-320k，不扫描R3超参数。后续比较使用R2-10k作为大Actor候选；R3-20k只作
   训练稳定性边界诊断，R3-40k只作失败对照。

