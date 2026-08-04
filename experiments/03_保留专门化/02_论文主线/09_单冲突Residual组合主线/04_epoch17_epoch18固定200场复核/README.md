# epoch 17/18固定200场复核

状态：`complete / rejected`。

## 目的

复核50场pilot中epoch 18相对epoch 17的full success从`0.300`提高到`0.420`是否
能够在更大的固定场景集上保持。

## 协议

- manifest：`dense_validation_monitor_v1`固定200场；
- seed：`20260803`；
- epoch 17：原epoch-16局部专家，作为冻结基线；
- epoch 18：短程全状态更新候选；
- 两者均为单Actor全程独立执行；
- 禁用5A接管、Oracle和Gate；
- 相同scenario ID、相同顺序串行运行。

当前测试输出为17列；旧比较脚本只接受13列，因此以下聚合读取定义不变的前13列
独立计算。原始逐场数组保存在`local_data/`。

## 结果

| 指标 | epoch 17 | epoch 18 | 变化 |
| --- | ---: | ---: | ---: |
| agent success | `0.749` | `0.726` | `-0.023` |
| collision | `0.111` | `0.157` | `+0.046` |
| unresolved | `0.140` | `0.117` | `-0.023` |
| full success | `61/200 = 0.305` | `52/200 = 0.260` | `-0.045` |
| timeout | `103/200 = 0.515` | `95/200 = 0.475` | `-0.040` |
| mean steps | `189.28` | `176.86` | `-12.42` |

逐场full success：

- both：`34`；
- epoch-17-only：`27`；
- epoch-18-only：`18`；
- neither：`121`；
- McNemar exact：`p=0.233`。

## 结论

epoch 18减少了一些timeout，但通过增加碰撞换取，full success反而下降。50场中的
`0.300 -> 0.420`属于小样本波动，不能证明全程化有效。

固定决策：

1. 拒绝直接复制epoch-16整网并继续全状态训练；
2. 不追加epoch或seed；
3. epoch-16保留为条件局部Actor；
4. 当时提出的Residual后续在R0被拒绝，当前直接冻结5A和epoch-16并训练Gate。

历史启动入口已移除，防止误运行被拒绝协议。
