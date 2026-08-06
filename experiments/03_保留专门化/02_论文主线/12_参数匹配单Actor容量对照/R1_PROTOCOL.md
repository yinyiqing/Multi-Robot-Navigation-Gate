# G12-R1：原宽度训练稳定性控制协议

状态：`completed / result archived`。日期：`2026-08-06`。结果见
[R1诊断](R1_DIAGNOSTIC.md)。

## 唯一问题

在P1前40k完全相同的训练条件下，把Actor保持为原始`24 -> 800 -> 600 -> 2`，是否仍在
解冻后发生相同的性能坍塌？R1只作机制诊断，不进入论文主性能表。

## 冻结配置

- 模型名：`capacity_original_width_r1_n5_seed20260810`；
- 5A Actor-only warm start，原宽度直接加载，fresh原始24维Critic；
- train/validation、哈希、seed、reward、优化器、batch、gamma和探索噪声均与P1相同；
- 前`20,000` agent samples只训练Critic，之后解冻Actor；
- 每`20,000` samples固定评测120场，硬上限`40,000` samples；
- 早停阈值与P1相同；设备固定为CUDA；
- 日志：`logs/active/capacity-original-width-r1/`，完成后归档到
  `logs/archive/diagnostic/g12_r1/`。

启动命令：

```bash
bash scripts/experiment.sh start actor-g12-r1-original-width
```

停止命令：

```bash
bash scripts/experiment.sh stop actor-g12-r1-original-width
```

## 判读规则

- R1也坍塌：P1不能归因于网络扩宽，R2/R3必须优先修正fresh-Critic无约束TD3稳定性；
- R1稳定而P1坍塌：重点检查扩宽初始化和新增参数优化尺度；
- 两者都稳定但没有交互改善：不属于本诊断问题，仍按R2从头课程路线建立公平baseline。

不得根据R1闭环结果修改本次超参数或追加同类seed。R1结束后先形成书面诊断，再冻结
R2/R3实现。
