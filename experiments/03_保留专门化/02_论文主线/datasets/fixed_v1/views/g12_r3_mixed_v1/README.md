# G12-R3完整五车混合训练视图

状态：`frozen derived train schedule`。日期：`2026-08-08`。

`train.json.gz`只派生自冻结的navigation train，不读取validation或test。它把完整五车
训练来源展开成确定性的四槽循环：

```text
standard -> strong -> dense -> strong
```

一个完整周期共24,000个episode槽位：standard/dense各6,000，strong 12,000。因此broad
与strong各占50%，broad内部standard/dense各占50%。strong槽位按
`deep/deep/close/close/margin`循环，对应40/40/20。

strong池是standard/dense train的确定性子集。调度中的重复只表示重采样；每个别名场景
都在`view.g12_r3_source_scenario_id`保留原ID，不能把24,000错误写成24,000个独立场景。

```text
train.json.gz
SHA-256 c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841
```

复现：

```bash
/usr/bin/python3 scripts/build_g12_r3_mixed_view.py
```
