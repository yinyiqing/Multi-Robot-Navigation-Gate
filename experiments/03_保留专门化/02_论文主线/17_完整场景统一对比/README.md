# G17 完整场景统一对比

状态：`registered / running`。登记日期：`2026-08-14`。

## 目的

在包含全部冲突类型的冻结五车场景上，统一比较：

1. `5A`：原宽单Actor；
2. `R2B-best`：按5A流程训练和validation选择的参数匹配大Actor；
3. `F-A1`：5A、epoch-17避障Actor与可部署时序Gate。

G11-F-C的50个独立场景只包含0-edge/edge-1，因此只作Gate准入pilot。本实验才用于
判断三方排序在完整场景中是否保持。

## 数据和预算

- manifest：`g12_full_scene_selection_v1/validation.json.gz`；
- SHA-256：`52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635`；
- 120个独立场景：standard/dense各60，zero/edge-1/multi-edge各40；
- repeat seeds：`20260824/20260825`；
- 3种方法 x 120场 x 2 repeats = `720 episodes`；
- 同一repeat内使用完全相同的场景顺序、seed、物理步长和终止条件。

两个repeat的运行顺序反转：

```text
repeat 1: 5A -> R2B-best -> F-A1
repeat 2: F-A1 -> R2B-best -> 5A
```

## 冻结输入

| 组件 | SHA-256 |
| --- | --- |
| 5A Actor | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| R2B-best Actor | `da28dd5820d09845eea07cb68da45a7afd262fe56e8a71f80bf6b5781551523a` |
| epoch-17 Actor | `149c2e42848ecc9bc478cbed7fd89b9062936dbd5c669b55e6964441685155a5` |
| detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |
| F-A1 Gate | `b28e81d341c145d6fa8c881dd98c7ece5285231e7d080b3f71afcd2dfe3a0beb` |

F-A1固定使用`on=0.29/off=0.19`、minimum hold `3`、stride `2`。不读取本实验结果
修改模型或阈值。本实验不读取sealed test。

## 报告

报告overall以及`standard/dense x zero/edge-1/multi-edge`六个层的full success、agent
success、collision、timeout和平均步数。对full success执行逐场配对McNemar exact检验。

运行入口：

```bash
bash scripts/start_g17_full_scene_comparison.sh
```

实时日志：`logs/active/g17-full-scene-comparison/`。完成后自动归档到
`logs/archive/validation/g17_full_scene_comparison/`。
