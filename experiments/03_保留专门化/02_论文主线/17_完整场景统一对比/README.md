# G17 完整场景统一对比

状态：`complete / sequentially audited`。
登记日期：`2026-08-14`。

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

## 结果

六个正式结果数组均为`(120, 17)`，场景ID顺序与冻结manifest完全一致，每组终止计数
均为`600`。seed `20260824`的5A和R2B-best已使用单Gazebo顺序复测；此前并行Gazebo
恢复结果已隔离，不进入正式汇总。两个repeat合并结果如下：

| 方法 | full success | agent success | collision | timeout | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5A | `0.5000` | `0.8075` | `0.1925` | `0` | `21.63` |
| R2B-best | `0.5292` | `0.8092` | `0.1908` | `0` | `19.55` |
| F-A1 | **`0.5958`** | **`0.8517`** | **`0.1483`** | `0` | `31.61` |

两个repeat中F-A1的full success均为最高：seed `20260824`上
`5A/R2B-best/F-A1 = 0.4917/0.5167/0.5750`，seed `20260825`上为
`0.5083/0.5417/0.6167`。合并配对结果：

- F-A1相对5A：42场改善、19场退化、179场持平，McNemar exact `p=0.00444`；
- F-A1相对R2B-best：35场改善、19场退化、186场持平，`p=0.04022`；
- R2B-best相对5A：22场改善、15场退化、203场持平，`p=0.32401`。

按冲突边数聚合：

| 分层 | episodes | 5A full | R2B-best full | F-A1 full |
| --- | ---: | ---: | ---: | ---: |
| zero | `80` | `0.8625` | `0.8500` | **`0.9500`** |
| edge-1 | `80` | `0.4625` | `0.5875` | **`0.6000`** |
| multi-edge | `80` | `0.1750` | `0.1500` | **`0.2375`** |

F-A1在multi-edge相对5A为13场改善、8场退化（`p=0.38331`），相对R2B-best为
13场改善、6场退化（`p=0.16707`）。因此完整场景结果支持可部署Gate相对两个公平
baseline的总体导航收益，也显示收益在multi-edge方向最大；单独层级样本仍小，不能把
multi-edge结果写成显著。F-A1平均步数比5A高`46.2%`，interaction Actor平均
占比为`0.5524`、每场平均切换`7.73`次，效率代价必须与成功率收益同时报告。

R2B-best相对5A为`+0.0292`，但配对检验不显著，没有形成可靠容量收益。该baseline的自动best
仍未使用新增容量，后续Actor更新坍塌，因此只能称为“5A流程和预算匹配的训练流程输出”，
不能称为训练充分的大Actor。

## 产物

结构化汇总：`local_data/summary.json`。正式日志归档：
`logs/archive/validation/g17_full_scene_comparison/`。

## 报告协议

报告overall以及`standard/dense x zero/edge-1/multi-edge`六个层的full success、agent
success、collision、timeout和平均步数。对full success执行逐场配对McNemar exact检验。

运行入口：

```bash
bash scripts/start_g17_full_scene_comparison.sh
```

运行时日志原位于`logs/active/g17-full-scene-comparison/`；本次完成后已归档到
`logs/archive/validation/g17_full_scene_comparison/`。
