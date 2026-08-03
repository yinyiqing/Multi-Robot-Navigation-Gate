# Exact-edge-2零样本确认

状态：`frozen baseline`。200场独立确认存在正收益但未通过预注册判定；禁止对该版Gate
调参或追加确认。新主线可在新Actor B和新Gate冻结后建立新的互斥协议。

## 收缩后的唯一主张

局部Actor只在`conflict_edge_count=1/max_degree=1`场景训练。部署时，可部署的本机
感知Gate能够在不显式构造冲突图的情况下反复调用该技能，并零样本改善恰好两条
冲突边的场景。

本文不再声称已经解决`edge>=3`或`max_degree>=3`：G3中`edge>=3`为8场改善、
8场退化，full success无净提升；这是明确的失败边界，而不是未来工作措辞。

## 确认集

[`build_manifest.py`](build_manifest.py)从完整dense validation中：

1. 排除G3开发使用的`dense_validation_monitor_v1`全部200个ID；
2. 只保留`conflict_edge_count=2`；
3. 用seed `20260802`对scenario ID做SHA-256排序；
4. 固定前200场，不按历史方法结果、reward或难度挑选。

sealed test仍未读取。生成后的[`validation.json`](validation.json)一经提交即冻结。

审计结果：完整validation共1000场，排除G3的200场后有233个exact-edge-2候选，
最终固定200场且ID重叠为0。确认集中`max_degree=1/2`为`102/98`场，
`simultaneous=1/2`为`119/81`场。

## 冻结方法

- 5A Actor、epoch-16 Actor、G0、G1、G2-A checkpoint保持G3哈希不变；
- learned Gate参数保持`switch-on/off=0.44/0.34`、`hold=3`；
- 5A和learned Gate均使用seed `20260803`、固定物理步`0.001 s`；
- 该确认集不再调阈值、不训练模型、不改变特征。

## 预注册判定

相同scenario ID配对比较，必须同时满足：

1. learned Gate相对5A的full success绝对提升`>=0.08`；
2. paired bootstrap 95% CI下界大于0，McNemar exact `p<0.05`；
3. 至少恢复同场历史Oracle full-success增益的`60%`；
4. agent success不下降，timeout增加不超过1个百分点。

任一条件失败，就不能把edge-2组合泛化写成论文主要结果。通过后只补standard/0-edge
能力保持和必要消融；不恢复edge>=3主张。

## 正式结果

5A和learned Gate均完成冻结manifest中的200个不同scenario ID，顺序与manifest
完全一致。历史Oracle只按相同ID抽取，用于计算预注册的增益恢复率。

| 方法 | agent success | collision | unresolved | full success | timeout | 平均环境步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5A | `737/1000 = 0.737` | `0.263` | `0.000` | `65/200 = 0.325` | `0/200` | `16.54` |
| learned Gate | `812/1000 = 0.812` | `0.188` | `0.000` | `81/200 = 0.405` | `0/200` | `26.63` |
| 历史Oracle | `877/1000 = 0.877` | `0.120` | `0.003` | `118/200 = 0.590` | `1/200` | `53.32` |

learned Gate相对5A为42场改善、26场退化、39场共同成功、93场共同失败。
full-success增益刚好为`+0.080`，20000次paired bootstrap 95% CI为
`[0.000, 0.160]`，McNemar exact `p=0.06812`。因此方向为正，但置信区间下界
没有大于0，McNemar也没有达到`p<0.05`。

同场历史Oracle的full-success增益为`0.590 - 0.325 = 0.265`，learned Gate只恢复：

```text
(0.405 - 0.325) / (0.590 - 0.325) = 0.302
```

即`30.2%`，远低于冻结的`60%`门槛。

## 分层结果

| 分层 | 场景数 | 5A full success | learned Gate | Oracle |
| --- | ---: | ---: | ---: | ---: |
| `max_degree=1` | 102 | `0.275` | `0.363` | `0.510` |
| `max_degree=2` | 98 | `0.378` | `0.449` | `0.673` |
| `simultaneous=1` | 119 | `0.361` | `0.412` | `0.630` |
| `simultaneous=2` | 81 | `0.272` | `0.395` | `0.531` |

四个预注册结构子组方向均为正，`simultaneous=2`的绝对增益最大，为`+0.123`。
这支持“局部技能在部分双冲突场景有用”，但不能覆盖总体显著性和Oracle恢复率失败。

## 判定

| 冻结条件 | 结果 | 判定 |
| --- | --- | --- |
| full success绝对提升`>=0.08` | `+0.080` | 通过 |
| bootstrap 95% CI下界`>0` | `0.000` | 未通过 |
| McNemar exact `p<0.05` | `0.06812` | 未通过 |
| Oracle增益恢复`>=60%` | `30.2%` | 未通过 |
| agent success不下降 | `0.737 -> 0.812` | 通过 |
| timeout增加不超过1个百分点 | `0.000 -> 0.000` | 通过 |

总判定为未通过。不能把exact-edge-2组合泛化作为论文主要已证实结论，也不能使用
这200场重新选择Gate阈值、checkpoint或特征。禁止追加新seed只为把`p=0.06812`
推过0.05；任何新验证必须服务于新的、独立预注册的问题。

5A在episode 123后、learned Gate在episode 87和140后分别遇到Gazebo fixed-step
stall。每次恢复前均核对checkpoint、结果行数和manifest游标；最终两种方法各有
200个不同ID，没有删除、替换或重复episode。stall是未完成episode的基础设施错误，
不是策略timeout。

完整统计由[`analyze.py`](analyze.py)生成，结构化结果见[`summary.json`](summary.json)。

## 入口

默认只跑1场smoke，CPU推理且使用两组独立ROS/Gazebo端口：

```bash
bash scripts/experiment.sh start edge2-confirmation-5a
bash scripts/experiment.sh start edge2-confirmation-learned-gate
```

smoke通过后才显式运行200场；两个任务不要同时启动，避免Gazebo资源竞争：

```bash
DRL_EDGE2_TARGET_EPISODES=200 \
  bash scripts/experiment.sh start edge2-confirmation-5a
DRL_EDGE2_TARGET_EPISODES=200 \
  bash scripts/experiment.sh start edge2-confirmation-learned-gate
```

已完成的1场smoke使用相同scenario
`dense-20260718008645-ea2b215742e9`。5A与learned Gate均为`5/5`成功、无timeout；
Gate强Actor激活比例`0.402`。两条路径均确认CPU设备、manifest、模型加载、结果落盘
和PID清理正常。smoke结果不进入正式统计，也不用于改参数。

固定run name可用于基础设施stall后的断点恢复：

```bash
DRL_EDGE2_TARGET_EPISODES=200 DRL_EDGE2_RUN_NAME=<run_name> \
  bash scripts/experiment.sh start edge2-confirmation-5a
```

启动器会在识别到Gazebo fixed-step stall时自动清理当前ROS/Gazebo子进程，并从
同一checkpoint最多恢复5次；其他异常仍立即退出。自动恢复不删除、替换或重复已
落盘episode，也不改变模型、seed、场景顺序或固定物理步参数。
