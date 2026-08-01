# ICRA 收敛路线：冲突拓扑组合泛化

状态：`G3 50场通过；Gate参数冻结；正在进入200场准入`。

本目录是投稿前两周的唯一执行台账。新实验只有在回答本文核心问题时才进入
这里；不再并行扩展Dense Actor、Critic、reward或大规模感知结构。

## 1. 核心问题

本文研究：只在单条名义冲突边场景中学习的局部避让技能，能否通过本机传感器
驱动的状态Gate反复调用，零样本处理训练中未见的多边、高度数和同时冲突，
同时保留无冲突场景中的普通导航能力。

这里的训练场景仍有5辆车，只是每场仅有一对主要冲突车辆。部署时不显式构造
冲突图、不把多车场景拆成独立pair，也不使用通信、场景标签或其他机器人真值。

简明描述：

> 在简单的单对冲突中学习局部避让，再按本地观测反复调用这项技能，处理复杂的
> 多车冲突组合。

## 2. 主张边界

创新不在“双Actor + Gate”架构，也不声称首次从双机器人推广到多机器人。CADRL、
分层策略切换和混合控制已有直接先例。本文只主张以下完整组合：

1. 用名义路径冲突图定义训练和测试的结构差异；
2. 局部技能只在`conflict_edge_count=1`、`max_conflict_degree=1`上训练；
3. 在未见的多边、高度数和同时冲突结构上测试组合泛化；
4. Actor和Gate部署输入只有本机传感器及本机导航状态；
5. 同时测量复杂冲突收益和0-edge能力保持。

若可部署Gate不能显著超过5A，论文不得用真值Oracle成绩冒充最终方法成绩。

## 3. 已完成证据

### 冻结组件

- 普通Actor：冻结5A；0-edge validation full success约`0.875`。
- 局部交互Actor：冻结epoch-16；原24维TD3 Actor。
- 局部Actor训练：2560/2560场均满足`edge_count=1`、`max_degree=1`、
  `simultaneous_conflict_count=1`。
- G0/G1：本机VLP-16候选形状分数与自运动补偿跟踪特征已完成。
- G2-A：前方180度交互Gate已训练，validation recall/FPR为
  `0.861/0.111`，standard/0-edge FPR为`0.070`。

### 组合上限

在完整dense validation 1000场上：

| 方法 | agent success | collision | full success | timeout |
| --- | ---: | ---: | ---: | ---: |
| 5A | `0.7064` | `0.2930` | `0.3090` | `0.0030` |
| 5A + epoch-16真值Oracle | `0.8476` | `0.1456` | `0.5450` | `0.0180` |

epoch-16全程运行会产生大量停车超时，因此它是局部技能，不是独立导航Actor。

### 未见冲突结构上的结果

| 测试层 | 场景数 | 5A full success | Oracle组合 | 提升 |
| --- | ---: | ---: | ---: | ---: |
| `edge=1` | 211 | `0.502` | `0.730` | `+0.227` |
| `edge>=2` | 747 | `0.221` | `0.470` | `+0.249` |
| `edge>=3` | 453 | `0.135` | `0.393` | `+0.258` |
| `max_degree>=2` | 601 | `0.201` | `0.466` | `+0.265` |
| `simultaneous>=2` | 480 | `0.156` | `0.396` | `+0.240` |

`edge>=2`中Oracle相对5A有`228`场改善、`42`场退化，配对McNemar exact
`p=4.15e-32`。这证明局部技能存在多冲突复用价值，但尚未证明可部署Gate能实现它。

### 已拒绝路线

- 多版独立Dense Actor未同时解决碰撞、超时和通行顺序，不再继续堆叠训练机制。
- 最小LiDAR距离不能区分静态障碍与机器人，只保留为可部署规则下界。
- G2-B v1单次8步反事实标签受传感器噪声放大，不可重复，禁止扩大采集。

## 4. 剩余工作与关口

### G3：现有G2-A端到端闭环

目标：先判断已经训练好的可部署Gate能否转化为导航收益，不先启动昂贵的新数据采集。

实现要求：

- 加载冻结5A、epoch-16、G0 detector和G2-A Gate；
- 每辆车独立维护G1 tracker；
- Gate使用24维Actor状态、形状和连续运动特征；
- 使用switch-on/off双阈值和最短保持时间抑制抖动；
- 记录强Actor激活比例、Gate均值和每episode切换次数；
- 不读取Gazebo机器人位置、冲突图、dense标签或scenario类别。

执行顺序：单场smoke -> 固定50场参数选择 -> 固定200场准入。

### G3准入线

固定200场上同时满足：

1. full success建议达到`>=0.45`，或至少恢复60%的同场Oracle增益；
2. 相对5A的配对改善明确多于退化；
3. timeout不出现系统性增加；
4. 0-edge/standard full success下降不超过3个百分点；
5. `edge>=2`和`max_degree>=2`仍有正收益。

只允许在50场上选择switch-on/off阈值和最短保持时间。200场用于准入，不能继续
反复调参。

### G4：Gate失败后的唯一升级

仅当G2-A能识别交互、但端到端收益明显低于Oracle时，才进入G2-B v2：

- 每个Actor从同一锚点运行多次独立带噪rollout；
- 比较碰撞率、到达率、净空和进展的期望及置信区间；
- 置信区间不能分离的状态标记为ambiguous并默认5A；
- 先做小规模标签稳定性pilot，再决定是否扩大采集。

如果G2-A在在线闭环中无法形成任何正收益，不默认认为增加反事实标签能够解决，
应停止Gate主线并收缩论文主张。

### G5：完整Validation

G3或G4通过后，在固定场景ID上比较：

1. 5A；
2. 5D历史基线；
3. epoch-16 always-on诊断；
4. 最小LiDAR规则Gate；
5. learned Gate without G0/G1；
6. learned Gate without hysteresis；
7. 完整learned Gate；
8. `2.0 m`真值Oracle上界。

先完成dense validation 1000场，再补standard/0-edge validation。报告
agent/full success、collision、unresolved、timeout、平均步数、激活比例、切换次数
和推理开销。

必须按以下三轴分层：

- `conflict_edge_count`；
- `max_conflict_degree`；
- `simultaneous_conflict_count`。

统计使用相同scenario ID配对，full success使用McNemar exact检验，并报告bootstrap
置信区间。固定200场至少重复运行以估计Gazebo轨迹波动。

### G6：冻结与Sealed Test

Actor、Gate checkpoint、特征、阈值、滞回和最短保持时间全部冻结后，才允许读取：

- dense sealed test 2000场；
- standard sealed test 1000场。

test只运行最终方法和预先固定的必要基线，不依据test结果修改任何参数。

## 5. 两周排期

| 时间 | 工作 | 交付物 |
| --- | --- | --- |
| 第1-2天 | 在线G2-A接入与离线测试 | `learned_gate`执行模式、结构化日志 |
| 第3-4天 | 单场smoke、50场参数选择、200场准入 | G3报告与继续/停止决策 |
| 第5-7天 | 完整validation与规则Gate | 主结果表、分层结果 |
| 第8-9天 | 必要消融、重复统计、失败案例 | 消融表、置信区间、案例图 |
| 第10天 | 冻结全部模型和参数 | checkpoint哈希、冻结配置 |
| 第11-12天 | sealed test | 最终test表 |
| 第1-14天并行 | 论文写作 | 方法、实验、图表和相关工作 |

## 6. 当前执行状态

- [x] 冻结5A和epoch-16。
- [x] 审计单冲突训练分布。
- [x] 完成1000场Oracle组合及冲突拓扑分层。
- [x] 完成G0/G1和G2-A离线验证。
- [x] 完成在线`learned_gate`模式。
- [x] 完成50场参数选择并冻结`0.44/0.34/hold=3`。
- [ ] 完成200场准入。
- [ ] 完成完整validation与必要消融。
- [ ] 冻结并运行sealed test。

每次实验完成后必须在本节更新状态，并在本目录新增对应结果README。禁止只留下日志
而不记录协议、checkpoint、场景清单、seed、结果和决策。

50场协议、配对结果和参数冻结决策见
[`G3_50场参数冻结`](G3_50场参数冻结/README.md)。

## 7. G3运行入口

默认只运行1场smoke，且使用独立端口、PID和本目录下的输出路径：

```bash
bash scripts/experiment.sh start learned-gate-validation
bash scripts/experiment.sh status
```

smoke通过后显式设置场景数。开发入口硬限制最多200场：

```bash
DRL_LEARNED_GATE_TARGET_EPISODES=50 \
  bash scripts/experiment.sh start learned-gate-validation
```

停止命令只管理该入口创建的进程组：

```bash
bash scripts/experiment.sh stop learned-gate-validation
```
