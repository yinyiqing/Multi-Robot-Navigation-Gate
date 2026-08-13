# I-E2-M多冲突Actor诊断

日期：`2026-08-13`。

## 结论

I-E2-M没有通过准入，不得替换旧epoch-16，也不得直接追加训练。失败不来自manifest没有
覆盖multi-edge，而来自训练目标和实际恢复任务错位：Actor梯度只更新“近距离且仍在闭合”
的安全状态，没有更新避让后停滞、重新推进和交还E2的恢复状态。40k预算又只走过
`422/2400=17.6%`个训练场景，不能把本次结果解释成“多冲突避障无法学习”。

## 最终闭环结果

同一N5 validation、120场、seed `20260818`：

| 方法 | full success | collision | timeout | mean steps |
| --- | ---: | ---: | ---: | ---: |
| E2 | `0.7417` | `0.060` | `0.0833` | `49.0` |
| E2 + old epoch-16 recovery | `0.7750` | `0.055` | `0.0500` | `41.98` |
| E2 + I-E2-M recovery | `0.6833` | `0.075` | `0.0833` | `48.44` |

I-E2-M相对E2为`7`场改善、`14`场退化、`99`场持平，McNemar exact `p=0.1892`。
multi-edge full success为`0.400`，低于E2的`0.450`和epoch-16的`0.600`。

## 诊断证据

### 1. 训练确实看到了multi-edge

422个实际训练episode中，edge-1/edge-2/edge-3+为`169/127/126`。首版I-E2的
“训练集全是单边冲突”问题已经修复，剩余失败不能再归因于没有运行multi-edge场景。

### 2. Actor更新没有带来有效增益

| 阶段 | full success | collision | timeout | mean steps |
| --- | ---: | ---: | ---: | ---: |
| 20k，Actor冻结 | `0.610` | `0.094` | `0.095` | `57.7` |
| 40k，Actor解冻 | `0.615` | `0.081` | `0.135` | `65.5` |

Actor解冻后full success只增加`0.005`，timeout增加`0.040`，平均步数增加`7.8`。训练
episode的collision也从冻结阶段的`0.118`上升到更新阶段的`0.145`。这不是P1/R1式动作
饱和坍塌，而是更新没有学到有效冲突退出，并加重了长尾拖延。

### 3. 学到的动作仍接近E2

在I-E2-M实际接管的2551个同观测帧上离线推理：

| Actor | mean linear | near-stop rate | mean abs angular |
| --- | ---: | ---: | ---: |
| E2 | `0.0489` | `0.9216` | `0.7154` |
| old epoch-16 | `0.1571` | `0.8103` | `0.8871` |
| first I-E2 | `0.0415` | `0.9240` | `0.7066` |
| I-E2-M | `0.0543` | `0.9106` | `0.6998` |

I-E2-M相对E2的平均线速度只变化`+0.0054`，平均绝对线速度差为`0.0297`；平均绝对
角速度差却为`0.2787`。它主要学成了“换一种转法”，没有形成epoch-16那样不同的冲突
解除动作。在失败episode接管帧上，I-E2-M mean linear为`0.0335`、near-stop rate为
`0.9482`；old epoch-16分别为`0.1584/0.8083`。

### 4. 训练目标与恢复任务错位

当前Actor只从“距离不超过1m且仍在闭合”的状态获得梯度。真正需要学习的避让后停滞、
冲突分离、恢复推进和交还E2状态通常不再满足closing条件，被排除在Actor更新之外。
safe-recovery reward又在机器人距离小于`1.2 m`时被抑制，无法直接教会Actor在最关键的
近车僵持窗口恢复。Critic safety ranking约束接近风险，也不等于学习冲突退出。

### 5. recovery规则不是唯一原因

I-E2-M相对E2的14个退化case中，10个确实调用了I-E2-M，4个完全没有调用；后4个属于
Gazebo闭环分叉，不能归因于Actor。10个实际调用的退化case包含3个timeout，存在大量
反复接管和低速帧，说明Actor本身有问题，而不仅是Gate误切。

## 下一步边界

当前不追加40k，也不扫描reward权重。若继续训练E2配套Actor，必须先书面重定义训练单元：

1. Actor更新集合同时包含approach、avoidance、stalled recovery和release四类阶段；
2. 直接奖励冲突距离恢复、目标进度恢复和成功交还E2，而不是只惩罚近车；
3. 加入E2行为锚点或短时动作约束，防止只有动作漂移而没有可验证收益；
4. 先做离线或短窗反事实准入，再运行新的闭环40k；
5. 预算同时登记场景覆盖，不能只按agent samples登记。

在该协议完成前，旧epoch-16仍是冻结fallback，I-E2-M只作为失败诊断。

## 输出

```text
local_data/ie2m_vs_epoch16_diagnosis/
local_data/ie2m_vs_ie2_diagnosis/
local_data/ie2m_offline_action_diagnosis/
```
