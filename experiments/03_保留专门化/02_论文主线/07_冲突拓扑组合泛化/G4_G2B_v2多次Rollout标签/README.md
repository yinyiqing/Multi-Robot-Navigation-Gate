# G4 G2-B v2多次Rollout标签

状态：`协议与代码已冻结；等待1场smoke，通过后才运行9场标签稳定性pilot`。

## 目的

G3证明交互存在性Gate有显著正收益，但在200场上只恢复Oracle增益的`45.1%`，
并且`edge>=3`没有净提升。G2-A回答“当前是否存在交互”，没有直接回答“此时哪个
Actor的长期结果更好”。G4只验证后一个问题能否在带噪Gazebo中得到稳定监督。

该pilot不训练Gate、不评估导航性能，也不读取sealed test。若标签不可稳定估计，
立即停止G4，不通过增加rollout次数反复寻找好看的门槛。

## 固定场景

[`select_pilot_cases.py`](select_pilot_cases.py)使用seed `20260802`从G3的200场中固定
三类诊断场景，每类各取`edge=1/2/3+`一场：

- learned Gate相对历史5A改善：候选40场；
- learned Gate相对历史5A退化：候选17场；
- 5A和learned Gate均失败但真值Oracle成功：候选34场。

最终9场见[`pilot_selection.json`](pilot_selection.json)，完整环境定义见
[`pilot_manifest.json`](pilot_manifest.json)。选择使用scenario ID哈希排序，不按
reward、碰撞数或激活比例人工挑选。

这9场来自G3 admission，只能用于标签可行性诊断。后续若训练G2-C，必须从独立
train split采集标签，并使用未参与训练的validation重新评估。

## 冻结采集协议

正式pilot参数：

```text
scenarios = 9
anchors_per_scenario = 2
anchor_steps = 4, 8
ego_per_anchor = 1 (nearest active robot)
horizon = 8 environment steps
rollouts_per_actor_per_batch = 5
independent_batches = 2
confidence = 0.95
bootstrap_resamples = 5000
seed = 20260802
device = cpu (avoid sharing the GPU with other running jobs)
```

每个锚点先用5A重放动作前缀恢复同一物理状态。每个batch分别对5A和epoch-16执行
5次独立带噪闭环rollout。标签按以下优先级比较strong-minus-generalist均值差的
bootstrap置信区间：

1. 本车碰撞率；
2. 碰撞机器人数量；
3. 本车到达率；
4. 最小车间距，同时限制目标进展退化；
5. 目标进展，同时限制车间距退化。

置信区间不能分离时，该batch标记为`ambiguous`。两个独立batch必须给出相同的
非ambiguous Actor标签，锚点才产生最终标签。Actor始终冻结，VLP-16噪声保持开启。

## 冻结准入线

9场正式pilot预计产生18个锚点，必须同时满足：

1. 物理锚点恢复通过率`>=90%`；
2. 双batch一致的非ambiguous标签比例`>=25%`；
3. 至少一个batch非ambiguous的锚点中，双batch一致率`>=70%`；
4. 最终5A和epoch-16标签各至少2个。

任一条件失败即停止G2-B v2，不训练收益Gate。通过只表示“标签可采集”，不表示
G2-C一定改善导航；下一阶段仍需独立train/validation。

## 入口

1场smoke缩短为2步、每Actor每batch 2次、1个锚点，只检查完整数据链和格式，
不用于准入：

```bash
bash scripts/experiment.sh start g4-counterfactual-smoke
bash scripts/experiment.sh status
```

smoke通过后，正式9场参数不可修改：

```bash
bash scripts/experiment.sh start g4-counterfactual-pilot
```

停止命令只管理对应独立端口下的进程组：

```bash
bash scripts/experiment.sh stop g4-counterfactual-smoke
bash scripts/experiment.sh stop g4-counterfactual-pilot
```

结果使用[`analyze_pilot.py`](analyze_pilot.py)汇总。smoke只能检查format-v3数组
维度、batch诊断、锚点误差和进程清理，不得据此修改正式参数。
