# Exact-edge-2零样本确认

状态：`收缩主张和200场确认manifest已冻结；双方法1场smoke通过；待正式运行`。

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
