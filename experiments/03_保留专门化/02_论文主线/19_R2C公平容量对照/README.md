# G19-R2C 公平容量对照

状态：`stopped / superseded by pre-5A-start fairness requirement`。登记日期：
`2026-08-15`，断点恢复及停止日期：`2026-08-16`。

首轮原宽控制因外部终止停在`33,430/60,000` agent samples，Actor仍处于冻结阶段。审计
确认`latest`完整保存replay、Critic优化器、场景采样状态和validation历史，Actor优化器
状态为空。恢复运行只接受SHA-256
`3520feda0d552ca6a04bee8082c9fea47e99d3c0c1f5ec3b5a50976022efaa73`的该断点，并保持
原超参数、manifest、seed、41k解冻边界和60k预算不变。原宽稳定闸门通过后才自动启动
加宽Actor；结果无论是否超过F-A1均如实报告，不以目标成绩区间选择模型。

恢复后在约`34.3k`、Actor仍冻结时主动停止。原因不是性能结果，而是公平性问题重新核对：
当前对照要求加宽Actor从5A之前的共同3D2起点进入五车训练阶段，不能从已经完成的5A函数
再扩宽。G19从5A开始只适合回答“5A后续稳定微调”的问题，不再作为当前容量对照。该次
恢复没有启动wide分支，也没有发生原宽Actor更新；后续不得从本目录的checkpoint继续。

## 目的

R2B严格复刻了历史5A的五车训练流程，但其自动best位于Actor有效更新前；Actor解冻后，
原宽控制和加宽Actor都因fresh Critic发生严重退化。因此R2B只能作为流程稳定性失败证据，
不能回答“一个真正训练过、参数量匹配的单Actor能否替代两个专门化Actor”。

R2C使用同一稳定化训练协议同时运行原宽控制和参数匹配加宽Actor。只有加宽Actor实际更新、
训练稳定并超过冻结5A，才允许作为论文的有效容量baseline候选。实验不以得到
`5A < R2C < F-A1`为目标；若R2C超过F-A1，必须如实报告并重新判断主方法。

## 成对控制

| 项目 | R2C-original | R2C-wide |
| --- | --- | --- |
| warm start | 冻结5A Actor | 同一5A Actor函数保持扩宽 |
| Actor结构 | `24-800-600-2` | `24-1137-855-2` |
| Actor参数 | `501,802` | `1,003,127` |
| 部署输入 | 本车24维观测 | 本车24维观测 |
| train/validation | 完全相同 | 完全相同 |
| seed、场景顺序、预算 | 完全相同 | 完全相同 |
| reward、Critic和优化器 | 完全相同 | 完全相同 |

加宽结构只匹配两个Actor的参数总量，不计训练时Critic；最终论文另行报告Gate和detector的
部署参数，不能把本实验描述为整个系统的总参数严格匹配。

## 数据

- train：`g12_r3_mixed_v1/train.json.gz`，SHA-256
  `c2ce37e51e8e98423d6ed6d295a7f5cf54d02e76c42f6459ce35003c899e0841`；
- train按`standard/strong/dense/strong`四槽循环，包含0/1/2/3+冲突，不是Gate的0/1
  冲突训练集；
- validation：`g12_full_scene_selection_v1/validation.json.gz`，120个冻结场景，SHA-256
  `52435d6c5bdf9914e7212dd29cb4bfec074257f72d85f0d71741deee7c63b635`；
- seed：`20260826`；不读取G17、G18或sealed test选择checkpoint。

## 稳定性修订

R2C不是继续R2B或R3 checkpoint，而是从同一冻结5A函数重新开始。相对已失败配置只做
预注册的稳定性修订：

1. fresh 87维ego-motion邻域Critic先校准40k agent samples；
2. Actor在41k后解冻，只在最后约20k中更新；
3. Actor LR从R3的`1e-5`降为历史5A量级`2e-6`，Critic LR从`8e-5`降为`2e-5`；
4. 使用Q尺度归一化、`1.0`梯度裁剪，并仅在无2m近车的安全状态锚定5A行为；
5. 两个宽度使用完全相同的动态reward和邻域Critic，不把训练侧真值输入部署Actor。

训练配置固定为：batch `256`、discount `0.99999`、每20k评测120场，共60k/模型。动态
reward使用`average`、self weight `0.8`和距离加权；邻域Critic使用ego-motion上下文并
以`0.5`比例抽取交互样本。Actor全程控制五车，不使用oracle Actor切换。

## 自动闸门

先运行R2C-original。相对40k Actor冻结边界，60k必须同时满足：

- full success下降不超过`0.05`；
- agent success下降不超过`0.03`；
- collision增加不超过`0.03`；
- timeout增加不超过`0.02`；
- Actor参数确实改变。

原宽控制未通过时自动停止，不启动R2C-wide，说明当前稳定性修订仍无效。原宽通过后才运行
加宽Actor。加宽Actor使用同一稳定闸门；其最终是否超过5A和原宽控制作为容量结论报告，
不反向改变checkpoint。

## 执行

```bash
bash scripts/start_training_g19_r2c_paired_pilot.sh

# 仅用于已审计的33,430样本断点
G19_RESUME_ORIGINAL=1 bash scripts/start_training_g19_r2c_paired_pilot.sh
```

实时日志：`logs/active/g19-r2c-paired-pilot/`。成功完成后归档到
`logs/archive/training/g19_r2c_paired_pilot/`。本pilot不修改5A、epoch17或Gate。
