# D5 Independent Dense Actor Full v2

状态：`ready / not started`。代码和协议已准备完成，按用户要求当前不启动训练。

## 唯一目标

训练一个能在Dense场景中从起点到终点全程独立导航的TD3 Actor。它不是局部避险模块，也不依赖另一个Actor在普通状态中接管。

## 独立性约束

- 一个新Actor控制每个episode的所有状态，包括起步、普通行进、强交互、避让后恢复和到达终点；
- rollout、validation和TD target均只使用这个新Actor；
- 5A只用于参数初始化，以及无近邻安全状态的轻量动作约束，从不输出实际控制动作；
- 不使用距离oracle、不在episode中切换Actor、不训练Gate。

## 训练协议

- 保持原TD3结构和24维Actor输入，从5A Actor warm-start，新建ego-motion Critic；
- Actor从完整Dense Replay更新，不启用`interaction-only`或`safety-focused-only`；
- Critic/Actor batch在数据充足时严格按`0.75`交互状态 + `0.25`非交互状态抽取；数据不足时才从另一类回填；
- `>2.0 m`无近邻状态对5A保留轻量anchor，但Q目标仍对这些状态更新；
- 交互状态不设5A动作anchor，线速度和角速度均允许学习；
- Actor学习率保持`1e-6`，取消曾将有效梯度缩小约58倍的Q尺度归一化；
- Actor解冻后每个训练episode输出相对5A的线速度/角速度变化。
- 协议单测固定检查无oracle、无Actor切换、无`interaction-only`和无`safety-focused-only`，防止启动参数退回局部避险模式。

## 短epoch与判断节奏

- 每`20,000 agent samples`为一个短epoch，默认`48`个，总预算仍为`960,000`样本；
- 每个epoch在固定100场`dense_validation_monitor_fast_v2`上评估并保存checkpoint；
- 上一轮实测推算每个短epoch约`40-50`分钟；
- epoch 1保持Actor冻结，用于建立5A基线；`21,000`样本后Actor才具备解冻资格，且仍必须通过危险状态梯度门检查；
- 不用单个100场epoch的波动下结论：至少看连续3个短epoch的同向趋势；
- 100场monitor只用于及时发现退化和挑选少量checkpoint，最终结论仍使用完整1000场dense validation。

## Reward修复

- 邻车距离减小立即处罚，距离增大立即奖励；
- 停车让行或短暂绕行不再要求同时朝目标前进；
- `1.2 m`内存在可见活动邻车时，暂停基础和额外停滞处罚；
- 距离不变不给正奖励，近距离持续处罚仍保留，避免永久停车。

## 准入条件

1. 100场快速monitor上的full success连续3个短epoch整体高于epoch 1的5A基线；
2. collision下降不能主要转化为timeout/unresolved；
3. 候选checkpoint必须在完整1000场dense validation上独立超过5A/5D的`0.3090/0.3140`；
4. 候选固定前不读取sealed test。

## 已知边界

Actor仍只看20维单帧激光、目标距离/方向和上一动作，无法可靠区分墙与机器人，也不能直接观测相对速度。这一轮只验证在不改Actor网络和输入的前提下，修正训练信号后能否学成独立Dense策略。

默认模型名为`independent_dense_actor_from_5a_full_v2_s20260729`，必须fresh start，禁止恢复v1 Replay或optimizer。
