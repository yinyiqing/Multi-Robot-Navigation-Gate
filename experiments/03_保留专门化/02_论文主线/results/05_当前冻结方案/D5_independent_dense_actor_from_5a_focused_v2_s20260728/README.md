# D5 Independent Dense Actor Focused v2

状态：`ready / not started`。代码和协议已经准备完成，按用户要求当前不启动训练。

## 修复目标

v1的epoch 7在危险Replay上的原始线速度相对5A仅变化`-0.0066`，并没有真正测试出独立Dense Actor能否学会避让。v2先修复“Actor几乎不更新”和“正确让行拿不到奖励”两个已确认问题。

## 固定协议

- rollout、validation和TD target均由新Actor独立完成，不使用距离oracle或另一个Actor接管；
- 保持原TD3网络和24维Actor输入，从5A Actor warm-start，新建ego-motion Critic；
- Actor只从`<=1.0 m`且正在靠近的危险样本学习Q目标；
- 普通状态使用独立采样的5A动作anchor，危险状态角速度也约束到5A，线速度允许学习减速；
- Critic batch中交互样本比例从`0.50`恢复为已验证过的`0.75`；
- Actor学习率保持`1e-6`，取消曾把有效梯度缩小约58倍的Q尺度归一化；
- Actor解冻后每个训练episode直接输出相对5A的线速度/角速度变化，避免再次跑多个epoch后才发现策略未改变。

## Reward修复

- 邻车距离减小：按距离变化立即处罚；
- 邻车距离增大：不再要求同时朝目标前进，停车让行或侧绕也能获得正反馈；
- 邻车距离不变：不给正奖励，近距离持续惩罚仍存在，因此不会奖励双方永久停车；
- `1.2 m`内存在可见活动邻车时，暂停基础和额外停滞惩罚；
- 风险解除后，原目标进度、前进奖励和停滞惩罚自动恢复；
- 日志同时记录signed/absolute clearance reward，避免正负抵消后误判为没有触发。

## 不可突破的边界

Actor仍只看20维单帧激光、目标距离/方向和上一动作，无法可靠区分墙与机器人，也不能直接观测相对速度。v2用于判断在不改变原Actor输入的前提下，正确训练信号能否得到稳定增益；如果Actor已经产生足够动作变化但完整Dense validation仍不提升，不再继续调reward或增加epoch，应重新讨论观测上限。

## 启动约束

默认模型名为`independent_dense_actor_from_5a_focused_v2_s20260728`，必须fresh start，禁止恢复v1 Replay或optimizer。当前不启动。
