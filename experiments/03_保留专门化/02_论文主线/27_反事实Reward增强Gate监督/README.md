# G27 反事实 Reward 增强 Gate 监督

状态：P0/P1/P2/P3 已完成；冻结 B3 的 P4 独立测试运行中。

登记日期：2026-09-10。

## 1. 背景与目标

导师指出，当前 Router 的二值监督

~~~text
y_t^i = 1[d_t^i <= 2.0 m]
~~~

只描述“最近活动机器人是否进入 2 m”，不直接回答“当前应由哪个冻结 Actor 控制”。因此，
2 m 内但没有实际冲突、已经分离或同向运动的状态仍可能被标为 interaction，造成持续保守控制。

G27 在保留上述交互阶段监督的同时，增加同一状态下两个冻结 Actor 的一步反事实 reward 差：

~~~text
Delta r_t^i = r_I(s_t, a_I) - r_N(s_t, a_N)
~~~

G27 的目标不是给现有 B2 增加一个补充实验，而是用导师提出的监督设计替换当前只学习 2 m
交互标签的 Router。研究问题是：Delta r 能否提供稳定、可学习的策略优势信息，并让 Gate 的
监督从“是否邻近其他机器人”扩展为“交互阶段 + 两个候选策略的一步相对结果”。成功、安全、
timeout 是闭环采用条件；interaction Actor 占比与完成步数作为代价完整报告，不是导师提出的
方法成立条件。若新独立测试确认，论文方法定义、主图、公式、训练流程和主结果均切换到 G27；
原 B2 仅保留为 proximity-only Router 对照。

## 2. 与历史反事实实验的边界

本项目此前已经进行过两类相近实验，均为历史失败证据，不得重复运行：

1. G2-B v1：同一锚点下两个 Actor 各执行一次 8 步分支；单次带噪 rollout 标签因重复性不足
   被拒绝。
2. G4 G2-B v2：每个 Actor 多次执行 8 步分支；同一 Gazebo 进程内 reset/replay 后的
   DiffDrive 内部状态、仿真时间、位置、朝向和 active mask 无法可靠恢复，smoke 为 0/1
   通过，因此正式 9 场 pilot 被封禁。

G27 不修补、不放宽也不重跑上述路径。唯一允许的新技术路线是：

- horizon 从 8 步改为严格 1 个环境步；
- 每个反事实分支使用全新、相互隔离的 Gazebo 进程；
- 从同一场景初始状态按相同确定性动作前缀重放到锚点；
- 在分支步只替换目标机器人的 Actor，其他活动机器人动作保持完全相同；
- 先做同 Actor 重复分支噪声审计，通过后才允许扩大。

## 3. 冻结项与数据边界

以下组件全程冻结，不允许训练或覆盖：

- Navigation Actor：generalist-5a；
- Interaction Actor：interaction-epoch16；
- G0/G1 感知与跟踪；
- 当前论文方法 B2、阈值、G25/G26 结果和全部 checkpoint（只读保留，作为替换对照）。

G27 只允许使用现有 navigation-train 场景和固定 validation 场景进行开发。不得读取 G25
sealed 结果来选择 reward、模型、损失权重、阈值或停止规则。原 G25 不能作为修改后 Router
的确认性测试。

所有新增产物统一放在本目录：

~~~text
local_data/
  protocol/       # 冻结配置、场景/锚点清单、哈希
  pilot/          # 一步反事实稳定性 pilot
  counterfactual/ # 通过 pilot 后的 train/validation Delta-r 数据
  training/       # 新 Router checkpoint 与训练摘要
  validation/     # development 闭环结果
  test/           # 仅在准入后生成的新独立测试
logs/             # 本实验入口产生的日志索引或链接
~~~

不得把新输出写回 G2-B、G4、G11、G25 或 G26 目录。

## 4. 共同评价 Reward

两个分支必须使用完全相同的评价 reward，不能分别使用两个 Actor 历史训练时的 reward。
G27 固定使用当前环境的基础单机器人任务 reward：

~~~text
goal                     +100
collision                -100
progress weight          20.0
forward-action weight     0.5
turn penalty weight       0.2
obstacle penalty weight   0.5
stagnation penalty        0.03
~~~

固定关闭 cooperative reward、robot proximity shaping、safe recovery、wall clearance、
local navigation、yield priority 及其他额外 shaping。主记录为目标机器人的一步总 reward；
goal、collision、progress、forward、turn、obstacle 和 stagnation 分量同时落盘，仅用于解释，
不得在看过 pilot 或 validation 结果后更换主 reward。

一步 reward 可能偏向立即推进而低估延迟避碰收益，因此 Delta r 与 2 m 交互标签共同构成新
Router 的监督，不能脱离交互标签单独生成最终硬标签。这里的“辅助”只描述多任务损失中的角色，
不表示 G27 是论文补充方法；最终部署方法将整体替换 B2。

## 5. P0：一次性稳定性 Pilot

### 5.1 样本

- 只从 training split 选择；
- 候选帧只来自已冻结的 G11-A1 5A navigation-train shards，不重用其他方法访问的状态；
- 候选固定为实际环境第 `4` 步且仍活跃的机器人；该步保留非空动作前缀，能测试
  fresh-process replay，同时避免旧 G11-A1 非固定步进轨迹在较晚步上的终止时刻偏差；
- 距离固定分为 `<=1.2 m`、`(1.2,2.0] m`、`(2.0,3.0] m`、`>3.0 m` 四层；
  部署动作的 L2 差异固定以 `0.25` 分为 low/high 两层；
- 使用 `20260910|scenario_id|step|ego_index` 的 SHA-256 排序，每个“距离×动作差异”单元
  取 4 个、每场最多一个，共 32 个锚点；不按 reward、episode 结局或图形可视效果挑选；
- 每个锚点运行 N1/N2/I1/I2 四个全新 Gazebo 分支；
- 每个分支只执行一步，其他机器人使用相同的 Navigation Actor 动作。

ROS/Gazebo 若在产物写入前发生服务未就绪或进程超时，同一 reference/branch 允许最多 3 次
基础设施启动尝试；每次日志分别保留。已有 JSON 绝不覆盖，三次均失败则停止队列。该机制只
恢复同一冻结输入的启动失败，不增加有效 rollout 次数，也不允许多次结果择优。

锚点的 G11-A1 帧只用于预先确定 `scenario/step/ego` 和分层；正式分支不直接使用
shard 中的物理状态。每场先由一个独立 reference 进程从 manifest 初始状态执行冻结
Navigation Actor，保存实际动作前缀、锚点物理状态和两个 Actor 的锚点动作；四个分支
再分别从全新进程重放该前缀。

### 5.1.1 P0 前的实现 smoke 记录

首次实现 smoke 暴露出一个在 P0 前必须关闭的协议问题：当候选范围为 `4--40`
步时，哈希最前的锚点位于第 16 步；该 ego 在旧 G11-A1 shard 中仍活跃，但在当前
固定物理步进的 reference 重放中已终止。G11-A1 当时未启用固定物理步进，因此旧
shard 可用于预分层，不能保证较晚的精确锚点仍有效。该 smoke 在 reference 阶段即停止，
没有运行 N1/N2/I1/I2、没有产生 reward 比较，不属于 P0 数据。

在不读取 reward 或 episode 结局的前提下，对旧 training shards 的候选数进行了一次性
可行性计数：第 4 步的八个“距离×动作差异”单元分别有 113--505 个不同场景可选。
因此在正式 P0 生成前将锚点步一次性改为固定第 4 步；除此之外不改变分层、哈希、
reward、对齐门槛或 P0 通过条件。旧清单和失败日志保留为 `pre-P0 diagnostic`。

修订后的单锚点端到端 smoke 已通过：reference 与 N1/N2/I1/I2 全部使用独立
Gazebo 进程，四个分支均通过对齐门槛；最大位置误差约 `1.9e-8 m`，最大线速度
误差约 `4.7e-8 m/s`，环境返回 reward 与固结分量重算一致，进程和端口清理通过。
该 smoke 仅验证实现，不并入 P0 的 32 个锚点。正式锚点清单 SHA-256 为
`4a1dc5d359f12ef0c0963c3ad69f12197503d1baf8d397663f77e25bceb7ea0b`。

正式 P0 首次启动后，前 6 个锚点的 24 个分支已全部对齐；随后一个 reference 完成原子
JSON 写入和 roslaunch 关闭后，ROS Python 在解构阶段出现 segmentation fault，且一个专属
Gazebo master 端口未在 30 秒内释放，编排器按保护逻辑中止。这是 P0 基础设施中断，
尚未计算或读取 P0 reward 判定。修复严格限于：完成 `env.close()` 和日志刷新后避免
rospy 解构路径，并使用每个 slot 专属的 ROS/Gazebo 端口定向清理残留进程。续跑仍使用
相同的 run manifest 和锚点清单，通过 artifact hash 跳过已完成分支；不重跑、不挑选、
不修改任何实验门槛。

### 5.2 沿用的状态对齐门槛

分支前必须满足：

~~~text
active mask 完全一致
最大位置误差          <= 0.02 m
最大朝向误差          <= 0.02 rad
最大线速度误差        <= 0.02 m/s
最大角速度误差        <= 0.03 rad/s
~~~

这些门槛沿用 G4 的预注册标准，不因 G27 结果放宽。

### 5.3 一次性通过条件

P0 必须同时满足：

1. 至少 90% 锚点通过全部状态对齐门槛；
2. 同 Actor 两次分支的 collision/goal 结果完全一致；
3. 以同 Actor 重复 reward 差绝对值的 95 分位数作为预先定义的噪声界；
4. 对每个锚点计算 `Delta r_1=r_I1-r_N1`、`Delta r_2=r_I2-r_N2` 和二者均值；
  若均值绝对值超过噪声界，则该锚点可分辨；可分辨锚点至少占 25%；
5. 在可分辨锚点中，`Delta r_1` 与 `Delta r_2` 同号的比例至少 80%；
6. 可分辨锚点不能全部只支持同一个 Actor。

任一条件失败即停止 G27，不增加 rollout 数、不放宽对齐门槛、不更换 reward、不改采样 strata，
并保留当前 B2 论文主线。

## 6. P1：有限规模反事实数据

仅在 P0 全部通过后执行：

- training：最多 512 个锚点；
- validation：最多 120 个锚点；
- 候选来自 G11-A1 对应 split 的第 `4--40` 个偶数环境步，使用与 P0 相同的四个
  距离层和两个动作差异层；
- 在不读取 reward 或 episode 结局的前提下做确定性全局一对一匹配：training 每层
  64 个，validation 每层 15 个；
- 每个场景最多一个锚点，train/validation 场景 ID 严格互斥；
- 继续使用 N1/N2/I1/I2 和全新 Gazebo 分支；
- train/validation 场景 ID 严格互斥；
- reference 使用冻结 G0 和确定性 tracker 在当前固定步轨迹上生成精确的 8 帧、
  82 维 Gate 输入窗口；不将旧 shard 的物理状态作为新标签的训练输入；
- 原始分支结果、状态对齐、reward 分量、y、Delta r、动作、Gate 输入与哈希全部落盘；
- 不给现有约 7 万个 Gate 帧逐帧补做反事实 rollout。

如果有效锚点不足目标数量，只报告实际数量，不通过修改选择规则补齐。

在 P1 完成前冻结数据充分性下限：training 至少 192 个、validation 至少 40 个可用锚点，
且八个预注册分层中 training 每层至少 8 个、validation 每层至少 3 个。该下限只判断能否训练
优势辅助头，不用于补样或改变分层；未通过则停止 G27 并保留 B2。

### 6.1 完成结果

P1 已完成并通过审计。training 从 512 个候选中保留 251 个四分支对齐锚点，八层可用数为
`25--39`；validation 从 120 个候选中保留 57 个，八层可用数为 `4--12`。两者均满足总数和
逐层下限，场景 ID 互斥。审计结果为 `local_data/counterfactual/audit/summary.json`，其
SHA-256 字段为 `205917dbeb6e37584e2b4a772d803038eeaae6f9f266e7e4a1d7b9b00c20b993`。

## 7. P2：唯一 Router 候选

新候选暂记为 B3-reward-aware。保持当前 82 维部署输入、8 帧 GRU、stride、双阈值滞回和
minimum hold，不改变两个 Actor。

共享 GRU 后使用两个输出头：

~~~text
interaction head: 预测 p_t，使用全部原 y 标签的 weighted BCE
advantage head:   预测 A_t，使用有反事实标签样本的 masked Huber loss
~~~

Delta r 的归一化统计只由 P1 training 计算。总损失固定为两个归一化损失等权相加，不进行
loss-weight sweep。部署只能使用预测的 p_t 和 A_t，不能输入真实 reward、真值位置或 Delta r。

B3 从冻结 B2 的 GRU 和 interaction head 初始化，advantage head 使用 seed `20260910` 初始化；
随后只更新 B3 Router。每个 epoch 分别遍历全部原交互标签帧和全部 P1 优势锚点，按各自样本
总权重累积两个任务的平均梯度，再以 `0.5 BCE + 0.5 Huber` 更新一次，避免样本量差异隐式
改变任务权重。checkpoint 仅按 P1 validation 上的等权联合损失选择，不扫描 loss 权重或结构。

首次 B3 闭环运行前，联合决策已冻结为：

~~~text
score_t = sigmoid(logit(p_t) + clip(A_t, -1, 1))
~~~

其中 `A_t` 直接预测 training-only 统计标准化后的 `Delta r`。部署继续使用 B2 的 on/off
阈值 `0.43/0.33`、最短保持 3 个 Router 更新和 stride 2。完整机器可读配置见
`local_data/protocol/router_decision.json`；不得根据 P3 结果修改。

最终路由分数和 on/off 阈值必须在首次闭环 validation 前写入
local_data/protocol/router_decision.json 并冻结。只允许一次预先登记的 validation 选择，
不得逐次修改公式或阈值追逐闭环结果。

## 8. P3：Development 闭环准入

唯一 B3 已完成训练：40 epoch 中由联合 validation loss 选择 epoch 2，checkpoint SHA-256 为
`b4fad03157a44a7eba1a318e08fe1e0c345e2c9ea3ea97507a1bc6a6c0b01aba`。P3 于 2026-09-10
启动，只新增运行 B3；5A、原 B2 和 2 m 特权 reference 复用同一 Dense256 protocol 的冻结结果。

在 Dense256 development 上同场比较：

~~~text
5A
原 B2
B3-reward-aware
2 m privileged reference（仅作诊断）
~~~

报告 full success、robot-level collision、timeout、raw steps、paired-success steps、
failure-penalized steps、interaction selection share 和 switches。

B3 进入新独立测试必须同时满足：

1. full-success 点估计不低于 B2；
2. robot-level collision 相对 B2 不增加超过 2 个百分点；
3. timeout 相对 B2 不增加超过 1 个百分点；
4. interaction selection share 至少下降 10 个百分点，或 paired-success steps 至少减少 5 步；
5. 结果文件、场景顺序、终止记账和冻结 checkpoint 哈希审计全部通过。

未通过即判定本次替换失败并恢复以 B2 为论文方法，不做第二个 B3 变体。

### 8.1 完成结果与停止决定

P3 以固定 seed `20260810` 一次完成全部 256 场，没有基础设施重启或有效场次重跑。结果文件
`local_data/validation/results/g27_dense256_b3_s20260810.npy` 的 SHA-256 为
`3b006bbff5597b9dfc54cc8f7503cd2644cbf86c7ec16df847a9ea9a15bce316`，正式摘要为
`local_data/validation/p3_summary.json`。

~~~text
                         B2          B3        B3 - B2
full success           0.4258      0.4531      +0.0273
robot collision        0.2102      0.1969      -0.0133
episode timeout        0.0078      0.0078       0.0000
raw steps             34.3984     35.7188      +1.3203
interaction share      0.70018     0.69979     -0.00039
paired-success steps                              +0.37 (75 pairs)
~~~

前三项准入通过，但 interaction share 没有下降 10 个百分点，paired-success steps 也没有减少
5 步，因此第 4 项效率准入失败，`admission_passed=false`。B3 的开发集成功/碰撞点估计虽优于
B2，但没有解决本次替换预先指定的问题，不能据此绕过停止线。G27 到此停止；论文保留 B2/G25，
不训练第二个 B3，不启动 P4，也不把 P3 点估计写成论文确认性结果。

### 8.2 P4 前目标澄清与继续授权

上述停止决定随后因研究目标解释错误而被显式修订，而不是被删除或伪装成 P3 原本通过。导师的
原始要求是解决 Gate 监督信号不合理：在现有 2 m 标签之外，加入同状态两个 Actor 的一步
reward 差，并联合两种预测决定切换。导师没有要求降低 interaction share 或完成步数。将旧 B2
的效率代价转成 B3 的硬性采用条件，是实验规划者额外引入且不符合该要求的约束。

研究者在看到完整 P3 后明确确认：新监督应作为论文主线替换，P3 中 full success 提升、collision
下降且 timeout 不变符合预期；效率继续如实报告，但不否决替换。该澄清发生在 P4 manifest
生成和 `[384:640]` 测试片段读取之前。机器可读记录为
`local_data/protocol/p4_continuation_amendment.json`。

继续授权严格限于冻结的唯一 B3 和已预先写好的 P4，不允许根据 P3 修改训练数据、checkpoint、
reward、损失、融合公式、阈值、hold 或 stride，也不训练第二个候选。P3 仍是 development
evidence，不能写成最终确认性结果。

## 9. P4：新独立测试与论文边界

P4 只能在 P3 完整审计和上述目标澄清均存在后运行 `scripts/prepare_g27_p4_manifest.py`。P4
固定使用 Dense test
原始顺序的 `[384:640]`，共 256 个五车场景；该片段不与 Dense train/validation、G25 使用的
`[0:256]` 或 G26 使用的 `[256:384]` 重叠。manifest 生成器同时核验 scene ID 和完整
`map + boxes + starts + goals + headings` 几何签名。选择只按冻结原始顺序，不按 B3 输出、
reward、冲突层或图形效果筛选。

确认评测固定为：

~~~text
methods       5A, proximity-only B2, reward-aware B3
scenes        256
repeat seeds  20260911, 20260912, 20260913
episodes      3 x 256 x 3 = 2304
horizon       300 environment steps
~~~

运行顺序按 repeat 循环轮换三个方法，以减小长队列时间顺序的系统偏差。模型、`0.43/0.33`
阈值、minimum hold 3、stride 2、Gazebo 固定步进、终止口径和 checkpoint 均不改变。
入口保存关键执行文件、manifest 和 checkpoint 的 SHA-256；基础设施中断只允许从已审计的
逐场状态继续，不允许重跑完成场次或挑选 attempt。

P4 继续以 full-team success 为 primary metric，robot-level collision 为预指定 safety metric，
并报告 agent success、unresolved、episode timeout、raw termination steps、paired-success
steps、failure-penalized completion steps、interaction selection share 和 switches。重复 seed
在场景层聚类；差值使用 20,000 次 scene-cluster BCa paired bootstrap，full-success 主检验
同时使用 100,000 次双侧 scene-level sign-flip，`alpha=0.05`，统计 seed 固定为 `20260910`。

B3 正式替换 B2 必须同时满足：

1. B3 相对 5A 的 full-success 差值 BCa 95% CI 下界大于 0，且双侧 sign-flip `p<0.05`；
2. B3 相对 5A 的 robot-level collision 差值 BCa 95% CI 上界小于 0；
3. B3 相对 B2 的 full-success 点估计不降低；
4. B3 相对 B2 的 robot-level collision 增幅不超过 2 个百分点；
5. B3 相对 B2 的 episode timeout 增幅不超过 1 个百分点；
6. manifest、场景顺序、终止记账、结果文件和冻结 checkpoint 哈希全部通过审计。

P4 不与 G25/G26 合并统计。新 Gate 的论文主结论只能来自 P4；只有完成 P4 并通过上述全部
冻结成功/安全判据，B3 才以确认性证据替换 B2。Interaction share、raw steps、paired-success
steps、failure-penalized steps 和 switches 无论方向如何都必须报告，但不作为方法否决条件。
若 P4 未完成或成功/安全任一判据未通过，不能把未确认的 B3 结果写成确认性结论。

目标澄清登记时：P4 manifest 未生成，Dense test `[384:640]` 未读取，P4 未启动。

### 9.1 启动记录

目标澄清登记完成后，P4 manifest 已按冻结原始顺序 `[384:640]` 一次性生成。manifest
SHA-256 为 `73e273f4b156a6286d66a08646f02aa16c1845717c83a532559fee8c690302c1`；所选 256 场与
Dense train/validation、G25 和 G26 的 scene ID 与完整几何重叠均为 0。冻结 B3 checkpoint、
P3 结果和 continuation amendment 的 SHA-256 均通过启动前核验。

2304-episode 队列已于 `2026-09-10 18:51 CST` 后台启动，固定运行
`5A / B2 / B3 x 256 scenes x 3 repeats`。入口 PID 文件为仓库根目录 `.g27_p4_test.pid`，
实时日志为 `logs/p4_test/runner.log`，逐方法结果与精确续跑状态分别写入
`local_data/test/results/` 和 `local_data/test/checkpoints/`。基础设施中断只能从已完成场次后
精确续跑；不得重跑已完成场次、选择 attempt 或改变任何冻结项。

## 10. 六天执行与停止安排

~~~text
Day 1       实现 fresh-process 一步分支；完成 P0
Day 2       P0 通过才运行 P1
Day 3       训练唯一 B3；完成 Dense256 development
Day 4--5    仅在 P3 通过后运行新独立测试与统计
Day 5--6    修改论文、图和 Overleaf；保留一天缓冲
~~~

每个阶段只允许一次按本协议执行。失败后不通过增加样本、放宽门槛、切换 reward、改标签公式或
重复扫描阈值挽救结果。
