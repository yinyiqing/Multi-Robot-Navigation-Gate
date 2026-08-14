# G12 参数匹配单Actor容量对照

状态：`P1/R1 archived / R2 cross-protocol only / R2B-30k frozen fair baseline`。
日期：`2026-08-08`。

`2026-08-14`新增[R2B 5A流程对照](R2B_5A_RECIPE_PROTOCOL.md)：从历史5A同一个3D2
Actor做函数保持扩宽，复制5A的五车standard、individual reward、fresh 24维Critic、
`20k` Actor冻结和`30k`总预算。R2B回答“相同训练流程下参数翻倍”的对照问题。
R2-10k的`0.760` full success保留为历史cross-protocol结果，但因额外课程与预算
不匹配，不进入论文公平容量baseline排名。

R2B现已完成。10k/20k/30k internal full success为`0.533/0.508/0.042`；10k仍是函数
等价的扩宽3D2，30k在约10k Actor更新后坍塌。项目接受退化作为对照结果，因此冻结
真正经过Actor更新的R2B-30k作为流程/预算匹配大Actor baseline，不使用冻结期自动best。
该对照证明同流程训练的加宽Actor优化不稳定，不证明大Actor容量不足。
完整结果见[R2B结果](R2B_5A_RECIPE_RESULTS.md)。

R2-S0加宽Actor从随机初始化完成100k单车broad训练，五次full success为
`0.983/1.000/1.000/0.992/1.000`，冻结best位于epoch 2。结果见
[R2-S0记录](R2_S0_RESULTS.md)。固定困难case诊断随后得到`22 pass / 0 borderline /
20 repair`，几何去重后为8个训练case。

S1首段已完成20,169 samples，但broad n1只有`69/120=0.575` full success并产生49次
timeout，未通过准入。该失败候选已回滚，不跑targeted、不作为S2 warm start、不追加预算。
动作与参数诊断确认这是8-case全网络更新造成的窄分布遗忘；详见
[R2-S1补课结果](R2_S1_REPAIR_RESULTS.md)。固定困难case不再阻塞多车课程；从S0 best
执行的[S2两车首段20k](R2_S2_N2_PROTOCOL.md)已经通过，10k/20k full success为
`0.9333/0.9250`并冻结10k best。完整结果见[R2-S2结果](R2_S2_N2_RESULTS.md)。随后
[S3三车首段](R2_S3_N3_PROTOCOL.md)的10k/20k full success为`0.8750/0.8833`，冻结20k
best；完整结果见[R2-S3结果](R2_S3_N3_RESULTS.md)。当前从该best进入
[S4五车首段20k](R2_S4_N5_PROTOCOL.md)。S4的10k/20k full success为`0.6667/0.6917`，
冻结20k best；完整结果见[R2-S4结果](R2_S4_N5_RESULTS.md)。进入R3前先执行
[五车底座配对准入](R2_N5_ADMISSION_PROTOCOL.md)。

20k配对准入的full success从5A的`0.5583`提高到`0.6917`，但timeout为`3/120`并超过
`0.020`上限，因此严格判定未通过。唯一预登记的10k fallback以`0.7000` full success、
`0` timeout和`20.39`平均步数通过全部五项准入，冻结为R2参考并允许进入R3设计。10k与
20k成功率逐场无显著差异，选择10k依据是约束和效率；完整结果见
[R2五车准入结果](R2_N5_ADMISSION_RESULTS.md)。

[R3 40k pilot](R3_PROTOCOL.md)已经完成：使用完整standard/dense train和strong子集构成
`standard/strong/dense/strong`四槽循环，只加载R2-10k Actor，fresh geometry Critic在
21k阈值后解冻Actor，随后采用归一化Q、safe-state anchor和梯度裁剪。R3-20k full success
为`0.667`，R3-40k降至`0.575`，未满足总体不低于R2-10k参考的准入条件；结果见
[R3结果](R3_RESULTS.md)。不启动R4，不扫描第二组配置。

P1在`40,007 agent samples`处按预注册规则早停。函数保持初始化通过，但Actor解冻后的
full success从`0.717`降至`0.050`并出现动作单侧饱和，因此该运行不能支持“参数匹配
单Actor容量不足”的结论。后续公平对照以
[G12-R修订路线](REVISED_PLAN.md)与[完整场景协议](FULL_SCENE_PROTOCOL.md)为准。本文件
以下内容保留P1原始预注册协议，其中`g11_a1_gate_v1`只适用于P1/R1诊断，不是正式
R2-R5大Actor的数据边界。

R1原宽度控制的full success同样从`0.717`降到`0.283`，因此P1不能归因于参数翻倍。
完整结果见[R1诊断](R1_DIAGNOSTIC.md)。P1/R1 checkpoint都不得作为R2 warm start。

## 研究问题

当前两个冻结Actor结构相同。G12只回答一个容量控制问题：

> 双Actor与Gate的收益，能否由一个拥有近似相同Actor参数总量的普通单Actor获得？

G12是论文对照，不替换`generalist-5a + interaction-epoch16 + Gate`主方法，也不重新开放
历史Actor路线搜索。Actor训练授权仅覆盖REVISED_PLAN登记的R1-R5；每次长跑仍需先冻结
具体manifest、seed、预算和停止条件。

## 参数匹配

| 模型 | 结构 | Actor参数 | 每步Actor MACs |
| --- | --- | ---: | ---: |
| 5A或epoch-16单个Actor | `24 -> 800 -> 600 -> 2` | `501,802` | `500,400` |
| 两个冻结Actor合计 | 两个上述网络 | `1,003,604` | `1,000,800` |
| G12加宽单Actor | `24 -> 1137 -> 855 -> 2` | `1,003,127` | `1,001,133` |

G12与双Actor相差`477`个参数，即`0.0475%`；Actor计算量相差`333` MACs，即
`0.0333%`。该比较只匹配Actor bank，不包含Gate感知前端和GRU，因此只能支持“24维
单帧单Actor的参数与Actor计算量对照”，不能声称匹配整个系统。

## 函数保持初始化

从冻结5A加载：复制原网络到加宽网络的左上子块；新增第一层特征到原第二层的连接为
零，新增第二层特征到输出层的连接为零，新增分支内部保留随机初始化。因此训练前输出在
数值精度内等价于5A，同时新增输出列在第一次Actor更新中具有非零梯度，不会成为永久
失活的零参数。

训练启动前由`scripts/audit_capacity_matched_actor.py`自动检查参数数目、真实5A checkpoint
输出等价和新增分支梯度；结果写入`local_data/initialization_audit.json`。

## 数据与训练

- train：`g11_a1_gate_v1/train.json.gz`，导航train内部`640`场；
- validation：同视图互斥`120`场；
- 四层均衡：`standard/dense x full-path 0-edge/edge-1`；
- train SHA-256：`a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026`；
- validation SHA-256：`e261a7afbac8f7341ab13609c2662a2824a0ff383789287ad7733290389cd99d`；
- seed：`20260810`；五车共享一个Actor，全程由该Actor控制；
- 不读取冲突pair、其他机器人真值、Gate标签或多冲突场景；
- individual navigation reward、原始24维fresh Critic；
- 前`20,000` agent samples冻结Actor，训练Critic；
- 每`20,000` samples在固定120场validation评估，pilot最多`4`个epoch，即`80,000`
  samples。

本pilot不直接宣称匹配epoch-16的完整`320,000`样本预算。只有在80k内出现明确正向趋势
且未触发停止条件，才允许在读取sealed test前登记G12-P2并扩展至320k；不得根据D2或
sealed test为加宽Actor扫描超参数。

若P1出现正向趋势，还必须补一个原宽度`24 -> 800 -> 600 -> 2`、相同数据和80k预算的
G12-P0，区分收益来自混合训练还是网络加宽；若P1没有超过运行内5A基线，则不为这一
区分追加长跑。

## 模型选择与停止

epoch 1位于Actor解冻边界，作为函数保持5A基线。后续checkpoint按120场validation的
`full_success`选择，同时满足：

1. 相对运行内最好结果full success下降`>=0.10`，停止；
2. agent success下降`>=0.08`，停止；
3. timeout增加`>=0.10`或绝对timeout达到`0.15`，停止；
4. 任一退化条件在epoch 2后出现一次即停止；
5. NaN、输出不等价、manifest/hash不符或D2未归档，运行无效并拒绝启动。

成功不是“超过5A”这么宽松。进入完整容量对照至少要求：full success高于运行内5A
基线，collision不升，timeout不升，并同时在0-edge与edge-1层保持非负趋势。

## 执行顺序

1. G11-D2生成`d2_summary.json`、归档日志并移除PID；
2. 运行无Gazebo单元测试与5A checkpoint输出审计；
3. 等待GPU 0至少有`8 GiB`可用显存且利用率不高于`20%`，随后固定使用CUDA训练；
4. 当前排队：`bash scripts/experiment.sh queue actor-g12-capacity-pilot`；D2已归档时也可直接
   `bash scripts/experiment.sh start actor-g12-capacity-pilot`；
5. 日志：`logs/active/capacity-matched-actor-g12-p1/`；
6. 完成后归档并决定停止或登记G12-P2。

G12完成前不读取sealed test。旧corrected edge-1完整Actor pilot仍是原宽度基础TD3失败
证据，不能替代本次参数匹配对照，也不能用其`best`文件作warm start。
