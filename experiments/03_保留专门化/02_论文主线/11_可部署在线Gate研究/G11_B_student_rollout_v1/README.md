# G11-B Student-Rollout 数据聚合

状态：`smoke passed / formal collection ready`。日期：`2026-08-04`。

## 目的

修复G11-A1只在冻结5A轨迹上训练造成的部署分布偏移。固定A1主seed T1运行
navigation-train场景，在student实际访问状态记录可部署观测，并用训练期其他机器人
真值生成`2.0 m` Oracle标签。两个Actor、detector和初始Gate全部冻结。

本阶段不读取导航validation/test或sealed test，不更新Actor，也不把采集过程中的导航
结果当成独立评测成绩。

## 冻结输入

| 组件 | SHA-256 |
| --- | --- |
| 5A Actor | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| epoch-16 Actor | `6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b` |
| G0 detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |
| A1主seed T1 | `d9b05d9f86e5bad4d2071c041187b618ebca6f1a3cc1f9c46e8b14b1a451537a` |
| A1 train manifest | `a97534ed22d1b4b12951cd42b80d515037774830f38cb432926c0e9f50379026` |

只使用A1的640场train manifest。已有A1 5A shard与新student shard按scenario ID一一
对应；validation仍固定为A1的120场内部validation，不采集student validation，不接触
导航validation或test。

## 在线执行契约

- Gate：A1主seed T1，8帧GRU；
- Gate输入：76维本机状态/track特征，加两个Actor动作和动作差，共82维；
- 每2个环境步评估一次Gate，与A1采集步幅一致，中间保持当前mode；
- switch-on：`0.28`，switch-off：`0.18`；
- minimum hold：`3`个Gate评估帧；
- 每个环境步仍由当前mode对应Actor根据最新24维状态计算动作；
- 推理和采集强制CPU，`CUDA_VISIBLE_DEVICES=""`，进程优先级`nice 10`。
- 每个shard内嵌`student_run_metadata.json`的规范化JSON，记录上述策略和全部输入哈希。

阈值和保持时间只是首轮student数据采集策略，不是最终闭环调参结果。后续只能在允许的
小validation上冻结最终滞回参数。

## 数据聚合

第一轮聚合两种行为分布：

1. G11-A1冻结5A轨迹，640场；
2. G11-B初始student轨迹，同一640场。

两类轨迹上的标签都由相同`2.0 m` Oracle查询。第一轮不额外采集Oracle行为轨迹：
DAgger的关键是给student实际访问状态补teacher标签，而不是要求teacher自己再走一遍。
训练时必须按`source + scenario_id`分组并平衡两种来源，不能让较长的student轨迹仅凭
帧数支配损失。

## Smoke准入

先运行固定manifest首场：

1. 日志确认加载`T1`、sequence length `8`、evaluation stride `2`且device为CPU；
2. 生成且只生成1个student shard，scenario ID等于manifest首项；
3. shard split为`train`，格式、有限值、帧顺序和Oracle标签通过审计；
4. 日志包含Gate概率、切换次数和epoch-16动作占比；
5. 不出现Actor、detector、Gate或manifest哈希漂移。

任一项失败时不启动640场。smoke数据只进入`local_data/smoke/`，不进入正式聚合。

smoke已通过。运行时确认`T1 / sequence length 8 / evaluation stride 2 / CPU`，只生成
manifest第一场`dense-20260718003837-629bc18bb33c`的一个shard。该场共106个环境步，
Gate动作占比`0.338`、切换8次；导航结局为`5/5`到达且无碰撞，只作健全性检查。

| shard审计 | 数值 |
| --- | ---: |
| Gate frames | `98` |
| candidates | `423` |
| Oracle positive frames | `37` |
| visible / missed robots | `72 / 0` |
| dataset SHA-256 | `7ffd2f64e361b58345a37e34ba3c5a246d6a64d53d1987ccbe61cf4bf6172d6c` |

格式、有限值、帧顺序、manifest身份和内嵌run metadata均通过审计，因此授权正式640场
student采集。smoke shard不进入训练。

## 正式采集与停止

smoke通过后正式采集640场。每个scenario只允许一个shard，允许通过state文件恢复，
但正式目录中已存在完整shard时不得静默覆盖。

```bash
bash scripts/experiment.sh start gate-g11-b-smoke
bash scripts/experiment.sh status

bash scripts/experiment.sh start gate-g11-b-train
bash scripts/experiment.sh stop gate-g11-b-train
```

正式采集完成后先做全量审计，再实现来源平衡的聚合训练。离线结果至少不得破坏A1的
FPR约束；真正的继续/停止判断仍由后续固定闭环50场pilot决定，而不是本目录中的分类峰值。
