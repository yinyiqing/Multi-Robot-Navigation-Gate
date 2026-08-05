# G11-C 固定50场闭环Pilot

状态：`complete / retained B2 for replication`。登记日期：`2026-08-05`，完成日期：`2026-08-05`。

## 问题

只回答一个问题：G11-B2的student-rollout聚合是否比原A1 Gate改善实际闭环导航。
本实验不是最终200场准入，不读取navigation test或sealed test。

## 固定协议

- 数据：A1内部validation中预先固定的50场；
- 分层：standard-zero 13、standard-edge1 12、dense-zero 12、dense-edge1 13；
- 策略：5A、A1主seed Gate、B2聚合Gate；
- 重复：两个seed，每次三种策略使用完全相同的场景顺序；
- 顺序：repeat 1为`5A -> A1 -> B2`，repeat 2反向为`B2 -> A1 -> 5A`；
- A1滞回：on `0.28`、off `0.18`、hold 3 Gate帧；
- B2滞回：on `0.43`、off `0.33`、hold 3 Gate帧；
- 两个Gate均每2个环境步评估一次；
- 所有Actor、detector和Gate冻结，强制CPU串行运行。

50场包含25个0-edge与25个edge-1，standard和dense各25场。0-edge/edge-1仍是静态
名义路径拓扑，不是执行期真值输入。

## 固定步进可靠性修订

运行到第106场时，两次出现`gz world --multi-step 200`返回成功但仿真时间完全不推进。
故障分别发生在5A和B2运行中，因此与Gate或Actor无关；连续重发同一异步Gazebo
transport消息也无法恢复，只能重启`gzserver`。

`2026-08-05`起，固定步进改由加载在同一`gzserver`内的常驻world plugin提供ROS服务，
直接执行`World::Step(steps)`并返回完成确认。环境仍逐步检查目标仿真时间、最大超调、
paused状态和五车传感器时间戳。G11-C强制要求该服务存在，不允许静默回退CLI。

这次修订不改变物理步长、每个控制步的物理步数、场景、模型、seed或指标。旧的成功
episode已经通过相同的仿真时间和传感器校验，可以保留；发生故障的未完成episode从未
写入结果，恢复时仍从第7个B2场景重新开始。

修订后的验证：

- 空载完整world连续1500次请求，每次200步，共300000步；仿真时间精确增加
  `300.000000 s`且最终保持paused；
- 真实五车固定场景完成1场、99个控制步，五车传感器同步和结果落盘通过；
- 固定步进服务单元测试7项通过。

## 冻结输入

| 输入 | SHA-256 |
| --- | --- |
| 50场manifest | `1bf044cb5ff9d7d80c14d860d1108481af1d422cf403b26869f8b963012f0e91` |
| 5A Actor | `fa28855049b67b3ee44c66d55d4f14441fc7c521e5429862c75b152f7d5cacc5` |
| epoch-16 Actor | `6ec1942fcd497ab1cc2a85a5aaec8f524395dc21ff21a442dca243a52e917c0b` |
| detector | `0b914c0d090bbaba0a2be63c0d75d88580bf4d71778a7f1749c63989e3dbbd56` |
| A1 Gate | `d9b05d9f86e5bad4d2071c041187b618ebca6f1a3cc1f9c46e8b14b1a451537a` |
| B2 Gate | `fc59b4f783f7c5461ebb0239fab4b34896ad910ee78e7223e88d29ce9c3f5a52` |

## 判断

1. 先看B2相对A1的同场full success改善/退化，再看agent collision、unresolved和timeout；
2. 若B2在两次重复中都没有相对A1的闭环优势，默认停止追加聚合seed并保留A1；
3. 若B2方向成立，再做多seed复核和更大独立准入；
4. A1/B2还必须在edge-1相对5A有收益，并检查0-edge能力保持；
5. 单次峰值、训练集`0.750`和离线分类指标都不能替代该配对结果。

## 最终结果

六组运行全部完成，结果文件均为`50 x 17`且各有50个唯一scenario ID。两次重复只是
仿真seed，不是重新训练Gate。

| repeat | 5A full | A1 full | B2 full | B2 vs A1改善/退化 |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.66` | `0.70` | `0.78` | `10 / 6` |
| `2` | `0.62` | `0.66` | `0.76` | `10 / 5` |

合并100场中，5A/A1/B2的full success为`0.64/0.68/0.77`，agent success为
`0.888/0.920/0.934`，collision为`0.110/0.076/0.060`。B2在两次重复中均超过A1，
因此通过本pilot的“是否保留student聚合”停止条件。

该结果不是最终准入：B2的interaction Actor占比为`0.789`，平均步数为`51.25`，timeout
为`0.03`；A1对应为`0.690/37.23/0.02`。B2相对A1的单次McNemar exact分别为
`p=0.4545/0.3018`，样本不足以声称显著。分层上B2主要改善`standard-edge1`，在
`dense-edge1`略低于A1；`dense-zero`虽然保持full success，但interaction Actor占比
达到`0.940`。后续不得在这50场上继续调阈值。

最终汇总位于`local_data/summary.json`，运行日志已归档到仓库根目录
`logs/archive/validation/g11_c/`；本目录只保留兼容链接。

运行入口：

```bash
bash scripts/experiment.sh start gate-g11-c-pilot
bash scripts/experiment.sh status
bash scripts/experiment.sh stop gate-g11-c-pilot
```

运行期间的统一日志目录为仓库根目录的`logs/active/gate-g11-c-pilot/`。六组运行和
`summary.json`全部完成后，日志自动归档到`logs/archive/validation/g11_c/`；本目录的
`local_data/logs`和`local_data/pilot_runner.log`只保留兼容链接。
