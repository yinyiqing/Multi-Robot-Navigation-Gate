# G11-C 固定50场闭环Pilot

状态：`registered / pending`。登记日期：`2026-08-05`。

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

运行入口：

```bash
bash scripts/experiment.sh start gate-g11-c-pilot
bash scripts/experiment.sh status
bash scripts/experiment.sh stop gate-g11-c-pilot
```
