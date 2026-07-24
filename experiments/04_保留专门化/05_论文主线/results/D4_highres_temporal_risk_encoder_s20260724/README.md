# D4 High-resolution Temporal Risk Encoder

状态：`rejected pilot`。180-bin时序激光明显优于20-bin，但在互斥holdout上仍有约一半负样本被误报，不接入Actor或Gate。

## 协议

- policy：冻结5D，仅采集数据，不修改导航Actor。
- development：旧`sensor_probe` 30场。每个`risk band x preset`组的5场按4 train / 1 validation分配，共24/6场。
- test：新`sensor_holdout` 30场，与development scenario ID完全互斥，只在模型和validation阈值固定后读取一次。
- input：8帧前视180-bin lidar，距离截断为`6 m`，加上上一线速度/角速度。
- development输入由原XYZ使用与环境完全相同的扇区边界投影；test直接记录180-bin。
- models：共用逐帧`182 -> 64 -> 32`编码器；对照仅使用最后一帧，时序候选使用8帧GRU。
- label：privileged CPA/TTC，与20-bin probe一致。
- threshold：只在validation上以recall `>=0.80`时FPR最低选择。
- 准入线：test precision `>=0.70`、recall `>=0.80`、FPR `<=0.10`。

## 结果

| Model | Test precision | Test recall | Test FPR | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| Single-frame | 0.173 | 0.736 | 0.638 | 27 |
| 8-frame GRU | 0.217 | 0.781 | 0.511 | 17 |

GRU相对单帧提高了precision/recall并降低FPR，说明时序和高分辨率都有信号；但validation FPR已高达`0.518`，test FPR为`0.511`，远不具备可部署性。该结果不能通过增加epoch解决：GRU validation loss在第17轮最优，第29轮已明显恶化。

## 决策

拒绝当前“小样本 + 逐帧MLP + GRU”pilot，不在已读取的holdout上继续调hidden size、epoch或seed。这一结果不等于证明高分辨率lidar原理上无效：当前只有24个训练scenario和992个逐车样本，而逐帧MLP也没有显式利用角度局部结构。

若继续这条路线，必须作为新的正式数据阶段：扩大并重新冻结scenario-level train/validation/test，事先固定能利用角度局部性的轻量时空编码器。在此之前不再训练强交互Actor。

## 产物

```text
c3759981fd253396a30141af6f1940dce7bd76d0321b88b7099c50c16b00b869  gru.pth
e02c50479aafc4876e8ab295c2aa67b6a1589ba63c9e79ec55f64e584ec4abe9  mlp.pth
a2103168b2c7b017710b109006715cd5222dbedaefbd3f264c7a65275365ea93  summary.json
d7d5cb5b6aa4810b5b67d6e4c668cfdbc6cb09c347cd357251364ef45ec4103c  temporal_risk_highres_holdout_5d_s20260724_20260724_135504.jsonl.gz
bf2a4d4bd7ba607adad3eeb60d251c41168337ccb1907bc7a5196a50d64df047  test_temporal_risk_highres_holdout_5d_s20260724_20260724_135504.log
```
