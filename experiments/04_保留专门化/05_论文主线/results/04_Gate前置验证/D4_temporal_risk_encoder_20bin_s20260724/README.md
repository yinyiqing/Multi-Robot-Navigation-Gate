# D4 Temporal Risk Encoder: 20-bin Lidar

状态：`rejected representation candidate`。时序GRU未能从5D Actor原有的20维扇区最小距离中可靠恢复紧迫交互风险，不接入Actor或Gate。

## 协议

- scenarios：固定30场sensor probe，按`risk band x standard/dense`分层。
- split：按scenario互斥划分为18 train / 6 validation / 6 test。
- input：连续8帧；每帧20维前视lidar最小距离和上一动作，共22维，均为可部署本地观测。
- label：仅训练时使用其他机器人位姿计算CPA/TTC；限定前视范围、距离`<=4.0 m`、TTC `<=4.0 s`、CPA `<=0.9 m`。
- controls：比较只看最后一帧的MLP和同输入的8帧GRU。
- threshold：只在validation上以recall `>=0.80`时FPR最低选取，test不参与训练或选阈值。
- 准入线：test precision `>=0.70`、recall `>=0.80`、FPR `<=0.10`。
- seed：`20260724`。

## 结果

| Model | Test precision | Test recall | Test FPR | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| Single-frame MLP | 0.158 | 0.737 | 0.525 | 18 |
| 8-frame GRU | 0.159 | 0.684 | 0.486 | 9 |

GRU对FPR只有小幅改善，同时recall更低，两者均远低于准入线。GRU的validation loss在第9轮最优，继续训练至第21轮已恶化，因此不存在“只是epoch不够”的证据。

## 决策

拒绝“20-bin扇区最小距离 + GRU”这一具体表示，不增加epoch、不换seed调参、不接Actor。原始点云对真实危险机器人的覆盖率曾达`97.92%`，而20-bin时序编码失败，说明主要问题位于输入压缩后丢失目标身份和运动结构。

若继续观测路线，下一个独立候选只能验证更高角分辨率的连续激光帧，且必须使用新的scenario-level test或把现有test视为已消耗，避免在同一个6场test上反复选方法。

## 产物

```text
b3a2b44f24a403115449a9d7340f8f75be4559a8103d6685942f7e9ef042c7f9  gru.pth
56f4519c115a48bd2c0f7c26e8db0c157456a95fa6090938591c7703f9c7032f  mlp.pth
20b8709a3facb45c503a7f23dd8b2e40b64d52c58596a4635ba374c0c524878c  summary.json
```
