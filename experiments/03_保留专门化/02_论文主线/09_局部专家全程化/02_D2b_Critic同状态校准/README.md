# D2b Critic同状态校准

## 目的

检查修正后的D2b fresh Critic能否用于epoch-16 Actor B解冻。D2b只保存当步受控ego的transition，消除了旧D2中未受控车辆样本的隐藏动作混杂。

## 协议

- checkpoint：`independent_dense_actor_simple_td3_hparam_d2b_s20260801_latest.pt`；
- checkpoint SHA256：`cc817d0af7d9fe853f1a3ea35ba6e3772a1b2adfa7f288a115b2b9d00231314d`；
- 2个固定dense scenario，每个最多4个锚点、每个锚点2台ego；
- 固定其他车辆动作和ego转向，只替换ego第一步线速度为`[-1,-0.5,0,0.5,1]`；
- 每个分支运行12步，比较Critic Qmin和真实N-step return的组内动作排序；
- 只使用通过位置、速度、Actor state重放一致性检查的分支。

## 结果

- 总分支：`40`；
- 可重复分支：`16`；
- 可比较状态组：`4`；
- 可比较动作对：`26`；
- 排序一致：`15/26 = 0.577`；
- Qmin与N-step target MAE：`104.53`；
- Qmin bias：`-45.42`。

可重复分支中，Critic Qmin全部位于`-0.64~-0.17`，真实12步return位于`-95.76~+136.13`。Critic输出主要随线速度单调增加，无法可靠区分碰撞和成功分支。

## 结论

D2b不通过Actor解冻准入。该结果不是单纯样本数不足：Q动态范围塌缩和全局速度单调偏好说明fresh Critic仍系统性欠拟合。停止继续基于D2b解冻Actor，也不增加同配置训练轮数。

Actor B改为从epoch-16完整训练checkpoint分叉，复用其已经产生有效局部避障策略的87维Critic和32万条replay；D2b只保留为失败对照。本地完整分支记录位于`local_data/critic_calibration.json`。
