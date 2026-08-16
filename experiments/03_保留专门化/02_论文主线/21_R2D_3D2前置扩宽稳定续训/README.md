# R2D：3D2前置扩宽大Actor稳定续训

状态：`registered / running`。登记日期：`2026-08-16`。

## 研究问题

R2D只回答：从与5A相同的3D2共同起点，在进入五车5A阶段之前将Actor函数保持扩宽到两个
Actor的参数量，并在整个五车阶段保持该结构时，能否得到一个实际更新且不坍塌的大Actor。
它不以低于F-A1为训练目标；若最终超过F-A1，必须如实报告。

## 起点

- source：`capacity_wide_r2b_5a_recipe_n5_seed20260823_best.pt`；
- source SHA-256：`6e4f47e2665d5040f3962e0283000ecdf4c9a6fb03b00ce3b2c45eda95b5ec60`；
- 结构：`24-1137-855-2`，`1,003,127`个Actor参数；
- 血缘：3D2 Actor函数保持扩宽，不经过训练完成的5A；
- 断点：`10,086` agent samples、10k Critic校准、Actor optimizer state为空；
- replay、Critic optimizer、场景计数和物理评测协议完整保留。

该起点的Actor仍与3D2函数等价，不是最终容量baseline。它只用于避免重复已经完成的前10k
Critic校准。

## 固定训练协议

与R2B保持一致：五车procedural `standard`、individual reward、24维Critic、无local
Critic、无dynamic reward、Actor/Critic LR=`2e-6/2e-5`、batch=40、同seed和同120场
validation。Actor在绝对`20,000` samples前冻结，总预算止于`30,000` samples。

相对R2B只增加三个预登记稳定项：

1. Actor Q目标按当前batch的Q绝对均值归一化；
2. 以R2B-10k Actor为固定参考，加入权重`1.0`的全状态动作MSE锚定；
3. Actor梯度范数裁剪为`1.0`。

Critic训练目标、reward、数据分布和部署输入均不改变。20k只检查Critic校准边界；30k是
唯一实际Actor更新候选。若30k相对20k出现full success下降超过`0.05`、collision增加超过
`0.03`、timeout增加超过`0.02`、动作持续饱和或Actor参数未改变，R2D判为失败，不追加
预算、不调anchor权重、不扫描checkpoint。

## 执行

```bash
bash scripts/start_training_g12_r2d_pre5a_stable.sh
```

实时日志：`logs/active/capacity-wide-g12-r2d-pre5a-stable/`。R2D不修改5A、epoch17、F-A1、
F-B2或sealed test。
