# 保留与专门化

状态：`current research branch`。唯一当前协议见[论文主线](02_论文主线/README.md)。

## 当前问题

普通导航Actor已经可靠，但直接在dense任务上继续更新会覆盖已有能力。项目因此不再
训练standard/dense两个完整场景专家，而研究：

> 单冲突中学到的局部避让修正，能否通过Gate反复调用，组合解决未训练的多冲突。

当前系统为：

```text
Actor A = 冻结5A
Actor B = 冻结5A + epoch-16指导的单冲突Residual
Gate    = 冻结A/B后的可部署状态选择器
```

Actor B必须能全程独立导航，但它和Gate的训练都只能看到0-edge与单冲突场景。多冲突
只用于零样本validation和最终test。

## 目录

| 目录 | 状态 | 内容 |
| --- | --- | --- |
| [`01_历史诊断/`](01_历史诊断/README.md) | historical / failed | 覆盖训练、旧双Actor和旧dense expert为何失败 |
| [`02_论文主线/`](02_论文主线/README.md) | current | 数据边界、Residual Actor B、Gate和组合泛化协议 |
| [`90_未启用方案/`](90_未启用方案/README.md) | inactive | 未启用的安全兜底，不与主线并行 |

## 已确认

- 5A在固定0-edge validation上的full success为`0.875`。
- epoch-16在匹配局部调用协议下将full success从`0.421`提高到`0.700`。
- 真值Gate在未见`edge>=2`结构上相对5A提升`0.249` full success。
- 可部署Gate在独立exact-edge-2确认上提升`0.080`，但未通过统计和收益恢复门槛。
- epoch-16整网全程续训在固定200场使full success从`0.305`降到`0.260`，已拒绝。
- 旧Residual只做过`5D + 零初始化Residual`；没有做过当前的
  `冻结5A + epoch-16指导Residual`。

## 当前只做

执行[单冲突Residual Actor B协议](02_论文主线/09_单冲突Residual组合主线/05_单冲突Residual_ActorB/README.md)：

1. 审计教师动作差和Residual表达范围；
2. 用教师组合与学生访问状态初始化Residual；
3. Actor B全程独立闭环训练；
4. 通过固定200场后再冻结A/B训练Gate；
5. 最后在未见多冲突拓扑上验证组合泛化。

禁止恢复完整dense Actor重训、epoch-16整网续训或5D零初始化Residual路线。
