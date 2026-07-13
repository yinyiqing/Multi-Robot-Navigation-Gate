# 集中式 Critic 核查

## 这条线在做什么

- 不先乱调 actor
- 先确认 critic 这条线有没有改对
- 看 critic 看到更完整的多车信息后，actor 解冻会不会更稳

## 现在这条线的定位

这条线已经完成第一轮关键判断：

- 旧版 critic 的多车信息确实不完整
- 补完整以后，前期效果会更好
- 但更长训练里还是会退化

所以这里的结论不是“critic 没用”，而是：

**critic 这条线有帮助，但它不是当前主要卡点的最终答案。**

当前主要卡点已经收缩成：

**单一 actor 在普通导航和密集交互之间，可能存在能力冲突。**

后续主线已转到：

- [04_保留专门化](/home/jiutian/Local-Critic-Multi-Robot-Navigation/experiments/04_保留专门化/README.md)

## 01 短对照

| 实验 | success_rate | collision_rate | full_success_rate |
| --- | --- | --- | --- |
| 新版 critic | 0.917 | 0.083 | 0.625 |
| 旧版 critic | 0.833 | 0.167 | 0.417 |

- critic 输入改法是有影响的
- 不是所有 actor 一解冻都会立刻崩

对应日志：

- [01_短对照](/home/jiutian/Local-Critic-Multi-Robot-Navigation/experiments/03_方法核查/集中式Critic/01_短对照)

## 02 中程验证

- 已跑完：
  - 修正后 `joint-action critic`
  - 同一 warm start
  - `5 epoch`
- 各轮验证：
  - Epoch 1: `success_rate=0.867`, `collision_rate=0.133`, `full_success_rate=0.542`
  - Epoch 2: `success_rate=0.875`, `collision_rate=0.133`, `full_success_rate=0.542`
  - Epoch 3: `success_rate=0.825`, `collision_rate=0.175`, `full_success_rate=0.417`
  - Epoch 4: `success_rate=0.783`, `collision_rate=0.200`, `full_success_rate=0.375`
  - Epoch 5: `success_rate=0.708`, `collision_rate=0.275`, `full_success_rate=0.208`
- 结论：
  - 前 2 个 epoch 明显好于旧版 critic
  - 但继续训练后还是出现了后期退化
  - 所以现在更准确的判断是：
    - critic 改法不是没用
    - 但它还不足以单独解决 actor 后期变差的问题

对应日志：

- [02_中程验证](/home/jiutian/Local-Critic-Multi-Robot-Navigation/experiments/03_方法核查/集中式Critic/02_中程验证)

## 已完成的代码核查

- 查过：
  - buffer 里有没有多车信息
  - critic 训练时到底看到了什么
  - 更新当前车时，其他车是不是只当背景
  - target 和 mask 有没有明显问题
- 其他车不会跟着当前车一起乱更新
- 这块不是主要 bug
- 但 target 动作构造原来不够标准

## 已做的小修正

- 已把 target 动作的噪声处理补标准：
  - 不再只给当前车加
  - 而是所有车统一处理

## 目录说明

- 这个目录只保留：
  - 一个总 README
  - `01_短对照/`
  - `02_中程验证/`

## 一句话总结

- 先发现旧版 critic 看到的多车信息不够完整
- 再把多车信息补进训练数据和 critic 输入里
- 然后确认到：critic 改对以后，前期会更稳，但更长训练里仍会退化
