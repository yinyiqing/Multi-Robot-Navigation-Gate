# epoch 17/18固定200场复核

状态：`运行中；2026-08-03 01:05启动，当前执行epoch 17基线`。

## 目的

确认50场pilot中full success从`0.300`提高到`0.420`的信号能否在更大的固定场景集上保持，并量化收益是否来自真实收尾改善而非碰撞/等待之间的随机波动。

## 冻结协议

- manifest：`dense_validation_monitor_v1`固定200场；
- seed：`20260803`；
- 基线：epoch 17冻结Actor，即原epoch-16局部专家；
- 候选：epoch 18短程全状态更新Actor；
- 两者均为单Actor全程独立执行，禁用5A接管、Oracle和Gate；
- 两者按相同scenario ID和相同顺序串行运行；
- 保存13列逐场统计，用`compare_actor_validation.py`做配对分析和冲突边分层。

## 判定

候选至少需要保持pilot中的方向：

1. full success提高，且逐场McNemar结果不支持“只有随机波动”的解释；
2. collision不能增加；
3. timeout必须下降，不能把碰撞重新换成等待；
4. 0-edge/低冲突分层不能出现明显普通导航退化；
5. 即使趋势通过，只要timeout仍高于`0.30`，Actor B仍不获得独立准入。

200场不用于选择新checkpoint或调参。未通过时停止当前配置，不追加seed追逐显著性。

## 脚本

- 串行启动：`scripts/start_validation_actor_b_epoch17_epoch18_200.sh`
- 停止：`scripts/stop_validation_actor_b_epoch17_epoch18_200.sh`
- 配对分析：`scripts/compare_actor_validation.py`

本地逐场数据和日志写入`local_data/`，完成后把聚合结果与决策写回本文件。
