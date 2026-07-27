# Robot Perception V1

用途：只从导航 `fixed_v1/standard/train` 和 `fixed_v1/dense/train` 内部重新划分机器人感知数据。导航 validation/test 未进入感知开发。

## 固定划分

| 场景层 | train | validation | sealed test |
| --- | ---: | ---: | ---: |
| standard / weak (`0 edge`) | 927 | 116 | 116 |
| standard / interaction (`>0 edge`) | 1473 | 184 | 184 |
| dense / weak (`0 edge`) | 197 | 25 | 25 |
| dense / interaction (`>0 edge`) | 4603 | 575 | 575 |
| 合计 | 7200 | 900 | 900 |

- 划分 seed：`20260727`。
- 四层分别按 `80% / 10% / 10%` 划分，再在 split 内打乱。
- 三个 split 的场景 ID 互不重叠。
- `test.json.gz` 只完成封存；模型结构、阈值和训练协议冻结前不得采集或评估它。
- 场景中的 `navigation_split=train` 记录其来源，`split` 表示新的感知用途。

重新生成：

```bash
source env.python.sh
python scripts/build_robot_perception_views.py
```
