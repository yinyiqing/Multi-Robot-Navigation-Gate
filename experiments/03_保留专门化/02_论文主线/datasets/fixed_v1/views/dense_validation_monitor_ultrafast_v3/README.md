# Dense Validation Monitor Ultrafast v3

该view只用于独立Dense Actor训练期间的快速趋势判断和checkpoint保存，不替代完整1000场`dense/validation`。

- 源数据：`fixed_v1/dense/validation.json.gz`；
- 选取方法：在stride 20的20个固定偏移中，只根据冲突边分布选择offset 18，共50场；
- 选取过程在任何v2 Actor运行前完成，不读取模型成绩；
- monitor平均冲突边`2.480`，完整validation为`2.457`；
- monitor中至少2/3条冲突边的比例为`74.0%/46.0%`，完整validation为`74.7%/45.3%`；
- SHA-256：`b88d23ee4f509d4c4427303b3cab656e804a6708cb1311d747ca167b89ed18cc`。

一个50场epoch的波动不用于提前停止；至少比较连续3个短epoch。训练结束后只允许将monitor选出的少量checkpoint放到完整1000场validation上比较。

重建：

```bash
python3 scripts/build_dense_validation_monitor.py \
  --stride 20 \
  --start-index 18 \
  --dataset-id dense-validation-monitor-ultrafast-v3 \
  --output experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/dense_validation_monitor_ultrafast_v3/validation.json.gz
```
