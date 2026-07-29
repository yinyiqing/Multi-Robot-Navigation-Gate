# Dense Validation Monitor Fast v2

该view只用于独立Dense Actor训练期间的快速趋势判断和checkpoint保存，不替代完整1000场`dense/validation`。

- 源数据：`fixed_v1/dense/validation.json.gz`；
- 选取方法：从冻结顺序的index 0开始，每10场取1场，共100场；
- 选取规则在任何v2 Actor运行前固定，不读取模型成绩；
- monitor平均冲突边`2.460`，完整validation为`2.457`；
- monitor中至少2/3条冲突边的比例为`75.0%/45.0%`，完整validation为`74.7%/45.3%`；
- SHA-256：`672b21bfc67ddab8c84ab614431ceed73fad17ff597b54f2b2db3f3900163300`。

一个100场epoch的波动不用于提前停止；至少比较连续3个短epoch。训练结束后只允许将monitor选出的少量checkpoint放到完整1000场validation上比较。

重建：

```bash
python3 scripts/build_dense_validation_monitor.py \
  --stride 10 \
  --dataset-id dense-validation-monitor-fast-v2 \
  --output experiments/03_保留专门化/02_论文主线/datasets/fixed_v1/views/dense_validation_monitor_fast_v2/validation.json.gz
```
