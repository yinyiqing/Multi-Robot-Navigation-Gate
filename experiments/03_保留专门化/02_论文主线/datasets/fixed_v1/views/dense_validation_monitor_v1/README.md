# Dense Validation Monitor v1

该view只用于独立Dense Actor训练期间的高频checkpoint选择，不替代完整1000场 `dense/validation`。

- 源数据：`fixed_v1/dense/validation.json.gz`；
- 选取方法：从冻结顺序的index 0开始，每5场取1场，共200场；
- 选取规则在任何Actor运行前固定，不读取模型成绩；
- monitor平均冲突边 `2.415`，完整validation为 `2.457`；
- monitor中至少2/3条冲突边的比例为 `74.0%/43.5%`，完整validation为 `74.7%/45.3%`。

训练结束后，只允许用完整1000场validation比较monitor挑选出的少量checkpoint。sealed test保持未读。

重建：

```bash
python3 scripts/build_dense_validation_monitor.py
```
