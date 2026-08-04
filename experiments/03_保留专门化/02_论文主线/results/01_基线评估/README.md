# 基线评估

这里保存冻结模型的参考成绩，不包含当前方法训练。

| 实验 | 状态 | 用途 |
| --- | --- | --- |
| `D3_generalist_baseline` | complete | 旧随机 `standard-5` 上的 5D 基线 |
| `D3_fixed_v1_generalist_baseline` | complete | fixed-v1 standard/dense test 上的 5D 基线 |
| `D3_fixed_v1_generalist_validation` | complete | fixed-v1 validation 及冲突边分层基线 |
| `D4_warmstart_baseline_5a_vs_5d_s20260725` | complete | 强交互训练前 5A/5D 同协议比较 |

这些结果用于定义难度和历史对照。结合`05_当前冻结方案/`中的弱交互配对验证，当前已
冻结5A为普通导航Actor；5D只保留为历史基线。
