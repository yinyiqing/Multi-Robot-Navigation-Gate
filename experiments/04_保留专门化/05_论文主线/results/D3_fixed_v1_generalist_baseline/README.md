# D3 Fixed-v1 Generalist Baseline

状态：`complete`。冻结的 `generalist-5d` 已在全部 fixed-v1 test manifest 上完成测试。

## 总结果

| Pool | Episodes | Agent success | Collision | Unresolved | Full success | Timeout episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 1000 | `4314/5000 = 0.8628` | `672/5000 = 0.1344` | `14/5000 = 0.0028` | `575/1000 = 0.5750` | `14/1000 = 0.0140` |
| dense | 2000 | `6976/10000 = 0.6976` | `3022/10000 = 0.3022` | `2/10000 = 0.0002` | `559/2000 = 0.2795` | `2/2000 = 0.0010` |

Episode bootstrap 95% CI，20,000 次重采样，seed `20260718`：

| Pool | Agent success | Collision | Unresolved | Full success |
| --- | ---: | ---: | ---: | ---: |
| standard | `[0.8512, 0.8742]` | `[0.1230, 0.1460]` | `[0.0014, 0.0044]` | `[0.5450, 0.6060]` |
| dense | `[0.6869, 0.7082]` | `[0.2914, 0.3130]` | `[0.0000, 0.0005]` | `[0.2595, 0.2990]` |

计数闭合：

```text
standard: 4314 + 672 + 14 = 5000
dense:    6976 + 3022 + 2 = 10000
```

## 按同步冲突边分桶

| Pool | Conflict edges | N | Agent success | Collision | Full success | Mean task distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 0 | 391 | 0.9540 | 0.0435 | 0.8159 | 1.604 m |
| standard | 1 | 424 | 0.8420 | 0.1552 | 0.4811 | 1.719 m |
| standard | 2 | 144 | 0.7333 | 0.2653 | 0.2917 | 1.857 m |
| standard | 3 | 32 | 0.7063 | 0.2812 | 0.2500 | 1.910 m |
| standard | 4+ | 9 | 0.5111 | 0.4889 | 0.2222 | 2.090 m |
| dense | 0 | 71 | 0.9915 | 0.0085 | 0.9577 | 1.159 m |
| dense | 1 | 400 | 0.8265 | 0.1730 | 0.4825 | 1.293 m |
| dense | 2 | 605 | 0.7336 | 0.2664 | 0.2893 | 1.392 m |
| dense | 3 | 521 | 0.6388 | 0.3612 | 0.1651 | 1.482 m |
| dense | 4+ | 403 | 0.5400 | 0.4596 | 0.0918 | 1.613 m |

## 结论

1. fixed standard 的 full success `0.5750` 与旧随机 standard 的 `0.5690` 接近，说明固定数据没有明显改变普通 benchmark 的总体难度。
2. dense 的 full success 降到 `0.2795`，collision 上升到 `0.3022`，为 specialist 留出了明确提升空间。
3. 两个 pool 内部都表现出随冲突边数增加而单调退化的趋势，支持使用 synchronized interaction graph 描述难度。
4. dense 的 0-edge 场景反而很简单，说明“缩短任务”确实降低难度；dense 总体更难来自它包含更多多边冲突，而不是距离本身。
5. 不能删除 5D 表现差的有效 test case。后续 specialist 和 gate 必须逐场复用相同 scenario ID。

standard 的 4+ 桶只有 9 场，只能作为趋势，不能单独做强统计结论。

## 辅助统计

| Pool | Mean episode steps | Mean reward | Mean final distance | Success histogram 0..5 |
| --- | ---: | ---: | ---: | --- |
| standard | 42.174 | 99.618 | 0.3948 m | `[1, 8, 37, 159, 220, 575]` |
| dense | 25.891 | 58.666 | 0.4737 m | `[17, 102, 275, 659, 388, 559]` |

## 复现信息

- Actor：`TD3_velodyne_multi_v4_curriculum_stage2_to_5d_geo_critic_from_5a_guarded_best`
- Actor SHA-256：`51e31499a8e7ec88a80e2ff657f74b3d14a64168c2afab9361c6cbe770818ff0`
- 测试启动代码：`d1d0836`
- standard manifest SHA-256：`699bc7237debadecb59400adafc075f20a4cc1fe5642ba82b74196d221ab35f8`
- dense manifest SHA-256：`03a744048102d7310db026e399e41c4ce664ed31b180438b2a2b519c78133eab`
- outcome：collision-priority mutually exclusive

归档文件：

- `standard_1000ep.log`：由 `logs/test_fixed_v1_standard_5d_20260717_233639.log` 移入的完整日志。
- `dense_2000ep.log`：由 `logs/test_fixed_v1_dense_5d_20260717_233639.log` 移入的完整日志。
- `standard_1000ep.npy` / `dense_2000ep.npy`：逐 episode 数据。
- `summary.json`：结构化统计、列定义和哈希。
