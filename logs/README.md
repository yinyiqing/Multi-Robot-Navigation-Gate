# 运行日志索引

本目录保存已经归档的历史运行输出，不负责判定模型是否可用。实验状态、关键指标和模型选择应以
`experiments/EXPERIMENT_REGISTRY.md` 及对应实验目录的 `README.md` 为准。

## 目录约定

- `logs/active/<run-id>/`：统一的运行中日志位置，不纳入Git；整组完成后自动归档。
- `logs/archive/validation/g11_c/`：G11-C闭环pilot完成后的日志归档位置。
- `logs/archive/validation/g12_dense_first256_pilot/`：G12 dense 前 256 场同场 pilot 的日志归档位置。
- `archive/rejected/independent_dense_actor/`：独立 Dense Actor 路线，19 个历史日志。
- `archive/rejected/epoch16_full_episode/`：epoch-16 全程续训路线，2 个历史日志。
- `archive/rejected/full_actor_edge1/`：单冲突完整 Actor 路线，6 个历史日志。
- `archive/validation/independent_dense_actor/`：独立 Dense Actor 对照验证，4 个历史日志。
- `archive/diagnostic/g11_a0/`：G11-A0已有数据时序表示诊断日志。
- `archive/diagnostic/g11_a1/`：G11-A1 smoke、正式采集、审计和五seed离线训练日志。
- `archive/diagnostic/g11_b/`：G11-B在线时序Gate、student-rollout smoke与正式采集日志。
- `archive/diagnostic/e2_ie2_multi_conflict/`：I-E2-M多冲突Actor训练、matched复测与诊断日志。
- `archive/rejected/g11_a1_duplicate_training/`：并发污染的主seed训练日志，不进入结果。
- `archive/rejected/g11_a1_launcher_artifacts/`：旧外层启动器残留日志，不进入结果。

整理日期：2026-08-05。G11-B1正式采集和G11-B2主seed训练均已结束；G11-C闭环pilot
当前写入`logs/active/gate-g11-c-pilot/`。

## 使用规则

1. 不移动仍被进程写入的日志。
2. 训练结束后，先在实验 README 中记录配置、结果和结论，再归档日志并更新引用路径。
3. 日志文件保持原名，便于从实验记录反查启动时间、种子和配置。
4. 日志末尾的 `ROSInterruptException: rospy shutdown` 常由正常清理或人工终止触发，不能单独作为实验无效的依据。
5. 原始日志体积较大且不纳入 Git；Git 只跟踪本索引和实验文档中的相对路径。
