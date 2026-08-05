# 运行日志索引

本目录只保存训练和验证的原始运行输出，不负责判定模型是否可用。实验状态、关键指标和模型选择应以
`experiments/EXPERIMENT_REGISTRY.md` 及对应实验目录的 `README.md` 为准。

## 目录约定

- `logs/*.log`：正在运行或尚未归档的日志。进程结束并完成结论登记后再移动。
- `active/`：预留给需要手工集中管理的运行中日志；当前为空。
- `archive/rejected/independent_dense_actor/`：独立 Dense Actor 路线，19 个历史日志。
- `archive/rejected/epoch16_full_episode/`：epoch-16 全程续训路线，2 个历史日志。
- `archive/rejected/full_actor_edge1/`：单冲突完整 Actor 路线，6 个历史日志。
- `archive/validation/independent_dense_actor/`：独立 Dense Actor 对照验证，4 个历史日志。
- `archive/diagnostic/g11_a0/`：G11-A0已有数据时序表示诊断日志。
- `archive/diagnostic/g11_a1/`：G11-A1 smoke、正式采集、审计和五seed离线训练日志。
- `archive/diagnostic/g11_b/`：G11-B在线时序Gate、student-rollout smoke与正式采集日志。
- `archive/rejected/g11_a1_duplicate_training/`：并发污染的主seed训练日志，不进入结果。
- `archive/rejected/g11_a1_launcher_artifacts/`：旧外层启动器残留日志，不进入结果。

整理日期：2026-08-05。G11-B1正式采集已结束；当前没有正在运行的G11采集或训练日志。

## 使用规则

1. 不移动仍被进程写入的日志。
2. 训练结束后，先在实验 README 中记录配置、结果和结论，再归档日志并更新引用路径。
3. 日志文件保持原名，便于从实验记录反查启动时间、种子和配置。
4. 日志末尾的 `ROSInterruptException: rospy shutdown` 常由正常清理或人工终止触发，不能单独作为实验无效的依据。
5. 原始日志体积较大且不纳入 Git；Git 只跟踪本索引和实验文档中的相对路径。
