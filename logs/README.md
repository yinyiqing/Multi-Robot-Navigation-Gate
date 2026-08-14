# 运行日志索引

本目录保存已经归档的历史运行输出，不负责判定模型是否可用。实验状态、关键指标和模型选择应以
`experiments/EXPERIMENT_REGISTRY.md` 及对应实验目录的 `README.md` 为准。

## 目录约定

- `logs/active/<run-id>/`：统一的运行中日志位置，不纳入Git；只允许保留仍有对应活进程、
  且仍在写入的run。整组完成或中止后必须移出。
- `logs/archive/validation/g11_c/`：G11-C闭环pilot完成后的日志归档位置。
- `logs/archive/validation/g12_dense_first256_pilot/`：G12 dense 前 256 场同场 pilot 的日志归档位置。
- `archive/rejected/independent_dense_actor/`：独立 Dense Actor 路线，19 个历史日志。
- `archive/rejected/epoch16_full_episode/`：epoch-16 全程续训路线，2 个历史日志。
- `archive/rejected/full_actor_edge1/`：单冲突完整 Actor 路线，6 个历史日志。
- `archive/validation/independent_dense_actor/`：独立 Dense Actor 对照验证，4 个历史日志。
- `archive/diagnostic/g11_a0/`：G11-A0已有数据时序表示诊断日志。
- `archive/diagnostic/g11_a1/`：G11-A1 smoke、正式采集、审计和五seed离线训练日志。
- `archive/diagnostic/g11_b/`：G11-B在线时序Gate、student-rollout smoke与正式采集日志。
- `archive/diagnostic/g11_f_epoch17_gate/`：epoch-17 Gate的离线A1重建及后续诊断日志。
- `archive/validation/g11_f_epoch17_gate_pilot/`：epoch-17 F-A1/F-B2固定50场、两个重复的
  闭环选择pilot日志；F-A1被选中，F-B2因成功率和效率代价未被选中。
- `active/g11_f_epoch17_gate_r2_pilot/`：在相同F-C manifest和两个repeat上补跑R2-10k
  参数匹配大Actor的实时日志；完成后归档到`archive/validation/`同名目录。
- `archive/diagnostic/e2_ie2_multi_conflict/`：I-E2-M多冲突Actor训练、matched复测与诊断日志。
- `archive/validation/e2_recovery_oracle_epoch16/`：E2 recovery-oracle 120场完整评测日志。
- `archive/validation/g12_dense_full1000_partial_20260811/`：未完成的dense 1000场并行评测；
  5A/B2/避障Actor/oracle/R2-10k实际只有`300/152/38/159/216`场，只作partial诊断。
- `archive/aborted/current_generalist_5a_local_critic_20260809/`：旧5A起点local-Critic中止运行。
- `archive/aborted/current_generalist_3d2_source_local_critic_20260809/`：旧3D2起点local-Critic
  中止运行。
- `archive/diagnostic/current_generalist_from_e2_local_critic_20260811/`：E2全场local-Critic
  四轮已完成但被拒绝的诊断日志。
- `archive/training/avoidance_actor_e17_e20/`：原避障Actor完整状态续训epoch 17-20日志；
  epoch 17未通过原效率硬门槛、后按修订主指标规则入选，18-20在internal validation后拒绝。
- `archive/validation/avoidance_actor_matched_admission/`：原epoch 16与候选epoch 17在两个
  seed、同一120场manifest上的`480/480`场配对准入日志；epoch 17未通过原效率门槛，后按
  full-success主指标规则冻结进入Gate重训。
- `archive/rejected/g11_a1_duplicate_training/`：并发污染的主seed训练日志，不进入结果。
- `archive/rejected/g11_a1_launcher_artifacts/`：旧外层启动器残留日志，不进入结果。
- `archive/rejected/avoidance_actor_matched_admission_cleanup_bug_20260814/`：配对准入首轮
  worker未先停止`roslaunch`造成的候选Actor启动失败日志；没有episode或结果数组。
- `archive/rejected/g11_f_epoch17_gate_smoke_nounset_20260814/`：首次epoch-17 Gate smoke
  在加载ROS环境前启用`nounset`而退出；未启动Gazebo且没有生成shard。

整理日期：2026-08-14。避障Actor epoch 17-20续训和独立matched admission均已完成并
归档，`logs/active/`中不保留该实验的文件。

## 使用规则

1. 不移动仍被进程写入的日志。
2. 训练结束后，先在实验 README 中记录配置、结果和结论，再归档日志并更新引用路径。
3. 日志文件保持原名，便于从实验记录反查启动时间、种子和配置。
4. 日志末尾的 `ROSInterruptException: rospy shutdown` 常由正常清理或人工终止触发，不能单独作为实验无效的依据。
5. 原始日志体积较大且不纳入 Git；Git 只跟踪本索引和实验文档中的相对路径。
6. 空文件、没有进入episode的重复启动残留和失效PID可以删除；有episode或结构化结论的
   唯一日志必须先归档，不能仅为清空`active`而删除。
