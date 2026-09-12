# 当前独立协议对照补测

状态：`registered / pilot pending`。

本目录补齐 Table 1 所需的对照实验。所有新增结果必须使用与 G34 独立主评测相同的
Dense test manifest 切片 `[640:896]`、固定物理步长、五车场景、终止规则和环境 seed
`20260914, 20260915, 20260916`。不得读取旧 G25/G26 结果作为当前协议结果，也不得根据
pilot 或完整队列结果调参。

## 当前目标

已有同协议结果：

- Navigation Actor：G34 independent test，256 场景 × 3 repeat；
- Reward-aware router：G34 independent test，256 场景 × 3 repeat。

待补的五个方法：

1. `always_on_interaction`：始终执行冻结 Interaction Actor；
2. `min_lidar_rule`：最小 LiDAR 距离规则；
3. `ttc_cpa_rule`：TTC/CPA 规则；
4. `capacity_matched_actor`：参数匹配的单 Actor，使用 R2B 流程匹配冻结 checkpoint；
5. `proximity_only_router`：旧 proximity-only Gate，但在当前 manifest/seed 下重新运行。

2 m 距离特权规则是不可部署 oracle，不在本补测中运行，也不进入 Table 1。

## 执行顺序

1. 先运行 pilot：默认每个方法 16 个场景、seed `20260914`。检查结果形状、场景顺序、终止
   记账、无 `timeout/unresolved` 异常记账和控制器是否实际生效。
2. pilot 通过后，将 `CONTROL_TARGET_EPISODES=256`、`CONTROL_SEEDS='20260914 20260915 20260916'`
   运行完整队列。脚本按方法和 seed 顺序串行运行，并支持基础设施中断后的精确续跑。
3. 只有五个方法全部完成并通过审计后，才生成统一统计并更新 Table 1。任何失败或未完成
   的结果都保留在本目录，不用占位数字填表。

## 冻结输入

- manifest：`34_双头RewardAwareGate/local_data/independent_test/manifest/dense_test_640_896.json.gz`；
- frozen Navigation Actor：`TD3_velodyne_multi_v4_curriculum_stage2_to_5a_shared_from_3d2_guarded_best`；
- frozen Interaction Actor：`interaction_focused_actor_from_5a_fullstrong_balanced_formal_s20260726_epoch_016`；
- G0 detector：`results/06_Gate开发/D5_G0_robot_detector_v1/local_data/model/pilot_v1/best.pt`；
- proximity-only checkpoint：`11_可部署在线Gate研究/G11_B_student_rollout_v1/local_data/training/seed20260804/any/T1/best.pt`；
- capacity-matched checkpoint：`capacity_wide_r2b_5a_recipe_n5_seed20260823_best`（运行器会追加 `_actor.pth`）。

所有 checkpoint、manifest、脚本和原始 `.npy` 的哈希在完整队列结束后写入
   `local_data/completion_n<N>.json`，统计脚本不得覆盖原始输出（`<N>` 为该次运行的目标场景数）。
