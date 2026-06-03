# 多机器人课程学习主线

本目录用于归档课程学习主线的配置、case 定义和后续阶段实验说明。

## 目录规划

- `cases/`：跨实验复用的课程 case 文件，只描述起点、目标和可选障碍盒。
- 诊断目录：只保留 README 摘要；原始日志、压缩日志、`.npy` 和中途废弃产物完成复盘后清理。
- 阶段结果：按机器人数量归档到对应实验目录，例如五车结果继续放在 `experiments/多智能体/5智能体/` 下。
- 根目录 `logs/`：只放正在运行的日志，完成后仍按既有规则归档。

## 当前阶段设计

| 阶段 | case 文件 | 目的 |
| --- | --- | --- |
| Stage 1 | `cases/stage1_single_local_cases.json` | 先用 1 agent 复现并修复目标隔墙、近目标捕获、近障局部停滞 |
| Stage 1b | `cases/stage1b_single_near_goal_sidewall_cases.json` | 单独复现近目标捕获、侧墙诱发振荡、目标贴墙恢复 |
| Stage 2a | `cases/stage2_three_dense_cases.json` | 用 3 agents 插入中等密集交互课程，避免从单车直接跳到五车密集 |
| Stage 2b | `cases/stage2_dense_multi_cases.json` | 用 5 agents 手动制造交错、密集目标、密集起点等多车交互样本 |
| Stage 3 | 暂未定义 | 如果 Stage 2 在手动密集 case 有提升但随机标准场景不稳，再进入五车随机 fine-tune |

## 使用原则

- 先跑 targeted test，确认 case 能复现当前失败模式，再训练。
- Stage 1 不讨论协同，只处理基础局部导航能力。
- Stage 2 先保持 individual reward，不急着重新引入 reward sharing 或 local critic。
- 阶段切换时 actor/critic warm-start，学习率和探索噪声通过环境变量调低或重置到中等水平。
- 同时测试手动课程 case 和标准随机 case，防止过拟合课程。

## 环境变量

课程 case 通过多车环境统一入口运行，包括 1 agent 和 5 agents：

- `DRL_MULTI_SCENARIO=curriculum`
- `DRL_MULTI_CURRICULUM_CASES=/path/to/cases.json`
- `DRL_MULTI_CURRICULUM_SAMPLING=cycle|random`

训练阶段可调参数：

- `DRL_MULTI_ACTOR_LR`
- `DRL_MULTI_CRITIC_LR`
- `DRL_MULTI_EXPL_NOISE`
- `DRL_MULTI_EXPL_MIN`
- `DRL_MULTI_EXPL_DECAY_STEPS`

## 当前判断

Z0-M 已经说明，继续盲调 reward/critic 没有稳定超过 baseline。课程学习主线先把问题拆成可复现的局部导航缺陷和多车密集交互缺陷，再逐阶段训练和评估。

Stage 1 单车课程的少量 targeted test 不足以证明基础局部导航缺陷已经解决。Stage 1 best 直接迁移到五车标准随机场景后出现高频左右摆动和超时，因此 Stage 2a 暂停推进，先补 Stage 1b 的近目标和侧墙振荡诊断。

当前保留原则是：case 定义和复盘摘要保留；错误启动、难度跳跃过大、reward shaping 冲突和已完成复盘的原始运行产物清理掉，避免后续误用。
