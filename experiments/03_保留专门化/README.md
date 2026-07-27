# 保留与专门化

状态：`current research branch`。具体方法、数据和实验顺序只以 [论文主线协议](02_论文主线/README.md) 为准。

## 为什么进入这条路线

历史实验反复出现同一现象：已有 Actor 在新 dense 课程中继续更新时，普通导航能力会被覆盖，而 high-interaction 能力也未稳定提高。

```text
5A / 5D 普通导航能力
  -> PAIR / THREE 渐进 dense：未形成稳定增益
  -> full Actor fine-tune：持续退化
  -> head-only：不崩，但表达力不足
  -> 5A + 5D switch/oracle：专家互补性不足
```

当前结论不再是训练两个完整的 standard/dense 场景专家。正式路线冻结 `generalist-5a` 作为普通导航 Actor，并冻结已经通过匹配重复验证的 `strong-interaction-5a-balanced` 作为条件交互 Actor。下一阶段只训练状态级 Gate；开始前必须先让本机传感器可靠地区分机器人与静态障碍。

## 目录逻辑

| 目录 | 状态 | 内容 |
| --- | --- | --- |
| [`01_历史诊断/`](01_历史诊断/README.md) | historical / failed | 解释为什么旧的覆盖训练和 5A+5D 切换不能作为最终方案 |
| [`02_论文主线/`](02_论文主线/README.md) | current | fixed-v1、条件交互 Actor、机器人感知和 Gate 的唯一当前协议 |
| [`90_未启用方案/`](90_未启用方案/README.md) | planned / inactive | 尚未进入实验流程的安全兜底备选 |

原来的五个平级目录实际混合了三种性质，现已归位：

| 原目录 | 现在的位置 | 含义 |
| --- | --- | --- |
| `01_冲突验证` | `01_历史诊断/01_冲突验证` | 证明继续训练会覆盖已有能力 |
| `02_双Actor切换` | `01_历史诊断/02_双Actor切换` | 证明旧 5A+5D 没有足够互补性 |
| `03_dense专家训练` | `01_历史诊断/03_dense专家训练` | 记录完整 dense expert、head 和 residual 的失败尝试 |
| `04_安全兜底` | `90_未启用方案/安全兜底` | 从未启用，不属于主线阶段 |
| `05_论文主线` | `02_论文主线` | 当前正式路线 |

逻辑顺序是：课程学习得到 5A/5D，历史诊断否定覆盖式训练和旧双 Actor 切换，当前主线重新构建互补的条件交互 Actor，再解决可部署 Gate。安全兜底只有在 Gate 仍无法覆盖极端风险时才考虑。

## 已确认的证据

- 5A 与 5D 在固定弱交互 validation 上能力等价，当前选择 5A 作为冻结普通导航 Actor。
- full Actor fine-tune 在 moderate fixed cases 上逐轮退化。
- head-only 限制了破坏，但没有超过冻结 5D。
- random dense 同时缩短了任务距离，不能证明策略擅长高交互。
- 五个 fixed moderate cases 能暴露同步冲突，但不是正式训练分布，只保留为 canonical held-out。
- 历史 5A + 5D 没有足够的 `specialist-only success`，不能直接支撑 gate。
- 正式条件交互 Actor 在冻结 5A 外围控制的匹配协议下，将强交互 full success 从 `0.421` 提高到 `0.700`；它不作为全程独立导航 Actor。
- 当前 oracle 按其他机器人真实距离切换，不可部署；现有 20 维激光又不能区分机器人和静态障碍，因此 Gate 感知是当前首要问题。

## 当前允许的工作

```text
D1  实现 conflict graph、standard/dense 生成器和 manifest 回放（已完成）
D2  完成 Gazebo 有效性筛选并冻结两个场景池的数据划分（已完成）
D3  完成 fixed-v1 generalist baseline 和交互分层（已完成）
D4  选择 5A 并训练条件交互 Actor，完成匹配重复 validation（已完成当前候选）
D5-G0  冻结两个 Actor，先解决机器人/静态障碍区分（当前）
```

当前禁止继续更新两个 Actor。机器人感知通过独立 validation 后，先建立可部署启发式 Gate，再训练 learned Gate；test 继续保持未读。

## 名称

新文档统一使用短 ID：

- 模型：`generalist-5a`, `strong-interaction-5a-balanced`, `interaction-gate`
- 场景池：`standard`, `dense`
- 当前阶段：`gate-robot-perception`

历史 artifact 原名不修改，映射见 [模型注册表](../../TD3/MODEL_REGISTRY.md)。
