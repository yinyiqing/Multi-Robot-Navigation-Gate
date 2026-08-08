# Dense 前 256 场同场 Pilot

状态：`registered / diagnostic pilot`。该实验不读取 sealed test，不更新任何 Actor 或
Gate，也不用于最终论文方法表。

## 目的

把 R2-10k 单 Actor 与 B2 Gate 从 G11-D2 的 0-edge/edge-1 准入集带到图片对应的 dense
多冲突场景，检查 D2 上的单 Actor优势是否只是简单场景造成的。

## 固定条件

- 来源 manifest：`datasets/fixed_v1/dense/validation.json.gz`；
- 取该 manifest 原始顺序的前 `256` 场；
- 五辆车、相同 ROS/Gazebo 物理参数、相同 evaluation seed `20260810`；
- 所有方法共享相同 scenario ID、顺序和终止条件；
- 运行入口：`scripts/start_g12_dense_first256_pilot.sh`。

## 方法

1. `5A` 全程；
2. `epoch-16` 全程；
3. `5A + epoch-16`，`2.0 m` 真值交互 Oracle；
4. `B2` 可部署 learned Gate；
5. `R2-10k` 参数匹配单 Actor。

Oracle 只作为不可部署上界，B2 才是当前可部署 Gate。最终分析按冲突边数分为
`0`、`1`、`2`、`3+`，并报告 full success、agent success、collision、unresolved、
timeout、平均步数、interaction action share 和逐场配对结果。

## 解释边界

256 场只用于快速诊断。如果结果值得进入最终比较，必须在同一冻结 dense validation
manifest 的完整 1000 场上重新运行所有方法。历史 D4 结果不能直接与本轮 B2/R2 结果拼表。
