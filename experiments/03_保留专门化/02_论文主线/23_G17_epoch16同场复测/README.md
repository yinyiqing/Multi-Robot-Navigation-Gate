# G17 epoch16同场复测

状态：`registered / running`。登记日期：`2026-08-17`。

## 问题

旧epoch16 Gate在0-edge/edge-1 D2上曾达到较高full success，但不能与当前epoch17
F-A1在完整场景上的结果跨数据集比较。本实验冻结所有模型，在G17同一完整五车manifest、
相同两个seed上复测：

- `epoch16 + A1`；
- `epoch16 + B2 student-rollout Gate`；
- `epoch16 + 2m privileged distance rule`。

现有G17的5A、epoch17+F-A1，以及G22的epoch17+2m规则结果直接复用，不重复运行。

## 固定协议

- manifest：`datasets/fixed_v1/views/g12_full_scene_selection_v1/validation.json.gz`；
- 120个固定完整五车场景，seeds为`20260824/20260825`；
- 每种策略240 episodes，共720 episodes；不读取sealed test；
- A1阈值`0.28/0.18`，B2阈值`0.43/0.33`，hold均为3，Gate stride均为2；
- 2m规则使用仿真机器人真值距离，只作不可部署机制诊断；
- CPU串行、固定物理步进，第二个seed反向运行以减小顺序影响；
- 不训练或修改Actor、Gate、detector和阈值。

## 判读

1. epoch16的2m规则若高于epoch17的2m规则，说明epoch16在当前完整场景中仍可能是更好的
   条件Actor；反之，Gate差距不能归因于epoch17 Actor本身。
2. epoch16+A1/B2若稳定高于epoch17+F-A1且效率可接受，才考虑把主线避障Actor换回
   epoch16；否则保留epoch17并诊断F-A1训练或切换时序。
3. 不以单个seed峰值选择方法，两个seed合并后做逐场配对检验。

运行日志先写入`logs/active/g17-epoch16-gate-comparison/`，成功后自动归档至
`logs/archive/validation/g17_epoch16_gate_comparison/`。
