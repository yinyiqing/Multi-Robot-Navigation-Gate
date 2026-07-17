# Fixed Scenario Datasets

这里只保留两个场景池：`standard` 和 `dense`。每个 JSON 场景都完整保存五台机器人的起点、目标、初始朝向、四个箱子、生成 seed、静态可行性和同步冲突指标。

`pilot/` 是管线测试样本，不能作为论文结果。正式数据按以下顺序产生：

```bash
source env.python.sh

python scripts/generate_fixed_scenarios.py \
  --preset dense \
  --output-dir /path/to/dense_candidates \
  --train 7200 --validation 1200 --test 2400 \
  --seed 20260717
```

上例先预留约 20% 候选。然后分别对每个 split 做 Gazebo reset 检查，并按正式目标数截取：

```bash
python scripts/validate_fixed_scenarios.py \
  --input /path/to/dense_candidates/test.json \
  --accepted /path/to/dense/test.json \
  --rejected /path/to/dense/rejected_test.json \
  --target-count 2000
```

验证命令需要已经 source ROS、catkin workspace 和 `env.python.sh`。它只检查传感器、初始碰撞和复位误差，不加载 Actor。

回放固定 split：

```bash
export DRL_MULTI_SCENARIO=manifest
export DRL_MULTI_MANIFEST_PATH=/path/to/dense/train.json
export DRL_MULTI_MANIFEST_SAMPLING=random  # 训练；测试使用 cycle
```

Gate 直接混合 standard 与 dense 的 train split，不生成第三类 gate 场景。
