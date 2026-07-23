# Strong Interaction Curriculum V1

这是从 fixed-v1 standard/dense 原始 train/validation split 确定性派生的强交互课程。只使用策略无关的同步名义路径最小间距分层，不修改原始场景，也不读取任何模型成绩。

## 三阶段课程

| Stage | Deep | Close | Margin | 目的 |
| --- | ---: | ---: | ---: | --- |
| 1 | 0 | 256 | 128 | 先学习中等冲突，同时保持不过度干预 |
| 2 | 256 | 256 | 128 | 引入深度冲突 |
| 3 | 512 | 128 | 128 | 形成强交互专门化并保留回归样本 |
| validation | 60 | 40 | 40 | 三阶段固定使用同一评测集 |

每个风险档在 standard/dense 来源池间等量抽取。三个训练阶段使用嵌套的确定性样本顺序；train 与 validation 继承原始 split，scenario ID 无交叉。

Actor和Critic不读取 `interaction_band`。该标签只用于构造课程和输出分层validation指标。

## Stage 1 协议

- 独立复制5D Actor/Critic完整warm-start。
- Actor保持原始单帧 `24 -> 800 -> 600 -> 2` 结构，全部参数可训练。
- reward保持 `0.8 self + 0.2` 距离加权邻居奖励。
- 前20000 agent samples只适配Critic，Actor从21000 samples后解锁。
- epoch 1是同协议冻结5D基线，epoch 2才包含Actor更新。
- 只有close提高且margin无明显退化，才允许进入Stage 2。

## 复现

```bash
python scripts/build_strong_interaction_curriculum.py \
  --output experiments/04_保留专门化/05_论文主线/datasets/fixed_v1/views/strong_interaction_curriculum_v1
```

SHA-256：

```text
ba165382e6f95ab2618b3f0551937bb0d108fbf493733e03c0785220941e8e10  stage1_train.json.gz
3fa59eee49c2e2ba544e9942f19c0992b56414515983a27827a98508deba5944  stage2_train.json.gz
ce113b1361a7ae3ee657bfbe2e0a7a63a631e89854b0435f53e7b627af3e193e  stage3_train.json.gz
3b2646a842b777f8c60dca4c452cb78eb3a223ffe59139b8501797aa1d23d583  validation.json.gz
```
