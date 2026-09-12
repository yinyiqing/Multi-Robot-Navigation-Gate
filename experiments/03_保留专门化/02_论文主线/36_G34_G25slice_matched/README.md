# G34 on G25 `[0:256]` matched evaluation

状态：已登记，先执行 16 场 pilot；通过后运行 256 场 × 3 repeat。

## 目的

按最新研究决定，G34 只在 G25 已冻结的 Dense test 前 256 个场景上进行 matched
评测。该评测复用 G25 的场景 manifest 和环境 repeat，使 G34 可以与已经归档的 5A、
B2 及其他 G25 方法进行横向比较。它不是新的独立场景集，因此论文中必须称为
`G25-slice matched evaluation`，不能称为对 G34 的全新 sealed test。

`[640:896]` 评测保留在 `34_双头RewardAwareGate/local_data/independent_test/` 作为历史
记录，但不进入当前论文主表、主图或主结论。

## 冻结输入

- Manifest：`25_最终消融与Sealed评测/local_data/sealed_manifest/dense_test_first256.json.gz`
- 场景数：256 个五车场景
- 环境 repeat：`20260901`、`20260902`、`20260903`
- 方法：G34；5A 直接复用 G25 同 manifest、同 repeat 的已归档结果
- G34 checkpoint：`34_双头RewardAwareGate/local_data/training/seed20260912/best_runtime.pt`
- Actor：`generalist-5a` 与 `interaction-epoch16`，均冻结
- G0/G1：冻结
- Router：stride 2、on/off `0.43/0.33`、minimum hold 3

G34 本次只新增 `768` 个 episode（256 场景 × 3 repeat），不会重复运行 5A。

## 预注册边界

primary 为与 G25 5A 的 episode-level full success 差异；collision 为 robot-level
secondary。另报告 agent success、完成步数、interaction-actor selection share 和
switches。结果只用于当前 G25 slice 的 matched 横向比较，不据此声称对未测试场景或
任意车数的泛化。

运行入口：

```bash
bash scripts/start_g34_g25slice_matched.sh --pilot
bash scripts/start_g34_g25slice_matched.sh
```

日志和结果位于本目录的 `logs/` 与 `local_data/matched/`。完整运行前必须保留 pilot
审计记录；不得根据 pilot 结果修改 checkpoint、阈值、manifest 或统计口径。
