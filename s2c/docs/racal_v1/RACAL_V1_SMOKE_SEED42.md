# RACAL-v1 Trainable K=1 Smoke（StackOverflow/KIR=0.50/seed=42）

本报告只覆盖 RACAL-v1 第一阶段的 seed=42。没有实现或运行 fixed K=2、中心激活、proxy-OOS、parent guard 或其他数据集。

## 工程检查

- 训练损失有限且从 `0.4209` 降至 `0.1630`。
- best checkpoint 为 epoch 4，选择指标为 Known calibration 的 `F1-K + 0.05 × Known Recall`。
- checkpoint 重新加载后，test 指标与保存结果的 `OOS F1`、`F1-All` 差异均为 0。
- 输出维度为 384；train/calibration/test sample-id 顺序由 manifest 固定。
- `test_used_for_selection=false`，训练和选模只使用 Known train/calibration。
- 前四个 Transformer block 冻结，仅最后两个 block 和 projection head 有梯度。

## 结果

|方法|OOS F1|F1-All|F1-K|Known Recall|False Acceptance|AUROC|AUPR-OOS|
|-|-|-|-|-|-|-|-|
|Frozen K=1|0.8197|0.7766|0.7723|0.8380|0.1930|0.9065|0.8927|
|Trainable K=1|0.8755|0.8577|0.8559|0.8350|0.0930|0.9268|0.9120|

相对 Frozen K=1：OOS F1 `+5.58pp`，F1-All `+8.11pp`，Known Recall `-0.30pp`，false acceptance `-10.00pp`。

## 决策

seed=42 通过工程 smoke 条件，允许继续运行 seed=13 和 seed=87。该结果仍不是 RACAL 多中心结果，不能授权 fixed K=2 或中心激活；最终是否进入下一阶段必须等待三个 seed 的聚合结果。
