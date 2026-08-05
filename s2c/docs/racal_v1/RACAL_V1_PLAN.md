# RACAL-v1 第一阶段计划

RACAL-v1（Risk-Aware Center Activation Learning）当前只验证两个问题：

1. RACAL 独立 runner 能否逐样本复现 StackOverflow、KIR=0.50 的 E2 K=1；
2. MiniLM 最后两层加 384 维残差投影的 Known-only 训练，是否改善单中心 Gate。

本阶段不实现固定 K=2、自适应中心激活、proxy-OOS、parent guard 或完整级联。

实验范围固定为 StackOverflow、KIR=0.50、seed 13/42/87。Frozen K=1 只读取冻结 E2
embedding cache；Trainable K=1 只使用 train_known 和 calibration_known，最终 test 只评价一次。

通过门槛：E2 K=1 的 sample/prediction mismatch 为 0，指标最大绝对差不超过 `1e-10`；
Trainable K=1 至少两个 seed 的 OOS F1 高于 Frozen，平均 F1-All 不下降超过 0.5 个百分点，
Known Recall 不下降超过 2 个百分点，且 false acceptance 没有明显增加。未通过则停止中心激活实现。
