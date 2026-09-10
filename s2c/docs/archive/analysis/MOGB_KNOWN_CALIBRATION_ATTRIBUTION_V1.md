# MOGB Known-only 校准与子中心损失归因 V1

## 结论

本实验完成 **45 个 dataset×KIR×seed 粒球单元、225 个边界工作点评价单元和 270 个损失契约评价单元**。
45/45 个重拟合单元与冻结的 `mogb_minilm` 参考结果达到 score/指标数值等价，因此以下差异不是缓存错位或重新实现漂移。

MOGB-Fair 的平均距离半径在 Known calibration 上普遍过窄。把半径扩大到仅由 `calibration_known` 决定的 95% 覆盖工作点，可以显著恢复 Known Recall，但同时增加 OOS false acceptance；这说明本地 MOGB 的弱综合性能包含明确的**工作点失配**，但不能仅靠半径放大自动得到更好的开放集排序，因为 AUROC/AUPR 不随单调阈值变化。

官方代码中的子中心损失先对每个样本的类别距离向量做 L1 归一化，再对负距离做 softmax。实际 MOGB-Fair 距离表上，`official_l1` 的平均真实类概率为 `0.060129`，平均距离梯度范数为 `0.050224`；未经这一步压缩的 `raw distance / tau=0.10` 分别为 `0.638877` 和 `3.828298`。这与严格兼容运行中“CE 收敛而 sub-centroid loss 接近均匀分布”的现象一致，是当前最强的代码级根因证据之一。

粒球筛选还会让部分 Known 类没有任何 selected ball。该现象不是脚本假设：45/45 Gate 重放与历史结果等价，但损失诊断必须排除没有中心的类别，不能伪造距离。

| 数据集 | 出现缺类的单元 | 平均缺失类数 | 最大缺失类数 | 平均排除训练行 | 最大排除训练行 |
|---|---:|---:|---:|---:|---:|
| banking77 | 3/15 | 0.33 | 3 | 25.3 | 294 |
| clinc150 | 8/15 | 0.73 | 3 | 73.3 | 300 |
| stackoverflow | 5/15 | 0.67 | 4 | 400.0 | 2400 |

## Calibration-95 相对默认平均变化

| 数据集 | 所需半径倍率 | OOS F1 (pp) | F1-All (pp) | Known Recall (pp) | False acceptance (pp) |
|---|---:|---:|---:|---:|---:|
| banking77 | 1.856 | -37.02 | +13.53 | +61.16 | +76.64 |
| clinc150 | 1.522 | -32.59 | +21.05 | +62.59 | +66.62 |
| stackoverflow | 1.343 | -37.78 | +17.65 | +68.40 | +78.90 |

这些变化是 3 个 KIR×5 个 seed 的均值。Calibration-95 是预注册 Known coverage 工作点，不使用 test Known 或 test OOS 选择；它不是声称最优的半径参数。

## 对“为什么没有复现论文 MOGB”的回答

1. **不是简单的训练轮数不足。** 既有 BERT 严格兼容单格中 CE 和 Known dev accuracy 已收敛。
2. **边界工作点确实有系统性失配。** 默认 mean radius 会拒绝大量 Known；Known-only 校准能恢复覆盖，但必然增加开放空间接受风险。
3. **子中心训练信号被公式压缩。** L1-normalized distance softmax 的概率动态范围随 Known 类数增长快速缩小；本实验在真实距离表上验证了这一点。
4. **仍不能把差距完全归因于代码缺陷。** 论文原始样本 ID、Known intent 列表、旧运行环境和完整作者数据契约不可恢复；本实验是官方公式和适配组件的归因，不是对论文结果真实性的否定。

## 当前自有方法与 MOGB-Fair 的同协议比较

这里比较的自有方法是 **S2C Trainable MiniLM K=1 Gate**：Known-only 训练 MiniLM 最后两层和 residual projection，随后使用单中心对角马氏 Gate；不是 `fulltex.tex` 中的完整 Gate→Router→Expert Cascade。MOGB-Fair 使用相同 TEXTOIR split 和冻结 MiniLM，但只适配 MOGB 粒球与边界组件；不是官方 BERT 完整 MOGB。

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---|---:|---:|---:|---:|
| clinc150 | S2C Trainable K=1 | 90.44 | 81.82 | 74.44 | 3.69 |
| clinc150 | MOGB-Fair default | 81.32 | 44.95 | 31.57 | 0.90 |
| clinc150 | MOGB-Fair cal-80 | 83.70 | 76.38 | 80.69 | 18.92 |
| banking77 | S2C Trainable K=1 | 83.56 | 81.67 | 82.21 | 15.74 |
| banking77 | MOGB-Fair default | 74.99 | 48.60 | 33.41 | 1.09 |
| banking77 | MOGB-Fair cal-80 | 70.39 | 72.10 | 82.03 | 36.10 |
| stackoverflow | S2C Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 |
| stackoverflow | MOGB-Fair default | 72.92 | 43.30 | 27.09 | 0.79 |
| stackoverflow | MOGB-Fair cal-80 | 74.64 | 73.15 | 80.28 | 27.74 |

在全部 45 个同 seed 配对中，S2C Trainable K=1 相对预注册的 MOGB cal-80 工作点取得 OOS F1 胜出 `45/45`、F1-All 胜出 `45/45`。这表明当前差异不能只用 MOGB 默认半径过窄解释；即使把 MOGB 调到较合理的 Known 覆盖，其开放集排序和综合 Known/OOS 权衡仍通常弱于当前 Trainable K=1。

## 图表

- `figures/archive/analysis/mogb_known_calibration_attribution_v1/known_recall_oos_f1_workpoints.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/radius_multiplier_by_coverage.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/cal95_delta_vs_default.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/subcentroid_loss_signal.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/trainable_k1_vs_mogb_cal80.png`

## 证据与边界

- 等价门：`equivalence.csv`，通过 `45/45`。
- 正式工作点表：`workpoint_per_seed.csv`，选择 split 始终为 Known calibration。
- 损失数值实验：`loss_contract_per_cell.csv`，只评价训练信号，不宣称测试性能提升。
- Manifest SHA256：`5e2d90bf4a793fcf19af240610f04245cab505ecdf63936853ca4fd20174656b`。
- 原始文本、embedding、checkpoint 和逐样本预测没有复制到轻量结果目录。
