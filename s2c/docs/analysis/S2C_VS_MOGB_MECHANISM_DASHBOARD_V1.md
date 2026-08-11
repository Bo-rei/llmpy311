# S2C 与 MOGB-Fair 机制对比仪表盘 V1

> 比较对象固定为当前 `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair`。两者共享 `protocol_v2_textoir_v1` 的 dataset/KIR/seed，但表示和边界合同不同。本报告不把历史 Cascade 或论文 BERT MOGB 混入公平排名。

## 实验覆盖

- 三个数据集 × 三个 KIR × 五个相同 seed，共 45 个严格配对单元。
- 主比较：Known-only Trainable MiniLM K=1 vs 冻结 MiniLM MOGB adaptive balls + mean radius。
- 组件桥：增加 `MOGB partition + S2C mean_std boundary`，用于分离边界工作点与表示适配的贡献。
- 统计：固定 seed=20260725、10,000 次配对 bootstrap。

## 核心结论

1. S2C 的优势不是“更保守地拒绝更多样本”。45 个配对中，OOS F1 win/loss=44/1，F1-All win/loss=45/0。
2. 相对 MOGB-Fair，S2C 平均 Known Recall 提高 +49.01pp、OOS Precision 提高 +20.58pp；代价是 OOS Recall -7.83pp、false acceptance +7.83pp。其 OOS F1 提升来自 precision–recall 工作点更平衡，而不是任一方向全面占优。
3. MOGB-Fair 的主要问题是大量 Known→OOS：mean-radius 粒球非常保守。换成 S2C mean_std 边界通常能恢复部分 F1-All，但完整 Trainable-K1 仍普遍更好，说明优势同时来自表示适配和边界工作点。
4. StackOverflow 固定 K=2 的失败与 MOGB-Fair 的失败方向相反：固定 K=2 是 OOS false acceptance 过多；MOGB-Fair 是 Known false rejection 过多。不能用同一个“多中心无效”概括二者。

## 45 单元平均

| method                        |   oos_f1_mean |   f1_all_mean |   id_recall_mean |   false_accept_rate_mean |
|:------------------------------|--------------:|--------------:|-----------------:|-------------------------:|
| MOGB-Fair                     |        0.7339 |        0.4626 |           0.3121 |                   0.0090 |
| MOGB partition + S2C boundary |        0.7774 |        0.6392 |           0.5221 |                   0.0265 |
| S2C Trainable K=1             |        0.8575 |        0.8336 |           0.8022 |                   0.0873 |

## 每个数据集与 KIR 的 S2C−MOGB 差值（百分点）

| dataset       |   kir |   f1_all |   false_accept_rate |   false_reject_rate |   id_recall |   oos_f1 |   oos_precision |   oos_recall |
|:--------------|------:|---------:|--------------------:|--------------------:|------------:|---------:|----------------:|-------------:|
| banking77     | +0.25 |   +22.63 |               +8.83 |              -41.87 |      +41.87 |    +1.35 |          +10.42 |        -8.83 |
| banking77     | +0.50 |   +33.07 |              +14.65 |              -48.80 |      +48.80 |    +8.57 |          +22.54 |       -14.65 |
| banking77     | +0.75 |   +41.74 |              +18.84 |              -53.65 |      +53.65 |   +21.02 |          +28.59 |       -18.84 |
| clinc150      | +0.25 |   +27.45 |               +2.58 |              -36.53 |      +36.53 |    +2.78 |           +7.28 |        -2.58 |
| clinc150      | +0.50 |   +36.86 |               +2.79 |              -42.87 |      +42.87 |    +9.12 |          +16.29 |        -2.79 |
| clinc150      | +0.75 |   +41.12 |               +3.19 |              -45.92 |      +45.92 |   +16.92 |          +23.50 |        -3.19 |
| stackoverflow | +0.25 |   +33.57 |               +2.91 |              -51.68 |      +51.68 |    +6.04 |          +13.30 |        -2.91 |
| stackoverflow | +0.50 |   +43.25 |               +8.55 |              -56.80 |      +56.80 |   +14.75 |          +27.26 |        -8.55 |
| stackoverflow | +0.75 |   +54.19 |               +8.13 |              -62.94 |      +62.94 |   +30.72 |          +36.06 |        -8.13 |

## 图表

1. `paired_delta_heatmaps.png`：六个指标的 dataset×KIR 配对差值。
2. `error_budget_arrows.png`：从 MOGB 保守拒识工作点移动到 S2C 平衡工作点。
3. `oos_precision_recall_decomposition.png`：OOS F1 的 precision/recall 来源。
4. `kir_performance_curves.png`：五 seed 的 KIR 趋势与标准差。
5. `paired_effect_forest.png`：OOS F1、F1-All、Known Recall 的配对置信区间。
6. `component_bridge_f1_all.png`：MOGB boundary、S2C boundary、Trainable representation 的组件桥。

## 解释边界

- 这能够解释当前 S2C Gate 为什么优于 MOGB 的冻结 MiniLM 公平组件。
- 它不能证明超过论文完整 BERT MOGB；本地官方逻辑 MOGB 的复现差距另见 `MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`。
- 它也不能替代完整 Gate→Router→Expert 与 ADB/DA-ADB/DCLOOS 的同协议主表。
- Manifest SHA256：`1f3c73102a094a0b087fdbe95d02e68d06e0275c75bb77b4dc3a313d8f2dbf4e`。
