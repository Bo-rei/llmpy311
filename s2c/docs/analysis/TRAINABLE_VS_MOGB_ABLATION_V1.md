# Trainable K=1 与 MOGB 组件归因对比 V1

> 本报告只连接已完成的 Trainable 五 seed K=1 和 MOGB frozen-MiniLM ablation，不是官方 BERT MOGB 复现，也不新增训练。

## 1. 研究问题

- 自有方法的优势是否来自动态粒球划分，还是来自表示/边界工作点？
- MOGB 组件中，递归划分、距离函数和半径规则分别贡献多少？
- Trainable K=1 的较高 OOS F1 是否伴随更好的 Known 覆盖？

## 2. KIR=.50 五 seed 均值

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False Accept |
|---|---|---:|---:|---:|---:|
| clinc150 | Trainable K=1 | 90.44±0.44 | 81.82±0.92 | 74.44±0.34 | 3.69±0.76 |
| clinc150 | MOGB Mahalanobis + mean | 80.70±0.32 | 40.73±1.68 | 27.94±1.43 | 0.57±0.10 |
| clinc150 | MOGB Mahalanobis + mean+std | 85.56±0.42 | 64.60±2.17 | 53.34±1.90 | 2.50±0.32 |
| clinc150 | MOGB Euclidean + mean+std | 86.77±0.42 | 70.30±1.62 | 60.36±1.58 | 3.57±0.38 |
| banking77 | Trainable K=1 | 83.56±1.83 | 81.67±1.05 | 82.21±0.48 | 15.74±3.43 |
| banking77 | MOGB Mahalanobis + mean | 74.36±0.68 | 45.07±2.65 | 30.26±2.26 | 0.60±0.28 |
| banking77 | MOGB Mahalanobis + mean+std | 79.40±0.78 | 64.81±1.98 | 52.18±2.02 | 3.50±0.87 |
| banking77 | MOGB Euclidean + mean+std | 79.95±0.74 | 68.22±1.54 | 57.29±1.85 | 5.68±1.53 |
| stackoverflow | Trainable K=1 | 87.67±1.66 | 86.55±1.30 | 83.89±0.36 | 9.34±3.26 |
| stackoverflow | MOGB Mahalanobis + mean | 72.41±0.67 | 40.71±3.48 | 24.82±2.59 | 0.60±0.24 |
| stackoverflow | MOGB Mahalanobis + mean+std | 79.25±1.49 | 63.34±5.50 | 50.39±4.91 | 1.86±0.70 |
| stackoverflow | MOGB Euclidean + mean+std | 80.21±1.55 | 66.91±5.81 | 54.97±5.32 | 2.95±1.12 |

## 3. 配对结论

- Trainable 与 `mahalanobis_diag_mean_std` 在同一 dataset×KIR×seed 下配对；差值、95% bootstrap CI 和 win/tie/loss 见 `paired_effects.csv`。
- 在 KIR=.50，Trainable 相对 MOGB partition + s2c boundary 的 OOS F1 增量为 CLINC150/Banking77/StackOverflow `+4.88/+4.17/+8.42pp`；同时 Known Recall 增量约 `+21.10/+30.03/+33.51pp`。这支持“更平衡的覆盖—拒识工作点”，不是简单的 OOS 阈值优势。
- `trainable_vs_mogb_ablation_pareto.png` 显示 Trainable 位于较高 Known Recall 区域；MOGB mean-radius 组件通常通过拒绝大量 Known 样本降低 false acceptance。

## 4. MOGB 组件归因

- MOGB closeout 的 540 个单元显示：将 mean radius 换成 mean+std，OOS F1 总体提升约 5.26pp，并在 45 个 paired cells 全部为正；这大于单纯改变 purity-get 阈值的收益。
- Euclidean 通常优于 diagonal Mahalanobis；因此在当前 Frozen MiniLM 空间，距离/半径合同比递归划分阈值更决定工作点。
- 但即使使用 Euclidean + mean+std，MOGB 仍常低于 Trainable 的 F1-All 和 Known Recall；这说明 Trainable 的主要可复现优势来自 Known-only 表示适配后的 score separation 和覆盖，而非已经证明动态多中心更好。
- StackOverflow 固定 K>1 的失败仍然是 union 接受区域风险，不能用更换 MOGB 半径规则解释为已解决。

## 5. 证据与边界

- 输入：`results/analysis/minilm_trainable_5seed_fair_v1/trainable_per_seed.csv` 和 `../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1/summary/boundary_component_runs.csv`。
- 输出：`results/analysis/trainable_vs_mogb_ablation_v1/` 和 `figures/trainable_vs_mogb_ablation_v1/`。
- 不包含历史 `fulltex.tex`、官方 BERT MOGB、DCLOOS 外部 OOS 监督；这些仍需按监督条件和系统层级单独报告。
- 不使用 test 指标选择参数；所有输出为对已完成运行的分析。
