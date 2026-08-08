# 同协议方法权衡与可视化分析 V1

> 本报告只汇总已完成的 `protocol_v2_textoir_v1` 五 seed 结果，不重新训练，不加入历史 `fulltex.tex`、官方 MOGB BERT 或 DCLOOS 外部 OOS 监督结果。

## 1. 分析范围

- 三个数据集、KIR={0.25, 0.50, 0.75}、5 个 seed；
- Trainable K=1、Frozen K=1、Frozen K=2、Random K=2、MOGB 组件三种变体；
- 主要观察 OOS F1、F1-All、Known Recall、false acceptance/rejection 和 seed 方差；
- 所有行仍是 Gate-only 或同协议组件，不是完整 Cascade 的 SOTA 排名。

## 2. KIR=0.50 的 Trainable 与组件对照

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | FA | FR |
|---|---|---:|---:|---:|---:|---:|
| clinc150 | Trainable K=1 | 90.44 | 81.82 | 74.44 | 3.69 | 25.56 |
| clinc150 | Frozen K=1 | 88.94 | 80.27 | 78.76 | 8.82 | 21.24 |
| clinc150 | Frozen K=2 | 89.20 | 80.09 | 75.14 | 6.44 | 24.86 |
| clinc150 | Random K=2 | 89.07 | 80.75 | 79.48 | 8.96 | 20.52 |
| clinc150 | MOGB partition + s2c boundary | 85.56 | 64.60 | 53.34 | 2.50 | 46.66 |
| clinc150 | s2c partition + MOGB boundary | 83.32 | 56.41 | 42.24 | 1.69 | 57.76 |
| clinc150 | MOGB-MiniLM | 81.32 | 44.95 | 31.57 | 0.90 | 68.43 |
| banking77 | Trainable K=1 | 83.56 | 81.67 | 82.21 | 15.74 | 17.79 |
| banking77 | Frozen K=1 | 72.43 | 74.51 | 85.01 | 34.85 | 14.99 |
| banking77 | Frozen K=2 | 75.46 | 76.16 | 82.55 | 29.05 | 17.45 |
| banking77 | Random K=2 | 71.62 | 74.72 | 85.96 | 36.50 | 14.04 |
| banking77 | MOGB partition + s2c boundary | 79.40 | 64.81 | 52.18 | 3.50 | 47.82 |
| banking77 | s2c partition + MOGB boundary | 78.86 | 65.68 | 53.25 | 5.24 | 46.75 |
| banking77 | MOGB-MiniLM | 74.99 | 48.60 | 33.41 | 1.09 | 66.59 |
| stackoverflow | Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 | 16.11 |
| stackoverflow | Frozen K=1 | 76.55 | 79.98 | 87.15 | 29.71 | 12.85 |
| stackoverflow | Frozen K=2 | 63.53 | 72.76 | 86.89 | 47.17 | 13.11 |
| stackoverflow | Random K=2 | 75.88 | 79.80 | 87.64 | 30.98 | 12.36 |
| stackoverflow | MOGB partition + s2c boundary | 79.25 | 63.34 | 50.39 | 1.86 | 49.61 |
| stackoverflow | s2c partition + MOGB boundary | 73.35 | 63.59 | 54.49 | 15.69 | 45.51 |
| stackoverflow | MOGB-MiniLM | 72.92 | 43.30 | 27.09 | 0.79 | 72.91 |

## 3. 机制结论

1. Trainable K=1 通常位于更好的覆盖—拒识折中区域：它保留较高 Known Recall/F1-All，同时比 Frozen K=1 降低 false acceptance。
2. MOGB 风格组件有时提高 OOS F1，但常以显著牺牲 Known Recall 和 F1-All 为代价；这说明它们更像保守拒识工作点，而不是全面替代。
3. StackOverflow 的 Frozen K=2 仍出现明显 OOS 误接受，说明训练表示的 K=1 收益不能直接外推为固定多中心安全性。
4. KIR 增大时，方法之间的差距和误差权衡发生变化；因此不能用单一 KIR 的最好数字宣称跨数据集统一优势。

## 4. 图和机器结果

- `results/analysis/cross_protocol_tradeoff_v1/per_seed.csv`
- `results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`
- `results/analysis/cross_protocol_tradeoff_v1/trainable_vs_components_paired.csv`
- `figures/cross_protocol_tradeoff_v1/pareto_oos_f1_known_recall.png`
- `figures/cross_protocol_tradeoff_v1/error_decomposition_kir050.png`
- `figures/cross_protocol_tradeoff_v1/kir_curves_oos_f1_f1_all.png`
- `figures/cross_protocol_tradeoff_v1/seed_variance_oos_f1_kir050.png`

## 5. 结论边界

- 历史 `fulltex.tex` 是完整 Gate→Router→Expert Cascade，不能混入本报告的 fair rows。
- MOGB 这里是 Frozen MiniLM 组件适配，不是作者 BERT 完整复现。
- DCLOOS 使用伪 OOS/外部 OOS 监督，不能与 Known-only 行直接排名。
- 后续应在统一监督、split、seed 和系统层级后再做强基线主表；本报告本身不启动新训练。
