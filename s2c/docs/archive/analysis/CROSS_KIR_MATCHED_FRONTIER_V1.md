# 跨 KIR 匹配 Known 覆盖率前沿分析 V1

该阶段只读取已完成 operating-curve 结果，不重新训练、不选择新的阈值或方法。每个工作点先用 Known 分数得到目标覆盖率，再报告对应 OOS F1；因此它是事后机制诊断，不是测试集调参。

输入：`/home/bo/bo01/llmpy311/s2c/results/analysis/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall.csv`；SHA256：`bbccc3389ede2bbe55c7d4059283981127b4ab5d345ee14f6157cca0d8804fe3`。覆盖三个数据集、KIR=0.25/0.50/0.75、五个 seed、Known Recall 80%/90%/95%。

## 结论

- 如果在相同 Known Recall 下 S2C 仍保持更高 OOS F1，优势更接近分数排序/表示分离，而不是默认阈值的偶然选择。
- 若 MOGB 在低覆盖率工作点更高而 S2C 在高覆盖率工作点更高，则说明二者主要是保守拒识与 Known 覆盖的工作点差异。
- 任何 heatmap 数值都不能写成新的 SOTA 排名，因为 MOGB-Fair 与 S2C 仍属于当前 MiniLM Known-only 组件合同，且目标覆盖率是测试后诊断。

## 输出

- `results/analysis/archive/analysis/cross_kir_matched_frontier_v1/paired_frontier.csv`：同 dataset×KIR×seed×目标覆盖率的配对差值。
- `results/analysis/archive/analysis/cross_kir_matched_frontier_v1/summary.csv`：五 seed 均值、标准差、win/tie/loss。
- `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_delta_heatmaps.png`：跨 KIR 的 OOS F1 差值热图。
- `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_oos_f1_curves.png`：匹配 Known 覆盖率曲线。

## 按 dataset/KIR 的平均差值（跨三个目标覆盖率）

| 数据集 | KIR | S2C−MOGB OOS F1 pp |
|---|---:|---:|
| banking77 | 0.25 | 16.75 |
| banking77 | 0.50 | 20.10 |
| banking77 | 0.75 | 24.92 |
| clinc150 | 0.25 | 15.45 |
| clinc150 | 0.50 | 22.23 |
| clinc150 | 0.75 | 25.67 |
| stackoverflow | 0.25 | 25.22 |
| stackoverflow | 0.50 | 20.36 |
| stackoverflow | 0.75 | 32.00 |
