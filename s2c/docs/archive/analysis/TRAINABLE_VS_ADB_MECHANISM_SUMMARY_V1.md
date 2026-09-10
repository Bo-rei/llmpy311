# Trainable-K1 与 ADB 数据集/KIR 机制汇总 V1

更新时间：2026-08-10

本报告把已经完成的 Trainable-K1/ADB 指标差值与逐样本错误预算差值按同一 `dataset×KIR` 对齐。它只做后验分析，不训练、不调阈值、不选择 checkpoint，也不导出原始文本或逐样本公共结果。ADB 仍是 BERT/TextOIR 外部合同，不能与 MiniLM fair 行合并成 SOTA 排名。

## 工作点表

| 数据集 | KIR | Δ OOS F1 | Δ F1-All | Δ Known Recall | Δ Known 误拒 | Δ OOS 误接收 | 机制 |
|---|---:|---:|---:|---:|---:|---:|---|
| banking77 | 0.25 | +8.85pp | +9.93pp | -5.63pp | +5.63pp | -16.08pp | conservative_rejection |
| banking77 | 0.50 | +8.60pp | +2.89pp | -6.93pp | +6.93pp | -17.92pp | conservative_rejection |
| banking77 | 0.75 | +2.39pp | -1.76pp | -7.94pp | +7.94pp | -16.21pp | conservative_rejection |
| clinc150 | 0.25 | +3.57pp | +3.08pp | -17.35pp | +17.35pp | -10.35pp | conservative_rejection |
| clinc150 | 0.50 | +1.05pp | -3.67pp | -16.40pp | +16.40pp | -10.66pp | conservative_rejection |
| clinc150 | 0.75 | -1.91pp | -5.61pp | -16.42pp | +16.42pp | -13.47pp | conservative_rejection |
| stackoverflow | 0.25 | +2.34pp | +4.52pp | +0.77pp | -0.77pp | -4.18pp | coverage_recovery |
| stackoverflow | 0.50 | +0.46pp | +0.75pp | +2.52pp | -2.52pp | +1.07pp | coverage_false_accept_tradeoff |
| stackoverflow | 0.75 | +2.60pp | +1.29pp | +3.27pp | -3.27pp | +0.68pp | coverage_false_accept_tradeoff |

## 数据集结论

| 数据集 | 平均 Δ OOS F1 | 平均 Δ F1-All | 平均 Δ Known 误拒 | 平均 Δ OOS 误接收 | 主要机制 |
|---|---:|---:|---:|---:|---|
| banking77 | +6.61pp | +3.69pp | +6.84pp | -16.74pp | 保守拒识：Known 误拒增加，但 OOS 误接收下降 |
| clinc150 | +0.90pp | -2.07pp | +16.72pp | -11.49pp | 保守拒识：Known 误拒增加，但 OOS 误接收下降 |
| stackoverflow | +1.80pp | +2.19pp | -2.19pp | -0.81pp | 覆盖恢复，但中高 KIR 出现误接收权衡 |

## 解释

- CLINC150 和 Banking77 的 Trainable 工作点更保守：Known 误拒上升，但 OOS 误接收明显下降，因此 OOS 指标改善不能只解释为 Known 覆盖恢复。
- StackOverflow 的方向不同：Trainable 在三个 KIR 都减少 Known 误拒；KIR=.25 同时减少 OOS 误接收，而 KIR=.50/.75 出现轻微 OOS 误接收增加，体现覆盖—开放空间风险权衡。
- 因此不存在一个跨数据集的“Trainable 必然减少所有错误”机制。后续解释必须同时报告 OOS F1、F1-All、Known Recall 和两类条件错误率。
- 这些差异同时包含 MiniLM 与 BERT、训练目标和边界实现差异；本报告用于机制归因，不用于宣称超过完整 ADB/MOGB/DCLOOS 或 SOTA。

## 产物

- `results/analysis/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_mechanism.csv`
- `results/analysis/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_mechanism_summary.csv`
- `figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_mechanism_heatmaps.png`
- `figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_error_budget_quadrants.png`

源文件 SHA256：metrics=`3b0ae26ec6f83140f437ff5f81f763e464429a987e8df8e1ac0182d8455d6d74`；error_budget=`ba83df8365327597a93569270ed4eb83b0beb3cb791681944270a58ec84c0eac`。
