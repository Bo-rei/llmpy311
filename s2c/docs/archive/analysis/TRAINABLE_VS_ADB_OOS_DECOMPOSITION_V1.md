# Trainable-K1 与 ADB 的 OOS Precision/Recall 分解 V1

更新时间：2026-08-10

本报告从已经完成的逐样本对齐状态计数重算 OOS precision、OOS recall、OOS F1 和 Known acceptance，按同一 dataset×KIR×seed 比较 Trainable-K1 与 ADB。它不训练、不调阈值、不选择 checkpoint。ADB 仍是 BERT/TextOIR 外部合同。

## 工作点分解

| 数据集 | KIR | Δ OOS precision | Δ OOS recall | Δ OOS F1 | Δ Known acceptance |
|---|---:|---:|---:|---:|---:|
| banking77 | 0.25 | -0.92pp | +16.08pp | +8.85pp | -5.63pp |
| banking77 | 0.50 | -3.31pp | +17.92pp | +8.60pp | -6.93pp |
| banking77 | 0.75 | -8.60pp | +16.21pp | +2.39pp | -7.94pp |
| clinc150 | 0.25 | -3.84pp | +10.35pp | +3.57pp | -17.35pp |
| clinc150 | 0.50 | -8.24pp | +10.66pp | +1.05pp | -16.40pp |
| clinc150 | 0.75 | -14.28pp | +13.47pp | -1.91pp | -16.42pp |
| stackoverflow | 0.25 | +0.48pp | +4.18pp | +2.34pp | +0.77pp |
| stackoverflow | 0.50 | +1.78pp | -1.07pp | +0.46pp | +2.52pp |
| stackoverflow | 0.75 | +4.11pp | -0.68pp | +2.60pp | +3.27pp |

## 数据集平均

| 数据集 | Δ OOS precision | Δ OOS recall | Δ OOS F1 | Δ Known acceptance |
|---|---:|---:|---:|---:|
| banking77 | -4.28pp | +16.74pp | +6.61pp | -6.84pp |
| clinc150 | -8.79pp | +11.49pp | +0.90pp | -16.72pp |
| stackoverflow | +2.12pp | +0.81pp | +1.80pp | +2.19pp |

## 解释

- CLINC150 和 Banking77 的 OOS F1 改善主要来自 OOS recall 增加，也就是 OOS false acceptance 减少；OOS precision 反而下降，原因是更多 Known 被判成 OOS，体现保守拒识代价。
- StackOverflow 的 Trainable 在 KIR=.25 同时提高 OOS precision 和 recall；KIR=.50/.75 的 recall 略降，但 precision 提升足以使 OOS F1 仍为正。其主要收益是减少 Known rejection，并没有重现固定 K>1 的 acceptance-union 爆炸。
- 这组分解支持“Trainable 的优势是数据集相关的分数/边界工作点”，不支持跨 BERT/MiniLM 合同的无条件 SOTA 结论。

## 产物

- `results/analysis/archive/analysis/trainable_vs_adb_oos_decomposition_v1/`
- `figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_decomposition_heatmaps.png`
- `figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_workpoints.png`

源文件 SHA256：`4ce790f6df804c67c7636c621520ac9345395b1fbd83e0fc115df79ba3d182fc`。
