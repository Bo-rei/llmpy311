# Trainable-K1 与 ADB 五 seed 配对推断

本报告只对相同 protocol_v2 registry、相同 dataset×KIR×seed 做配对统计。ADB 使用 BERT/TextOIR，
Trainable-K1 使用 Known-only MiniLM，因此 CI 和胜负只描述工作点差异，不构成同 backbone SOTA 排名。

bootstrap：`10000` 次；RNG seed：`20260810`；输入 seed：`13,42,87,100,123`。

| 数据集 | KIR | 指标 | 均值差(pp) | 95% CI(pp) | 胜/平/负 | sign-test p | Cohen dz |
|---|---:|---|---:|---:|---:|---:|---:|
| clinc150 | 0.25 | OOS F1 | 3.57 | [2.96, 4.02] | 5/0/0 | 0.0625 | 5.18 |
| clinc150 | 0.25 | F1-All | 3.08 | [1.95, 4.25] | 5/0/0 | 0.0625 | 2.11 |
| clinc150 | 0.25 | Known Recall | -17.35 | [-18.53, -16.18] | 0/0/5 | 0.0625 | -11.24 |
| clinc150 | 0.25 | False Acceptance | -10.35 | [-11.22, -9.28] | 0/0/5 | 0.0625 | -7.85 |
| clinc150 | 0.50 | OOS F1 | 1.05 | [0.33, 1.82] | 5/0/0 | 0.0625 | 1.09 |
| clinc150 | 0.50 | F1-All | -3.67 | [-4.73, -2.74] | 0/0/5 | 0.0625 | -2.92 |
| clinc150 | 0.50 | Known Recall | -16.40 | [-17.01, -15.80] | 0/0/5 | 0.0625 | -20.68 |
| clinc150 | 0.50 | False Acceptance | -10.66 | [-12.10, -9.32] | 0/0/5 | 0.0625 | -5.95 |
| clinc150 | 0.75 | OOS F1 | -1.91 | [-2.97, -0.67] | 1/0/4 | 0.3750 | -1.29 |
| clinc150 | 0.75 | F1-All | -5.61 | [-6.45, -4.91] | 0/0/5 | 0.0625 | -5.45 |
| clinc150 | 0.75 | Known Recall | -16.42 | [-17.08, -15.76] | 0/0/5 | 0.0625 | -18.83 |
| clinc150 | 0.75 | False Acceptance | -13.47 | [-15.51, -11.60] | 0/0/5 | 0.0625 | -5.41 |
| banking77 | 0.25 | OOS F1 | 8.85 | [7.46, 10.29] | 5/0/0 | 0.0625 | 5.01 |
| banking77 | 0.25 | F1-All | 9.93 | [7.31, 12.31] | 5/0/0 | 0.0625 | 3.05 |
| banking77 | 0.25 | Known Recall | -5.63 | [-6.97, -4.29] | 0/0/5 | 0.0625 | -3.19 |
| banking77 | 0.25 | False Acceptance | -16.08 | [-18.16, -14.04] | 0/0/5 | 0.0625 | -5.79 |
| banking77 | 0.50 | OOS F1 | 8.60 | [7.04, 10.22] | 5/0/0 | 0.0625 | 4.21 |
| banking77 | 0.50 | F1-All | 2.89 | [2.02, 3.75] | 5/0/0 | 0.0625 | 2.59 |
| banking77 | 0.50 | Known Recall | -6.93 | [-7.78, -6.07] | 0/0/5 | 0.0625 | -6.20 |
| banking77 | 0.50 | False Acceptance | -17.92 | [-20.88, -15.27] | 0/0/5 | 0.0625 | -4.95 |
| banking77 | 0.75 | OOS F1 | 2.39 | [-0.41, 5.34] | 3/0/2 | 1.0000 | 0.65 |
| banking77 | 0.75 | F1-All | -1.76 | [-2.47, -1.05] | 0/0/5 | 0.0625 | -1.96 |
| banking77 | 0.75 | Known Recall | -7.94 | [-8.27, -7.57] | 0/0/5 | 0.0625 | -18.36 |
| banking77 | 0.75 | False Acceptance | -16.21 | [-20.66, -11.76] | 0/0/5 | 0.0625 | -2.81 |
| stackoverflow | 0.25 | OOS F1 | 2.34 | [1.32, 3.02] | 5/0/0 | 0.0625 | 2.07 |
| stackoverflow | 0.25 | F1-All | 4.52 | [2.78, 5.76] | 5/0/0 | 0.0625 | 2.35 |
| stackoverflow | 0.25 | Known Recall | 0.77 | [-0.59, 2.13] | 3/0/2 | 1.0000 | 0.44 |
| stackoverflow | 0.25 | False Acceptance | -4.18 | [-5.33, -2.65] | 0/0/5 | 0.0625 | -2.30 |
| stackoverflow | 0.50 | OOS F1 | 0.46 | [-0.69, 1.38] | 3/0/2 | 1.0000 | 0.35 |
| stackoverflow | 0.50 | F1-All | 0.75 | [0.02, 1.30] | 4/0/1 | 0.3750 | 0.89 |
| stackoverflow | 0.50 | Known Recall | 2.52 | [1.65, 3.48] | 5/0/0 | 0.0625 | 2.13 |
| stackoverflow | 0.50 | False Acceptance | 1.07 | [-0.81, 3.14] | 3/0/2 | 1.0000 | 0.43 |
| stackoverflow | 0.75 | OOS F1 | 2.60 | [0.79, 4.13] | 4/0/1 | 0.3750 | 1.24 |
| stackoverflow | 0.75 | F1-All | 1.29 | [0.82, 1.75] | 5/0/0 | 0.0625 | 2.31 |
| stackoverflow | 0.75 | Known Recall | 3.27 | [2.44, 4.30] | 5/0/0 | 0.0625 | 2.77 |
| stackoverflow | 0.75 | False Acceptance | 0.68 | [-3.04, 4.36] | 2/0/3 | 1.0000 | 0.13 |

## 解释

- OOS F1 的正差值表示 Trainable-K1 高于 ADB，负差值表示 ADB 更高。
- Known Recall 和 false acceptance 必须与 OOS F1 一起解释；只看 OOS F1 会隐藏拒识/覆盖工作点。
- 五个 seed 的配对 CI 只量化当前固定 split 与两个训练合同的差异，不能消除 BERT/MiniLM、训练目标和监督差异。

fair source SHA256：`31ccdd433b38b86ec92b0cd81b460114a7fe1e7f12b94d4a5d031aab7d703145`
ADB source SHA256：`0a287bf154bdb0d2db06ec916cfbb4aea40c8b536fabf419942f684e587e1223`
图：`figures/archive/analysis/adb_paired_inference_v1/trainable_minus_adb_paired_forest.png`
