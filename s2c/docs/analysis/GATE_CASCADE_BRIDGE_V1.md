# Gate→Cascade 桥接与误差分解 V1

> 本报告只读取当前 protocol_v2_textoir_v1 已完成的 3-seed Cascade 行和 Trainable K=1 Gate 行，不重训、不改变历史结果。Gate-only 与 Cascade 的 Known intent label-space 不完全相同，因此只对共享的 OOS/ID 指标做直接桥接，Accuracy/F1-K 只在 Cascade 内解释。

## 1. 关键发现

1. Trainable K=1 Gate-only 的 OOS F1 已经改善，但历史高分还依赖后续 Router/Expert 和完整 Cascade；不能把 Gate-only 数字直接当作论文系统数字。
2. `ce_recon_selected_k` 和 `best_controlled_baseline` 的 OOS F1 变化不只来自 Gate：Router/Expert error 与 Gate false-reject/false-accept 共同决定端到端结果。
3. 同一个 OOS F1 可能对应不同 Known coverage；因此只看 OOS F1 会掩盖 Cascade 牺牲 Known Recall 换取拒识的情况。

## 2. KIR=0.50 汇总（3 个 seed）

| 数据集 | 变体 | OOS F1 | ID/ Known Recall | Gate FA | Known FR | Expert error |
|---|---|---:|---:|---:|---:|---:|
| clinc150 | Trainable K=1 Gate-only | 90.43 | 74.27 | 3.61 | 25.73 | -- |
| clinc150 | Frozen K=1 Cascade | 88.02 | 73.67 | 7.06 | 26.33 | 4.03 |
| clinc150 | Frozen selected-K Cascade | 88.02 | 73.67 | 7.06 | 26.33 | 4.03 |
| clinc150 | CE-Recon selected-K Cascade | 90.00 | 73.30 | 3.07 | 26.70 | 2.85 |
| clinc150 | Best controlled Cascade | 90.84 | 84.90 | 8.09 | 15.10 | 3.80 |
| banking77 | Trainable K=1 Gate-only | 84.77 | 81.91 | 13.46 | 18.09 | -- |
| banking77 | Frozen K=1 Cascade | 84.82 | 88.83 | 23.66 | 11.17 | 12.24 |
| banking77 | Frozen selected-K Cascade | 88.11 | 80.37 | 16.08 | 19.63 | 11.28 |
| banking77 | CE-Recon selected-K Cascade | 89.48 | 79.37 | 13.59 | 20.63 | 10.10 |
| banking77 | Best controlled Cascade | 84.17 | 88.40 | 24.35 | 11.60 | 10.19 |
| stackoverflow | Trainable K=1 Gate-only | 86.71 | 83.92 | 11.14 | 16.08 | -- |
| stackoverflow | Frozen K=1 Cascade | 79.02 | 83.22 | 23.52 | 16.78 | 7.50 |
| stackoverflow | Frozen selected-K Cascade | 78.00 | 84.61 | 25.86 | 15.39 | 8.15 |
| stackoverflow | CE-Recon selected-K Cascade | 87.62 | 86.15 | 11.06 | 13.85 | 6.15 |
| stackoverflow | Best controlled Cascade | 83.24 | 85.49 | 18.16 | 14.51 | 6.88 |

## 3. 证据

- `results/analysis/gate_cascade_bridge_v1/per_seed.csv`
- `results/analysis/gate_cascade_bridge_v1/summary_mean_std.csv`
- `figures/gate_cascade_bridge_v1/gate_vs_cascade_oos_f1.png`
- `figures/gate_cascade_bridge_v1/cascade_error_decomposition.png`
- `figures/gate_cascade_bridge_v1/cascade_oos_known_tradeoff.png`

## 4. 结论边界

- 该桥接证明“当前 Trainable 低于历史 fulltex”不能归因于一个 MiniLM checkpoint；必须把 Gate、Router、Expert 和校准合同分开。
- 它不是新的 SOTA 结果，也不把 3-seed 当前 Cascade 变体冒充 fulltex 历史主表。
