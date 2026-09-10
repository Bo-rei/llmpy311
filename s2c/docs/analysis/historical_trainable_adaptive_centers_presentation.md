# H1 自适应 per-intent 中心数量实验

所有主结果由 validation OOS F1 选择，test 只确认；Known F1、Accuracy、Known Recall 均无硬约束。
本实验属于 historical_v19_paper_main 的 H1 controlled evidence，论文 Ours 仅作历史数值参照。

## 三 seed 测试结果

| 数据集 | 选择范围 | OOS F1 均值±std | 相对论文 pp | 超过论文 seed 数 | 真实 pipeline |
|---|---|---:|---:|---:|---|
| banking77_oos | overall | 91.89±0.19 | +3.66 | 3/3 | True |
| banking77_oos | fixed | 91.56±0.17 | +3.33 | 3/3 | True |
| banking77_oos | adaptive | 91.89±0.19 | +3.66 | 3/3 | True |
| clinc150 | overall | 90.99±0.78 | -0.97 | 0/3 | True |
| clinc150 | fixed | 90.83±0.51 | -1.13 | 0/3 | True |
| clinc150 | adaptive | 90.99±0.78 | -0.97 | 0/3 | True |
| stackoverflow | overall | 89.96±1.50 | +0.25 | 2/3 | True |
| stackoverflow | fixed | 89.99±1.47 | +0.28 | 2/3 | True |
| stackoverflow | adaptive | 89.76±1.50 | +0.05 | 1/3 | True |

std 使用 ddof=0，与历史报告一致；三 seed 不代表统计显著性。overall 从统一和自适应候选共同选择，fixed/adaptive 是各自 validation 最优。

## 逐 seed 锁定配置与辅助指标

| 数据集/seed | 策略 | λ / threshold / mode | Val OOS | Test OOS | Known Recall | False acceptance | Pipeline macro F1 / Accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| clinc150/13 | mean_distance_top50_k2 | 2.5 / 0.9 / nearest_sphere | 88.97 | 89.99 | 83.29 | 8.74 | 79.19 / 85.51 |
| clinc150/42 | variance_top75_k2 | 2.5 / 0.9 / normalized_union | 88.43 | 91.09 | 83.96 | 7.08 | 80.92 / 86.98 |
| clinc150/87 | mean_distance_top25_k2 | 1.0 / 1.1 / nearest_sphere | 90.07 | 91.90 | 88.53 | 8.25 | 81.87 / 87.62 |
| stackoverflow/13 | fixed_k1 | 3.0 / 0.75 / normalized_union | 90.03 | 89.84 | 81.94 | 3.71 | 82.23 / 86.01 |
| stackoverflow/42 | fixed_k1 | 2.0 / 0.85 / nearest_sphere | 92.15 | 91.85 | 85.47 | 2.74 | 85.78 / 88.73 |
| stackoverflow/87 | sse_reduction_top25_k2 | 2.0 / 0.76 / normalized_union | 89.07 | 88.18 | 80.66 | 5.91 | 84.12 / 86.08 |
| banking77_oos/13 | mean_distance_top75_k2 | 0.25 / 1.0 / nearest_sphere | 87.88 | 91.61 | 67.40 | 6.53 | 67.02 / 85.66 |
| banking77_oos/42 | variance_top75_k2 | 2.5 / 0.7 / nearest_sphere | 89.51 | 92.02 | 61.10 | 4.03 | 65.11 / 86.18 |
| banking77_oos/87 | mean_distance_top75_k2 | 0.25 / 1.01 / nearest_sphere | 89.34 | 92.03 | 66.60 | 5.52 | 66.72 / 86.03 |

## 中心数、边界与 seed 工作点

对所有 27 个唯一策略完成同一粗网格；top100% 拆分策略与 fixed K=2/3 完全相同，以 alias 登记而不重复计算。
排序仅用归一化 Known train embedding：平均欧氏距离、总类内方差、K=2 相对 K=1 的 SSE 相对下降。拆分 intent 数向上取整，排序并列按 intent 名。
每个策略的 validation 粗网格最优 λ/mode 固定后，在其 threshold±.04 内以 .01 细化，并限制在 [.70,1.30]。所有配置、选择和中心映射先保存，再开始 test 指标计算。
Known F1/Accuracy 搜索表为 Gate 最近 intent 分类指标；full_pipeline_* 为真实下游指标，两者不能混用。

固定 K 与自适应 K 各自在 validation 上选参后的 test 差值（adaptive−fixed）：
- clinc150：+0.16 pp。
- stackoverflow：-0.23 pp。
- banking77_oos：+0.32 pp。

所以不能说自适应 K 普遍优于固定 K；CLINC/Banking 有小幅收益，SO 的 adaptive 子集均值反而较低。overall 是 validation 在两类候选间的选择，不根据 test 在 fixed/adaptive 间改选；其 test 均值也不保证高于两个子集。

## CLINC score ranking 与 intent 错误诊断

| Seed | Split | 选定 OOS F1 | 固定 score 最优阈值 oracle | AUROC | 全拒绝 OOS F1 |
|---|---|---:|---:|---:|---:|
| 13 | val | 88.97 | 89.11 | 93.39 | 68.09 |
| 13 | test | 89.99 | 90.20 | 93.74 | 74.29 |
| 42 | val | 88.43 | 88.53 | 94.01 | 68.09 |
| 42 | test | 91.09 | 91.39 | 95.25 | 74.29 |
| 87 | val | 90.07 | 90.16 | 94.71 | 68.09 |
| 87 | test | 91.90 | 92.08 | 95.35 | 74.29 |

oracle 是已选 score 的事后阈值上限诊断，不用于选择或作为泛化结果，也不是所有可能表示的理论上限。near-boundary 定义为 |score/threshold−1|≤.05，只作边界难度代理，不能等同人工语义 Near-OOS 标签。

| CLINC test OOS intent | 误接收/样本数（三 seed 汇总） | 边界附近样本数 |
|---|---:|---:|
| oos | 63/3000 | 75 |
| pto_used | 46/90 | 9 |
| no | 44/90 | 13 |
| change_ai_name | 41/90 | 18 |
| what_is_your_name | 40/60 | 13 |
| todo_list_update | 38/60 | 11 |
| report_lost_card | 34/60 | 24 |
| time | 24/90 | 36 |
| ingredients_list | 24/60 | 22 |
| cancel_reservation | 19/60 | 31 |
| interest_rate | 19/60 | 9 |
| last_maintenance | 19/90 | 9 |

## 原始 K=1 Gate 与 full pipeline 逐样本对账

以下为历史未调边界的 Trainable checkpoint，不是上方 validation-selected 自适应配置。数值均为百分数，差值为百分点。
Gate 分类采用最近中心 intent；pipeline 分类采用 Router/Expert。两者在相同全体测试样本和 Known intents+OOS 标签集合上计算 macro F1/Accuracy。

| 数据集/seed | Gate / pipeline OOS F1 | Gate / pipeline macro F1 | Gate / pipeline Accuracy | Δmacro / ΔAcc pp | Router / Expert error |
|---|---:|---:|---:|---:|---:|
| clinc150/13 | 88.63 / 88.63 | 79.51 / 75.72 | 85.27 / 83.87 | -3.79 / -1.40 | 2.38 / 3.61 |
| clinc150/42 | 89.56 / 89.56 | 81.12 / 77.26 | 86.56 / 85.16 | -3.87 / -1.40 | 2.44 / 2.81 |
| clinc150/87 | 90.30 / 90.30 | 82.10 / 77.88 | 87.35 / 85.80 | -4.22 / -1.55 | 2.26 / 4.04 |
| stackoverflow/13 | 88.78 / 88.78 | 84.63 / 81.51 | 86.48 / 84.96 | -3.12 / -1.52 | 0.00 / 7.90 |
| stackoverflow/42 | 91.49 / 91.49 | 87.23 / 85.41 | 89.30 / 88.40 | -1.82 / -0.90 | 0.00 / 5.97 |
| stackoverflow/87 | 86.08 / 86.08 | 85.44 / 83.10 | 85.44 / 84.26 | -2.34 / -1.19 | 0.00 / 3.60 |
| banking77_oos/13 | 88.41 / 88.41 | 70.16 / 67.51 | 81.99 / 81.40 | -2.66 / -0.59 | 0.00 / 10.63 |
| banking77_oos/42 | 87.68 / 87.68 | 68.41 / 64.41 | 81.00 / 80.27 | -4.00 / -0.74 | 0.00 / 11.22 |
| banking77_oos/87 | 89.32 / 89.32 | 70.40 / 65.74 | 82.97 / 82.08 | -4.66 / -0.88 | 0.00 / 12.69 |

Router error 是 accepted Known 中的 domain 错误比例；Expert error 是 accepted Known 中 domain 正确但 intent 错误的比例，两者使用相同分母且不重复计数。Gate Known Recall 指 Known 被接受的比例，不是 Known 分类正确率。
这些下游错误总数不等于相对最近中心分类新增的错误数：Router/Expert 也可能修复原来的 intent 错误；表中的 ΔAccuracy 才是净变化。
逐样本 CUDA 对账另保存 downstream_repaired_count/regressed_count，满足 ΔAccuracy=(修复数−新增错误数)/测试样本数。历史保存值和本次重算值分别保留；replay_minus_saved_pipeline_* 报告差异，不通过放宽断言假装精确复现。
OOS False Acceptance 分母为所有真实 OOS。legacy known_macro_f1 仅在真实 Known 子集上计算，不等于全测试样本上的 Known-class macro F1，本表统一使用包含 OOS 的 macro F1。
对账证据：fresh_sample_level_cuda_replay。无 --replay-pipeline 的重建保留已完成的逐样本 CUDA 对账，不降级为旧聚合证据。

## 判因与失败实验

H0/H1 协议与表示均有差异，现有对照不能把论文差距定量归因给其中某一项。当前 semantic_gate_enabled=False 的 H1 实现中，Router/Expert 不改 is_oos；直接运行核验的是该配置下的 Gate 编码/边界接线一致性，不能外推到历史 semantic gate 或其他 pipeline。
用户指出整体 pipeline 指标比单独 Gate 低：该现象成立，必须与 OOS F1 单独区分。原始 Gate prediction 重算与 full pipeline 逐 seed 对账见 historical_gate_pipeline_metric_reconciliation.csv。例如 CLINC seed42 的 Gate OOS Recall=96.37%、OOS F1=89.56%、macro F1=81.12%，full pipeline OOS F1=89.56%、macro F1 历史保存值为77.21%（本次重算见上表）；下游确实降低了整体分类表现。当前报告不能据 OOS 一致性断言整个 pipeline 无损失，也不能在未定位用户指向的另一组高 OOS F1 记录时把差异归因为指标混用。历史聚合与当前重算的小幅分类差异尚未定位到具体运行时原因，不能声称 byte-identical 重现。
统一增加 K、扩展 λ/threshold、细 threshold 和 per-sphere calibration 的历史负结果仍保留；本轮只检验 train-only 稀疏拆分，不能从更多中心推断一定收益。
如果 CLINC 自适应结果仍低于论文，应根据固定 score oracle 与 intent 错误决定表示训练实验，停止无目的扩大边界网格。

## 证据与复现

- `results/analysis/historical_trainable_adaptive_centers/MANIFEST.json`：设备与协议。
- 每个 dataset/seed 的 `center_assignments.json`、`selection_lock.json`、`validation_workpoints.csv`、`test_workpoints.csv`：所有策略映射、选择锁和候选指标。
- `selected_validation_per_seed.csv`、`selected_test_per_seed.csv`、`selected_test_summary.csv`：锁定配置与完整确认。
- `strategy_validation_selected_test.csv`、`ranking_diagnostics.csv`、`intent_errors.csv`：逐策略和聚合错误诊断。
- `python tools/analysis/build_historical_trainable_adaptive_centers_report.py --replay-pipeline`：仅用现有 checkpoint 重算九个单元的逐样本对账，不训练。
- `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python scripts/experiments/run_historical_trainable_adaptive_centers.py --device cuda`。历史目录已存在时须使用新的 `--output-root`，不覆盖。
