# 当前实验可视化证据包 V2

更新时间：2026-08-10
活动协议：`protocol_v2_textoir_v1`

本文件只引用当前已冻结的五 seed fair matrix、已完成的 Cascade bridge 和 MOGB 归因结果。它是实验阶段的可视化阅读入口，不是论文正文，也不把不同监督合同混成一个 SOTA 排名。

历史 `fulltex.tex` 完整 Cascade 与当前 Trainable-K1 的逐格合同差距见
`docs/analysis/CROSS_CONTRACT_GAP_V1.md`；该报告只做描述性对齐，不改变本文件的 fair matrix 主结论。

机器可读图索引和统一合同见 [`UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`](UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md) 及 `results/analysis/archive/analysis/unified_comparison_v1/figure_manifest.json`。

## 1. 当前主要比较对象

当前自有最佳对象是 `S2C-Trainable-K1`：Known-only 训练 MiniLM 最后两层和 projection，随后使用 K=1 单中心 Gate。

同协议 fair matrix 还包括 Frozen K=1、Frozen K=2、Random K=2、MOGB partition + S2C boundary、S2C partition + MOGB boundary 和 MOGB-MiniLM，共 3 数据集 × 3 KIR × 5 seeds。

历史 `fulltex.tex` Cascade、官方 BERT MOGB、ADB、DA-ADB 和 DCLOOS 属于独立合同层。

## 2. 当前五 seed fair 结果

来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。

| 方法 | OOS F1 九格均值 | F1-All 九格均值 | OOS F1 第一名次数 | F1-All 第一名次数 |
|---|---:|---:|---:|---:|
| **S2C Trainable K=1** | **85.75%** | **83.36%** | **8/9** | **9/9** |
| Frozen K=1 | 78.64% | 78.30% | 0/9 | 0/9 |
| Random K=2 | 78.39% | 78.51% | 0/9 | 0/9 |
| Frozen K=2 | 75.51% | 76.79% | 0/9 | 0/9 |
| MOGB partition + S2C boundary | 77.74% | 63.92% | 1/9 | 0/9 |
| S2C partition + MOGB boundary | 75.90% | 62.30% | 0/9 | 0/9 |
| MOGB-MiniLM | 73.39% | 46.26% | 0/9 | 0/9 |

主图：

- `figures/archive/analysis/experiment_analysis_master_v1/oos_f1_heatmap.png`
- `figures/archive/analysis/experiment_analysis_master_v1/f1_all_heatmap.png`
- `figures/archive/analysis/experiment_analysis_master_v1/pareto_oos_f1_f1_all_kir050.png`
- `figures/archive/analysis/statistical_stability_v1/paired_oos_f1_forest.png`

## 3. 为什么当前方法优于 MOGB-Fair

45 个相同 dataset/KIR/seed 配对单元显示：

- OOS F1：Trainable 胜出 44/45；
- F1-All：45/45 胜出；
- AUROC：45/45 胜出；
- 平均 Known Recall：Trainable 比 MOGB-Fair 高 49.01pp；
- 平均 F1-All：高 37.10pp；
- 平均 OOS F1：高 12.36pp。

机制不是“单纯拒绝更多 OOS”：

1. MOGB-Fair 的平均半径较保守，大量 Known 样本被拒绝；
2. Trainable K=1 恢复了这些 Known 样本，并只付出有限的 OOS Recall 损失；
3. 固定 K=2 则相反，StackOverflow 的多个球并集增加了 OOS false acceptance。

对应图：

- `figures/s2c_vs_mogb_mechanism_dashboard_v1/error_budget_arrows.png`
- `figures/s2c_vs_mogb_mechanism_dashboard_v1/oos_precision_recall_decomposition.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/paired_correctness_gain_decomposition.png`
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png`

## 4. StackOverflow 的具体失败结构

KIR=.50 五 seed 结果：

| 方法 | OOS F1 | F1-All | Known Recall | false acceptance | false rejection |
|---|---:|---:|---:|---:|---:|
| Trainable K=1 | 87.67% | 86.55% | 83.89% | 9.34% | 16.11% |
| Frozen K=1 | 76.55% | 79.98% | 87.15% | 29.71% | 12.85% |
| Frozen K=2 | 63.53% | 72.76% | 86.89% | 47.17% | 13.11% |
| MOGB-MiniLM | 72.92% | 43.30% | 27.09% | 0.79% | 72.91% |

固定 K=2 的主要错误是 OOS 误接收增加；MOGB-MiniLM 的主要错误是 Known 误拒绝。StackOverflow 前五个高风险 intent 约贡献 64.9% 的固定 K=2 OOS false acceptance。

对应图：

- `figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_error_source_waterfall.png`
- `figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_intent_oos_acceptor_heatmap.png`
- `figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/fixed_k2_oos_acceptor_rank_curve.png`
- `figures/archive/analysis/mechanism_closure_v1/mogb_risk_workpoints.png`

## 5. MOGB 论文复现差距

StackOverflow/KIR=.50/seed=0 的官方逻辑现代兼容运行：

| 指标 | 本地运行 | 论文参考 |
|---|---:|---:|
| Accuracy | 75.17 | 88.67 |
| F1-All | 68.35 | 87.49 |
| F1-U | 79.97 | 89.71 |
| F1-K | 67.19 | 87.27 |

目前已验证的差距来源包括：

- 官方子中心损失的 L1 距离归一化压缩了类别训练信号；
- 平均半径过窄，导致 Known Recall 只有约 51.53%；
- selected-ball 可能遗漏 Known 类；
- 作者数据快照、Known 列表、旧环境和最终粒球随机状态未完整恢复。

修正损失和扩大 Known-only 半径均不能单独恢复论文工作点。因此正确状态是 `official_code_not_reproduced_under_available_materials`，不能写成 MOGB 算法已经被否定。

对应图：

- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/paper_gap_metrics.png`
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/ce_vs_subcentroid_loss.png`
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/dev_accuracy_vs_known_recall.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/subcentroid_loss_signal.png`
- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/radius_workpoints.png`

## 6. 当前协议 Cascade bridge

同一 Known-only Expert 下，Trainable K=1 相对 Frozen K=1 的 OOS F1 配对提升为：

| 数据集 | OOS F1 | F1-All | Known Recall | false acceptance |
|---|---:|---:|---:|---:|
| CLINC150 | +1.12pp | +1.52pp | −1.33pp | −2.86pp |
| Banking77 | +5.18pp | +3.13pp | −1.95pp | −10.00pp |
| StackOverflow | +9.42pp | +6.70pp | +0.21pp | −15.40pp |

图：`figures/archive/analysis/cascade_bridge_cross_dataset_v1/` 下的三张 Cascade 图。

## 7. 外部方法合同边界

- ADB：StackOverflow/KIR=.50/5 seeds，OOS F1 `87.21±1.17%`，但为 BERT/TextOIR 合同；
- DA-ADB：当前 protocol_v2 三 seed 已有有限、非单类塌缩预测，均值 OOS F1=`72.48±6.24%`；仍是 BERT/TextOIR 外部合同；
- DCLOOS：reduced 单元 OOS F1 `87.05%`，使用 pseudo-OOS 与外部 SQuAD，不能进入 Known-only 排名；
- MSP：最新 runtime 探测超时，没有指标。

因此当前不能声称跨合同 SOTA，只能声称：**在当前统一 Known-only MiniLM Gate 合同中，Trainable K=1 是最稳定且最平衡的自有候选，并且相对 MOGB-Fair 的优势可由 Known 覆盖、分数排序和错误预算图解释。**

## 8. 同一 Trainable 表示下的检测器对照

在同一 Trainable MiniLM 表示和同一 Known calibration 下，Gate、MSP、Energy、kNN、LOF 的 3 数据集控制已完成。Gate 的 OOS F1 在四种原生 detector 上均为 3/3 seed 胜出，配对 bootstrap 的平均增益为：

- CLINC150：相对 MSP/Energy/kNN/LOF 为 `+2.05/+2.13/+3.62/+11.73pp`；
- Banking77：`+12.21/+15.25/+17.42/+27.11pp`；
- StackOverflow：`+41.10/+43.76/+17.88/+23.17pp`。

StackOverflow 上 MSP/Energy 的 Known Recall 约 95.6%，但 false acceptance 约 69%--71%；Gate 将 false acceptance 降到 11.14%，代价是 Known Recall 降到 83.92%。这说明当前优势来自：

```text
Trainable MiniLM 表示 + Known-only 几何 Gate
```

而不是“微调后任何 detector 都自动变好”。

图：`figures/archive/analysis/detector_mechanism_v1/`；详细报告：`docs/archive/analysis/DETECTOR_MECHANISM_ANALYSIS_V1.md`。

## 9. 权威数据与报告

- `docs/archive/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`
- `docs/analysis/COMPARISON_ATLAS_V2.md`
- `docs/archive/analysis/MECHANISM_CLOSURE_V1.md`
- `docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`
- `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`
- `results/analysis/cross_protocol_tradeoff_v1/`
- `results/analysis/s2c_vs_mogb_mechanism_dashboard_v1/`

## 10. DA-ADB 当前协议三 seed 新证据

DA-ADB 当前 protocol_v2 的 StackOverflow/KIR=.50 三个 seed 已完成，均为有限、非单类塌缩预测。其 OOS F1 为
`70.82/67.22/79.38%`，均值 `72.48±6.24%`；同 seed S2C Trainable K=1 为
`87.55/86.92/90.16%`，均值 `88.21±1.72%`。配对差值为 OOS F1 `+15.73pp`、F1-All `+12.67pp`，
但这仍是 BERT/TextOIR 与 Known-only MiniLM 的合同对照，不是同骨干 SOTA 排名。

图与机器可读结果：

- `figures/da_adb_current_protocol_summary_v1/current_protocol_seed_metrics.png`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_tradeoff.png`
- `docs/analysis/DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md`
- `results/analysis/da_adb_current_protocol_summary_v1/`

## 11. DCLOOS 当前 registry 单格阻塞证据

已新增固定 Known-list adapter，并在 StackOverflow/KIR=.50/seed=42 当前 registry 上启动 DCLOOS。该
运行保留 BERT、pseudo-OOS 与外部 SQuAD 监督，但约 3530 秒后没有生成最终 metrics；中间预测不进入
任何图表。现阶段只展示既有 reduced 结果的合同标签，不绘制伪造的当前协议 DCLOOS 点。

- 阻塞说明：`docs/archive/analysis/DCLOOS_CURRENT_PROTOCOL_BLOCKER_V1.md`
- 运行 manifest：`../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_v1/run_manifest.json`
- reduced 外部参照仍见 `figures/archive/analysis/experiment_comparison_overview_v2/external_supervision_reference.png`
