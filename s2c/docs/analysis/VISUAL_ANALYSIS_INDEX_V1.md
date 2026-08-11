# 可视化分析索引 V1

更新时间：2026-08-10
活动协议：`protocol_v2_textoir_v1`

本索引把已有图按“性能—表示—决策—外部合同”四层组织，避免只看一张 OOS F1 表。图表均为已完成实验或 analysis-only 后验分析，不使用测试集选择新参数。

统一收口入口：[`UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`](UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md)；机器可读图索引：`results/analysis/archive/analysis/unified_comparison_v1/figure_manifest.json`。

> 如果只想回答“历史 Ours 是什么、它超过了谁、当前 S2C 与 MOGB 到底怎么比、MOGB 为什么没复现”，
> 先读 [`S2C_BASELINE_MOGB_COMPARISON_OVERVIEW_V1.md`](S2C_BASELINE_MOGB_COMPARISON_OVERVIEW_V1.md)。

> 如果只想看当前阶段的一页总览（当前 fair 矩阵、StackOverflow 外部 ADB 参照、MOGB 论文/本地差距、历史 Cascade 优势），
> 先读 [`EXPERIMENT_COMPARISON_OVERVIEW_V2.md`](EXPERIMENT_COMPARISON_OVERVIEW_V2.md)，并打开
> `figures/archive/analysis/experiment_comparison_overview_v2/` 下的四张图。

> 如果只想回答“当前自有方法为什么更好、MOGB 的损失来自哪里、哪些结论仍不能下”，
> 先读 [`MECHANISM_CLOSURE_V1.md`](MECHANISM_CLOSURE_V1.md)，并打开
> `figures/archive/analysis/mechanism_closure_v1/mechanism_closure_dashboard.png`。

> 如果要区分“收益来自 Trainable 表示还是 Gate 决策器”，请读
> [`DETECTOR_MECHANISM_ANALYSIS_V1.md`](DETECTOR_MECHANISM_ANALYSIS_V1.md)，并打开
> `figures/archive/analysis/detector_mechanism_v1/native_detector_frontier.png`。

## 一、建议首先查看的 6 张主图

| 层次 | 图 | 回答的问题 |
|---|---|---|
| 性能 | `figures/archive/analysis/experiment_analysis_master_v1/oos_f1_heatmap.png` | 三数据集、三 KIR 下谁整体稳定 |
| 性能 | `figures/archive/analysis/experiment_analysis_master_v1/pareto_oos_f1_f1_all_kir050.png` | OOS F1 提升是否以 Known 分类退化为代价 |
| 稳定性 | `figures/archive/analysis/experiment_analysis_master_v1/oos_f1_kir_curves.png` | 开放比例增加时谁退化更快 |
| 决策 | `figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_error_source_waterfall.png` | 固定 K=2 的主要错误来自 OOS 误接收还是 Known 误拒绝 |
| 决策 | `figures/archive/analysis/representation_geometry_visuals_v1/score_gap_false_accept.png` | 表示训练是否改善 score separation，而不只是收紧阈值 |
| 合同 | `figures/archive/analysis/baseline_contract_visuals_v1/stackoverflow_baseline_pareto_contract.png` | ADB/DA-ADB/DCLOOS 的数值为什么不能直接混排 |
| 监督条件 | `figures/archive/analysis/experiment_comparison_overview_v2/external_supervision_reference.png` | DCLOOS reduced 与 S2C/ADB 的监督、KIR、骨干和预算差异；只作数值参照 |
| 机制闭环 | `figures/archive/analysis/mechanism_closure_v1/mechanism_closure_dashboard.png` | 当前自有方法的性能、Known/OOS 正确预算和 MOGB 工作点一次收束；不是新训练结果 |
| 机制闭环 | `figures/archive/analysis/mechanism_closure_v1/performance_operating_curves.png` | 3 数据集×3 KIR 的 OOS F1、F1-All、Known Recall 曲线 |
| 机制闭环 | `figures/archive/analysis/mechanism_closure_v1/trainable_mogb_error_budget.png` | Trainable-K1 相对 MOGB-Fair 的 Known 恢复与 OOS 正确数代价 |
| 机制闭环 | `figures/archive/analysis/mechanism_closure_v1/mogb_risk_workpoints.png` | MOGB 组件的 false acceptance/false rejection 与粒球支持风险 |
| Detector 归因 | `figures/archive/analysis/detector_mechanism_v1/native_detector_comparison.png` | 同一 Trainable MiniLM 表示下 Gate、MSP、Energy、kNN、LOF 的 OOS/覆盖比较 |
| Detector 归因 | `figures/archive/analysis/detector_mechanism_v1/native_detector_frontier.png` | OOS F1 与 false acceptance 的方法前沿，突出 StackOverflow 的差异 |
| Detector 归因 | `figures/archive/analysis/detector_mechanism_v1/representation_vs_detector_gain.png` | Trainable 相对 Frozen 的原生 detector 表示增益 |
| 外部合同 | `figures/archive/analysis/trainable_vs_adb_kir_v1/trainable_vs_adb_kir_delta_heatmaps.png` | 3 数据集×3 KIR×5 seed 下 Trainable-K1 与 ADB 的多指标差值；同 split，不同 BERT/MiniLM 合同 |
| 外部合同 | `figures/archive/analysis/trainable_vs_adb_kir_v1/trainable_vs_adb_f1_all_false_acceptance.png` | Trainable-K1 与 ADB 的 F1-All—false acceptance 工作点 |
| 外部合同错误预算 | `figures/archive/analysis/trainable_vs_adb_error_budget_v1/state_error_budget.png` | StackOverflow/KIR=.50 三 seed 同测试样本的 Known/OOS 错误状态预算 |
| 外部合同错误预算 | `figures/archive/analysis/trainable_vs_adb_error_budget_v1/state_transition_heatmap.png` | Trainable-K1 与 ADB 的逐样本状态转移；仅导出聚合计数 |
| 外部合同错误预算 | `figures/archive/analysis/trainable_vs_adb_error_budget_v1/known_reject_oos_accept_tradeoff.png` | Known 误拒与 OOS 误接收的直接交换关系 |
| 外部合同错误预算（跨数据集） | `figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/known_rejected_delta_heatmap.png` | Trainable−ADB 的 Known 条件误拒差异；显示数据集/KIR 异质性 |
| 外部合同错误预算（跨数据集） | `figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/oos_false_accept_delta_heatmap.png` | Trainable−ADB 的 OOS 条件误接收差异；区分 CLINC/Banking 与 StackOverflow |
| 外部合同错误预算（跨数据集） | `figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/known_rejection_vs_oos_accept_frontier.png` | Known 覆盖与 OOS 拒识的工作点前沿 |
| 外部合同错误预算（跨数据集） | `figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/error_budget_by_dataset_kir.png` | 三数据集×三 KIR 的五状态错误预算 |
| 外部合同机制合并 | `figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_mechanism_heatmaps.png` | 将 OOS F1/F1-All 与 Known/OOS 错误预算按 dataset×KIR 对齐 |
| 外部合同机制合并 | `figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_error_budget_quadrants.png` | 区分保守拒识、覆盖恢复和覆盖—误接收权衡 |
| OOS 指标分解 | `figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_decomposition_heatmaps.png` | OOS precision、recall、F1 和 Known acceptance 的差值热图 |
| OOS 指标分解 | `figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_workpoints.png` | 三数据集的 OOS precision–recall 工作点轨迹 |
| 意图级错误归因 | `figures/archive/analysis/trainable_vs_adb_intent_error_v1/known_rejection_delta_by_intent.png` | 各数据集/KIR 的 Known 误拒差异按真实意图展开 |
| 意图级错误归因 | `figures/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_budget_scatter.png` | 意图级 Known 覆盖变化与 OOS 误接收变化的联合工作点 |
| 意图级错误归因 | `figures/archive/analysis/trainable_vs_adb_intent_error_v1/top_intent_oos_acceptor_changes.png` | OOS 被哪些预测意图吸收最多或最少 |
| Detector 归因 | `figures/archive/analysis/detector_mechanism_v1/gate_vs_native_oos_f1_ci.png` | 10,000 次配对 bootstrap 下 Gate 相对原生 detector 的 OOS F1 区间 |
| Detector 归因 | `figures/archive/analysis/detector_mechanism_v1/trainable_vs_frozen_native_oos_f1_ci.png` | 10,000 次配对 bootstrap 下 Trainable 相对 Frozen 的原生 detector 增益 |
| 跨 KIR 工作点 | `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_delta_heatmaps.png` | 在 Known Recall≈80/90/95% 时，S2C 相对 MOGB-Fair 的 OOS F1 差异是否仍存在 |
| 跨 KIR 工作点 | `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_oos_f1_curves.png` | 三数据集×三 KIR 的匹配覆盖率前沿；区分排序优势与默认阈值优势 |
| 跨 KIR 错误预算 | `figures/archive/analysis/cross_kir_transition_attribution_v1/cross_kir_transition_attribution.png` | Trainable-K1 相对 MOGB-Fair 的 F1-All、Known 净增、OOS 净增和 Known 恢复来源 |
| 跨 KIR 错误预算 | `figures/archive/analysis/cross_kir_transition_attribution_v1/known_oos_budget_cross_kir.png` | 三数据集随 KIR 变化时，Known 覆盖恢复与 OOS 正确拒识的相对贡献 |
| 跨 KIR 多指标前沿 | `figures/archive/analysis/cross_kir_pareto_frontier_v1/cross_kir_pareto_frontier.png` | 315 个 fair Gate 行的 OOS F1、F1-All、Known Recall 与 false acceptance 前沿及五 seed 误差线 |
| 跨 KIR 多指标前沿 | `figures/archive/analysis/cross_kir_pareto_frontier_v1/pareto_cell_counts.png` | 各方法进入多指标 Pareto 前沿的 dataset×KIR 工作点数量 |
| MOGB 归因 | `figures/archive/analysis/mogb_known_calibration_attribution_v1/known_recall_oos_f1_workpoints.png` | MOGB 默认半径与 Known-only 校准工作点为何都弱于当前 Trainable K=1 |
| MOGB 归因 | `figures/archive/analysis/mogb_known_calibration_attribution_v1/subcentroid_loss_signal.png` | 官方 L1 距离归一化为何压缩子中心训练信号 |
| MOGB BERT 配对 | `figures/s2c_baseline_mogb_overview_v1/mogb_published_local_corrected_metrics.png` | 论文公开值、本地官方逻辑与只改损失单格的差距 |
| MOGB BERT 配对 | `figures/s2c_baseline_mogb_overview_v1/mogb_error_budget.png` | 修正损失后恢复多少 Known、又误收多少 OOS |
| MOGB BERT 配对 | `figures/s2c_baseline_mogb_overview_v1/mogb_training_contract_curves.png` | 官方损失近似常数与 raw-distance 损失有效下降的训练对照 |
| MOGB 筛选归因 | `figures/archive/analysis/mogb_selected_class_rescue_v1/known_recovery_vs_oos_cost.png` | 回补被筛掉的 Known 类时，恢复 Known 与新增误收 OOS 的样本级代价 |
| MOGB 筛选归因 | `figures/archive/analysis/mogb_selected_class_rescue_v1/rescue_vs_trainable_k1.png` | 缺类回补是否缩小当前 S2C 与 MOGB-Fair 的同协议差距 |
| MOGB 粒球归因 | `figures/archive/analysis/mogb_ball_risk_attribution_v1/top10pct_ball_error_concentration.png` | 错误是否集中在少数危险粒球 |
| MOGB 粒球归因 | `figures/archive/analysis/mogb_ball_risk_attribution_v1/support_quartile_error_budget.png` | tiny ball 还是较大粒球承担主要错误预算 |
| MOGB 粒球归因 | `figures/archive/analysis/mogb_ball_risk_attribution_v1/ball_risk_vs_trainable_gap.png` | 粒球风险与 Trainable K1 性能差距是否同步变化 |
| 开放意图转移 | `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/open_intent_transition_heatmaps_kir050.png` | S2C 与 MOGB 的 Known正确/错类/拒绝和 OOS正确/误收如何逐样本转移 |
| 开放意图转移 | `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/paired_correctness_gain_decomposition.png` | S2C 优势来自 Known 恢复还是 OOS 拒识 |
| 开放意图转移 | `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/known_recovery_by_mogb_score_bin.png` | MOGB 边界外的 Known 样本有多少仍被 S2C 正确分类 |
| S2C--MOGB 总览 | `figures/s2c_vs_mogb_mechanism_dashboard_v1/paired_delta_heatmaps.png` | 45 个严格配对单元的六指标差值集中在哪里 |
| S2C--MOGB 总览 | `figures/s2c_vs_mogb_mechanism_dashboard_v1/error_budget_arrows.png` | 两种方法在 Known 覆盖、OOS precision/recall 上如何移动工作点 |
| S2C--MOGB 总览 | `figures/s2c_vs_mogb_mechanism_dashboard_v1/component_bridge_f1_all.png` | MOGB 分区、S2C 边界和 Trainable 表示各恢复多少 F1-All |
| 排序/工作点 | `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/threshold_free_delta_heatmap.png` | 优势是否在阈值无关AUROC/AUPR上仍成立 |
| 排序/工作点 | `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png` | 相同Known覆盖下谁的OOS检测前沿更好 |
| 排序/工作点 | `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/calibration_gap_heatmap.png` | 默认阈值离事后最优工作点有多远 |

## 二、表示层证据

- `figures/archive/analysis/representation_geometry_visuals_v1/geometry_tradeoff.png`：effective rank、relative separation 与 K=2 变化的关系。
- `figures/archive/analysis/representation_geometry_visuals_v1/near_oos_delta.png`：Frozen、CE、SupCon 的 near-OOS 变化。
- `figures/archive/analysis/representation_geometry_visuals_v1/stackoverflow_error_quadrants.png`：Trainable 与 MOGB 逐样本错误互补性。
- `figures/archive/analysis/native_baselines_trainable_v1/kir050_trainable_representation_tradeoff.png`：固定 Trainable 表示下 Gate、MSP、Energy、kNN、LOF 的检测器差异。

当前解释：Trainable MiniLM 的主要收益在 K=1 的分数排序和覆盖—拒识平衡；几何改善并没有自动修复固定多中心的 acceptance-union 风险。

## 三、决策层和错误来源

- `figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_oos_transition_heatmap.png`：同一 OOS 样本在不同方法间的正确/错误转移。
- `figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_intent_oos_acceptor_heatmap.png`：哪些 intent 吸收了 OOS。
- `figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_intent_known_reject_heatmap.png`：哪些 Known intent 被过度拒绝。
- `figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/intent_error_delta_scatter_kir050.png`：三数据集逐 intent 风险异质性。
- `figures/archive/analysis/raw_gate_error_visualization_v1/roc_pr_curves.png`：score 排序层面的 ROC/PR 证据。
- `figures/archive/analysis/oos_error_budget_v1/oos_precision_recall_workpoints.png`：OOS precision/recall 工作点。

StackOverflow/KIR=.50 的核心事实：Trainable K=1 的 OOS F1/FA/FR 为 87.67/9.34/16.11%；Frozen K=2 为 63.53/47.17/13.11%；MOGB-MiniLM 为 72.92/0.79/72.91%。因此 MOGB 组件的低 FA 主要来自保守拒识，固定 K=2 的失败主要来自 OOS 过接收。

## 四、跨 KIR 与方法对比

- `figures/archive/analysis/kir_sensitivity_decomposition_v1/oos_f1_kir_sensitivity.png`：OOS F1 曲线。
- `figures/archive/analysis/kir_sensitivity_decomposition_v1/coverage_oos_trajectory_by_kir.png`：Known Recall—OOS F1 轨迹。
- `figures/archive/analysis/fair_effect_landscape_v1/trainable_multimetric_effect_summary.png`：Trainable 相对各组件的多指标配对效应。
- `figures/archive/analysis/statistical_stability_v1/paired_oos_f1_forest.png`：五 seed 配对 bootstrap 区间。
- `figures/archive/analysis/experiment_analysis_master_v1/method_rank1_counts.png`：跨九个 dataset×KIR 单元的排名稳定性。

## 五、外部 baseline 的边界

`figures/archive/analysis/baseline_contract_visuals_v1/` 中的 ADB、DA-ADB、DCLOOS 和 MOGB 官方行只用于显示合同差异：表示、监督、数据划分、seed 数或指标层级不同。当前不能据此宣称 Trainable 已超过完整 SOTA。

ADB 三数据集 KIR=.50 外部参照图：

- `figures/archive/analysis/adb_cross_dataset_v1/adb_vs_trainable_oos_f1.png`：CLINC150、Banking77、StackOverflow 各 3 seed 的 ADB BERT 与 S2C Trainable K=1 绝对 OOS F1；
- `figures/archive/analysis/adb_cross_dataset_v1/adb_vs_trainable_paired_oos_delta.png`：同 seed 配对 OOS F1 差值，展示数据集依赖；
- `results/analysis/archive/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_REPORT.md`：逐 seed、配对摘要和合同边界。

ADB 跨 KIR 扩展（五 seed 完整入口）：

- `tools/analysis/build_adb_kir_sensitivity.py`：读取 KIR=.25/.50/.75 的三数据集外部运行并与 Trainable-K1 配对；
- `figures/adb_kir_sensitivity_v2/adb_kir_curves.png`：五 seed ADB 的 OOS F1、F1-All 和 Known Recall 随 KIR 变化；
- `figures/adb_kir_sensitivity_v2/trainable_minus_adb_oos_f1_heatmap.png`：五 seed S2C Trainable-K1 相对 ADB 的逐数据集/KIR差值；
- `results/analysis/adb_kir_sensitivity_v2/ADB_KIR_SENSITIVITY_REPORT.md`：45/45 外部单元、五 seed 摘要和配对差异。三 seed v1 已归档，不作为当前入口。

跨 KIR 合同可视化总览（五 seed 当前入口）：

- `tools/analysis/build_cross_kir_contract_atlas.py`：由冻结 fair 汇总和五 seed ADB 摘要生成合同分层图；
- `figures/cross_kir_contract_atlas_v2/cross_kir_contract_atlas.png`：按数据集并列展示 S2C Trainable-K1、MOGB-MiniLM-Fair 与 ADB 的 OOS F1、F1-All 和 Known Recall；每个图例保留 backbone、监督合同和 seed 数；
- `results/analysis/cross_kir_contract_atlas_v2/CROSS_KIR_CONTRACT_ATLAS_REPORT.md` 与 `rows.csv`：保留源哈希、合同说明和机器可读行。三 seed v1 已归档，不作为当前入口。

ADB 五 seed 配对推断（已完成 36/36 指标格）：

- `figures/archive/analysis/adb_paired_inference_v1/trainable_minus_adb_paired_forest.png`：OOS F1、F1-All、Known Recall 和 false acceptance 的五 seed 配对均值与 95% bootstrap CI；
- `results/analysis/archive/analysis/adb_paired_inference_v1/ADB_PAIRED_INFERENCE_REPORT.md`：固定 RNG、10,000 次 bootstrap、胜平负、sign test 和 Cohen dz；
- `results/analysis/archive/analysis/adb_paired_inference_v1/paired_inference.csv`：机器可读的 dataset×KIR×metric 配对统计。

外部 baseline 状态：

- MOGB 官方 BERT：已完成兼容单格，但未达到论文数字；
- ADB：已完成 CLINC150、Banking77、StackOverflow 各 3 seed 的 KIR=.50 BERT/TextOIR 外部参照；三数据集摘要和配对差值见 `results/analysis/archive/analysis/adb_cross_dataset_v1/` 与 `results/analysis/archive/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_REPORT.md`。它仍不进入 MiniLM fair 主排名；
- DA-ADB：仍为历史兼容单格/无效预测，不能引用 runner 的伪高分；
- DCLOOS：reduced-budget 结果存在，但使用伪 OOS/外部 OOS，不进入 Known-only fair 主表。

详细数字和证据边界见 [`EXPERIMENT_ANALYSIS_MASTER_V1.md`](EXPERIMENT_ANALYSIS_MASTER_V1.md) 与 [`BASELINE_EXECUTION_STATUS_V1.md`](BASELINE_EXECUTION_STATUS_V1.md)。

## 六、当前协议 Cascade bridge

这些图使用当前 `protocol_v2_textoir_v1` 的 Known-only Expert，配对比较 Frozen K=1 与 Trainable K=1，
不混入历史 `fulltex.tex` Cascade：

- `figures/archive/analysis/cascade_bridge_cross_dataset_v1/trainable_vs_frozen_cascade_metrics.png`：三数据集的 OOS F1、F1-All、Known Recall 和 false acceptance。
- `figures/archive/analysis/cascade_bridge_cross_dataset_v1/trainable_minus_frozen_cascade_effect.png`：Trainable 相对 Frozen 的配对差值。
- `figures/archive/analysis/cascade_bridge_cross_dataset_v1/cascade_error_budget_cross_dataset.png`：Gate 与 Expert 错误预算分解。

它们支持的结论是 Trainable K=1 的 Gate 优势能够传递到当前协议下游；不支持跨合同 SOTA 排名。

### MOGB 论文差距的直接证据

- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/ce_vs_subcentroid_loss.png`：CE 已收敛，但公开子中心损失受 L1 距离归一化约束，长期停留在接近均匀分类的窄区间。
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/dev_accuracy_vs_known_recall.png`：Known dev accuracy 约 92%，最终 Known Recall 却只有约 44%–52%，定位到边界覆盖而非纯闭集分类失败。
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/paper_gap_metrics.png`：论文公开参考与本地官方逻辑兼容单格的四指标差距；Banking 合同未完全对齐，只作描述。
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/ball_size_radius_distribution.png`：动态粒球确实生成，排除“实际退化为单中心”的解释。
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/five_seed_short_run_stability.png`：十个五 epoch 兼容运行仅证明执行稳定，不能作为严格论文复现。

中文解释与数学动态范围计算见 [`MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`](MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md)。

### MOGB Known-only 校准与损失公式实证

- `figures/archive/analysis/mogb_known_calibration_attribution_v1/radius_multiplier_by_coverage.png`：45 个同协议单元上，为达到 80%–95% Known calibration 覆盖所需的半径倍率。
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/cal95_delta_vs_default.png`：恢复 Known coverage 与 OOS false acceptance 增长的直接权衡。
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/trainable_k1_vs_mogb_cal80.png`：当前 S2C Trainable K=1 相对预注册 MOGB cal-80 的跨 KIR 配对差值。

该实验 45/45 重放与冻结 `mogb_minilm` score/指标等价；校准只使用 `calibration_known`，没有用 test Known/OOS 选择工作点。完整报告见 [`MOGB_KNOWN_CALIBRATION_ATTRIBUTION_V1.md`](MOGB_KNOWN_CALIBRATION_ATTRIBUTION_V1.md)。

### corrected-loss BERT-MOGB 半径覆盖诊断

- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/radius_workpoints.png`：Known-dev覆盖目标、全局半径倍率、
  test Known Recall 与F1-U的工作点曲线。
- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/false_accept_top_balls.png`：默认半径与cal-95下，OOS误接受
  增量最大的selected balls。

该图组复用 corrected-loss checkpoint，不重训、不用 test OOS 选半径；但源运行没有保存最终粒球随机
状态，因此属于确定性固定-checkpoint重建，不是原球结构严格重放。完整报告见
[`MOGB_CORRECTED_RADIUS_COVERAGE_V1.md`](MOGB_CORRECTED_RADIUS_COVERAGE_V1.md)。

### MOGB selected-ball 缺类回补

- `figures/archive/analysis/mogb_selected_class_rescue_v1/affected_cell_metric_delta.png`：仅在16个确实缺类的单元上显示回补前后多指标变化；其余29个单元严格零变化。
- `figures/archive/analysis/mogb_selected_class_rescue_v1/known_recovery_vs_oos_cost.png`：每个受影响单元的 Known 恢复量与新 OOS 误接收量。
- `figures/archive/analysis/mogb_selected_class_rescue_v1/missing_class_correct_recovery.png`：被遗漏类的直接分类恢复。
- `figures/archive/analysis/mogb_selected_class_rescue_v1/rescue_vs_trainable_k1.png`：KIR=.50 的 MOGB cal-80、回补版和 S2C Trainable K=1 工作点。

StackOverflow 的 cal-80 回补使 OOS F1 平均下降15.66个百分点并使 false acceptance 增加18.61个百分点。完整报告见 [`MOGB_SELECTED_CLASS_RESCUE_V1.md`](MOGB_SELECTED_CLASS_RESCUE_V1.md)。

### MOGB 逐粒球风险归因

- `figures/archive/analysis/mogb_ball_risk_attribution_v1/top10pct_ball_error_concentration.png`：每单元最高风险10%粒球承担的 OOS 误接收和 Known 误拒绝比例。
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/ball_support_radius_risk.png`：训练支持、结构半径与两类错误的逐球关系。
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/support_quartile_error_budget.png`：按训练支持四分位拆分错误预算，检验 tiny-cluster 假设。
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/ball_risk_vs_trainable_gap.png`：MOGB-Fair、共享粒球+S2C边界与 Trainable K1 的跨数据集差距。
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/intent_ball_risk_heatmap.png`：KIR=.50 下不同 Known intent 的粒球数量、误接收和误拒绝风险。

结论是错误集中但并非由 tiny ball 主导；粒球纯度也不能预测开放空间风险。完整报告见 [`MOGB_BALL_RISK_ATTRIBUTION_V1.md`](MOGB_BALL_RISK_ATTRIBUTION_V1.md)。

### S2C 与 MOGB-Fair 完整开放意图结果转移

- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/open_intent_transition_heatmaps_kir050.png`：KIR=.50下三数据集五状态逐样本转移矩阵。
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/paired_correctness_gain_decomposition.png`：各dataset×KIR的净Known正确增量与净OOS正确拒绝增量。
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/known_recovery_by_mogb_score_bin.png`：按MOGB归一化分数区间观察两方法Known正确率。
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/top_intent_known_recovery_kir050.png`：至少3/5 seeds进入Known集合的意图中，S2C恢复幅度最大的意图。

这组图说明S2C的F1-All优势来自大量恢复MOGB拒绝的Known样本，同时接受有限的OOS拒识损失；完整报告见[`TRAINABLE_MOGB_OPEN_INTENT_TRANSITIONS_V1.md`](TRAINABLE_MOGB_OPEN_INTENT_TRANSITIONS_V1.md)。

### Trainable-K1 与 MOGB-Fair 错误预算重算

- `figures/archive/analysis/trainable_mogb_error_budget_v1/error_budget_decomposition.svg`：按 dataset×KIR 显示净 Known
  正确恢复、净 OOS 正确拒绝变化和 F1-All 增量；
- `results/analysis/archive/analysis/trainable_mogb_error_budget_v1/budget_summary.csv`：9 个 dataset×KIR 组的五 seed 均值、
  标准差和正向 seed 数；
- `results/analysis/archive/analysis/trainable_mogb_error_budget_v1/integrity.json`：45 行逐样本转移分解的算术一致性审计。

完整中文说明见 [`TRAINABLE_MOGB_ERROR_BUDGET_V1.md`](TRAINABLE_MOGB_ERROR_BUDGET_V1.md)。

### Trainable-K1 与 MOGB-Fair 的 KIR 趋势

- `figures/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.svg`：OOS F1、F1-All、Known Recall 三面板的 KIR 趋势；
- `results/analysis/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.csv`：当前方法和 MOGB-Fair 的绝对值、逐 KIR 差值、
  seed 数和源哈希。

该图显示 Trainable-K1 相对 MOGB-Fair 的覆盖/排序优势并非只出现在单个 KIR；StackOverflow 的差距随 KIR
扩大最明显。完整中文说明见 [`TRAINABLE_MOGB_KIR_TREND_V1.md`](TRAINABLE_MOGB_KIR_TREND_V1.md)。

### S2C 与 MOGB-Fair 45 单元机制总览

- `figures/s2c_vs_mogb_mechanism_dashboard_v1/paired_delta_heatmaps.png`：三个数据集、三个 KIR 的
  OOS F1、F1-All、Known Recall、precision、recall 和 false acceptance 配对差值。
- `figures/s2c_vs_mogb_mechanism_dashboard_v1/oos_precision_recall_decomposition.png`：S2C 的 OOS F1
  优势如何由 precision 提升和 recall 损失共同构成。
- `figures/s2c_vs_mogb_mechanism_dashboard_v1/paired_effect_forest.png`：五 seed、10,000 次配对
  bootstrap 的 OOS F1、F1-All 和 Known Recall 区间。
- `figures/s2c_vs_mogb_mechanism_dashboard_v1/component_bridge_f1_all.png`：固定 MOGB 分区后替换边界，
  再替换为 Trainable K1 时的组件桥。

完整中文报告见 [`S2C_VS_MOGB_MECHANISM_DASHBOARD_V1.md`](S2C_VS_MOGB_MECHANISM_DASHBOARD_V1.md)。

### S2C 与 MOGB-Fair 排序能力和工作点归因

- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/threshold_free_delta_heatmap.png`：九个
  dataset×KIR 设置的 AUROC、AUPR-OOS 和标准化分数分离差值。
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/default_vs_oracle_oos_f1.png`：默认阈值与各自
  事后最优阈值的 OOS F1 上限；仅作测试敏感性分析。
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png`：KIR=.50 时
  50%--95% Known coverage 下的 OOS F1 前沿。
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/score_quantiles_kir050.png`：Known/OOS分数10%--90%
  区间和正式阈值1的位置。
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/calibration_gap_heatmap.png`：默认到oracle的校准损失。
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/ranking_vs_workpoint.png`：AUROC差值与默认OOS F1差值。

完整报告见 [`S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1.md`](S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1.md)。

### S2C 与 MOGB-Fair 逐意图结构--收益桥接

- `figures/archive/analysis/s2c_mogb_intent_structure_bridge_v1/intent_known_recovery_distribution.png`：逐 intent Known
  覆盖恢复的全分布，验证收益是否广泛存在。
- `figures/archive/analysis/s2c_mogb_intent_structure_bridge_v1/ball_count_vs_recovery.png`：selected-ball 数量与 Known
  恢复幅度，显示不存在统一单调关系。
- `figures/archive/analysis/s2c_mogb_intent_structure_bridge_v1/known_recovery_concentration.png`：Top intents 的恢复贡献和
  Gini，排除少数类别主导解释。
- `figures/archive/analysis/s2c_mogb_intent_structure_bridge_v1/known_recovery_vs_oos_cost.png`：九个 dataset×KIR 的 Known
  覆盖恢复与新增 OOS 接受样本数。

完整报告见 [`S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md`](S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md)。

## 六、当前唯一实验重点

### 历史 Cascade 与当前 Trainable-K1 的合同差距

- `figures/archive/analysis/cross_contract_gap_v1/current_vs_historical.svg`：逐格对齐 3 个数据集 × 3 个 KIR 的历史
  `fulltex.tex` 完整 Cascade 与当前 Trainable-K1，并在下方显示描述性 OOS F1 差值。
- `results/analysis/archive/analysis/cross_contract_gap_v1/current_vs_historical.csv`：包含源合同、五 seed 当前均值、历史值、
  差值和源文件哈希对应关系。

这组图只回答“当前自有方法与历史论文方法是否是同一个对象”，不把历史 Cascade、MOGB 论文、ADB 或
DCLOOS reduced 结果混成一个排名。完整中文说明见 [`CROSS_CONTRACT_GAP_V1.md`](CROSS_CONTRACT_GAP_V1.md)。

结论口径审计见 [`RESULT_CLAIM_AUDIT_V1.md`](RESULT_CLAIM_AUDIT_V1.md)。

继续补同协议外部 baseline 的可运行单格和多 seed，并沿用上述四层图表输出；不要重复 E2/E3、扩大 K、增加新损失项或把不同监督合同强行合并。
