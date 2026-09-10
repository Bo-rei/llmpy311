# 关键图索引

更新时间：2026-09-03

这里只保留 OOS 主线需要阅读的图。Known intent 分类图仍不进入活跃入口；一个严格限定为
OOS gold subtype 残余风险的诊断图保留在新 H1 机制 bundle 中。历史 bundle 仍保留在原路径，
但不与当前主线混排。

协议边界先记清楚：后续历史主线使用 `historical_v19_paper_main`，当前首选图片来自
[`historical_oos_visual_explanation`](../../results/analysis/historical_oos_visual_explanation/MANIFEST.json)，
其四张核心图是局部边界、score 分布、配对 crossing 和精确 score 分解。
下面三张低阅读成本图属于冻结的 `protocol_v2_textoir_v1` 参考，面板中的 `Banking77` 是标准
`banking77`，不是历史主表的 `BANKING77-OOS`；两组图不能混排。

## 低阅读成本视觉合同（当前生效）

版式参考 `oos_intent论文.pdf` 后部的 Figure 3–4（PDF 第 7–8 页）。参考重点是信息层级，
不是复制具体配色：每张图只回答一个问题，阈值位置明确，图例短，正文结论不依赖读者逐点扫描。

- **主文最多三类图。** 顺序固定为 `OOS score 分布 → OOS 工作点 → OOS 机制`；一张图最多
  3 个数据集小面板，默认 1×3，不使用 3×3、3×4 或多方法全矩阵作为主图。
- **OOS 是信号色。** OOS 使用单一强调色，Known/ID 仅用低饱和中性色作必要参照；阈值
  `normalized score = 1` 使用一条黑色虚线。不要同时叠加“方法 × 正确性 × Known/OOS × 状态”四层编码。
- **标题直接给读图问题。** 每个面板最多保留一个关键统计标注（例如 OOS false acceptance、
  OOS correct rate 或相关系数）；样本数和精确数值放在图注或源表，不塞进画布。
- **主图只保留 OOS 结果。** OOS F1、AUROC、AUPR、false acceptance 是主读数；Known Recall
  只作为一个 coverage guard，不展开 intent、Router/Expert 或 Known 错误子类。
- **复杂图降级为补充证据。** 原始点云、per-ball 风险、detector rank transfer、五状态
  转移矩阵和外部 ADB 图只在回答一个明确追问时单独展示；不在同一页连续堆叠。
- **读者看不出结论就换图。** 如果图需要先读长图例、记住多个颜色含义，或必须在多个面板之间
  来回对照才能判断 OOS 好坏，则该图只能留在分析 bundle，不能进入主文入口。

### 冻结 protocol_v2 参考的三类低阅读成本图

| 主图 | 建议版式 | 只回答的问题 | 当前状态 |
|---|---|---|---|
| OOS score 分布 | 1×3 数据集小面板；Known 淡色、OOS 强调色、阈值线 | Trainable 相对 Frozen 是否把 OOS 推向拒绝侧？ | [`oos_score_distribution_kir050.png`](../../figures/oos_readability_figures_v1/oos_score_distribution_kir050.png) |
| OOS 工作点 | 1×3 数据集小面板；只标 OOS F1/false acceptance，coverage 作为旁注 | S2C 的 OOS 收益和 coverage 代价是什么？ | [`oos_operating_point_kir.png`](../../figures/oos_readability_figures_v1/oos_operating_point_kir.png) |
| OOS 机制 | 1×3 数据集小面板；OOS 状态转移率配合 `mean Δscore`，每面板一个关键数字 | 哪些 OOS 状态改善同时伴随拒绝侧 score 移动？ | [`oos_mechanism_transitions_kir050.png`](../../figures/oos_readability_figures_v1/oos_mechanism_transitions_kir050.png) |

三张主图已由 `tools/analysis/build_oos_readability_figures_v1.py` 从冻结源表生成；结果表、输入哈希、输出哈希和逐图合同见
`results/analysis/oos_readability_figures_v1/MANIFEST.json`。每张图同时提供 PNG、TIFF、SVG 和 PDF；主图只使用
`Frozen E2 K=1` 与 `Trainable K=1`，不把 78.64% 的 Euclidean MOGB-Fair single-centroid 组件混入同几何比较。

### 当前历史 v19 H1 OOS 机制 bundle

新 bundle 由 `tools/analysis/build_historical_oos_visual_explanation.py` 生成，固定使用 H1
controlled v19、K=1、对角 Mahalanobis 和 score=1；结果与图清单见
`results/analysis/historical_oos_visual_explanation/MANIFEST.json`。图件按“局部几何 → score
分布 → 样本 crossing → distance/radius 分解 → 跨数据集密度 → 难度剖面 → KIR/seed 稳定性 →
残余 subtype 风险 → 表示位移”组织，汇报入口为 `RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md`。

Fig.1 的主图现为 [`paper_style_local_boundary_geometry_3d.png`](../../figures/historical_oos_visual_explanation/paper_style_local_boundary_geometry_3d.png)：
使用共同 PCA-3D 的局部白化坐标和半透明 score=1 单位球，并把真实 384 维 OOS accepted/rejected 状态分开标记；
二维版本仅作为补充。对应真实判决计数见
`results/analysis/historical_oos_visual_explanation/local_boundary_geometry_3d_summary.csv`。

新增的 [`oos_representation_movement_map.png`](../../figures/historical_oos_visual_explanation/oos_representation_movement_map.png)
只显示 Gate 决策发生变化的 OOS：线段表示 Frozen 到 Trainable 的共同 PCA 位移，红/蓝端点分别表示
OOS 修复和退化；它不使用 sample-level 文本，也不把 PCA 当作高维判决空间。

### 当前 H1 Trainable full pipeline 结果图

当前 Trainable checkpoint 已通过既有 v19 Router/Expert 完成 3 数据集×3 seed 的 9 个 full-pipeline
推理单元。结果和 device/protocol 记录见
`results/analysis/historical_trainable_full_pipeline/MANIFEST.json`；图只作为结果辅助：

- [`full_pipeline_workpoint.png`](../../figures/full_pipeline_visual_explanation/full_pipeline_workpoint.png)：Frozen K=1 与当前 H1 Trainable K=1 的 OOS F1 和最终 macro F1；
- [`full_pipeline_error_paths.png`](../../figures/full_pipeline_visual_explanation/full_pipeline_error_paths.png)：Gate、Router、Expert 的错误路径；
- [`full_pipeline_gate_to_pipeline.png`](../../figures/full_pipeline_visual_explanation/full_pipeline_gate_to_pipeline.png)：Gate OOS F1 与最终 full-pipeline macro F1 的关系；
- [`full_pipeline_gate_variants.png`](../../figures/full_pipeline_visual_explanation/full_pipeline_gate_variants.png)：已有 Gate 变体的完整 pipeline trade-off。

这些图的核心数值不应从图上估读，优先使用
`results/analysis/full_pipeline_visual_explanation/full_pipeline_paired_effects.csv` 和
`results/analysis/full_pipeline_visual_explanation/full_pipeline_performance.csv`。

### H1 Trainable Gate 参数化 full pipeline 图组

针对“训练后的 MiniLM 作为 Gate 时，哪些边界参数能把完整 pipeline 的 OOS F1 推近或超过论文 Ours”，
新增参数化实验图组。它固定历史 H1 数据、KIR=.50、seed={13,42,87} 和 Router/Expert，先在
validation 选择，再用 test confirmation；`banking77_oos` 仍是论文历史主线数据键。

- [`validation_parameter_frontier.png`](../../figures/historical_trainable_parameter_full_pipeline/validation_parameter_frontier.png)：每个点是一组 K/λ/threshold/acceptance 参数，星标是 validation-selected，虚线是论文 Ours OOS F1。
- [`test_oos_f1_vs_paper_ours.png`](../../figures/historical_trainable_parameter_full_pipeline/test_oos_f1_vs_paper_ours.png)：所有测试候选与论文 Ours 的 OOS F1 对照；红叉只表示后验最高点，不参与选参。
- [`selected_full_pipeline_error_budget.png`](../../figures/historical_trainable_parameter_full_pipeline/selected_full_pipeline_error_budget.png)：selected full pipeline 的 OOS acceptance、Known rejection 和 Expert error。

结果和图合同见 `results/analysis/historical_trainable_parameter_full_pipeline/`，汇报入口为
[`historical_trainable_parameter_full_pipeline_presentation.md`](historical_trainable_parameter_full_pipeline_presentation.md)。
该图组的核心 selected 结果是：CLINC150 `91.04±0.81`、StackOverflow `89.15±1.54`、
Banking77-OOS `91.53±0.05` OOS F1；只有后者在 validation-selected 主结果上超过论文参考值。

### Archive 数据协议下的最新全链路数值

本轮新增的数字结果不与上面的 H1 `banking77_oos` 机制图混排：它使用标准 `banking77`、
archive 数据、`K=1`、`lambda=1` 和关闭 semantic gate，并重新训练 archive Router/Expert。
Frozen/Partial/LoRA 的 9 个 full-pipeline 单元、参数量和 CUDA 证据见
[`HISTORICAL_ARCHIVE_FULL_PIPELINE_REPORT.md`](HISTORICAL_ARCHIVE_FULL_PIPELINE_REPORT.md)，
轻量表见 [`comparison.csv`](../../results/analysis/historical_archive_full_pipeline/comparison.csv)。
上述 H1 图不能直接代表 LoRA 或 tuned archive 的新结果；本轮新增的 pipeline 汇报图见下节。

### Pipeline 实验阶段汇报图组

本轮针对“Gate 单独变好后是否在 pipeline 中保持”的 archive full-pipeline 汇报图位于
`figures/historical_archive_tuned_full_pipeline/`，源数据和图合同见
`results/analysis/historical_archive_tuned_full_pipeline/FIGURE_MANIFEST.json`，完整汇报为
[`historical_pipeline_experiment_presentation.md`](historical_pipeline_experiment_presentation.md)。

- [`pipeline_gate_full_oos_invariance.png`](../../figures/historical_archive_tuned_full_pipeline/pipeline_gate_full_oos_invariance.png)：Gate 与 full pipeline 的 OOS F1 严格一致；
- [`pipeline_oos_f1_vs_paper.png`](../../figures/historical_archive_tuned_full_pipeline/pipeline_oos_f1_vs_paper.png)：当前候选与论文 KIR=.50 `Ours` 的 OOS F1；
- [`pipeline_error_budget_oos_tradeoff.png`](../../figures/historical_archive_tuned_full_pipeline/pipeline_error_budget_oos_tradeoff.png)：OOS acceptance、Known rejection 与 Expert error 的代价分解。
- [`pipeline_workpoint_search_frontier.png`](../../figures/historical_archive_tuned_full_pipeline/pipeline_workpoint_search_frontier.png)：在 Known F1/Accuracy 不低于基线 1 pp 的验证集约束下，K、lambda 和 threshold 工作点如何改变 OOS F1。

### 跨方法 OOS 比较与机制图组

跨方法图组的汇报入口为 [`cross_method_oos_mechanism_presentation.md`](cross_method_oos_mechanism_presentation.md)，
机器清单为 `results/analysis/cross_method_oos_mechanism/MANIFEST.json`。该组图使用标准
`banking77`、CLINC150 和 StackOverflow 的 MiniLM fair 结果，并将 TextOIR native BERT、ADB/DA-ADB
外部兼容和 DCLOOS 监督状态单独展示：

- `fair_oos_boundary_tradeoff.png`：OOS F1 与 Known coverage 的工作点；
- `fair_oos_precision_recall.png`：OOS precision/recall 与 iso-F1 的平衡；
- `textoir_native_oos_kir_stability.png`：MSP/DOC/ADB 的 TextOIR-native KIR 曲线；
- `s2c_vs_adb_oos_frontier.png`：S2C 与 ADB 的 dataset/KIR 外部 frontier；
- `mogb_component_bridge.png`：MOGB partition/boundary 与 S2C Trainable 的组件桥接；
- `oos_error_budget_frontier.png`：MOGB under-coverage 与 S2C coverage trade-off；
- `mogb_ball_risk_surface.png`：MOGB 粒球支持度与局部 OOS 污染；
- `same_sample_mogb_s2c_transitions.png`：同样本状态转移；
- `cross_method_evidence_status.png`：KNNCL、DCLOOS 等未闭合结果的证据状态。

### 历史协议 v3 predecessor 图组

旧 v3 机制图组保留为 supporting evidence：

- [`main_oos_performance.png`](../../figures/historical_protocol_oos_v3/main_oos_performance.png)：Frozen/Trainable 的 OOS F1 与 false acceptance。
- [`main_oos_mechanism.png`](../../figures/historical_protocol_oos_v3/main_oos_mechanism.png)：OOS score 分布、阈值和 OOS 决策转移，并标出 `Δdistance` 与 `Δscore` 的样本级相关性。
- [`supp_training_dynamics.png`](../../figures/historical_protocol_oos_v3/supp_training_dynamics.png)：训练 loss 与 ID-only checkpoint selection score。
- [`supp_distance_radius_score.png`](../../figures/historical_protocol_oos_v3/supp_distance_radius_score.png)：把总 score 变化拆成 distance contribution 和 radius contribution。
- [`supp_oos_geometry.png`](../../figures/historical_protocol_oos_v3/supp_oos_geometry.png)：只显示 OOS 密度的 Frozen/Trainable PCA 视图。

源数据为 `results/analysis/historical_protocol_v3/`。旧的 v1/v2 图仍保留为历史或 rejected draft，不再作为主入口。

历史 v19 汇报使用上面的历史协议对账图；`protocol_v2` 三张图只在明确标注冻结参考协议时使用。
现有图件不删除，只按“历史主线 / 冻结参考 / 补充证据”分层。

## 阅读入口

1. [当前状态](../CURRENT_STATUS.md)：进展、结论和剩余风险。
2. [近期机制分析汇报](RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md)：直接用于汇报，含图片占位和读图说明。
3. [统一对比与机制报告](UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md)：完整综合结果入口。
4. 本页：按问题定位关键图和准确路径。

## 证据库：真实表示、边界和 baseline 失败机制图（不作为主文整包展示）

统一源：`results/analysis/deep_geometry_mechanism_v1/MANIFEST.json`。范围是 3 个数据集、KIR=.50、seed={13,42,87}；图用 seed=42 作为可读代表，聚合表保留三 seed。原始 embedding 只在构图进程内使用，不写入轻量结果。

| 图 | 位置 | 直接回答的问题 |
|---|---|---|
| 真实空间边界叠加 | `figures/deep_geometry_mechanism_v1/real_space_boundary_overlay.png` | Frozen 的 K=1/MOGB 接受区域与 Trainable K=1 的投影边界，是否覆盖了不同的样本区域？ |
| Frozen→Trainable 位移 | `figures/deep_geometry_mechanism_v1/frozen_trainable_movement.png` | OOS 样本是否发生系统性位移，并有多少样本被推到拒绝侧？ |
| 局部邻域失败图 | `figures/deep_geometry_mechanism_v1/local_neighborhood_failure_map.png` | OOS 即使更像 Known 邻域时，Trainable 是否仍把 score 推到拒绝侧？ |
| 边界组件分解 | `figures/deep_geometry_mechanism_v1/boundary_component_decomposition.png` | OOS score 变化主要来自 nearest distance 的变化，还是来自 radius 的变化？ |

下面五张图不再只是展示最终分数，而是把 OOS 错误映射回接受区域、K=2 几何、ball 和 detector 排序：

| 图 | 位置 | 直接回答的问题 |
|---|---|---|
| 接受区域重叠风险 | `figures/deep_geometry_mechanism_v1/acceptance_overlap_risk.png` | K=2 的接受区域并集具体新增了多少 OOS；MOGB 具体保留了多少 OOS 拒识？ |
| K=2 局部接受区域探针 | `figures/deep_geometry_mechanism_v1/acceptance_surface_exposure.png` | 在 K=2 的局部 mode 内，K=1、K=2 和 MOGB 的 OOS 接受覆盖如何不同？ |
| MOGB ball 支持—风险图 | `figures/deep_geometry_mechanism_v1/mogb_ball_support_risk_map.png` | MOGB 的 OOS 污染是否集中在少数 ball？coverage 只作为 guard 记录。 |
| detector 排序转移 | `figures/deep_geometry_mechanism_v1/detector_rank_transfer.png` | 在完全相同的 Trainable MiniLM 表示上，MSP/Energy/kNN/LOF 与 S2C 的 OOS 排序在哪里分叉？ |
| 多方法 OOS 转移矩阵 | `figures/deep_geometry_mechanism_v1/multi_method_error_transitions.png` | S2C 与各 baseline 对同一 OOS 样本的拒绝/误接收状态如何转移？ |

统一清单仍是 `results/analysis/deep_geometry_mechanism_v1/MANIFEST.json`。当前 OOS 机制图的源表分别是：

- `results/analysis/deep_geometry_mechanism_v1/acceptance_overlap_summary.csv`
- `results/analysis/deep_geometry_mechanism_v1/acceptance_surface_summary.csv`
- `results/analysis/deep_geometry_mechanism_v1/mogb_ball_risk_summary.csv`
- `results/analysis/deep_geometry_mechanism_v1/detector_rank_transfer_summary.csv`
- `results/analysis/deep_geometry_mechanism_v1/multi_method_error_transitions.csv`

StackOverflow/KIR=.50/seed42 的 OOS 结果在 `results/analysis/deep_geometry_mechanism_v1/`：Trainable 相对 Frozen 的
OOS correct rate 为 90.70% vs 80.70%；相对 MOGB，OOS 中 MOGB-only correct 为 8.43%。

## 补充证据：ADB 原生边界机制图

统一源：`results/analysis/adb_deep_mechanism_v1/MANIFEST.json`。范围是 StackOverflow、KIR=.50、seed42；ADB 仍是 BERT/TextOIR external backbone，不能与 MiniLM embedding 共用坐标。

| 图 | 位置 | 直接回答的问题 |
|---|---|---|
| ADB distance/radius 机制 | `figures/adb_deep_mechanism_v1/adb_native_distance_radius.png` | Known coverage guard 与 OOS false accept 分别位于 `distance / radius = 1` 的哪一侧？ |
| ADB 中心空间 OOS 错误图 | `figures/adb_deep_mechanism_v1/adb_centroid_pca_error_map.png` | BERT 表示中的 OOS false accept 是否集中在局部区域？ |
| ADB–S2C OOS 状态转移 | `figures/adb_deep_mechanism_v1/adb_s2c_native_error_transition.png` | 两种方法对同一 OOS 样本的拒绝/误接收是否发生系统性转移？ |
| ADB–S2C OOS score rank transfer | `figures/adb_deep_mechanism_v1/adb_s2c_score_rank_transfer.png` | 两种 backbone 的 OOS score ordering 在哪些样本上分叉？ |

这组图的 probe 源表位于 `results/analysis/adb_deep_mechanism_v1/`。其中 `adb_native_distance_radius.png` 是最重要的边界图，
`adb_s2c_native_error_transition.png` 是最重要的跨方法 OOS 样本解释图。机制 probe 是独立复跑，和 canonical ADB seed42
预测有 `113/6000` 个样本差异；它不替换 `results/analysis/adb_kir_sensitivity_v2/` 的主性能表。

## 补充证据：已有 score、错误和 detector 机制图

统一源：`results/analysis/mechanism_explanation_v1/MANIFEST.json`。

| 图 | 位置 | 用途 |
|---|---|---|
| 同样本 score 重排 | `figures/mechanism_explanation_v1/paired_score_reordering.png` | 观察 Trainable 是否改变 score ordering，而不是只换阈值。 |
| OOS 边界两侧 score mass | `figures/mechanism_explanation_v1/boundary_margin_fingerprint.png` | 观察 OOS score 是否集中在拒绝边界的安全侧。 |
| MOGB ball OOS 风险面 | `figures/mechanism_explanation_v1/mogb_ball_risk_surface.png` | 联合看 ball 支持、半径和 OOS 污染。 |
| 表示—边界耦合 | `figures/mechanism_explanation_v1/geometry_boundary_coupling.png` | 解释表示分离变好后 K=2 仍可能更危险。 |
| 同表示不同 detector 的 OOS 指纹 | `figures/mechanism_explanation_v1/native_detector_error_fingerprint.png` | 区分表示收益与 detector 对 OOS 的排序收益。 |

## 历史/归档证据

这些图仍是有效的历史证据，但不再作为当前 OOS bundle 的新入口；路径保留在这里，避免与当前 OOS 图混淆：

- score 分布：`figures/archive/analysis/raw_gate_error_visualization_v1/score_distributions.png`
- ROC/PR：`figures/archive/analysis/raw_gate_error_visualization_v1/roc_pr_curves.png`
- 表示几何 trade-off：`figures/archive/analysis/representation_geometry_visuals_v1/geometry_tradeoff.png`
- Near-OOS K=1/K=2：`figures/archive/analysis/representation_boundary_pack_v1/representation_near_oos_f1_k1_k2.png`
- OOS 状态转移：`figures/archive/analysis/raw_gate_error_visualization_v1/oos_transitions.png`
- StackOverflow 错误四象限：`figures/archive/analysis/representation_geometry_visuals_v1/stackoverflow_error_quadrants.png`
- 匹配 Known Recall frontier：`figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png`
- S2C–MOGB 配对差值：`figures/s2c_vs_mogb_mechanism_dashboard_v1/paired_delta_heatmaps.png`

## 第三优先级：性能主表和外部合同参照

- Fair 主表：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`
- ADB 外部工作点：`results/analysis/adb_kir_sensitivity_v2/`
- ADB/DA-ADB 样本级机制包：`results/analysis/da_adb_current_protocol_summary_v1/`
- MOGB 机制面板：`figures/s2c_vs_mogb_mechanism_dashboard_v1/`
- 统一逐样本合同与对齐：`results/analysis/unified_prediction_contract_v1/`
- 外部/历史图：`figures/archive/analysis/`，不与 same-protocol fair 图混作排名。

### 外部 backbone 机制图

范围固定为 StackOverflow、KIR=.50、seed={42,87,100}；三种方法共享同一测试文本/标签顺序，但 ADB/DA-ADB 仍是 BERT/TextOIR 外部合同。图只做 OOS 样本状态归因，不把外部表示投影到 MiniLM 空间，也不进入 fair 排名。

| 图 | 位置 | 直接回答的问题 |
|---|---|---|
| 外部 OOS 状态转移 | `figures/da_adb_current_protocol_summary_v1/external_error_transition_heatmaps.png` | S2C 的 OOS 正确/错误样本在 ADB、DA-ADB 中具体变成了哪一种状态？ |
| 外部 OOS 错误构成 | `figures/da_adb_current_protocol_summary_v1/external_error_signature.png` | 外部方法的 OOS 差距来自误接收还是拒识不足？ |

对应机器可读源：

- `results/analysis/da_adb_current_protocol_summary_v1/external_alignment_audit.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/external_error_transitions_summary.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/external_state_composition_summary.csv`

三 seed 聚合结果显示 OOS false acceptance：S2C 为 `8.18%`，ADB 为 `7.52%`，DA-ADB 为 `29.07%`。
这解释了外部 baseline 的 OOS 代价，但由于外部 manifest 未声明 test-selection 且未保存可复用 score/embedding，不能进一步声称差异来自某个单一内部机制。

## 读图边界

- PCA 只用于可视化，不参与训练、阈值、K 或结构选择；边界轮廓是高维边界在二维 PCA 平面上的投影。
- 测试标签只用于事后错误分层；`selection_used_test_oos=false`。
- `movement_transition_summary.csv` 登记 `trainable_vs_frozen` 和 `trainable_vs_mogb` 的聚合转移，不导出 sample_id、文本或 raw embedding。
- `acceptance_surface_summary.csv` 是固定随机方向的高维几何探针，不使用测试标签，也不参与模型、阈值或 K 的选择。

## 审计入口

最新 full-pipeline 实验解释见 [自适应中心与逐样本指标对账](historical_trainable_adaptive_centers_presentation.md)：
分别展示 OOS F1、macro F1、Accuracy 和 Router/Expert 错误；该表格分析没有新增表示空间图片，不替代上述 3D Fig.1。
新训练的选择、full-pipeline 测试、score oracle 和残余 intent 见 [checkpoint 汇报](historical_trainable_checkpoint_selection_presentation.md)，不把单 seed 超过论文包装成稳定提升。

- 资产注册：`configs/experiment_registry.yaml`
- 深层图清单：`results/analysis/deep_geometry_mechanism_v1/MANIFEST.json`
- 运行回放审计：`results/analysis/deep_geometry_mechanism_v1/reconstruction_audit.json`
- 资产检查：`python tools/maintenance/audit_asset_catalog.py`
