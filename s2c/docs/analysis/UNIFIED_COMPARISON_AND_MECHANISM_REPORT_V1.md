# S2C 统一对比实验与机制报告 V1

> 本报告是 analysis-only 收口。它统一已有结果的行级合同、统计入口和证据索引；没有新增模型、元学习、自适应 K、椭球边界或复杂聚合，也没有重跑 E2/E3/MOGB/DCLOOS。

## 1. 结论边界

同协议 Known-only fair 层覆盖 `63` 个 dataset × KIR × method 汇总行，方法为：Frozen K=1, Frozen K=2, MOGB partition + s2c boundary, MOGB-MiniLM, Random K=2, Trainable K=1, s2c partition + MOGB boundary。它是唯一可以直接做 Trainable-K1 与 Frozen/MOGB 组件的公平主层。
统一 prediction contract 已写入 `results/analysis/unified_prediction_contract_v1`；本地逐样本压缩 JSONL 位于 `artifacts/s2c/analysis/unified_prediction_contract_v1`，只含匿名 `sample_id`，不含原始文本或 embedding。共 `2394360` 行、`486` 个运行组。
对齐审计中 `441` 个方法-cell 完全匹配参考样本序列，`45` 个被保留为外部/失败对齐，不会被补成伪 fair row。

## 2. 性能现象

`method_summary.csv` 保留 OOS F1、F1-All、F1-K、Accuracy、Known Recall、false acceptance/rejection、AUROC 和 AUPR-OOS 的均值/标准差。`paired_statistics.csv` 复用同一 `dataset × KIR × seed` 配对单位的 10,000 次 bootstrap、95% CI、win/tie/loss 和 effect size；它只用于事后分析，不用于阈值或方法选择。

当前 fair 层可以回答的是：Known-only MiniLM 表示下，S2C-Trainable-K1 与 Frozen/KIR/MOGB 组件的 coverage–open-space 工作点差异。它不能回答“超过完整 MOGB、完整 DCLOOS 或历史 fulltex SOTA”。

## 3. 样本级错误转移与分数层

统一行级字段保留 `true_label/is_true_oos/predicted_label/predicted_oos/accepted_known`，因此 Known false rejection、OOS false acceptance、wrong-intent 和共同正确/共同错误可以按匿名 sample_id 做后续 paired set 分析。Gate 行额外保留 nearest intent/distance、center/radius/normalized distance；MOGB 行额外保留 ball id/count/size/radius/purity。
已有的 score gap/错误预算、KIR 曲线、Pareto、seed 稳定性和 MOGB risk workpoint 被统一收录到 `figure_manifest.json`。当前 manifest **没有新的 error-aware UMAP**：checkout 中没有可公开登记的统一 embedding artifact，因此不从 score 汇总伪造 UMAP。现有 geometry/error-quadrant 图只用于解释，未进入训练或参数选择。

## 4. 表示、边界与 MOGB 机制

本收口把表示控制和边界控制分开：native MiniLM MSP/Energy/kNN/LOF 是同一 trainable representation 上的检测器控制；MOGB-MiniLM 与 partition/boundary swap 是同协议组件归因；S2C-Trainable-K1 是 Known-only 适配后的单中心 Gate。它们不与 BERT 外部方法混为一个排名。

MOGB 的逐样本 ball 字段来自 `balls.jsonl` 和 `ball_statistics.json`。selected ball 缺类、tiny-ball、radius/purity 风险继续由现有 MOGB 机制图和 `docs/archive/analysis/MOGB_OPERATING_POINT_VISUALS_V1.md` 解释，本阶段不复制或覆盖 MOGB 原始 artifact。

## 5. 监督差异与阻塞项

ADB 的完整外部 BERT 标签结果可被 sample_id 对齐，但其 score 语义和 test-selection 声明不属于同一 MiniLM fair 合同，因此只进入 `external_backbone` 层。DA-ADB 保留已有 current-protocol 外部单元/汇总，不进入 fair paired table。
DCLOOS 当前分为 `official_timeout_no_final_metrics`、`reduced_complete_different_supervision` 和固定 registry 的 `interrupted_no_final_metrics`。reduced 单元含 pseudo-OOS 与外部 SQuAD OOS，且来自 validation-best intermediate prediction；它可以作为监督强度参考，不能与 Known-only 方法直接排名。
KNNCL、OpenMax、DOC、DeepUnk 等传统 TextOIR 路线若只有历史 fulltex 或没有当前最终 metrics，则登记为 blocked/historical，不用中间预测填表。详见 `unified_prediction_contract_v1/blocked_methods.csv`。

## 6. 成本、失败场景和后续停止条件

当前交付优先完成可审计主矩阵、逐样本合同、配对统计和机制索引；没有在 baseline 合同未闭合前扩展 low-resource、threshold robustness 或新的复杂模型。后续只有在同一 split、Known list、seed、评估器和 final metrics 全部闭合后，才可把外部方法提升为 fair 主表。

## 7. 机器可读入口

- 汇总：`results/analysis/archive/analysis/unified_comparison_v1/method_summary.csv`、`results/analysis/archive/analysis/unified_comparison_v1/paired_statistics.csv`。
- 合同：`results/analysis/archive/analysis/unified_comparison_v1/contract_summary.csv`、`results/analysis/archive/analysis/unified_comparison_v1/prediction_schema_manifest.json`。
- 图索引：`results/analysis/archive/analysis/unified_comparison_v1/figure_manifest.json`；图状态统计：{'generated': 3, 'reused': 14}。
- 对齐：`results/analysis/unified_prediction_contract_v1/alignment.csv`、`results/analysis/unified_prediction_contract_v1/failed_alignment.csv`。
- 现有证据入口：`docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`、`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V2.md`、`docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`、`docs/对比实验/MOGB_DCLOOS_对比结果报告.md`。
- 计划差距审计：`docs/analysis/PLAN_ALIGNMENT_AUDIT_V1.md`；补充执行计划：`docs/analysis/UNIFIED_COMPARISON_SUPPLEMENT_PLAN_V1.md`。
