# Historical H1 Trainable MiniLM Gate：参数化 full pipeline 实验汇报

本页保留最初的固定参数网格及其选择约束，不代表最新实验已止步于此。后续 OOS-only 选择、自适应中心和 Gate/pipeline 逐样本分类损失见[自适应中心汇报](historical_trainable_adaptive_centers_presentation.md)；当前进展以 [CURRENT_STATUS](../CURRENT_STATUS.md) 为准。

> 目标：在历史论文主线的 H1 controlled 协议下，用已有训练后的 Trainable MiniLM 作为 Gate，搜索 K、λ、threshold 和接受规则，并将真实 Gate→Router→Expert 结果与 `fulltex.tex` KIR=.50 的 `Ours` OOS F1 对比。

## 结论

- `banking77_oos` 的验证集选中配置在真实 full pipeline 上达到 **91.53±0.05 OOS F1**，比论文 `Ours=88.23` 高 **+3.30 pp**；三个 seed 均超过论文值。
- `CLINC150` 的选中配置为 **91.04±0.81**，比论文 `91.96` 低 **0.92 pp**；`StackOverflow` 为 **89.15±1.54**，比论文 `89.71` 低 **0.56 pp**。当前网格没有让这两个数据集的三-seed 均值超过论文值。
- 当前结果说明：训练后的 MiniLM Gate 可以在同一 full pipeline 中产生明显收益，但是否超过论文 Ours 仍受数据协议/H1 下游组件和边界 work-point 共同影响，不能把 Gate-only 提升直接等同为论文系统复现。

## 1. 协议与实验设计

| 项目 | 本实验 |
|---|---|
| 数据 | `clinc150`、`stackoverflow`、`banking77_oos` |
| KIR / seed | KIR=.50；seed=13,42,87 |
| Gate | 已有 Trainable `last2 MiniLM + 384→256→384 residual projection` checkpoint |
| 边界 | diagonal Mahalanobis；`mean + λ·std` |
| 参数 | K∈{1,2,3}；λ∈{.50,.75,1,1.25,1.5,2}；threshold∈{.85,.90,.95,1,1.05,1.10,1.20}；两种 acceptance mode |
| 选参 | 仅 validation；目标 OOS F1；Known F1 与 Accuracy 相对 K=1 baseline 不下降超过 1 pp（逐 seed及均值） |
| 下游 | 现有 H1 Router/Expert checkpoint 固定不变；selected 配置另做直接 pipeline 核验 |
| GPU | CUDA；实验 manifest 记录为 `cuda` |

论文对照值来自 `fulltex.tex` KIR=.50 的 `Ours` 行；论文表头写作 Banking77，本实验历史主线的实际数据键为 `banking77_oos`。

## 2. 验证集选中的配置与测试结果

| 数据集 | selected Gate | Val OOS F1 | Test full-pipeline OOS F1 | Known F1 | Acc | False Acceptance | 相对论文 OOS F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | K=2, λ=2.0, t=0.95, normalized_union | 88.48 | 91.04* | 83.94 | 86.78 | 6.56% | -0.92 pp |
| StackOverflow | K=1, λ=1.0, t=0.95, nearest_sphere | 89.67 | 89.15* | 84.48 | 86.14 | 4.32% | -0.56 pp |
| Banking77-OOS | K=1, λ=0.75, t=0.95, normalized_union | 88.05 | 91.53* | 75.84 | 85.59 | 7.85% | +3.30 pp |

`*` 上表的 selected test 数字来自直接 Gate→Router→Expert 推理，并与固定下游 replay 的全部指标逐项核对，最大绝对差为 0。

## 3. 参数搜索保留结果

测试集只用于确认，不用于选参。作为后验审计，本实验另外保留所有测试均值超过论文 OOS F1 的候选：

| 数据集 | 超过论文的候选数 | 测试后验最高配置 | OOS F1 | Δ paper |
|---|---:|---|---:|---:|
| CLINC150 | 0 | K=2, λ=2.0, t=0.95, normalized_union | 91.04 | -0.92 pp |
| StackOverflow | 0 | K=1, λ=1.5, t=0.9, normalized_union | 89.25 | -0.46 pp |
| Banking77-OOS | 110 | K=1, λ=0.5, t=0.95, nearest_sphere | 91.73 | +3.50 pp |

测试后验最高配置只作为保留的 confirmation，不是论文式选参结果；完整候选表见 [`full_pipeline_candidate_summary.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_candidate_summary.csv)，论文超越候选见 [`paper_beating_candidates.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/paper_beating_candidates.csv)。

## 4. 图形解释

![Validation parameter frontier](../../figures/historical_trainable_parameter_full_pipeline/validation_parameter_frontier.png)

图 1：每个点是一组 Gate 参数。星标只由 validation 选出；灰色/蓝色区分 Known F1 与 Accuracy guard，虚线是论文 Ours 的 OOS F1。

![Test OOS F1 versus paper](../../figures/historical_trainable_parameter_full_pipeline/test_oos_f1_vs_paper_ours.png)

图 2：所有测试候选的 OOS F1 与 Known F1 分布。星标是 validation-selected，红色叉号只是 test 后验最高点，用于回答“当前网格能否超过论文”，不参与选参。

![Selected error budget](../../figures/historical_trainable_parameter_full_pipeline/selected_full_pipeline_error_budget.png)

图 3：selected full pipeline 的错误预算。OOS F1 的变化首先来自 Gate 的 OOS 接受/拒绝边界；只有被 Gate 接受的样本才会暴露 Router/Expert error。

## 5. 可信度与限制

- 所有 9 个 selected dataset×seed 均执行了真实 pipeline；`direct_vs_derived_verification.csv` 的最大绝对差为 0。
- 训练后的 MiniLM checkpoint 已有，本文实验不重新训练 Gate；本轮只改变边界 work-point。
- 这是 H1 controlled evidence：当前数据、训练后的 Gate 和现有 Router/Expert 与论文原始 H0 完整 Cascade 并非字节级相同，因此“高于论文”应理解为当前 H1 对论文公开 reference 的比较，而不是严格复现声明。
- 没有修改 `fulltex.tex`，没有覆盖历史 artifact，也没有保存逐样本 raw prediction。

## 6. 文件

- 结果 manifest：[`MANIFEST.json`](../../results/analysis/historical_trainable_parameter_full_pipeline/MANIFEST.json)
- 验证集选择：[`selected_validation.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/selected_validation.csv)
- 全部 Gate 候选：[`gate_workpoints.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/gate_workpoints.csv)
- full pipeline 候选：[`full_pipeline_candidates_test.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_candidates_test.csv)
- 测试后验最高配置：[`best_test_candidate_by_dataset.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/best_test_candidate_by_dataset.csv)
- selected 直接结果：[`full_pipeline_selected_per_seed.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_selected_per_seed.csv)
- 直接/派生核验：[`direct_vs_derived_verification.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/direct_vs_derived_verification.csv)
