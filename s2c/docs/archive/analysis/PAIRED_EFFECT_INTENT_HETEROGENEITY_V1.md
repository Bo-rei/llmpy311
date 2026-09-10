# 配对效应与 intent-level 多中心异质性 V1

更新时间：2026-08-08
活动协议：`protocol_v2_textoir_v1`
证据类型：analysis-only；不训练、不调参、不覆盖历史 artifact。

## 做了什么

1. 对同一 `dataset × KIR × seed` 配对 Trainable K=1 与六个 Frozen/MOGB 组件，计算 OOS F1、F1-All、Known Recall 和 false acceptance 的差值。
2. 使用固定 RNG seed、10,000 次 bootstrap 生成配对均值的 95% CI，并输出 win/tie/loss。
3. 从已有 intent-level test-oracle 诊断中统计不同数据集/KIR/距离下的多中心安全收益比例和 oracle-best-K 分布。

## Trainable K=1 对 Frozen K=1 的配对结果

- Banking77 KIR=0.25：相对 Frozen K=1 的 OOS F1 差值 +6.48pp，95% CI [+5.47, +7.50]。
- Banking77 KIR=0.50：相对 Frozen K=1 的 OOS F1 差值 +11.14pp，95% CI [+9.68, +13.41]。
- Banking77 KIR=0.75：相对 Frozen K=1 的 OOS F1 差值 +14.99pp，95% CI [+13.22, +17.23]。
- CLINC150 KIR=0.25：相对 Frozen K=1 的 OOS F1 差值 +0.96pp，95% CI [+0.87, +1.08]。
- CLINC150 KIR=0.50：相对 Frozen K=1 的 OOS F1 差值 +1.50pp，95% CI [+1.36, +1.64]。
- CLINC150 KIR=0.75：相对 Frozen K=1 的 OOS F1 差值 +1.10pp，95% CI [+0.74, +1.47]。
- StackOverflow KIR=0.25：相对 Frozen K=1 的 OOS F1 差值 +6.00pp，95% CI [+2.55, +9.08]。
- StackOverflow KIR=0.50：相对 Frozen K=1 的 OOS F1 差值 +11.12pp，95% CI [+7.93, +14.32]。
- StackOverflow KIR=0.75：相对 Frozen K=1 的 OOS F1 差值 +10.70pp，95% CI [+5.55, +15.08]。

这一步只说明表示适配在同一 split/seed 下的后验差异，不把结果解释成外部 SOTA。

## intent-level 异质性（Euclidean）

- Banking77 KIR=0.25：safe_gain_oracle 比例 43.2%，平均 best K=3.25，平均 ΔOOS F1=+10.39pp，平均 ΔKnown Recall=-0.68pp。
- Banking77 KIR=0.50：safe_gain_oracle 比例 47.9%，平均 best K=2.81，平均 ΔOOS F1=+8.13pp，平均 ΔKnown Recall=+1.14pp。
- Banking77 KIR=0.75：safe_gain_oracle 比例 52.1%，平均 best K=2.70，平均 ΔOOS F1=+6.74pp，平均 ΔKnown Recall=+2.74pp。
- CLINC150 KIR=0.25：safe_gain_oracle 比例 37.9%，平均 best K=2.15，平均 ΔOOS F1=+3.91pp，平均 ΔKnown Recall=+1.42pp。
- CLINC150 KIR=0.50：safe_gain_oracle 比例 38.1%，平均 best K=1.96，平均 ΔOOS F1=+3.98pp，平均 ΔKnown Recall=+2.59pp。
- CLINC150 KIR=0.75：safe_gain_oracle 比例 40.7%，平均 best K=1.88，平均 ΔOOS F1=+3.70pp，平均 ΔKnown Recall=+3.15pp。
- StackOverflow KIR=0.25：safe_gain_oracle 比例 28.0%，平均 best K=1.80，平均 ΔOOS F1=+0.98pp，平均 ΔKnown Recall=-0.85pp。
- StackOverflow KIR=0.50：safe_gain_oracle 比例 28.0%，平均 best K=1.58，平均 ΔOOS F1=+0.81pp，平均 ΔKnown Recall=-0.69pp。
- StackOverflow KIR=0.75：safe_gain_oracle 比例 17.3%，平均 best K=1.40，平均 ΔOOS F1=+0.53pp，平均 ΔKnown Recall=-0.45pp。

`safe_gain_oracle` 和 `oracle-best-K` 依赖测试标签，只能用于解释研究空间，不能用于正式选择中心数、半径或阈值。它们的价值是说明：多中心潜在收益集中在部分 intent，而不是整个数据集统一受益。

## 图表与数据

- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/paired_oos_f1_forest.png`：配对 OOS F1 及 95% bootstrap CI。
- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/kir50_tradeoff_heatmap.png`：KIR=0.50 下四项核心指标的配对差值。
- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/safe_gain_heatmap.png`：逐意图安全收益比例。
- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/oracle_best_k_distribution.png`：Euclidean 下 oracle-best-K 分布。
- `results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/paired_effects.csv`、`paired_summary.csv`、`intent_summary.csv`、`best_k_distribution.csv`。

## 解释边界

- 该阶段没有新增训练或测试选择；所有输入来自已完成协议结果。
- 外部 ADB、DA-ADB、DCLOOS 不在本阶段被强行并入同协议统计。
- 配对 CI 反映五个正式 seed 的不确定性，不替代同监督外部基线复现。

输入与输出 SHA256 见 `results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/MANIFEST.json`。生成摘要：{"analysis_only": true, "best_k_distribution_rows": 88, "bootstrap_replicates": 10000, "bootstrap_seed_base": 20260808, "inputs": {"results/analysis/cross_protocol_tradeoff_v1/per_seed.csv": "31ccdd433b38b86ec92b0cd81b460114a7fe1e7f12b94d4a5d031aab7d703145", "results/analysis/archive/analysis/intent_kir_stability_pack_v1/intent_kir_rows.csv": "73800ccc0105d4a8888e14c5787612bff2185e7b00f649b48f43b970b141c3bc"}, "intent_rows": 3700, "intent_summary_rows": 18, "outputs": {"results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/best_k_distribution.csv": "ced7c948049e6c2569dad0b6821d1cfe6c2aaa68d509ad6c2bb77fcc3b6a614e", "results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/intent_summary.csv": "bc3d6a497b1ca5bbe04e09e3ce5db84ef9608aa0a60cf91f7583c60e60a7b8d2", "results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/paired_effects.csv": "5751196ff02b6e8fb820c9085c51561c86eafa726e82c928545ea972566544cf", "results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/paired_summary.csv": "d3e866ff3d4dac1dd68e43a0e38839c742fcad740c72addafc40d6c661e1d2b0"}, "paired_rows": 270, "paired_summary_rows": 216, "protocol_version": "protocol_v2_textoir_v1", "schema_version": "paired_effect_intent_heterogeneity_v1"}