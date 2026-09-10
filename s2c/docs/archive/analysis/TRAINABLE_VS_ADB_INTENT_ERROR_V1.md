# Trainable-K1 与 ADB 意图级错误归因 V1

更新时间：2026-08-10

本报告只读取已经完成且通过 sample-id、文本、标签和 manifest 对齐审计的 Trainable-K1 与 ADB 预测。
Known 错误按真实 Known intent 聚合；OOS 误接收按预测吸收 OOS 的 Known intent 聚合。两者语义不同，不能合并解释。
本分析不训练、不调阈值、不选择 checkpoint，也不把 BERT/TextOIR ADB 与 MiniLM Trainable 混成 SOTA 排名。

## 覆盖范围

- 1,850 条 intent×dataset×KIR×seed 记录；663 条 intent×dataset×KIR 聚合记录。
- 对齐单元：45 个 dataset×KIR×seed；全部 `aligned=true`。
- 输入：Trainable/ADB 逐样本预测和 protocol test view；轻量输出不包含文本。

## 主要观察

- **clinc150**：意图级 Known 误拒差值均值 `+16.30pp`；OOS 误接收差值均值 `-0.17pp`；按意图×KIR 观察到 `400` 个聚合点。
- **banking77**：意图级 Known 误拒差值均值 `+7.29pp`；OOS 误接收差值均值 `-0.51pp`；按意图×KIR 观察到 `210` 个聚合点。
- **stackoverflow**：意图级 Known 误拒差值均值 `-2.42pp`；OOS 误接收差值均值 `-0.19pp`；按意图×KIR 观察到 `53` 个聚合点。

按意图×KIR 的方向计数如下：CLINC150 有 `7/400` 个点恢复 Known 覆盖、`338/400` 个点减少 OOS 误接收；Banking77 为 `23/210` 和 `165/210`；StackOverflow 为 `42/53` 和 `33/53`。因此 StackOverflow 的覆盖恢复比另外两个数据集更普遍，但仍存在少数意图的 OOS 误接收上升。

- 如果点落在散点图右上方，Trainable 同时增加 Known 误拒和 OOS 误接收，属于更保守但有额外开放空间风险的意图。
- 如果点落在左下方，Trainable 同时恢复 Known 覆盖并减少 OOS 误接收，是最有利的工作点变化。
- 该图用于解释平均差异由哪些意图贡献，不用于从测试结果选择意图、中心数或阈值。

## 产物

- `results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_per_seed.csv`
- `results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_mean.csv`
- `results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1/ALIGNMENT.json`
- `figures/archive/analysis/trainable_vs_adb_intent_error_v1/known_rejection_delta_by_intent.png`
- `figures/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_budget_scatter.png`
- `figures/archive/analysis/trainable_vs_adb_intent_error_v1/top_intent_oos_acceptor_changes.png`

Manifest SHA256：`beffb22bc4ecbcbbb50f4c5f0720d7baa29f081cc7a9a2353141592544524f9b`。
