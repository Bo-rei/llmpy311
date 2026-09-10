# 跨 KIR 合同分层图与结果摘要

本报告把当前 MiniLM fair 结果、MOGB-MiniLM 组件结果和 ADB BERT/TextOIR 外部合同放在同一坐标系中，
但不把它们当作同骨干 SOTA 排名。误差线来自 5 seed（S2C/MOGB）和 ADB 的 `5` seed。

## 关键观察

- Trainable-K1 的 OOS F1 在 9 个 dataset×KIR 组中有 8 个高于 ADB，但 CLINC150 的 F1-All 与 Known Recall 全部低于 ADB。
- MOGB-MiniLM 的低 false acceptance 伴随明显 Known Recall 损失；它不是简单的“更强拒识”。
- KIR 增加后，ADB 的 OOS F1 在三个数据集都下降，幅度在 Banking77 和 StackOverflow 更明显。
- 因此应同时报告 OOS F1、F1-All、Known Recall 和 false acceptance；只画 OOS F1 会掩盖工作点差异。

## 机器可读来源

- fair source SHA256：`c3f5b5cb9701bfd2e111da85575226393e3baf9c8bcace7df6fc39eb2d396ae8`
- ADB source SHA256：`b948e56cadf0edd7480cf6a5d9d4ad6b12c8dce98fb973ab2c62ab95eb5023ad`
- 图：`/home/bo/bo01/llmpy311/s2c/figures/cross_kir_contract_atlas_v2/cross_kir_contract_atlas.png`
- 行级数据：`/home/bo/bo01/llmpy311/s2c/results/analysis/cross_kir_contract_atlas_v2/rows.csv`
