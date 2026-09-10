# 外部基线合同与可比性可视化 V1

> 状态提示（2026-08-10）：本文件保留 V1 历史图表；DA-ADB 当前协议三 seed 的最新结果已单独收口到
> [`DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md`](DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md)，不要把本页早期的
> `90.90%` 单格视为当前协议代表值。

## 结论

本报告把同协议 Frozen/Trainable 组件、ADB/DA-ADB 兼容性单格、MOGB 官方严格单格和 DCLOOS reduced-budget 结果放在一个带合同标签的证据层中。它不把不同监督、表示、数据或 seed 合同混成 SOTA 排名。

在 StackOverflow/KIR=0.50 的已有数字中，最新三 seed ADB 外部均值高于当前 Trainable K=1；但它是端到端 BERT/TextOIR 合同，不是当前 protocol_v2 的五 seed MiniLM fair matrix，因此这里只能记录外部参照工作点，不能称为公平优胜结论。

## StackOverflow/KIR=0.50 可见结果

| 方法 | OOS F1 | F1-All | Known macro-F1 | seed 数 | 合同 | 是否进入当前同协议主排名 |
|---|---:|---:|---:|---:|---|---|
| DA-ADB | 90.90 | 89.23 | 89.06 | 1 | compatibility single-cell | 否 |
| Trainable K=1 | 87.67 | 86.55 | nan | 5 | same protocol | 是 |
| ADB | 87.36 | 85.66 | nan | 3 | compatibility single-cell | 否 |
| BRAK | 81.34 | 80.10 | 79.97 | 1 | official/other | 否 |
| MOGB partition + s2c boundary | 79.25 | 63.34 | nan | 5 | same protocol | 是 |
| Single centroid | 76.55 | 79.98 | nan | 5 | same protocol | 是 |
| MOGB-MiniLM | 72.92 | 43.30 | nan | 5 | same protocol | 是 |
| Fixed K=2 | 63.53 | 72.76 | nan | 5 | same protocol | 是 |

## 监督与表示差异

- 当前 Trainable K=1：Known-only MiniLM 适配，当前协议，五 seed fair summary。
- MOGB-MiniLM 与 MOGB partition 组件：冻结 MiniLM，Known-only，五 seed fair component。
- ADB：最新三 seed 的端到端 BERT/TextOIR 兼容性参照；DA-ADB 仍为旧兼容性单格/invalid 记录，不能与当前 MiniLM 五 seed 结果直接做显著性结论。
- MOGB official strict：官方 BERT 逻辑单格，当前 OOS F1 字段与 Gate 表不完全同构，不能强行补值。
- DCLOOS reduced-budget：使用 pseudo-OOS/外部 OOS，且当前记录为不同数据/不同 KIR 合同，不进入 StackOverflow fair scatter。

## 图表

- `figures/archive/analysis/baseline_contract_visuals_v1/stackoverflow_baseline_pareto_contract.png`：性能工作点与合同形状。
- `figures/archive/analysis/baseline_contract_visuals_v1/baseline_contract_table.png`：方法、监督、表示和可比性矩阵。

## 研究含义

当前证据说明：Trainable K=1 已经是当前自有 Gate 的稳定工作点，但还没有同合同证据证明它超过端到端 ADB/DA-ADB。下一步应优先补齐同协议的端到端基线或明确将其保留为兼容性参照，而不是继续用外部数字宣称 SOTA。

所有结果来自已完成的 summary 文件；没有训练、调参或覆盖历史 artifact。
