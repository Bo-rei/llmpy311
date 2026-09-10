# StackOverflow 同协议外部基线单格对比

> 当前表只用于审计同一 StackOverflow/KIR=0.50 数据合同下的兼容运行，不把不同监督条件混成一个 SOTA 排名。

## 结果边界

- S2C 与 MOGB-MiniLM 行来自 protocol_v2_textoir_v1 的统一 fair Gate 矩阵。
- ADB 行来自 TextOIR BERT 兼容 runner，但输入 split 根由 protocol_v2 导出，属于同数据/同 KIR 单格参考，不等同于 MiniLM fair Gate。
- DA-ADB 适配尝试即使进程返回 0，也必须以逐样本预测审计为准；全类预测或 NaN 训练不能作为有效结果。

## 当前可用观察

- ADB 的同协议 seed 结果可以作为外部单格参照；它与 S2C 的差异同时包含 BERT 表示、端到端训练和边界训练，不能归因于单一边界组件。
- MOGB-MiniLM 的低 false acceptance 伴随极低 Known Recall，说明其保守拒识工作点不能只看 OOS F1。
- DA-ADB 若出现 NaN 或单类预测，必须保留为 invalid，不得引用官方 results.csv 中与预测不一致的数字。

## 机器可读输出

- `stackoverflow_kir50_external_and_fair_cells.csv`：公平行和外部兼容行，含 layer/contract/status。
- `external_invalid_semantic_runs.csv`：无效运行及原因。
- `adb_vs_trainable_paired_effects.csv`：三个 seed 的配对差值。
- `stackoverflow_external_single_cell_comparison.png`：有效单格工作点图。
- `adb_vs_trainable_paired_delta.png`：同 seed 差值及标准差。
