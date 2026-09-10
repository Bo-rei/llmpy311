# ADB 隔离 CUDA 运行收口（StackOverflow / KIR=0.50）

## 结论先行

本次完成了 ADB 在当前 StackOverflow/KIR=0.50 固定 split 上的三个 seed 兼容运行：
`42、87、100` 均有完整逐样本预测，返回码为 0，artifact audit 通过。
这是一组 **BERT/TextOIR 外部合同参照**，不是与当前 MiniLM Gate 的同表示公平实验，不能据此宣布跨合同 SOTA。

在这三个配对 seed 上，当前 `S2C Trainable K=1` 的 OOS F1 平均高于 ADB
`+0.85pp`，F1-All 高 `+1.03pp`，Known Recall 高 `+2.90pp`；但 false acceptance
高 `+0.66pp`。因此当前证据支持“Trainable K=1 在 Known 覆盖与 OOS 工作点之间取得更好的同数据工作点”，不支持“MiniLM 方法全面击败 ADB”，因为两者 backbone、训练实现和边界合同不同。

## 运行合同

| 项目 | 固定值 |
|---|---|
| 数据集 | StackOverflow |
| KIR | 0.50 |
| split / Known 列表 | `protocol_v2_textoir_v1` 导出的固定数据根和 Known labels |
| 外部方法 | TextOIR ADB，BERT/TextOIR 训练路径 |
| seed | 42、87、100 |
| 运行时 | PyTorch 2.9.1+cu128，Transformers 4.46.3，CUDA RTX 5070 |
| 评价 | 从 `y_true.npy/y_pred.npy` 重算；不信任与预测不一致的 `results.csv` |
| OOS 训练 | 否；ADB 仅使用该 runner 的已知训练合同 |

Run manifest：

- `../artifacts/s2c/external/adb_gpu_runtime_v1/stackoverflow/ADB/kir50/seed42/run_manifest.json`
  （SHA256 `d4c358a1cb3e568561484a109c3ed4043ce6b45a88e7d64ac5c274e2966cb778`）
- `../artifacts/s2c/external/adb_gpu_runtime_v1/stackoverflow/ADB/kir50/seed87/run_manifest.json`
  （SHA256 `3cd0f9ee73b912af114749785c3bac25a54e95a8065407cbe16063f97047c568`）
- `../artifacts/s2c/external/adb_gpu_runtime_v1/stackoverflow/ADB/kir50/seed100/run_manifest.json`
  （SHA256 `78a018efd0eec3c8cda37d6aaf8c8dde9dd0ee15f05f758197d9fab29ace3fde`）

## 三个 seed 的指标（百分数）

| seed | OOS F1 | F1-All | Known Recall | False Acceptance | False Rejection | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 85.95 | 84.25 | 79.30 | 9.03 | 20.70 | 84.85 |
| 87 | 87.02 | 85.35 | 80.87 | 8.23 | 19.13 | 86.02 |
| 100 | 89.12 | 87.38 | 82.17 | 5.30 | 17.83 | 88.23 |
| **均值 ± 标准差** | **87.36 ± 1.61** | **85.66 ± 1.59** | **80.78 ± 1.44** | **7.52 ± 1.97** | **19.22 ± 1.44** | **86.37 ± 1.72** |

机器可读汇总：
`results/analysis/archive/analysis/external_gpu_runtime_comparison_v1/stackoverflow_kir50_external_summary.csv`。

## 与当前 S2C Trainable K=1 的配对差异

这里仅取相同 seed `42、87、100` 的当前 protocol_v2 结果，避免把五 seed 均值与三 seed 外部运行混算。

| 指标 | S2C Trainable K=1 均值 | ADB 均值 | S2C − ADB |
|---|---:|---:|---:|
| OOS F1 | 88.21 | 87.36 | **+0.85pp** |
| F1-All | 86.69 | 85.66 | **+1.03pp** |
| Known Recall | 83.68 | 80.78 | **+2.90pp** |
| False Acceptance | 8.18 | 7.52 | **+0.66pp** |
| False Rejection | 16.32 | 19.22 | **−2.90pp** |
| Accuracy | 87.23 | 86.37 | **+0.87pp** |

逐 seed 配对差异和标准差见：
`results/analysis/archive/analysis/external_gpu_runtime_comparison_v1/adb_vs_trainable_paired_effects_summary.csv`。

## 如何解释

1. ADB 的工作点比 MOGB-MiniLM 更平衡，但它仍是 BERT/TextOIR 外部实现；不能把它放进当前 Frozen/Trainable MiniLM 的 fair Gate 排名。
2. S2C Trainable K=1 在这三个 seed 上提高 Known Recall 并减少 Known false rejection，同时 OOS F1 仍略高；代价是 false acceptance 平均高约 0.66pp。
3. 当前证据回答的是“同数据 split 下的合同参照”，不是“同表示、同训练预算、同指标选择流程下的严格基线复现”。
4. DA-ADB 仍保持 invalid（NaN/单类预测），DCLOOS 仍属于 pseudo/external-OOS 监督合同；二者没有被本报告伪装成可比结果。

## 产物与复现

- 比较审计：`results/analysis/archive/analysis/external_gpu_runtime_comparison_v1/`
- 运行入口：`tools/compat/textoir/run_external_textoir.py`
- 外部比较入口：`tools/analysis/build_external_single_cell_comparison_v1.py --external-root ../artifacts/s2c/external/adb_gpu_runtime_v1/stackoverflow`
- 决策面板：`docs/analysis/EXPERIMENT_DECISION_DASHBOARD_V1.md`

本阶段没有修改 E0–E3、MiniLM fair 矩阵或历史 artifacts；没有使用测试 OOS 选参；没有执行 `git add/commit/push`。
