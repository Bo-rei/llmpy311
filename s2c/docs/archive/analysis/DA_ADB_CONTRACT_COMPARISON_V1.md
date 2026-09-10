# DA-ADB 新旧合同差异分析 V1

## 目的

旧兼容单格报告了 OOS F1=`90.90%`，新的 protocol_v2 隔离 CUDA 单格为
`70.82%`。本报告不挑选更高的数字，而是在同一新 runtime 下复跑旧 seed=0
数据/Known-list 合同，拆分“运行时/兼容层差异”和“seed/Known-list/split 差异”。

## 三个单元

| 单元 | 数据/Known 合同 | runtime | OOS F1 | F1-All | Known Recall | FA |
|---|---|---|---:|---:|---:|---:|
| 历史 seed0 | 旧 `textoir/data`，旧 Known list | textoir-py39，旧兼容 | 90.90% | 89.23% | 83.13% | 2.63% |
| 隔离 seed0 | 与历史相同的 split/Known list | 新 CUDA runtime | 84.05% | 80.98% | 74.50% | 9.03% |
| 隔离 seed42 | protocol_v2 split/Known list | 新 CUDA runtime | 70.82% | 72.03% | 72.93% | 30.33% |

## 差值解释

历史 → 隔离 seed0（只近似控制数据/Known list，改变 runtime/兼容层）的差值为：

`OOS F1 -6.85pp`，`F1-All -8.25pp`，
`Known Recall -8.63pp`，`FA +6.40pp`。

隔离 seed0 → 隔离 seed42（保持新 runtime，改变 seed、Known list 和 protocol split）的差值为：

`OOS F1 -13.22pp`，`F1-All -8.95pp`，
`Known Recall -1.57pp`，`FA +21.30pp`。

这不是严格的因果分解：seed、Known list 和数据快照不能只靠三个单元完全分离。但它已经证明，旧
`90.90%` 不能直接当作当前 protocol_v2 DA-ADB 性能；新 runtime 下同旧合同的结果必须先作为独立
复现结果记录。所有三单元的逐样本预测均有限、形状一致且非单类塌缩。

## 证据和图

- 图：`figures/archive/analysis/da_adb_contract_comparison_v1/da_adb_contract_metrics.png`
- 表：`results/analysis/archive/analysis/da_adb_contract_comparison_v1/da_adb_contract_comparison.csv`
- Manifest：`results/analysis/archive/analysis/da_adb_contract_comparison_v1/DA_ADB_CONTRACT_COMPARISON_MANIFEST.json`
- 新旧运行根：`/home/bo/bo01/llmpy311/artifacts/s2c/external/da_adb_compat_single_cell_v3/stackoverflow/DA-ADB/kir50/seed0`、`/home/bo/bo01/llmpy311/artifacts/s2c/external/da_adb_gpu_runtime_v1_legacy/stackoverflow/DA-ADB/kir_0.50/seed_0`、`/home/bo/bo01/llmpy311/artifacts/s2c/external/da_adb_gpu_runtime_v1/stackoverflow/DA-ADB/kir_0.50/seed_42`

结论仍然是合同分层：DA-ADB 可作为外部 BERT/TextOIR 参照，但当前没有证据支持它与
`S2C-Trainable-K1` 的同骨干公平排名，也没有证据支持跨合同 SOTA 宣称。
