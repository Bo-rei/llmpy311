# 实验决策面板 V1

本面板只读取已冻结结果，集中回答当前比较对象、同协议结果和外部参照的合同边界。它不新增训练、不调参、不读取测试样本。

## 同协议公平矩阵

| 方法 | 九格 OOS F1 均值 | 状态 |
|---|---:|---|
| Trainable K=1 | 85.75% | `valid_same_protocol` |
| Frozen K=1 | 78.64% | `valid_same_protocol` |
| Random K=2 | 78.39% | `valid_same_protocol` |
| Frozen K=2 | 75.51% | `valid_same_protocol` |
| MOGB partition + s2c boundary | 77.74% | `valid_same_protocol` |
| s2c partition + MOGB boundary | 75.90% | `valid_same_protocol` |
| MOGB-MiniLM | 73.39% | `valid_same_protocol` |

## 解释

- 当前最强自有对象是 `S2C-Trainable-K1`，不是历史 `fulltex.tex` Cascade。
- MOGB-Fair 是冻结 MiniLM 组件对照；MOGB 官方 BERT 单格仍是 `not_reproduced_strict`。
- ADB（三个 seed 的隔离 CUDA/BERT 兼容运行）和 DCLOOS reduced 只作外部合同参照，不能与 fair Gate 行直接排名。
- ADB 三个 StackOverflow/KIR=.50 单格已经完成；需要继续完成的是其合同说明和其他外部基线，而不是重复 E2/E3 或继续扩大固定 K。

## 产物

- `results/analysis/archive/analysis/experiment_decision_dashboard_v1/decision_rows.csv`
- `figures/archive/analysis/experiment_decision_dashboard_v1/decision_dashboard.svg`
- `results/analysis/archive/analysis/experiment_decision_dashboard_v1/MANIFEST.json`
