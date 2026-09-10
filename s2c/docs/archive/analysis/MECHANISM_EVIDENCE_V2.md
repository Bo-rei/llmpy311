# 机制证据 V2：到底是在和谁比较

更新时间：2026-08-09  
活动协议：`protocol_v2_textoir_v1`

本报告只使用已经完成的五 seed 同协议结果和已有表示诊断，不训练新模型、不调阈值、不把外部兼容性单格混入公平排名。

## 1. 对比对象先分层

当前真正可直接比较的是七行 fair matrix：

- `S2C-Trainable-K1`：当前最强自有候选；Known-only 训练 MiniLM，单中心 Gate。
- `S2C-Frozen-K1`：冻结 MiniLM，单中心。
- `S2C-Frozen-K2`：冻结 MiniLM，固定 KMeans 双中心。
- `S2C-Random-K2`：冻结 MiniLM，随机平衡双中心。
- `MOGB-Fair`：冻结 MiniLM，MOGB 粒球和 MOGB 边界。
- `MOGB-Partition/S2C-Boundary`：只换成 MOGB 分区，边界仍用 S2C。
- `S2C-Partition/MOGB-Boundary`：只换成 MOGB 边界，分区仍用 S2C。

七行合计 63 个 dataset×KIR summary rows，每格 5 个 seed。详细定义见
[`METHOD_COMPARISON_MAP_V1.md`](METHOD_COMPARISON_MAP_V1.md)。

RC-AMBL 和 Joint-Adaptive 是自适应多中心 pilot，当前都安全回退到 `K_y=1`，不应伪装成已经成功的第八种方法。

## 2. 新增的机制图回答什么问题

### 图 A：OOS F1—false acceptance 前沿

文件：`figures/archive/analysis/mechanism_evidence_v2/oos_f1_vs_false_acceptance.png`

它把所有数据集和 KIR 的工作点放在同一图中。右上角不是自动更好：OOS F1 需要同时检查 false acceptance、Known Recall 和 F1-All。Trainable K=1 的优势主要是位于较高 OOS F1、较低 FA 的平衡区域；MOGB-Fair 更靠近低 FA 但高 false rejection 的保守区域。

### 图 B：Known 覆盖恢复—新增 OOS 误接收预算

文件：`figures/archive/analysis/mechanism_evidence_v2/known_recovery_vs_oos_acceptance.png`

横轴是相对 Frozen K=1 的 Known Recall 变化，纵轴是相对 Frozen K=1 的 false acceptance 变化。StackOverflow 中固定 K=2 的点明显位于“Known 几乎没有恢复、OOS 误接收大幅增加”的危险方向；MOGB 组件则位于“误接收下降但 Known 覆盖大量损失”的保守方向。

### 图 C：表示几何—K=2 风险

文件：`figures/archive/analysis/mechanism_evidence_v2/representation_geometry_vs_k2_risk.png`

该图只作描述性诊断。CE/SupCon 可以提高相对分离，但 StackOverflow 上 K=2 仍然明显退化，说明表示几何改善不等于多球接受区域安全。不能把这些点解释为因果证明，也不能用其选择 K。

### 图 D：Trainable 表示的 K=1 工作间隔

文件：`figures/archive/analysis/mechanism_evidence_v2/score_gap_and_false_acceptance.png`

Trainable MiniLM 在三个数据集都扩大了 OOS 与 Known 的中位 score gap，同时降低固定 threshold 下的 false acceptance。这是目前“为什么 K=1 有效”的最直接表示—决策证据；它不证明 K>1 有效。

## 3. 当前可以说什么

当前最稳妥的表述是：

> 在相同 Known-only 协议下，`S2C-Trainable-K1` 比冻结单中心、固定双中心、随机双中心和 MOGB 的 MiniLM 组件更接近平衡的 OOS/覆盖工作点；其主要收益来自表示适配后更大的 score separation 和更低的 OOS false acceptance，而不是来自增加中心。

当前不能说：

- 已经超过完整 MOGB；
- 已经超过 ADB、DA-ADB 或 DCLOOS；
- 自适应多中心已经成功；
- 低 false acceptance 的 MOGB 行就是更好的方法。

## 4. 数据与合同边界

- 外部 ADB/DA-ADB 的同协议 runtime 仍未通过 torch/CUDA smoke；旧兼容单格不进入本报告排名。
- DCLOOS 使用 pseudo-OOS 和外部 OOS，监督条件不同，必须单列。
- MOGB 官方 BERT 单格未达到论文数字；`MOGB-Fair` 只表示冻结 MiniLM 组件比较。
- 所有测试指标均来自已冻结结果；本分析没有使用测试结果选择参数或模型。

## 5. 机器可读输出

- `results/analysis/archive/analysis/mechanism_evidence_v2/fair_matrix_audited.csv`
- `results/analysis/archive/analysis/mechanism_evidence_v2/acceptance_budget.csv`
- `results/analysis/archive/analysis/mechanism_evidence_v2/trainable_k1_k2_budget.csv`
- `results/analysis/archive/analysis/mechanism_evidence_v2/representation_geometry_k2_risk.csv`
- `results/analysis/archive/analysis/mechanism_evidence_v2/score_gap_false_acceptance.csv`
- `results/analysis/archive/analysis/mechanism_evidence_v2/MANIFEST.json`
