# 当前实验综合分析 V1（权威分析入口）

更新时间：2026-08-10  
活动协议：`protocol_v2_textoir_v1`  
分析范围：三数据集 × KIR={0.25, 0.50, 0.75} × 7 个同协议 Gate/组件方法 × 5 seeds。

> 如果不清楚“我的方法”具体指哪一行，请先阅读
> [`METHOD_COMPARISON_MAP_V1.md`](METHOD_COMPARISON_MAP_V1.md)。当前主候选是
> `S2C-Trainable-K1`；固定多中心和自适应多中心 pilot 是不同实验对象。

本文件是当前结果的分析入口，不是论文稿件，也不把尚未完成统一运行的外部基线写成排名。所有数字来自已冻结的五 seed 汇总：
`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。本轮只做结果重排和可视化，没有重训、调参或覆盖历史 artifact。

当前阶段的短版进度快照见 [`EXPERIMENT_PROGRESS_SNAPSHOT_V2.md`](EXPERIMENT_PROGRESS_SNAPSHOT_V2.md)；它只保留当前主方法、已完成实验规模、外部合同边界和现有可视化入口，便于每次实验开始前快速定位。

外部 ADB 的跨数据集/KIR/seed 分解见 [`ADB_TRAINABLE_KIR_ANALYSIS_V1.md`](ADB_TRAINABLE_KIR_ANALYSIS_V1.md)；它共享 split/KIR/seed 但保留 BERT/TextOIR 与 MiniLM 合同差异，不进入 fair 主排名。

StackOverflow/KIR=.50 的 ADB 三 seed 逐样本错误预算见 [`TRAINABLE_VS_ADB_ERROR_BUDGET_V1.md`](TRAINABLE_VS_ADB_ERROR_BUDGET_V1.md)。该审计证明两套预测与对应 protocol test view 对齐，共 18,000 条记录；它把 Trainable 的优势边界明确为“少一些 Known 误拒、略多一些 OOS 误接收”的工作点交换，不把跨 backbone 结果写成 SOTA。

跨数据集/KIR/seed 的逐样本错误预算扩展见 [`TRAINABLE_VS_ADB_CROSS_DATASET_ERROR_BUDGET_V1.md`](TRAINABLE_VS_ADB_CROSS_DATASET_ERROR_BUDGET_V1.md)。45/45 个配对单元、共 221,700 条协议测试记录通过文本、标签、sample-id 和 manifest 哈希审计。结果显示：CLINC150 与 Banking77 的 Trainable 主要以更高 Known 误拒换取更低 OOS 误接收；StackOverflow 的 Trainable 则减少 Known 误拒但在中高 KIR 略增 OOS 误接收。该分析把“当前方法为什么更好”限定为数据集相关的错误预算重分配，不支持跨骨干 SOTA 排名。

最新的合并机制表见 [`TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md`](TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md)。它把每个 `dataset×KIR` 的 OOS F1、F1-All、Known Recall 差值与错误预算放在同一行，并生成四格热图和错误预算象限图；该表用于区分“保守拒识”和“Known 覆盖恢复”，不新增训练或测试集调参。

OOS 指标的 precision/recall 分解见 [`TRAINABLE_VS_ADB_OOS_DECOMPOSITION_V1.md`](TRAINABLE_VS_ADB_OOS_DECOMPOSITION_V1.md)。它从同一批状态计数独立重算 OOS precision、OOS recall 和 OOS F1，避免只看 F1 而忽略“更多拒绝 OOS”与“更多误拒 Known”的代价。

意图级错误归因见 [`TRAINABLE_VS_ADB_INTENT_ERROR_V1.md`](TRAINABLE_VS_ADB_INTENT_ERROR_V1.md)。它把 Known 误拒按真实 intent、OOS 误接收按预测吸收 intent 分开聚合，回答平均工作点差异是否集中在少数类别；该分析只读已对齐预测，不参与选择。

最新补充的 detector 机制证据见 [`DETECTOR_MECHANISM_ANALYSIS_V1.md`](DETECTOR_MECHANISM_ANALYSIS_V1.md)：
它在 KIR=.50、3 seed 控制上进一步区分了 Trainable 表示收益和几何 Gate 收益，并复用固定
10,000 次 paired bootstrap 生成置信区间；该控制不替代下文的五 seed fair matrix。

## 1. 方法和比较边界

当前同协议 fair matrix 包含：

| 方法 | 表示/边界 | 监督条件 | 可回答的问题 |
|---|---|---|---|
| Trainable K=1 | Trainable MiniLM + 单中心 Gate | Known-only train/calibration | 训练表示能否改善单中心工作点 |
| Frozen K=1 | Frozen MiniLM + 单中心 | Known-only | 冻结表示基准 |
| Frozen K=2 | Frozen MiniLM + 固定 KMeans 双中心 | Known-only | 固定多中心的直接消融 |
| Random K=2 | Frozen MiniLM + random-balanced 双中心 | Known-only | 多中心收益是否只是增加中心 |
| MOGB partition + S2C boundary | Frozen MiniLM + MOGB 粒球划分 + S2C 边界 | Known-only | 动态划分与 S2C 边界的组合 |
| S2C partition + MOGB boundary | Frozen MiniLM + S2C 划分 + MOGB 边界 | Known-only | MOGB 边界规则的组件贡献 |
| MOGB-MiniLM | Frozen MiniLM + MOGB 粒球/欧氏平均半径 | Known-only | 相同轻量表示下的 MOGB 组件工作点 |

这里的 MOGB 行不是作者 BERT 端到端完整复现；ADB、DA-ADB 和 DCLOOS 也没有进入这张同协议主表。当前外部基线状态见 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`。

## 2. 当前最可靠的整体结果

全 9 个 dataset×KIR 单元的均值（不是把不同监督合同的论文数字混在一起）：

| 方法 | OOS F1 | F1-All | F1-K | Known Recall | FA | FR | OOS F1 第1名次数 | F1-All 第1名次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trainable K=1 | 85.75 | 83.36 | 83.14 | 80.22 | 8.73 | 19.78 | 8/9 | 9/9 |
| Frozen K=1 | 78.64 | 78.30 | 78.17 | 83.50 | 23.01 | 16.50 | 0/9 | 0/9 |
| Random K=2 | 78.39 | 78.51 | 78.39 | 84.25 | 23.86 | 15.75 | 0/9 | 0/9 |
| MOGB partition + S2C boundary | 77.74 | 63.92 | 63.04 | 52.21 | 2.65 | 47.79 | 1/9 | 0/9 |
| S2C partition + MOGB boundary | 75.90 | 62.30 | 61.48 | 49.98 | 7.35 | 50.02 | 0/9 | 0/9 |
| Frozen K=2 | 75.51 | 76.79 | 76.79 | 81.44 | 25.81 | 18.56 | 0/9 | 0/9 |
| MOGB-MiniLM | 73.39 | 46.26 | 44.57 | 31.21 | 0.90 | 68.79 | 0/9 | 0/9 |

排名方向已经在 `cell_ranking.csv` 中显式固定：F1/OOS/Recall 越高越好，FA/FR 越低越好。Trainable 的 OOS F1 唯一不是第一名的单元是 Banking77/KIR=.25，第一名为 MOGB partition + S2C boundary；但 Trainable 在该单元仍需同时查看 F1-All、Known Recall 和错误预算，不能只看 OOS F1。

相对其他方法的 9 个 dataset×KIR 单元平均配对差值（Trainable 减比较方法，单位 pp）为：

| 比较对象 | ΔOOS F1 | ΔF1-All | ΔKnown Recall | ΔFA | ΔFR |
|---|---:|---:|---:|---:|---:|
| Frozen K=1 | +7.11 | +5.06 | −3.28 | −14.28 | +3.28 |
| Frozen K=2 | +10.25 | +6.57 | −1.23 | −17.07 | +1.23 |
| Random K=2 | +7.36 | +4.85 | −4.03 | −15.12 | +4.03 |
| MOGB partition + S2C boundary | +8.01 | +19.43 | +28.01 | +6.08 | −28.01 |
| S2C partition + MOGB boundary | +9.86 | +21.05 | +30.24 | +1.39 | −30.24 |
| MOGB-MiniLM | +12.36 | +37.10 | +49.01 | +7.83 | −49.01 |

这张表解释了“为什么当前自有方法看起来更好”：相对 Frozen/固定多中心，它减少 FA 的同时只付出有限 Known Recall；相对 MOGB 组件，它主要赢在保留 Known 覆盖和 F1-All，而不是在 FA 单指标上更激进。换言之，这是工作点差异证据，不是对完整 MOGB 的无条件击败证明。

## 3. 三个数据集的机制结论

### CLINC150

Trainable K=1 在 KIR=.50 达到 OOS F1 90.44、F1-All 81.82、Known Recall 74.44、FA 3.69%。Frozen K=2 的 OOS F1 为 89.20，FA 为 6.44%。固定双中心没有稳定改善；MOGB 组件进一步降低 FA，却把 Known Recall 降到 53.34%（MOGB partition + S2C boundary）或 31.57%（MOGB-MiniLM）。因此 CLINC150 的主要问题不是“拒不掉 OOS”，而是如何在拒识与 Known 覆盖之间保持平衡。

### Banking77

Trainable K=1 在 KIR=.50 的 OOS F1 为 83.56，F1-All 为 81.67，Known Recall 为 82.21%。MOGB partition + S2C boundary 的 OOS F1 为 79.40，FA 仅 3.50%，但 Known Recall 只有 52.18%。这说明动态粒球组件可形成保守拒识工作点，却没有在当前冻结 MiniLM 合同下同时保留细粒度 Known 分类能力。Banking77 仍是固定多中心相对最有可能产生条件性收益的数据集，但不能把局部 OOS F1 优势扩展成普遍结论。

### StackOverflow

StackOverflow 是最清晰的边界风险证据。KIR=.50 时：

| 方法 | OOS F1 | F1-All | Known Recall | FA | FR |
|---|---:|---:|---:|---:|---:|
| Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 | 16.11 |
| Frozen K=1 | 76.55 | 79.98 | 87.15 | 29.71 | 12.85 |
| Frozen K=2 | 63.53 | 72.76 | 86.89 | 47.17 | 13.11 |
| Random K=2 | 75.88 | 79.80 | 87.64 | 30.98 | 12.36 |
| MOGB partition + S2C boundary | 79.25 | 63.34 | 50.39 | 1.86 | 49.61 |
| MOGB-MiniLM | 72.92 | 43.30 | 27.09 | 0.79 | 72.91 |

固定 K=2 的退化主要来自新增 OOS 误接受，而不是 Known Recall 崩溃：FA 从 Frozen K=1 的 29.71% 增到 47.17%。相反，MOGB-MiniLM 的低 FA 是以 72.91% 的 Known false rejection 换来的。因此不能用低 FA 单独证明边界更好。

### 3.4 逐意图风险：三个数据集不是同一种失败

跨数据集逐样本审计覆盖 `1,890,000` 条对齐记录；结果见
[`CROSS_DATASET_ERROR_ATTRIBUTION_V1.md`](CROSS_DATASET_ERROR_ATTRIBUTION_V1.md) 和
[`CROSS_DATASET_INTENT_RISK_VISUALS_V1.md`](CROSS_DATASET_INTENT_RISK_VISUALS_V1.md)。

- StackOverflow/KIR=.50 的 Frozen K=2 OOS 误接收中，`cocoa`、`sharepoint`、`osx`、`spring`、`scala`
  五个 intent 合计约占 `64.9%`，说明退化是少数边界吸收器主导的 union-risk，而不是所有 intent
  均匀变差。
- CLINC150 的固定多中心风险相对有限，更接近局部工作点波动；Banking77 的风险更分散，部分 intent
  仍可能从局部结构受益，但其整体收益必须同时检查 Known 覆盖。
- MOGB 组件的低 false acceptance 在逐意图层面仍伴随大量 Known false rejection，因此不能用“误收更少”
  单独解释成更好的 OOS 检测。

这组证据把当前解释从“StackOverflow 数据集不适合多中心”收紧为：**数据集和 intent 的局部边界风险
分布不同，固定多个接受区域会在高风险 intent 上产生不成比例的 OOS 吸收。**

## 4. KIR 和中心数告诉我们的事情

已有 KIR 分解显示，Trainable K=1 的 OOS F1 从 KIR=.25 到 .75 的下降为 CLINC150 −12.3pp、Banking77 −23.0pp、StackOverflow −19.2pp；Frozen K=2 的下降为 −13.0pp、−26.5pp、−43.0pp。StackOverflow Frozen K=2 同时出现约 +42.2pp 的 FA 增量。

这支持一个明确的工程判断：固定多个局部接受区域的并集随开放程度扩大而更容易吸收 OOS。当前实验不支持继续枚举更大 K 来寻找统一最优；应优先解释和控制 FA/FR 工作点。

## 5. 表示层与决策层的证据链

当前证据链分为三层：

1. **性能层**：`oos_f1_heatmap.png`、`f1_all_heatmap.png` 和 `pareto_oos_f1_f1_all_kir050.png` 显示 Trainable K=1 在当前同监督矩阵中同时保持较高 F1-All 与 OOS F1。
2. **表示层**：Trainable MiniLM 的收益来自已完成的训练动态、表示几何和 detector-control 分析；它不是简单把阈值调得更严格。详见 `docs/archive/analysis/REPRESENTATION_GEOMETRY_VISUALS_V1.md`、`docs/archive/analysis/TRAINABLE_DETECTOR_MECHANISM_V1.md`。
3. **决策层**：`docs/archive/analysis/OOS_ERROR_BUDGET_V1.md`、`docs/archive/analysis/CROSS_DATASET_INTENT_RISK_VISUALS_V1.md` 以及 KIR 曲线把 FA 与 FR 拆开，说明 MOGB 组件的优势集中在保守拒识，而 Trainable K=1 更接近平衡工作点。

在 detector 控制中，同一 Trainable MiniLM 表示下，Gate 相对 MSP、Energy、kNN、LOF 在三个数据集均为
3/3 seed 胜出；StackOverflow 的 OOS F1 配对区间相对 MSP/Energy 分别为 `+41.10/+43.76pp`。
但 Trainable-Frozen 的原生 detector 差值并非全部为正。因此“当前方法更好”不能简化成“MiniLM
训练本身更好”，而应理解为表示适配与 Known-only 几何 Gate 的组合。

## 5.1 跨 KIR 的工作点与多指标前沿补充

两项不重训的后验分析进一步把“Trainable 为什么更好”拆开：

- [`CROSS_KIR_MATCHED_FRONTIER_V1.md`](CROSS_KIR_MATCHED_FRONTIER_V1.md) 在相同 Known Recall≈80%/90%/95% 的工作点比较 Trainable-K1 与 MOGB-Fair；全部 27 个 dataset×KIR×target 组合均为 5/5 seed 配对胜出。它支持分数排序/表示分离优势，而不是默认阈值偶然有利，但目标覆盖率来自测试后诊断，不能作为调参。
- [`CROSS_KIR_TRANSITION_ATTRIBUTION_V1.md`](CROSS_KIR_TRANSITION_ATTRIBUTION_V1.md) 对 45 个逐样本配对单元做五状态错误预算归因。跨三个 KIR，Trainable 的 F1-All 提升主要来自恢复 MOGB 判为 `Known rejected` 的样本；OOS 正确净增通常为负。因此当前优势不是更激进地拒绝 OOS，而是减少 MOGB 平均半径造成的 Known 覆盖损失。
- [`CROSS_KIR_PARETO_FRONTIER_V1.md`](CROSS_KIR_PARETO_FRONTIER_V1.md) 在 OOS F1、F1-All、Known Recall 和 false acceptance 四个目标上计算五 seed 均值前沿。Trainable-K1 在 9/9 工作点未被其他 fair 方法同时支配，但也不存在单一方法全面支配所有方法，说明结论是平衡工作点优势而非单指标统治。

当前还缺少的是同一监督合同下的官方 ADB/DA-ADB/DCLOOS 多 seed 主表，以及作者 BERT 完整 MOGB 的可重复主表；因此目前不能宣称“超过 SOTA”。DA-ADB 已有一个有效隔离 CUDA 单格，但仍处于 `valid_external_cell_pending_replication`。

## 6. 外部 baseline 的当前可见结果与边界

StackOverflow/KIR=.50 的历史兼容性单格如下，仅作合同参照：

| 方法 | OOS F1 | F1-All | 表示/监督 | seed | 是否进入 fair 主表 |
|---|---:|---:|---|---:|---|
| DA-ADB | 90.90（旧）/70.82（新） | 89.23（旧）/72.03（新） | BERT，端到端 Known-only | 1/1 | 否；旧新合同待对齐 |
| ADB | 89.47 | 87.63 | BERT，端到端 Known-only | 1 | 否 |
| Trainable K=1 | 87.67 | 86.55 | Trainable MiniLM，Known-only | 5 | 是 |
| DCLOOS reduced | 87.05 | 90.26 | BERT，伪 OOS+外部 OOS | 1 | 否 |
| MOGB 官方兼容单格 | 79.97（指标合同不同） | 68.35 | BERT，官方逻辑兼容运行 | 1 | 否 |

因此，当前可以明确说：Trainable K=1 在自己的五 seed、同协议矩阵中优于冻结 MiniLM/MOGB 组件；ADB/DA-ADB 的 BERT 外部 cell 只能作合同参照，DCLOOS 使用额外未知监督，不能直接与 Known-only 结果排名。DA-ADB 新单格已可审计但远低于旧兼容单格，完整外部 baseline 仍需 split/监督/环境合同审计。

## 6.1 外部基线运行的最新补充

在前述历史兼容单格之外，已经完成 StackOverflow/KIR=.50、seed=42/87/100 的 ADB 同数据合同单元：

| 方法 | OOS F1 均值±std | F1-All 均值±std | 结果层 |
|---|---:|---:|---|
| S2C Trainable K=1 | 87.67±1.66 | 86.55±1.30 | current protocol_v2 fair Gate |
| ADB | 87.36±1.61 | 85.66±1.59 | BERT/TextOIR external compatibility，3 个 CUDA seed |

ADB 数字来自逐样本 `y_true.npy/y_pred.npy`，不是直接信任 legacy `results.csv`；它仍不能与 MiniLM fair
Gate 视作同一训练条件。DA-ADB 新隔离 CUDA 单格已出现有限且非塌缩预测（OOS F1=70.82%），但旧/新
合同尚未对齐，因此只进入外部单格审计，不进入表格排名。DCLOOS 的外部 SQuAD 负样本来源已经闭合，但默认预算单元超时；reduced 兼容结果继续
单独保存，且不得与当前 Known-only fair 行混排。

详情：`docs/analysis/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`、
`results/analysis/archive/analysis/comparison_atlas_v1/stackoverflow_kir50_external_summary.csv`。

## 7. 可视化入口

本轮新生成的权威图：

- `figures/archive/analysis/experiment_analysis_master_v1/oos_f1_heatmap.png`
- `figures/archive/analysis/experiment_analysis_master_v1/f1_all_heatmap.png`
- `figures/archive/analysis/experiment_analysis_master_v1/pareto_oos_f1_f1_all_kir050.png`
- `figures/archive/analysis/experiment_analysis_master_v1/method_rank1_counts.png`
- `figures/archive/analysis/experiment_analysis_master_v1/oos_f1_kir_curves.png`
- `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_delta_heatmaps.png`
- `figures/archive/analysis/cross_kir_transition_attribution_v1/cross_kir_transition_attribution.png`
- `figures/archive/analysis/cross_kir_transition_attribution_v1/known_oos_budget_cross_kir.png`
- `figures/archive/analysis/cross_kir_pareto_frontier_v1/cross_kir_pareto_frontier.png`
- `figures/archive/analysis/cross_kir_pareto_frontier_v1/pareto_cell_counts.png`

机器可读结果：

- `results/analysis/archive/analysis/experiment_analysis_master_v1/audited_summary.csv`
- `results/analysis/archive/analysis/experiment_analysis_master_v1/cell_ranking.csv`
- `results/analysis/archive/analysis/experiment_analysis_master_v1/method_global_summary.csv`
- `results/analysis/archive/analysis/experiment_analysis_master_v1/trainable_paired_effects.csv`
- `results/analysis/archive/analysis/experiment_analysis_master_v1/MANIFEST.json`
- `results/analysis/archive/analysis/cross_kir_matched_frontier_v1/`
- `results/analysis/archive/analysis/cross_kir_transition_attribution_v1/`
- `results/analysis/archive/analysis/cross_kir_pareto_frontier_v1/`

## 8. 目前的结论边界与下一步

当前能够确认的是：在相同 `protocol_v2_textoir_v1`、Known-only、五 seed 的 Gate/组件比较中，Trainable K=1 是最稳定且最平衡的自有结果；固定 K>1 在 StackOverflow 上存在明确的 FA 过覆盖；MOGB 轻量组件不是全面替代，而是更保守的工作点。

当前不能确认的是：Trainable K=1 已经超过作者完整 MOGB、ADB、DA-ADB 或 DCLOOS，也不能把历史 `fulltex.tex` Cascade 数字直接与本 Gate-only 表合并。

唯一下一步是收口 DA-ADB 数值稳定性与 DCLOOS 外部 OOS 合同，并把已完成 ADB 三 seed 单格持续保留在
外部合同层；不新增 K、损失项或自适应多中心矩阵。当前分析重点继续是错误预算、Known/OOS 工作点和
表示/边界归因，而不是重新设计方法。

## 9. DCLOOS 端到端合同收口（2026-08-10）

DCLOOS 官方代码现已完成外部 SQuAD 快照接入和隔离 smoke。当前 protocol 的
StackOverflow/KIR=.50/seed=42 registry 固定 10 个 Known intent，而 DCLOOS 在正样本快照上按自身
随机规则重新选 5 个 Known intent，并使用 pseudo-OOS 与外部 SQuAD。因此 DCLOOS 不能进入当前
Known-only fair 主表。

完整 1 epoch smoke 的 OOS F1=0、F1-All=8.31% 只说明训练/验证/测试/指标导出链路已通，不是性能
结论；既有 KIR=.75/seed=888 reduced 单元也只能作为不同监督合同的外部参考。下一步应先实现固定
registry Known-list adapter，再运行一个收敛单格，之后才考虑多 seed。详见
`docs/archive/analysis/DCLOOS_CONTRACT_STATUS_V1.md`。
