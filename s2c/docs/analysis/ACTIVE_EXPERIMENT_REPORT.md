# 当前 s2c 实验证据总览（active_experiment_dashboard_v1）

> 本报告只汇总已有轻量 CSV，不重新训练、不覆盖历史 artifact。结果先按实验合同分层，再讨论性能；外部兼容性数字不自动并入同协议主排名。

中文集中版：[`EXPERIMENT_COMPARISON_ZH.md`](EXPERIMENT_COMPARISON_ZH.md)。

## 1. 总体判断

当前项目已经完成大量固定 K、KIR、随机分簇、聚类诊断、MiniLM 表示、MOGB 组件和端到端兼容性实验。当前主要缺口不是没有数字，而是数字属于不同实验合同。现有证据支持：

- 固定 K 的收益依赖数据集和 KIR，不存在跨数据集统一最优 K；
- Banking77 在部分条件下从多中心获益，CLINC150 收益小且不稳定，StackOverflow 固定多中心持续过覆盖；
- Known-only Trainable MiniLM 对 K=1 有稳定正收益，但没有自动修复 K=2 的多球 false acceptance；
- 训练参与式 joint-adaptive pilot 已真正执行，但候选 split 全部被 Known calibration 安全门拒绝，最终 K_y=1；
- MOGB 官方 BERT、ADB/DA-ADB 和 DCLOOS 当前主要是严格负复现或兼容性证据，不能和冻结 MiniLM fair matrix 直接排名。

## 2. StackOverflow 当前最关键对照

| 方法 | OOS F1 | F1-All | Known Recall | False acceptance | 协议层 |
|---|---:|---:|---:|---:|---|
| frozen_k1 | 77.29% | 78.60% | 83.71% | 26.54% | 当前 Gate |
| trainable_k1 | 86.71% | 85.65% | 83.92% | 11.14% | 当前 Gate |
| trainable_fixed_k2 | 67.65% | 76.81% | 93.62% | 45.26% | 当前 Gate |

Trainable K=1 相对 Frozen K=1 的 OOS F1 提升约 9.42 个百分点，说明表示适配本身有效；Trainable K=2 的 OOS F1 大幅下降并伴随 false acceptance 上升，说明当前瓶颈是多球接受区域组合，而不是单纯缺少训练。

## 3. K/KIR 与数据集差异

`k_sweep_oos_f1.png` 展示固定 Gate-only 结果：CLINC150 通常在 K=2 附近达到局部峰值后回落；Banking77 在部分距离下随 K 增大受益但 Known 覆盖有代价；StackOverflow 从 K>1 开始显著退化，且 KIR 越高过覆盖风险越明显。

这说明多中心是否有效不是单一方法属性，而是数据集语义结构、表示空间、半径规则和接受区域并集共同决定的结果。

`k_selection_tradeoff.png` 和 `k_selection_summary.csv` 将测试集 oracle 最优 K 与 Known Recall 约束下的诊断 K 分开记录；这些结果只用于解释 KIR/数据集异质性，不能被当作正式验证选 K。
`fair_component_gaps.png` 和 `fair_component_gaps.csv` 则在同一冻结 MiniLM、同一 KIR=0.50 下，以 Single centroid 为基准分解随机分簇、固定 K=2、MOGB 粒球和边界替换的增量。

## 4. MiniLM 表示实验

表示对照图同时画 Frozen、CE、SupCon、CE-Recon 及其 K=1/K=2 结果，用于区分表示训练能否改善单中心 OOS 排序，以及这种改善能否传递到固定多中心边界。当前已有结果显示前者较明确，后者不稳定。

## 5. MOGB 与外部基线

同协议 MOGB fair matrix 可用于组件归因；官方 BERT MOGB 未复现论文参考数值；ADB/DA-ADB 只有单 seed compatibility artifact；DCLOOS 使用 pseudo-OOS 与外部 OOS，且正式运行未完成。因此不能用一张柱状图宣称我的方法已经超过 MOGB/DCLOOS。

`stackoverflow_known_oos_tradeoff.png` 将 Known Macro-F1 与 OOS F1 放在同一坐标系，并把 `fulltex.tex` 的历史表面值单独标为不可比点；它不是当前协议的 SOTA 排名。
`historical_latex_metric_audit.csv` 逐项记录论文表值、JSON 覆盖值和 raw `primary_metrics`，用于防止历史 override 被误当成当前可复算结果。
`current_cascade_gate_comparison.png` 只显示当前 36-unit Cascade 的 Gate 对照，不与历史论文表混合。

## 6. KIR=0.50 方法分层对照

`KIR50_METHOD_COMPARISON_V1.md` 将 Trainable K=1/K=2、冻结 MiniLM 组件和 ADB/DA-ADB/BRAK 兼容结果放在同一张分层表中。StackOverflow 的 Trainable K=1 为 86.71%，高于同协议 Frozen Single centroid 76.55%、MOGB-MiniLM 72.92% 和 MOGB partition+s2c boundary 79.25%；ADB/DA-ADB 分别为 89.47%/90.90%，但属于 BERT/不同训练合同的兼容单格，不能直接视为公平超越或落后。
详见 `docs/analysis/KIR50_METHOD_COMPARISON_V1.md`、`kir50_method_layers.png` 和 `kir50_method_tradeoff.png`。

## 7. Trainable MiniLM 的 λ/K 受控分析

`minilm_trainable_lambda_control_v1` 在同一 checkpoint 上评价 λ={0.50,0.75,1.00,1.25,1.50,2.00}，选择规则只使用 Known calibration。
- banking77：K=2−K=1 OOS F1 `+2.08pp`，Known Recall `-1.21pp`，false acceptance `-3.27pp`。
- clinc150：K=2−K=1 OOS F1 `+0.79pp`，Known Recall `-1.88pp`，false acceptance `-2.44pp`。
- stackoverflow：K=2−K=1 OOS F1 `-9.51pp`，Known Recall `+2.27pp`，false acceptance `+11.83pp`。
这说明 StackOverflow 的 K=2 退化不是 λ=1 单点设置造成，Trainable MiniLM 的主要收益仍属于 K=1 表示和分数排序。
详见 `docs/analysis/MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md` 和 `trainable_lambda_k_interaction.png`。

## 8. 自适应多中心的实际结果

`adaptive_decision_summary.png` 显示 RC-AMBL、joint adaptive 和 contract repair 的候选 split 均被拒绝，最终平均 K_y=1。当前实现已经包含候选分裂、共同训练、Known-only calibration 选择和安全回退，但 StackOverflow 上没有找到安全的新增中心。

## 9. 下一步实验优先级

1. 对已有 Trainable/Frozen/MOGB fair rows 做逐数据集、逐 KIR、逐 seed 的主表与置信区间汇总；
2. 对 MOGB fair matrix 做逐意图 false-accept/false-reject 归因，不继续盲目扩大官方 BERT 复现；
3. 在统一监督条件、数据划分和随机种子后，再把 ADB、DA-ADB、DCLOOS 纳入正式比较；
4. 接入完整 Cascade 前先冻结 Gate 候选，分别验证 Frozen K=1 与 Trainable K=1 的下游传递；
5. 继续多中心前保留 Trainable K=1 为安全基线，新增中心必须通过 Known-only 风险门。

## 10. 图表文件

- `figures/active_experiment_dashboard_v1/k_sweep_oos_f1.png`
- `figures/active_experiment_dashboard_v1/trainable_k1_k2_tradeoff.png`
- `figures/active_experiment_dashboard_v1/representation_k_interaction.png`
- `figures/active_experiment_dashboard_v1/stackoverflow_baseline_layers.png`
- `figures/active_experiment_dashboard_v1/stackoverflow_known_oos_tradeoff.png`
- `figures/active_experiment_dashboard_v1/current_cascade_gate_comparison.png`
- `figures/active_experiment_dashboard_v1/k_selection_tradeoff.png`
- `figures/active_experiment_dashboard_v1/fair_component_gaps.png`
- `figures/active_experiment_dashboard_v1/trainable_lambda_k_interaction.png`
- `figures/active_experiment_dashboard_v1/kir50_method_layers.png`
- `figures/active_experiment_dashboard_v1/kir50_method_tradeoff.png`
- `figures/active_experiment_dashboard_v1/adaptive_decision_summary.png`

历史 R1/R1-full 中已被 contract audit 标记为 superseded 或 exploratory 的几何和 test-defined near-OOS 结果不进入本报告正式结论。
