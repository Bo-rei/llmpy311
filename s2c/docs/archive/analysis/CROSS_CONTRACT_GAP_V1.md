# 跨合同实验差距审计（V1）

> 这是实验阶段的统一阅读入口。它把结果对齐到同一数据集和 KIR，但保留训练监督、骨干、系统层级和评价合同；因此不生成一个不公平的 SOTA 排名。

## 1. 当前到底比较哪个自有方法

当前 fair 矩阵中的自有最佳对象是 **S2C-Trainable-K1**：只使用 Known train/calibration 训练 MiniLM 最后两层和 projection，随后使用 K=1 Gate；不是 `fulltex.tex` 中的完整 Gate–Router–Expert Cascade，也不是 RC-AMBL/joint-adaptive 候选。

历史论文的 `Ours` 是旧合同下的完整 Cascade：Frozen MiniLM Gate + Router/Expert。它的高 OOS F1 不能直接归因于单独的 Gate 或固定多中心。

## 2. 当前可复算的同合同结论

来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。在 3 数据集 × 3 KIR × 5 seeds 的同一 protocol_v2 Gate 矩阵中：

- Trainable-K1 的 OOS F1 九格均值为 85.75%，F1-All 九格均值为 83.36%；
- 它在 OOS F1 上 9 格赢 8 格，在 F1-All 上赢 9 格；
- 相比 Frozen-K1，平均 OOS F1 提升 7.11pp；相比 Frozen-K2，提升 10.25pp；
- 这证明的是当前 Known-only MiniLM Gate 合同内的表示适配收益，不是跨论文 SOTA。

## 3. 与 fulltex 历史 Ours 的逐格差距

逐格表见 `results/analysis/archive/analysis/cross_contract_gap_v1/current_vs_historical.csv`。当前 Trainable-K1 在 OOS F1 上 9 格中高于历史 Ours 的格数为 3/9，平均差值为 -3.42pp；StackOverflow/KIR=.50 当前 OOS F1 为 87.67%、历史为 89.71%，差值 -2.04pp。这个差距主要反映 Gate-only 与完整 Cascade 的系统层级差异，不能被解释为 Trainable encoder 本身“没有效果”。

同一格的其他指标更能说明问题：StackOverflow/KIR=.50 当前 F1-K 为 86.44%、历史 Known F1 为 75.48%，当前高 +10.96pp；当前 Accuracy 为 86.81%、历史为 85.54%，差值 +1.27pp。也就是说，当前单中心 Gate 的 Known 分类指标并不弱，历史 OOS F1 优势不能简单归因为 encoder 更强，而应归因于完整 Cascade 的系统合同和 OOS 工作点。

可视化：`figures/archive/analysis/cross_contract_gap_v1/current_vs_historical.svg`。紫色是历史完整 Cascade，绿色是当前 Trainable-K1；下半图只展示描述性差值。

## 4. MOGB、ADB、DCLOOS 应如何放置

| 对象 | 当前证据 | 能否与 Trainable-K1 直接排名 |
|---|---|---|
| MOGB-MiniLM-Fair | 同 TEXTOIR registry、Frozen MiniLM、MOGB 粒球/边界；45 配对单元 | 可以作为同表示组件对照 |
| MOGB 官方逻辑单格 | BERT + 现代兼容层；StackOverflow KIR=.50/seed0 未复现论文工作点 | 不能当作完整论文 MOGB 排名 |
| ADB | BERT/TextOIR 兼容 3 cell，StackOverflow KIR=.50 OOS F1 87.47±1.48% | 只能作外部兼容参照 |
| DA-ADB | 当前 NaN/全类预测 | 无效，不能进入排名 |
| DCLOOS reduced | BERT + pseudo-OOS + 外部 SQuAD OOS，KIR=.75/seed888 | 监督合同更强、不能进入 Known-only 排名 |

外部机器可读来源：`results/analysis/archive/analysis/comparison_atlas_v1/external_contract_reference.csv`。当前 StackOverflow/KIR=.50 的分层图为 `figures/archive/analysis/experiment_comparison_overview_v2/stackoverflow_kir50_contract_layers.png`。

## 5. MOGB 论文差距的正确解释

MOGB 本地 BERT 兼容单格为 Accuracy=75.17、F1-All=68.35、F1-U=79.97、F1-K=67.19；论文公开参考为 88.67、87.49、89.71、87.27。当前代码审计和 Known-only 归因已经验证：

1. 官方子中心损失的 L1 距离归一化压窄了训练信号；
2. 平均半径工作点过窄，严重误拒 Known；
3. selected-ball 可能遗漏注册 Known 类；
4. 作者数据快照、Known 列表、旧运行环境和最终粒球随机状态未完整恢复。

修复单一损失或单纯扩大半径都不能恢复论文工作点，所以当前状态必须写成 `official_code_not_reproduced_under_available_materials`，不能写成“本地低分证明 MOGB 无效”。详细证据见 `docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`。

## 6. 当前实验阶段的下一步

1. 不再重复 E2/E3、旧 R1、固定 K/KIR 扫描或 MOGB-Fair 组件矩阵；
2. 保留 Trainable-K1 作为当前自有安全基线；
3. 若继续做外部实验，只在独立运行时先完成 DA-ADB 数值审计和 DCLOOS 默认单格；不拿 reduced/invalid 结果填主表；
4. 继续补的分析应围绕现有 CSV：逐数据集/KIR/seed 的错误预算、Known/OOS 工作点、MOGB 粒球风险和历史 Cascade 合同差距；
5. 在外部方法合同闭合前，不宣称跨合同 SOTA。

## 7. 证据入口

- 当前对比总览：`docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`
- 当前可视化证据包：`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V2.md`
- MOGB 机制闭环：`docs/archive/analysis/MECHANISM_CLOSURE_V1.md`
- 历史 fulltex 对照：`docs/archive/analysis/HISTORICAL_SOTA_AND_CURRENT_COMPARISON_V1.md`
- 外部状态：`docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`

本文件和 SVG 只使用已完成轻量结果，不修改训练 artifact、不读取测试 OOS 进行选参。
