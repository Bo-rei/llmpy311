# 历史 SOTA 方法、基线对比与 MOGB 差距（V1）

> 本报告只解释 `fulltex.tex` 的历史结果与当前 `protocol_v2_textoir_v1` 的公平 Gate 结果，不能把两种合同合并排名。

## 1. 论文中真正取得高 OOS F1 的方法

论文表格中的 `Ours` 是完整 Gate–Router–Expert Cascade，而不是当前实验中的 `S2C-Trainable-K1`。历史 Gate 使用冻结的 all-MiniLM-L6-v2、多中心 KMeans 和局部 `μ+λσ` 半径；Router/Expert 使用 SmolLM-135M 与 LoRA。历史主表将它与 MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB、DA-ADB 比较。

本报告直接解析 `fulltex.tex` 的 `tab:main_results_all`，源文件 SHA256 为 `83322d7dff65ba22e489c67aaa2bfe70a8afb8ef8526e0a53294dfaf4557a61e`，共 72 个 dataset×KIR×method 单元。

## 2. 历史表中的优势是什么

历史 `Ours` 的优势主要体现在 OOS F1；Known F1 和 Accuracy 并非在每个格子都第一。例如 KIR=0.50 时，StackOverflow 的 Ours OOS F1=89.71，高于 ADB=87.70 和 DA-ADB=88.86，但 DA-ADB 的 Known F1=86.71、Accuracy=87.78 高于 Ours 的 75.48、85.54。也就是说，历史论文的 SOTA 叙述是“完整 Cascade 的 OOS F1 优势”，不是所有指标都全面领先。

在 9 个历史 dataset×KIR 格子中，Ours 的 OOS F1 高于该格所有七个列出的基线的格子数为 9。逐格差值见 `results/analysis/historical_sota_comparison_v1/ours_minus_best_baseline.csv`。

## 3. 当前实验与历史方法不是同一个比较对象

当前最强自有行是 `S2C-Trainable-K1`：Known-only 训练 MiniLM、单中心 Gate、五 seed；它没有 Router/Expert，也没有历史 Cascade 的数据/阈值合同。因此它不能直接宣称复现或超过 fulltex 的 `Ours`。当前七行 fair matrix 仍然只支持：Trainable K=1 在当前统一协议下优于 Frozen/fixed-K 和冻结 MiniLM 的 MOGB 组件。

完整的 MOGB、ADB、DA-ADB、DCLOOS 尚未形成同一 protocol_v2 多 seed 主表：MOGB 官方 BERT 只完成了现代兼容单格，ADB/DA-ADB 被 torch/CUDA runtime 阻塞，DCLOOS reduced 结果使用 pseudo-OOS 与外部 OOS。

## 4. MOGB 为什么与论文数字差很多：当前可验证的差距类别

1. **数据合同差异**：MOGB 论文使用作者快照和原始抽样/划分；当前 fair matrix 使用 TEXTOIR 固定快照、固定 registry、Known intent 列表与 split。样本 ID、类别列表、train/dev/test 计数未证明完全一致。
2. **训练合同差异**：论文 MOGB 是 BERT+投影层、最近子中心损失、粒球递归划分和交替表示更新；当前 MOGB-Fair 是冻结 MiniLM 组件，官方 BERT 单格则是旧代码的现代兼容运行，不能把二者等同。
3. **兼容层风险**：官方仓库依赖旧版 PyTorch/BERT、硬编码 CUDA 和旧张量/计算图语义。现代兼容层可能改变 optimizer、梯度流、聚类更新顺序、checkpoint 选择或设备行为；在这些逐阶段差异未对齐前，不能说“算法复现失败”。
4. **指标合同差异**：论文报告 Acc、F1-All、F1-U、F1-K，并依赖作者自己的 OOS 映射与评估器；当前统一评估器使用 protocol_v2 的 canonical/registry 和 Known/OOS 语义。数值字段同名不等于样本集合相同。
5. **超参数/随机性差异**：纯度阈值、最小粒球大小、聚类更新 step、epoch、学习率、投影维度和 seed 都需要逐项锁定；论文表格可能是单次或作者选择的运行点，当前单格不能推断其统计分布。

目前已有的四组合静态诊断还给出两个更具体的事实：本地 pinned MOGB checkout 的最早失败点是 `MOGB.py:3` 导入缺失的 `utils` 包；作者配套数据目录也未在本地材料中发现。因此“原代码+原数据”“原代码+当前数据”“现代兼容+原数据”三个隔离组合均不能运行，当前可见的现代兼容+当前 TEXTOIR 结果只是已有 D 组合 artifact。四组合证据见 `results/diagnostics/mogb_diff/stage_comparison.csv` 和 `first_divergence.json`。

因此当前 MOGB 结果应标记为 `official_code_not_reproduced_under_available_materials`，而不是用低分证明 MOGB 算法本身无效。真正的下一步是逐阶段 discrepancy audit：先复核作者数据计数/样本 ID/已知类别，再锁定 BERT、tokenizer、优化器、粒球统计和 checkpoint，最后用独立评估器重算四个论文指标。

## 5. 已有与新增可视化

已有当前协议可视化位于 `figures/experiment_analysis_master_v1/`、`figures/mechanism_evidence_v2/`、`figures/mogb_operating_point_visuals_v1/` 和 `figures/baseline_contract_visuals_v1/`；它们解释了 Trainable K=1、固定多中心和 MOGB 组件的工作点与错误预算。

本报告新增：
- `figures/historical_sota_comparison_v1/historical_oos_f1_heatmap.png`：fulltex 历史表的 OOS F1 热力图；
- `figures/historical_sota_comparison_v1/ours_minus_best_baseline.png`：历史 Ours 相对每格最强列出基线的差值；
- `figures/historical_sota_comparison_v1/stackoverflow_contract_layers.png`：StackOverflow/KIR=.50 历史 Cascade 与当前 Gate 合同分层。

## 6. 当前唯一可信结论

历史论文方法是完整 Cascade `Ours`，它在旧合同下取得了很强的 OOS F1；当前 Trainable K=1 是新的、Gate-only 的自有候选，在当前五 seed Known-only 合同下表现最好，但不能直接称为论文历史 SOTA，也不能直接称为超过完整 MOGB/ADB/DA-ADB/DCLOOS。下一步不是继续堆固定 K，而是完成 MOGB discrepancy audit，并在稳定 runtime 下补同协议外部 baseline。
