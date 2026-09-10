# S2C 统一对比计划与附件要求差距审计 V1

更新时间：2026-08-11
活动协议：`protocol_v2_textoir_v1`

## 结论

上一版十阶段 plan 与附件的研究方向**高度一致**，没有把任务偷换成新模型开发：两者都把
`S2C-Trainable-K1` 固定为 Known-only MiniLM、K=1 单中心 Gate，并要求通过统一合同、同协议
比较、错误迁移、表示/边界机制和监督差异来解释收益与代价。

但上一版 plan 对应的**完整实验交付尚未完成**。当前可以确认完成的是：

- 当前 fair MiniLM Gate/MOGB 组件的主矩阵和跨 Dataset/KIR/Seed 汇总；
- 对已有可用方法的匿名逐样本合同、sample-id/标签序列审计和合同分层；
- S2C–Frozen、S2C–MOGB 以及 S2C–ADB 的部分错误、score、Pareto、KIR/seed 和机制证据；
- MOGB 粒球/半径/工作点/错误转移的大部分专项分析。

尚不能写成“附件要求已经全部满足”的部分是：当前 final-metrics 的 KNNCL、OpenMax、DOC、
DeepUnk 合同，DA-ADB 的完整矩阵，DCLOOS 的收敛监督前沿，S2C-BERT/ADB-MiniLM backbone
控制，error-aware UMAP，固定 near-OOS 全方法分桶，low-resource 和统一 performance-cost
Pareto。相关状态均保留为缺失或阻塞，没有用历史表格或中间预测填空。

机器可读审计矩阵见 [`requirement_matrix.csv`](../../results/analysis/archive/analysis/plan_alignment_audit_v1/requirement_matrix.csv)，
补充执行计划见 [`UNIFIED_COMPARISON_SUPPLEMENT_PLAN_V1.md`](UNIFIED_COMPARISON_SUPPLEMENT_PLAN_V1.md)。

## 审计口径

本审计把“文件/脚本存在”与“实验完成”分开。只有同时满足 final metrics、逐样本来源、split/Known
list/registry 对齐、选择过程可审计、manifest 完整且图有源 CSV，才把一项视为完成。已有 test OOS
标签可以用于事后错误解释，但不能用于主结果选阈值、选 checkpoint、选分桶或晋级方法。

当前统一合同的事实基线为：`2,394,360` 行、`486` 个运行组、`441` 个同序列对齐 method-cell，
另有 `45` 个 ADB 外部 BERT 合同 cell；fair 汇总为 `63` 行。ADB 的 sample-id 桥接结果仍保持
`external_backbone`，不是 MiniLM fair 行。DCLOOS 的 timeout、smoke 和 validation-best 中间预测
不进入正式性能表。

## 一致性判断

| 附件的主要求 | 上一版 plan 的对应阶段 | 一致性 | 当前完成度 |
|---|---|---|---|
| 不增加新模型，先闭合公平主表 | 总目标、阶段二、停止条件 | 一致 | 部分完成 |
| 统一 prediction schema 与对齐审计 | 阶段一 | 一致 | 对已有方法部分完成 |
| 缺失 baseline 先 smoke、再扩展矩阵 | 阶段二 | 一致 | 未完成 |
| Backbone × Method 控制 | 阶段三 | 一致 | 未完成 |
| 性能层图和 calibration-only 工作点 | 阶段四 | 一致 | 部分完成 |
| score/几何/UMAP/near-OOS 机制链 | 阶段五 | 一致 | 部分完成；UMAP 未完成 |
| MOGB 专项机制面板 | 阶段六 | 一致 | 大部分完成，最终合并未闭合 |
| DCLOOS 监督差异层 | 阶段七 | 一致 | 未完成 |
| 样本级错误迁移 | 阶段八 | 一致 | 部分完成 |
| low-resource、threshold、efficiency | 阶段九 | 一致 | 部分完成；low-resource/cost 未完成 |
| 中文总报告、机器可读 manifest、验收 | 阶段十 | 一致 | 部分完成 |

因此，上一版 plan 与附件之间没有需要推翻的方向性冲突；需要补的是执行门槛、证据覆盖和依赖顺序，
而不是再设计一个新方法。

## P0–P5 完成判定

详细逐条判定位于 CSV；下面只列对最终结论有影响的聚合结果。

### P0：合同和对齐

P0 对**已有 fair/native/ADB 输入**已经落地：schema 包含附件要求的 base fields、Gate/MOGB/DCLOOS
method-specific fields，匿名行级 JSONL 保持在本地 artifact，不含文本和 embedding。对齐报告也没有
把 ADB 的外部合同强行提升为 fair。

P0 尚未“全完成”的原因是附件要求最终覆盖所有可用方法，而 KNNCL、OpenMax/DOC/DeepUnk 没有当前
final-metrics 行，DCLOOS 也没有 converged final row。它们目前正确的状态是 blocked，不是 0 分或伪预测。

### P1：baseline 和 backbone

当前 fair 主矩阵可支撑 S2C-Trainable-K1 与 Frozen/MOGB 组件的同协议比较；ADB 已有外部 BERT
工作点，DA-ADB 有 StackOverflow/KIR=.50/三个 seed 的外部 summary。它们不能替代附件要求的
KNNCL、传统 TextOIR detector、完整 DA-ADB 以及 DCLOOS 监督层，也不能回答“优势来自方法还是
MiniLM”的 2×2 backbone 控制。

### P2–P3：性能与机制

已有 heatmap、Pareto、KIR/seed、错误预算、score 分布、AUROC/AUPR、几何汇总、near-OOS 诊断和
多个 intent/MOGB 热图；这些证据已经能支持“Trainable-K1 的收益是数据集/KIR 相关的 coverage–
rejection 工作点与 score separation 变化”，但还不能支持附件要求的完整方法覆盖。

特别需要纠正一个容易误读的地方：当前统一 figure manifest 中没有 error-aware UMAP，也没有公开
可审计的统一 embedding artifact。现有 error-quadrant/geometry 图不能等价替代 UMAP；若之后仍取不到
合规 embedding，应保留 blocked，而不是从 score 汇总反推 UMAP。

### P4：MOGB 与 DCLOOS

MOGB 证据已经最接近闭合：ball count/radius/risk、Known rescue、component swap、Trainable–MOGB
transition 和工作点曲线都有独立源表。剩余工作主要是把它们作为最终 dashboard 统一登记，并明确
哪些中心/ball 字段缺失。

DCLOOS 则相反：当前已有的价值是监督合同和资源阻塞审计，不是性能前沿。固定 registry 的运行未产出
final metrics；KIR=.75/seed=888 reduced 结果含 pseudo-OOS 与外部 SQuAD，不能与 Known-only
主表并列。

### P5：鲁棒性、成本和最终交付

threshold/radius stability 已有部分 Gate/MOGB 证据；但没有 10/25/50/75/100% low-resource
包，也没有覆盖 S2C、Frozen、MOGB、ADB、DA-ADB、DCLOOS 的统一参数/时间/内存/吞吐 Pareto。
因此“主文 8 图”目前只是目标集合，不是已经全部生成并验收的交付集合。

## 当前允许写出的结论

可以写：在 `protocol_v2_textoir_v1` 的 Known-only MiniLM fair 层，S2C-Trainable-K1 相对 Frozen
和 MOGB-Fair 组件表现出更平衡的 Known coverage–OOS rejection 工作点；已有 score/错误/MOGB
组件分析支持“表示适配、score separation 与边界工作点共同作用”的解释。对 ADB/DA-ADB，只能写
外部 BERT/TextOIR 合同下的工作点参照和配对错误预算。

不可以写：已经超过完整 MOGB、完整 DCLOOS、KNNCL 或 TextOIR 全部方法；已经完成附件中的完整
8 图；或者仅凭当前 fair 层就证明收益只来自 MiniLM 而非方法/训练合同。

## 交付状态

本文件是当前研究状态的审计解释，不把缺失工作伪装成完成。后续按补充计划推进；每个阶段都必须
先满足 entry gate，再生成下一层图表或扩展矩阵。
