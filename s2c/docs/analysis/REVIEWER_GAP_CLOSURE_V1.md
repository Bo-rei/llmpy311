# CCF A/B 审稿意见逐项闭环审计

更新时间：2026-09-09。

这份文档把审稿意见与当前工作区的可审计证据逐项对齐。它不把不同数据协议、backbone、监督条件或系统层级的结果混成一个 SOTA 排名。

## 总体判断

实验完整性已经明显提高，但仍不能声称“完整 MOGB/DCLOOS/H0 SOTA 已严格复现”。当前最稳妥的论文主张是：

> 在 historical v19 H1、Known-only、MiniLM Gate、固定 Router/Expert 的统一合同下，Trainable Gate 相对 Frozen Gate 改善 OOS F1、Accuracy 和完整 Known F1，并降低 OOS false acceptance；这种收益在 K=1 单中心边界下稳定，不能外推为 Trainable 表示自动使固定多中心边界有效。

## 要求逐项核对

| 审稿人要求 | 当前证据 | 状态 | 可以写的结论 |
|---|---|---|---|
| 加入 MOGB 直接对比 | cross_method_oos_mechanism_presentation.md、[`official_mogb_status.csv`](../../results/analysis/cross_method_oos_mechanism/official_mogb_status.csv)、cross_protocol_tradeoff_v1、MOGB-Fair component、MOGB ball 风险图 | 部分完成 | 已同时报告同协议 MiniLM/MOGB component 与官方代码的本地兼容状态；不能称为完整官方 MOGB 严格复现 |
| K=1,2,3,4,5 与 adaptive K | fulltex.tex 受控 K 表、1650 个 Gate 单元、adaptive-center 和 random/KMeans 分析 | 已完成（Gate层） | 多中心收益依赖数据集；K=2 不是普适最优，StackOverflow 多中心会扩大 OOS 接受区域 |
| Known-only vs OOS-aware tuning | historical_known_oos_tradeoff.md、historical_oos_sota_tradeoff.md、OOS-first/均衡搜索 | 已完成 | 必须明确 OOS validation 是否参与 λ/threshold 选择；两种合同不能混排 |
| Known Recall / FA / F1-All / AUROC / AUPR | historical_gate_ablation_report.md、historical_oos_sota_tradeoff.md、historical_oos_visual_explanation | 已完成 | OOS F1 不能单独作为“整体更好”证据；同时报告 coverage/rejection 代价 |
| Trainable Gate 论文式四变体消融 | trainable_paper_four_variants_paper_geometry.md，36/36 CUDA 单元 | 已完成（H1 extension） | 当前 Trainable Gate 的四变体结果可用；必须标记为 H1 extension，不能写成严格 H0 |
| Frozen vs Trainable 直接消融 | historical_gate_ablation_report.md，54/54 CUDA 单元 | 已完成 | 在相同 Router/Expert 和 K=1 合同下，Trainable 的收益可以直接归因到 Gate 表示变化 |
| 延迟、吞吐、参数量、显存 | [`historical_deployment_benchmark`](historical_deployment_benchmark.md) 已在 3 个数据集、KIR=.50、seed42 对 Frozen/Trainable 的 Gate-only 与完整 Cascade 分别测量参数量、batch=1/32 延迟、吞吐和峰值 CUDA 显存 | 已完成（两种 H1 K=1 范围） | 完整四变体/CPU/更多 batch size 的部署结论仍不声明 |
| Related Work 事实核对 | fulltex.tex 仍保留旧稿表述 | 未完成 | 需要在稿件修订中修正 Hendrycks、DOC、MC dropout 和 multi-agent LLM 的描述 |

## MOGB 的正确写法

当前已有两层 MOGB 证据：

1. MOGB-Fair component：在统一 MiniLM、Known-only、固定数据划分和评估器下，比较粒球/分区/边界组件。该层可以说明 coverage–rejection 工作点和局部 ball 风险。
2. 官方 MOGB compatibility/reproduction：当前材料、旧环境和最终粒球状态没有完全恢复，不能把兼容运行结果写成完整官方方法的严格复现。

因此，论文应写成：

> We compare against a same-backbone MOGB-style component under the controlled MiniLM contract and separately report the status of the official MOGB reproduction. The component comparison is not presented as a byte-identical reproduction of the full BERT-based MOGB system.

2026-09-09 GPU preflight：当前 RTX 5070/CUDA 可用，`run_mogb_exact_reproduction.py --dry-run` 已通过 StackOverflow 数据快照审计。但已登记的 exact-compatible 单元已经完成；该 runner 仍通过现代兼容层运行，重新训练只会增加另一组非严格兼容结果，不能关闭原始环境、数据合同和官方代码逐字执行的差距。因此本轮不重复训练，并保留官方严格复现为未闭合状态。

## K 与多中心结论

当前证据不支持“多中心普遍有效”。可以支持的条件性结论是：

- BANKING77-OOS 在部分 KIR 和对角 Mahalanobis 设置下受益于多中心；
- CLINC150 的 K=2 只有小幅收益，继续增加 K 后下降；
- StackOverflow 的 K>1 稳定退化；
- KMeans 与随机分簇、碎片化、near-OOS 错误分解说明：新增中心会扩大接受区域，收益不等于更安全的 OOS 边界；
- Trainable 表示改善 K=1，不会自动使固定 K=2 更可靠。

## 调参协议必须分开

当前已经存在三种不同选择合同：

- 论文历史合同：允许使用 OOS validation 学习 λ；
- H1 均衡合同：验证集 OOS F1 选择，同时限制 Known/Accuracy 代价；
- H1 OOS-first 合同：只用 validation OOS F1 选择，Known 指标作为代价。

论文中不能把三者合成一条曲线，也不能把 OOS-aware 结果与完全 Known-only baseline 无标注比较。

## 指标修正

旧的 known_macro_f1 只在真实 Known 子集上计算，忽略 OOS 被误接受造成的 Known false positives。当前报告统一使用全测试集 Known 类 macro F1：

\[
\mathrm{KnownF1}
=
\frac{(C+1)\mathrm{F1\mbox{-}All}-\mathrm{OOSF1}}{C},
\]

其中 \(C\) 是 Known intent 数量。这个修正已经应用于：

- Frozen/Trainable Gate 消融；
- OOS-SOTA tradeoff；
- 论文设置 Trainable Gate extension；
- 当前四变体报告的补充指标。

## 论文 claim 收口

当前不应继续使用以下无条件表述：

- “我们的 multi-cluster boundary 普遍优于现有方法”；
- “Trainable 表示使多中心边界更有效”；
- “严格超过完整 MOGB/DCLOOS”；
- “lightweight models have outstanding deployment advantages”；
- “threshold strictly separates Known and OOS distributions”。

更稳妥的贡献表述是：

1. 轻量级显式 OOS Gate 与 Router/Expert 的级联解耦；
2. Known-only Trainable MiniLM 表示适配对 OOS score ordering 的改善；
3. 多中心边界的适用条件、接受区域过覆盖风险和 Known/OOS 权衡；
4. 在同一 H1 合同下的可复查 Gate 表示消融。

## Deployment benchmark

当前 Frozen/Trainable K=1 的 Gate-only 与 full Cascade CUDA benchmark 已闭合，机器可读表为 `results/analysis/historical_deployment_benchmark/summary.csv`。Frozen Gate 为 22.71M 总参数、0 个训练参数；Trainable Gate 为 22.91M 总参数、3.75M 个训练参数。该表分别记录 CLINC150、StackOverflow、BANKING77-OOS 的 batch=1/32 延迟、吞吐和峰值显存；测量口径与限制见 [benchmark 报告](historical_deployment_benchmark.md)。

这组结果足以支持“Trainable Gate 是部分参数更新的 Gate，并且在当前设备上具有可测量的推理开销”，不能外推为完整四变体或 CPU 部署 SOTA。

## 尚需补的稿件工作

当前实验资产已经覆盖主要证据缺口；仍需在论文源稿中完成：

- 将 MOGB 放入主对比或明确放入同协议 component supplement；
- 将 KIR/K 消融、Known-only/OOS-aware tuning 和完整 Known F1 口径写入实验部分；
- 修正 Related Work 中 Hendrycks、DOC、MC dropout 和 multi-agent LLM 的事实描述；
- 如果保留更广 deployment claim，仍需扩展到完整四变体/CPU/更多 batch size；当前 Frozen/Trainable K=1 的 Gate-only 与 full Cascade benchmark 已完成；
- 将所有 SOTA 表述绑定到数据键、KIR、backbone、监督和 evaluator 合同。

Related Work 的可直接替换文案已整理在 [RELATED_WORK_CLAIM_CORRECTIONS_V1.md](RELATED_WORK_CLAIM_CORRECTIONS_V1.md)。

按当前 `fulltex.tex` 的实际内容，已经生成了带基准 hash、原句、替换句和应用检查项的 [精确稿件修订补丁](FULLTEX_REVIEW_REVISION_PATCH_V1.md)。
对应的 unified diff 已通过 `patch --dry-run`，见 [FULLTEX_REVIEW_REVISION_V1.patch](FULLTEX_REVIEW_REVISION_V1.patch)；实际应用仍留给下一次投稿稿件修订。

fulltex.tex 当前保持不修改；本文件用于指导下一次论文稿件修订。

## 可直接用于 rebuttal 的证据化回应草稿

下面的文字只使用当前已经登记的证据，不把 H1 extension 写成严格 H0，也不把 MOGB compatibility 结果写成官方复现。

### Response to the novelty concern

> We agree that multi-center boundary construction alone is not a sufficient novelty claim. We therefore narrow the contribution statement. The revised claim is a lightweight explicit OOS Gate coupled with a decoupled Router–Expert cascade, together with a Known-only Trainable MiniLM representation analysis. The experiments also show a limitation of the original multi-center assumption: fixed K>1 is not uniformly beneficial, and can enlarge the acceptance region for near-OOS samples. The contribution is thus not “the first multi-center boundary,” but the controlled characterization of when local geometry helps and when representation adaptation with a single-center Gate is safer.

证据：[`historical_gate_ablation_report.md`](historical_gate_ablation_report.md)、[`UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`](UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md)。

### Response to the missing MOGB comparison

> We added a same-backbone MOGB-style component comparison under the controlled MiniLM contract and report it separately from the official BERT-based MOGB reproduction. The component comparison is not presented as a byte-identical reproduction of the full MOGB system. The pinned official checkout was also audited in an isolated compatibility layer; because the original environment, data snapshot and complete final ball state are not recoverable, the official run remains labeled non-strict and is not used for an unconditional SOTA claim.

证据：[`cross_method_oos_mechanism_presentation.md`](cross_method_oos_mechanism_presentation.md)、[`historical_deployment_benchmark.md`](historical_deployment_benchmark.md)、[`MOGB reproduction gap analysis`](../../docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md)。

### Response to the K and adaptive-K concern

> We added a controlled K=1,...,5 study across three datasets, 11 KIR values, five seeds and two distance functions, together with adaptive-center and random-partition diagnostics. The results do not support a universal K=2 claim: Banking77-OOS benefits from additional centers in several settings, CLINC150 shows only limited K=2 gains, and StackOverflow degrades for K>1. We therefore report K as a dataset- and protocol-dependent design choice rather than a universally superior mechanism.

### Response to the unknown-aware tuning concern

> We separate three contracts in the revised experimental description: the historical paper contract, which uses OOS validation to learn the radius coefficient; the H1 balanced contract, which constrains the Known/Accuracy cost; and the H1 OOS-first contract, which explicitly prioritizes validation OOS F1. These results are not pooled into one ranking. Test OOS labels are not used for checkpoint, threshold or K selection in the new H1 analyses.

证据：[`historical_known_oos_tradeoff.md`](historical_known_oos_tradeoff.md)、[`historical_oos_sota_tradeoff.md`](historical_oos_sota_tradeoff.md)。

### Response to the OOS–Known trade-off concern

> We corrected the Known F1 calculation to use the full test label set, including the precision cost of accepting OOS as Known. We now report OOS F1 together with Known F1, Known Recall, False Acceptance, F1-All and Accuracy; AUROC/AUPR are retained for the Gate-level fair matrix. The revised text does not claim all-metric SOTA. For example, the current StackOverflow KIR=.25 workpoint reaches 95.38 OOS F1 with 80.07 Known F1 and 91.58 Accuracy, whereas the Banking77-OOS OOS-first workpoint reaches higher OOS F1 only with a substantially larger Known-side cost.

### Response to the deployment and claim-strength concerns

> We added a CUDA benchmark covering Gate-only inference and the matching full Cascade for Frozen/Trainable K=1 on three datasets. It records total/trainable parameters, batch-1 and batch-32 latency, derived throughput and peak CUDA memory. We restrict the claim accordingly: these measurements quantify the tested H1 K=1 paths on one device; they do not establish four-variant or CPU deployment superiority.

证据：[`historical_deployment_benchmark.md`](historical_deployment_benchmark.md)。Related Work 的事实修正文案见 [`RELATED_WORK_CLAIM_CORRECTIONS_V1.md`](RELATED_WORK_CLAIM_CORRECTIONS_V1.md)；应用到 `fulltex.tex` 仍属于下一次稿件修订，当前仓库明确保持源稿不变。

## 最终状态审计（2026-09-09）

| 审稿要求 | 当前可证明状态 | 权威证据 | 尚未完成的部分 |
|---|---|---|---|
| MOGB 直接比较 | 证据部分完成 | MiniLM fair component、official MOGB status table、compatibility audit | 原作者旧环境/完整 artifact 不可恢复，因此不能证明严格官方复现 |
| K=1…5 与 adaptive K | Gate 证据完成 | 1650 个 Gate 单元、adaptive-center 和 random/KMeans 诊断 | 论文正文尚未应用修订 patch |
| Known-only vs OOS-aware tuning | 证据完成 | Known/OOS trade-off 与 OOS-SOTA reports | 论文正文尚未应用修订 patch |
| Known/OOS 多指标 | 证据完成 | corrected Known F1、Known Recall、FA、F1-All、AUROC/AUPR 表 | 论文正文尚未应用修订 patch |
| Trainable Gate 消融 | H1 证据完成 | 54 个 CUDA Frozen/Trainable cells、36 个 paper-geometry variant cells | 不能标成严格 H0 |
| deployment claim | H1 K=1 Gate/Cascade 范围完成 | 6 个 CUDA rows、双调用入口：参数、latency、throughput、显存 | 完整四变体/CPU/更多 batch size 未测量 |
| Related Work 与过强 claim | 修订材料完成 | `RELATED_WORK_CLAIM_CORRECTIONS_V1.md`、`FULLTEX_REVIEW_REVISION_V1.patch` | patch 尚未应用到投稿源稿 |

因此，当前仓库已经完成“审稿意见对应的实验补充、证据整理和可执行稿件修订材料”，但不能声称投稿 PDF 已经修订，也不能声称完整官方 MOGB 已严格复现。

## ARR 十三项扩展对账

| ARR 关注点 | 当前判断 | 证据或缺口 |
|---|---|---|
| 1. Li/MOGB 创新重叠 | 部分完成 | MOGB-Fair、official status table 和 novelty patch 已有；严格官方 MOGB 仍缺原始环境与完整 artifact |
| 2. multi-cluster 是否真的有效 | 已完成 Gate 证据 | K=1…5、random/KMeans、adaptive-center 和 acceptance-risk 分析；结论为条件性收益 |
| 3. λ 的 OOS validation 使用 | 已完成口径审计 | 已分开 historical、H1 balanced、H1 OOS-first 三种合同；正文仍需应用 patch |
| 4. 随机种子 | 已完成主要实验 | 关键 H1/Frozen/Trainable/MOGB-Fair 矩阵含 3 或 5 seed；论文表格仍需统一 mean±std |
| 5. baseline 公平性 | 部分完成 | MOGB、ADB、DA-ADB、DCLOOS 状态和监督条件已登记；KNNCL/OpenMax/DeepUnk 部分路线仍无同合同 final metrics |
| 6. 高 OOS F1 与低 Known F1 | 已完成机制证据 | corrected Known F1、Known Recall、FA/FR、F1-All、AUROC/AUPR、KIR 错误分解已生成 |
| 7. Method/Gate 定义清晰度 | 修订材料完成 | `FULLTEX_REVIEW_REVISION_V1.patch` 明确 Known-only 数据、intent-wise clustering 和 Gate→Router→Expert 数据流 |
| 8. `mean+λσ` 与距离 heuristic | 部分完成 | Euclidean/diagonal Mahalanobis、radius/geometry 搜索已有；严格统计理论推导未新增 |
| 9. MiniLM encoder 控制 | 部分完成 | Frozen、Trainable、LoRA、SmolLM 和 fair detector 对照已有；尚未形成完整 sentence-encoder factorial 表 |
| 10. deployment claim | H1 K=1 完成 | Gate-only 与 full Cascade 已分别测量参数、latency、throughput、显存；完整四变体/CPU未测 |
| 11. incremental intent | 未完成 | 当前没有新增 intent append/update 实验；只能在稿件中暂不声称该部署优势 |
| 12. hard OOS/failure cases | 部分完成 | near-OOS、误接收、误拒绝和 subtype residual-risk 已有；尚未新增 synthetic/adversarial OOS 训练或测试集 |
| 13. 全文重写 | 修订材料完成 | Related Work、Introduction、Method、Results、Conclusion 的 patch 已通过临时副本检查；尚未应用投稿源稿 |
