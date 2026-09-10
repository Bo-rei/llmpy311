# 机制闭环 V1：当前自有方法为何优于 MOGB 组件、又不能冒充跨合同 SOTA

更新时间：2026-08-10  
活动协议：`protocol_v2_textoir_v1`

本报告是分析阶段的统一收口，不是新的训练实验。它只读取已经冻结的五 seed fair matrix、MOGB 逐样本转移摘要、粒球风险摘要和 KIR 敏感性摘要；没有重新训练、重新选 K、重新调阈值或修改历史 artifact。

## 1. 先说清楚当前比较的是哪个方法

当前“我的方法”默认指 **`S2C-Trainable-K1`**：Known-only 训练 MiniLM 的最后两层和残差投影，随后使用单中心 Gate（K=1）。它不是 `fulltex.tex` 中的完整 Gate–Router–Expert Cascade，也不是 RC-AMBL 或尚未完成的自适应 K 方法。

当前可直接排名的对象是同一 `protocol_v2_textoir_v1`、同一 Known registry、同一 Gate evaluator 下的七行 fair matrix：

1. S2C Trainable K=1；
2. S2C Frozen K=1；
3. S2C Frozen K=2；
4. S2C Random K=2；
5. MOGB partition + S2C boundary；
6. S2C partition + MOGB boundary；
7. MOGB-MiniLM。

ADB、DA-ADB、DCLOOS 和历史 `fulltex.tex` Cascade 单独报告，不能直接拼入这七行的同协议排序。

## 2. 当前 fair matrix 的结果

来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`，3 数据集 × 3 KIR × 5 seed，共 63 个汇总行。

| 方法 | OOS F1 九格均值 | F1-All 九格均值 | 解释 |
|---|---:|---:|---|
| **S2C Trainable K=1** | **85.75%** | **83.36%** | 当前自有候选最平衡；OOS F1 为 9 格中 8 格第一，F1-All 为 9 格中 9 格第一 |
| S2C Frozen K=1 | 78.64% | 78.30% | 冻结表示单中心基线 |
| S2C Random K=2 | 78.39% | 78.51% | 增加中心但不学习子结构 |
| S2C Frozen K=2 | 75.51% | 76.79% | 在 StackOverflow 出现明显 acceptance-union 风险 |
| MOGB partition + S2C boundary | 77.74% | 63.92% | 误接受较低，但 Known 拒绝较多 |
| S2C partition + MOGB boundary | 75.90% | 62.30% | MOGB 平均半径/欧氏边界工作点偏保守 |
| MOGB-MiniLM | 73.39% | 46.26% | 低误接受主要以大量拒绝 Known 为代价 |

这证明的是：在当前统一 **MiniLM + Gate** 合同中，Trainable-K1 是最好的自有候选；不证明它已经超过完整官方 MOGB 或所有外部端到端方法。

已有 `statistical_stability_v1` 进一步以同一 `dataset × KIR × seed` 为配对单位做了 10,000 次
paired bootstrap：Trainable-K1 相对 Frozen-K1、Frozen-K2 和 Random-K2 在 9/9 个 dataset×KIR
组合上 OOS F1 全 seed 胜出；相对三个 MOGB 组件比较则有 8/9 个组合的 95% CI 稳定为正。
这支持“当前 fair matrix 的优势不是单个 seed 偶然”，但仍只适用于当前 Gate 合同。

## 3. 为什么 Trainable-K1 看起来更好

### 3.1 主要收益是 Known 覆盖与分数分离的平衡

在同一五 seed、同一 registry 的 Trainable-K1 与 MOGB-Fair 逐样本转移中，Trainable 的 F1-All 优势主要来自恢复被 MOGB 的平均半径球拒绝的 Known 样本，而不是简单把更多 OOS 拒绝掉：

| 数据集 | Trainable 相对 MOGB 的平均 Known 正确数增量比例 | 平均 OOS 正确数变化比例 | F1-All 增量 |
|---|---:|---:|---:|
| CLINC150 | +36.30% 至 +45.04% | -2.58% 至 -3.19% | +27.45 至 +41.12 pp |
| Banking77 | +41.16% 至 +50.91% | -8.83% 至 -18.84% | +22.63 至 +41.74 pp |
| StackOverflow | +51.19% 至 +61.23% | -2.91% 至 -8.55% | +33.57 至 +54.19 pp |

因此当前方法的准确表述是：**它在 Known-only 条件下改善了覆盖—拒识工作点**。不能写成“Trainable 对 OOS 的绝对拒识能力在所有协议都更强”。

### 3.2 StackOverflow 的固定 K=2 为什么失败

StackOverflow/KIR=.50/5 seed：

| 方法 | OOS F1 | F1-All | Known Recall | False acceptance | False rejection |
|---|---:|---:|---:|---:|---:|
| S2C Trainable K=1 | 87.67% | 86.55% | 83.89% | 9.34% | 16.11% |
| S2C Frozen K=1 | 76.55% | 79.98% | 87.15% | 29.71% | 12.85% |
| S2C Frozen K=2 | 63.53% | 72.76% | 86.89% | 47.17% | 13.11% |
| MOGB-MiniLM | 72.92% | 43.30% | 27.09% | 0.79% | 72.91% |

固定双中心没有显著恢复 Known Recall，却把任意一个局部球接受即可的并集区域扩大，导致 OOS false acceptance 从 29.71% 增至 47.17%。MOGB-MiniLM 反方向工作：其 false acceptance 很低，但 Known false rejection 达 72.91%，所以 F1-All 很低。Trainable-K1 的优势是把 score separation 和单边界覆盖放在相对平衡的工作点，而不是中心数量本身。

## 4. MOGB 本地结果为什么远低于论文数字

当前 MOGB BERT 单格是“公开代码逻辑的现代兼容运行”，不是作者旧环境、数据快照和最终随机粒球状态的逐字节重现。

StackOverflow/KIR=.50/seed=0 的本地兼容结果：`Acc=75.17%`、`F1-All=68.35%`、`F1-U=79.97%`、`F1-K=67.19%`、Known Recall=`51.53%`；论文公开参考为 `88.67/87.49/89.71/87.27%`。现有代码审计和 corrected-loss/radius 诊断支持以下分层解释：

1. 官方子中心损失对非负距离做 L1 归一化，压缩类别 logit 动态范围，训练信号接近均匀分类；
2. 选择粒球的随机状态和作者数据/Known 列表没有完整恢复，导致不能宣称严格同球重放；
3. 默认平均半径偏窄，本地 Known Recall 很低；仅扩大半径虽能恢复 Known，却很快增加 OOS 误接受，仍不能回到论文工作点；
4. 因此差距不是单一 lambda、单一半径或“StackOverflow 天生不适合 MOGB”可以解释的。

这也解释了为什么 **MOGB-MiniLM** 不能作为完整 MOGB 的替代：它只保留粒球/边界组件，去掉了 MOGB 的 BERT 与最近子中心交替训练。

## 5. 外部基线的合同边界

- 历史 `fulltex.tex` 的 `Ours` 是完整 Gate–Router–Expert Cascade；旧合同下它在 9/9 个数据集×KIR 单元超过表内基线，但不是当前 Trainable-K1。
- ADB 有 StackOverflow/KIR=.50/3 seed 的 BERT/TextOIR 兼容参照：OOS F1=`87.36±1.61%`、F1-All=`85.66±1.59%`；同 seed Trainable-K1 为 `88.21±1.72%`、`86.69±1.45%`，但两者骨干和训练合同不同。
- DCLOOS 已恢复作者 README 所指外部 SQuAD 负样本；默认预算单元超时，reduced 单元为 KIR=.75/seed=888、BERT、pseudo-OOS+外部 OOS，OOS F1=`87.05%`、F1-All=`90.26%`。它不是当前 Known-only fair 主表，不能据此宣称 S2C 超过 DCLOOS。
- DA-ADB 当前仍没有通过语义有效性审计的结果，NaN/全类预测单元不进入排名。

## 6. 当前真正完成的证据

本轮新增机制闭环包：

- `results/analysis/archive/analysis/mechanism_closure_v1/method_kir_performance.csv`（63 行）
- `results/analysis/archive/analysis/mechanism_closure_v1/trainable_mogb_transition_summary.csv`（9 行）
- `results/analysis/archive/analysis/mechanism_closure_v1/mogb_risk_summary.csv`（18 行）
- `results/analysis/archive/analysis/mechanism_closure_v1/kir_robustness.csv`（21 行）
- `results/analysis/archive/analysis/mechanism_closure_v1/MANIFEST.json`
- `figures/archive/analysis/mechanism_closure_v1/mechanism_closure_dashboard.png`
- `figures/archive/analysis/mechanism_closure_v1/performance_operating_curves.png`
- `figures/archive/analysis/mechanism_closure_v1/trainable_mogb_error_budget.png`
- `figures/archive/analysis/mechanism_closure_v1/mogb_risk_workpoints.png`

这些是后验分析图，不是新模型结果；所有来源哈希和“不得用于测试选择”的说明都在 manifest 中。

统计入口：[`STATISTICAL_STABILITY_V1.md`](STATISTICAL_STABILITY_V1.md)，其 paired bootstrap 结果位于
`results/analysis/archive/analysis/statistical_stability_v1/`，不与本报告重新生成或混合统计口径。

## 7. 当前未完成、因此不能下的结论

目前仍不能写：

- 当前 Trainable-K1 已超过完整官方 MOGB；
- 当前 Trainable-K1 已超过 DCLOOS；
- 当前方法已经达到跨协议 SOTA；
- 自适应多中心已经被证明有效。

仍缺少：完整官方合同下的 MOGB 多 seed、有效 DA-ADB、DCLOOS 默认预算/同 KIR 同 seed 的可比单元，以及当前协议下重新运行完整 Cascade。下一步应先补齐可验证外部单格或继续做逐样本错误/工作点分析；不要重复 E2/E3 和旧 K 扫描。

## 8. 复现入口

```bash
python tools/analysis/build_mechanism_closure_v1.py
```

可视化总入口：[`VISUAL_ANALYSIS_INDEX_V1.md`](VISUAL_ANALYSIS_INDEX_V1.md)；当前方法和协议总览：[`EXPERIMENT_COMPARISON_OVERVIEW_V2.md`](EXPERIMENT_COMPARISON_OVERVIEW_V2.md)。
