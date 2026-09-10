# 三数据集逐意图风险可视化（v1）

## 1. 目的与边界

本阶段只消费已经完成的 `cross_dataset_error_attribution_v1` 汇总，不重新训练、不改变阈值、不选择中心数，也不读取原始文本。目标是把总体 false acceptance/false rejection 拆到具体 Known intent，回答：

1. 哪些 intent 在固定 K=2 下吸收最多 OOS；
2. MOGB 的低 OOS 误接收是否伴随逐意图 Known 过拒绝；
3. 三个数据集的风险是否具有相同结构；
4. KIR 增大时，错误是否集中到少数高风险 intent。

输入为 `results/analysis/archive/analysis/cross_dataset_error_attribution_v1/intent_error_attribution_summary.csv`，共 4,641 行，覆盖三数据集、三 KIR 和七种已有方法。正式判定仍为 `score <= 1` 接受 Known。

## 2. 新增产物

- `tools/analysis/build_cross_dataset_intent_risk_visuals_v1.py`
- `results/analysis/archive/analysis/cross_dataset_intent_risk_visuals_v1/top_oos_acceptor_intents.csv`
- `results/analysis/archive/analysis/cross_dataset_intent_risk_visuals_v1/oos_acceptance_concentration.csv`
- `results/analysis/archive/analysis/cross_dataset_intent_risk_visuals_v1/intent_error_deltas_vs_trainable.csv`
- `results/analysis/archive/analysis/cross_dataset_intent_risk_visuals_v1/MANIFEST.json`
- `figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/intent_risk_scatter_kir050.png`
- `figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/oos_acceptance_concentration_kir050.png`
- `figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/intent_error_delta_scatter_kir050.png`
- `figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/fixed_k2_oos_acceptor_rank_curve.png`

## 3. 主要可视化结论

### 3.1 StackOverflow 的固定 K=2 错误高度集中

在 KIR=0.50 下，固定 K=2 的 OOS false acceptance 中，前五个 intent（`cocoa`、`sharepoint`、`osx`、`spring`、`scala`）合计约占 **64.9%**。这不是均匀的小幅退化，而是少数 Known 边界成为 OOS 吸收器。固定 K=2 的前十个 intent 已输出在 `top_oos_acceptor_intents.csv`。

对应的累计排名曲线显示，StackOverflow 比 CLINC150、Banking77 更早达到高累计错误占比；KIR 从 0.25 提升到 0.75 时仍保持明显的集中结构。这与逐样本分析中固定 K=2 的 acceptance-union 过覆盖一致。

### 3.2 MOGB 的低误接收来自更保守的逐意图决策

KIR=0.50 时，前五个 OOS 误接收 intent 的占比为：

| 数据集 | Trainable K=1 | Frozen K=2 | MOGB-MiniLM | MOGB partition + ours |
|---|---:|---:|---:|---:|
| CLINC150 | 33.7% | 27.6% | 57.0% | 41.5% |
| Banking77 | 31.4% | 22.5% | 51.7% | 55.7% |
| StackOverflow | 81.1% | 64.9% | 83.7% | 85.0% |

MOGB 组件的误接收总量较低，因此其错误更容易集中在少数仍被接受的 intent；但这不能单独解释为更好的检测，因为其 Known false rejection 同时大幅上升。

### 3.3 逐意图 trade-off 不是同一方向

相对 Trainable K=1，在 KIR=0.50：

- Frozen K=2 的逐意图平均 OOS false acceptance 增量约 **+0.49 个百分点**，但平均 Known false rejection 反而约低 **0.95 个百分点**，说明它主要是以更宽的开放区域换取覆盖；
- MOGB-MiniLM 的 OOS false acceptance 平均约低 **0.22 个百分点**，但 Known false rejection 平均高 **45.98 个百分点**；
- MOGB partition + ours 的 OOS false acceptance 平均约低 **0.19 个百分点**，但 Known false rejection 平均高 **25.10 个百分点**。

因此 MOGB 的优势不能只按 false acceptance 判断，而应同时查看 Known coverage、F1-All 和逐意图拒识率。

### 3.4 数据集差异

- **CLINC150**：intent-level OOS 误接收整体较低，固定 K=2 相对 Trainable 的差距有限；多中心风险更像局部而非全局崩溃。
- **Banking77**：风险分布更分散，部分 intent 可从局部结构受益，但 MOGB 的保守性代价很明显。
- **StackOverflow**：固定 K=2 的 OOS 误接收集中在少数高风险 intent，且这些 intent 的错误随 KIR 增大更容易主导整体 OOS F1。

## 4. 与已有结论的关系

这批图没有改变已有实验结论，而是补充了机制层证据：

1. Trainable MiniLM K=1 的改进不是某一个 intent 的偶然现象；
2. StackOverflow 的固定多中心失败可以定位到具体 OOS 吸收 intent；
3. MOGB 组件更像保守拒识工作点，不是无代价的全面改进；
4. “多中心是否有效”必须同时考虑数据集、KIR 和 intent，而不能只看一个全局均值。

## 5. 限制

- 这是对已有测试预测的后验归因，不能把 `top_oos_acceptor_intents` 用作正式选择器；
- 没有新增 ADB、DA-ADB 或 DCLOOS 结果，外部基线仍受 runtime/监督合同限制；
- 不能据此宣称已经超过 MOGB 或达到 SOTA；
- 逐意图的 OOS 误接收贡献是误差诊断，不是新的训练目标。

## 6. 复现

```bash
python tools/analysis/build_cross_dataset_intent_risk_visuals_v1.py
```

