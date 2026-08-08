# 实验机制分析包 V3（现有结果复核）

> 本报告只读取已经提交到 `s2c/results/analysis/` 的轻量 CSV；不读取 checkpoint、embedding 或原始文本，也不重新训练。当前工作区的 `../artifacts/s2c/runs/` 已不在磁盘，因此报告同时标记重产物缺失风险。

## 1. 数据与证据边界

- 输入：`results/analysis/minilm_trainable_5seed_fair_v1/all_methods_per_seed.csv`，SHA256=`c981f5f887c49ed2360d51a6ad0bfb061fb60a79b42f95fda13ca4485cfb9a3d`。
- 主表：315 行（3 数据集 × 3 KIR × 5 seed × 7 方法）；Trainable K=1、Frozen 单/双中心、随机分簇和三种 MOGB 组件。
- Trainable native 归因轻量表：36 行（仅 KIR=.50、3 seed、四种 detector）；不与主表混合。
- 所有主表差值按 dataset×KIR×seed 配对；Bootstrap RNG=20260806、10,000 次。
- `fulltex.tex`、ADB/DA-ADB、DCLOOS 只作为外部/历史层参照，不参与同协议配对排名。

## 2. 当前最稳定的实验事实

| 数据集 | KIR | Trainable K=1 OOS F1 | Frozen 单中心 OOS F1 | 差值 | Trainable false acceptance | Trainable Known Recall |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 0.25 | 91.65±1.80 | 85.17±2.53 | +6.48 pp | 10.33% | 81.84% |
| banking77 | 0.50 | 83.56±1.83 | 72.43±2.77 | +11.14 pp | 15.74% | 82.21% |
| banking77 | 0.75 | 68.65±2.71 | 53.66±4.17 | +14.99 pp | 19.61% | 82.41% |
| clinc150 | 0.25 | 95.16±0.65 | 94.21±0.74 | +0.96 pp | 3.24% | 73.63% |
| clinc150 | 0.50 | 90.44±0.44 | 88.94±0.50 | +1.50 pp | 3.69% | 74.44% |
| clinc150 | 0.75 | 82.85±0.51 | 81.75±0.80 | +1.10 pp | 4.02% | 75.12% |
| stackoverflow | 0.25 | 95.48±1.21 | 89.48±3.77 | +6.00 pp | 3.85% | 84.29% |
| stackoverflow | 0.50 | 87.67±1.66 | 76.55±5.34 | +11.12 pp | 9.34% | 83.89% |
| stackoverflow | 0.75 | 76.30±2.91 | 65.61±5.34 | +10.70 pp | 8.80% | 84.12% |

## 3. 机制解释

### 3.1 Trainable 的主要收益在 K=1 分数排序

Trainable K=1 在三个数据集和多数 KIR 上提高 OOS F1，尤其是 StackOverflow 的 KIR=.50/.75；同时 false acceptance 通常下降。这个现象在同一 seed 的配对差值中保持，说明收益不是单个 seed 的偶然峰值。它支持“最后两层 MiniLM+投影改变 Known/OOS 分数排序”的解释，而不是“增加中心数量带来收益”。

### 3.2 多中心不是表示训练的自动副产物

StackOverflow 的 fixed K=2 仍然明显差于 Trainable K=1；MOGB 分区组件通常进一步牺牲 Known Recall 换取低 false acceptance。E2/E3/RACAL 的逐样本诊断已经显示稳定聚类仍会新增大量 OOS 误接受，因此当前瓶颈是接受区域的组合语义，而非 KMeans 是否收敛。

### 3.3 数据集差异是真实的

Banking77 在某些 KIR 下固定多中心或 MOGB 分区能提高 OOS F1，但往往伴随 Known Recall/F1-All 下降；CLINC150 的收益小且不稳定；StackOverflow 在高 KIR 下多中心风险最大。下一步不能用一个跨数据集的“最优 K”解释这些现象。

## 4. 可视化解读

本节说明图表支持的机制结论，而不只是列出 PNG 路径。所有图均来自同一份 315 行轻量结果，未重新训练，因此不能被解释为新的实验单元。

### 4.1 KIR×方法热图：Trainable K=1 是最稳定的自有工作点

图：[OOS F1 方法×KIR 热图](../../figures/experimental_mechanism_pack_v3/oos_f1_method_kir_heatmap.png)

- CLINC150 中 Trainable K=1 在三个 KIR 都处于最高或接近最高位置，说明表示适配后的单中心分数排序较稳定；固定 K=2 只有很小收益。
- Banking77 中 MOGB split + ours boundary 在 KIR=.25 略高于 Trainable，但在 KIR=.50/.75 Trainable 更高，说明多中心收益是条件性的。
- StackOverflow 中 Trainable K=1 在三个 KIR 都明显高于固定 K=2 和 MOGB 组件，且 KIR 越高差距越大，支持“多球接受区域过覆盖”而非“训练未收敛”的解释。

该图支持“当前最稳的自有 Gate 是 Trainable K=1”，不支持“已经在所有协议上达到 SOTA”。

### 4.2 覆盖—拒识散点图：MOGB 更保守，Trainable 更平衡

图：[KIR=.50 覆盖—拒识权衡](../../figures/experimental_mechanism_pack_v3/kir050_coverage_oos_tradeoff.png)

横轴是 Known Recall，纵轴是 OOS F1，点大小表示 false acceptance。

- MOGB MiniLM 和 MOGB split + ours boundary 位于较低 Known Recall 区域，false acceptance 较低，但拒绝了大量 Known 样本。
- Trainable K=1 位于较高 Known Recall、高 OOS F1 区域，优势不是简单地“拒绝更多样本”，而是覆盖—拒识折中更好。
- StackOverflow 的固定 K=2 位于高 Known Recall、低 OOS F1 区域，且点较大，直接显示“多中心提高覆盖，却新增大量 OOS 误接受”。

因此主表必须同时报告 OOS F1、Known Recall 和 false acceptance，不能只看单一 F1。

### 4.3 历史 fulltex 对照图：当前结果并非在所有数据集和 KIR 都更差

图：[历史 fulltex 与当前 Trainable 对照](../../figures/minilm_trainable_5seed_fair_v1/trainable_vs_fulltex_reference.png)

这张图只是描述性参照，因为两条曲线来自不同协议。

- CLINC150：低 KIR 接近历史值，高 KIR 差距扩大。
- Banking77：当前 Trainable 随 KIR 增大下降更快，提示数据/split、λ 校准和完整 Cascade 均可能影响历史差距。
- StackOverflow：KIR=.25 和 .75 的当前值接近或略高于历史参考，KIR=.50 略低，说明“Trainable 在所有情况下都比论文差”并不成立。

它的主要用途是提醒：历史 fulltex 数字不能直接作为当前 Gate-only 训练结果的统一上限。

### 4.4 Trainable−Frozen 差值热图：训练改善的是 K=1 分数分离

图：[Trainable 相对 Frozen 的 KIR 差值](../../figures/minilm_trainable_kir_sweep_v1/trainable_minus_frozen_kir_heatmap.png)

- CLINC150 和 StackOverflow 的差值在三个 KIR 均为正，StackOverflow 在 KIR=.50/.75 的提升尤其明显。
- Banking77 在 KIR=.75 出现明显负差值，说明高 Known 密度下 Known-only 表示适配不保证 OOS 边界稳定。
- 该图只比较 K=1，不能证明训练解决了固定 K>1；已有 K=2 结果仍显示 StackOverflow false acceptance 爆炸。

综合图表得到的机制链条是：

```text
Known-only MiniLM 适配
        ↓
K=1 的 Known/OOS 分数分离改善
        ↓
更好的覆盖—拒识工作点
        但不等于
固定多中心接受区域安全
```

## 5. Pareto 工作点

Pareto 标记同时考虑 OOS F1、F1-All、Known Recall（越高越好）和 false acceptance（越低越好）。它不是 SOTA 排名，而是展示为什么需要同时报告覆盖和拒识。每个 dataset/KIR 的候选方法见 `pareto_flags.csv`。

## 6. 与外部方法的边界

- ADB/DA-ADB：已有单 seed、BERT/兼容环境数字；不是当前 protocol_v2 五 seed 同合同结果。
- MOGB：作者 BERT 单格严格复现未达到论文数字；Frozen MiniLM MOGB 行是公平组件对照，不是官方 MOGB。
- DCLOOS：使用伪 OOS/外部 OOS 监督；reduced-budget 结果不能与 Known-only Trainable 直接排名。
- 因此当前最稳妥的“自有方法胜出”表述是：在相同当前协议的 Gate-only、Known-only 条件下，Trainable K=1 通常比 Frozen/MOGB 组件有更好的覆盖—拒识折中；尚不能声称超过完整 MOGB 或端到端 DCLOOS。

## 7. 当前实验瓶颈与下一步

1. 现有轻量结果已经足以支持 KIR/K/表示/组件的机制分析；不应继续重复相同矩阵。
2. 当前工作区缺少原始 run/checkpoint，因此要重新运行新实验，必须先恢复并核对 artifacts provenance，或重新登记一套最小可复现实验。
3. 恢复产物后，最高价值的下一步是同一 Known-only 工作点下复算 Trainable/Frozen Gate、native detectors 和 MOGB 组件；不要再用 test-oracle threshold 做正式选择。
4. 之后再决定是否把最稳定的 Gate 接入 Cascade；外部基线必须单独标注监督条件。

## 8. 输出与复核

- `paired_effects.csv`：Trainable 相对每个冻结组件的 paired bootstrap 差值。
- `method_summary.csv`：dataset×KIR×method 的均值、标准差。
- `pareto_flags.csv`：覆盖—拒识多目标 Pareto 标记。
- `figures/experimental_mechanism_pack_v3/`：热图、KIR 曲线、工作点散点和差值热图。

## 9. 结论

当前最可信的结果不是“固定多中心达到了 SOTA”，而是：Trainable MiniLM K=1 在当前统一协议下是最稳定的自有 Gate 候选；它的收益来自表示适配后的分数分离和更好的覆盖—拒识平衡。固定 K>1、MOGB 组件和训练参与式自适应 split 尚未在 StackOverflow 上形成安全正收益。
