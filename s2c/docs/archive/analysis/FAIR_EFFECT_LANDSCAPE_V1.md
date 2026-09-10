# Fair Effect Landscape V1：全指标同协议效果景观

## 1. 目的与边界

本阶段是对已经完成的 `protocol_v2_textoir_v1` 五 seed fair matrix 做的
analysis-only 汇总，不启动训练、不重算边界、不选择阈值，也不修改 E2、E3、R1
或历史结果。目标是把“Trainable K=1 的收益来自哪里、固定多中心和 MOGB 组件付出了什么代价”
放在同一个多指标坐标系中，而不是只看 OOS F1 排名。

输入为：

```text
results/analysis/archive/analysis/statistical_stability_v1/paired_effects.csv
```

该源表有 486 行配对效应。本阶段预先登记并选取 3 个数据集 × 3 个 KIR × 6 个比较方法 ×
8 个指标，共 432 行；未纳入源表中的其它指标不会被伪装为已分析。统计量沿用
`statistical_stability_v1` 的 10,000 次 paired bootstrap、固定 RNG 和 95% CI。

输出：

```text
results/analysis/archive/analysis/fair_effect_landscape_v1/effect_landscape.csv
results/analysis/archive/analysis/fair_effect_landscape_v1/effect_landscape_summary.csv
results/analysis/archive/analysis/fair_effect_landscape_v1/MANIFEST.json
figures/archive/analysis/fair_effect_landscape_v1/
```

## 2. 比较合同与符号

每个比较都以同一 `dataset × KIR × seed` 为配对单位，Trainable MiniLM K=1 为参照。

比较方法包括 Frozen K=1、Frozen K=2、Random K=2、MOGB MiniLM、MOGB partition + ours
boundary、ours partition + MOGB boundary。正的 `effect_pp` 表示 Trainable 更好；对于
false acceptance 和 false rejection，错误率已转换为“Trainable 风险降低量”，所以正值仍表示
Trainable 更安全。该约定只用于解释，不改变源结果。

## 3. 主要发现

### 3.1 OOS F1 的跨数据集稳定性

Trainable K=1 相对于 Frozen K=1、Frozen K=2 和 Random K=2 的 OOS F1 平均优势分别为
约 `+7.1`、`+10.2` 和 `+7.4` 个百分点，9 个数据集×KIR 单元均达到配对 CI 排除零的
稳定结果。StackOverflow 的优势最明显，说明训练表示在该数据集主要改善了单中心的 Known/OOS
分数分离，而不是靠增加局部球数量。

相对于 MOGB 组件，Trainable 的 OOS F1 平均优势为：

| 比较 | 平均优势 | 95% CI 排除零的单元 |
|---|---:|---:|
| MOGB MiniLM | +12.4 pp | 8/9 |
| MOGB partition + ours boundary | +8.0 pp | 8/9 |
| ours partition + MOGB boundary | +9.9 pp | 8/9 |

唯一未形成稳定正差异的区域集中在 Banking77/KIR=0.25：MOGB partition + ours boundary
相对 Trainable 的差异约为 `-0.4 pp`，CI 跨过零。这是一个局部工作点，不能外推为
MOGB 全面优胜。

### 3.2 不是单指标的“拒识越强越好”

Trainable 相对 Frozen K=1 的平均变化为：OOS F1 `+7.1 pp`、F1-All `+5.1 pp`、F1-K
`+5.0 pp`、AUROC `+3.8 pp`、AUPR-OOS `+5.4 pp`，同时 Known Recall 下降约 `3.3 pp`、
false acceptance 降低约 `14.3 pp`。因此当前证据支持“更安全且较平衡的工作点”，但不支持
“所有指标均提高”。Known Recall 的代价必须在主表中保留。

Frozen K=2 和 Random K=2 的 OOS F1 与 false acceptance 也有正向变化，但其 Known Recall
分别下降约 `1.2` 和 `4.0 pp`。这说明新增中心的收益伴随覆盖风险变化，不能将 OOS F1 的提升
解释为多中心在所有数据集都更合理。

### 3.3 MOGB 组件的工作点是“保守拒识”

MOGB MiniLM 相对 Trainable 的主要差异不是简单的 OOS 失败，而是极高的 Known 拒绝：
Trainable 相对它的 Known Recall/F1-K 平均优势约 `+49.0/+38.6 pp`，同时 Trainable 的
false acceptance 反而高约 `7.8 pp`。MOGB partition + ours boundary 和 ours partition +
MOGB boundary 也分别带来约 `28.0` 和 `30.2 pp` 的 Known Recall 优势，但 Trainable 的
false acceptance 更低并不成立（对应优势约 `-6.1` 和 `-1.4 pp`）。

因此，这些行揭示的是不同的 coverage–open-space 工作点：MOGB 组件更保守，Trainable K=1
更平衡；它们不是同监督、同训练条件下的 SOTA 排名，也不是官方 BERT MOGB 复现。

## 4. 图表阅读方式

1. `trainable_oos_f1_effect_heatmap.png`：看每个数据集/KIR 的 OOS F1 配对优势；星号表示
   95% paired bootstrap CI 不跨零。
2. `trainable_oos_fa_effect_landscape.png`：横轴是 false-acceptance 风险降低量，纵轴是
   OOS F1 优势；右上方表示同时更安全且 OOS F1 更高。
3. `trainable_multimetric_effect_summary.png`：看整体指标向量，避免只挑 OOS F1。

这些图用于机制解释，不用于从测试集反选 K、阈值、半径或训练配置。

## 5. 当前结论与限制

当前最稳妥的结论是：在已有统一 fair matrix 内，Known-only Trainable MiniLM K=1 是最稳定
的自有工作点；固定多中心和 MOGB 组件的差异主要表现为“边界覆盖与拒识保守性”的取舍。
这批证据仍不能回答 Trainable 是否超过 ADB、DA-ADB 或 DCLOOS，因为这些外部方法尚未在
同一监督合同、split、seed 和运行环境下完成统一矩阵。MOGB 官方 BERT 复现也仍保持
`audited_not_reproduced`，不能把 MiniLM 组件适配结果称为严格论文复现。

下一步不是继续扩展 K 或新增损失，而是：先恢复一个可验证的外部 baseline runtime，完成
StackOverflow/KIR=0.50/seed=42 的同协议单格，再按相同监督条件扩展多 seed；同时保留本阶段
作为性能—风险机制证据。

