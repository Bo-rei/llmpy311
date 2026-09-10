# 同协议五 seed 统计稳定性与方法排名（v1）

## 1. 目的与范围

本阶段使用已经完成的 `protocol_v2_textoir_v1` fair prediction 指标，按同一 `dataset × KIR × seed` 做配对比较。它不新增训练、不重新计算边界、不选择阈值，专门回答：

1. Trainable K=1 的优势是否跨 seed 稳定；
2. 固定 K=2、随机划分和 MOGB 组件的差异是否只是某个 seed 偶然；
3. 不同数据集/KIR 下，谁在 OOS F1 上稳定占优；
4. 结果差异是否伴随 Known Recall 或 false acceptance 的代价。

输入为 `results/analysis/cross_protocol_tradeoff_v1/per_seed.csv`，覆盖三数据集、三 KIR、七方法和五个正式 seed，共 315 行。

## 2. 方法

每个比较都以同一 `dataset × KIR × seed` 为配对单位。对 Trainable K=1 与六个比较方法计算：

- OOS F1；
- F1-All；
- F1-K/Known Recall；
- AUROC/AUPR-OOS（若源表提供）；
- false acceptance；
- false rejection。

使用固定 RNG seed `20260725` 的 10,000 次配对 bootstrap，输出均值、median、95% CI、标准差、win/tie/loss、effect size 和 sign-test p 值。p 值只作描述性证据，因为每个 dataset×KIR 只有五个 seed。

## 3. 主要结果

### 3.1 Trainable K=1 的 OOS F1 稳定性

Trainable K=1 在 27 个 dataset×KIR×比较方法组合中：

- 相对 Frozen K=1、Frozen K=2、Random K=2 的 9/9 个组合，五个 seed 全部胜出，配对 95% CI 均为正；
- 相对 MOGB-MiniLM、MOGB partition + ours、ours partition + MOGB 的 8/9 个组合，五个 seed 全部胜出，配对 95% CI 均为正；
- 唯一没有稳定正 CI 的区域是 Banking77/KIR=.25 的部分 MOGB 组件比较，这与该工作点的 MOGB partition + ours 平均 OOS F1 略高于 Trainable 一致。

因此，Trainable K=1 的优势不是由一个偶然 seed 造成，但也不是“所有 dataset/KIR 都绝对第一”。更准确的表述是：它在 8/9 个 dataset×KIR 工作点取得最高平均 OOS F1，并在大多数关键配对中保持一致胜出。

### 3.2 平均配对 OOS F1 优势

对全部 9 个 dataset×KIR 汇总，Trainable 相对比较方法的平均 OOS F1 优势约为：

| 比较方法 | 平均优势 |
|---|---:|
| Frozen K=1 | +7.11 pp |
| Frozen K=2 | +10.25 pp |
| Random K=2 | +7.36 pp |
| MOGB-MiniLM | +12.36 pp |
| MOGB partition + ours | +8.01 pp |
| Ours partition + MOGB | +9.86 pp |

这些数值是同协议 Frozen/Trainable/MOGB 组件比较，不包含 ADB、DA-ADB 或 DCLOOS，也不是跨监督条件的 SOTA 证明。

### 3.3 数据集和 KIR 差异

- **CLINC150**：Trainable 在三个 KIR 都是 OOS F1 平均排名第一；MOGB-MiniLM 差距在 KIR 增大时扩大。
- **Banking77**：KIR=.25 时 MOGB partition + ours 略高于 Trainable；KIR=.50 和 .75 时 Trainable 排名第一，说明多中心收益只在部分开放程度下出现。
- **StackOverflow**：Trainable 在三个 KIR 都是第一；Frozen K=2 在 KIR=.50 的平均排名为第 7，固定多中心退化不是单个 seed 现象。

## 4. 图表

- `figures/archive/analysis/statistical_stability_v1/paired_oos_f1_forest.png`：Trainable 相对各 fair 方法的五 seed 配对 CI；
- `figures/archive/analysis/statistical_stability_v1/method_rank_heatmap_oos_f1.png`：每个 dataset/KIR 的 OOS F1 平均排名；
- `figures/archive/analysis/statistical_stability_v1/five_seed_oos_f1_kir_curves.png`：均值和标准差随 KIR 的变化。

对应轻量数据：

- `results/analysis/archive/analysis/statistical_stability_v1/paired_effects.csv`；
- `results/analysis/archive/analysis/statistical_stability_v1/rank_per_seed.csv`；
- `results/analysis/archive/analysis/statistical_stability_v1/rank_summary.csv`；
- `results/analysis/archive/analysis/statistical_stability_v1/MANIFEST.json`。

## 5. 解释边界

这批统计证明的是现有同协议 fair 结果的 seed 稳定性，不证明：

- Trainable 已超过 BERT 版 ADB/DA-ADB；
- Trainable 已超过使用额外 pseudo/external OOS 的 DCLOOS；
- MOGB 官方论文结果已被严格复现；
- Gate-only OOS F1 已经等同完整 Cascade 性能。

外部 baseline 仍需在同一 split、监督说明和可验证 runtime 下单独运行后才能进入正式主表。

## 6. 复现

```bash
python tools/analysis/build_statistical_stability_v1.py
```

