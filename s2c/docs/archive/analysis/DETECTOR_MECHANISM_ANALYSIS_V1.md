# Detector 机制分析 V1：收益来自表示、Gate，还是二者交互

更新时间：2026-08-10  
活动协议：`protocol_v2_textoir_v1`

## 1. 目的和合同

这是一组已经完成结果的 analysis-only 对照，不新增训练。范围固定为：CLINC150、Banking77、StackOverflow，KIR=.50，seed=13/42/87。

比较两层：

1. **同一 Trainable MiniLM 表示**：Trainable K=1 Gate、MSP、Energy、kNN、LOF；
2. **同一原生 detector 换表示**：Trainable MiniLM 与 Frozen MiniLM 的 MSP/Energy/kNN/LOF。

原实验只使用 Known calibration 选择 detector 工作点，测试 OOS 只做最终评价。本报告不把这些三 seed 控制扩充成五 seed 主矩阵，也不与 BERT/外部 OOS 监督方法混排。

## 2. 关键结果

### 2.1 同一 Trainable MiniLM 表示下，Gate 是最强的 OOS 工作点

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---|---:|---:|---:|---:|
| CLINC150 | Trainable Gate K=1 | **90.43%** | 81.90% | 74.27% | **3.61%** |
| CLINC150 | MSP | 88.38% | **86.27%** | 95.07% | 18.21% |
| CLINC150 | Energy | 88.30% | 85.59% | 95.02% | 18.37% |
| CLINC150 | kNN | 86.81% | 83.67% | 94.61% | 20.60% |
| CLINC150 | LOF | 78.70% | 78.91% | 95.64% | 33.21% |
| Banking77 | Trainable Gate K=1 | **84.77%** | **82.31%** | 81.91% | **13.46%** |
| Banking77 | MSP | 72.56% | 80.25% | 93.86% | 39.66% |
| Banking77 | Energy | 69.52% | 78.07% | 92.85% | 43.01% |
| Banking77 | kNN | 67.35% | 77.55% | 94.78% | 46.65% |
| Banking77 | LOF | 57.66% | 74.59% | 95.04% | 57.46% |
| StackOverflow | Trainable Gate K=1 | **86.71%** | **85.65%** | 83.92% | **11.14%** |
| StackOverflow | kNN | 68.83% | 77.22% | 95.17% | 44.59% |
| StackOverflow | LOF | 63.53% | 76.06% | 94.93% | 50.70% |
| StackOverflow | MSP | 45.61% | 70.95% | 95.60% | 69.04% |
| StackOverflow | Energy | 42.95% | 69.80% | 95.66% | 71.43% |

这说明当前 Gate 的作用不是单纯提高 Known Recall。它选择了更严格的 Known/OOS 工作点：Known Recall 下降，但 false acceptance 大幅下降，OOS F1 在三个数据集都最高；F1-All 则需要同时报告，CLINC150 上 MSP/Energy 的 Known 覆盖更高，所以 F1-All 反而更好。

### 2.2 表示训练本身有收益，但不能解释全部 Gate 优势

Trainable 相对 Frozen 的原生 detector 差值：

- CLINC150：OOS F1 提升约 `+2.97--+15.59 pp`，false acceptance 降低约 `4.98--19.41 pp`；
- Banking77：OOS F1 提升约 `+6.33--+11.95 pp`，false acceptance 降低约 `7.01--11.90 pp`；
- StackOverflow：kNN/LOF 分别提升 `+7.23/+30.03 pp`，但 Energy/MSP 分别下降 `-2.19/-4.42 pp`。

因此 Trainable MiniLM 的表示适配对原生 detector 通常有帮助，但在 StackOverflow 上并不保证所有 detector 都改善。Trainable Gate 仍显著高于这些原生 detector，说明当前优势来自：

```text
Trainable representation
        +
Known-only calibrated geometric Gate
```

而不是表示训练单独造成的。

### 2.3 StackOverflow 是最清晰的机制证据

StackOverflow/KIR=.50 上，Trainable Gate 的 OOS F1 为 `86.71%`，而同一表示上的 MSP/Energy 只有 `45.61%/42.95%`；两者 Known Recall 都约 `95.6%`，但 false acceptance 达到 `69%--71%`。这与固定 K=2 的过覆盖现象一致：如果决策器过度接受 Known 区域，Known Recall 看起来很好，但 OOS F1 会崩溃。

Gate 把 false acceptance 降到 `11.14%`，代价是 Known Recall 降到 `83.92%`。因此这里的核心不是“Gate 让所有分类指标都更高”，而是它更好地控制开放空间风险；F1-All、Known Recall 和 OOS F1 必须一起报告。

### 2.4 配对 bootstrap：优势不是单个 seed 偶然

为避免只比较三 seed 的均值，本分析直接复用源实验已经登记的 10,000 次 paired
bootstrap（固定 RNG seed=`20260725`），不重新抽样、不读取测试结果做选择。Gate 与原生
detector 的源表方向是 `native - Gate`，报告中已统一改为 `Gate - native`。

在 OOS F1 上，Gate 相对同一 Trainable 表示的四种原生 detector 均为 3/3 seed 胜出，
且 95% bootstrap 区间全部为正：

| 数据集 | 相对 MSP | 相对 Energy | 相对 kNN | 相对 LOF |
|---|---:|---:|---:|---:|
| CLINC150 | `+2.05 pp [0.27, 4.44]` | `+2.13 pp [1.41, 2.94]` | `+3.62 pp [2.73, 4.58]` | `+11.73 pp [9.33, 15.21]` |
| Banking77 | `+12.21 pp [11.70, 13.20]` | `+15.25 pp [14.51, 16.24]` | `+17.42 pp [16.02, 18.50]` | `+27.11 pp [23.80, 29.55]` |
| StackOverflow | `+41.10 pp [36.38, 44.50]` | `+43.76 pp [41.23, 45.48]` | `+17.88 pp [13.67, 26.07]` | `+23.17 pp [16.92, 31.37]` |

表示层的 Trainable-Frozen 差值则不是所有 detector 都同方向：StackOverflow 的 MSP/Energy
分别为 `-4.42/-2.19 pp`，而 kNN/LOF 为 `+7.23/+30.03 pp`。这进一步说明当前结果不是
“微调后任何 detector 都变好”，而是表示适配与几何 Gate 的组合产生了主要收益。

新增的机器可读配对表为：

- `results/analysis/archive/analysis/detector_mechanism_v1/detector_paired_ci.csv`（48 行）；
- `results/analysis/archive/analysis/detector_mechanism_v1/representation_paired_ci.csv`（48 行）。

新增图为：

- `figures/archive/analysis/detector_mechanism_v1/gate_vs_native_oos_f1_ci.png`；
- `figures/archive/analysis/detector_mechanism_v1/trainable_vs_frozen_native_oos_f1_ci.png`。

## 3. 图表

- `figures/archive/analysis/detector_mechanism_v1/native_detector_comparison.png`：同一 Trainable 表示下的三指标柱状图；
- `figures/archive/analysis/detector_mechanism_v1/native_detector_frontier.png`：OOS F1—false acceptance 前沿；
- `figures/archive/analysis/detector_mechanism_v1/representation_vs_detector_gain.png`：Trainable 与 Frozen 原生 detector 的表示增益。

机器可读结果：

- `results/analysis/archive/analysis/detector_mechanism_v1/detector_summary.csv`（27 行）；
- `results/analysis/archive/analysis/detector_mechanism_v1/representation_gain_summary.csv`（12 行）；
- `results/analysis/archive/analysis/detector_mechanism_v1/gate_vs_native_paired.csv`（36 行）；
- `results/analysis/archive/analysis/detector_mechanism_v1/detector_paired_ci.csv`（48 行）；
- `results/analysis/archive/analysis/detector_mechanism_v1/representation_paired_ci.csv`（48 行）；
- `results/analysis/archive/analysis/detector_mechanism_v1/MANIFEST.json`。

## 4. 当前边界

这组结果只能证明当前 Trainable MiniLM + Gate 相对原生 detector 的机制差异，不能替代：

- 五 seed 的 native detector 正式主矩阵；
- ADB/DA-ADB 的同一骨干公平比较；
- MOGB 官方 BERT 严格复现；
- DCLOOS 的同 KIR、同 seed、同预算比较。

下一步可以继续补充同一 Trainable 表示下的 score 分布、ROC/PR 和逐样本错误交集；不需要重新扫描 K/KIR。

## 5. 复现

```bash
python tools/analysis/build_detector_mechanism_v1.py
```
