# KIR 敏感性分解 V1：性能退化来自哪里

## 1. 目的与边界

本阶段对已经完成的 `protocol_v2_textoir_v1` 五 seed fair summary 做趋势分析，量化
Known Intent Ratio 从 `0.25` 增加到 `0.75` 时，各方法的 OOS F1、Known Recall 和错误率
如何变化。它不是新训练实验，也不使用测试集选择方法、阈值或半径。

输入：

```text
results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv
```

范围：3 个数据集 × 3 个 KIR × 7 个当前协议方法 × 6 个指标。对每个
`dataset × method × metric` 用三个 KIR 点计算端点变化 `value(.75)-value(.25)` 和线性斜率。

输出：

```text
results/analysis/archive/analysis/kir_sensitivity_decomposition_v1/
figures/archive/analysis/kir_sensitivity_decomposition_v1/
```

## 2. 结果概览

### 2.1 Trainable K=1 对开放比例最稳

从 KIR=.25 到 .75 的 OOS F1 下降：

| 数据集 | Trainable K=1 | Frozen K=1 | Frozen K=2 |
|---|---:|---:|---:|
| CLINC150 | −12.3 pp | −12.5 pp | −13.0 pp |
| Banking77 | −23.0 pp | −31.5 pp | −26.5 pp |
| StackOverflow | −19.2 pp | −23.9 pp | −43.0 pp |

Trainable 的 KIR 敏感性在 Banking77 和 StackOverflow 明显小于 Frozen K=1/K=2；
StackOverflow 的固定 K=2 下降幅度最大，说明 KIR 增大时其多球接受区域风险被进一步放大。

### 2.2 MOGB 组件的低误接收来自拒绝更多 Known

MOGB MiniLM 从 KIR=.25 到 .75 的 Known Recall 变化为：CLINC150 `−7.9 pp`、Banking77
`−11.2 pp`、StackOverflow `−11.4 pp`；MOGB partition + ours boundary 分别为
`−12.1/−15.8/−20.0 pp`。相应的 false acceptance 基本不升，部分数据集甚至略降。

这说明 MOGB 组件在当前 Frozen MiniLM 合同下采取了更保守的拒识工作点：随着 Unknown
intent 变少，仍有大量 Known 被拒绝。它不能仅凭低 false acceptance 被判定为整体更好。

### 2.3 Trainable 的风险曲线更平衡

Trainable K=1 从 KIR=.25 到 .75 的 false acceptance 增量约为：CLINC150 `+0.8 pp`、
Banking77 `+9.3 pp`、StackOverflow `+5.0 pp`；Known Recall 基本稳定
（`+1.5/+0.6/−0.2 pp`）。因此其 OOS F1 下降主要反映开放比例变难，而不是固定多中心
那样由新增误接受或由 MOGB 那样由 Known 过拒绝主导。

## 3. 机制解释

当前结果可以分成三种轨迹：

1. **Trainable K=1：** Known-only 表示适配改善单中心分数分离，KIR 变化时保持较稳定的
   Known coverage 和较低 false acceptance。
2. **Frozen K=2/Random K=2：** KIR 提高后，局部球的并集扩大风险更明显；StackOverflow
   Frozen K=2 的 false acceptance 端点增量约 `+42.2 pp`，与 OOS F1 下降 `43.0 pp`
   同步出现。
3. **MOGB 组件：** 多粒球/平均半径在当前 Frozen 表示下压低 false acceptance，但代价是
   Known Recall 和 F1-All 持续下降；这是保守工作点，不是无代价的 OOS 分离。

因此，KIR 不是单纯的难度横轴，它还暴露了不同边界机制的风险：固定多中心更容易开放空间
过覆盖，MOGB 组件更容易过度拒绝 Known，而 Trainable K=1 的退化相对平滑。

## 4. 图表

- `oos_f1_kir_sensitivity.png`：三数据集的 OOS F1–KIR 曲线。
- `kir_endpoint_delta_heatmap.png`：KIR=.25→.75 的六指标端点变化；正值表示指标值上升，
  对错误率则表示错误率增加。
- `coverage_oos_trajectory_by_kir.png`：每个方法在 Known Recall–OOS F1 平面上的 KIR 轨迹。

图中使用的是已有五 seed 均值，不是新的测试选择；它们用于解释“谁随 KIR 退化、为什么退化”，
不能替代显著性检验或外部基线公平排名。

## 5. 结论与限制

在当前协议下，Trainable K=1 是对 KIR 变化最稳的自有 Gate 工作点；StackOverflow 固定
K=2 的失败主要是 KIR 增大后 false acceptance 快速上升，而 MOGB 组件的低误接收主要伴随
Known Recall 大幅下降。该分析加强了“表示适配改善单中心分数分离、多中心和 MOGB 组件改变
coverage–open-space 工作点”的解释，但仍不能声称超过完整 MOGB、ADB、DA-ADB 或 DCLOOS。

下一步应继续补齐同监督外部 baseline 的可验证单格；在此之前不重复已有 K 网格，也不把
KIR 趋势图解释成 SOTA 证据。

