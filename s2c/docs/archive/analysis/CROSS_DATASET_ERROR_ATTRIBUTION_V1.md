# 三数据集 × 三 KIR 逐样本错误归因（v1）

## 1. 目标

在 StackOverflow 的逐样本分析基础上，继续使用已有 `protocol_v2_textoir_v1` 预测文件，对 CLINC150、Banking77、StackOverflow 在 KIR=0.25/0.50/0.75 下的 7 个方法做同一套错误归因。这里不训练、不调参、不改变任何正式实验结果，重点回答：

- Trainable MiniLM K=1 的优势是否跨数据集/KIR 保持；
- 固定 K=2 的 OOS 过覆盖是否集中在 StackOverflow，还是普遍存在；
- MOGB 组件的低 false acceptance 是否以 Known 大量误拒为代价；
- 哪些差异是表示训练造成的，哪些是边界/粒球组合造成的。

## 2. 覆盖与审计

| 项目 | 值 |
|---|---|
| 数据集 | CLINC150、Banking77、StackOverflow |
| KIR | 0.25、0.50、0.75 |
| seeds | 13、42、87、100、123 |
| 方法 | Trainable K=1、Frozen K=1/K=2、Random K=2、MOGB-MiniLM、MOGB partition + ours boundary、Ours partition + MOGB boundary |
| 源 run | 315 |
| 对齐逐样本记录 | 315 × 6,000 = 1,890,000 |
| 正式判定 | score `<=1` 为 Known，`>1` 为 OOS |

每个 `dataset × KIR × seed` 内检查了所有方法的 `sample_id` 顺序和 `gold_intent` 完全一致，并用 score 重新计算 OOS F1、Known Recall、false acceptance、false rejection；重算值与源 metrics 在浮点误差内一致。

结果文件：

- [方法级逐 seed 指标](../../results/analysis/archive/analysis/cross_dataset_error_attribution_v1/method_metrics_per_seed.csv)
- [方法级均值汇总](../../results/analysis/archive/analysis/cross_dataset_error_attribution_v1/method_metrics_summary.csv)
- [Trainable 相对其它方法的 OOS 转移](../../results/analysis/archive/analysis/cross_dataset_error_attribution_v1/trainable_oos_transition_effects.csv)
- [false acceptance 差值](../../results/analysis/archive/analysis/cross_dataset_error_attribution_v1/false_acceptance_delta_vs_trainable.csv)
- [false rejection 差值](../../results/analysis/archive/analysis/cross_dataset_error_attribution_v1/false_rejection_delta_vs_trainable.csv)
- [manifest](../../results/analysis/archive/analysis/cross_dataset_error_attribution_v1/MANIFEST.json)

## 3. 主要结果

### 3.1 Trainable K=1 相对 Frozen K=1/K=2/Random K=2

相对 Frozen K=1、Frozen K=2 和 Random K=2，Trainable K=1 的 false acceptance 在全部 9 个 dataset×KIR 组合中都更低。差距在开放程度升高时扩大，尤其是 Banking77 和 StackOverflow：

- Banking77 KIR=.75：相对 Frozen K=1/K=2/Random K=2 分别减少约 28.39、16.47、29.71 个百分点的 false acceptance；
- StackOverflow KIR=.75：相对 Frozen K=1/K=2/Random K=2 分别减少约 24.03、52.87、25.76 个百分点；
- CLINC150 差距较小，KIR=.75 时分别减少约 6.09、3.31、6.21 个百分点。

这说明 Trainable MiniLM 的优势不是只出现在 StackOverflow，但 StackOverflow 的多中心边界风险最明显。

### 3.2 OOS 错误转移

`trainable_oos_transition_effects.csv` 用 `Trainable-only correct - comparison-only correct` 表示 Trainable 在 OOS 样本上的正确拒识优势：

- 与 Frozen K=2 比较：Banking77 为 +0.10/+0.13/+0.16，CLINC150 为 +0.02/+0.03/+0.03，StackOverflow 为 +0.16/+0.38/+0.53（KIR 从 .25 到 .75）；
- 与 Random K=2 比较：Banking77 为 +0.12/+0.21/+0.30，CLINC150 为 +0.03/+0.05/+0.06，StackOverflow 为 +0.12/+0.22/+0.26；
- 与 Frozen K=1 比较：Banking77 为 +0.12/+0.19/+0.28，CLINC150 为 +0.03/+0.05/+0.06，StackOverflow 为 +0.11/+0.20/+0.24。

因此，Trainable K=1 相对固定/随机多中心的 OOS 优势随 KIR 增大而增强，StackOverflow 的增幅最大。

### 3.3 MOGB 组件的不同机制

MOGB-MiniLM 和 MOGB partition + ours boundary 在绝大多数组合下的 false acceptance 低于 Trainable，但 false rejection 显著高于 Trainable。例如：

- Banking77 KIR=.50：MOGB-MiniLM 的 false rejection 比 Trainable 高约 48.80 个百分点；MOGB partition + ours boundary 高约 30.03 个百分点；
- CLINC150 KIR=.50：分别高约 42.87 和 21.10 个百分点；
- StackOverflow KIR=.50：分别高约 56.80 和 33.51 个百分点。

所以 MOGB 组件的低 false acceptance 不能直接解释成更好的 OOS 检测；它更多体现为极保守工作点。与 Trainable 的差异是“拒绝多少 Known”，而不只是“拒绝多少 OOS”。

## 4. 可视化证据

- [Trainable OOS 正确性优势热力图](../../figures/archive/analysis/cross_dataset_error_attribution_v1/trainable_oos_win_heatmap.png)
- [false acceptance 差值热力图](../../figures/archive/analysis/cross_dataset_error_attribution_v1/false_acceptance_delta_vs_trainable.png)
- [false rejection 差值热力图](../../figures/archive/analysis/cross_dataset_error_attribution_v1/false_rejection_delta_vs_trainable.png)
- [跨数据集 OOS F1–false acceptance 散点图](../../figures/archive/analysis/cross_dataset_error_attribution_v1/cross_dataset_oos_f1_false_acceptance.png)

图中正值表示对比方法比 Trainable 更高的风险率；对 MOGB 列，正的 false-rejection 差值尤其明显，说明其保守性是以 Known 覆盖换来的。

## 5. 数据集差异的实验解释

### CLINC150

固定多中心相对 Trainable 的 false-acceptance 差距较小，Trainable 的 OOS 正确性优势也较小；这与 CLINC150 的固定多中心接近中性、收益有限的结果一致。

### Banking77

固定/随机多中心在较高 KIR 下更容易出现 OOS 误接收，但某些 MOGB 组件仍能通过极保守边界压低 false acceptance。Banking77 的多中心收益不能只由“中心更多”解释，需要同时考虑 Known 覆盖和工作点。

### StackOverflow

固定 K=2 的 false acceptance 随 KIR 增大快速恶化；Trainable K=1 相对 Frozen K=2 的 OOS 正确性优势在 KIR=.75 达到约 53 个百分点。这里最清楚地显示了“固定多中心 acceptance union 过覆盖”的机制。

## 6. 当前实验结论

在当前已有同协议 fair 结果中，Trainable MiniLM K=1 的优势可以概括为：

1. 相对 Frozen K=1/K=2/Random K=2，跨数据集、跨 KIR 都降低 false acceptance；
2. 在 StackOverflow，降低幅度随 KIR 增加而显著扩大；
3. 相对 MOGB 组件，它不是单纯追求最低 false acceptance，而是在 OOS 拒识和 Known 覆盖之间取得更好的平衡；
4. MOGB 组件的低 false acceptance 与高 Known false rejection 同时出现，因此不能只按 OOS F1 或 FA 单列排名。

这仍然不是跨监督条件的 SOTA 证明。ADB、DA-ADB、DCLOOS 尚未在同一 runtime、同一 split 和同一多 seed 协议下完成，不能从本报告推出对它们的最终排名。

## 7. 复现命令

```bash
python tools/analysis/build_cross_dataset_error_attribution_v1.py
```

脚本只读取已有逐样本预测，生成 analysis-only CSV/PNG/manifest，不修改训练 artifact。
