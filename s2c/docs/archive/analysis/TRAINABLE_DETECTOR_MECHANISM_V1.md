# Trainable MiniLM 检测器机制对照 V1

## 1. 目的与协议

本阶段固定 `protocol_v2_textoir_v1`、KIR=`0.50`、数据集
`CLINC150/Banking77/StackOverflow` 和 seed=`13,42,87`，只读取已完成结果，不训练新
模型。所有原生检测器都使用同一批 Trainable MiniLM 表示、同一数据划分和 Known-only
校准；比较对象为：

1. `Trainable Gate K=1`；
2. `MSP`；
3. `Energy`；
4. `kNN`；
5. `LOF`。

测试 OOS 未用于训练、阈值或 checkpoint 选择。需要注意：Gate 和原生检测器的校准
目标不完全相同（Gate 使用当前正式 Gate 规则，原生检测器使用 Known-only conformal
alpha=`0.05`），因此本报告是**同一表示下的检测器机制诊断**，不是外部 SOTA 排名。

## 2. 输出文件

- `results/analysis/archive/analysis/trainable_detector_mechanism_v1/paired_effects_per_seed.csv`
- `results/analysis/archive/analysis/trainable_detector_mechanism_v1/paired_effects_summary.csv`
- `results/analysis/archive/analysis/trainable_detector_mechanism_v1/performance_summary.csv`
- `results/analysis/archive/analysis/trainable_detector_mechanism_v1/MANIFEST.json`
- `figures/archive/analysis/trainable_detector_mechanism_v1/trainable_detector_pareto.png`
- `figures/archive/analysis/trainable_detector_mechanism_v1/trainable_detector_paired_effects.png`
- `figures/archive/analysis/trainable_detector_mechanism_v1/trainable_detector_error_balance.png`

生成脚本：`tools/analysis/build_trainable_detector_mechanism_v1.py`。

## 3. 主要结果

Trainable Gate 相对每个原生检测器的 OOS F1 配对提升（百分点）如下：

| 数据集 | MSP | Energy | kNN | LOF |
|---|---:|---:|---:|---:|
| CLINC150 | +2.05 | +2.13 | +3.62 | +11.73 |
| Banking77 | +12.21 | +15.25 | +17.42 | +27.11 |
| StackOverflow | +41.10 | +43.76 | +17.88 | +23.17 |

三个数据集、四个检测器的 12 个比较中，Gate 的 OOS F1 都在 3/3 seed 上更高。

但这不是“所有指标都更好”：

- Gate 的 Known Recall 相对原生检测器通常下降约 11–21 pp；
- Gate 的 false acceptance 显著降低，CLINC150 约降低 14.6–29.6 pp，Banking77
  约降低 26.2–44.0 pp，StackOverflow 约降低 33.4–60.3 pp；
- CLINC150 的 F1-All 只有 LOF 比 Gate 低，MSP/Energy/kNN 仍略高；
- Banking77 和 StackOverflow 的 F1-All 则普遍由 Gate 占优。

## 4. 机制解释

这组实验把表示因素控制住了：Trainable MiniLM checkpoint 不变，只换检测器。因此
可以得到两个较清晰的结论。

### 4.1 表示适配不是唯一来源

如果收益完全来自 Trainable 表示，那么换成 MSP、Energy、kNN、LOF 后应大致保留相同
的 OOS 优势。实际结果显示它们都保留了部分表示收益，但 OOS F1 明显落后于 Trainable
Gate，尤其是 StackOverflow。这说明当前优势还来自 Gate 的类内中心、半径和分数校准
方式，而不只是 encoder 变好了。

### 4.2 Gate 选择了更开放集导向的工作点

原生检测器的 Known Recall 很高，但 false acceptance 很大；Trainable Gate 主动牺牲
一部分 Known coverage，换取大幅减少 OOS 被误接收。因而 Gate 在 OOS F1 上更强，但
不能把它描述为无代价全面优于原生检测器。`trainable_detector_pareto.png` 的点位和
`trainable_detector_error_balance.png` 的双错误率图应一起阅读。

### 4.3 与固定多中心失败的联系

这组结果与 StackOverflow 固定 K=2 的诊断一致：主要风险不是 Known 样本完全无法
覆盖，而是接受区域过宽导致 OOS 进入 Known。Trainable K=1 的边界规则减少了这种
开放空间误接收；增加局部中心则可能重新引入 union-over-acceptance 风险。

## 5. 不能作出的结论

- 不能将该结果写成超过 MOGB、ADB、DA-ADB 或 DCLOOS；这些方法的表示、训练监督、
  校准合同和运行状态不同；
- 不能把 Gate 的 OOS F1 提升解释为 Known 分类全面提升；CLINC150 的 F1-All 已显示
  这种权衡；
- 不能把三 seed 诊断替代正式五 seed 外部 baseline 对比。

## 6. 下一步

在不重复 E2/E3/R1 的前提下，下一步应继续优先做同协议的误差对齐和可视化；外部
ADB/DA-ADB/DCLOOS 只有在独立 torch/BERT runtime 通过最小 smoke 后，才进入同监督
单格对照。当前应保留 Trainable K=1 作为安全自有基线，并把固定多中心结果作为
StackOverflow 的边界风险对照。
