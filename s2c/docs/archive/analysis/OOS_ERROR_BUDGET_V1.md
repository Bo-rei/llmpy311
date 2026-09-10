# OOS 误接收—误拒绝预算 V1

## 1. 目的与边界

本阶段把已有同协议 fair matrix 的 OOS F1 拆成可解释的错误预算，回答：

1. Trainable MiniLM 的收益来自减少 OOS→Known 误接收，还是来自增加 Known→OOS 误拒绝？
2. 固定多中心和 MOGB 组件的 OOS 工作点分别偏向哪一种风险？
3. 在相同 `dataset × KIR × seed` 配对下，Trainable K=1 是否比这些冻结组件更平衡？

本阶段是 **analysis-only**：不重新训练、不重拟合中心、不改变阈值、不选择超参数，也不修改 E2、
E3、R1 或历史运行。输入是已经通过 sample-level 对齐检查的 315 行方法指标摘要：

```text
results/analysis/archive/analysis/cross_dataset_error_attribution_v1/method_metrics_per_seed.csv
```

范围为 3 个数据集 × 3 个 KIR × 5 个 seed × 7 个当前协议方法。所有重采样使用固定 RNG seed
`20260808`，共 10,000 次；原始逐样本预测和大文件仍只在本地 artifact 中保存。

## 2. 指标重构与审计

令 `FA` 为 OOS 被错误接受为 Known 的比例，`FR` 为 Known 被错误拒绝为 OOS 的比例，
`N_oos` 和 `N_known` 分别为两类测试样本数量，则：

```text
OOS Recall    = 1 - FA
OOS TP        = N_oos × (1 - FA)
Known FP      = N_known × FR
OOS Precision = OOS TP / (OOS TP + Known FP)
OOS F1        = 2 × Precision × Recall / (Precision + Recall)
```

重构后的 OOS F1 与审计源表逐行核对，最大绝对误差小于 `1e-9`；输出文件没有 NaN 或重复键。
因此这里的 precision/recall 不是重新用测试集调参，而是对已冻结预测的错误计数进行透明分解。

## 3. KIR=.50 的工作点

下表是五 seed 均值。`FA` 是 OOS→Known，`FR` 是 Known→OOS；百分比仅为阅读方便。

| 数据集 | 方法 | OOS F1 | OOS Precision | OOS Recall | Known Recall | FA | FR |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | Trainable K=1 | 90.44 | 85.25 | 96.31 | 74.44 | 3.69 | 25.56 |
| CLINC150 | Frozen K=1 | 88.94 | 86.81 | 91.18 | 78.76 | 8.82 | 21.24 |
| CLINC150 | Frozen K=2 | 89.20 | 85.23 | 93.56 | 75.14 | 6.44 | 24.86 |
| CLINC150 | MOGB MiniLM 组件 | 81.32 | 68.95 | 99.10 | 31.57 | 0.90 | 68.43 |
| Banking77 | Trainable K=1 | 83.56 | 82.93 | 84.26 | 82.21 | 15.74 | 17.79 |
| Banking77 | Frozen K=1 | 72.43 | 81.69 | 65.15 | 85.01 | 34.85 | 14.99 |
| Banking77 | Frozen K=2 | 75.46 | 80.66 | 70.95 | 82.55 | 29.05 | 17.45 |
| Banking77 | MOGB MiniLM 组件 | 74.99 | 60.39 | 98.91 | 33.41 | 1.09 | 66.59 |
| StackOverflow | Trainable K=1 | 87.67 | 84.91 | 90.66 | 83.89 | 9.34 | 16.11 |
| StackOverflow | Frozen K=1 | 76.55 | 84.48 | 70.29 | 87.15 | 29.71 | 12.85 |
| StackOverflow | Frozen K=2 | 63.53 | 80.09 | 52.83 | 86.89 | 47.17 | 13.11 |
| StackOverflow | MOGB MiniLM 组件 | 72.92 | 57.65 | 99.21 | 27.09 | 0.79 | 72.91 |

读法不是“FA 越低的方法必然最好”。例如 MOGB MiniLM 组件几乎拒绝所有 OOS，但同时把约
三分之二的 Known 样本拒绝掉；它是保守拒识工作点，不是同监督下的整体优胜证据。Trainable
K=1 的特点是同时把 FA 压到较低水平，并保留明显更高的 Known Recall，因此 OOS F1/F1-All
更平衡。固定 K=2 在 StackOverflow 的主要问题清晰表现为 FA 达到 `47.17%`，而不是 Known
覆盖不足。

## 4. Trainable 的配对变化

以同一 `dataset × KIR × seed` 为配对单位，KIR=.50 时 Trainable K=1 相对于 Frozen K=1 的
变化为：

| 数据集 | OOS F1 | OOS Precision | OOS Recall | Known Recall | FA | FR |
|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | +1.50pp | −1.57pp | +5.12pp | −4.31pp | −5.12pp | +4.31pp |
| Banking77 | +11.14pp | +1.24pp | +19.10pp | −2.80pp | −19.10pp | +2.80pp |
| StackOverflow | +11.12pp | +0.43pp | +20.37pp | −3.25pp | −20.37pp | +3.25pp |

这说明 Trainable 的主要作用是改善 OOS 排序/分离并降低误接收；代价是 Known Recall
下降约 2.8–4.3 个百分点，而不是通过大幅放弃 Known 来换取 OOS F1。相对于 Frozen K=2，
StackOverflow 的 OOS F1 提升 `+24.14pp`、FA 降低 `37.83pp`；相对于 MOGB MiniLM 组件，
Trainable 的 Known Recall 高 `56.80pp`，但 OOS Recall 低 `8.55pp`。后者再次说明两者处在
不同的 coverage–rejection 工作点。

## 5. 可视化证据

输出目录：

```text
results/analysis/archive/analysis/oos_error_budget_v1/
figures/archive/analysis/oos_error_budget_v1/
```

- `oos_precision_recall_workpoints.png`：三个数据集在 KIR=.50 的 OOS precision–recall 工作点。
- `oos_error_budget_kir050.png`：OOS→Known 与 Known→OOS 两类错误率并列比较。
- `trainable_oos_error_budget_effects.png`：Trainable 相对各方法的配对效应和 bootstrap CI。

第一张图适合判断“谁在同时追求 precision 与 recall”；第二张图直接显示 StackOverflow 固定
K=2 的 FA 爆炸，以及 MOGB 组件的 FR 代价；第三张图用于检查五 seed 配对方向是否一致。
图表是机制诊断，不是从 test OOS 反选阈值或配置。

## 6. 当前研究结论

1. 当前 fair matrix 内，Trainable MiniLM K=1 是最平衡的自有 Gate 工作点：它明显降低
   OOS 误接收，同时保持比 MOGB 组件高得多的 Known 覆盖。
2. StackOverflow 固定多中心的失败主要是 boundary-union 过覆盖，而不是单纯的 Known
   false rejection；新增中心恢复的 Known 样本不足以抵消新增 OOS 接受。
3. MOGB MiniLM/组件结果不能直接写成“低 FA 因而优于 Trainable”：它们显著提高 Known
   false rejection，且使用冻结 MiniLM 与不同边界/半径合同。
4. 这些数字仍不能回答 Trainable 是否超过完整 MOGB、ADB、DA-ADB 或 DCLOOS。官方 MOGB
   BERT 复现仍是 `not_reproduced_strict`；ADB/DA-ADB/DCLOOS 同协议运行仍需独立 runtime
   和监督条件审计。DCLOOS 的外部/伪 OOS 监督不能与 Known-only 方法直接排名。

## 7. 复现入口

```bash
python tools/analysis/build_oos_error_budget_v1.py
```

脚本只读取已审计的轻量指标摘要，重新生成 CSV、manifest 和 PNG；它不会读取 `textoir/data`，
不会访问测试文本，也不会覆盖历史 run。阶段 ledger 为 `oos_error_budget_v1`，策略为
`do_not_repeat`。
