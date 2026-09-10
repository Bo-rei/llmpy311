# CLINC150：KIR=.25 Trainable MiniLM full pipeline 验证

## 结论

找到了一个比论文 `Ours` 更高的 CLINC 配置：在 `historical_v19_paper_main` 的 H1 controlled 协议下，KIR=.25、Trainable MiniLM、K=1、对角 Mahalanobis、`mean+1×std` 边界接入同协议重新训练的 Router/Expert 后，三 seed full pipeline OOS F1 为 **95.10±0.30%**，论文同 KIR 的 Ours 为 **95.01%**，均值高 **+0.09 pp**。

这不是单 seed 偶然值：seed13=`95.40%`、seed42=`95.21%` 超过论文，seed87=`94.68%` 低于论文；因此应表述为“均值略高、2/3 seed 超过”，不能表述为三个 seed 稳定超过。

## 结果对比

| KIR / 配置 | OOS F1 | Full-pipeline macro F1 | Accuracy | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|---:|
| 论文 Ours / KIR=.25 | 95.01 | 历史表未提供统一 macro F1 | 90.45 | — | — |
| Trainable K=1 / seed13 | 95.40 | 75.23 | 91.56 | 72.54 | 2.25 |
| Trainable K=1 / seed42 | 95.21 | 73.49 | 91.04 | 71.05 | 2.27 |
| Trainable K=1 / seed87 | 94.68 | 77.57 | 91.00 | 75.79 | 4.40 |
| **Trainable K=1 / 三 seed** | **95.10±0.30** | **75.43±1.68** | **91.20±0.26** | **73.13±1.98** | **2.97±1.01** |

Router error 平均为 `2.05%`，Expert error 平均为 `3.76%`。Gate OOS F1 与 pipeline OOS F1 逐单元一致，符合 `semantic_gate_enabled=False` 的当前级联逻辑；Router/Expert 主要影响 Known 分类，因此 macro F1 不能用 OOS F1 代替。

## 与 KIR=.50 的当前最佳结果对比

| 配置 | OOS F1 | 相对论文 Ours |
|---|---:|---:|
| KIR=.50，checkpoint-selected full pipeline | 91.64±0.27 | −0.32 pp |
| KIR=.25，既有 Trainable K=1 full pipeline | **95.10±0.30** | **+0.09 pp** |

KIR=.25 是目前最有价值的突破方向。它不是通过增加中心数量获得收益：这里仍然是 K=1。主要变化是已知意图比例降低，OOS/known 几何重叠减弱；因此 KIR 必须作为实验条件明确报告，不能把 KIR=.25 的结果写成 KIR=.50 的改进。

## 参考方案的筛选结论

- 固定 K=2/3、多中心和 adaptive K 已在 CLINC 上完成，不能稳定超过论文；CLINC adaptive overall 约为 `90.99±0.78%`。
- KIR=.50 的长训练 checkpoint 选择得到 `91.64±0.27%`，但只有 1/3 seed 超过论文；固定 score test oracle 均值 `91.78%`，不足以靠继续调阈值补齐差距。
- 在 KIR=.50 当前 checkpoint 上快速验证 competition margin 后，validation-only 选择的 test 均值约 `91.49%`，低于 `91.64%`，因此简单加入 `margin_gamma` 暂不作为改进方案。
- 旧资料中的 MOGB partition、Random partition 和其他 Gate-only 数字属于不同组件/协议，不能替代本次 matched full pipeline 结果。

## 协议与证据边界

- 数据：`assets/datasets/s2c/prepared/data/multidataset/v19/clinc150/kir25_seed{13,42,87}`。
- Gate：既有 Known-only Trainable MiniLM K=1 checkpoint，未使用测试 OOS 训练或选参。
- 下游：针对 KIR=.25 重新训练的 3 个 Router 和 30 个 Expert，全部组件审计为 ready；训练日志记录 `Device: cuda`。
- 推理：3 个 full pipeline 单元使用 CUDA；`test_used_for_selection=false`，`oos_used_for_training=false`。
- 结果：`results/analysis/historical_trainable_kir25_full_pipeline/per_seed.csv`。
- manifest：`results/analysis/historical_trainable_kir25_full_pipeline/MANIFEST.json`。
- 论文参照：`fulltex.tex` KIR=.25 的 CLINC Ours 行：OOS F1 `95.01`、Accuracy `90.45`。

该结果仍属于 H1 controlled evidence，不是严格 H0 原始 Cascade 的字节级复现。下一步若要形成主结论，应在不改变协议的前提下补充 KIR=.25 的独立重复或把 KIR=.25/.50/.75 作为完整 KIR 曲线一起汇报。
