# CLINC KIR=.50 Trainable MiniLM Gate 搜索汇报

## 结论

在固定 CLINC KIR=.50、Known-only 训练和现有 Gate→Router→Expert 框架下，完成了两阶段搜索：复用四类已有 checkpoint 搜索几何配置，并训练4种配方×3 seed后确认 full pipeline。

当前最优 full-pipeline OOS F1 为 **91.93±0.48%**，比此前 `91.64±0.27%` 提升约 `+0.29 pp`，但相对论文 Ours `91.96%` 仍低 `0.03 pp`。本轮没有得到稳定超过论文的 KIR=.50 配置。

## 1. 三组结果

| 实验线 | seed13 | seed42 | seed87 | 均值±std | 相对论文 |
|---|---:|---:|---:|---:|---:|
| 论文 Ours | — | — | — | 91.96 | — |
| 已有 checkpoint selection | 91.37 | 91.56 | 92.01 | 91.64±0.27 | −0.32 pp |
| geometry search | 91.40 | 91.58 | 92.39 | 91.79±0.43 | −0.17 pp |
| recipe search | 91.55 | 91.64 | 92.61 | **91.93±0.48** | **−0.03 pp** |

所有结果都是 validation OOS F1 选择后再读取 test；测试集没有参与训练、checkpoint、中心数、协方差或阈值选择。

## 2. Geometry search

搜索了 `existing_last2`、`last2_long`、`projection_only`、`lora_long` 四类表示，以及 K=1/2/3/5、top25/top50 adaptive K=2、局部/收缩协方差、mean+std/95%分位数半径和两种 acceptance mode。

最终 validation 锁定：

| Seed | 表示 | 几何 | 半径 / λ | threshold / mode | Test OOS F1 |
|---|---|---|---|---|---:|
| 13 | last2_long | adaptive50 K=2 | mean+2.5std | .90 / nearest | 91.40 |
| 42 | lora_long | K=2 | q95 | 1.02 / normalized union | 91.58 |
| 87 | lora_long | adaptive50 K=2 | q95 | 1.00 / nearest | 92.39 |

几何改进带来小幅收益，但中心数量和协方差变化没有形成跨 seed 的稳定突破。新增中心仍需承担开放空间覆盖风险，不能由 CLINC 的局部 seed 结果推导出多中心普遍有效。

## 3. Recipe search

在 Known-only 条件下训练12个单元：last2 的低学习率、较高 temperature、去掉 inter loss，以及较低学习率 LoRA；每个配方3个 seed，每个 epoch 由 validation OOS F1 选择。

最终配置：

| Seed | 配方 / epoch | λ / threshold / mode | Test OOS F1 | Pipeline macro F1 | Accuracy |
|---|---|---|---:|---:|---:|
| 13 | last2_temp10 / 9 | 1.0 / 1.20 / nearest | 91.55 | 81.88 | 87.53 |
| 42 | last2_no_inter / 9 | .75 / 1.25 / nearest | 91.64 | 82.64 | 87.91 |
| 87 | last2_temp10 / 9 | 1.0 / 1.30 / nearest | 92.61 | 83.19 | 88.64 |

三 seed 的 Gate OOS F1 与 pipeline OOS F1 逐单元一致；Router/Expert 不改变当前 `semantic_gate_enabled=False` 下的 OOS 二分类。配方搜索的 full-pipeline macro F1 为 `82.57±0.54%`，Accuracy 为 `88.02±0.46%`。

## 4. 判因与下一步

当前 score ranking 已接近但没有跨过论文参照；同一表示上的简单 margin/competition 规则此前约为 `91.49%`，没有收益。继续扩大 threshold 网格的价值很低。

下一步真正可能改变 ranking 的方向是：Known-only 的语义锚定损失、局部支持度/协方差模型和核心语义双视图。它们需要重新定义 score 并保持 validation-only 选择，不能直接把现有距离阈值结果外推。

KIR=.25 的独立 matched full pipeline 已达到 `95.10±0.30%`，但它是不同 KIR 条件，不能作为 KIR=.50 的突破；应在最终汇报中单独画 KIR 曲线。

## 5. 证据入口

- Geometry：`results/analysis/historical_trainable_gate_search/`，含 validation candidates、selection lock、selected test 和 manifest。
- Recipe：`results/analysis/historical_trainable_gate_recipe_search/`，含12个训练历史、validation candidates、selection lock、selected test 和 manifest。
- 主脚本：[geometry search](../../scripts/experiments/run_historical_trainable_gate_search.py) 和 [recipe search](../../scripts/experiments/run_historical_trainable_gate_recipe_search.py)。
- 当前证据属于 `historical_v19_paper_main` 的 H1 controlled；未修改 `fulltex.tex`，未公开 raw predictions、embedding 或逐样本 score。
