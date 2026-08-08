# Trainable MiniLM KIR Sweep V1（K=1）

> 本报告只汇总新完成的 Trainable MiniLM K=1 KIR 控制，不把它冒充新的多中心方法。训练、checkpoint 选择和 Gate 评价均使用 `protocol_v2_textoir_v1`；测试 OOS 未用于选择。

## 1. 实验完成情况

- 范围：CLINC150、Banking77、StackOverflow；KIR=0.25/0.50/0.75；seed=13/42/87。
- 计划 27 个单元，完成 27 个；失败、缺失、重复、无效指标均为 0。
- 训练目标与已有 Trainable K=1 控制完全相同：最后两层 MiniLM + 残差 projection；只使用 Known train 和 Known calibration。
- 结果根目录：`artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1/`。

## 2. Trainable K=1 结果（均值±标准差）

| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Accept | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 0.25 | 92.17±0.78 | 79.27±2.29 | 81.93±0.75 | 9.45±1.61 | 92.87±0.39 |
| banking77 | 0.50 | 84.77±0.71 | 82.31±0.83 | 81.91±0.17 | 13.46±1.22 | 90.80±0.64 |
| banking77 | 0.75 | 69.53±3.37 | 84.20±1.07 | 82.61±0.42 | 18.33±5.89 | 89.17±2.20 |
| clinc150 | 0.25 | 94.94±0.30 | 79.01±0.95 | 73.33±0.61 | 3.61±0.54 | 95.32±1.03 |
| clinc150 | 0.50 | 90.43±0.62 | 81.90±1.01 | 74.27±0.32 | 3.61±1.06 | 95.13±0.61 |
| clinc150 | 0.75 | 82.55±0.42 | 82.43±0.36 | 74.57±0.55 | 4.06±0.50 | 94.79±0.41 |
| stackoverflow | 0.25 | 95.05±1.40 | 86.59±2.52 | 84.18±0.76 | 4.64±2.43 | 94.86±1.83 |
| stackoverflow | 0.50 | 86.71±0.96 | 85.65±0.37 | 83.92±0.39 | 11.14±2.02 | 91.75±1.07 |
| stackoverflow | 0.75 | 76.68±1.59 | 87.07±0.29 | 84.27±0.91 | 8.42±4.41 | 93.16±0.46 |

## 3. 相同 KIR、相同对角马氏距离下 Trainable−Frozen

| 数据集 | KIR | OOS F1 差值 | Known Recall 差值 | Trainable OOS F1 | Frozen OOS F1 |
|---|---:|---:|---:|---:|---:|
| banking77 | 0.25 | +0.59 pp | -4.46 pp | 92.17 | 91.58 |
| banking77 | 0.50 | -0.05 pp | -6.93 pp | 84.77 | 84.82 |
| banking77 | 0.75 | -14.02 pp | -7.28 pp | 69.53 | 83.55 |
| clinc150 | 0.25 | +0.64 pp | -0.35 pp | 94.94 | 94.29 |
| clinc150 | 0.50 | +2.41 pp | +0.59 pp | 90.43 | 88.02 |
| clinc150 | 0.75 | +3.59 pp | -0.24 pp | 82.55 | 78.95 |
| stackoverflow | 0.25 | +1.19 pp | +1.10 pp | 95.05 | 93.85 |
| stackoverflow | 0.50 | +7.69 pp | +0.70 pp | 86.71 | 79.02 |
| stackoverflow | 0.75 | +13.54 pp | +0.67 pp | 76.68 | 63.13 |

## 4. 机制解读

1. Trainable K=1 的收益不是只出现在 KIR=0.50：CLINC150 三个 KIR 均为正，StackOverflow 三个 KIR 均为正；Banking77 在 KIR=0.25 仅小幅为正，KIR=0.50 接近持平，KIR=0.75 明显下降。
2. KIR 增大时，Trainable K=1 的 OOS F1 普遍下降，尤其 Banking77/StackOverflow；这说明已知意图变密集后，Known/OOS 重叠仍是主要限制。
3. Banking77 的 KIR=0.75 训练结果虽然 F1-All 较高，但 OOS F1 下降，说明只看 Known 分类会掩盖 OOS 拒识退化。
4. 当前结果支持‘Known-only 表示适配改善单中心分数排序’，不支持‘训练后固定多中心自然有效’；StackOverflow 的 K=2 负结果仍需单独看作 union-risk 证据。
5. Trainable 与 Frozen、MOGB 组件结果的 seed 数和监督条件不同，图表用于机制对照，不构成无条件 SOTA 排名。

## 5. 图表

- `figures/minilm_trainable_kir_sweep_v1/trainable_vs_frozen_kir.png`：三数据集 KIR 曲线。
- `figures/minilm_trainable_kir_sweep_v1/trainable_known_oos_tradeoff_kir.png`：Known Recall/OOS F1 权衡。
- `figures/minilm_trainable_kir_sweep_v1/trainable_minus_frozen_kir_heatmap.png`：训练相对冻结的 KIR 热图。
- `figures/minilm_trainable_kir_sweep_v1/trainable_fair_component_context.png`：与同协议 Frozen/MOGB 组件的上下文比较。

## 6. 下一步

- 先把本轮 Trainable K=1 与现有 Frozen/MOGB 组件结果合并到同一分层总表；
- 再决定是否为 Trainable 表示运行 K=2 的跨 KIR 诊断；不把 K=2 直接当正式方法；
- 完整 Cascade 和 DCLOOS 仍需单独标注数据/监督合同，不能用本轮 Gate-only 数字代替。
