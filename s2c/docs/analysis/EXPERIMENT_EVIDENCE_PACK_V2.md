# 实验优先证据包 V2：多 KIR、多方法与误差权衡

更新时间：2026-08-06  
本报告只做实验结果整理和机制分析，不提出新的模型，也不把不同协议的结果混成一个 SOTA 排名。

## 1. 本轮做了什么

在已有 E2/E3、MOGB fair matrix、RACAL/Trainable MiniLM 和 Cascade 产物基础上，新增了一个
analysis-only evidence pack：

- 读取 MOGB fair matrix 的 3 数据集×3 KIR×5 seeds×6 frozen-component 方法；
- 对每个 `dataset×KIR×seed` 做 paired comparison，以 Single centroid 为参考；
- 对 OOS F1、F1-All、Known Recall、false acceptance 计算均值差、95% paired bootstrap CI、
  win/tie/loss；bootstrap seed=20260725、重采样 10,000 次；
- 将当前 Trainable MiniLM K=1 的 27 个 KIR 单元作为单独的上下文层加入图表；
- 生成每个数据集的 OOS F1-KIR 曲线、OOS F1–Known Recall 权衡图、差值热图和 Cascade 错误分解图。

可复现入口：

```bash
cd /home/bo/bo01/llmpy311/s2c
python tools/analysis/build_experiment_evidence_pack_v2.py
```

结果：`results/analysis/experiment_evidence_pack_v2/`；图：`figures/experiment_evidence_pack_v2/`。

## 2. 同协议方法的主要结果

### StackOverflow

| 方法 | KIR=.25 | KIR=.50 | KIR=.75 | Known Recall（KIR=.50） |
|---|---:|---:|---:|---:|
| Single centroid | 89.5 | 76.5 | 65.6 | 87.1 |
| Fixed K=2 | 86.7 | 63.5 | 43.6 | 86.9 |
| Random partition | 89.1 | 75.9 | 65.0 | 87.6 |
| MOGB-MiniLM | 89.4 | 72.9 | 45.6 | 27.1 |
| MOGB partition + s2c boundary | 92.6 | 79.2 | 51.6 | 50.4 |
| s2c partition + MOGB boundary | 90.6 | 73.4 | 48.5 | 54.5 |
| Trainable MiniLM K=1 | 95.1 | 86.7 | 76.7 | 83.9 |

解释：在 StackOverflow，Trainable K=1 的 OOS F1 高于 Frozen 单中心和 MOGB fair 组件，且 Known
Recall 仍保持在 83.9%；MOGB partition+s2c boundary 虽然 KIR=.50 的 OOS F1 为 79.2%，但 Known
Recall 只有 50.4%，主要是通过拒绝大量 Known 样本换取 OOS 召回，不能只看 OOS F1。

### Banking77

Banking77 的 MOGB partition+s2c boundary 在 KIR=.25/.50 的 OOS F1 分别为 92.1/79.4，说明
动态粒球在该数据集存在条件性收益；但其 Known Recall 约 61.7/52.2%，明显低于单中心。当前
Trainable K=1 在 KIR=.50 为 OOS F1 84.8、Known Recall 81.9，整体权衡更平衡；KIR=.75 的
Trainable OOS F1 降至 69.5，说明高 KIR 仍是主要困难。

### CLINC150

CLINC150 上 MOGB fair 组件没有稳定优势：KIR=.50 时 Single centroid OOS F1 为 88.9，MOGB
partition+s2c boundary 为 85.6，且 Known Recall 从 78.8 降到 53.3。Trainable K=1 为 90.4、
Known Recall 74.3，表现为较好的 OOS/F1-All 权衡，而不是简单扩大拒绝区域。

## 3. 配对差值揭示的机制

在 KIR=.50、以 Single centroid 为参考的五 seed paired comparison 中：

| 数据集 | 方法 | Δ OOS F1 | Δ F1-All | Δ Known Recall | Δ False Accept |
|---|---|---:|---:|---:|---:|
| Banking77 | Fixed K=2 | +3.03pp | +1.65pp | -2.46pp | -5.79pp |
| Banking77 | MOGB partition+s2c | +6.97pp | -9.70pp | -32.83pp | -31.35pp |
| CLINC150 | Fixed K=2 | +0.26pp | -0.18pp | -3.62pp | -2.38pp |
| CLINC150 | MOGB partition+s2c | -3.39pp | -15.67pp | -25.41pp | -6.32pp |
| StackOverflow | Fixed K=2 | -13.02pp | -7.22pp | -0.26pp | +17.46pp |
| StackOverflow | MOGB partition+s2c | +2.70pp | -16.64pp | -36.76pp | -27.85pp |

这里 `false acceptance` 越低越好，因此 MOGB partition 的负差值表示 OOS 误接收减少，但同时
Known Recall 大幅下降。该结果支持以下判断：

1. MOGB 的局部划分可以收紧 OOS 接受区域；
2. 当前 MOGB fair 适配并没有保留足够 Known 覆盖；
3. Trainable K=1 的优势是表示/分数排序的平衡，而不是单纯把边界收窄；
4. StackOverflow 固定 K=2 的失败主要是多球并集导致误接收增加，而不是中心数量本身必然无效。

## 4. KIR 趋势

所有 frozen fair 方法在三个数据集上随 KIR 增大普遍下降；StackOverflow 的下降最陡：

- Single centroid：89.5 → 76.5 → 65.6；
- Fixed K=2：86.7 → 63.5 → 43.6；
- MOGB partition+s2c：92.6 → 79.2 → 51.6。

Trainable K=1 的 StackOverflow 曲线为 95.1 → 86.7 → 76.7，说明表示适配能改善排序但不能
消除高 KIR 的语义重叠。Banking77 的 Trainable 曲线 92.2 → 84.8 → 69.5，则显示训练过程
在高 KIR 下会牺牲部分 OOS 分离能力；这也是当前不能只报告 Known 分类或只报告低 KIR 的原因。

## 5. Cascade 误差分解

已有 `cascade_error_decomposition_summary.csv` 显示，完整 Cascade 的差异不能只看 Gate 的
OOS F1：

- StackOverflow 的 `frozen_k1` 同时有约 8.39% Known 被 Gate 拒绝、11.76% OOS 被 Gate 接受；
  `ce_recon_selected_k` 把 OOS 接受率降到约 5.53%，但 Known→OOS 与 Router/Expert 错误仍存在，
  因此最终 F1-All 不等同于 Gate-only OOS F1。
- CLINC150 的 `ce_recon_selected_k` 将 OOS 接受率降到约 1.81%，但 Known Gate rejection
  上升到约 10.92%；这是典型的 OOS/ID 覆盖权衡。
- Banking77 的 `ce_recon_selected_k` 比 `frozen_k1` 减少 OOS 接受，但 Known rejection 上升，
  说明更强 OOS 过滤不必然带来更好的完整分类。

因此，历史 `fulltex.tex` 的高系统级分数很可能来自 Gate、Router 和 Expert 的联合误差结构，
不能由当前单独训练 MiniLM 的 Gate 结果直接复现。

## 6. 当前能说“自己的方法比 MOGB 好”到什么程度

可以说：

- 在当前统一 Frozen MiniLM 协议下，MOGB 的两种 fair 组件适配经常以显著 Known Recall 损失
  换取 OOS F1；
- 在 StackOverflow 和 CLINC150，Trainable K=1 的 OOS/Known 权衡优于 MOGB fair 组件；
- 在 Banking77，MOGB partition 的 OOS F1 在部分 KIR 更高，但 Known Recall 代价很大；
- 当前自有方案最可靠的优势是“轻量 Known-only 表示适配下的单中心 Gate 平衡”，不是已经证明
  了自适应多中心普遍优于 MOGB。

不能说：

- 已经公平超过 MOGB 论文；
- 已经超过 DCLOOS；
- ADB/DA-ADB 的兼容单格数字可直接作为同协议排名；
- Trainable K=1 已经证明多中心机制成功。

## 7. 图表索引

- `figures/experiment_evidence_pack_v2/{dataset}_oos_f1_by_kir.png`：各方法 OOS F1-KIR 曲线；
- `figures/experiment_evidence_pack_v2/{dataset}_oos_known_tradeoff.png`：OOS F1–Known Recall 权衡；
- `figures/experiment_evidence_pack_v2/{dataset}_delta_oos_f1.png`：相对单中心的 OOS 差值热图；
- 同目录下另有 F1-All、Known Recall 和 False Accept 差值热图。
- `figures/experiment_evidence_pack_v2/{dataset}_cascade_error_decomposition.png`：Gate 拒绝 Known、
  接受 OOS、Router/Expert 错误分解。

## 8. 下一轮实验重点

1. 继续补齐同协议的 Trainable/Frozen/MOGB 逐 seed 对齐，而不是新增模型结构；
2. 对 StackOverflow 重点做逐样本 false acceptance、Known→OOS 和 OOS→Known 错误分解；
3. 对 Banking77 分析为什么 MOGB OOS F1 高但 Known Recall 低；
4. 对 CLINC150 分析近 OOS 与多中心收益为何较弱；
5. 最后再决定是否需要补同协议 DCLOOS/ADB/DA-ADB 条件对照；
6. 在这些分析完成前，不把任何一条兼容性结果称为最终 SOTA。
