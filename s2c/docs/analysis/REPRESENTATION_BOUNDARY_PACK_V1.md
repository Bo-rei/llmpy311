# 表示训练与多中心边界交互分析

更新时间：2026-08-06  
活动协议：`protocol_v2_textoir_v1`  
证据范围：已有 `representation_fixed_results.csv` 与 `representation_geometry_summary.csv`，不新增训练。

## 结论先行

已有结果支持一个清晰的分层结论：表示训练可以改善单中心 OOS 检测，但不能自动使固定多中心边界有效。StackOverflow 是最明确的反例：CE 和 SupCon 的 K=1 OOS F1 分别为 88.13% 和 89.63%，但切换到 K=2 后分别降到 73.44% 和 71.90%。同时 Known Recall 分别上升到 90.46% 和 93.32%，说明退化主要表现为接受区域过宽，而不是已知样本覆盖不足。

因此，不能把“更有判别性的 MiniLM 表示”直接解释为“更适合多中心 OOS 边界”。当前证据更支持：表示学习和边界几何是两个需要分别评价的因素。

## 1. K=1 与 K=2 的主结果

下表为三个 data seed 的均值；`false_accept` 使用现有结果文件中的 OOS 接受率口径。完整逐 seed 数据见 [`representation_k1_k2_summary.csv`](../../results/analysis/representation_boundary_pack_v1/representation_k1_k2_summary.csv)。

| 数据集 | 表示 | K=1 OOS F1 | K=2 OOS F1 | K=2−K=1 | K=1 Known Recall | K=2 Known Recall | K=2−K=1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Banking77 | Frozen | 84.82 | 85.57 | +0.75 | 88.83 | 86.10 | −2.73 |
| Banking77 | CE | 91.15 | 88.28 | −2.87 | 82.07 | 80.20 | −1.87 |
| Banking77 | SupCon | 90.41 | 87.81 | −2.60 | 85.30 | 81.40 | −3.90 |
| CLINC150 | Frozen | 88.02 | 88.07 | +0.04 | 73.67 | 70.00 | −3.67 |
| CLINC150 | CE | 89.73 | 89.46 | −0.28 | 72.96 | 71.21 | −1.75 |
| CLINC150 | SupCon | 89.86 | 89.47 | −0.39 | 75.29 | 72.46 | −2.83 |
| StackOverflow | Frozen | 79.02 | 72.80 | −6.22 | 83.22 | 87.25 | +4.03 |
| StackOverflow | CE | 88.13 | 73.44 | −14.69 | 82.34 | 90.46 | +8.12 |
| StackOverflow | SupCon | 89.63 | 71.90 | −17.74 | 83.59 | 93.32 | +9.73 |

## 2. Near-OOS 结果

Near/medium/far 分桶来自已有表示实验，不能替代正式测试协议中的统一 OOS 结果；这里仅作为机制诊断。

| 数据集 | 表示 | K=1 Near-OOS F1 | K=2 Near-OOS F1 | 变化 |
|---|---|---:|---:|---:|
| Banking77 | Frozen | 11.14 | 36.20 | +25.06 |
| Banking77 | CE | 39.00 | 47.02 | +8.02 |
| Banking77 | SupCon | 25.41 | 48.63 | +23.22 |
| CLINC150 | Frozen | 45.67 | 48.47 | +2.81 |
| CLINC150 | CE | 53.46 | 53.76 | +0.30 |
| CLINC150 | SupCon | 52.39 | 53.19 | +0.80 |
| StackOverflow | Frozen | 39.50 | 29.70 | −9.80 |
| StackOverflow | CE | 46.34 | 35.65 | −10.69 |
| StackOverflow | SupCon | 55.55 | 23.28 | −32.27 |

StackOverflow 上，表示训练虽然提高了 K=1 的整体 OOS F1 和近邻结构指标，但 K=2 会明显破坏 near-OOS 拒识。这表明 near-OOS 不是只由类内紧致度决定，还受到多个接受区域的并集和边界方向影响。

## 3. 表示几何与检测性能不能等价

现有几何汇总（均值）显示：

| 表示 | Linear probe Val Macro-F1 | Purity@10 | Relative separation | Effective rank | Same-intent alignment |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.9349 | 0.8672 | 0.5444 | 122.92 | 0.4355 |
| CE | 0.9438 | 0.9328 | 0.9440 | 34.24 | 0.8908 |
| SupCon | 0.9469 | 0.9247 | 0.8393 | 47.42 | 0.7832 |

CE/SupCon 提高了已知意图可分性并显著增强类内对齐，但 effective rank 同时降低。更关键的是，这些几何改善没有在 StackOverflow 的 K=2 OOS F1 上兑现。当前证据不能支持“分类表示越紧，多个球越安全”的推论。

## 4. 机制解释

1. K=1 时，训练表示让类中心和 Known 样本的距离排序更稳定，因此 OOS F1 提升。
2. K=2 时，类内压缩和局部中心拟合会让多个局部半径覆盖更多边缘区域。
3. 评分采用多个局部区域的接受并集；任意一个子球接受即可通过 Gate，因此 OOS false acceptance 可能上升。
4. StackOverflow 的技术短文本具有词汇和主题交叉，KMeans 子簇方向不一定与 OOS 分离方向一致；SupCon 只优化 Known 类别间隔，也没有直接看到 OOS 风险。

## 5. 与已有多中心结论的关系

这组结果与 E2/E3 的数据集差异一致：Banking77 的多中心收益是条件性的，CLINC150 接近中性，StackOverflow 固定多中心退化。它进一步排除了“只要把 MiniLM 训练得更判别，就能统一修复多中心”的解释。

## 6. 当前可作为论文/汇报证据的范围

- 可以报告：表示训练改善 K=1；不同表示的 K=1→K=2 效应不同；StackOverflow 存在明显 Known Recall–OOS F1 冲突。
- 不能报告：CE/SupCon 已经证明多中心普遍有效；effective rank 提升/下降可以单独预测 OOS 性能；near-OOS 已被解决。
- 本文件是已有结果的分析汇总，不包含新的训练、调参或测试集选模。

## 7. 原始证据

- [`representation_k1_k2_summary.csv`](../../results/analysis/representation_boundary_pack_v1/representation_k1_k2_summary.csv)
- [`representation_k2_minus_k1.csv`](../../results/analysis/representation_boundary_pack_v1/representation_k2_minus_k1.csv)
- [`representation_geometry.csv`](../../results/analysis/representation_boundary_pack_v1/representation_geometry.csv)
- [`representation_oos_f1_k1_k2.png`](../../figures/representation_boundary_pack_v1/representation_oos_f1_k1_k2.png)
- [`representation_near_oos_f1_k1_k2.png`](../../figures/representation_boundary_pack_v1/representation_near_oos_f1_k1_k2.png)
- [`representation_known_recall_k1_k2.png`](../../figures/representation_boundary_pack_v1/representation_known_recall_k1_k2.png)
- [`representation_geometry_summary.png`](../../figures/representation_boundary_pack_v1/representation_geometry_summary.png)
