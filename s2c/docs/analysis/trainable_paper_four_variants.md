# Trainable Gate 完整系统：论文四变体消融

新评估完成 36/36 个单元。三个数据集 × KIR=.25/.50/.75 × 四个变体，seed=42。主表使用论文布局：每个数据集两列 Acc、OOS F1；下表所有已填数字均来自本轮运行。

| KIR | Variant | CLINC Acc | CLINC OOS F1 | StackOverflow Acc | StackOverflow OOS F1 | BANKING77-OOS Acc | BANKING77-OOS OOS F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.25 | Ours | 91.02 | 95.21 | 92.17 | 95.99 | 93.41 | 96.57 |
| 0.25 | Without Gate | 73.00 | 81.75 | 66.84 | 74.95 | 82.72 | 89.81 |
| 0.25 | Cascade-MiniLM | 92.05 | 95.21 | 92.74 | 95.99 | 93.58 | 96.57 |
| 0.25 | Cascade-SmolLM | 80.85 | 88.09 | 40.33 | 45.10 | 51.15 | 64.41 |
| 0.50 | Ours | 87.93 | 91.64 | 88.76 | 91.85 | 85.51 | 91.46 |
| 0.50 | Without Gate | 73.00 | 76.03 | 78.36 | 80.16 | 82.43 | 88.75 |
| 0.50 | Cascade-MiniLM | 89.67 | 91.64 | 89.80 | 91.85 | 85.93 | 91.46 |
| 0.50 | Cascade-SmolLM | 67.65 | 69.71 | 44.41 | 6.38 | 47.89 | 55.38 |
| 0.75 | Ours | 79.82 | 81.33 | 83.04 | 78.45 | 81.47 | 88.49 |
| 0.75 | Without Gate | 71.65 | 61.76 | 77.40 | 65.94 | 76.00 | 82.32 |
| 0.75 | Cascade-MiniLM | 81.89 | 81.33 | 84.76 | 78.45 | 82.70 | 88.49 |
| 0.75 | Cascade-SmolLM | 68.44 | 55.02 | 64.86 | 0.13 | 45.88 | 42.07 |

## 四行实际含义

- Ours：当前 Trainable MiniLM Gate + 固定 SmolLM LoRA Router/Expert，逐数据集/KIR使用此前已确定的主配置，见各cell config.json。
- Without Gate：移除几何 Gate，所有样本进入原 Router/Expert；用下游 Expert 意图置信度拒识。这遵循原论文执行代码的 intent_confidence 分支；论文文字的 Router rejection 在单域数据中没有可用置信信息。
- Cascade-MiniLM：保留当前 Trainable MiniLM Gate，复用论文 MiniLM cascade 的域分类头和域内意图分类头（Known train 上 logistic regression），替换 SmolLM 下游。表示取当前 Trainable MiniLM 的输出；三阶段均为 MiniLM。
- Cascade-SmolLM：Gate改为 SmolLM mean-pooled/L2-normalized 意图原型余弦分数，复用论文原型 Gate 实现；下游仍是相同 SmolLM Router/Expert。CLINC使用router backbone；单域常量router没有模型参数，使用预训练SmolLM backbone。

Without Gate和Cascade-SmolLM的拒识分数阈值在验证集上按完整macro F1选择，搜索分数阈值为0.20至0.95，步长0.05。所有阈值与验证集分数单独保存。Ours/Cascade-MiniLM共用已确定的Gate及边界，所以OOS F1相同是设计预期，Known分类和Acc可以不同。

本表属于当前H1同数据、同下游条件下的结构/模型替换实验。旧论文四行历史数字未参与本表填充或参数选择。当前主配置的中心数和边界未被强制改回论文K=2。

## 补充完整指标

| Dataset | KIR | Variant | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance |
|---|---:|---|---:|---:|---:|---:|---:|
| banking77_oos | 0.25 | Ours | 65.77 | 96.57 | 93.41 | 61.88 | 1.89 |
| banking77_oos | 0.25 | Without Gate | 51.59 | 89.81 | 82.72 | 70.83 | 15.33 |
| banking77_oos | 0.25 | Cascade-MiniLM | 67.55 | 96.57 | 93.58 | 61.88 | 1.89 |
| banking77_oos | 0.25 | Cascade-SmolLM | 26.83 | 64.41 | 51.15 | 75.42 | 50.94 |
| banking77_oos | 0.50 | Ours | 66.71 | 91.46 | 85.51 | 69.80 | 7.47 |
| banking77_oos | 0.50 | Without Gate | 65.24 | 88.75 | 82.43 | 72.80 | 13.18 |
| banking77_oos | 0.50 | Cascade-MiniLM | 68.89 | 91.46 | 85.93 | 69.80 | 7.47 |
| banking77_oos | 0.50 | Cascade-SmolLM | 40.44 | 55.38 | 47.89 | 83.40 | 59.64 |
| banking77_oos | 0.75 | Ours | 70.22 | 88.49 | 81.47 | 80.72 | 11.56 |
| banking77_oos | 0.75 | Without Gate | 67.47 | 82.32 | 76.00 | 79.47 | 21.52 |
| banking77_oos | 0.75 | Cascade-MiniLM | 73.77 | 88.49 | 82.70 | 80.72 | 11.56 |
| banking77_oos | 0.75 | Cascade-SmolLM | 51.31 | 42.07 | 45.88 | 91.84 | 72.07 |
| clinc150 | 0.25 | Ours | 72.86 | 95.21 | 91.02 | 71.05 | 2.27 |
| clinc150 | 0.25 | Without Gate | 53.27 | 81.75 | 73.00 | 67.81 | 25.05 |
| clinc150 | 0.25 | Cascade-MiniLM | 78.40 | 95.21 | 92.05 | 71.05 | 2.27 |
| clinc150 | 0.25 | Cascade-SmolLM | 61.19 | 88.09 | 80.85 | 76.23 | 16.40 |
| clinc150 | 0.50 | Ours | 82.56 | 91.64 | 87.93 | 85.96 | 7.20 |
| clinc150 | 0.50 | Without Gate | 70.95 | 76.03 | 73.00 | 80.67 | 30.46 |
| clinc150 | 0.50 | Cascade-MiniLM | 87.04 | 91.64 | 89.67 | 85.96 | 7.20 |
| clinc150 | 0.50 | Cascade-SmolLM | 67.40 | 69.71 | 67.65 | 85.02 | 40.95 |
| clinc150 | 0.75 | Ours | 78.11 | 81.33 | 79.82 | 74.02 | 3.50 |
| clinc150 | 0.75 | Without Gate | 77.49 | 61.76 | 71.65 | 89.08 | 47.66 |
| clinc150 | 0.75 | Cascade-MiniLM | 81.99 | 81.33 | 81.89 | 74.02 | 3.50 |
| clinc150 | 0.75 | Cascade-SmolLM | 75.28 | 55.02 | 68.44 | 94.29 | 58.64 |
| stackoverflow | 0.25 | Ours | 80.10 | 95.99 | 92.17 | 83.37 | 2.60 |
| stackoverflow | 0.25 | Without Gate | 60.79 | 74.95 | 66.84 | 82.03 | 36.48 |
| stackoverflow | 0.25 | Cascade-MiniLM | 82.40 | 95.99 | 92.74 | 83.37 | 2.60 |
| stackoverflow | 0.25 | Cascade-SmolLM | 42.02 | 45.10 | 40.33 | 75.95 | 68.55 |
| stackoverflow | 0.50 | Ours | 85.24 | 91.85 | 88.76 | 85.47 | 2.74 |
| stackoverflow | 0.50 | Without Gate | 76.91 | 80.16 | 78.36 | 78.72 | 18.89 |
| stackoverflow | 0.50 | Cascade-MiniLM | 87.47 | 91.85 | 89.80 | 85.47 | 2.74 |
| stackoverflow | 0.50 | Cascade-SmolLM | 60.66 | 6.38 | 44.41 | 97.66 | 96.63 |
| stackoverflow | 0.75 | Ours | 85.04 | 78.45 | 83.04 | 83.01 | 2.60 |
| stackoverflow | 0.75 | Without Gate | 82.09 | 65.94 | 77.40 | 81.23 | 23.15 |
| stackoverflow | 0.75 | Cascade-MiniLM | 87.55 | 78.45 | 84.76 | 83.01 | 2.60 |
| stackoverflow | 0.75 | Cascade-SmolLM | 74.97 | 0.13 | 64.86 | 99.89 | 99.93 |

Known F1直接在全测试集上对Known类求macro F1，保留真实OOS引起的Known false positives。全部四行同一数据/KIR/seed；这是单seed结果，不是三个seed平均。

[源表](../../results/analysis/trainable_paper_four_variants/summary.csv)
