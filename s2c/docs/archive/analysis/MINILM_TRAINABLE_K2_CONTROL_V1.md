# Trainable MiniLM K=1/K=2 跨数据集控制

本阶段只评价已经训练完成的 Trainable MiniLM checkpoint，不重新训练、不重新划分数据，也不使用测试集选择 checkpoint、K、半径或阈值。CLINC150/Banking77 使用 `minilm_trainable_control_v1` checkpoint；StackOverflow 读取已经完成的同协议 RACAL-v1 K=1/K=2 配对结果。

## 协议与完整性

- 数据集：CLINC150、Banking77、StackOverflow；KIR=0.50；seed=[13, 42, 87]。
- 表示：Trainable MiniLM last-2 layers + projection；K=1 与 K=2 共用同一 seed 对应 checkpoint。
- 距离/边界：对角 Mahalanobis、`mean + 1.0*std`、threshold=1.0、固定 partition seed=42。
- 配对行数：9；每个数据集 3 个 seed；CLINC/Banking 新增 6 个评价单元，StackOverflow 为既有只读结果。
- 所有新增 run 的 `k1_replay_max_abs_delta` 均为 0（浮点容差内）；历史 artifacts 未覆盖。

## 逐数据集均值（0-1 指标）

| 数据集 | K | OOS F1 | F1-All | F1-K | Known Recall | False Acceptance | AUROC | AUPR-OOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 1 | 0.8477 ± 0.0071 | 0.8231 ± 0.0083 | 0.8224 ± 0.0083 | 0.8191 ± 0.0017 | 0.1346 ± 0.0122 | 0.9080 ± 0.0064 | 0.8922 ± 0.0073 |
| banking77 | 2 | 0.8490 ± 0.0038 | 0.8176 ± 0.0018 | 0.8168 ± 0.0018 | 0.7996 ± 0.0004 | 0.1184 ± 0.0071 | 0.9071 ± 0.0017 | 0.8893 ± 0.0118 |
| clinc150 | 1 | 0.9043 ± 0.0062 | 0.8190 ± 0.0101 | 0.8179 ± 0.0101 | 0.7427 ± 0.0032 | 0.0361 ± 0.0106 | 0.9513 ± 0.0061 | 0.9619 ± 0.0068 |
| clinc150 | 2 | 0.9015 ± 0.0042 | 0.8036 ± 0.0126 | 0.8023 ± 0.0127 | 0.7101 ± 0.0109 | 0.0242 ± 0.0034 | 0.9548 ± 0.0030 | 0.9649 ± 0.0036 |
| stackoverflow | 1 | 0.8671 ± 0.0096 | 0.8565 ± 0.0037 | 0.8555 ± 0.0033 | 0.8392 ± 0.0039 | 0.1114 ± 0.0202 | 0.9175 ± 0.0107 | 0.8878 ± 0.0248 |
| stackoverflow | 2 | 0.6765 ± 0.0753 | 0.7681 ± 0.0154 | 0.7772 ± 0.0097 | 0.9362 ± 0.0036 | 0.4526 ± 0.0957 | 0.8945 ± 0.0166 | 0.8754 ± 0.0302 |

## K=2 相对 K=1 的配对差值

| 数据集 | OOS F1 Δ | F1-All Δ | F1-K Δ | Known Recall Δ | False Acceptance Δ | AUROC Δ | 方向（OOS F1） |
|---|---:|---:|---:|---:|---:|---:|---|
| banking77 | +0.13 pp | -0.55 pp | -0.56 pp | -1.95 pp | -1.62 pp | -0.10 pp | K=2较好 |
| clinc150 | -0.28 pp | -1.54 pp | -1.56 pp | -3.26 pp | -1.20 pp | +0.35 pp | K=1较好 |
| stackoverflow | -19.06 pp | -8.85 pp | -7.83 pp | +9.70 pp | +34.11 pp | -2.30 pp | K=1较好 |

## 当前解释边界

- 该阶段只回答“同一 Trainable MiniLM 表示下，固定 K=2 是否仍优于 K=1”。它不是自适应 K，也不是完整 Cascade 结果。
- K=2 若提高 Known Recall 但降低 OOS F1，说明新增局部球扩大了接受区域；不能只看 Known Recall 判断多中心有效。
- StackOverflow 的 K=2 结果来自既有 RACAL-v1 Stage 2，不与本阶段 CLINC/Banking 的新 run 混写；来源列保留了这一差异。
- 需要把这些配对结果与 Frozen、CE-Recon、MOGB、DCLOOS 等方法放到同一 split/指标协议后，才可作 baseline 排名。

## 机器可读文件

- `per_seed.csv`：逐 seed 的 K=1/K=2 配对结果。
- `mean_std.csv`：各数据集、各 K 的均值和标准差。
- `delta_summary.csv`：K=2−K=1 的配对汇总。
