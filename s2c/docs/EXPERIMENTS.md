# s2c 实验登记与结果

这是当前唯一的实验结果入口。历史目录名（如 `v19`、`v20`、`v21`、`v22`）是 artifact provenance，不是新的实验文档体系。

## 先区分两种评价

### Gate-only

只评价 MiniLM Gate，回答多簇、表示碰撞、near-OOS 和支持区域几何问题。主要看 `OOS F1`、`AUROC`、`AUPR-OOS`、`FPR95`、`ID Recall`。

### 完整 Cascade

评价 `Gate → Router → Expert`，回答 Gate 改善是否传递到最终系统。主要看 `OOS F1`、`Known macro-F1`、`overall accuracy`、`ID Recall` 和错误阶段。

两种评价不能合并成一个“总 F1”。

## 唯一登记表

```text
configs/experiment_registry.yaml
```

登记分三层：

| 层 | 内容 | 当前用途 |
| --- | --- | --- |
| `pipeline` | 代表性修复和三 seed 完整 Cascade | 系统级验证 |
| `gate_mechanism` | KIR/K、随机分簇、MiniLM、样本效率和机制分析 | 论文机制主体 |
| `external_validation` | hard-negative 和 MOGB 协议审计 | 外部有效性/缺口 |

审计命令：

```bash
/home/bo/anaconda3/envs/bo/bin/python tools/analysis/audit_experiment_registry.py
```

它检查入口、manifest、summary 和 unit count，并写出 `configs/active_entrypoints.json`、`configs/unreferenced_entrypoints_report.json` 及 artifact 下的 `study_closeout/pipeline_freeze_manifest.json`。不会重跑模型，也不会覆盖实验结果。

## 完整 Cascade 主结果

来源：

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/
```

`evaluations/matrix_manifest.json` 是完成性事实来源：`expected_unit_count=36`、`completed_unit_count=36`、`missing_components=[]`。

### 三 seed 均值 ± 标准差

| Dataset | Gate | OOS F1 | Known macro-F1 | Accuracy | ID Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| CLINC150 | Frozen K=1 | 0.8802 ± 0.0100 | 0.7850 ± 0.0036 | 0.8306 ± 0.0116 | 0.7367 |
| CLINC150 | CE-Recon selected-K | 0.9000 ± 0.0075 | 0.7968 ± 0.0087 | 0.8579 ± 0.0088 | 0.7330 |
| CLINC150 | Best controlled baseline | 0.9084 ± 0.0107 | 0.8518 ± 0.0193 | 0.8677 ± 0.0127 | 0.8490 |
| BANKING77-OOS | Frozen K=1 | 0.8482 ± 0.0144 | 0.8246 ± 0.0189 | 0.7674 ± 0.0158 | 0.8883 |
| BANKING77-OOS | CE-Recon selected-K | 0.8948 ± 0.0069 | 0.7921 ± 0.0181 | 0.8270 ± 0.0096 | 0.7937 |
| BANKING77-OOS | Best controlled baseline | 0.8417 ± 0.0498 | 0.8401 ± 0.0186 | 0.7657 ± 0.0606 | 0.8840 |
| StackOverflow | Frozen K=1 | 0.7902 ± 0.0463 | 0.8398 ± 0.0147 | 0.7673 ± 0.0379 | 0.8322 |
| StackOverflow | CE-Recon selected-K | 0.8762 ± 0.0278 | 0.8668 ± 0.0088 | 0.8485 ± 0.0322 | 0.8615 |
| StackOverflow | Best controlled baseline | 0.8324 ± 0.0473 | 0.8562 ± 0.0161 | 0.8072 ± 0.0318 | 0.8549 |

数值来自 `cascade_gate_summary.csv`；表中只保留三 seed、同一 dataset/seed 下共用下游组件的比较。

## 为什么 Banking77 的 CE-Recon Known 指标下降

这不是 Expert 没修好，而是 Gate 工作点的明确取舍。相对同 seed 的 Frozen K=1，CE-Recon 的 Banking77 三 seed 平均：

```text
OOS F1                 +0.0466
OOS false-accept rate  -0.1006
ID Recall              -0.0947
Known macro-F1         -0.0325
Known false-reject     +0.0947
```

也就是说，CE-Recon 拒绝了更多 Known 样本，从而显著减少 OOS 被接受；被 Gate 拦截的 Known 不再进入 Expert，因此 Known macro-F1/ID Recall 下降。这是 OOS 拒识与 Known 保留之间的 operating-point trade-off，不是继续重训下游模型就能自动消除的 bug。逐 seed 的配对数字在：

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/cascade_banking_tradeoff.csv
```

StackOverflow 则显示 CE-Recon 同时改善 OOS F1、Known macro-F1 和 accuracy；Frozen 多簇退化仍然是 Gate 支持结构的负结果。

## 错误分解

样本级阶段限定为：

```text
correct_oos_rejection
oos_accepted_by_gate
known_rejected_by_gate
known_wrong_domain
known_wrong_expert
correct_known_prediction
```

原始逐单元表是 `cascade_error_decomposition.csv`，三 seed 聚合表是 `cascade_error_decomposition_summary.csv`。只用 `known_wrong_domain` 和 `known_wrong_expert` 判断下游质量；前两项是 Gate 的 OOS/ID 决策。

## 其他研究目录

| Artifact 根目录 | 研究问题 | 关键入口 |
| --- | --- | --- |
| `cluster_separability_v19/` | KIR × K × distance 和受控 Baseline | `tools/experiments/cluster_separability` |
| `cluster_separability_v20/` | 随机分簇、selected-K、near-OOS、效率 | `v20_*` 脚本 |
| `cluster_separability_v21/` | MiniLM 邻域纯度、语义间隔、表示适配 | `v21_*` 脚本 |
| `minilm_semantic_collision_v22/` | 表示碰撞/过覆盖与语义结构 | artifact-only 历史结果 |
| `sample_efficiency/` | 每意图样本量与多簇稳定性 | artifact-only |
| `mechanism_analysis/` | 碰撞、碎片化与 near-OOS 的描述性关系 | artifact-only |
| `external_validation/hard_negative_oos/` | zero-adjustment 外部困难 OOS | artifact-only |
| `external_validation/mogb_*` | MOGB 协议审计 | 没有公平性能数字 |

## 解释边界

- 不再扩展 KIR25/75、更多 K、更多普通 Baseline 或新的 MiniLM 损失。
- `selected K=1` 是 validation 的有效结论，不是失败。
- MOGB 审计通过不等于 MOGB 已复现。
- 任何新结果必须使用新的 artifact 根目录；不覆盖上述历史结果。
