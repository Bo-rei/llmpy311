# 结构必要性 + 异构骨干价值：正式执行计划

## 1. 目标

这套消融只回答两个问题，不再引入 gate family、prototype family 或 decision family 作为主线：

1. `Gate -> Router -> Expert` 三层结构是否必要。
2. `MiniLM` 做 Gate、`SmolLM` 做 Router/Expert 的异构骨干组合是否有价值。

## 2. 统一锚点

所有后续结果都以当前可复现的完整系统为锚点：

- `full_pipeline_group2_ref`

锚点结果来自当前稳定 pipeline 配置，不使用旧的 13 实验矩阵作为主证据。

## 3. 主实验矩阵

### 3.1 结构必要性表

| 组别 | 说明 | 作用 |
|------|------|------|
| `full_pipeline_group2_ref` | 完整三级结构 | 主锚点 |
| `no_gate_confidence` | 去掉 Gate，仅保留 Router confidence OOS | 验证 Gate 是否必要 |
| `single_stage_minilm` | MiniLM + LogisticRegression 单阶段控制 | flat 架构对照 |
| `single_stage_smollm` | 单阶段 SmolLM 控制 | flat 架构对照 |

### 3.2 异构骨干表

| 组别 | 说明 | 作用 |
|------|------|------|
| `full_pipeline_group2_ref` | MiniLM Gate + SmolLM Router/Expert | 混合骨干锚点 |
| `cascade_minilm` | MiniLM Gate + MiniLM Router/Expert | 保持级联结构的纯 MiniLM 对照 |

## 4. 诊断项

以下结果只保留为诊断或边界分析，不进入主结论：

- `no_gate_random`
- `oracle_oos`

## 5. 执行顺序

推荐按下面顺序执行：

1. 先在 validation 上搜索 `router_confidence` 阈值。
2. 再跑 `full_pipeline_group2_ref`。
3. 再跑 `no_gate_confidence`。
4. 再跑 `single_stage_minilm`。
5. 再跑 `single_stage_smollm`。
6. 再跑 `cascade_minilm`。
6. 如有需要，再补 `no_gate_random` 和 `oracle_oos` 作为诊断。

## 6. 协议要求

- 数据必须固定为 `data/current -> v19`。
- 所有阈值只能在 validation 上选择。
- 每个实验必须保留 `eval_results.json`、`predictions.json`、`run_manifest.json`。
- 主结论只比较同一协议下的 delta，不跨协议拼接因果。

## 7. 输出目录建议

建议总输出目录为：

`outputs/experiments/pipeline/ablations/structure_backbone_v19/<exp_id>/`

建议分层如下：

- `00_threshold_validation/`
- `01_structure/full_pipeline_group2_ref/`
- `01_structure/no_gate_confidence/`
- `02_backbone/cascade_minilm/`
- `03_diagnostics/no_gate_random/`
- `03_diagnostics/oracle_oos/`

## 8. 结果写法

报告时只写两类结论：

- 结构结论：`full_pipeline_group2_ref` vs `no_gate_confidence` vs `flat_*`
- 骨干结论：`MiniLM + SmolLM` 混合骨干 vs `cascade_minilm`

不写 gate family，不写 prototype/decision 主表，不把历史失败矩阵当主证据。
