# s2c 运行手册

本手册只记录活动协议 `protocol_v2_textoir_v1` 的检查、审计和可恢复实验入口。历史协议不在此
运行；其资料保留在审计或 archive 目录。

## 环境与准入

```bash
cd /home/bo/bo01/llmpy311/s2c
python --version
python -c "import torch, sentence_transformers; print(torch.cuda.is_available())"
python tools/maintenance/check_data_tracking.py
```

当前 Gate smoke 使用本地 `assets/models/all-MiniLM-L6-v2`。不允许运行时从 `textoir/data`、网络或
旧 v19 路径读取输入。

## E0：验证数据契约

```bash
python -m protocol_v2.data.validate_protocol --require-views --require-exports
python tools/maintenance/check_data_tracking.py
```

E0 通过条件：三个 source/canonical、165 registry、165 views、990 exports 完整；StackOverflow
严格是 20,000 行/20 标签；所有 export 指向同一个 registry；完整数据未被 Git 跟踪。

## E1：36 单元 Gate smoke

```bash
python -m protocol_v2.experiments.runner \
  --config configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml \
  --resume --shard-name e1_smoke
python -m protocol_v2.experiments.verify \
  --config configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml \
  --require-complete
python -m protocol_v2.experiments.summarize \
  --config configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml \
  --output ../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e1_gate_smoke.csv
```

## E2：1,650 单元密集 Gate 网格

```bash
python -m protocol_v2.experiments.plan \
  --config configs/experiments/protocol_v2_textoir_v1/gate_core_dense.yaml
python -m protocol_v2.experiments.runner \
  --config configs/experiments/protocol_v2_textoir_v1/gate_core_dense.yaml \
  --resume --shard-name e2_core
```

Runner 每个完成/失败单元都更新 state JSON；`--resume` 只跳过 config hash 相同的完整 run。E2
已经完成且只读引用，E2 run/config/data/cache 不得修改。

## E3：多中心机制诊断（当前阶段）

先只检查计划，不写正式 run：

```bash
PYTHONPATH=src python scripts/experiments/plan_e3_mechanisms.py
PYTHONPATH=src python -m protocol_v2.experiments.mechanism_runner partition-control \
  --config configs/experiments/protocol_v2_textoir_v1/e3_partition_control.yaml --dry-run
```

完成 E3-0 provenance 冻结后，运行 E3-A：

```bash
PYTHONPATH=src python -m protocol_v2.experiments.mechanism_runner partition-control \
  --config configs/experiments/protocol_v2_textoir_v1/e3_partition_control.yaml --resume
PYTHONPATH=src python scripts/experiments/verify_e3_mechanisms.py \
  --partition-config configs/experiments/protocol_v2_textoir_v1/e3_partition_control.yaml \
  --diagnostic-config configs/experiments/protocol_v2_textoir_v1/e3_cluster_diagnostics.yaml \
  --check-equivalence
```

E3-A 完整性通过后再运行稳定性/覆盖诊断：

```bash
PYTHONPATH=src python -m protocol_v2.experiments.mechanism_runner cluster-diagnostics \
  --config configs/experiments/protocol_v2_textoir_v1/e3_cluster_diagnostics.yaml --resume
PYTHONPATH=src python scripts/experiments/summarize_e3_mechanisms.py
```

E3 已完成 720 个 partition-control 单元和 180 个 train/calibration-only 诊断组。收口复核：

```bash
PYTHONPATH=src python scripts/experiments/verify_e3_mechanisms.py \
  --partition-config configs/experiments/protocol_v2_textoir_v1/e3_partition_control.yaml \
  --diagnostic-config configs/experiments/protocol_v2_textoir_v1/e3_cluster_diagnostics.yaml \
  --require-complete --check-equivalence
PYTHONPATH=src python scripts/experiments/summarize_e3_mechanisms.py
```

摘要位于 `../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/summaries/`。
E3 不启动 ADB、DA-ADB、MOGB、表示学习、边界网格或完整 Cascade；E4--E7 仍未启动。

## R1：Geometry-Preserving CE-Recon pilot

R1 已完成且只写入独立目录；不得重复 E2/E3，也不自动进入 R1_full：

```bash
python scripts/experiments/plan_r1_geometry_preserving.py \
  --output ../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/plans/R1_plan.json
python tools/maintenance/check_research_state.py \
  --plan ../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/plans/R1_plan.json
python scripts/experiments/run_r1_geometry_preserving.py freeze \
  --config configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving.yaml
python scripts/experiments/run_r1_geometry_preserving.py select-beta \
  --config configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving.yaml
python scripts/experiments/run_r1_geometry_preserving.py pilot \
  --config configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving.yaml
```

R1 训练只使用 Known train，checkpoint 只用 Known calibration macro-F1 选择；beta 选择不读取
OOS。Pilot 收口文件为 `R1_CLOSEOUT.md`、`R1_method_decision.md`、`R1_gate_summary.csv` 和
`R1_geometry_analysis.csv`。运行前后必须执行 `check_research_state.py`，并确认 E2/E3 根目录未被写入。

## 审计与公开结果

```bash
python tools/audit/generate_protocol_v2_implementation_report.py \
  --dataset-version protocol_v2_textoir_v1
python tools/analysis/audit_experiment_registry.py --check-only
python tools/maintenance/export_public_results.py --verify
git diff --check
```

`results/` 只能含 `configs/public_results.yaml` 显式允许的小型证据；完整语料和模型绝不导出。
