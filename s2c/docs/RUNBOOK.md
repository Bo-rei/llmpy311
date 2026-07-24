# s2c 运行手册

本手册只保留环境检查、数据协议核验、实验登记审计和公开结果导出命令。`protocol_v2` TEXTOIR
candidate 已被来源裁决冻结；`protocol_v2_official_v1` 只允许 CLINC150 与 Banking77。历史
artifact 保持不覆盖。

## 环境检查

```bash
cd /home/bo/bo01/llmpy311/s2c
# 项目环境是 bo；未激活时也可把下面的 python 替换为该环境的绝对路径
conda activate bo
python --version
python -c "import numpy, pandas, yaml; print('runtime imports: ok')"
```

## 结果登记审计

```bash
python tools/analysis/audit_experiment_registry.py
```

该命令只检查当前登记表指向的入口、manifest、汇总文件和 unit count；它不会选择 K、
阈值或重跑模型。完整原始产物仍在 `../artifacts/s2c/`。

## 官方重建数据与已准入 E1 Gate-only smoke

```bash
S2C_DATASET_VERSION=protocol_v2_official_v1 \
  python -m s2c.data.validate_protocol --dataset clinc150 --dataset banking77 --seed 0 --kir 0.5 --require-views --require-exports
python tools/maintenance/check_data_tracking.py
```

全量 KIR/seed registry 已生成；其余 view/export 使用同一环境变量按需 materialize。不得 resume
TEXTOIR candidate grid，也不得对 StackOverflow 或 BANKING77-OOS 生成 embedding、训练或公平比较。

已登记的官方 E1 子矩阵只覆盖 CLINC150 和 Banking77（24 个 Gate-only 单元；不是三数据集
36-cell 矩阵）。可复验已完成单元：

```bash
export S2C_DATASET_VERSION=protocol_v2_official_v1
python scripts/experiments/verify_grid.py \
  --config configs/experiments/protocol_v2/smoke_gate.yaml \
  --dataset clinc150 --dataset banking77 --seed 42 \
  --kir 0.25 --kir 0.50 --kir 0.75 --require-complete
python scripts/experiments/summarize_grid.py \
  --config configs/experiments/protocol_v2/smoke_gate.yaml \
  --dataset clinc150 --dataset banking77 --seed 42 \
  --kir 0.25 --kir 0.50 --kir 0.75 --shard-name official_e1_admitted \
  --output ../artifacts/s2c/runs/protocol_v2_official_v1/summaries/official_e1_admitted.csv
```

该命令只验证/汇总现有结果；不要借此扩大到 StackOverflow、candidate run root 或完整 Cascade。

## TEXTOIR candidate 数据与 Gate（仅审计）

```bash
python -m s2c.data.validate_protocol --require-views --require-exports
python tools/maintenance/check_data_tracking.py
python -m s2c.experiments.plan --config configs/experiments/protocol_v2/smoke_gate.yaml
```

正式 runner 命令已由 `configs/data/protocol_v2_admission.json` fail closed。先完成
`docs/DATASETS.md` 指定的官方 raw 重建和新的数据裁决，才可重新登记新的 run root；不得
resume 当前 candidate grid。详见 `DATA_PROTOCOL_V2.md` 与 `REPRODUCIBILITY.md`。

该段只适用于 `protocol_v2` candidate。官方重建版本的 dataset-level admission 已单独
控制；仍必须先 materialize 对应 view/export 并通过验证，且绝不准入 StackOverflow。

## protocol_v2 E4 外部 Baseline 可运行性

先做无模型的固定 registry/export 预检：

```bash
python -m s2c.experiments.external_baselines \
  --config configs/experiments/protocol_v2/external_baselines.yaml --smoke
```

`msp`、`energy`、`knn` 和 `lof` 是未来可用的 local frozen-MiniLM controls；当前
`--execute` 会被数据准入门槛拒绝。仅在新的官方 canonical 数据通过后，才可运行：

```bash
python -m s2c.experiments.external_baselines \
  --config configs/experiments/protocol_v2/external_baselines.yaml \
  --smoke --method msp --execute --resume
```

`DOC`、`ADB`、`DA-ADB`、`MOGB` 和 `(K+1)-way` 没有满足协议的执行环境或训练契约时，
应让该命令生成 `blocked`/`unsupported` manifest；不得用零值、旧 TextOIR 数字或
MiniLM fallback 代替方法结果。详情见 `EXPERIMENTS.md`。

## 公开结果导出

先查看白名单和容量：

```bash
python tools/maintenance/export_public_results.py --dry-run
```

确认后执行并校验 SHA256：

```bash
python tools/maintenance/export_public_results.py --execute
python tools/maintenance/export_public_results.py --verify
```

导出脚本只复制 `configs/public_results.yaml` 明确列出的文件，并拒绝 checkpoint、模型、
embedding、逐样本输出和超过 10MB 的单文件。

## 测试与静态检查

```bash
pytest tests/unit -q
python -m py_compile \
  tools/maintenance/export_public_results.py \
  tools/analysis/audit_experiment_registry.py
git diff --check
```

## 读取结果

- GitHub 结果：先看 `results/README.md`，再用 `results/MANIFEST.csv` 校验文件来源。
- 完整证据：先读对应 artifact 根目录的 manifest，再读 summary CSV。
- 不要将 Gate-only 的 Frozen/CE/SupCon 数字解释为完整 Cascade。
