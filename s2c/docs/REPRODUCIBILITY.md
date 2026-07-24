# 可复现性

> **当前不允许重建或运行 protocol_v2 candidate。** 三方来源审计已将 CLINC150 和
> Banking77 标为 `reconstructed_from_official`，StackOverflow 标为 `blocked_unverified`。
> 下列历史命令保留为实现说明，不能在当前 checkout 执行；正式重建必须从
> `docs/DATASETS.md` 所列官方 raw source 开始，并使用新的 dataset_version/run root。

## 历史 candidate 重建顺序（禁止执行）

```bash
python -m pip install -e .
python -m s2c.data.import_textoir
python -m s2c.data.build_canonical
python -m s2c.data.build_registries
python -m s2c.data.build_views
python -m s2c.data.export_protocol
python -m s2c.data.validate_protocol --require-views --require-exports
```

这些命令描述旧候选的可重放工程路径，不授权新的实验。导入需要本地 TEXTOIR snapshot；
其运行时独立性不能替代官方 raw source、split 与许可证的三方裁决。

## 官方重建数据（CLINC150 与 Banking77）

以下只导入和验证数据，不编码、不训练、不运行 Gate。两条源 checkout 必须先通过
固定 commit 与许可证核验；命令会把字节副本写入新的 dataset_version，不会覆盖 candidate：

```bash
export S2C_DATASET_VERSION=protocol_v2_official_v1
python -m s2c.data.official_import --dataset clinc150 --clinc-root ../assets/datasets/s2c/source/clinc150
python -m s2c.data.official_import --dataset banking77 --banking-root /path/to/task-specific-datasets
python -m s2c.data.build_canonical --dataset clinc150 --dataset banking77
python -m s2c.data.build_registries --dataset clinc150 --dataset banking77
python -m s2c.data.validate_protocol --dataset clinc150 --dataset banking77
```

Banking77 的 calibration 由 source manifest 中的 stratified SHA256 规则从官方 train 派生；
它不是 TextOIR dev。StackOverflow 和 BANKING77-OOS 没有此命令路径，直到其来源与许可证核验完成。

## 官方 E1 Gate-only 子矩阵（已完成）

`protocol_v2_official_v1` 已完成 CLINC150 与 Banking77 的 24 个 frozen-MiniLM 固定边界 Gate
单元：KIR `{.25,.50,.75}`、seed 42、K `{1,2}` 和两种距离。运行前先用同一环境变量物化对应
views/exports；运行/验证/汇总命令见 `RUNBOOK.md`。该结果不包含 StackOverflow、Router、Expert 或
完整 Cascade，且不能与被封锁的 candidate `protocol_v2` 混合。

## 历史 candidate Gate 网格（禁止执行）

```bash
python -m s2c.experiments.plan --config configs/experiments/protocol_v2/smoke_gate.yaml
python -m s2c.experiments.runner --config configs/experiments/protocol_v2/smoke_gate.yaml --resume
python -m s2c.experiments.verify --config configs/experiments/protocol_v2/smoke_gate.yaml --require-complete
python -m s2c.experiments.summarize --config configs/experiments/protocol_v2/smoke_gate.yaml
```

Gate runner 现在读取 `configs/data/protocol_v2_admission.json` 并 fail closed；因此任何非
`--dry-run` 的候选实验命令都会被拒绝。旧 run 目录和 embedding cache 保留作审计，但不属于
正式结果。

## 证据链

每个正式结果至少需要：source manifest、canonical manifest、registry SHA256、view/export
manifest、resolved config、run manifest 和 metrics。`results/` 的轻量文件必须能回指 raw
artifact；v19 与 protocol_v2 数字不得无标记混合。
