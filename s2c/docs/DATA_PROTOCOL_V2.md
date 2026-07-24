# protocol_v2 数据契约

> **当前状态：官方重建数据已局部准入。** `protocol_v2` 仍是被阻断的 TEXTOIR candidate；
> `protocol_v2_official_v1` 只准入 CLINC150 与 Banking77。StackOverflow 与 legacy
> BANKING77-OOS 因来源/许可证未核验继续被阻断。现有 candidate 运行产物不能作为论文正式数字。
> 详见 `docs/DATASETS.md` 和
> `../artifacts/s2c/reports/data_provenance_audit/2026-07-22_three_way_verification/`。

protocol_v2 将数据来源、Known intent 抽样、固定 views 与方法格式导出分开，避免方法在
运行时自行改变样本集合。

## 固定来源（重建完成后）

正式运行时只读取 `s2c/data/` 的指定 `S2C_DATASET_VERSION`。`../textoir/data/` 只允许由显式导入命令在导入阶段读取，
不属于训练、评估或实验 runner 的依赖。每个 canonical dataset 只能有一个明确 raw source：
CLINC150 与 Banking77 为官方 raw source；StackOverflow 在许可证通过前没有 canonical source。
TEXTOIR 只可作为格式/转换对照，不能替代官方 raw source。

## Canonical 记录

每条 canonical JSONL 记录保留原文本、原标签、原 split、来源行和确定性 `sample_id`。
构建不 lower-case、不去重、不修改标签，也不提前将 held-out intent 改为 OOS。CLINC150
的 native OOS 通过 `native_oos=true` 与 held-out intent OOS 保持可区分。

## KIR registry 和 views

registry 固定 Known 与 held-out intent 列表。所有 Gate、Router、Expert 和外部方法导出
必须消费同一 registry，不能自行重抽样。`train_known` 与 `calibration_known` 只含 Known
intent；test views 才包含 held-out/native OOS。任何 K、阈值和边界选择不得使用 test OOS。

## 运行前门禁

```bash
python -m s2c.data.validate_protocol --require-views --require-exports
python tools/maintenance/check_data_tracking.py
```

若来源裁决、许可证、manifest、SHA256、view 分割或 exporter sample-id 映射任一项不一致，
命令必须失败；不能以旧 v19 结果替代 protocol 数据。训练、embedding、MOGB/DCL 复现和
TEXTOIR 公平比较只能在 dataset-level admission 为 `admitted` 且对应 canonical/view/export
manifest 已生成后开始；这不准入 StackOverflow。
