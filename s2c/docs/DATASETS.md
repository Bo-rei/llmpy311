# 数据来源裁决（2026-07-22）

本页只给出当前可执行的裁决。完整逐样本差异在
`../artifacts/s2c/reports/data_provenance_audit/2026-07-22_three_way_verification/`；
这里的 JSON 与 CSV 只保存轻量、可审计的结论，绝不复制语料。

| 数据集 | 官方/原始来源 | 三方内容核验 | split / 许可证 | 当前裁决 |
| --- | --- | --- | --- | --- |
| CLINC150 | `clinc/oos-eval@828f809…`, `data/data_full.json` | 官方、TEXTOIR、s2c raw 精确记录均为 23,700 | TEXTOIR 将 1,200 条 native OOS 合并进 test；CC BY 3.0 | `reconstructed_from_official`，已写入 `protocol_v2_official_v1` |
| Banking77 | `PolyAI-LDN/task-specific-datasets@57ec275…`, `banking_data/{train,test}.csv` | 官方→TEXTOIR 精确 13,080/13,083；空白规范化后 13,083/13,083 | TEXTOIR 额外生成 1,000 条 dev，且 3 条文本与 raw 不同；CC BY 4.0 | `reconstructed_from_official`，已写入 `protocol_v2_official_v1` |
| StackOverflow | Xu et al. (2015) 处理数据仓库 `jacoxu/StackOverflow@7c207f5…` | TEXTOIR 为 20,000 行；历史 s2c 为去重后 19,980 行 | 上游和 TEXTOIR 均未提供可核验的数据许可证 | `blocked_unverified` |

`BANKING77-OOS` 是历史 s2c 扩展，不是官方标准 Banking77，也不在 TEXTOIR 中；它保持
`blocked_unverified`，不得和标准 Banking77 合并或比较。

## 当前门槛

- 现有 `s2c/data/protocol_v2` 是从 TEXTOIR 快照构建的**候选副本**，不是本页认可的
  canonical dataset；不得将其结果称为正式 protocol_v2 结果。
- CLINC150 必须由官方 `data_full.json` 重建；Banking77 必须由官方 raw train/test 重建，并
  将任何 train/dev 派生规则写入版本化 manifest。
- StackOverflow 在许可证和原始文本一致性完成独立核验前，不得进入 canonical、训练、embedding、
  MOGB/DCL 复现或与 TEXTOIR 的公平比较。

只有 `clinc150` 与 `banking77` 可以在 `protocol_v2_official_v1` 下按需 materialize view/export；
`stackoverflow` 与 `banking77_oos` 仍被 admission gate 拒绝。已完成 canonical、全量 registry
和 `seed=0/KIR=0.50` 的 view/export 验证；其余 view/export 是从 immutable canonical records
按需生成的派生产物，不属于训练或 embedding 运行。
- 历史 v19–v22/Cascade 读取的是
  `../assets/datasets/s2c/prepared/data/multidataset/v19/`，仅作 legacy evidence；不能替代 raw
  source，也不能混入新协议。

审计输出包括 `official_vs_textoir.csv`、`official_vs_s2c.csv`、`textoir_vs_s2c.csv`、
`missing_samples.csv`、`extra_samples.csv`、`label_conflicts.csv`、`split_conflicts.csv`、
`normalized_text_matches.csv`、`source_license_report.csv` 及各数据集的
`dataset_decision.json`。
