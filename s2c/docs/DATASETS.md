# 活动数据集裁决

唯一活动协议是 `protocol_v2_textoir_v1`。它固定使用本地 TEXTOIR commit
`dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 的 train/dev/test 快照；导入后所有运行时
只读取被 Git 忽略的 `s2c/data/`，不读取 `../textoir/data`。

| 数据集 | TEXTOIR 目录 | 快照规模 | 准入 | 本地训练/评估 | s2c 重新分发 |
| --- | --- | ---: | --- | --- | --- |
| CLINC150 | `data/oos` | 23,700 / 150 labels / 1,200 native OOS | `admitted_official` | 是 | 遵循上游条款 |
| Banking77 | `data/banking` | 13,083 / 77 labels / 无 native OOS | `admitted_official` | 是 | 遵循上游条款 |
| StackOverflow | `data/stackoverflow` | 20,000 / 20 labels / 无 native OOS | `admitted_benchmark_local_only` | 是 | 否 |

## StackOverflow 的有界准入

StackOverflow 是固定的 TEXTOIR-compatible 本地 benchmark 快照，不是 s2c 自采数据，也不是
Stack Overflow 官方发布的分类数据集。当前可以用于 canonical、embedding、Gate、Pipeline 和
外部 baseline 复现；但完整文本不得进入 Git、论文附件或任何 s2c 打包发布，且不得声称已完成
逐条帖子归属或许可证核验。

## 历史协议

- `protocol_v2_official_v1`：冻结审计版本，保留但不再是新实验默认值。
- `protocol_v2`：被拒绝的 legacy candidate，只能显式读取作历史审计。
- v19--v22/Cascade：历史证据，不与活动协议数字混表。

每个实际快照的文件 hash、行数、标签数和 redistribution policy 都写在
`data/manifests/protocol_v2_textoir_v1/<dataset>/SOURCE_MANIFEST.json`。
