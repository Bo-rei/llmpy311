# 实验状态

## 活动协议：protocol_v2_textoir_v1

活动数据是固定 TEXTOIR commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 的 CLINC150、
Banking77 与 StackOverflow snapshot。StackOverflow 是 local-only benchmark，不阻止本地实验，
但不允许完整语料进入公开 Git 或 s2c 再分发。

| 阶段 | 目的 | 当前状态 | 证据位置 |
| --- | --- | --- | --- |
| E0 | source/canonical/registry/views/exports 与 runtime independence | complete | `docs/audits/protocol_v2_implementation/` |
| E1 | 3 datasets × 3 KIR × K{1,2} × 2 distances | complete: 36/36 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e1_gate_smoke.csv` |
| E2 | 11 KIR × 5 seeds × K{1..5} × 2 distances | complete: 1,650/1,650 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/` |
| E3 | KMeans/random-balanced, stability, Known-only coverage/reliability | complete: 720 + 180 groups | `../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/summaries/` |
| R1 | Geometry-Preserving CE-Recon 表示 pilot | complete: 108/108；条件性支持 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/summaries/R1_CLOSEOUT.md` |
| R1_full | KIR `0.25/0.50/0.75`、5 seeds 的几何保持扩展 | complete: 135 cells / 270 Gate units；0 failed | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/summaries/R1_FULL_CLOSEOUT.md` |
| E4--E7 | boundary grid, external baselines, representation, Pipeline | not started | R1 closeout 后另行决定 |

E1 和 E2 是 Gate-only 证据，不应与历史完整 Cascade 或 v19--v22 结果混合解释。E3 的 K=1 只读引用
E2；每个新 run ID、manifest、
cache 与 summary 均包含 `protocol_v2_textoir_v1`，防止误 resume 历史 protocol。

## E3 收口结论（机制诊断）

E3-A 的 720 个分簇控制单元和 E3-B/C 的 180 个诊断组均完成且无失败。结论是数据集条件性的，
不是“多中心默认有效”：Banking77 中 KMeans 相对 random-balanced 的 combined OOS F1 平均差约
`+0.0437`，且 KMeans 稳定性较高，但 Known false rejection 随 K 增长；CLINC150 的差值约
`-0.0137`，random-balanced 略优而 KMeans 的 Known Recall 下降；StackOverflow 的差值约
`-0.1543`，KMeans 多中心显著退化。三者都不能据此宣布最终 adaptive-K；这些数字来自
`E3_partition_paired_effects.csv` 的配对汇总，必须结合 distance、KIR、seed 和 OOS source 解读。

Known-only 诊断显示 KMeans 的初始化稳定性通常高于 random-balanced，但稳定不等于有利于 OOS：
StackOverflow 的稳定子簇仍导致更高的 false acceptance，符合 boundary-union/支持区域失配解释。
可靠性特征仅作事后关联分析，不能用于本阶段的测试集选 K。

## R1 表示几何保持 pilot

R1 在同一 Frozen MiniLM teacher 上比较 `CE-Recon` 与加入 batch 内 pairwise cosine relation
preservation 的 `CE-Recon-Geometry`。全局 `beta=1.0` 只由三个数据集 seed=42 的 Known
train/calibration 指标选择；108 个 Gate 单元（3 dataset × 3 seed × 3 representation ×
2 K × 2 distance）全部完成。相对 CE-Recon，K=1 OOS F1 三个数据集均改善，但 Banking77
near-OOS 下降，StackOverflow K=2 仍严重退化；因此 R1 不是普遍多中心或完整 Pipeline 结论。
详细几何、碰撞和 K=1/K=2 对照见 R1 summaries；R1_full 已完成完整性审计，closeout 明确区分
K=1 表示结果与 K=2 结构诊断。

## 历史版本

- `protocol_v2_official_v1`：冻结 provenance audit；不再作为新实验默认。
- `protocol_v2`：拒绝的 legacy candidate；仅可显式读取用于历史审计。
- v19--v22、旧 Cascade：保留原始 artifacts，不覆盖也不重新命名。

后续主表只使用带有该 protocol version、dataset manifest SHA、registry SHA、encoder revision 和 run
manifest 的结果。详细命令见 `RUNBOOK.md`，结果字段约束见 `RESULTS_CONTRACT.md`。
