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
| R1 pilot | Geometry-Preserving CE-Recon 表示 pilot | completed but superseded by contract audit | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/summaries/R1_CLOSEOUT.md` |
| R1_full | KIR `0.25/0.50/0.75`、5 seeds 的几何保持扩展 | completed but superseded by contract audit | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/summaries/R1_FULL_CLOSEOUT.md` |
| R1 contract repair | pooled/normalized head、student/teacher geometry、validation-only buckets | complete: 12 checkpoints / 30 Gate units；0 failed；near-OOS exploratory only | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_contract_repair_v1/R1_CONTRACT_REPAIR_CLOSEOUT.md` |
| Multi-center boundary attribution | StackOverflow covariance、选球规则与 Known-only 半径归因 | complete: 60/60；停止固定 KMeans 多中心救援 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/multicenter_boundary_attribution/BOUNDARY_ATTRIBUTION_CLOSEOUT.md` |
| MiniLM training and StackOverflow repair | Frozen/head-only CE/full CE/SupCon/CE-Recon 与逐样本 K=1/K=2 审计 | complete: 36 checkpoints + 180/180 Gate；停止固定多中心救援 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_training_and_stackoverflow_repair_v1/summaries/MINILM_PILOT_CLOSEOUT.md` |
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

## R1 表示几何保持 pilot（历史，已被 contract audit supersede）

R1 在同一 Frozen MiniLM teacher 上比较 `CE-Recon` 与加入 batch 内 pairwise cosine relation
preservation 的 `CE-Recon-Geometry`。全局 `beta=1.0` 只由三个数据集 seed=42 的 Known
train/calibration 指标选择；108 个 Gate 单元（3 dataset × 3 seed × 3 representation ×
2 K × 2 distance）全部完成。相对 CE-Recon，K=1 OOS F1 三个数据集均改善，但 Banking77
near-OOS 下降，StackOverflow K=2 仍严重退化；因此 R1 不是普遍多中心或完整 Pipeline 结论。
详细几何、碰撞和 K=1/K=2 对照见 R1 summaries；R1_full 已完成完整性审计，closeout 明确区分
K=1 表示结果与 K=2 结构诊断。旧 R1 的几何字段已被 contract audit 标记为
`invalid_metric_implementation`，旧 near-OOS 分桶标记为 `exploratory_test_defined_bucket`。

## R1 contract repair

该阶段不重复 E2/E3，也不启动新数据集或 KIR。它只在 StackOverflow/KIR50、seed `{42,87,100}`
下修复三个可比性问题：CE classifier 显式使用 pooled 或 normalized_pooled；student 的
intra/inter distance 与 teacher 指标分开；near/medium/far 只有在 validation OOS 存在时才正式定义。
当前 protocol 的 calibration 是 Known-only，因此 30 个 Gate 行的 bucket status 均为
`exploratory_unavailable_validation_oos`，不能用于正式 near-OOS 成功标准。修复结果显示 pooled
head 对 K=1/K=2 的差异分别很小/明显，Geometry loss 在 pooled-head K=1 只有小幅描述性变化，K=2
仍是结构性退化。不要把该 pilot 写成 corrected R1_full 的授权。

## 多中心边界归因

该阶段不训练 encoder，只复用 Frozen、CE-Recon pooled-head 和 Geometry pooled-head 表示，
逐步替换 covariance scope、选球 score 和 Known-train radius。shared-intent diagonal covariance
可明显缓解 per-cluster covariance 的 K=2 退化，但没有表示通过预注册安全门；按 `d/r` 选球和
q95 半径均增加 false acceptance。结论是停止继续修补 StackOverflow 固定 KMeans 多中心，
保留其作为 boundary-union failure 证据。

## 历史版本

- `protocol_v2_official_v1`：冻结 provenance audit；不再作为新实验默认。
- `protocol_v2`：拒绝的 legacy candidate；仅可显式读取用于历史审计。
- v19--v22、旧 Cascade：保留原始 artifacts，不覆盖也不重新命名。

后续主表只使用带有该 protocol version、dataset manifest SHA、registry SHA、encoder revision 和 run
manifest 的结果。详细命令见 `RUNBOOK.md`，结果字段约束见 `RESULTS_CONTRACT.md`。
