# s2c 研究状态台账

> 唯一当前状态入口。实验任务开始先读本文件和同目录 ledger；任务结束必须更新。

## 当前状态

- 活动协议：`protocol_v2_textoir_v1`
- 当前阶段：`R1_geometry_preserving_representation_full` 已收口（135 cells / 270 Gate units）
- Git：`main`，基准 commit `1f299d33bee949d934a74cadbf6adb1962d620ea`
- Git dirty：`true`（E3 源码、文档和本轮 R1 变更尚未提交）
- 最近冻结代码快照：R1_full `R1_FULL_CODE_SNAPSHOT.patch`，SHA256 `21b923d67d6b4169e6261c0e4675f508a4caa67f9e4c9577d42c8c6bdc373b81`；R1 pilot 与 E3 快照仍保留用于历史复现。
- GitHub 状态：落后于本地；本轮不得自动 commit/push
- 最近更新：2026-07-28

## 已完成且禁止重复

| 阶段 | 状态 | 规则 | 证据 |
| --- | --- | --- | --- |
| E0 | complete | `do_not_repeat` | `../artifacts/s2c/runs/protocol_v2_textoir_v1/` 数据审计 |
| E1 | complete 36/36 | `do_not_repeat` | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e1_gate_smoke.csv` |
| E2 | complete 1650/1650 | `do_not_repeat` | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/` |
| E3-A | complete 720/720 | `do_not_repeat` | `../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/summaries/` |
| E3-B/C | complete 180/180 | `do_not_repeat` | 同上 |
| R1 pilot | complete 108/108；24 次实际训练、3 个 beta=1.0 复用引用 | `do_not_repeat`；仅可另行计划 R1_full | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/summaries/R1_CLOSEOUT.md` |
| R1_full | complete 270 个 Gate 单元；0 失败/无效；独立 provenance 已冻结 | `do_not_repeat`；不与 R1 pilot 重复 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/summaries/R1_FULL_CLOSEOUT.md` |

E2/E3 的 K、KIR、random partition、聚类稳定性和 tiny-cluster 矩阵不得再次运行。

## 已确认结论

1. Banking77 的多中心收益是条件性的，且伴随 Known false rejection。
2. CLINC150 没有稳定的默认多中心收益。
3. StackOverflow 的 K>1 明显有害。
4. 聚类稳定性不是多中心有效的充分条件，Known-only reliability 也没有稳定跨数据集选 K 的证据。
5. 不存在跨数据集统一最优 K；`oracle_test_best_k` 不能作为正式选择规则。
6. R1_full 的 Geometry-Preserving CE-Recon 相对普通 CE-Recon 在三个数据集的 K=1 OOS F1 均有正向变化，平均 Known Recall 变化约为 `-0.0057`；但 Banking77/StackOverflow near-OOS 下降，StackOverflow 的 K=2 仍严重退化，因此只能称为条件性表示层证据。
7. R1 的 StackOverflow K=2 大幅退化已经完成配对审计：pilot 的 `-0.5908` 是 Geometry CE-Recon 的 combined OOS F1 配对均值；R1_full Geometry 的 15 个单元均值为 `-0.4852`，Frozen MiniLM 为 `-0.0915`。历史 Frozen v19 的约 `-0.0622` 属于不同协议/表示，不能直接混比。
8. 代表性 Geometry 单元中 K=2 的 ID Recall 上升而 false acceptance 大幅上升，当前更支持“多球接受区域并集过覆盖”是主要失败方向，而不是 Known 误拒绝。

## 历史依据（不混入当前主表）

Frozen、CE、SupCon、CE-Recon、effective-rank、representation-collision、near/far OOS 和历史
Pipeline 结果仍可作为研究依据，但它们来自旧实验族，必须在 ledger 中标记为历史或
`protocol_migration_control`。

## 当前唯一下一步

R1_full 已完成完整性审计：135 个表示 cell、270 个 Gate 单元、0 失败/无效；固定全局 `beta=1.0`，
只使用 Known calibration 选择 checkpoint。K=1 的 OOS F1 相对 CE-Recon 三个数据集均为正，但
near-OOS 不一致，StackOverflow K=2 仍严重退化。当前唯一下一步是将 R1_full 结果接入论文 claim
审阅并决定外部 baseline；不得继续扫描更多 K 或启动完整 Pipeline。

## 当前风险

- R1 只验证表示适配是否保留局部几何，不能宣称完整 Cascade 改善。
- E3 输出中的全局 tiny-cluster 字段有解释限制；论文分析优先使用 intent-level features。
- 任何未在 `EXPERIMENT_LEDGER.csv` 登记的实验均不得启动。
- R1 结果只属于本轮表示方法证据；不得与历史 Frozen/CE/SupCon/CE-Recon 数字直接合并为
  `protocol_v2_textoir_v1` 主表。
- StackOverflow 的 K=2 幅度必须同时标注 metric、representation、KIR、seed 聚合和 distance；
  `-0.5908` 不得脱离 R1 pilot 的具体协议解释为所有多中心方法的通用退化幅度。
