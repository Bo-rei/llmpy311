# s2c 研究状态台账

> 唯一当前状态入口。实验任务开始先读本文件和同目录 ledger；任务结束必须更新。

## 当前状态

- 活动协议：`protocol_v2_textoir_v1`
- 当前阶段：`r1_contract_repair_v1` 已收口（12 trainable checkpoints / 30 Gate units）
- Git：`main`，基准 commit `5880a339c809a3dada72b1a21f92c4a9ece42676`
- Git dirty：`true`（E3 源码、文档和本轮 R1 变更尚未提交）
- 最近冻结代码快照：contract repair `R1_CONTRACT_REPAIR_CODE.patch`，SHA256 `a1f8302d5c512348e8adffeca01c9245b9da1a79d6c1d62b9f4973947650802d`；旧 R1 pilot/full 与 E3 快照仍保留用于历史复现。
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
| R1 pilot | completed but superseded by contract audit；Gate 预测保留，几何字段 invalid，near-OOS exploratory | `do_not_repeat`；不覆盖原始 artifacts | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/summaries/R1_CLOSEOUT.md` |
| R1_full | completed but superseded by contract audit；Gate 预测保留，几何字段 invalid，near-OOS exploratory | `do_not_repeat`；不覆盖原始 artifacts | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/summaries/R1_FULL_CLOSEOUT.md` |
| R1 contract repair | complete 12 checkpoints + 30 Gate units；0 失败/无效；独立 provenance 已冻结 | `do_not_repeat`；不覆盖旧 R1 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_contract_repair_v1/R1_CONTRACT_REPAIR_CLOSEOUT.md` |

E2/E3 的 K、KIR、random partition、聚类稳定性和 tiny-cluster 矩阵不得再次运行。

## 已确认结论

1. Banking77 的多中心收益是条件性的，且伴随 Known false rejection。
2. CLINC150 没有稳定的默认多中心收益。
3. StackOverflow 的 K>1 明显有害。
4. 聚类稳定性不是多中心有效的充分条件，Known-only reliability 也没有稳定跨数据集选 K 的证据。
5. 不存在跨数据集统一最优 K；`oracle_test_best_k` 不能作为正式选择规则。
6. 旧 R1 pilot/full 的几何统计和 test-defined near-OOS 结果已被 contract audit 标记为 superseded；旧 Gate prediction 保留为历史证据。
7. R1 的 StackOverflow K=2 大幅退化已经完成配对审计：pilot 的 `-0.5908` 是 Geometry CE-Recon 的 combined OOS F1 配对均值；R1_full Geometry 的 15 个单元均值为 `-0.4852`，Frozen MiniLM 为 `-0.0915`。历史 Frozen v19 的约 `-0.0622` 属于不同协议/表示，不能直接混比。
8. 代表性 Geometry 单元中 K=2 的 ID Recall 上升而 false acceptance 大幅上升，当前更支持“多球接受区域并集过覆盖”是主要失败方向，而不是 Known 误拒绝。
9. Contract repair 在 StackOverflow/KIR50/3 seeds 下确认：pooled 与 normalized classifier head 对 K=1 影响很小、对 K=2 影响明显；修复后的 student intra/inter 距离与 teacher 指标分离；当前 Known-only calibration 没有合法 validation OOS，因此 near/medium/far 只能标记 exploratory，不能进入正式成功标准。
10. Geometry loss 在 pooled-head 契约下的 K=1 平均 OOS F1 仅有小幅描述性变化（`+0.0009`），K=2 仍严重退化；该 pilot 不授权 corrected R1_full。

## 历史依据（不混入当前主表）

Frozen、CE、SupCon、CE-Recon、effective-rank、representation-collision、near/far OOS 和历史
Pipeline 结果仍可作为研究依据，但它们来自旧实验族，必须在 ledger 中标记为历史或
`protocol_migration_control`。

## 当前唯一下一步

只进行 contract-repair 结果审阅和论文 claim 边界更新：确认 pooled-head、student geometry 和
validation-only bucket 契约后，决定是否结束表示实验或另行提出经过批准的 corrected R1_full 计划。
当前不得自动运行 corrected R1_full、外部 baseline、ADB、DA-ADB、MOGB 或完整 Pipeline。

## 当前风险

- R1 只验证表示适配是否保留局部几何，不能宣称完整 Cascade 改善。
- E3 输出中的全局 tiny-cluster 字段有解释限制；论文分析优先使用 intent-level features。
- 任何未在 `EXPERIMENT_LEDGER.csv` 登记的实验均不得启动。
- R1 结果只属于本轮表示方法证据；不得与历史 Frozen/CE/SupCon/CE-Recon 数字直接合并为
  `protocol_v2_textoir_v1` 主表。
- StackOverflow 的 K=2 幅度必须同时标注 metric、representation、KIR、seed 聚合和 distance；
  `-0.5908` 不得脱离 R1 pilot 的具体协议解释为所有多中心方法的通用退化幅度。
