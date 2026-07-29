# s2c 研究状态台账

> 唯一当前状态入口。实验任务开始先读本文件和同目录 ledger；任务结束必须更新。

## 当前状态

- 活动协议：`protocol_v2_textoir_v1`
- 当前阶段：`minilm_training_and_stackoverflow_repair_v1` 已收口；StackOverflow
  逐样本 K=1/K=2 审计和 MiniLM Known-only pilot 均完成
- Git：`main`，阶段基准 commit `bca13b51221a5c327fa0197229e783c42f57bba7`
- Git dirty：`true`（包含源码布局整理、本阶段实现与状态记录；本阶段不自动 commit/push）
- 最近冻结代码快照：`BOUNDARY_ATTRIBUTION_CODE.patch`，SHA256
  `16ecffdea31faded6305b9a9d5d5d165ac3a59a008315fdc4c1158f1edcca0d7`；
  本阶段另冻结 `R1_MINILM_STAGE_PROVENANCE.json` 和对应 code patch，旧 R1/E2/E3 快照保持不变。
- GitHub 状态：`origin/main` 已同步到 `bca13b5`；本轮新增改动尚未提交，任务不得自动 commit/push
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
| Multi-center boundary attribution | complete 60/60；0 失败；无 encoder 训练 | `do_not_repeat`；停止固定 KMeans 多中心救援 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/multicenter_boundary_attribution/BOUNDARY_ATTRIBUTION_CLOSEOUT.md` |
| MiniLM training and StackOverflow repair | complete 36 checkpoints + 180/180 Gate；0 失败；sample/sphere audit complete | `do_not_repeat`；停止通过表示训练救活固定多中心 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_training_and_stackoverflow_repair_v1/summaries/MINILM_PILOT_CLOSEOUT.md` |

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
11. StackOverflow 边界归因中，shared-intent diagonal covariance 是唯一一致缓解 K=2 的组件，
    但 Frozen/CE-Recon/Geometry 三种表示仍未通过预注册安全门；normalized-score 选球和
    Known-train q95 半径反而扩大 false acceptance。
12. 因此固定 KMeans 多中心的“救活”路线正式停止；StackOverflow 保留为 boundary-union
    failure 的负面机制证据，不再增加损失、K、半径或选择器。
13. StackOverflow/KIR50/seed42 的当前 E2 cache 与 canonical view 按 sample ID 和实际 embedding
    bytes 对齐；TEXTOIR ADB 兼容 smoke 接近其仓库参考结果，未发现“数据快照损坏”这一简单解释。
14. 历史 `nearest_sphere` 是 E2 的冻结打分契约；`normalized_union` 已实现为显式诊断模式，单格
    对照没有消除 StackOverflow K=2 退化，因此不能把该缺陷修复包装成多中心方法提升。
15. 新 MiniLM pilot 的 StackOverflow cache/view 逐样本审计通过；Frozen、Head-only CE、Full CE、
    SupCon 和 CE-Recon 均未把 K=2 的 OOS false acceptance 降到预注册安全门内。Full CE/CE-Recon
    的 K=1 有明显提升，但其 K=2 退化更大；SupCon 也未修复多球过覆盖。
16. MiniLM pilot closeout 的 distance 汇总已完成后处理校正：表格现在使用 42/87/100 三个 seed
    的配对均值；原始 Gate run、paired delta 和逐样本审计未修改，校正不改变停止决策。

## 历史依据（不混入当前主表）

Frozen、CE、SupCon、CE-Recon、effective-rank、representation-collision、near/far OOS 和历史
Pipeline 结果仍可作为研究依据，但它们来自旧实验族，必须在 ledger 中标记为历史或
`protocol_migration_control`。

## 当前唯一下一步

冻结并审阅 `minilm_training_and_stackoverflow_repair_v1` 的单中心结果，选择论文中可保留的
Frozen/训练表示对照；不扩展 KIR、seed 或 K=1..5，不运行 corrected R1_full、ADB、DA-ADB、MOGB
或完整 Pipeline。StackOverflow 的固定后处理多中心路线已停止，后续任何新方法必须先登记新的
研究问题和独立 artifact root。

## 当前风险

- R1 只验证表示适配是否保留局部几何，不能宣称完整 Cascade 改善。
- E3 输出中的全局 tiny-cluster 字段有解释限制；论文分析优先使用 intent-level features。
- 任何未在 `EXPERIMENT_LEDGER.csv` 登记的实验均不得启动。
- R1 结果只属于本轮表示方法证据；不得与历史 Frozen/CE/SupCon/CE-Recon 数字直接合并为
  `protocol_v2_textoir_v1` 主表。
- StackOverflow 的 K=2 幅度必须同时标注 metric、representation、KIR、seed 聚合和 distance；
  `-0.5908` 不得脱离 R1 pilot 的具体协议解释为所有多中心方法的通用退化幅度。
- `normalized_union` 是新诊断契约，不得用于改写 E2/E3 历史表；所有 M1 run 必须显式记录
  acceptance mode、cache sample-id hash 和实际 embedding hash。
- 本阶段 Gate 指标只属于新的独立 artifact root；由于当前 protocol 没有合法 validation OOS，
  没有生成正式 near/medium/far 成功标准，任何近邻分桶只能另行登记为探索性分析。
