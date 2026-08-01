# s2c 研究状态台账

> 唯一当前状态入口。实验任务开始先读本文件和同目录 ledger；任务结束必须更新。

## 当前状态

- 活动协议：`protocol_v2_textoir_v1`
- 当前阶段：`MOGB_OFFICIAL_CONVERGED`、`BRAK_PILOT` 已完成；`DCLOOS_PREFLIGHT` 已完成但因缺失
  官方 open-domain negative corpus 阻断。fixed-K mean-radius 180/180、MOGB fair/OFAT 和 BRAK
  均保持独立 artifact root，不覆盖旧实验。
- Git：`main`，当前基准 commit `2ff028e`；本阶段新增代码、配置、报告和第三方审计目录尚未提交
- Git dirty：`true`
- 最近冻结代码快照：`BOUNDARY_ATTRIBUTION_CODE.patch`，SHA256
  `16ecffdea31faded6305b9a9d5d5d165ac3a59a008315fdc4c1158f1edcca0d7`；
  本阶段另冻结 `R1_MINILM_STAGE_PROVENANCE.json` 和对应 code patch，旧 R1/E2/E3 快照保持不变。
  MOGB 消融由 `MOGB_ABLATION_EXECUTION_PROVENANCE.json` 绑定 runner/config/plan/source hash，
  SHA256 `e7c6b961cab050921750901c770b145ae2be0ccb7e9b7f0780d73d7325baf61f`。
- GitHub 状态：`origin/main` 尚未包含本轮官方 MOGB、BRAK、DCLOOS 审计和汇总表；本轮不自动 commit/push
- 最近更新：2026-08-01

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
| MOGB MiniLM fair matrix | complete 270/270；0 failed；6 methods、3 datasets、3 KIR、5 seeds | `do_not_repeat`；官方旧 BERT 复现另行审计 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/summary/all_runs.csv` |
| CLMSG Version A--C | complete 78/78；seed13 pilot 加 seeds42/87 确认性扩展，三 seed 结论一致 | `do_not_repeat`；不实现 manifold/entropy | `../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/summary/CLMSG_M4_CONFIRMATION_CLOSEOUT.md` |
| KNN fixed-k10 full protocol | complete 45/45；42 fresh + 3 exact reuse | `do_not_repeat`；只作 nonparametric baseline | `../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/summary/knn_pareto_v1/KNN_PARETO_CLOSEOUT.md` |
| KNN k sensitivity | complete 180/180；`k={5,10,20,30}`；135 个新增 k 单元 + 45 个 k10 单元，其中 3 格从确认阶段精确复用 | `do_not_repeat`；禁止 test-selected k；停止继续扩普通 KNN | `../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/summary/knn_k_sensitivity_v1/KNN_K_SENSITIVITY_CLOSEOUT.md` |
| MOGB frozen-MiniLM OFAT | complete 540/540；0 missing/invalid/duplicate；复用 135 个控制单元 | `do_not_repeat`；停止邻近 purity/support 网格 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1/summary/MOGB_ABLATION_CLOSEOUT.md` |
| MOGB fixed-K mean-radius | complete 180/180；135 新格 + 45 个 K=2 hash-validated reuse；45 adaptive reference | `do_not_repeat`；停止 frozen-partition granularity 扩展 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_fixed_k_mean_ablation_v1/summary/MOGB_FIXED_K_MEAN_CLOSEOUT.md` |
| MOGB official-logic smoke | partial_blocked；StackOverflow 1/1 工程单元完成，Banking77 未形成完整单元；无可比较收敛结果 | `do_not_repeat`；只保留 blocker 记录，不写入主表 | `../artifacts/s2c/external/mogb_official_modernized_smoke_v1/MOGB_OFFICIAL_MODERNIZED_PROVENANCE.json` |
| MOGB official-logic converged | complete 10/10；StackOverflow 与 Banking77 各 5 个 seed；兼容层下收敛，不是严格官方复现 | `do_not_repeat`；不与 MiniLM fair 结果混合 | `../artifacts/s2c/external/mogb_official_converged_v1/` |
| BRAK pilot | complete 21/21 summary cells；3 seeds；30 个 intent 全部选择 K=1；0 失败 | `do_not_repeat`；未通过扩展门，不启动三数据集扩展 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/brak_v1/` |
| DCLOOS official/unified preflight | blocked；源码可编译，但缺失 `squad_placeh.tsv` 和明确 negative corpus，未生成指标 | `do_not_repeat`；不得用 protocol OOS 替代额外监督 | `../artifacts/s2c/external/dcloos_v1/DCLOOS_PREFLIGHT.json` |

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
17. CLMSG seed13 验证确认：普通 KNN 的排序指标有竞争力（AUROC `0.8786`、AUPR-OOS
    `0.8431`），但 alpha=0.05 的 OOS F1 仅 `0.4520`；support-point local-scale 会把 AUPR
    降至约 `0.67--0.70`，不能优于 Single-centroid OOS F1 `0.6957`。
18. Global/class-conditional/hybrid split conformal 在 alpha=0.05 能把 Known false rejection
    控制在约 `3.5%--5.1%`，但 false acceptance 仍为 `79%--86%`；共形校准控制 coverage，
    不能修复 local-scale 已损失的 Known/OOS 排序。
19. 因 Version C 的所有预注册支持模式均未通过 seed13 门槛，seeds42/87、local manifold、
    label entropy、cross-conformal 与 CLMSG full sweep 均未启动；当前 CLMSG 不进入论文主方法。
20. CLMSG 收口验证完整通过：26 个授权输出、156,000 条逐样本预测与 1,000 条 Known 校准分数
    均可审计；完整 unit/integration/smoke 为 `258/8/3 passed`，无 split 泄漏或 test selection。
21. CLMSG 的 seed42/87 确认性扩展保持同一失败方向：三 seed primary local-scale conformal 平均
    OOS F1 为 `0.3604`，相对 Single-centroid 为 `-0.3924`，因此局部尺度路线正式停止。
22. 普通 KNN 的 `k={5,10,20,30}` 敏感性已覆盖 180/180 格；整体 OOS F1 依次为
    `0.6492/0.6335/0.5957/0.5565`。即使最好的描述性 `k=5`，相对 Single-centroid 仍为
    `-0.1373`，45 格 W/T/L 为 `2/0/43`；不得据此 test-select k。
23. MOGB frozen-MiniLM OFAT 已覆盖 540/540 新格。纯 partition 调整的最佳描述性配置
    `purity_get=0.90` 仅比默认 MOGB 提高 OOS F1 `+0.0135`，仍比 Single-centroid 低 `0.0391`；
    邻近 purity/minimum-support 网格不能弥补表示学习缺失。
24. `Euclidean + mean+std` 是四种 distance-radius 组合中最强者，相对默认 MOGB OOS F1
    提高 `+0.0526`，但相对 Single-centroid 的 F1-All 和 Known Recall 仍低 `0.1048/0.2593`。
    因而只看 OOS F1 会掩盖严重 Known 覆盖代价。
25. diagonal Mahalanobis 在 mean 与 mean+std 半径下都落后于 Euclidean；`min_select=5` 将平均
    粒球数增至约 193.7 而无实质 OOS F1 收益，说明更多细粒度球本身不是有效改进。
26. 在完全相同的 Frozen MiniLM、Euclidean 与 mean-radius 契约下，fixed K=1 的整体 OOS F1
    为 `0.7808`，高于 adaptive MOGB 的 `0.7339`，45 个配对单元全部获胜；fixed K=2/3/4
    也分别高 `0.0251/0.0138/0.0143`，说明动态粒球划分本身不是公开 MOGB 性能的充分来源。
27. fixed K 随 K 增大总体退化：K=1/2/3/4 的 F1-All 为
    `0.6616/0.6230/0.5922/0.5812`，Known Recall 为
    `0.5385/0.4998/0.4671/0.4547`。Banking77 存在少量 K2/K4 单元胜出，但 CLINC150 的
    15/15 与 StackOverflow 的 14/15 protocol cells 均由 K=1 获得最高 fixed-K OOS F1。
28. 官方 MOGB 兼容层已完成 StackOverflow 与 Banking77 各 5 个 seed 的 10/10 收敛单元；
    官方格式 F1-All 均值分别为 `40.7243` 与 `19.2843`，但该结果使用现代化兼容层和本地快照，
    不是严格论文复现，也不能与 MiniLM-fair 主表直接混合。
29. BRAK 在 StackOverflow/KIR50/42,87,100 的 30 个 Known intent 选择中全部保留 K=1；
    K>1 的 Known-only union-risk、交叉意图泄漏和 bootstrap 不稳定性均上升，未通过预注册扩展门。
    因此 BRAK 当前是安全的 Known-only 负控制，不是已验证的新 adaptive-K 方法。
30. DCLOOS 源码可编译，但官方所需 `squad_placeh.tsv` 外部 open-domain negative corpus 缺失；
    official/unified 均未启动，状态为 `blocked_missing_official_open_domain_oos`，没有伪造指标。

## 历史依据（不混入当前主表）

Frozen、CE、SupCon、CE-Recon、effective-rank、representation-collision、near/far OOS 和历史
Pipeline 结果仍可作为研究依据，但它们来自旧实验族，必须在 ledger 中标记为历史或
`protocol_migration_control`。

## 当前唯一下一步

完成本轮外部基线收口审计：验证 MOGB 官方兼容层、BRAK Known-only 选择和 DCLOOS blocker 的
provenance/报告/台账/回归测试一致性，然后把 `results/final_baselines/summary.csv` 作为当前
基线汇总。不得继续扩大 BRAK、自适应 K、MOGB 官方 BERT 矩阵或用 protocol OOS 替代 DCLOOS
缺失的外部负样本；若要推进论文主结果，下一阶段应先在统一指标中比较现有 MOGB-fair 组件与
Single/FK controls，再决定是否申请独立可复现环境完成严格官方复现。

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
- MOGB 官方仓库 pinned at `5b689e2a03de0d86ec41212825e5db8d7f0e5c02`，缺少 `utils` 且使用旧
  BERT/TextOIR 数据契约；当前 270-cell 结果是冻结 MiniLM 的公平适配矩阵，不是官方论文复现。
