# 当前研究状态

这是当前唯一状态入口。活动协议为 `protocol_v2_textoir_v1`；当前基准 commit 为
`169a5a5bd2b5ba3ff640e4c7b2b404c9cac1f7ea`，当前工作分支为
`main`。本地分支比 `origin/main` 超前 1 个 commit；当前工作树因本地 contract-repair 代码、轻量结果和
文档尚未提交而 dirty，父仓库没有运行中的实验。`third_party/mogb_official` 仍是独立只读来源
checkout，其本地审计元数据保持在子仓库工作树中，不修改第三方源码。

> 当前工作区复核（2026-08-08）：`../artifacts/s2c/runs/` 与 `../artifacts/s2c/cache/` 当前存在，E2、RACAL、
> Trainable control 等原始运行目录和部分 checkpoint 可读取；但每个阶段仍必须以自身 `PROVENANCE`、manifest
> 和完整性检查为准，不能仅凭目录存在就宣称可重跑。`results/analysis/` 下的轻量 CSV/JSON/图表仍是当前报告
> 的主要阅读入口。

## 数据源与 TextOIR baseline 范围校正（2026-08-11）

StackOverflow、Banking77，以及当前 S2C 名为 `clinc150` 的 OOS 数据，已与独立
`../textoir` checkout 的对应文件完成 SHA256/字节级核对；其中 `clinc150` 实际对应 TextOIR 的
`data/oos`，不是 `data/clinc`。因此 ADB、DA-ADB、KNNCL 等路线不是“换了数据”，而是使用同源数据上的
不同 backbone、Known-list、seed、训练过程、阈值和评估合同。S2C 的本地数据副本只用于运行隔离，不改变
数据内容。

TextOIR 代码和 README 中的检测路线不止 ADB/DA-ADB，还包括 MSP、SEG、OpenMax、LOF、DOC、DeepUnk、
`(K+1)-way`、MDF、ARPL、KNNCL-last/all、DA-ADB-llama 和 EliDecide。当前 published/legacy reference
必须保留在外部 baseline 层；当前统一 final-metrics 合同尚未闭合的方法，继续标记为 historical/blocked，
不能用中间预测补齐排名。ADB 已完成 45/45 个当前外部 BERT 单元；DA-ADB 已完成 StackOverflow/KIR=.50
的 3-seed 有效外部汇总，但仍是 `valid_external_cell_pending_replication`，均不并入 MiniLM fair 主排名。

## 统一对比计划与附件要求审计（2026-08-11）

上一版十阶段统一对比 plan 与附件要求的方向高度一致，但完整实验交付尚未完成。当前已闭合的是
Known-only MiniLM fair Gate/MOGB 组件主矩阵、已有方法的匿名 prediction contract、441 个同序列对齐
method-cell、MOGB 大部分机制包以及部分 ADB/DA-ADB 外部合同分析；未闭合的是 KNNCL、OpenMax/DOC/DeepUnk
当前 final metrics，DA-ADB 完整矩阵，DCLOOS fixed-registry final metrics/监督前沿，S2C-BERT/ADB-MiniLM
控制，error-aware UMAP，固定全方法 near-OOS，low-resource 和统一 performance-cost Pareto。

逐项状态矩阵：`results/analysis/plan_alignment_audit_v1/requirement_matrix.csv`；读者版审计：
`docs/analysis/PLAN_ALIGNMENT_AUDIT_V1.md`；后续只按 entry/exit gate 执行的补充计划：
`docs/analysis/UNIFIED_COMPARISON_SUPPLEMENT_PLAN_V1.md`。该审计不把 blocked、timeout、smoke 或中间
prediction 转写成性能结果，也不改变已完成 E0–E3/MOGB artifact 的不得重复边界。

## 当前对比总入口（2026-08-10）

如需先看“当前究竟在比哪个方法、历史 Cascade 与当前 Trainable-K1 如何区分、ADB/MOGB 外部参照和
MOGB 论文差距如何解释”，请先读：
`docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`。
如需核对哪些结论可以写、哪些结论不能写，请读 `docs/analysis/RESULT_CLAIM_AUDIT_V1.md`。
如果需要逐格查看当前 Trainable-K1 与 `fulltex.tex` 历史完整 Cascade 的描述性差距，以及外部方法为何不能
直接混排，请读 `docs/analysis/CROSS_CONTRACT_GAP_V1.md`；其九格 CSV 和 SVG 由源哈希固定的轻量结果生成。
配套四张总览图位于 `figures/experiment_comparison_overview_v2/`，机器可读源哈希位于
`results/analysis/experiment_comparison_overview_v2/MANIFEST.json`。
当前最简决策入口为 `docs/analysis/EXPERIMENT_DECISION_DASHBOARD_V1.md`；它汇总 63 条同协议 fair
Gate 行，并单独标记外部/MOGB 参照。ADB 最新隔离 CUDA 三 seed 的跨数据集结果见
`results/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_REPORT.md`，StackOverflow 运行细节见
`docs/analysis/ADB_GPU_RUNTIME_THREE_SEED_V1.md`；任何 ADB 行都不要与 MiniLM fair 行合并排名。
ADB 跨 KIR=.25/.50/.75 的五 seed `45/45` 结果和曲线见 `results/analysis/adb_kir_sensitivity_v2/` 与
`figures/adb_kir_sensitivity_v2/`，三 seed 历史汇总仍保留在 v1；两者只用于外部 BERT 工作点/机制分析。
对应的五 seed 合同感知图见 `results/analysis/cross_kir_contract_atlas_v2/` 与
`figures/cross_kir_contract_atlas_v2/`，不构造跨 backbone 的 SOTA 排名。
ADB 与 Trainable-K1 的五 seed 配对 bootstrap 推断见 `results/analysis/adb_paired_inference_v1/` 和
`figures/adb_paired_inference_v1/`；该分析只量化同 split 配对工作点差异，不改变外部合同边界。
StackOverflow/KIR=.50 的三 seed 逐样本错误预算已完成：`docs/analysis/TRAINABLE_VS_ADB_ERROR_BUDGET_V1.md`；
ADB 测试快照、protocol view 和 Trainable `sample_id` 顺序全部通过对齐审计，共 18,000 条记录。结果显示
Trainable 少 2.90 个百分点的 Known 条件误拒，但多 0.66 个百分点的 OOS 条件误接收；这是工作点交换，不是无条件超过 ADB。
跨数据集扩展也已完成：45/45 个 `dataset×KIR×seed` 单元、共 221,700 条协议测试记录通过文本顺序、标签映射、
sample-id 和 manifest SHA256 对齐审计。Trainable 相对 ADB 的机制方向随数据集变化：CLINC150 的 Known
条件误拒增加约 16--17pp、OOS 条件误接收减少约 10--13pp；Banking77 分别增加约 6--8pp、减少约
16--18pp；StackOverflow 则减少约 1--3pp Known 误拒，但在 KIR=.50/.75 的 OOS 误接收略增约
0.7--1.1pp。该证据说明 Trainable 的优势不是单一的“全面降低错误”，而是数据集相关的覆盖--拒识工作点变化；
详见 `docs/analysis/TRAINABLE_VS_ADB_CROSS_DATASET_ERROR_BUDGET_V1.md` 和
`figures/trainable_vs_adb_cross_dataset_error_budget_v1/`。
将性能差值与错误预算合并后的机制表见 `docs/analysis/TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md`；它把
CLINC150/Banking77 标记为“保守拒识”，把 StackOverflow 标记为“覆盖恢复但中高 KIR 有误接收权衡”，
避免把一个全局平均数误写成统一机制。对应图位于 `figures/trainable_vs_adb_mechanism_summary_v1/`。
OOS precision/recall 的状态重算见 `docs/analysis/TRAINABLE_VS_ADB_OOS_DECOMPOSITION_V1.md`；该分析显示
CLINC150/Banking77 的 OOS F1 增益主要来自 recall 增加，而 StackOverflow 的中高 KIR 增益更多来自
precision 改善并伴随轻微 recall 下降。
新增意图级错误归因见 `docs/analysis/TRAINABLE_VS_ADB_INTENT_ERROR_V1.md`；它把 Known 误拒按真实意图、
OOS 误接收按预测吸收意图分开，三个数据集均显示差异分散在多个意图，不能归因于单一类别。
DA-ADB 最新隔离 CUDA 单格（StackOverflow/KIR=.50/seed=42）已完成逐样本审计：OOS F1=`70.82%`、
F1-All=`72.03%`、Known Recall=`72.93%`，预测无 NaN/Inf 且未塌缩为单一类别；证据见
`docs/analysis/DA_ADB_GPU_RUNTIME_SINGLE_CELL_V1.md`。该结果是有效的 BERT/TextOIR 外部兼容 cell，
但仍不是 MiniLM fair 主表，且不能用来覆盖旧 seed=0 的 `90.90%` 兼容单格；两者差异需要单独的
数据/seed/环境审计，当前不做 DA-ADB 的正式跨合同排名。

若要看“为什么当前 Trainable-K1 在 fair matrix 中更好、MOGB-Fair 的低误接受为何伴随高 Known 拒绝、外部
DCLOOS/ADB 为什么不能直接混排”，请读 `docs/analysis/MECHANISM_CLOSURE_V1.md`；其四张机制图和来源哈希
位于 `figures/mechanism_closure_v1/`、`results/analysis/mechanism_closure_v1/MANIFEST.json`。

当前 Cascade 与 MOGB/外部方法的合同分层可视化入口为
`docs/analysis/COMPARISON_ATLAS_V2.md`；它包含当前 protocol_v2 Cascade bridge、同 MiniLM MOGB
组件和 MOGB 论文/本地差距图，不构造跨合同 SOTA 排名。
当前五 seed 可视化证据的唯一收口入口为
`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V2.md`；它替代早期图索引中的旧三 seed 数字，
并将性能、MOGB 机制、Cascade 和外部合同分开。
MSP 的最新外部探测仅完成了数据合同 dry-run，运行时在环境探针阶段超时，没有指标；详情见
`docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`，因此 MSP 仍不进入任何比较排名。
Cascade 的逐样本 Gate/Expert 误差预算见 `docs/analysis/CASCADE_ERROR_BUDGET_V1.md`；该分析显示
Trainable 的主要收益仍来自 Gate 降低 OOS false acceptance，而不是 Expert 单独创造 OOS 信号。
三数据集当前协议 Cascade 桥接见 `docs/analysis/CASCADE_BRIDGE_CROSS_DATASET_V1.md`；在同一
Known-only Expert 下，Trainable K=1 相对 Frozen K=1 的 OOS F1 配对差值为 Banking77 `+5.18pp`、
CLINC150 `+1.12pp`、StackOverflow `+9.42pp`，FA 分别下降 `10.00pp`、`2.86pp`、`15.40pp`。
这仍是每数据集 3 seed 的当前协议下游证据，不是与 MOGB 论文/DCLOOS 的跨合同 SOTA 排名。

## 已完成且不得重复

| 阶段 | 状态 | 证据 |
|---|---|---|
| E0 | complete：3 数据集、165 registry、165 views、990 exports、runtime independence | `docs/audits/protocol_v2_implementation/` |
| E1 | complete：36/36 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e1_gate_smoke.csv` |
| E2 | complete：1,650/1,650，0 failed/missing/duplicate/invalid | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/` |
| E3 | complete：720 partition-control、180 诊断组 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/` |
| R1/M1 | 已完成但按 contract audit superseded；不得重跑 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_*`、`minilm_training_and_stackoverflow_repair_v1/` |
| MOGB/ADB/DA-ADB/DCLOOS | 已审计或完成隔离单元；不扩展旧矩阵 | `docs/archive/mogb_reproduction/`、`docs/archive/external_baselines/` |
| λ 泄漏与敏感性审计 | complete：9 个 split 审计、216 行评分（18 个唯一中心拟合；论文默认 K 行复用 K=2） | `results/diagnostics/lambda_leakage/`、`results/diagnostics/lambda_sensitivity/` |
| RC-AMBL adaptive_v1 pilot | complete：StackOverflow/KIR=.50、3 seeds、KnownOnly/ProxyOOS 共 6/6；0 failed/missing/duplicate/invalid；全部分裂安全回退 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/` |
| joint_adaptive_multicenter_v1 | complete：StackOverflow/KIR=.50、3 seeds；MiniLM、projection 和 intent prototypes 共同训练；3/3 候选 split 由 Known calibration 拒绝，最终 mean `K_y=1.0` | `../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1/repair6/` |
| joint_adaptive_multicenter_contract_repair_v1 | complete：冻结 K=1 父边界、guarded score、负载/分离约束；3/3 候选实际训练后均被 Known calibration 拒绝，最终 mean `K_y=1.0` | `../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1/repair3/` |
| consistency_gate_v1 | complete：复用 Trainable K=1 checkpoint；原始、MC-dropout 和表面归一化多视图；3/3；Known-only 选择证据 margin/冲突容忍度 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/consistency_gate_v1/` |
| minilm_trainable_control_v1 | complete：CLINC150/Banking77 新增 6/6，StackOverflow 复用同协议 RACAL K=1；KIR=.50、3 seeds | `docs/analysis/MINILM_TRAINABLE_CONTROL_V1.md` |
| cascade_bridge_cross_dataset_v1 | complete：CLINC150/Banking77 新增 12/12，与 StackOverflow 既有 6 个桥接单元合并分析；同一 Known-only Expert 下 Trainable/Frozen 配对与 Gate→Cascade 错误预算已完成 | `docs/analysis/CASCADE_BRIDGE_CROSS_DATASET_V1.md` |
| minilm_trainable_k2_control_v1 | complete：同一 Trainable checkpoint 的 K=1/K=2 跨数据集配对，9/9（含 StackOverflow 只读行） | `docs/analysis/MINILM_TRAINABLE_K2_CONTROL_V1.md` |
| minilm_trainable_lambda_control_v1 | complete：3 数据集×3 seeds×2 K×6 λ，108/108；Known-only 选择，不用 test OOS | `docs/analysis/MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md` |
| minilm_trainable_kir_sweep_v1 | complete：3 数据集×3 KIR×3 seeds、Trainable K=1，27/27；同距离 Frozen 对照 | `docs/analysis/MINILM_TRAINABLE_KIR_SWEEP_V1.md` |

## 2026-08-06 新增分析证据

## 2026-08-08 MOGB 与 Trainable MiniLM 同工作点诊断

- 新增 `MOGB_OPERATING_POINT_VISUALS_V1.md` 与 4 张图，把 315 个已有 protocol_v2 fair run 在 Known Recall≈0.75/0.85/0.90/0.95 的事后工作点对齐；没有新增训练，也没有改变正式 E2/E3/R1 结果。
- 在 KIR=.50、Known Recall≈.85 下，Trainable MiniLM K=1 的 OOS F1/false acceptance 分别为：CLINC150 `0.9164/0.0715`、Banking77 `0.8241/0.1964`、StackOverflow `0.8705/0.1132`；StackOverflow Frozen K=2 为 `0.6692/0.4216`，MOGB-MiniLM 为 `0.6860/0.3913`，MOGB partition + ours boundary 为 `0.7209/0.3413`。
- 诊断强化了现有机制判断：Trainable MiniLM K=1 的收益来自更好的分数排序和更低的 open-space false acceptance；固定多中心/MOGB 组件在 StackOverflow 的主要问题仍是 acceptance-region 过覆盖。
- 这些阈值来自 test-known 分数事后对齐，只能作为可视化诊断，不是正式调参或 SOTA 排名；F1-All/F1-K/Accuracy 因 compact prediction 文件不含完整意图映射而未重新计算。

## 2026-08-09 S2C--MOGB-Fair 逐意图结构桥接

- `s2c_mogb_intent_structure_bridge_v1` 已完成：1,850 条 intent×seed 桥接记录、45 个单元、663 个
  intent 汇总、90 个结构相关结果和 6 张图；没有新训练、参数选择或历史 artifact 修改。
- 九个 dataset×KIR 组的正向 Known 恢复 intent 比例均为 100%；S2C 相对 MOGB-Fair 的平均逐 intent
  Known 拒绝率降低 `46.70pp`。缺少 selected ball 的行仅占 `1.41%`，所以缺类不是主要全局解释。
- ball count 与 Known 恢复的 Spearman rho 为 `-0.382--+0.445`，恢复贡献也不集中在少数 intent。
  当前差距应解释为广泛的表示排序与边界覆盖差异，不解释为统一的“粒球越多越好”。
- 证据：`docs/analysis/S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md`、
  `results/analysis/s2c_mogb_intent_structure_bridge_v1/`、
  `figures/s2c_mogb_intent_structure_bridge_v1/`；manifest SHA256=
  `2b139c5b23fea877e5c01cf1ba73ed8e65b6c9f3088c72fdc77d968b45ccc0f9`。

## 2026-08-09 MOGB Known-only 校准与子中心损失归因

- 新增独立实验 `mogb_known_calibration_attribution_v1`：3 数据集×3 KIR×5 seed，共 45 个粒球重拟合单元、225 个 Known-only 边界工作点和 270 个损失契约评价单元。
- 45/45 单元与冻结 `mogb_minilm` 参考 score/指标等价，最大指标绝对误差为 0、score 最大绝对误差为 `4.44e-16`、accepted-label mismatch 为 0；因此结果不是缓存错位或重实现漂移。
- MOGB mean-radius 默认工作点系统性过窄。Calibration-95 将 Known Recall 提高约 61–68pp，但 false acceptance 增加约 67–79pp，OOS F1 下降约 33–38pp；边界工作点失配是真实问题，但单纯放大半径不能恢复论文级综合性能。
- 在实际 MOGB-Fair 类别距离表上，官方 `L1-normalize distances → softmax(-distance)` 的平均真类概率/梯度范数为 `0.0601/0.0502`，raw-distance/tau=.10 为 `0.6389/3.8283`。这验证了子中心损失训练信号被压缩，而不是只凭代码阅读推断。
- 粒球筛选在 45 单元中会出现 Known 类没有 selected ball：平均缺失类数为 Banking77 `0.33`、CLINC150 `0.73`、StackOverflow `0.67`，最坏 StackOverflow 单元排除 2,400 条训练行；这是额外的系统性 Known 覆盖风险。
- 与预注册 MOGB cal-80 工作点同 seed 比较，当前 S2C Trainable K=1 在 OOS F1 和 F1-All 上均为 45/45 胜出。该结论只适用于当前 TEXTOIR-compatible、Known-only、MOGB-Fair 组件合同，不能写成超过官方 BERT MOGB 论文。
- 证据：`docs/analysis/MOGB_KNOWN_CALIBRATION_ATTRIBUTION_V1.md`、`results/analysis/mogb_known_calibration_attribution_v1/`、`figures/mogb_known_calibration_attribution_v1/`；manifest SHA256=`5e2d90bf4a793fcf19af240610f04245cab505ecdf63936853ca4fd20174656b`。

## 2026-08-09 MOGB 粒球筛选缺类回补归因

- 新增独立实验 `mogb_selected_class_rescue_v1`：45 个 dataset×KIR×seed 粒球重拟合单元、180 个评分单元；只在 selected balls 完全遗漏某个注册 Known 类时，用该类 `train_known` 样本补一个欧氏平均半径单中心球。
- 16/45 单元存在缺类（Banking77 3、CLINC150 8、StackOverflow 5）；29 个无缺类单元为严格零变化对照。回补后的结构和 cal-80 工作点均不使用 test Known/OOS 选择。
- 默认窄半径下，StackOverflow 受影响单元的 Known Recall 平均恢复 `+8.40pp`，但 false acceptance 增加 `+6.23pp`、AUROC 下降 `-13.87pp`；累计恢复 1,726 个 Known，同时新增误收 580 个 OOS。
- 在各结构分别使用 Known-calibration 80% 工作点后，StackOverflow 回补使 OOS F1 `-15.66pp`、false acceptance `+18.61pp`。这证明缺类是实际 Known 覆盖缺陷，但简单补球会改变全局排序并扩大开放空间风险，不能解释或消除 S2C 与 MOGB-Fair 的主要差距。
- 当前 S2C Trainable K=1 相对回补后的 MOGB cal-80 仍在 OOS F1/F1-All 上 45/45 胜出。该结论仍只属于同 TEXTOIR split、Known-only、MOGB-Fair 组件合同，不是完整官方 BERT MOGB 排名。
- 证据：`docs/analysis/MOGB_SELECTED_CLASS_RESCUE_V1.md`、`results/analysis/mogb_selected_class_rescue_v1/`、`figures/mogb_selected_class_rescue_v1/`；manifest SHA256=`0968ada93bcbdb04b19bc5b733c4451267feba357cdaf1820f1369152717a05e`。

## 2026-08-09 MOGB 逐粒球开放空间风险归因

- 新增 analysis-only 阶段 `mogb_ball_risk_attribution_v1`：覆盖 3 数据集×3 KIR×5 seed×2 种共享粒球分区的方法，共 `90/90` 方法单元、`8,930` 条 selected-ball 记录，0 失败。
- `MOGB-Fair` 与 `MOGB partition + S2C boundary` 的粒球 ID、标签、训练支持、深度和纯度逐单元完全一致；逐样本重新归因的 OOS assignment、false acceptance 和 Known false rejection 与冻结 `balls.jsonl` 计数完全一致。
- 每个单元风险最高的 10% 粒球承担约 `89%--98%` 的 OOS false acceptance；但 MOGB-Fair 的绝对 false-accept rate 很低，因此该比例表示错误集中度，不等于总误接收量很大。
- tiny ball（训练支持小于 20）占 `36%--51%`，但 Q3/Q4 大粒球承担了更多错误预算；selected-ball 纯度几乎全为 1。由此排除“只删小球”或“只按 Known 纯度筛球”作为可靠修复。
- 在完全相同粒球分区上换用 S2C 边界可恢复 Known coverage/F1-All，但也提高 false acceptance，且仍落后于 Trainable K1。三数据集上 Trainable K1 相对 MOGB-Fair 的平均 OOS F1/F1-All 差值为 CLINC150 `+9.61/+35.14pp`、Banking77 `+10.31/+32.48pp`、StackOverflow `+17.17/+43.67pp`。
- 结论：当前差距来自表示排序、粒球划分与边界工作点的联合失配，不能归因于单一 tiny-cluster、缺类或半径缺陷；test OOS 粒球风险只作事后解释，禁止用于删球或选参。
- 证据：`docs/analysis/MOGB_BALL_RISK_ATTRIBUTION_V1.md`、`results/analysis/mogb_ball_risk_attribution_v1/`、`figures/mogb_ball_risk_attribution_v1/`；manifest SHA256=`aca7b154c60aedbe964c4582c9bfa0d5acd02ab0cd6640ae24a59ff64e047e28`。

## 2026-08-09 S2C 与 MOGB-Fair 开放意图五状态转移

- 新增 `trainable_mogb_open_intent_transitions_v1`：对齐三数据集×3 KIR×5 seed 的 `45/45` 配对单元，读取 `443,400` 条冻结预测，将每条样本分为 Known 正确、Known 错类、Known 被拒、OOS 正确拒绝、OOS 误接收五种互斥结果。
- sample_id、gold_intent、Known/OOS 标记逐单元完全一致；F1-All 重新计算与冻结汇总的最大绝对误差为0。没有训练、调阈值、改变split或输出逐样本ID到轻量结果。
- S2C-Trainable-K1 相对 MOGB-Fair 的平均 F1-All 增量为 Banking77 `+32.48pp`、CLINC150 `+35.14pp`、StackOverflow `+43.67pp`。
- 差距主要来自 Known 覆盖：每单元平均净增正确 Known 为 Banking77 `+734.87`、CLINC150 `+958.73`、StackOverflow `+1734.93`；平均净 OOS 正确拒绝反而分别为 `-192.20/-96.13/-169.80`。S2C 不是比 MOGB 更保守，而是以有限 OOS 拒识损失换取更大的 Known 正确分类恢复。
- 在 KIR=.50，S2C 正确分类而 MOGB 拒绝的 Known 样本平均为 Banking77 `713.8`、CLINC150 `953.8`、StackOverflow `1681.0`；S2C 修正 MOGB Known 错类仅 `0.4/1.0/2.2`。因此当前 MOGB-Fair 差距首先来自平均半径边界覆盖，其次才是粒球间错类竞争。
- 证据：`docs/analysis/TRAINABLE_MOGB_OPEN_INTENT_TRANSITIONS_V1.md`、`results/analysis/trainable_mogb_open_intent_transitions_v1/`、`figures/trainable_mogb_open_intent_transitions_v1/`；manifest SHA256=`ed7cef1d3a208e25d5dd4201de2670059e254fe0ed1a9087823d02f7b71189f7`。

## 2026-08-10 跨 KIR 匹配 Known 覆盖率前沿分析

- 新增 analysis-only 阶段 `cross_kir_matched_frontier_v1`：读取已经冻结的 Trainable-K1/MOGB-Fair operating curves，覆盖 3 数据集×3 KIR×5 seeds×3 个目标 Known Recall（80%、90%、95%），共 270 个方法行、135 个配对单元。
- 在相同 Known Recall 工作点下，S2C 的 OOS F1 在全部 27 个 dataset×KIR×target 组合中均高于 MOGB-Fair，且每个组合都是 5/5 seed 配对胜出；跨三个目标覆盖率的平均提升范围约为 CLINC150 `15.45–25.67pp`、Banking77 `16.75–24.92pp`、StackOverflow `20.36–32.00pp`。
- 这强化了“Trainable 的优势来自分数排序/表示分离，而不是默认阈值偶然有利”的解释；同时仍属于事后 Known-coverage 诊断，不能作为阈值选择或新的 SOTA 排名。
- 图：`figures/cross_kir_matched_frontier_v1/matched_frontier_delta_heatmaps.png`、`matched_frontier_oos_f1_curves.png`；报告：`docs/analysis/CROSS_KIR_MATCHED_FRONTIER_V1.md`。

## 2026-08-10 跨 KIR 错误预算归因

- 新增 analysis-only 阶段 `cross_kir_transition_attribution_v1`：读取已审计的 Trainable-K1/MOGB-Fair 45 个逐样本配对单元，覆盖 3 数据集×3 KIR×5 seeds，共 9 个汇总单元。
- 九个 dataset×KIR 组的 F1-All 差值和 Known 正确净增均为正，OOS 正确净增均为负；跨三个 KIR 的平均 F1-All 提升为 CLINC150 `35.14pp`、Banking77 `32.48pp`、StackOverflow `43.67pp`。
- 主要来源是恢复 MOGB 判为 `Known rejected` 的样本：跨 KIR 平均恢复比例为 CLINC150 `41.41%`、Banking77 `46.52%`、StackOverflow `56.14%`。这支持“Trainable 的优势主要是 Known 覆盖和分数排序恢复”，而不是更激进地接受 OOS。
- 输出：`docs/analysis/CROSS_KIR_TRANSITION_ATTRIBUTION_V1.md`、`results/analysis/cross_kir_transition_attribution_v1/`、`figures/cross_kir_transition_attribution_v1/`；不修改训练、split、阈值或历史 artifact。

## 2026-08-10 跨 KIR 多指标 Pareto 分析

- 新增 analysis-only 阶段 `cross_kir_pareto_frontier_v1`：读取 315 个已完成的 fair Gate 行（7 方法×3 数据集×3 KIR×5 seed），同时分析 OOS F1、F1-All、Known Recall 和 false acceptance。
- Trainable-K1 在全部 9 个 dataset×KIR 工作点都处于四指标均值 Pareto 前沿；这不是显著性检验，而是说明没有其他当前 fair 方法同时在四项指标上支配它。
- Pareto 结果同时显示不存在跨数据集统一的单一支配方法：MOGB-MiniLM/动态边界常以低 false acceptance 换取低 F1-All 和低 Known 覆盖；Frozen/Random 多中心的点位受 union-risk 影响；Trainable 位于较高 F1-All/OOS F1 的平衡区域。
- 输出：`docs/analysis/CROSS_KIR_PARETO_FRONTIER_V1.md`、`results/analysis/cross_kir_pareto_frontier_v1/`、`figures/cross_kir_pareto_frontier_v1/`；没有新增训练或测试选择。

## 2026-08-08 StackOverflow 逐样本错误归因

- 新增 `STACKOVERFLOW_ERROR_ATTRIBUTION_V2.md` 与 4 张图；对齐 7 方法×5 seed×6,000 条预测，共 210,000 条 sample-level 记录，检查 sample_id、gold_intent 和 Known/OOS 划分一致。
- 正式 score<=1 下，Trainable K=1 的 OOS F1/FA/FR 为 `0.8767/0.0934/0.1611`；Frozen K=2 为 `0.6353/0.4717/0.1311`；MOGB-MiniLM 为 `0.7292/0.0079/0.7291`。
- 固定 K=2 的 OOS false acceptance 主要由 `cocoa、sharepoint、osx、spring、scala` 五个 Known intent 吸收；MOGB-MiniLM 则以大量 Known false rejection 换取低误接收。
- 新证据把“Trainable K=1 为什么更好”具体化为错误结构差异：它同时降低 OOS 误接收并保持 Known 覆盖，而不是单纯调低/调高阈值。
- 当前仍不做跨数据集 SOTA 宣称；外部 ADB/DA-ADB/DCLOOS 仍需可用 runtime 后进入统一工作点和错误集合对齐。

- `representation_boundary_pack_v1`：对已有 Frozen/CE/SupCon 的 K=1/K=2 结果生成 18 行汇总、36 行配对效应和 4 张图。StackOverflow 上 CE 的 OOS F1 从 88.13% 降至 73.44%，SupCon 从 89.63% 降至 71.90%，而 Known Recall 分别上升 8.12pp 和 9.73pp；表示训练改善 K=1，不等于多中心安全。证据：`docs/analysis/REPRESENTATION_BOUNDARY_PACK_V1.md`、`results/analysis/representation_boundary_pack_v1/`。
- `stackoverflow_intent_diagnostic_v1`：对已有 RACAL Stage-2 的 30 个 intent-seed 诊断行做按意图汇总和 4 张图。平均恢复 29.73 个 Known、却新增接受 102.53 个 OOS，净收益 -72.80；平均 bootstrap ARI 0.91，说明稳定聚类仍可能造成 OOS 过接受。证据：`docs/analysis/STACKOVERFLOW_INTENT_MULTI_CENTER_DIAGNOSTIC_V1.md`、`results/analysis/stackoverflow_intent_diagnostic_v1/`。
- 两个分析包均为 analysis-only，不新增训练、不修改 E2/E3/RACAL 历史 artifact，也不使用 test OOS 选择参数。
- `intent_kir_stability_pack_v1`：读取已有 13,580 条 intent-level K/KIR 敏感性审计行，生成 66 个
  dataset×KIR×distance 汇总、4 张图和 seed 稳定性表。oracle 结果显示 Banking77 的多中心候选比例
  最高、StackOverflow 最低且平均收益接近 1pp；这些是 test-sensitivity 诊断，不能直接作为 K 选择规则。
  证据：`docs/analysis/INTENT_KIR_STABILITY_PACK_V1.md`、`results/analysis/intent_kir_stability_pack_v1/`。

## 已确认结论

- 不存在跨数据集统一最优 `K`；Banking77 的多中心收益是条件性的，CLINC150 弱，
  StackOverflow 的固定 KMeans 多中心显著退化。
- E3 表明 KMeans 稳定性不等于 OOS 有效性；StackOverflow 更符合 boundary-union
  过覆盖，而非单纯 tiny-cluster 失败。
- R1 contract repair 未证明 Geometry loss 能救活固定多中心；其 near-OOS 字段在
  当前 Known-only calibration 下只能作为 exploratory。
- MOGB 官方 BERT 逻辑已有 StackOverflow KIR=.50/seed=0 和 Banking KIR=.75/seed=0
  的严格负复现证据，论文数字不可直接混入主表。四组合定位还缺作者配套数据，故分类为
  `public_code_not_reproduced_under_available_materials`。
- 固定 K 逐样本审计覆盖 1,650/1,650 run、13,580 个 intent-seed 诊断行；Banking77
  的 dataset-level test oracle 常选 K=4/5，CLINC150 多数为 K=1/2，StackOverflow 的
  22 个 dataset×KIR×distance 组合全部选 K=1。intent-level 结果是异质的：在同一
  intent×KIR×distance 组内，K>1 至少一次胜过 K=1 的组数为 Banking77 1,096/1,476、
  CLINC150 1,460/2,794、StackOverflow 148/376；任意 K>1 在至少 3/5 seeds 胜出的
  intent×KIR×distance 组数分别为 Banking77 509、CLINC150 694、StackOverflow 73。
  因此这些行只能说明候选结构，且明确标为 test-sensitivity，不能当作选择规则。
- Meta-review 所要求的直接基线、五 seed 统计、K/λ 消融和泄漏证明被列为后续论文证据
  backlog；本轮只整理已有 MOGB/ADB/DA-ADB/DCLOOS 证据，不重跑这些冻结单元。
- RC-AMBL pilot 的 6/6 单元均完成且没有测试选择泄漏，但 3 个 seed 的候选分裂均因
  bootstrap median ARI（0.7051--0.7712）低于 0.80 或 Known-only 安全门而拒绝；10 个
  Known intent 最终全部为 `K_y=1`。RC-AMBL OOS F1 `0.5785±0.0926`，相对 E2 K=1
  下降 `19.44pp`，false acceptance 增加 `29.14pp`，不能称为新方法成功。
- joint_adaptive_multicenter_v1 的 `repair6` 是第一次真正把训练参与扩展到多中心候选：从 RACAL
  Trainable K=1 checkpoint 初始化，MiniLM 最后两层、残差 projection 和 intent prototypes 一起
  训练候选 split；候选只由 Known train/ calibration 决定。StackOverflow/KIR=.50 的 3 个 seed
  均完成，3/3 候选 split 被拒绝，最终 `mean K_y=1.0`。joint adaptive OOS F1 为
  `0.8661±0.0111`，F1-All `0.8563±0.0050`，Known Recall `0.8388±0.0045`，false acceptance
  `0.1129±0.0228`；这是真正的训练链路负诊断，不是 adaptive-K 正结果。
- `joint_adaptive_multicenter_contract_repair_v1/repair3` 修复了两个契约问题：候选分裂始终继承冻结的
  K=1 父边界，且 compactness 使用 parent-guarded score；训练损失显式加入子中心负载平衡和中心分离项。
  StackOverflow/KIR=.50 的 seed 13/42/87 均完成候选训练，3/3 因 Known calibration Recall 下降而拒绝，
  最终 `mean K_y=1.0`。结果为 OOS F1 `0.8661±0.0091`、F1-All `0.8563±0.0041`、Known Recall
  `0.8388±0.0037`、false acceptance `0.1129±0.0187`；没有产生安全的 `K_y>1`。这修复了评估合同，
  但没有改变 StackOverflow 上多中心候选的负结论。
- `consistency_gate_v1` 在不增加中心、不重新训练 encoder 的前提下复用 Trainable K=1，加入原始、两次
  MC-dropout 和表面归一化视图的一致性/证据 margin Gate。3/3 seed 完成；evidence-margin 变体 OOS F1
  `0.8673±0.0076`、F1-All `0.8580±0.0027`、Known Recall `0.8376±0.0020`、false acceptance
  `0.1099±0.0145`，相对 Trainable K=1 仅为描述性小幅变化，不能称 SOTA。
- `minilm_trainable_control_v1` 说明最后两层 MiniLM+projection 的 Known-only 适配能改善 K=1：相对同 seed
  Frozen，CLINC150/Banking77/StackOverflow 的 OOS F1 分别为 `+1.12pp/+5.18pp/+9.42pp`，但
  CLINC150/Banking77 Known Recall 分别下降 `1.33pp/1.95pp`。
- `minilm_trainable_k2_control_v1` 在同一 checkpoint 上显示 K=2−K=1 OOS F1 为 `-0.28pp/+0.13pp/-19.06pp`
  （CLINC150/Banking77/StackOverflow），因此表示训练收益不能推出固定多中心安全。
- `minilm_trainable_lambda_control_v1` 完成 108/108 个 λ/K 评价。Known-only 选择后 K=2−K=1 OOS F1 为
  CLINC150 `+0.79pp`、Banking77 `+2.08pp`、StackOverflow `-9.51pp`；StackOverflow false acceptance
  仍增加 `11.83pp`。部分 K=1 的 Known calibration 5% false-reject 约束在 λ≤2 网格内不可满足，
  因此 λ=2 仅是诊断上限，不能称正式最优。
- `minilm_trainable_kir_sweep_v1` 完成 27/27 个 Trainable K=1 单元。相对同一对角马氏距离的 Frozen K=1，
  CLINC150 在 KIR=.25/.50/.75 的 OOS F1 分别 `+0.64/+2.41/+3.59pp`，StackOverflow 分别
  `+1.19/+7.69/+13.54pp`；Banking77 仅在 `.25` 为 `+0.59pp`，`.50` 近似持平，`.75` 为 `-14.02pp`。
  这说明当前可训练 MiniLM 的主要增益是 K=1 分数排序，并非跨数据集或跨 KIR 的 SOTA 结论。

## λ 选择与数据泄漏审计（2026-08-02）

- 当前 `protocol_v2_textoir_v1` 的 E2 `mean_std` 运行固定 `lambda=1.0`，不从 test OOS
  选择；历史 v19 tuned runner 和旧 corrected runner 确实在 validation 行上搜索 λ，均已
  标为 `validation_oos_selected`，不得与当前 Known-only 主协议混写。
- 9 个 `dataset×seed` split 的 train/calibration/test-known/test-OOS 样本 ID 均不重叠；当前
  没有合法的 validation OOS 或 validation unknown-intent split，因此主审计政策是
  `known_only_calibration`，`test_used_for_selection=false`。
- λ 网格为 `{0,0.25,0.5,0.75,1,1.25,1.5,2}`，使用相同冻结 embedding、中心和对角马氏
  距离。λ=1 的 18 个 E2 对照均达到 `max_abs_delta=0` 且预测不匹配为 `0`。
- Known-only 的 5% calibration false-rejection 守门在当前网格内并不总能满足：CLINC150
  的 K=1/K=2、Banking77 的 K=2、StackOverflow 的 K=1 到 λ=2 仍未满足。因此 λ=2
  只能作为“网格上限且约束未满足”的诊断结果，不能宣称为合法最优边界。
- 在 Known-only 选择的诊断条件下，Banking77 K=2 相对 K=1 的平均 OOS F1 增量为
  `+0.04395`、F1-All 增量为 `+0.00919`、Known Recall 增量为 `-0.00702`；它未达到
  预注册的 F1-All `+0.01` 门槛，且缺少合法的 intent-level Known-only 正收益比例。
  StackOverflow K=2 平均 OOS F1/F1-All 分别下降 `-0.26054/-0.07648`，false acceptance
  增加约 `+0.24056`，继续支持 boundary-union 过覆盖方向。
- 因此 `split_merge adaptive-K pilot` 当前未获授权；已有 intent-level 行仍是
  test-oracle 描述性证据，不能升级为选择器或新方法结论。
- λ 审计摘要已加入 `configs/public_results.yaml`；公开白名单校验通过，102 个轻量文件共
  13,820,620 bytes，未包含原始文本、embedding、checkpoint 或逐样本 score。

## URCSG pilot（2026-08-04）

- `urcsg_pilot_v1` 已在固定 `protocol_v2_textoir_v1`、KIR=.50、Banking77/StackOverflow、
  seeds 13/42/87、`mahalanobis_diag`、`mean+lambda*std (lambda=1.0)` 下完成 6/6 cell。
- 选择器只使用 proper-train 和 Known calibration 的 leave-one-known-intent 伪 OOS 风险，
  并保留 shuffled episode negative control；冻结 all-MiniLM cache，没有重新编码或训练。
- Banking77 的 URCSG-primary 相对 single-centroid：OOS F1 `-0.39pp`、F1-All `-0.35pp`、
  Known Recall `-0.20pp`；StackOverflow：OOS F1 `-6.10pp`、F1-All `-2.67pp`，false acceptance
  增加 `+10.93pp`。两数据集预注册门槛均失败，未进入 full matrix。
- 产物：`results/diagnostics/urcsg/{pilot_summary.csv,intent_selection.csv,mechanism_analysis.csv,shuffled_control.csv,decision.json}`；
  详细输入和 run manifest 在 `../artifacts/s2c/runs/protocol_v2_textoir_v1/urcsg_pilot_v1/`。

## DCLOOS 端到端基线合同收口（2026-08-10）

- 官方代码和外部 SQuAD 负样本快照已接入隔离 runner；负样本 SHA256 为
  `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。
- 当前 StackOverflow/KIR=.50/seed=42 registry 固定 10 个 Known intent；DCLOOS 在同一 train.tsv 上按
  自身随机规则重新选 5 个 Known intent，故不能称 `same_protocol`，只能称
  `dcloos_adapted_protocol_migration`。
- 一次 1 epoch 的隔离 smoke 已完整落盘 metrics/predictions：OOS F1=0、F1-All=8.31%。这是低预算
  链路验证，不是性能结果；长预算与后续 smoke 的失败 artifact 均保留，未进入 fair matrix。
- 运行时 overlay 定位并记录了旧 DCLOOS 的标量 `.item()`、schedule 越界、AdamW/tokenizer/本地 BERT
  兼容问题；第三方 checkout 未修改。
- 详见 `docs/analysis/DCLOOS_CONTRACT_STATUS_V1.md`。

### DCLOOS 固定 registry 资源阻塞更新

- 另启动 `dcloos_stackoverflow_kir050_seed42_fixed_registry_reduced10_v1`（固定 Known list、10 epoch、
  patient=3）；运行约 1,246 秒后仍未生成最终 `metrics.json`，确认当前环境中的 DCLOOS 子进程实际
  走 CPU 路径，随后停止并保留 `run_manifest.json`，状态为 `timeout_incomplete`。
- 其中的 `predictions.npz` 仍是中间 validation-best 预测，禁止进入汇总、可视化或任何性能结论。
- 因此 DCLOOS 当前仍只有不同合同的 reduced 参考结果，没有可用于当前 registry 的最终指标；这不是
  “DCLOOS 性能为零”，而是外部端到端基线的资源/运行合同未闭合。

## Trainable Gate 与旧 Cascade 合同审计（2026-08-10）

- 新增 `cascade_trainable_contract_gap_v1`，只审计当前 `protocol_v2_textoir_v1` 的 Trainable K=1
  checkpoint 与旧 v19 Router/Expert 的数据、split、sample ID 和组件来源，没有运行不合法的混合 Cascade。
- 当前 Trainable StackOverflow/KIR=.50/seed=42 为 train/calibration/test=`6000/1000/6000`，test
  Known/OOS=`3000/3000`；旧 v19 Cascade 为 `5995/1998/5990`，test Known/OOS=`2993/2997`，且旧
  gate JSON 没有当前 `sample_id`。因此直接替换 Gate 会同时改变下游训练输入和评价集合，不能作为
  protocol_v2 的端到端结果。
- 当前必须保持的分层是：Trainable K=1（当前协议 Gate-only）、旧 v19 Frozen/CE-Recon Cascade、
  `fulltex.tex` 历史 Cascade、外部 MOGB/ADB/DCLOOS；四者不能合并排名。
- 证据：`docs/analysis/CASCADE_TRAINABLE_CONTRACT_GAP_V1.md`、
  `results/analysis/cascade_trainable_contract_gap_v1/`、
  `figures/cascade_trainable_contract_gap_v1/split_count_contract_gap.png`。

## 当前协议 Cascade bridge 已完成（2026-08-10）

- 新增 `cascade_bridge_v1`：在当前 `protocol_v2_textoir_v1` 的 StackOverflow/KIR=.50、
  seeds=13/42/87 上，各 seed 只用 `train_known` 训练一个 SmolLM Expert，并用
  `calibration_known` 选择 checkpoint；同一 Expert 分别接 Frozen K=1 与 Trainable K=1 Gate。
- 完成 6/6 评价行，0 failed/missing/duplicate；`test_used_for_selection=false`、
  `oos_used_for_training=false`。StackOverflow 只有单一 domain，因此 Router 是常量路由，未伪造多领域结果。
- 三 seed 均值：Frozen K=1 的 OOS F1/F1-All/Known Recall/FA 为
  `77.29%/76.55%/83.71%/26.54%`；Trainable K=1 为
  `86.71%/83.25%/83.92%/11.14%`。配对差值为 OOS F1 `+9.42pp`、F1-All `+6.70pp`、
  Known Recall `+0.21pp`、FA `-15.40pp`。
- 该结果确认 Trainable Gate 的收益能传递到同协议下游，但仍不是 `fulltex.tex` 历史 Cascade、
  官方 BERT MOGB 或 DCLOOS 外部监督结果；对应报告为 `docs/analysis/CASCADE_BRIDGE_V1.md`。

## 当前唯一下一步

停止 URCSG、CCSG、RC-AMBL、joint-adaptive、固定多中心和新表示损失扩展；这些阶段继续作为
Known-only 自适应 K 与 StackOverflow union-risk 的负结果证据。MOGB corrected-loss 单格已经完成，
证明公开 L1 归一化子中心损失是复现差距的一项来源，但 F1-All 仍比论文参考低 14.85pp。

corrected-loss BERT checkpoint 的 Known-only 半径覆盖归因现已完成。确定性重建的默认 mean-radius
工作点仅有 58.63% Known Recall；dev-known 80% 覆盖校准将其提高到 77.00%，但 F1-U 从79.76降到
73.45。95%覆盖时 Known Recall 达87.17%，F1-U却降到35.68，OOS误接受从197增到2311。因此默认
半径过窄是差距来源之一，但放大半径不能恢复论文级权衡。

当前唯一下一步是整理并审计同协议的 Cascade bridge、fair Gate、MOGB 组件和外部 baseline 的
分层主表与可视化；在该分层收口前不扩展其他数据集的 Cascade、不复用 v19 Router/Expert、不把
reduced/smoke 数字称为严格复现，也不扩新的 adaptive-K 或 Cascade 矩阵。

补充定义：当前 s2c 还不能称为“已完成的自适应多中心方法”。E2/E3 是固定 K 的后处理中心，
RACAL 阶段一只训练 K=1 表示，RACAL 阶段二只复用表示做固定 K=2 归因；RC-AMBL 和
joint_adaptive_multicenter_v1 虽然分别实现了冻结表示风险门和训练参与式候选 split，但候选均未
通过 Known-only 安全门。因此后续任何 adaptive-K 论文表述都必须标记为“未完成/待验证”，不能把
固定 K 结果或本轮共同训练诊断误写成自适应多中心成功。

## 当前阻断和风险

- MOGB A/C 组合缺少原始配套数据；不能以 TextOIR 快照冒充原始数据。
- adaptive-K intent-level 指标是从 E2 test predictions 做的描述性 oracle 敏感性审计，
  不能用于正式选 K。
- λ sensitivity 的 Known-only 5% 误拒绝约束在部分数据集/ K 上不可满足；当前结果只能作为
  选择稳定性和泄漏审计证据，不能把 λ=2 解释为跨数据集正式默认值。
- StackOverflow 完整文本、模型、embedding、checkpoint 和逐样本结果仍只保留在本地，
  不进入 GitHub 结果快照。

## 2026-08-10 最新跨方法机制收口

- 新增 `trainable_vs_adb_mechanism_summary_v1`：将 9 个 `dataset×KIR` 的 Trainable−ADB 指标差值与已经完成的逐样本错误预算合并；没有新增训练、阈值选择或测试数据导出。
- CLINC150 和 Banking77 都表现为“保守拒识”：Known 条件误拒增加，但 OOS 条件误接收明显下降；StackOverflow 表现为“覆盖恢复”，其中 KIR=.50/.75 的 OOS 误接收略有回升。
- 这说明当前方法的“更好”不是一个跨数据集统一机制，而是表示适配改变了不同数据集上的覆盖—拒识工作点；当前仍不能据此宣称跨 BERT/MiniLM 合同 SOTA。
- 证据：`docs/analysis/TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md`、`results/analysis/trainable_vs_adb_mechanism_summary_v1/`、`figures/trainable_vs_adb_mechanism_summary_v1/`。
- 原台账中的 `adb_kir_sensitivity_v1` 已修正为 `superseded_incomplete`，明确其 9/27 部分执行由 v2/v3 五 seed sweep 替代，不再显示为正在运行。

## CCSG pilot（2026-08-04）

- 固定 `protocol_v2_textoir_v1`、KIR=.50、三数据集、seeds 13/42/87、冻结 all-MiniLM、
  diagonal Mahalanobis、mean+std radius 和 Known calibration 阈值；每个 cell 同时拟合 K=1/2。
- 比较当前 K=1/K=2 union、mixture-support、margin-only、CCSG joint 和 independent-AND 消融；
  未使用测试 OOS 选择阈值，未重新编码或训练。
- 9/9 cells、72 metric rows、0 failed/missing/duplicate/invalid。CCSG-K2 相对 CCSG-K1 的
  F1-All 增量为 Banking77 -1.26pp、CLINC150 -3.68pp、StackOverflow -0.39pp；StackOverflow
  false acceptance 增量为 +3.00pp，超过 +1pp 安全门。最终决策为 `stop_ccsg_pilot`。
- 证据入口：`results/diagnostics/ccsg/` 和
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/ccsg_pilot_v1/`。

## RACAL-v1 阶段一（2026-08-05）

- RACAL-v1 是独立的新阶段，未修改或覆盖 E2、E3、R1、URCSG、CCSG、BRAK、RC-AMBL 和 MOGB 历史产物。
- StackOverflow、KIR=0.50、seeds 13/42/87 的 frozen K=1 E2 精确回放已完成 3/3：sample_id 与预测零不匹配，score 和指标最大绝对差均为 0。
- Trainable MiniLM K=1 已完成 3/3；仅使用 Known train/calibration，先 projection warm-up，再解冻 MiniLM 最后两层，未使用测试 OOS 选 epoch、边界或阈值。
- 三 seed 均值显示，Trainable 相对 Frozen：OOS F1 +9.42pp、F1-All +7.06pp、Known Recall +0.21pp、false acceptance -15.40pp、AUROC +3.56pp；Trainable OOS F1 标准差为 0.79pp，低于 Frozen 的 4.17pp。
- 阶段一结论：K=1 表示训练是当前可复现且有正收益的方向，允许登记下一阶段，但不得自动启动固定 K=2、中心激活、Proxy-OOS 或其他数据集扩展；当前仍不能声称多中心或 SOTA 已解决。
- 证据入口：`docs/racal_v1/RACAL_V1_REPORT.md`、`docs/racal_v1/RACAL_V1_CLOSEOUT.md`、`results/diagnostics/racal_v1/`、`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/`。

## RACAL-v1 阶段二（2026-08-05）

- 在阶段一同一 Trainable MiniLM checkpoint 上完成 StackOverflow/KIR=.50、seeds 13/42/87 的纯 K=1/K=2 对照；没有重新训练或选择 K=2 表示，也没有加入风险门控或 proxy-OOS。
- K=1 重新编码结果与阶段一指标最大差异为 0；K=2 每个 intent 内使用 KMeans-2，三 seed 完成 3/3。
- K=2 相对 K=1 的均值变化：OOS F1 `-19.06pp`、F1-All `-8.85pp`、Known Recall `+9.70pp`、false acceptance `+34.11pp`、AUROC `-2.30pp`；新增 OOS false acceptance 为 1169/753/1154，恢复 Known false rejection 为 298/309/285。
- 判定为 `A_primary_with_C_heterogeneity`：固定 K=2 明显退化，但 intent-level 存在异质性。停止 K=3--5；RACAL 不停止，但只能登记最小 risk-gated intent 激活，不得自动运行。
- 证据入口：`docs/racal_v1/RACAL_V1_STAGE2_REPORT.md`、`docs/racal_v1/RACAL_V1_STAGE2_CLOSEOUT.md`、`results/diagnostics/racal_v1/stage2_fixed_k2/`、`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/`。

## joint_adaptive_multicenter_v1（2026-08-05）

- 这是本项目第一次真正把多中心候选放进训练闭环：encoder 最后两层、residual projection 和
  intent prototypes 共同优化；候选 split 从 Known train 的 PCA 残差提出，候选模型只用 Known
  train 训练，结构只由 Known calibration 的 recall、compactness、objective 和父边界约束决定。
- `repair6` 固定 StackOverflow/KIR=.50、seed=13/42/87，3/3 完成，0 failed/missing/duplicate/invalid；
  `test_used_for_selection=false` 且 `oos_used_for_training=false`。
- 3/3 候选 split 均被拒绝，最终 10 个 Known intent 全部为 `K_y=1`。均值：OOS F1
  `0.8661±0.0111`、F1-All `0.8563±0.0050`、Known Recall `0.8388±0.0045`、false acceptance
  `0.1129±0.0228`。因此本阶段证明了训练参与式实现存在且执行了候选训练，但没有证明自适应多中心
  在当前 StackOverflow 条件下有收益。
- 证据入口：`docs/joint_adaptive_multicenter_v1/JOINT_ADAPTIVE_V1_REPORT.md`、
  `docs/joint_adaptive_multicenter_v1/REPRODUCE_JOINT_ADAPTIVE_V1.md`、
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1/repair6/`。

## MiniLM Trainable K=1 跨数据集控制（2026-08-06）

- 在当前 `protocol_v2_textoir_v1`、KIR=0.50、Mahalanobis K=1 下新增 CLINC150/Banking77 ×
  seeds 13/42/87，共 6/6 单元；StackOverflow 复用 RACAL-v1 同协议的 3-seed Trainable K=1。
- Frozen 参考逐 seed 直接读取同一 E2 K=1 单元，避免预聚合 CSV 造成 seed 或 runner 口径错配。
- Trainable 相对 Frozen 的 OOS F1 差值：CLINC150 `+1.12pp`、Banking77 `+5.18pp`、
  StackOverflow `+9.42pp`；Known Recall 差值分别为 `-1.33pp`、`-1.95pp`、`+0.21pp`。
- 结论：Known-only 表示适配的 K=1 收益不是偶然只出现在 StackOverflow，但也不是无代价的
  普遍替代方案；CLINC150/Banking77 需要继续做表示范围与阈值校准分析，不能直接扩展 K。
- 证据入口：`docs/analysis/MINILM_TRAINABLE_CONTROL_V1.md`、
  `results/diagnostics/minilm_trainable_control_v1/`、
  `figures/active_experiment_dashboard_v1/trainable_cross_dataset.png`、
  `../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_control_v1/`。

## MiniLM Trainable K=1/K=2 跨数据集配对控制（2026-08-06）

- 在同一 Trainable MiniLM checkpoint、同一 KIR=0.50、同一 Mahalanobis/mean+std/threshold=1 合同下，CLINC150 与 Banking77 新增 6/6 个 K=2 评价单元；StackOverflow 合并 RACAL-v1 Stage2 的 3 个只读配对行。
- K=2−K=1 的 OOS F1：CLINC150 `-0.28pp`、Banking77 `+0.13pp`、StackOverflow `-19.06pp`；Known Recall：`-3.26pp`、`-1.95pp`、`+9.70pp`；StackOverflow false acceptance `+34.11pp`。
- 结论：Trainable 表示能稳定改善 K=1，但不能把固定 K=2 变成跨数据集安全配置；Banking77 仅有微小条件性收益，CLINC150 更支持 K=1，StackOverflow 仍是多球并集过覆盖的结构性失败。
- 证据入口：`docs/analysis/MINILM_TRAINABLE_K2_CONTROL_V1.md`、`results/diagnostics/minilm_trainable_k2_control_v1/`、`figures/active_experiment_dashboard_v1/trainable_k_interaction_cross_dataset.png`、`../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_k2_control_v1/`。

## 当前证据入口

- 可训练 MiniLM 与 `fulltex.tex` 历史 Cascade、MOGB/ADB/DA-ADB 分层解释：
  `docs/analysis/MINILM_TRAINABLE_VS_FULLTEX_AND_BASELINES_V1.md`。
- Trainable K=1 的跨 KIR 结果与图：`docs/analysis/MINILM_TRAINABLE_KIR_SWEEP_V1.md`、
  `results/analysis/minilm_trainable_kir_sweep_v1/`、`figures/minilm_trainable_kir_sweep_v1/`。
- 当前协议的分层轻量总表：`results/analysis/unified_layered_summary_v1/all_layers.csv`；该表不把
  `fulltex.tex` 历史 Cascade 或外部兼容单格混成公平排名。
- 多 KIR、多方法 paired effects 与图表证据包：`docs/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md`、
  `results/analysis/experiment_evidence_pack_v2/`、`figures/experiment_evidence_pack_v2/`。
- 当前方法、实验进展和基线差异总览：`docs/CURRENT_METHOD_AND_EXPERIMENT_STATUS.md`。

- MOGB/DCLOOS 中文对比主报告：`docs/对比实验/MOGB_DCLOOS_对比结果报告.md`；该报告区分同协议 Frozen MiniLM 组件比较、MOGB 官方兼容负复现和 DCLOOS reduced-budget 参考结果。

- 固定 K 审计：`results/diagnostics/adaptive_k/`；
- RC-AMBL：`docs/adaptive_v1/ADAPTIVE_V1_REPORT.md`、`docs/adaptive_v1/REPRODUCE_ADAPTIVE_V1.md`、
  `results/diagnostics/adaptive_v1/`；
- MOGB 四组合审计：`results/diagnostics/mogb_diff/` 和
  `docs/archive/mogb_reproduction/MOGB_DIAGNOSIS.md`；
- λ 数据集契约与敏感性：`results/diagnostics/lambda_leakage/data_split_audit.json`、
  `results/diagnostics/lambda_sensitivity/summary.csv`、`mean_std.csv`、
  `lambda_k_interaction.csv`、`adaptive_k_decision.json`；
- 机器总账：`docs/EXPERIMENT_LEDGER.csv`；
- 历史决策与 claim audit：`docs/archive/protocol_and_data/`、
  `docs/archive/external_baselines/`。

## MiniLM Trainable K=1 五 seed 公平扩展（2026-08-06）

- 在 `protocol_v2_textoir_v1` 下补齐了 Trainable K=1 的 seed=100/123：新增 18/18 个训练与评估单元，
  与原有 27 个单元合计 45/45；范围为三个数据集、KIR={.25,.50,.75}、seeds={13,42,87,100,123}、
  diagonal Mahalanobis、mean+std、threshold=1。
- 每个 seed 的 Frozen 参考直接读取同一 E2 K=1 单元；Trainable 只用 Known train 训练、Known calibration
  选 checkpoint，test OOS 不参与训练、选模、半径或阈值选择。
- Trainable 相对 Frozen 的配对 OOS F1 增量（KIR=.25/.50/.75）为：CLINC150
  `+0.45/+1.12/+1.38pp`，Banking77 `+2.47/+4.72/+6.94pp`，StackOverflow
  `+5.06/+9.55/+10.50pp`。Known Recall 变化分别为 CLINC `-1.33/-1.37/-0.70pp`、
  Banking `-1.21/-2.05/-2.90pp`、StackOverflow `+0.77/+0.43/+0.13pp`。
- 这说明当前可训练 MiniLM 的稳定收益属于 K=1 表示/分数排序，不能解释为固定多中心恢复；Banking77
  高 KIR 的收益伴随 Known Recall 代价，后续仍需校准分析。该结果仍不是 SOTA 证据。
- 五 seed 分析报告：`docs/analysis/MINILM_TRAINABLE_5SEED_FAIR_COMPARISON_V1.md`；CSV 及图表分别位于
  `results/analysis/minilm_trainable_5seed_fair_v1/` 和 `figures/minilm_trainable_5seed_fair_v1/`。

## MiniLM 表示—边界诊断（analysis-only，2026-08-06）

- 读取五 seed Trainable/Frozen K=1 的 443,400 条测试 score 记录，仅做事后机制分析，不重新训练、不选阈值、不改历史结果。
- KIR=.50 时，OOS−Known median score gap 从 Frozen 到 Trainable 分别变为：CLINC150 `0.258→0.438`、
  Banking77 `0.259→0.334`、StackOverflow `0.116→0.434`；false acceptance 分别下降 `2.88pp`、
  `9.14pp`、`15.69pp`。
- 这支持“Trainable 的主要收益是单中心 score 排序和表示分离”，而不是“训练自动让固定多中心安全”。
- 诊断报告：`docs/analysis/MINILM_BOUNDARY_DIAGNOSTICS_V1.md`；CSV 位于
  `results/analysis/minilm_boundary_diagnostics_v1/`，图位于 `figures/minilm_boundary_diagnostics_v1/`。

## Trainable 与 MOGB 组件五 seed 配对分析（analysis-only，2026-08-06）

- 将 45 个 Trainable K=1 单元与 135 个 Frozen MiniLM MOGB/fixed-K 组件行按 dataset×KIR×seed 配对，计算 OOS F1、F1-All、Known Recall、false acceptance 的 paired bootstrap。
- KIR=.50 时，Trainable 相对 `MOGB partition + s2c boundary` 的 OOS F1 提升为 CLINC150 `+4.88pp`、
  Banking77 `+4.17pp`、StackOverflow `+8.42pp`；Known Recall 提升分别为 `+21.10pp`、`+30.03pp`、
  `+33.51pp`，但 false acceptance 更高。
- 结论不是“无条件超过 MOGB”：MOGB 组件通过更保守拒识降低 false acceptance，同时牺牲大量 Known 覆盖；
  Trainable 的优势是更平衡的覆盖—拒识工作点。报告：`docs/analysis/TRAINABLE_VS_MOGB_COMPONENT_V1.md`。

## 统一实验索引

当前所有主要实验、方法定义、MOGB/DCLOOS/ADB 分层状态和可视化入口，统一见
`docs/analysis/EXPERIMENTAL_EVIDENCE_INDEX_V1.md`。该索引只负责组织证据，不把不同监督条件和系统层级
合并成一个 SOTA 排名。

当前实验阶段短入口：`docs/analysis/EXPERIMENT_PROGRESS_SNAPSHOT_V2.md`。该快照明确当前自有主方法为
`S2C-Trainable-K1`，列出 E0--E3、MiniLM、MOGB、ADB、DA-ADB、DCLOOS 的证据层和现有图表；不把外部
监督或历史 Cascade 数字混成 SOTA 排名。

ADB 的完整跨数据集/KIR/五 seed 对比已补为 `docs/analysis/ADB_TRAINABLE_KIR_ANALYSIS_V1.md`，图表位于
`figures/trainable_vs_adb_kir_v1/`；这是同 split 的 BERT/TextOIR 外部合同分析，不是 MiniLM 同骨干排名。

## MiniLM 训练动态诊断（analysis-only，2026-08-06）

- 读取五 seed Trainable K=1 的 45 个 training manifest 和 179 条 Known calibration history；不重训、不改 checkpoint。
- 选择目标固定为 `calibration F1-K + 0.05×Known Recall`，最佳 epoch 主要集中在 3--4；但随着 KIR 增大，
  Known calibration 目标仍可稳定选模，而测试 OOS F1 在 Banking77、StackOverflow 明显下降。
- 这说明当前训练目标与开放空间风险并不完全对齐：不能通过盲目增加 epoch 解决历史 fulltex 差距，必须区分
  Known-only 选模和 OOS 边界校准。
- 报告：`docs/analysis/MINILM_TRAINING_DYNAMICS_V1.md`；CSV/图表位于
  `results/analysis/minilm_training_dynamics_v1/` 和 `figures/minilm_training_dynamics_v1/`。
- calibration→test Known Recall 的转移差异在 KIR=.50 仅为 CLINC150 `-0.64pp`、Banking77 `-0.21pp`、
  StackOverflow `+0.33pp`，所以高 KIR 的 OOS 下降更像 OOS score/边界校准问题，而不是已知覆盖完全崩溃。

## MiniLM score 标度与半径稳定性诊断（analysis-only，2026-08-06）

- 对 45 个 Trainable 与 45 个 Frozen K=1 run 做了统一的事后 threshold/radius 分析，生成 810 条阈值敏感性行、90 条半径稳定性行和 3 张图；不修改任何历史 run，也不使用 test oracle 选择正式阈值。
- KIR=.50 的诊断性 oracle threshold（仅用于说明 score 标度差异）为：CLINC150 Frozen/Trainable `1.00/1.05`，Banking77 `0.90/0.95`，StackOverflow `0.95/0.95`。这说明固定 `threshold=1` 对不同表示和数据集不是同一个工作点，但不能把 oracle 值写成正式调参结果。
- 因此当前 Trainable 与历史 `fulltex.tex` 的差距不能只归结为 MiniLM 没有训练；训练确实改善 K=1 score 分离，但当前 Gate 仍使用固定阈值和 Known-only 选模，历史 Cascade 还包含 Router/Expert、不同表示、K=2 和历史 OOS 校准合同。
- 半径 CV 约为 0.02--0.04，未显示“半径估计完全失稳”；StackOverflow 的关键风险仍是固定多中心 union 的新增 OOS 误接受，而非 K=1 半径噪声。证据见 `docs/analysis/THRESHOLD_RADIUS_STABILITY_V1.md`、`results/analysis/threshold_radius_stability_v1/` 和 `figures/threshold_radius_stability_v1/`。

## 同协议方法权衡与可视化（analysis-only，2026-08-06）

- 将 315 个已完成的五 seed 行（Trainable K=1、Frozen K=1/K=2、Random K=2、MOGB 三种 Frozen MiniLM 组件）按 dataset×KIR×seed 统一整理，生成 63 个均值/标准差单元、486 个 Trainable 相对组件的 paired bootstrap effect，以及 4 张图。
- 该 fair 组件包中的 Frozen K=1 使用 MOGB 矩阵的 Euclidean/mean-radius 工作点，不能与 E2 的 Mahalanobis Frozen K=1 混同；在这个组件口径下，KIR=.50 的 Trainable OOS F1 增量为 CLINC150 `+1.50pp`、Banking77 `+11.14pp`、StackOverflow `+11.12pp`。相对 MOGB partition+s2c boundary 的增量为 `+4.88/+4.17/+8.42pp`，但后者 false acceptance 更低、Known Recall 明显更低。
- 可视化显示：MOGB 风格组件位于“保守拒识”区域，Trainable 位于“覆盖—拒识平衡”区域；StackOverflow 固定 K=2 同时出现低 OOS F1 和高 false acceptance，说明问题不是单纯中心数量，而是接受区域组合语义。
- 证据：`docs/analysis/CROSS_PROTOCOL_TRADEOFF_V1.md`、`results/analysis/cross_protocol_tradeoff_v1/`、`figures/cross_protocol_tradeoff_v1/`。历史 fulltex、官方 BERT MOGB 和 DCLOOS 外部 OOS 监督均未混入该 fair 包。

## Gate→Cascade 桥接与误差分解（analysis-only，2026-08-06）

- 读取当前协议已经完成的 3-seed Cascade 变体（Frozen K=1、Frozen selected-K、CE-Recon selected-K、best controlled）和 3-seed Trainable K=1 Gate-only，生成 45 行、15 个 summary 和 3 张图。
- KIR=.50 时，Trainable Gate-only 的 OOS F1 为 CLINC150/Banking77/StackOverflow `90.43/84.77/86.71`；当前 CE-Recon selected-K Cascade 为 `90.00/89.48/87.62`。这表明后续 Cascade 确实能改变工作点，但作用依赖 Gate 误拒、误接和 Expert error，不能把 Gate-only 与完整系统直接排名。
- 桥接报告补出了 `Gate false accept`、`Known false reject` 和 `Expert error` 的分解；它是解释当前可训练 MiniLM低于历史 fulltex 的直接证据。入口：`docs/analysis/GATE_CASCADE_BRIDGE_V1.md`、`results/analysis/gate_cascade_bridge_v1/`、`figures/gate_cascade_bridge_v1/`。
- 证据：`docs/analysis/GATE_CASCADE_BRIDGE_V1.md`、`results/analysis/gate_cascade_bridge_v1/`、`figures/gate_cascade_bridge_v1/`。

## 原生 Frozen MiniLM baseline 与工作点诊断（2026-08-06）

- `native_baselines_v1` 已完成 180/180：3 数据集 × KIR={.25,.50,.75} × 5 seed × MSP/Energy/kNN/LOF；使用相同 registry、Known-only calibration 和冻结 `all-MiniLM-L6-v2`。
- KIR=.50 时 Trainable K=1 的 OOS F1 为 CLINC150/Banking77/StackOverflow `90.44/83.56/87.67%`；原生 MSP 为 `85.58/65.44/53.48%`。Trainable Known Recall 为 `74.44/82.21/83.89%`，MSP 为 `95.46/93.71/95.28%`，所以默认阈值不是同一工作点。
- `operating_point_diagnostic_v1` 生成 900 个回顾性 target-recall 行；名义 Known Recall=.85 附近，Trainable OOS F1 仍为 `91.64/82.40/87.05%`。该诊断使用 test 标签对齐工作点，仅用于解释，不改变正式阈值。
- 证据：`docs/analysis/NATIVE_BASELINES_V1.md`、`docs/analysis/OPERATING_POINT_DIAGNOSTIC_V1.md`、`results/analysis/native_baselines_v1/`、`results/analysis/operating_point_diagnostic_v1/`。

## 对 `fulltex.tex` 历史数字的当前解释

当前可训练 MiniLM K=1 低于 `fulltex.tex` 的历史 OOS 数字，主要不是“训练失败”：当前结果是 `protocol_v2_textoir_v1` 的 Known-only、Gate-only、K=1；历史表是旧数据/split、旧阈值与 Gate→Router→Expert Cascade 的系统级结果，部分历史 artifact 还存在 `main_table_ours` 指标覆盖。Trainable 已相对同协议 Frozen K=1 提高，且在匹配 Known Recall 的回顾性诊断中仍保留 OOS 排序优势；尚未完成的是同一当前协议下的端到端闭环。

## MOGB 组件归因分析（2026-08-06）

新增 `trainable_vs_mogb_ablation_v1`：将 45 个 Trainable K=1 五 seed 行与 180 个已完成的 MOGB frozen-MiniLM 距离/半径组件行配对，输出 324 个 bootstrap effects、180 个机制行和 3 张图。KIR=.50 时，Trainable 相对 MOGB partition+s2c boundary 的 OOS F1 增量为 CLINC150/Banking77/StackOverflow `+4.88/+4.17/+8.42pp`，同时 Known Recall 增量约 `+21.10/+30.03/+33.51pp`。该结果支持“Trainable 的主要优势是表示适配后的覆盖—拒识平衡”，不能称为官方 MOGB 复现或无条件 SOTA。

证据：`docs/analysis/TRAINABLE_VS_MOGB_ABLATION_V1.md`、`results/analysis/trainable_vs_mogb_ablation_v1/`、`figures/trainable_vs_mogb_ablation_v1/`。

## Trainable 表示上的原生检测器归因（2026-08-06）

- 新增 `native_baselines_trainable_v1`：使用已经完成的 Trainable MiniLM checkpoint，在 KIR=.50、seed=13/42/87 上运行 MSP、Energy、kNN、LOF，共 36/36 个检测器单元；不重新训练编码器。
- Trainable native 与 Trainable Gate K=1 用于区分检测器贡献；Trainable native 与 Frozen native 用于区分表示贡献。阈值只由 Known calibration 的 conformal alpha=.05 选择。
- Trainable 表示相对 Frozen native 在 CLINC150 和 Banking77 的四类 detector 上总体改善；StackOverflow 上 kNN/LOF 改善更明显，而 MSP/Energy 下降，说明表示收益依赖检测器几何。
- 同一 Trainable 表示上，Gate K=1 仍明显高于 native detector，但 Known Recall 更低，结论应按工作点而不是单指标排名解释。
- 证据：`docs/analysis/NATIVE_BASELINES_TRAINABLE_V1.md`、`results/analysis/native_baselines_trainable_v1/`、`figures/native_baselines_trainable_v1/`。

## 实验机制分析包 V3（2026-08-06）

- 基于现有 `all_methods_per_seed.csv` 的 315 行五 seed 轻量结果，新增 324 个 Trainable 相对冻结组件的
  paired bootstrap effect、63 个 dataset×KIR×method 汇总和 63 个多目标 Pareto 标记；未读取 checkpoint、embedding
  或原始文本，未重新训练。
- 新增中文报告：`docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`；数据和图表分别位于
  `results/analysis/experimental_mechanism_pack_v3/` 与 `figures/experimental_mechanism_pack_v3/`。
- 该分析进一步确认：Trainable K=1 的主要收益是 K=1 分数分离和覆盖—拒识折中；StackOverflow 的 fixed K=2 和
  MOGB 组件没有形成安全的多中心正收益。外部 ADB/DA-ADB/DCLOOS 仍保持监督条件不同的兼容性/历史层。
- 2026-08-08 复核发现 `../artifacts/s2c/runs/`、`../artifacts/s2c/cache/` 和 Trainable/E2 相关目录当前存在；
  但 V3 本身是基于轻量 CSV 的 analysis-only 结果，没有读取这些原始产物。后续如需重跑或扩展，仍须先逐阶段核对
  provenance、checkpoint hash 和 registry manifest，不能把目录存在当作完整性证明。

## 原始产物与可视化面板复核（2026-08-08）

- E2 目录当前可见 1,650 个 run；RACAL、Trainable control、缓存和外部基线目录也可见。
- `racal_v1` verifier 通过：Frozen/Trainable K=1 共 6/6（seed 13/42/87）；实验 registry audit 无错误输出。
- 重新运行 `tools/analysis/build_active_experiment_dashboard_v1.py`，刷新 12 张图和 `DASHBOARD_MANIFEST.json`；manifest 记录论文历史、Cascade、RACAL、MOGB 和表示实验输入的 SHA256。
- 该面板仍为 analysis-only，不新增训练、不覆盖历史 run；图表解读见 `docs/analysis/ACTIVE_EXPERIMENT_REPORT.md` 和 `docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`。
- 结论不变：Trainable K=1 是当前最稳的自有 Gate 工作点；固定 K>1 在 StackOverflow 仍有 union false-accept 风险；外部基线仍需按监督条件分层，不能合并成无条件 SOTA 排名。

## 原始逐样本误差证据链与可视化（2026-08-08）

- 新增 analysis-only 入口 `tools/analysis/build_raw_gate_error_visualization_v1.py`，读取 StackOverflow/KIR=.50、seed `{13,42,87}` 的 RACAL Trainable K=1、Frozen K=1、Fixed K=2 和 MOGB frozen-MiniLM fair component raw predictions。
- 逐 seed 校验四种方法的 `sample_id`、gold OOS 标签和样本数完全对齐，共读取 72,000 行（4 方法×3 seed×6,000），没有复制原始文本，也没有修改任何历史 run。
- 生成逐样本重算指标、pairwise 错误转移、Known 误拒绝/OOS 误接收按 intent 归因、分数分位数、ROC/PR 点和外部官方 MOGB 隔离摘要；报告和轻量结果位于 `docs/analysis/RAW_GATE_ERROR_VISUALIZATION_V1.md`、`results/analysis/raw_gate_error_visualization_v1/`、`figures/raw_gate_error_visualization_v1/`。
- 同协议三 seed 均值：Trainable K=1 OOS F1 `0.8671`、F1-All `0.8565`、Known Recall `0.8392`、FA `0.1114`；Fixed K=2 OOS F1 `0.6765`、Known Recall `0.9362`、FA `0.4526`；MOGB fair component OOS F1 `0.7319`、Known Recall `0.2834`、FA `0.0092`。因此 MOGB fair 更保守，不能只按 OOS F1 排名。
- Trainable K=1→Fixed K=2 的逐样本扩张平均恢复约 `297.3` 个 Known，但新增误接收约 `1,025.3` 个 OOS；这直接支持 StackOverflow 的 boundary-union 过覆盖机制。
- 外部官方 BERT MOGB 5-seed 兼容复现单独记录为 Known F1 `37.59±1.73`、Open F1 `72.05±0.39`、F1-score `40.72±1.61`、Accuracy `61.49±0.69`，与当前 MiniLM Gate 合同不同，禁止合并排名。
- 当前唯一下一步仍是：在不扩展训练矩阵的前提下，继续完成同协议误差/表示可视化解释；不得把这些 analysis-only 图表写成 SOTA 或官方 MOGB 公平胜负结论。

## 跨数据集/KIR 可视化证据链（2026-08-08）

- 新增 `tools/analysis/build_visual_evidence_chain_v1.py`，读取 63 个五 seed fair-component 汇总行、90 个固定 K=1…5 的 3-seed light sweep 行和 16 个 StackOverflow intent 诊断汇总行。
- 输出 7 张图：方法×数据集×KIR 性能热图、OOS F1—F1-All Pareto、KIR 曲线、方法排名、Trainable 相对差值、K 扩张折中和 intent 风险散点；轻量 CSV/manifest 位于 `results/analysis/visual_evidence_chain_v1/`，中文解释位于 `docs/analysis/VISUAL_EVIDENCE_CHAIN_V1.md`。
- 同协议 OOS F1 排名中，Trainable K=1 在 9 个 dataset×KIR 单元中 8 次排名第一、1 次排名第二；这只是当前 fair matrix 的稳定性证据，不是 SOTA 声明。
- 跨 KIR 均值的 Trainable OOS F1/F1-All/Known Recall/FA：CLINC150 `89.48/81.47/74.40/3.65%`，Banking77 `81.29/81.44/82.16/15.23%`，StackOverflow `86.48/87.16/84.10/7.33%`。
- StackOverflow intent 诊断图显示平均恢复约 `30.0` 个 Known、却新增约 `131.4` 个 OOS 误接收，平均 bootstrap ARI `0.926`；这进一步说明稳定聚类不等于 OOS 边界收益。
- 外部 ADB、DA-ADB、官方 BERT MOGB 和 DCLOOS（若有记录）仅写入隔离表，不与同协议 fair matrix 合并；本阶段没有新增训练或测试选择。

## 表示几何与 Near-OOS 可视化证据（analysis-only，2026-08-08）

- 新增 `tools/analysis/build_representation_geometry_visuals_v1.py`，读取已有 Frozen/CE/SupCon 几何汇总、K=1/K=2 结果、Trainable/Frozen score 诊断以及 StackOverflow raw transition；没有训练、调参、重建中心或修改历史 artifact。
- 生成 4 张图：`geometry_tradeoff.png`、`near_oos_delta.png`、`score_gap_false_accept.png`、`stackoverflow_error_quadrants.png`，以及 4 份轻量 CSV 和 `MANIFEST.json`；入口为 `docs/analysis/REPRESENTATION_GEOMETRY_VISUALS_V1.md`。
- 结果显示：CE/SupCon 提高类内对齐和 relative separation，但 StackOverflow 的 K=2 near-OOS F1 分别下降 `10.7pp/32.3pp`；因此表示几何改善不能单独保证多中心安全。
- KIR=.50 的 score-gap 诊断中，Trainable 的 OOS−Known median score gap 高于 Frozen，且三个数据集 false acceptance 均下降；这支持“Trainable K=1 的主要收益来自 score separation”。
- StackOverflow Trainable 与 MOGB fair 的错误四象限显示：Trainable 独有的 Known 正确样本占比约 `54.6%`，MOGB 独有的 OOS 正确样本约 `10.2%`；这反映两者是不同工作点，不是单一方法全面支配。
- 解释边界：所有数字仍为事后机制分析；不构成 SOTA、官方 MOGB 公平排名或完整 Cascade 结论。
- 验证：脚本运行、`py_compile`、ruff、图像人工检查已通过；后续仍需完成研究状态/日志/数据跟踪和 registry 最终检查。

## 状态修正（2026-08-08）

- 上一阶段“补表示几何和 near-OOS 图”的下一步已完成；当前不再新增 K、表示损失或多中心训练矩阵。
- 下一步改为：将已有性能、逐样本错误、表示几何和 near-OOS 证据整理成统一的同协议对比包，并继续把 ADB/DA-ADB/MOGB/DCLOOS 按监督合同隔离；任何新训练实验必须先登记新的研究问题和重复实验检查。

## 当前实验与可视化证据总览（2026-08-08）

- 新增中文总入口：`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V1.md`，统一链接性能热图、StackOverflow raw 错误链、表示几何/near-OOS 图和外部基线合同说明。
- 该总览不新增实验、不改变历史结果，也不把 MOGB/ADB/DA-ADB/DCLOOS 与同协议 Trainable/Frozen fair matrix 混成一个排名。
- 当前推荐阅读顺序：先看总览，再看 `VISUAL_EVIDENCE_CHAIN_V1.md`、`RAW_GATE_ERROR_VISUALIZATION_V1.md` 和 `REPRESENTATION_GEOMETRY_VISUALS_V1.md`。
- 当前唯一下一步：以总览识别仍缺少的同协议 baseline 字段；没有明确缺口前不再扩展 K、训练损失或 adaptive-K 矩阵。

## 外部基线合同与可比性可视化（analysis-only，2026-08-08）

- 新增 `tools/analysis/build_baseline_contract_visuals_v1.py` 和中文报告 `docs/analysis/BASELINE_CONTRACT_VISUALS_V1.md`。
- 将 StackOverflow/KIR=.50 的 Trainable、Frozen/MOGB 组件、BRAK、ADB、DA-ADB 和 MOGB strict 单格放入带合同标签的 Pareto 图；DCLOOS reduced-budget 保留在合同表中，不放入同一散点。
- 当前已有数字显示 ADB/DA-ADB 兼容单格的 OOS F1 高于 Trainable，但它们是端到端 BERT、单 seed/单格，不能据此宣布公平优胜；同协议 fair matrix 仍以五 seed Trainable K=1 为自有稳定工作点。
- 输出：`results/analysis/baseline_contract_visuals_v1/contract_matrix.csv`、`figures/baseline_contract_visuals_v1/`、`MANIFEST.json`。
- 该阶段仍为 analysis-only，没有训练、调参或历史 artifact 修改。

## ADB/DA-ADB 同协议数据适配与执行预检（2026-08-08）

- 已将 StackOverflow/KIR=0.50、seed={42,87,100} 的 protocol_v2 ADB TSV 从
  `data/exports/protocol_v2_textoir_v1/adb/` 复制到独立 external artifact 数据根；train/dev/test、Known labels
  复制前后 SHA256 一致，Known 列表与 registry 一致。
- `tools/compat/textoir/run_external_textoir.py` 现在支持显式 `--data-root` 和 `--known-labels-file`，外部兼容 runner
  不再必须读取 `textoir/data`；新增 `tools/compat/textoir/build_protocol_data_root.py` 负责复制和哈希记录。
- ADB 的 protocol_v2 seed=42 首次尝试因旧版 Transformers 只接受 `pytorch_model.bin` 失败；隔离转换后再次预检时
  `textoir-py39` 的 CUDA/torch probe 在 90 秒内超时，尚未生成可用同协议指标。DA-ADB 与 ADB 共用该环境阻断，未重复启动。
- 旧 ADB/DA-ADB 单格仍保持历史 `legacy_compatibility_only`；最新隔离 CUDA ADB 三 seed 为 OOS F1
  `87.36±1.61`、F1-All `85.66±1.59`，仍不能与当前五 seed Trainable/Frozen fair rows 合并，也不能据此
  宣称超过或达到 SOTA。
- 详细状态：`docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`；轻量 CSV/manifest：
  `results/analysis/baseline_execution_status_v1/`。当前唯一外部基线下一步是恢复可验证的独立旧版运行环境后先完成
  StackOverflow/KIR=0.50/seed=42 单格，再扩展其余 seed；不启动其他新训练矩阵。

## 配对效应与 intent-level 异质性分析（2026-08-08）

- 新增 analysis-only 脚本 `tools/analysis/build_paired_effect_intent_heterogeneity_v1.py`，读取已完成的
  `cross_protocol_tradeoff_v1/per_seed.csv` 和 `intent_kir_stability_pack_v1/intent_kir_rows.csv`，没有重新训练或覆盖 artifact。
- 生成 `results/analysis/paired_effect_intent_heterogeneity_v1/`（270 个配对行、216 个指标汇总、18 个 dataset×KIR×distance intent 汇总）和四张可视化图。
- 配对统计固定 10,000 次 bootstrap，比较同一 dataset×KIR×seed 下 Trainable K=1 与 Frozen/MOGB 组件；intent-level 的
  `safe_gain_oracle` 和 `oracle-best-K` 明确标记为测试 oracle 后验诊断，不能作为正式选 K 规则。
- 新增中文入口 `docs/analysis/PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md`，并链接到可视化证据总览和实验对比总报告。
- 初步机制结论：Trainable K=1 相对 Frozen K=1 的 OOS F1 配对收益在三个数据集/KIR 下方向一致；Banking77 的逐 intent 多中心潜在安全收益比例高于 StackOverflow，StackOverflow 在 KIR=.75 时进一步下降。
- 风险：该阶段仍不提供同监督 ADB/DA-ADB/DCLOOS 的正式排名；外部基线 runtime blocker 未改变。

## 外部 baseline runtime follow-up（2026-08-08）

- 为排除单一解释器问题，检查了当前项目默认环境、`bo`、`implicit_intent` 和 `textoir-py39` 等可见 runtime；多个环境的最小 `import torch`/CUDA 检查均在 12–15 秒内超时，历史尝试还出现 signal 6 与 D-state 进程。
- `nvidia-smi` 可见 RTX 5070，但当前没有稳定的 torch 计算进程；因此 ADB/DA-ADB 同协议运行仍是 runtime 阻断，不是算法或 StackOverflow 数据结论。
- 详细边界追加在 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`；下一次外部基线运行必须先通过独立环境的 torch import、BERT forward 和 CPU/GPU smoke，再启动 seed=42。

## 跨数据集错误归因（2026-08-08）

- 新增 analysis-only 脚本 `tools/analysis/build_cross_dataset_error_attribution_v1.py`，读取已有 315 个 fair prediction run，按同一 `dataset × KIR × seed × sample_id` 对齐 1,890,000 条预测记录；没有重跑训练、改动阈值或覆盖任何历史 artifact。
- 生成 `results/analysis/cross_dataset_error_attribution_v1/`（方法指标、配对转移、逐 intent 归因、manifest）和 `figures/cross_dataset_error_attribution_v1/`（Trainable OOS 正确率优势、false-accept 差值、false-reject 差值、OOS F1–false acceptance 工作点图）。中文入口为 `docs/analysis/CROSS_DATASET_ERROR_ATTRIBUTION_V1.md`。
- 结果：Trainable K=1 相对 Frozen K=1、Frozen K=2 和 Random K=2，在全部 9 个 dataset×KIR 单元降低 false acceptance；StackOverflow 的差距随 KIR 增大最明显。MOGB-MiniLM 及 MOGB partition 组件的 false acceptance 较低，但 false rejection 分别显著增加，属于保守拒识而非同监督下的整体优胜证据。
- 边界：本阶段只归因已有 frozen/trainable 预测，不能用于调参或声明 SOTA；ADB、DA-ADB、DCLOOS 同协议结果仍受 runtime/监督合同阻断，不能与这批 Known-only fair rows 混排。

## 跨数据集逐意图风险可视化（2026-08-08）

- 新增 `tools/analysis/build_cross_dataset_intent_risk_visuals_v1.py`，消费已审计的逐 intent 汇总，不读取原始文本、不重跑模型、不调整阈值。
- 生成 `results/analysis/cross_dataset_intent_risk_visuals_v1/` 与 `figures/cross_dataset_intent_risk_visuals_v1/`，包括 OOS 吸收 intent、错误集中度、逐 intent 相对 Trainable 的差值和四张图；中文报告为 `docs/analysis/CROSS_DATASET_INTENT_RISK_VISUALS_V1.md`。
- KIR=.50 时，StackOverflow Frozen K=2 的前五个 OOS 吸收 intent（cocoa、sharepoint、osx、spring、scala）贡献约 64.9% 的 false acceptance；这说明退化集中在少数边界，而非均匀发生。
- MOGB-MiniLM 与 MOGB partition + ours 在逐 intent OOS false acceptance 上更保守，但平均 Known false rejection 分别比 Trainable 高约 45.98 和 25.10 个百分点；因此仍是风险取舍证据，不是统一 SOTA 排名。

## 同协议五 seed 统计稳定性（2026-08-08）

- 新增 `tools/analysis/build_statistical_stability_v1.py`，使用已有 315 行 fair per-seed 指标做同一 `dataset × KIR × seed` 配对比较；固定 bootstrap seed=20260725、10,000 次重采样，不重跑模型、不选择阈值。
- 生成 `results/analysis/statistical_stability_v1/` 和 `figures/statistical_stability_v1/`，包含配对 CI、方法排名热图和五 seed KIR 曲线；中文报告为 `docs/analysis/STATISTICAL_STABILITY_V1.md`。
- Trainable K=1 相对 Frozen K=1、Frozen K=2、Random K=2 在 9/9 个 dataset×KIR 组合中五 seed 全部胜出；相对 MOGB 组件在 8/9 个组合中配对 CI 为正。
- 按 OOS F1 平均排名，Trainable 在 8/9 个 dataset×KIR 单元排名第一；唯一例外是 Banking77/KIR=.25，MOGB partition + ours 略高。

## Gate→Cascade 配对桥接（2026-08-08）

- 新增 analysis-only 脚本 `tools/analysis/build_gate_cascade_paired_bridge_v2.py`，读取已有
  `gate_cascade_bridge_v1/per_seed.csv` 的 45 行；没有重训、调参或覆盖历史结果。
- 生成 `results/analysis/gate_cascade_paired_bridge_v2/`、
  `figures/gate_cascade_paired_bridge_v2/` 和中文报告
  `docs/analysis/GATE_CASCADE_PAIRED_BRIDGE_V2.md`。
- 该阶段明确标注评价层差异：Trainable K=1 是 Gate-only，其他行是 Cascade，不能据此做统一端到端排名。
- 相对 Frozen K=1 Cascade，Trainable Gate OOS F1 桥接差值为 CLINC150 +2.41 pp、Banking77 −0.05 pp、
  StackOverflow +7.69 pp；false acceptance 分别降低约 3.44、10.20、12.38 pp。
- 同层 Cascade 中，CE-Recon selected-K 相对 Frozen K=1 的 OOS F1 差值为 CLINC150 +1.97 pp、
  Banking77 +4.66 pp、StackOverflow +8.60 pp，但 Banking77 Known macro-F1 下降约 3.25 pp。
- 该阶段仍不能支持超过 MOGB/ADB/DA-ADB/DCLOOS 或 SOTA 的结论；外部 baseline 统一运行仍受独立 torch/BERT runtime 阻塞。

## 同一 Trainable 表示下的检测器机制对照（2026-08-08）

- 新增 analysis-only 脚本 `tools/analysis/build_trainable_detector_mechanism_v1.py`，固定 KIR=.50、3 个数据集、3 个 seed，读取 9 个 Trainable Gate 行和 36 个原生检测器行。
- 生成 `results/analysis/trainable_detector_mechanism_v1/`、`figures/trainable_detector_mechanism_v1/` 和中文报告 `docs/analysis/TRAINABLE_DETECTOR_MECHANISM_V1.md`。
- 在同一 Trainable MiniLM 表示下，Gate 相对 MSP/Energy/kNN/LOF 的 OOS F1 在 12/12 个比较中三 seed 全部更高；StackOverflow 提升约 17.88–43.76 pp。
- 主要代价是 Known Recall 下降约 11–21 pp，同时 false acceptance 显著降低；因此这是“检测器工作点/边界机制”证据，不是所有指标全面优胜或 SOTA 证据。
- 新图包括同表示 Pareto 图、配对效应森林图和 false acceptance/false rejection 构成图。

## 全指标 Fair Effect Landscape（2026-08-08）

- 新增 analysis-only 脚本 `tools/analysis/build_fair_effect_landscape_v1.py`、中文报告
  `docs/analysis/FAIR_EFFECT_LANDSCAPE_V1.md`、结果目录
  `results/analysis/fair_effect_landscape_v1/` 和三张多指标图
  `figures/fair_effect_landscape_v1/`。
- 该阶段从 `statistical_stability_v1/paired_effects.csv` 的 486 行源表中按预注册范围筛选
  432 行：3 数据集 × 3 KIR × 6 fair comparison × 8 指标；没有重跑模型、调参或改动历史 artifact。
- Trainable K=1 相对 Frozen K=1/K=2/Random K=2 的 OOS F1 平均优势约为 +7.1/+10.2/+7.4 pp，
  且 9 个 dataset×KIR 单元均由 95% paired bootstrap CI 支持；相对 MOGB 组件的优势约为
  +12.4/+8.0/+9.9 pp，8/9 单元为稳定正差异，Banking77/KIR=.25 是局部例外区域。
- 多指标图同时显示 Known Recall 与 false acceptance 的代价：MOGB 组件更保守、Known 误拒明显更高，
  Trainable K=1 更接近平衡工作点。该阶段不构成 SOTA、官方 MOGB 复现或外部 baseline 公平排名。
- 当前唯一外部实验下一步仍是恢复可验证 runtime 后先做 ADB/DA-ADB/DCLOOS 的同协议单格；不再盲目扩展 K、
  新损失或 adaptive-K 矩阵。

## KIR 敏感性分解（2026-08-08）

- 新增 analysis-only 脚本 `tools/analysis/build_kir_sensitivity_decomposition_v1.py`、中文报告
  `docs/analysis/KIR_SENSITIVITY_DECOMPOSITION_V1.md`、结果目录
  `results/analysis/kir_sensitivity_decomposition_v1/` 和三张图
  `figures/kir_sensitivity_decomposition_v1/`。
- 从已有五 seed summary 计算 3 数据集 × 7 方法 × 6 指标的 KIR=.25→.75 端点变化和线性斜率，
  没有重跑模型或使用测试结果选参。
- 结果显示 Trainable K=1 的 OOS F1 下降幅度为 CLINC150/Banking77/StackOverflow 的
  −12.3/−23.0/−19.2 pp；Frozen K=2 为 −13.0/−26.5/−43.0 pp。StackOverflow 固定 K=2
  同时出现约 +42.2 pp 的 false acceptance 增量，支持多球接受区域随开放比例扩大而过覆盖。
- MOGB 组件的 false acceptance 基本不随 KIR 增大，但 Known Recall 大幅下降，说明其当前
  工作点更保守；Trainable K=1 的 Known Recall 基本稳定。该阶段是机制分析，不是 SOTA 排名。
## OOS 误接收—误拒绝预算（analysis-only，2026-08-08）

- 新增 `tools/analysis/build_oos_error_budget_v1.py`、中文报告
  `docs/analysis/OOS_ERROR_BUDGET_V1.md`、结果目录
  `results/analysis/oos_error_budget_v1/` 和三张图
  `figures/oos_error_budget_v1/`。读取已审计的 315 行 fair per-seed 指标，不重跑模型、不调参、
  不修改历史 artifact。
- 由已审计的 `n_known/n_oos`、false acceptance 和 false rejection 重构 OOS precision/recall，
  并逐行核对源 OOS F1（最大误差 `<1e-9`）；输出 315 行重构明细、63 行五 seed 汇总和 324 行
  Trainable 配对效应，固定 bootstrap seed=20260808、10,000 次重采样。
- KIR=.50 时，Trainable K=1 的 OOS F1/FA/FR 为 CLINC150 `90.44/3.69/25.56%`、Banking77
  `83.56/15.74/17.79%`、StackOverflow `87.67/9.34/16.11%`。StackOverflow Frozen K=2 的
  FA 为 `47.17%`，而 MOGB MiniLM 组件的 FA 仅 `0.79%` 但 FR 达 `72.91%`；这把“多中心过覆盖”
  与“MOGB 组件过度保守”明确分开。
- 当前最稳妥的结论是：Trainable K=1 在已有同监督 fair matrix 内提供更平衡的 coverage—rejection
  工作点，但这不是完整 MOGB、ADB、DA-ADB 或 DCLOOS 的 SOTA 排名；外部 baseline 仍需独立 runtime
  和监督合同通过后再进入主表。
## 综合实验分析主入口（2026-08-09）

为避免多个历史报告使用不同快照或评价层级，当前五 seed 同协议结果的权威分析入口统一为：
`docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`。

该分析消费 `cross_protocol_tradeoff_v1/summary_mean_std.csv` 的 63 行汇总，重新计算了正确方向的
dataset×KIR 方法排名，并生成 OOS F1/F1-All 热力图、KIR 曲线、KIR=.50 Pareto 图和排名计数图。
核心判断保持为：Trainable K=1 在当前同监督 Gate/组件矩阵中最稳定且更平衡；StackOverflow 固定 K=2
的主要失败是 false acceptance 过覆盖；MOGB MiniLM 组件更保守但 Known false rejection 很高。
这些结果仍不是完整 MOGB、ADB、DA-ADB、DCLOOS 的 SOTA 排名；外部基线需在合同一致且 runtime 可复现后再并入。

外部 runtime 复核的早期 `runtime_blocked_no_metrics` 条目属于 shim 修复前历史记录；随后已通过隔离
EasyDict/MKL/Transformers 兼容层完成 ADB 三个 StackOverflow/KIR=.50 单元。ADB 逐样本重算结果已进入
单独外部合同表；DA-ADB 兼容运行仍因 NaN/全类预测被标记无效。详见 `docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`。

可视化入口已整理为 `docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`，按性能、表示、决策和外部合同四层指向现有图表；当前主图均来自已审计结果，不是新模型选择。

维护回归方面，分析脚本、compileall、research-state、development-log、data-tracking 和 diff 检查通过；
`pytest tests/unit -q` 当前已通过（349 passed，3 个旧依赖 deprecation warnings）；DA-ADB overlay 定向测试此前为 19 passed。

## 方法名称和对比边界（2026-08-09）

当前“我的方法”默认专指 `S2C-Trainable-K1`；固定 K=2、随机 K=2 和 RC-AMBL/Joint-Adaptive
pilot 是独立的消融/负诊断，不得混称为已经成功的自适应多中心。MOGB-Fair 与 MOGB 官方 BERT
复现也严格分开。完整方法地图见 `docs/analysis/METHOD_COMPARISON_MAP_V1.md`，新增机制图和机器可读
结果见 `docs/analysis/MECHANISM_EVIDENCE_V2.md` 与 `results/analysis/mechanism_evidence_v2/`。

## 历史 SOTA 与当前结果分层（2026-08-09）

已新增 `docs/analysis/HISTORICAL_SOTA_AND_CURRENT_COMPARISON_V1.md`。该报告直接解析
`fulltex.tex` 的历史 `Ours` 主表，并明确：论文中的 SOTA 候选是完整 Gate–Router–Expert
Cascade，不是当前 `S2C-Trainable-K1` Gate-only。历史表与当前五 seed fair matrix 不得直接合并排名。
同一报告新增历史 OOS F1 热力图、Ours 相对最强列出基线差值图和 StackOverflow 合同分层图；MOGB
论文差距被拆成数据、训练、兼容层、指标和超参数/随机性五类待审计合同差异。

## MOGB 复现差距定量审计（2026-08-09）

新增 `docs/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`，只读取两个已冻结的官方 BERT
exact 单格、十个五 epoch 兼容运行和 pinned MOGB 源码，没有重训或修改历史 artifact。审计确认：

- StackOverflow/Banking exact 单格分别训练 38/54 epoch，Known dev accuracy 为 91.60%/91.84%，
  CE loss 已从 2.307/4.064 降到约 0.120/0.126，故“只是训练未收敛”不是主要解释；
- 官方 `myloss.py` 对类别距离先做 L1 归一化再 `softmax(-distance)`，使 10/58 类下的
  true-class 理论最大概率仅约 0.1105/0.01754，子中心目标的动态范围被明显压缩；
- 最终生成 28/99 个粒球，说明自适应划分实际运行；但 OOS Recall 约 98%，Known Recall 只有
  51.53%/43.71%，本地差距主要表现为平均半径边界过度拒绝 Known；
- 结论仍是“现有材料下未复现论文”，不是“MOGB 算法无效”。原始数据、旧环境和逐样本合同缺失，
  且现代兼容层修复了 stale-graph 训练路径。

机器可读表与五张图位于 `results/analysis/mogb_reproduction_gap_analysis_v2/` 和
`figures/mogb_reproduction_gap_analysis_v2/`。当前方法边界保持不变：历史 SOTA 是完整 Cascade
`Ours`；当前自有 Gate 候选是 `S2C-Trainable-K1`；两者以及 MOGB paper/exact/fair 不能混排。

## MOGB corrected-loss 单格与统一对比入口（2026-08-09）

- 新增 `mogb_corrected_subcentroid_loss_v1`：StackOverflow/KIR=.50/seed=0，复用 pinned 官方
  BERT/数据/粒球/mean-radius/nearest-ball 合同，只把公开代码的
  `L1-normalize(class distances)` 改为预注册的 `raw_distance / temperature=1.0`。
- 61 epoch 完成，best epoch=51，Known dev accuracy=92.10%；数值兼容测试的 loss 和梯度差均为0。
  第三方 MOGB 源码未修改，结果属于 adapted diagnostic，不是 official reproduction。
- 相对本地官方逻辑单格，Accuracy `+2.08pp`、F1-All `+4.29pp`、F1-K `+4.62pp`、Known Recall
  `+7.67pp`；同时 OOS Recall `-3.50pp`。这说明损失压缩是实际原因之一，但仍不能关闭论文差距。
- 修正后 Acc/F1-All/F1-U/F1-K 为 `77.25/72.64/80.96/71.81`，相对论文公开参考仍低
  `11.42/14.85/8.75/15.46pp`。剩余差距继续指向平均半径 Known 覆盖、作者原始数据/Known 列表和
  旧运行时合同，而不是“训练根本没有运行”。
- 统一中文入口为 `docs/analysis/S2C_BASELINE_MOGB_COMPARISON_OVERVIEW_V1.md`；新增四张图位于
  `figures/s2c_baseline_mogb_overview_v1/`，同时链接历史 Ours 对七个基线的既有热力图和逐格优势图。

## S2C 与 MOGB-Fair 机制对比仪表盘（2026-08-09）

- 新增权威同协议入口 `docs/analysis/S2C_VS_MOGB_MECHANISM_DASHBOARD_V1.md`，严格配对三个数据集、
  三个 KIR 和五个 seed，共 45 个 dataset×KIR×seed 单元；没有重跑模型或修改历史结果。
- 当前公平比较对象固定为 `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair`。前者是 Known-only 训练最后
  两层 MiniLM 与 residual projection 后的单中心 Gate；后者是相同 Frozen MiniLM 表示上的 MOGB
  自适应粒球、欧氏距离和 mean-radius 边界。二者都不是 `fulltex.tex` 的完整 Cascade，也不是论文
  BERT MOGB 的同合同排名。
- 45 个配对中，S2C 的 OOS F1 为 44 胜 1 负，F1-All 为 45 胜 0 负。相对 MOGB-Fair，平均
  Known Recall `+49.01pp`、OOS Precision `+20.58pp`，代价是 OOS Recall `-7.83pp`、false
  acceptance `+7.83pp`。因此优势来自更平衡的 Known 覆盖—OOS precision/recall 工作点，不是
  在所有风险指标上同时占优。
- 组件桥 `MOGB partition + S2C boundary` 的 45 单元平均 F1-All 为 `63.92%`，高于 MOGB-Fair
  的 `46.26%`，但仍低于 Trainable K1 的 `83.36%`。当前差距同时包含边界工作点和表示适配贡献。
- 六张图位于 `figures/s2c_vs_mogb_mechanism_dashboard_v1/`；轻量结果与 manifest 位于
  `results/analysis/s2c_vs_mogb_mechanism_dashboard_v1/`，manifest SHA256 为
  `1f3c73102a094a0b087fdbe95d02e68d06e0275c75bb77b4dc3a313d8f2dbf4e`。

## S2C 与 MOGB-Fair 阈值曲线和排序归因（2026-08-09）

- 新增 `s2c_mogb_operating_curve_attribution_v1`，逐样本重放 45 个相同 dataset×KIR×seed 配对，
  90/90 方法单元的 `score > 1` 均与冻结预测完全一致；没有训练、改阈值或覆盖历史 run。
- S2C 在阈值无关 AUROC 上 45/45 胜出，平均 `+7.93pp`；AUPR-OOS 平均 `+12.00pp`。
  默认 OOS F1 为44胜1负、平均 `+12.36pp`。这证明当前优势不只是 mean-radius 默认工作点不同。
- 即使使用 test-defined 事后最优阈值，S2C 的 OOS F1 上限仍平均高 `+6.43pp`。MOGB-Fair 的默认到
  oracle 校准缺口为 `7.12pp`，S2C 为 `1.19pp`，说明 MOGB 同时存在排序和校准两层损失。
- 在事后匹配约80% Known Recall时，S2C/MOGB-Fair 的 OOS F1 为 `86.05%/74.94%`，false
  acceptance 为 `7.32%/26.20%`。该结果只解释同覆盖检测前沿，禁止用于正式选阈值。
- 中文报告、六张图和轻量表分别位于 `docs/analysis/S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1.md`、
  `figures/s2c_mogb_operating_curve_attribution_v1/`、
  `results/analysis/s2c_mogb_operating_curve_attribution_v1/`；manifest SHA256 为
  `500bf78e3c2347d8d96a6df1f2804b72a851e112acdc9ed4171b971f60f19d58`。

## corrected-loss BERT-MOGB 半径覆盖归因（2026-08-09）

- 复用 checkpoint `127bd347...`，没有重训；只用 Known dev 固定 80%/85%/90%/95% 覆盖工作点，
  test OOS 仅评价。
- 源运行未保存最终 `cluster3` Python RNG 状态和中心向量，因此当前 33 球结构明确标记为
  `deterministic_fixed_checkpoint_reconstruction`，不是原 artifact 的严格 selected-ball 重放；默认指标
  与源记录最大绝对差1.87pp。
- 默认工作点 F1-U/F1-All/Known Recall 为 `79.76/71.42/58.63`；cal-80 为
  `73.45/73.97/77.00`；cal-95 为 `35.68/63.80/87.17`。Known 覆盖恢复伴随 OOS 误接受
  `197→2311`，否定“只需放大平均半径即可复现论文”的解释。
- 中文报告与图：`docs/analysis/MOGB_CORRECTED_RADIUS_COVERAGE_V1.md`、
  `figures/mogb_corrected_radius_coverage_v1/`；manifest SHA256=
  `1e5ee8361243d052e53a4db34efe121c1adf0ebdfd81f571e1546cafcaf9b927`。

## MOGB 官方代码合同逐行审计（2026-08-09）

新增 `docs/analysis/MOGB_CODE_CONTRACT_LINE_AUDIT_V1.md` 和
`results/analysis/mogb_code_contract_line_audit_v1/evidence.csv`。逐行核对 pinned upstream 后，已把
实际观测到的差距对应到：`myloss.py` 的 L1 距离归一化、`cluster.py` 的硬编码 `cuda:0`、`cluster3.py`
的随机递归拆分与 selected-ball 过滤、均值半径以及 `gb_test.py` 的最近球接受语义。该阶段没有训练、
没有重评分，也没有改变 MOGB 或 S2C 结果；它只加强 MOGB discrepancy audit，仍不能称严格复现。

## 统一对比实验图谱与可视化证据链（2026-08-09）

新增 `docs/analysis/COMPARISON_ATLAS_V1.md` 和 `comparison_atlas_v1` 轻量结果/图表。该阶段只读取
已冻结结果，覆盖当前 fair 汇总 63 行、per-seed 315 行、历史 `fulltex.tex` 72 行、外部合同参考 10 行
和 MOGB exact 本地参考 2 行。

图谱将历史完整 Cascade、当前 protocol_v2 Gate、MOGB 公平组件和外部兼容单格严格分层；直接排名只允许
当前 `protocol_v2_fair_gate`。新增图包括公平 OOS/F1-All 热图、Trainable-Frozen 增量、固定 K=2 风险
归因、MOGB/S2C 组件工作点、OOS F1--Known Recall Pareto 和五 seed 稳定性。

## 外部基线同协议单格审计（2026-08-09）

在不改变当前 fair Gate 矩阵的前提下，已通过隔离 EasyDict/MKL/Transformers 兼容层完成 ADB 的三个
StackOverflow/KIR=.50 单元（seed=42、87、100）。逐样本重算得到 OOS F1=86.30%/86.99%/89.13%、
F1-All=84.30%/85.32%/87.40%，并与 S2C Trainable K=1、Frozen K=1、固定 K=2、MOGB-MiniLM
放在单独的外部合同表中。
DA-ADB 原始与 clamp30 适配运行虽返回0，但训练出现 NaN 或预测全为类别0，已标记为 invalid，不采信
其旧 `results.csv` 的不一致数字。seed=100 ADB 单元仍在收口。

新增：`tools/analysis/build_external_single_cell_comparison_v1.py`、
`results/analysis/comparison_atlas_v1/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`、
`stackoverflow_kir50_external_and_fair_cells.csv`、`external_invalid_semantic_runs.csv` 和
`figures/comparison_atlas_v1/stackoverflow_external_single_cell_comparison.png`；ADB 三 seed 已收口。

## 原生 detector 配对置信区间补充（2026-08-10）

- 在不新增训练的前提下，复用 KIR=.50、3 seed 的 Trainable/Frozen MiniLM detector 控制，
  将已登记的 10,000 次 paired bootstrap 结果规范化为 `Gate - native` 和
  `Trainable - Frozen` 两种方向。
- 新增 `results/analysis/detector_mechanism_v1/detector_paired_ci.csv`、
  `representation_paired_ci.csv`（各 48 行）以及两张 OOS F1 forest 图。
- 同一 Trainable 表示下，Gate 相对 MSP/Energy/kNN/LOF 在三个数据集均为 3/3 seed 胜出，
  OOS F1 的 95% 区间全部为正；StackOverflow 相对 MSP/Energy 的平均提升约 `+41.10/+43.76pp`。
- 表示层并非所有 detector 都同向改善：StackOverflow 的 MSP/Energy 为 `-4.42/-2.19pp`，
  kNN/LOF 为 `+7.23/+30.03pp`，进一步支持“Trainable 表示 + 几何 Gate”是组合收益。
- 此阶段仍是 analysis-only，不替代五 seed 主矩阵，也不改变外部基线合同边界。

## DA-ADB 当前 protocol_v2 三 seed 收口（2026-08-10）

- 新增当前协议 StackOverflow/KIR=.50/seed=`42,87,100` 的 DA-ADB 有效运行；三个 manifest 均 complete，逐样本预测有限、形状一致且包含全部 11 个标签。
- DA-ADB OOS F1=`72.48±6.24%`、F1-All=`74.02±3.13%`、F1-Known=`74.18±2.86%`、Known Recall=`75.97±4.10%`、FA=`29.07±11.19%`。
- 同 seed S2C Trainable K=1 为 OOS F1=`88.21±1.72%`、F1-All=`86.69±1.45%`；描述性配对差值（S2C−DA-ADB）为 OOS F1 `+15.73pp`、F1-All `+12.67pp`、Known Recall `+7.71pp`、FA `−20.89pp`。
- 该结果仍是 BERT/TextOIR 与 Known-only MiniLM 的合同对照，不能作为同骨干 SOTA 排名；旧兼容 DA-ADB 单格 `90.90%` 已被三单元合同审计拆分，不能代表当前协议。
- 报告：`docs/analysis/DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md`；图与轻量结果：`figures/da_adb_current_protocol_summary_v1/`、`results/analysis/da_adb_current_protocol_summary_v1/`。

## DCLOOS 固定 Registry 单格收口（2026-08-10）

- 已为 DCLOOS 运行时 overlay 增加 `--known-labels-file`，使其读取 protocol_v2 的 StackOverflow/KIR=.50/seed=42 Known 列表，避免上游再次随机抽类；第三方 checkout 未修改。
- 当前 run：`../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_v1/`；保留 DCLOOS 的 BERT、pseudo-OOS 和外部 SQuAD 监督，故仍是外部监督参考，不进入 Known-only fair 主排名。
- BERT 训练约运行 3530 秒后结束，但没有生成最终 `metrics.json`、`stdout.log` 或 `stderr.log`；只留下中间预测，已明确排除出结果汇总。历史 reduced/timeout/smoke artifact 保持只读。
- 阻塞报告：`docs/analysis/DCLOOS_CURRENT_PROTOCOL_BLOCKER_V1.md`；当前不据此比较性能，也不启动 DCLOOS 多 seed。

### DCLOOS GPU 预算尝试（2026-08-10）

- 新运行 `../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_gpu_v1/` 已实际使用 RTX 5070（峰值显存约 11.8GB），说明训练链路确实进入 GPU。
- 在预先声明的 30 分钟上限内仍未生成最终 `metrics.json`；manifest 为 `timeout_incomplete`。仅保留中间 `predictions.npz`，并明确排除出指标和图表。
- 因此 DCLOOS 仍只有不同监督合同的 reduced 参考，没有当前 registry 的可比最终单格；本次只完成资源/预算诊断，不产生新的性能数字。
- 继续实验前必须解决可恢复训练或流式数据预算问题；不得重复同一超时命令，也不得启动 DCLOOS 多 seed。

## 统一对比合同收口（2026-08-11）

- 已将当前 protocol_v2 的 Trainable/Frozen MiniLM Gate、native detector、MOGB fair component 和完整 ADB 外部标签 artifact 转换为统一匿名 prediction contract；输出入口为 `docs/analysis/UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`。
- 机器可读合同位于 `results/analysis/unified_prediction_contract_v1/`，统一比较位于 `results/analysis/unified_comparison_v1/`；逐样本压缩 JSONL 仅保存在 `../artifacts/s2c/analysis/unified_prediction_contract_v1/`，不进入轻量公开结果。
- 当前合同审计确认 2,394,360 行、486 个运行组、441 个同序列对齐方法-cell；45 个 ADB 单元保留为 external BERT 合同。DCLOOS、DA-ADB 和缺失 final metrics 的传统 TextOIR 路线继续停留在 blocked/different-supervision 层，不进入 fair 排名。
- 本轮不新增训练、不调 test OOS 阈值、不重跑 E2/E3/MOGB/DCLOOS；后续只有在 split、Known list、seed、评估器和 final metrics 全部一致时，才提升外部方法到 fair 主表。
