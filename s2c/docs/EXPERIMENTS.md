# 实验索引

这里只列当前论文可引用的实验家族和证据边界；原始输出永远以
`../artifacts/s2c/` 下的 manifest 为准。

## 主协议结果

| 家族 | 覆盖 | 证据与解释 |
|---|---|---|
| E2 Gate-only | 3 datasets × 11 KIR × 5 seeds × K=1..5 × 2 distances，1,650/1,650 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/`；描述 K/KIR/距离敏感性，不是完整 Cascade |
| E3 mechanism controls | KMeans/random-balanced、ARI、tiny-cluster、Known-only coverage | `../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/`；机制诊断，不产生 adaptive-K 正式方法 |
| 表示对照 | Frozen、CE、SupCon、CE-Recon 与 contract repair/M1 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/{r1_*,minilm_training_and_stackoverflow_repair_v1}/`；只作为 Gate 表示证据 |
| 历史 Cascade | 代表性 Gate→Router→Expert 单元 | `../artifacts/s2c/outputs/experiments/cascade_repair/` 与 `cascade_full/`；协议不同，不能与 E2 直接混表 |

## 外部基线

MOGB、ADB、DA-ADB、DCLOOS 均单独记录训练监督、数据契约和复现状态。MOGB 四组合差异
诊断见 `results/diagnostics/mogb_diff/`；原始代码和 negative reproduction 见
`docs/archive/mogb_reproduction/`。DCLOOS 的外部 OOS 监督缺口和 ADB/DA-ADB 兼容审计
见 `docs/archive/external_baselines/`。这些结果不能被改写成 s2c 原创方法或统一 SOTA 数字。

## 当前 adaptive-K 证据

`results/diagnostics/adaptive_k/` 复用 E2 的固定 K=1..5 逐样本预测，输出 dataset、intent
和 seed 层面的描述性 oracle 对照。dataset-level 提供全局 F1-All/Known Recall 差值；
intent-level 的 `delta_intent_class_f1_vs_k1` 是单个 Known 类的类别 F1 差值，不冒充全局
F1-All。它回答“哪些意图在测试敏感性分析中偶尔受益”，不提供合法的验证集选 K。正式实验
在 split–merge 原型通过 dry-run 和 Known-only 门之前不得启动。

## URCSG Known-only 自适应 K pilot

`urcsg_pilot_v1` 是一次独立的、已完成的选择器诊断：在每个 intent 上只替换目标意图的
K=1..5，使用 leave-one-known-intent-out calibration 估计新增接受风险，并以 coverage 安全门
和 Wilson UCB95 约束选择 K。它复用了冻结 all-MiniLM cache，未重新训练/编码，测试 OOS 只用于
最终描述性评价；`oracle_test_k` 明确不参与选择。

6/6 cell（Banking77、StackOverflow × 3 seeds）完成，但 URCSG-primary 未通过预注册门：
Banking77 相对 single-centroid 的 OOS F1/F1-All 为 -0.39/-0.35 pp，StackOverflow 为
-6.10/-2.67 pp 且 false acceptance +10.93 pp。选择器在 StackOverflow 仍有过多 K>1，
shuffled episode control 也没有形成可用于晋级的优势。因此该阶段仅作为负结果和机制证据，
不启动 full matrix，不把它称为最终 adaptive-K 方法。

证据入口：`results/diagnostics/urcsg/` 和
`../artifacts/s2c/runs/protocol_v2_textoir_v1/urcsg_pilot_v1/`。

## RC-AMBL risk-calibrated pilot

`adaptive_v1` 是一个独立的结构自适应 pilot，不重复 E2/E3/URCSG/CCSG/BRAK。它固定
StackOverflow、KIR=.50、seeds 13/42/87，使用冻结 MiniLM、PCA median split、父级边界保护、
收缩对角协方差和类级加权 evidence。KnownOnly 与 ProxyOOS 共 6/6 单元完成，测试 OOS 不参与
选择。所有候选分裂均被安全门拒绝，最终 `K_y=1`；RC-AMBL 的 OOS F1 `0.5785±0.0926`，
相对 E2 K=1 下降 `19.44pp`，false acceptance 增加 `29.14pp`，因此该阶段停止，不进入其他
数据集、KIR 或 K 网格。

证据入口：`docs/adaptive_v1/ADAPTIVE_V1_REPORT.md`、
`../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/` 和
`results/diagnostics/adaptive_v1/`。

## CCSG competition-calibrated support pilot

`ccsg_pilot_v1` 是 URCSG 失败后单独登记的评分机制实验，不是 E2/E3 或 adaptive-K 的
重复。它复用冻结 all-MiniLM、相同 registry/view/export、diagonal Mahalanobis 和
mean+std 半径；只把子中心从独立 union 接受器改为意图级混合支持分数，并加入 top-1/top-2
竞争间隔。阈值完全由 Known calibration 的固定 5% false-rejection 目标产生。

覆盖为 3 datasets × KIR=.50 × 3 seeds × 8 机制配置（72 metric rows）。配置包括当前
K=1/K=2 union、mixture-support、margin-only、CCSG joint 和 independent-AND 消融。9/9
cells 完成且无失败、缺失、重复或测试选择泄漏；但 Banking77、CLINC150 和 StackOverflow
的预注册晋级门均未通过，StackOverflow K=2 的 false acceptance 仍比 CCSG-K1 高 3.00pp。
因此 CCSG 不扩展到 full matrix，当前仅作为“类级支持聚合不能在冻结表示上自动救活多中心”的
负结果。

证据入口：`results/diagnostics/ccsg/` 和
`../artifacts/s2c/runs/protocol_v2_textoir_v1/ccsg_pilot_v1/`。

## RACAL-v1 Trainable K=1 阶段一

`racal_v1_stage1` 是独立的表示训练控制实验，不重复 E2/E3/R1/URCSG/CCSG/RC-AMBL，也不运行
多中心。它先对 StackOverflow、KIR=.50、seeds 13/42/87 做 Frozen K=1 的 E2 精确回放，再比较
Known-only 选择 checkpoint 的 Trainable MiniLM K=1。训练使用 projection warm-up 和最后两层
解冻，测试 OOS 不参与 epoch、阈值或边界选择。

Frozen 与 Trainable 均完成 3/3。Trainable 的 OOS F1 为 `0.8671±0.0079`，相对 Frozen
`0.7729±0.0417` 提升 `+9.42pp`；F1-All 提升 `+7.06pp`，Known Recall 变化 `+0.21pp`，
false acceptance 下降 `15.40pp`。这只证明 K=1 表示适配有效，不证明固定多中心或 RACAL 完整方法
已经成立；固定 K=2、中心激活、Proxy-OOS 和其他数据集均未启动。

证据入口：`docs/racal_v1/RACAL_V1_REPORT.md`、`docs/racal_v1/RACAL_V1_CLOSEOUT.md`、
`results/diagnostics/racal_v1/` 和
`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/`。

## RACAL-v1 阶段二：Trainable 表示下的固定 K=2 归因

`racal_v1_stage2_fixed_k2` 复用阶段一已经按 Known-only 选定并冻结的 Trainable MiniLM
checkpoint，不重新训练编码器。范围固定为 StackOverflow、KIR=.50、seeds 13/42/87；在每个
Known intent 内比较纯 K=1 与 KMeans K=2，距离为对角 Mahalanobis，半径为 mean+1 std，
阈值为 1。该阶段不使用 proxy-OOS、测试 OOS 选参、风险门、K=3--5 或其他数据集。

3/3 runs 完成且阶段一 K=1 replay 最大指标差为 0。Trainable K=2 相对 K=1 的均值为：OOS F1
`-19.06pp`、F1-All `-8.85pp`、Known Recall `+9.70pp`、false acceptance `+34.11pp`、
AUROC `-2.30pp`；三个 seed 的 OOS F1 方向一致下降。新增 OOS 误接收分别为 1169、753、1154，
恢复 Known 分别为 298、309、285。10 个 intent 的诊断显示存在异质性，但整体并集过覆盖风险占主导，
因此判定为 `A_primary_with_C_heterogeneity`，停止 K=3--5。

证据入口：`docs/racal_v1/RACAL_V1_STAGE2_REPORT.md`、`docs/racal_v1/RACAL_V1_STAGE2_CLOSEOUT.md`、
`results/diagnostics/racal_v1/stage2_fixed_k2/` 和
`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/`。

## 训练参与式自适应多中心合同修复

`joint_adaptive_multicenter_contract_repair_v1` 是在 StackOverflow/KIR=.50、seeds 13/42/87 上对
真正共同训练候选中心的独立合同修复。它冻结 K=1 父边界，使用 parent-guarded compactness，并在
候选训练损失中加入子中心负载平衡和中心分离。3/3 候选均实际训练，但 Known calibration Recall
明显下降而被拒绝，最终 `mean K_y=1.0`；OOS F1 `0.8661±0.0091`。该阶段证明训练链路真实执行，
但未证明当前数据上有安全多中心收益。

证据入口：`docs/joint_adaptive_multicenter_contract_repair_v1/CONTRACT_REPAIR_REPORT.md` 和
`../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1/repair3/`。

## consistency_gate_v1 单中心拒识 pilot

`consistency_gate_v1` 复用 RACAL Trainable K=1 checkpoint，不增加中心、不重新训练 encoder；使用
原始、两次固定 MC-dropout 和表面归一化视图，Known calibration 选择证据 margin 与冲突容忍度。
3/3 完成。evidence-margin 的 OOS F1 `0.8673±0.0076`、F1-All `0.8580±0.0027`、Known Recall
`0.8376±0.0020`、false acceptance `0.1099±0.0145`。收益很小，只能作为单中心拒识候选，
不能称为 SOTA 或多中心成功。

证据入口：`docs/consistency_gate_v1/CONSISTENCY_GATE_REPORT.md` 和
`../artifacts/s2c/runs/protocol_v2_textoir_v1/consistency_gate_v1/`。

## MiniLM Trainable K=1 跨数据集控制

`minilm_trainable_control_v1` 是当前新增的跨数据集表示训练控制，不扩展 K、不引入新多中心规则。
范围为 KIR=.50、Mahalanobis K=1、seeds 13/42/87；CLINC150 和 Banking77 新运行 6/6，
StackOverflow 复用 RACAL-v1 同协议 K=1。Frozen 基线逐 seed 读取同一 E2 K=1 单元，避免使用
不同 seed 或预聚合 CSV。

Trainable 相对 Frozen 的 OOS F1 变化为：CLINC150 `+1.12pp`、Banking77 `+5.18pp`、
StackOverflow `+9.42pp`；Known Recall 变化为 `-1.33pp`、`-1.95pp`、`+0.21pp`。该阶段说明
Known-only MiniLM 表示适配值得继续做，但不是无条件优于 Frozen；它仍然不能解释或修复 StackOverflow
固定 K=2 的 false acceptance 爆炸。

证据入口：`docs/analysis/MINILM_TRAINABLE_CONTROL_V1.md`、
`results/diagnostics/minilm_trainable_control_v1/`、
`../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_control_v1/`。

## MiniLM Trainable K=1/K=2 跨数据集配对控制

`minilm_trainable_k2_control_v1` 复用上一个阶段的已冻结 checkpoint，不重新训练模型；CLINC150 和
Banking77 各完成 3 个 seed 的 K=1/K=2 评价，StackOverflow 读取已完成的同协议 RACAL Stage2 配对结果。
K=2−K=1 的 OOS F1 为 CLINC150 `-0.28pp`、Banking77 `+0.13pp`、StackOverflow `-19.06pp`。
对应 Known Recall 变化为 `-3.26pp`、`-1.95pp`、`+9.70pp`，StackOverflow false acceptance 增加
`34.11pp`。这说明 Trainable MiniLM 的收益主要属于 K=1 表示和分数排序，不能据此推出固定多中心安全。

证据入口：`docs/analysis/MINILM_TRAINABLE_K2_CONTROL_V1.md`、
`results/diagnostics/minilm_trainable_k2_control_v1/`、
`figures/active_experiment_dashboard_v1/trainable_k_interaction_cross_dataset.png`。

## MiniLM Trainable λ/K 交互控制

`minilm_trainable_lambda_control_v1` 复用同一批 Trainable MiniLM checkpoint，在不重新训练的前提下
评价 λ={0.50,0.75,1.00,1.25,1.50,2.00} 与 K={1,2} 的组合。每个 dataset×seed×K 的 λ 选择只看
Known calibration 的 false-reject 约束，测试 OOS 不参与选择。108/108 个单元完成；三数据集的
Known-only 选择后 K=2−K=1 OOS F1 分别为 CLINC150 `+0.79pp`、Banking77 `+2.08pp`、
StackOverflow `-9.51pp`，StackOverflow false acceptance 仍增加 `+11.83pp`。因此固定 K=2 的
StackOverflow 退化不是 λ=1 的偶然问题，Trainable K=1 仍是当前安全基线。

证据入口：`docs/analysis/MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md`、
`results/diagnostics/minilm_trainable_lambda_control_v1/`、
`figures/active_experiment_dashboard_v1/trainable_lambda_k_interaction.png`。

## MiniLM Trainable KIR sweep（K=1）

`minilm_trainable_kir_sweep_v1` 将同一最后两层 MiniLM+projection Trainable K=1 控制扩展到
CLINC150、Banking77、StackOverflow 的 KIR={.25,.50,.75} 和 seeds={13,42,87}，共 27/27 个单元。
训练、checkpoint 选择和 Gate 评价只访问 Known train/calibration；不使用 test OOS 选择或训练。
该阶段只回答“表示适配在不同已知意图密度下是否改善单中心 Gate”，不重新扫描 K，也不构成
自适应多中心方法。

结果显示 CLINC150 和 StackOverflow 的 K=1 OOS F1 在三个 KIR 均高于同距离 Frozen；Banking77
只有低 KIR 有收益，KIR=.75 明显下降。证据入口：
`docs/analysis/MINILM_TRAINABLE_KIR_SWEEP_V1.md`、
`results/analysis/minilm_trainable_kir_sweep_v1/`、
`../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1/`。

## 指标契约

Gate-only 主指标为 OOS F1、AUROC、AUPR-OOS、Known Recall、false acceptance/rejection；
完整开放意图分类另报 Accuracy、F1-All、F1-U、F1-K。任何表格必须注明
`protocol_version`、dataset/registry SHA、representation、K、distance、boundary 和 seed，
并明确是否使用真实或伪 OOS 监督。

## 多 KIR、多方法 evidence pack

`experiment_evidence_pack_v2` 是对已完成结果的 analysis-only 汇总，不运行训练、不修改历史 artifacts。
它读取 MOGB fair matrix 和 Trainable K=1 KIR sweep，输出 paired bootstrap、win/tie/loss、KIR 曲线、
Known Recall–OOS F1 权衡图和差值热图。入口为：

`docs/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md`、
`results/analysis/experiment_evidence_pack_v2/`、
`figures/experiment_evidence_pack_v2/`。

## 表示训练与多中心边界交互分析（analysis-only）

`representation_boundary_pack_v1` 读取已有 Frozen、CE 和 SupCon 结果，比较 K=1/K=2 的 OOS F1、
Near-OOS F1、Known Recall 和表示几何。StackOverflow 上三种表示均出现 K=2 OOS 退化，
但 Known Recall 上升；这说明表示训练收益不能直接解释为多中心收益。

`stackoverflow_intent_diagnostic_v1` 读取已有 RACAL Stage-2 intent diagnostics，按意图统计新增
OOS 接受量、恢复 Known 量、ARI、silhouette、簇规模和半径比，并生成 4 张诊断图。平均新增 OOS
接受量明显高于恢复 Known 量，支持“稳定聚类仍可能导致 boundary-union 过覆盖”的机制解释。

证据入口：`docs/analysis/REPRESENTATION_BOUNDARY_PACK_V1.md`、
`docs/analysis/STACKOVERFLOW_INTENT_MULTI_CENTER_DIAGNOSTIC_V1.md`、
`results/analysis/representation_boundary_pack_v1/`、
`results/analysis/stackoverflow_intent_diagnostic_v1/`、
`figures/representation_boundary_pack_v1/`、`figures/stackoverflow_intent_diagnostic_v1/`。

## 意图级 KIR 稳定性分析（analysis-only）

`intent_kir_stability_pack_v1` 读取已完成的 `results/diagnostics/adaptive_k/intent_level.csv`，按
dataset×KIR×distance 汇总 oracle best-K、OOS F1/Known Recall 差值、K>1 比例和跨 seed 稳定性，
并生成 4 张图。它用于回答“多中心收益是否只集中在部分意图/开放程度”，不用于正式选择 K，且不新增训练。

证据入口：`docs/analysis/INTENT_KIR_STABILITY_PACK_V1.md`、
`results/analysis/intent_kir_stability_pack_v1/`、`figures/intent_kir_stability_pack_v1/`。

## MiniLM Trainable K=1 五 seed 公平扩展

`minilm_trainable_kir_sweep_extension_v1` 补齐了原有 Trainable K=1 KIR sweep 的 seed=100/123，
使 CLINC150、Banking77、StackOverflow 在 KIR={.25,.50,.75} 下均达到五个正式 seed。该阶段没有
重跑旧单元、没有扩大 K、没有使用测试 OOS 选 checkpoint 或边界参数。

配对结果以同一 E2 Frozen K=1 为参考：Trainable 的 OOS F1 在 CLINC150/Banking77/StackOverflow
分别提高 `0.45--1.38pp`、`2.47--6.94pp`、`5.06--10.50pp`；Known Recall 的最大下降分别为
`1.37pp`、`2.90pp`，StackOverflow 则小幅上升。结论仍限定为 Known-only 单中心表示适配证据，
不能写成 SOTA 或固定多中心成功。

证据入口：`docs/analysis/MINILM_TRAINABLE_5SEED_FAIR_COMPARISON_V1.md`、
`results/analysis/minilm_trainable_5seed_fair_v1/`、`figures/minilm_trainable_5seed_fair_v1/`。

## MiniLM 表示—边界诊断（analysis-only）

`minilm_boundary_diagnostics_v1` 读取同一批五 seed Trainable/Frozen K=1 预测，比较 Known/OOS
normalized score 分布、最近半径、score gap、false acceptance 和 KIR 趋势；不重训、不选择参数。
KIR=.50 时，StackOverflow 的 median OOS−Known score gap 从 `0.116` 增至 `0.434`，false acceptance
从 `25.03%` 降至 `9.34%`；CLINC150 和 Banking77 也分别出现 `0.258→0.438`、`0.259→0.334` 的
score-gap 增益。该结果把 Trainable K=1 的收益定位为表示/分数排序改善，而非多中心安全性证明。

证据入口：`docs/analysis/MINILM_BOUNDARY_DIAGNOSTICS_V1.md`、
`results/analysis/minilm_boundary_diagnostics_v1/`、`figures/minilm_boundary_diagnostics_v1/`。

## MiniLM 训练动态诊断（analysis-only）

`minilm_training_dynamics_v1` 读取 45 个 Trainable K=1 run 的 Known calibration history，分析
checkpoint 选择目标、最佳 epoch 与最终 test OOS F1 的关系。它不改变任何 checkpoint，也不以 test
OOS 重新选 epoch；结果用于判断“继续训练”是否是当前差距的合理解释。

KIR=.50 时 calibration→test Known Recall 转移差异为 CLINC150 `-0.64pp`、Banking77 `-0.21pp`、
StackOverflow `+0.33pp`，说明当前高 KIR OOS 下降主要需要从 OOS score 和边界校准解释，而不是简单
增加训练轮数。

证据入口：`docs/analysis/MINILM_TRAINING_DYNAMICS_V1.md`、
`results/analysis/minilm_training_dynamics_v1/`、`figures/minilm_training_dynamics_v1/`。

## Trainable 与 MOGB 组件五 seed 配对分析（analysis-only）

`trainable_vs_mogb_component_v1` 将 Trainable K=1 与 Frozen MiniLM 下的 MOGB 动态分区、MOGB 边界和
固定 K=2 组件按 dataset×KIR×seed 配对，输出 OOS F1、F1-All、Known Recall、false acceptance 的
10,000 次 paired bootstrap、win/tie/loss 和三张图。它明确把“更高 OOS F1”和“更低 false acceptance”
分开报告，不把组件适配称为官方 MOGB 复现。

证据入口：`docs/analysis/TRAINABLE_VS_MOGB_COMPONENT_V1.md`、
`results/analysis/trainable_vs_mogb_component_v1/`、`figures/trainable_vs_mogb_component_v1/`。

## 统一实验结果索引

`docs/analysis/EXPERIMENTAL_EVIDENCE_INDEX_V1.md` 是当前中文实验地图，集中链接 E2/E3、Trainable、
表示—边界诊断、训练动态、MOGB 组件、官方 MOGB 复现、ADB/DA-ADB 和 DCLOOS 的证据，并明确每类结果
是否可以公平排名。

## Threshold 与半径稳定性诊断（analysis-only，2026-08-06）

新增 `threshold_radius_stability_v1`，读取五 seed Trainable/Frozen K=1 的既有预测，扫描
`threshold={.75,.85,.90,.95,1.00,1.05,1.10,1.20,1.30}`，并统计训练球半径和测试样本被分配半径的
变异系数。该阶段不训练、不修改 checkpoint、不选择正式阈值；test score 网格只用于解释表示依赖的
工作点差异。

KIR=.50 的诊断性最佳 threshold（Frozen/Trainable）为 CLINC150 `1.00/1.05`、Banking77 `0.90/0.95`、
StackOverflow `0.95/0.95`。因此当前 Trainable 与 `fulltex.tex` 的差距不能简单归因于“MiniLM 没有训练”：
Trainable 已改善 K=1 score 分离，但仍处于 Gate-only、固定 threshold=1、Known-only 选模协议；历史
fulltex 是包含 Router/Expert、不同表示和历史 OOS 校准合同的 Cascade。正式结果仍只能使用 threshold=1。

证据：`docs/analysis/THRESHOLD_RADIUS_STABILITY_V1.md`、
`results/analysis/threshold_radius_stability_v1/`、`figures/threshold_radius_stability_v1/`。

## 同协议方法权衡与可视化（analysis-only，2026-08-06）

`cross_protocol_tradeoff_v1` 将 315 个已完成五 seed 行按 dataset×KIR×seed 重新整理，覆盖 Trainable K=1、
Frozen K=1/K=2、Random K=2、MOGB partition+s2c boundary、s2c partition+MOGB boundary 和 MOGB-MiniLM，
并计算 486 个 Trainable 相对组件的 paired bootstrap effects。该 fair 包不包含历史 `fulltex.tex`、官方
BERT MOGB 或 DCLOOS 外部 OOS 监督。

KIR=.50 的图形显示：MOGB 风格组件通常通过拒绝更多 Known 样本换取较低 false acceptance；Trainable K=1
保留更高的 Known Recall/F1-All，形成更平衡的工作点。StackOverflow 固定 K=2 的低 OOS F1 和高 false
acceptance 同时出现，支持“多球接受区域组合”而不是“MiniLM 没训练”是主要风险。

证据：`docs/analysis/CROSS_PROTOCOL_TRADEOFF_V1.md`、
`results/analysis/cross_protocol_tradeoff_v1/`、`figures/cross_protocol_tradeoff_v1/`。

## Gate→Cascade 桥接与误差分解（analysis-only）

`gate_cascade_bridge_v1` 读取当前协议已经完成的 3-seed Cascade 变体和 3-seed Trainable K=1 Gate-only，
将共享的 OOS F1、ID/ Known Recall、Gate false acceptance、Known false rejection 与 Expert error 分开
展示。该分析明确：完整 Cascade 的 OOS 结果不是 MiniLM Gate 单独决定的，Router/Expert 和拒识工作点会改变
最终数字；因此当前 Trainable Gate-only 低于 `fulltex.tex` 不能简单归因于 MiniLM 训练失败。

证据：`docs/analysis/GATE_CASCADE_BRIDGE_V1.md`、
`results/analysis/gate_cascade_bridge_v1/`、`figures/gate_cascade_bridge_v1/`。

## Trainable 与 MOGB 组件归因（2026-08-06）

`trainable_vs_mogb_ablation_v1` 只连接已完成结果：45 个 Trainable K=1 五 seed 行和 180 个 MOGB frozen-MiniLM 的 Euclidean/Mahalanobis、mean/mean+std 组件行。它生成 324 个 paired effects、180 个机制行和 3 张图，回答“自有方法相对 MOGB 组件的优势是表示/覆盖，还是动态划分本身”。

报告：`docs/analysis/TRAINABLE_VS_MOGB_ABLATION_V1.md`；结果：`results/analysis/trainable_vs_mogb_ablation_v1/`；图：`figures/trainable_vs_mogb_ablation_v1/`。

## 当前协议原生 baseline 矩阵（2026-08-06）

`native_baselines_v1` 已完成 180/180：3 数据集 × KIR={.25,.50,.75} × 5 seeds × {MSP, Energy, kNN, LOF}，冻结 `all-MiniLM-L6-v2`、Known-only calibration、同一 registry/view。该矩阵用于补足同协议的原生 score/OOD 控制，不冒充 ADB、DA-ADB、MOGB 或 DCLOOS。

KIR=.50 的均值和方差见 `docs/analysis/NATIVE_BASELINES_V1.md`；Trainable K=1 默认有更高 OOS F1，但通常有更低 Known Recall。`OPERATING_POINT_DIAGNOSTIC_V1.md` 将各方法回顾性对齐到共同 Known Recall，用于解释工作点差异；由于阈值使用 test 标签，该文件只能作诊断。

证据：`results/analysis/native_baselines_v1/`、`results/analysis/operating_point_diagnostic_v1/`、`figures/native_baselines_v1/`、`figures/operating_point_diagnostic_v1/`。

## Trainable 表示上的原生 detector 归因（2026-08-06）

`native_baselines_trainable_v1` 复用已完成的 Trainable MiniLM checkpoint，在 KIR=.50、三个 seed 上
运行 MSP/Energy/kNN/LOF（36/36），并与同协议 Frozen native 行配对。该实验不重新训练、不使用测试
OOS 选阈值，专门区分“表示改变”与“当前 Gate 几何”的贡献。

- 逐 seed：`results/analysis/native_baselines_trainable_v1/trainable_native_per_seed.csv`；
- 配对效果：`trainable_native_vs_gate_paired.csv`、`trainable_vs_frozen_native_paired.csv`；
- 图表：`figures/native_baselines_trainable_v1/`；
- 中文报告：`docs/analysis/NATIVE_BASELINES_TRAINABLE_V1.md`。

## 实验机制分析包 V3（2026-08-06）

新增 `tools/analysis/build_experimental_mechanism_pack_v3.py`，只读取已有的 315 行五 seed 轻量结果，
生成配对 bootstrap、方法汇总、Pareto 工作点和四张图；不读取或重建大型 artifact。报告入口：
`docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`。

本包不是新训练结果，而是对现有 Trainable/Frozen/MOGB 组件实验的可视化和机制收口：Trainable K=1 的优势
主要在表示适配后的分数排序和覆盖—拒识平衡，固定 K>1 仍不能作为 StackOverflow 的默认结构。当前重型
`../artifacts/s2c/runs/` 已不在工作区，后续如需重新训练必须先恢复并核对 provenance。
