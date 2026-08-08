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

## 指标契约

Gate-only 主指标为 OOS F1、AUROC、AUPR-OOS、Known Recall、false acceptance/rejection；
完整开放意图分类另报 Accuracy、F1-All、F1-U、F1-K。任何表格必须注明
`protocol_version`、dataset/registry SHA、representation、K、distance、boundary 和 seed，
并明确是否使用真实或伪 OOS 监督。
