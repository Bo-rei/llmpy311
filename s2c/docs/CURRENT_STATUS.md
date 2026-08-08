# 当前研究状态

这是当前唯一状态入口。活动协议为 `protocol_v2_textoir_v1`；当前基准 commit 为
`a5a96fed4a779afdfb1586dfbe91efeb9565d541`，当前工作分支为
`main`。GitHub 已包含该基准 commit；当前工作树因本地 contract-repair 代码、轻量结果和
文档尚未提交而 dirty，父仓库没有运行中的实验。`third_party/mogb_official` 仍是独立只读来源
checkout，其本地审计元数据保持在子仓库工作树中，不修改第三方源码。

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

## 当前唯一下一步

停止 URCSG、CCSG、RC-AMBL、joint_adaptive_multicenter_v1、contract-repair、consistency_gate_v1 和固定多中心扩展；保留这些 pilot 作为
Known-only 自适应 K 与 StackOverflow union-risk 的负结果证据。训练参与式链路已经验证可运行，但
当前候选分裂没有通过 Known-only 安全门。本次 contract-repair 已完成且仍为负结果；下一步只能在明确登记新的
目标函数/拒识机制修订后再决定是否做另一个同规模小 pilot，后续不得自动扩展其他数据集、KIR、
K=3--5、Proxy-OOS、完整 Pipeline 或外部 baseline。

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

## 当前证据入口

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
