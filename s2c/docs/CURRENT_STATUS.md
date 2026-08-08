# 当前研究状态

这是当前唯一状态入口。活动协议为 `protocol_v2_textoir_v1`；当前基准 commit 为
`a5a96fed4a779afdfb1586dfbe91efeb9565d541`，当前工作分支为
`main`。GitHub 已包含该基准 commit；当前工作树因本地 contract-repair 代码、轻量结果和
文档尚未提交而 dirty，父仓库没有运行中的实验。`third_party/mogb_official` 仍是独立只读来源
checkout，其本地审计元数据保持在子仓库工作树中，不修改第三方源码。

> 当前工作区核查（2026-08-06）：`../artifacts/s2c/runs/` 与 `../artifacts/s2c/cache/` 不存在；下文引用的
> E2/E3/Trainable 原始运行目录是历史完成记录，不代表这些大型产物仍可在当前 checkout 直接重跑。当前可直接
> 复核的是 `results/analysis/` 下的轻量 CSV/JSON/图表；重新训练前必须恢复或重新生成并冻结 provenance。

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
| minilm_trainable_k2_control_v1 | complete：同一 Trainable checkpoint 的 K=1/K=2 跨数据集配对，9/9（含 StackOverflow 只读行） | `docs/analysis/MINILM_TRAINABLE_K2_CONTROL_V1.md` |
| minilm_trainable_lambda_control_v1 | complete：3 数据集×3 seeds×2 K×6 λ，108/108；Known-only 选择，不用 test OOS | `docs/analysis/MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md` |
| minilm_trainable_kir_sweep_v1 | complete：3 数据集×3 KIR×3 seeds、Trainable K=1，27/27；同距离 Frozen 对照 | `docs/analysis/MINILM_TRAINABLE_KIR_SWEEP_V1.md` |

## 2026-08-06 新增分析证据

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

## 当前唯一下一步

停止 URCSG、CCSG、RC-AMBL、joint_adaptive_multicenter_v1、contract-repair、consistency_gate_v1、lambda-only rescue 和固定多中心扩展；保留这些 pilot 作为
Known-only 自适应 K 与 StackOverflow union-risk 的负结果证据。训练参与式链路已经验证可运行，但
当前候选分裂没有通过 Known-only 安全门。Trainable K=1 的五 seed/KIR 公平扩展和协议分层汇总已经完成；
Frozen/Trainable 的类内距离、半径分布、calibration coverage、训练动态和阈值稳定性分析也已完成。
下一步只能在明确登记新的目标函数/拒识机制修订后再决定是否做另一个同规模小 pilot；优先整理同协议强
baseline 的监督条件和工作点，而不是用 test oracle 追历史分数。后续不得自动扩展其他数据集、KIR、K=3--5、
Proxy-OOS、完整 Pipeline 或外部 baseline。

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
- 当前工作区的 `../artifacts/s2c/runs/` 不存在，故原始 run/checkpoint 不能在本地重跑；轻量 CSV 仍可审计，不能将
  状态文档中的旧 artifact 路径当作当前可复现实验证据。恢复原始产物或重新登记最小实验前，不启动依赖 checkpoint 的新矩阵。
