# 当前研究状态

这是当前唯一状态入口。活动协议为 `protocol_v2_textoir_v1`；本轮整理已冻结在
`03a6e26ed373747baccabbbb459d0a355af935ae`，当前工作分支为
`experiment/lambda-leakage-audit`。当前工作树因 λ 审计代码、轻量结果和文档尚未提交而
dirty；父仓库没有运行中的实验。`third_party/mogb_official` 仍是独立只读来源 checkout，
其本地审计元数据保持在子仓库工作树中，不修改第三方源码。GitHub 尚未推送本轮提交或
本阶段诊断结果。

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

## 当前唯一下一步

冻结并审阅 λ 审计结果，将其纳入论文的参数选择和泄漏说明；不注册或运行新的
split–merge adaptive-K pilot。若未来仍要研究自适应 K，必须先提供独立的 validation-OOS
或严格的 Known-only intent-level 选择契约，并重新登记实验，不得复用本阶段 test-oracle 行。

## 当前阻断和风险

- MOGB A/C 组合缺少原始配套数据；不能以 TextOIR 快照冒充原始数据。
- adaptive-K intent-level 指标是从 E2 test predictions 做的描述性 oracle 敏感性审计，
  不能用于正式选 K。
- λ sensitivity 的 Known-only 5% 误拒绝约束在部分数据集/ K 上不可满足；当前结果只能作为
  选择稳定性和泄漏审计证据，不能把 λ=2 解释为跨数据集正式默认值。
- StackOverflow 完整文本、模型、embedding、checkpoint 和逐样本结果仍只保留在本地，
  不进入 GitHub 结果快照。

## 当前证据入口

- 固定 K 审计：`results/diagnostics/adaptive_k/`；
- MOGB 四组合审计：`results/diagnostics/mogb_diff/` 和
  `docs/archive/mogb_reproduction/MOGB_DIAGNOSIS.md`；
- λ 数据集契约与敏感性：`results/diagnostics/lambda_leakage/data_split_audit.json`、
  `results/diagnostics/lambda_sensitivity/summary.csv`、`mean_std.csv`、
  `lambda_k_interaction.csv`、`adaptive_k_decision.json`；
- 机器总账：`docs/EXPERIMENT_LEDGER.csv`；
- 历史决策与 claim audit：`docs/archive/protocol_and_data/`、
  `docs/archive/external_baselines/`。
