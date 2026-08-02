# 当前研究状态

这是当前唯一状态入口。活动协议为 `protocol_v2_textoir_v1`；当前 checkout 基准
commit 为 `7c9008d7e85c637334139783a91a80841725d628`（`origin/main` 同步）。当前
工作树为 dirty，尚未提交；本轮代码/脚本/工具/测试/配置文件内容清单 SHA256 为
`171e89996d68f418710455fad00453fdebe02af476cf94af744e31d2ed72d27d`。GitHub 仍只
显示基准 commit，不包含本轮本地整理。父仓库没有运行中的实验；pinned
`third_party/mogb_official` 是独立只读来源 checkout，其审计元数据尚未进入父仓库提交。

## 已完成且不得重复

| 阶段 | 状态 | 证据 |
|---|---|---|
| E0 | complete：3 数据集、165 registry、165 views、990 exports、runtime independence | `docs/audits/protocol_v2_implementation/` |
| E1 | complete：36/36 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e1_gate_smoke.csv` |
| E2 | complete：1,650/1,650，0 failed/missing/duplicate/invalid | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/` |
| E3 | complete：720 partition-control、180 诊断组 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/` |
| R1/M1 | 已完成但按 contract audit superseded；不得重跑 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_*`、`minilm_training_and_stackoverflow_repair_v1/` |
| MOGB/ADB/DA-ADB/DCLOOS | 已审计或完成隔离单元；不扩展旧矩阵 | `docs/archive/mogb_reproduction/`、`docs/archive/external_baselines/` |

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

## 当前唯一下一步

审查本轮 `results/diagnostics/adaptive_k/` 与
`results/diagnostics/mogb_diff/`，然后只在 Known-only 证据支持时注册一个新的
split–merge adaptive-K pilot；在此之前不启动新的 MOGB、DCLOOS、ADB、DA-ADB、BRAK
扩展、表示学习矩阵或完整 Pipeline。若注册 pilot，必须先完成 λ sensitivity 与
Known/OOS 调参隔离审计，不能把 test-oracle 结果当作正式选 K 证据。

## 当前阻断和风险

- MOGB A/C 组合缺少原始配套数据；不能以 TextOIR 快照冒充原始数据。
- adaptive-K intent-level 指标是从 E2 test predictions 做的描述性 oracle 敏感性审计，
  不能用于正式选 K。
- StackOverflow 完整文本、模型、embedding、checkpoint 和逐样本结果仍只保留在本地，
  不进入 GitHub 结果快照。

## 当前证据入口

- 固定 K 审计：`results/diagnostics/adaptive_k/`；
- MOGB 四组合审计：`results/diagnostics/mogb_diff/` 和
  `docs/archive/mogb_reproduction/MOGB_DIAGNOSIS.md`；
- 机器总账：`docs/EXPERIMENT_LEDGER.csv`；
- 历史决策与 claim audit：`docs/archive/protocol_and_data/`、
  `docs/archive/external_baselines/`。
