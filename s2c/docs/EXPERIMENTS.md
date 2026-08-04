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

## 指标契约

Gate-only 主指标为 OOS F1、AUROC、AUPR-OOS、Known Recall、false acceptance/rejection；
完整开放意图分类另报 Accuracy、F1-All、F1-U、F1-K。任何表格必须注明
`protocol_version`、dataset/registry SHA、representation、K、distance、boundary 和 seed，
并明确是否使用真实或伪 OOS 监督。
