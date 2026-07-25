# 结果契约

## Gate-only

`protocol_v2_textoir_v1` 中 OOS 为正类，分数越高越倾向 OOS。每个 Gate run 至少保存 OOS F1、
precision、recall、ID recall、AUROC、AUPR-OOS、FPR95、false accept/reject、有效 cluster 数、
最小 cluster size、评分时间、吞吐和峰值内存。CLINC150 还分别保留 held-out/native/combined OOS。

## 完整 Pipeline

`Gate → Router → Expert` 必须与 Gate-only 分表，至少报告 Known macro-F1、OOS F1、overall
accuracy、Gate false accept/reject、Router error、Expert error 和端到端耗时。Frozen/CE/SupCon 的
Gate-only 表不等于完整 Pipeline 结论。

## 路径隔离

- 活动 raw run：`../artifacts/s2c/runs/protocol_v2_textoir_v1/`。
- 活动轻量结果：`results/protocol_v2_textoir_v1/`（只含小型 CSV/JSON/manifest）。
- 原始语料、embedding、checkpoint、Parquet、逐样本分数和运行日志只留在 Git 忽略的本地路径。
- `protocol_v2_official_v1` 与 legacy `protocol_v2` 只能作审计，不能与活动协议混表。
