# 结果契约

## Gate-only

protocol_v2 Gate-only 输出 OOS 为正类、分数越大越 OOS，并至少记录 OOS F1、precision、
recall、ID recall、AUROC、AUPR-OOS、FPR95、false accept/reject、评分时间、吞吐、峰值
内存、有效 cluster 数和最小 cluster size。CLINC150 还必须分别给出 held-out intent、native
OOS 与 combined 结果。

## 完整 Pipeline

`Gate → Router → Expert` 结果与 Gate-only 分表，且至少包括 Known macro-F1、OOS F1、
overall accuracy、router/expert error 和 Gate false accept/reject。不能把 Frozen/CE/SupCon
Gate-only 表述为完整 pipeline 效果。

## 文件位置

- Raw、逐样本、模型和日志：`../artifacts/s2c/`。
- 被封锁 candidate run：`../artifacts/s2c/runs/protocol_v2/`。
- 官方重建 run：`../artifacts/s2c/runs/protocol_v2_official_v1/`；当前只含 CLINC150、Banking77
  的 E1 Gate-only 子矩阵，公开 CSV 为 `results/gate_only/protocol_v2_official_e1_admitted.csv`。
- Git 可公开汇总：`results/`（只含 `configs/public_results.yaml` 白名单中的小型 CSV/JSON）。
- 旧 `protocol_v2` candidate 的公开快照仅作历史审计，位于
  `docs/archive/historical_repro_bundle/protocol_v2_candidate_results/`；它不属于当前公开结果，
  也不得用于 official 或 TEXTOIR 公平可比性声明。

结果没有 manifest 或输入 SHA256 时不应进入论文表格。
