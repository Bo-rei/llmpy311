# s2c 公开结果快照

这里是可以随源码提交 GitHub 的轻量、可审计结果。历史原始产物在
`../../artifacts/s2c/outputs/experiments/`，candidate raw run 在
`../../artifacts/s2c/runs/protocol_v2/`，官方重建 run 在
`../../artifacts/s2c/runs/protocol_v2_official_v1/`；三者都不会整体复制到本目录。

- 结果文件由 `configs/public_results.yaml` 白名单生成。
- `MANIFEST.csv` 记录每个公开文件的来源相对路径、大小和 SHA256。
- `gate_only/` 只包含 Gate、多簇、near-OOS 和表示对照结果。
- `pipeline/` 包含完整 `Gate → Router → Expert` 的汇总和 provenance。
- `representation/` 包含 Frozen、CE、SupCon、CE-Recon 和 MiniLM 语义分析。
- `robustness/` 包含样本效率、边界形状和机制汇总。
- `external/` 包含 hard-negative 结果和 MOGB 协议审计；MOGB 没有公平性能数字时不生成伪结果。
- 被来源裁决冻结的 candidate Gate 汇总保留在
  `../docs/archive/historical_repro_bundle/protocol_v2_candidate_results/`，不属于此目录，也不得与
  历史 v19–v22 或官方重建数字混合。官方 `protocol_v2_official_v1` 的已准入 E1 Gate-only 摘要位于
  `gate_only/protocol_v2_official_e1_admitted.csv`，并由根 `MANIFEST.csv` 追溯。

Frozen、CE、SupCon 的表示结果属于 Gate-only 对照，不能解释为完整 Pipeline 结果。
CE-Recon 同时有 Gate-only 证据和完整 Pipeline 汇总，但两者仍必须分开阅读。

本目录不包含模型、embedding、checkpoint、Parquet 或逐样本输出。

`MANIFEST.csv` 本身不含绝对路径；复制的原始 provenance JSON 保留其原始字段，可能仍包含
生成机器路径，便于与 artifact 原文逐字核对。
