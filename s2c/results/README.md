# s2c 公开结果快照

这里是可以随源码提交 GitHub 的轻量、可审计结果。完整原始产物仍在
`../../artifacts/s2c/outputs/experiments/`，不会复制到本目录。

- 结果文件由 `configs/public_results.yaml` 白名单生成。
- `MANIFEST.csv` 记录每个公开文件的来源相对路径、大小和 SHA256。
- `gate_only/` 只包含 Gate、多簇、near-OOS 和表示对照结果。
- `pipeline/` 包含完整 `Gate → Router → Expert` 的汇总和 provenance。
- `representation/` 包含 Frozen、CE、SupCon、CE-Recon 和 MiniLM 语义分析。
- `robustness/` 包含样本效率、边界形状和机制汇总。
- `external/` 包含 hard-negative 结果和 MOGB 协议审计；MOGB 没有公平性能数字时不生成伪结果。

Frozen、CE、SupCon 的表示结果属于 Gate-only 对照，不能解释为完整 Pipeline 结果。
CE-Recon 同时有 Gate-only 证据和完整 Pipeline 汇总，但两者仍必须分开阅读。

本目录不包含模型、embedding、checkpoint、Parquet 或逐样本输出。

`MANIFEST.csv` 本身不含绝对路径；复制的原始 provenance JSON 保留其原始字段，可能仍包含
生成机器路径，便于与 artifact 原文逐字核对。
