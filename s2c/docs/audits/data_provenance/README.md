# 三方数据来源审计

此目录保存可提交的轻量裁决；完整逐样本比较、冲突样本散列和历史输入清单在
`../artifacts/s2c/reports/data_provenance_audit/2026-07-22_three_way_verification/`。

审计固定比较三方：官方/原始来源、TEXTOIR commit
`dffe2b1b848a069a6808f8089b4cb9bd16e2062b`，以及当前 `assets/datasets` 和历史
prepared 输入。`exact_record_match` 不清理文本；`normalized_text_match` 只做 NFKC、
casefold 和空白折叠，并仅用于诊断差异，绝不写回数据。

CLINC150 与 Banking77 已由官方 raw source 写入 protocol_v2_official_v1，且官方→重建
canonical 的 exact record、label、split、intent-set 与 per-class count match rate 均为 1.0。
它们只能在对应 view/export materialized 后运行；StackOverflow 与 BANKING77-OOS 继续被阻塞。
不得把旧候选 run 或 legacy 结果称为 protocol_v2/TEXTOIR 公平比较。

StackOverflow 的补充 source/license trace 位于 `stackoverflow/source_trace.md`：已固定公开内容仓库、
README 和两份 raw 文件的 SHA256，但该仓库没有许可证文件，且语料缺少可将每条标题绑定到 Stack
Overflow 原帖许可版本的 post metadata；因此裁决保持 `blocked_unverified`。

`summary.csv` 中的 `official_reconstruction_split_match_rate` 专指官方 raw source 与
`protocol_v2_official_v1` 重建版本的 split 一致性；它不表示 TEXTOIR 或历史 s2c prepared
split 相同。三方之间的完整 split 冲突仍保留在 artifact 审计。`source_license_report.csv`
列出每个来源的许可证证据、固定 revision 与 SHA256；它不是对未验证 StackOverflow 语料的许可补充。
