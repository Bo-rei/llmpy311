# 从 v19 到 protocol_v2

v19–v22 artifacts 与脚本是历史血缘，禁止移动、重命名、覆盖或删除。protocol_v2 是并行的
可复现数据与实验协议，不会隐式修改旧 `WorkspacePaths`、旧 `assets/datasets` 或旧结果。

| 项目 | 历史协议 | protocol_v2 |
| --- | --- | --- |
| 数据根 | `assets` 与准备后快照 | `s2c/data` 本地导入、canonical、views |
| Banking | 可能包含 Banking77-OOS 扩展 | 仅标准 Banking77 |
| StackOverflow | 旧去重 19,980 行 | TEXTOIR 快照完整 20,000 行 |
| Known 抽样 | 由各历史流程决定 | 429 个固定 registry |
| 结果 | 历史 experiment artifacts | 独立 `runs/protocol_v2` |

因此旧数值只能作为历史参照；它们不能与 protocol_v2 主表混合，也不能替代新协议的 smoke
或 dense grid。
