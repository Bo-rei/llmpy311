# 实验状态

## 活动协议：protocol_v2_textoir_v1

活动数据是固定 TEXTOIR commit `dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 的 CLINC150、
Banking77 与 StackOverflow snapshot。StackOverflow 是 local-only benchmark，不阻止本地实验，
但不允许完整语料进入公开 Git 或 s2c 再分发。

| 阶段 | 目的 | 当前状态 | 证据位置 |
| --- | --- | --- | --- |
| E0 | source/canonical/registry/views/exports 与 runtime independence | complete | `docs/audits/protocol_v2_implementation/` |
| E1 | 3 datasets × 3 KIR × K{1,2} × 2 distances | complete: 36/36 | `../artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e1_gate_smoke.csv` |
| E2 | 11 KIR × 5 seeds × K{1..5} × 2 distances | in progress: 1,650 planned | `../artifacts/s2c/runs/protocol_v2_textoir_v1/plans/` |
| E3--E7 | mechanisms, boundaries, baselines, representation, Pipeline | not started | wait for E2 summary |

E1 是 Gate-only smoke，不应与历史完整 Cascade 或 v19--v22 结果混合解释。每个 run ID、manifest、
cache 与 summary 均包含 `protocol_v2_textoir_v1`，防止误 resume 历史 protocol。

## 历史版本

- `protocol_v2_official_v1`：冻结 provenance audit；不再作为新实验默认。
- `protocol_v2`：拒绝的 legacy candidate；仅可显式读取用于历史审计。
- v19--v22、旧 Cascade：保留原始 artifacts，不覆盖也不重新命名。

后续主表只使用带有该 protocol version、dataset manifest SHA、registry SHA、encoder revision 和 run
manifest 的结果。详细命令见 `RUNBOOK.md`，结果字段约束见 `RESULTS_CONTRACT.md`。
