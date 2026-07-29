# s2c 项目知识库

## 当前边界

这是 `Gate → Router → Expert` 开放世界意图识别项目。当前活动源码、配置、测试、
文档和轻量公开结果均在 `s2c/`；本地原始产物在 `../artifacts/s2c/`，数据和基础模型
在 `../assets/`，独立 TextOIR 仓库在 `../textoir/`。

当前文档入口只有：

- `README.md`
- `docs/PROJECT.md`
- `docs/CODE_LAYOUT.md`
- `docs/EXPERIMENTS.md`
- `docs/RUNBOOK.md`
- `docs/DEVELOPMENT_LOG.md`
- `docs/research/RESEARCH_STATUS.md`（唯一研究状态入口）
- `docs/research/EXPERIMENT_LEDGER.csv`（追加式实验总账）
- `docs/research/DECISION_LOG.md`
- `docs/research/PAPER_CLAIM_AUDIT.md`

`docs/archive/` 仅保存历史资料，不是当前事实来源。不要重新建立平行文档索引或版本号
文档树。

## 源码位置

```text
src/protocol_v2/      当前活动协议（data/evaluation/experiments/gate/runtime/tracking）
src/legacy/           v19/历史兼容实现，不作为当前协议默认依赖
src/legacy/gate/      历史 Gate；活动多球 detector 位于 protocol_v2/gate
src/legacy/router/    历史 Router
src/legacy/models/    历史 Expert、SVDD 和 Transformer 封装
src/legacy/pipeline/  历史完整级联推理
src/legacy/runtime/   历史 WorkspacePaths 和 artifact 约定
tools/                训练、评价、分析、兼容和维护入口
configs/              运行配置、实验登记和公开导出白名单
tests/                单元、协议和回归测试
results/              GitHub 可提交的轻量 CSV/JSON 快照
```

完整放置规则见 `docs/CODE_LAYOUT.md`。禁止重新创建 `src/s2c/`，也禁止从 `src`
根目录导入；活动包使用 `protocol_v2.*`，历史包使用 `legacy.*`。

`tools/maintenance/export_public_results.py` 只按
`configs/public_results.yaml` 导出公开文件；不得复制 `../artifacts` 整个目录。

## 维护规则

- 所有 Codex 或其他智能体的实质性修改都必须同步追加
  `docs/DEVELOPMENT_LOG.md`；纯只读且不产生文件的任务可以例外。日志必须记录
  base commit、修改文件、数据影响、artifact 影响、测试、风险和下一步。
- 不运行训练来完成工作区整理，不修改或重命名 `../artifacts` 原始实验目录。
- 不把 Gate-only 的 Frozen/CE/SupCon 结果写成完整 Pipeline 结果。
- 不提交模型、checkpoint、embedding、Parquet、逐样本 scores 或运行日志。
- 新实验入口使用功能命名；历史 `_v19/_v20/_v21` 文件保留为兼容入口，不再扩展同类版本号。
- 涉及实验、指标、数据协议或论文论断的任务，开始前必须读取
  `docs/research/RESEARCH_STATUS.md`、`EXPERIMENT_LEDGER.csv` 和 `DECISION_LOG.md`，结束前运行
  `python tools/maintenance/check_research_state.py` 并追加状态台账、开发日志和阶段 closeout。
- 新计划若与 ledger 中 `do_not_repeat` 且已完成的 protocol/dataset/KIR/seed/representation/K/distance/
  partition/boundary 完全相同，必须拒绝为 `duplicate_completed_experiment`；只有带明确 rerun reason
  的显式覆盖才允许继续。
- 当前协议只从 `protocol_v2.*` 导入；历史 Router 只从 `legacy.router` 导入，禁止在
  `legacy.models` 重新导出 Router。
- 训练循环不要添加会破坏 LoRA 梯度的 `torch.no_grad()`。
- 不在初始化阶段调用 `torch.cuda.is_available()`；部分环境会触发原生运行时问题。
- `configs/data/protocol_v2_admission.json` 是唯一数据准入开关。只有同时满足 dataset_version、
  dataset-level admission 和 materialized view/export 的任务才可运行；不得绕过 Gate runner 或 E4
  adapter 向任何 `../artifacts/s2c/runs/<dataset_version>/` 写入。唯一活动版本是
  `protocol_v2_textoir_v1`；StackOverflow 为 `admitted_benchmark_local_only`，允许本地实验但
  禁止完整语料进入 Git、论文附件或任何 s2c 再分发包。`protocol_v2_official_v1` 冻结审计，
  legacy `protocol_v2` 仍被拒绝。

## 最小验证

```bash
pytest tests/unit -q
python -m py_compile tools/maintenance/export_public_results.py
python tools/analysis/audit_experiment_registry.py
python tools/maintenance/export_public_results.py --verify
```

任何公开结果数字都必须能通过 `results/MANIFEST.csv` 或对应 artifact manifest 追溯。
