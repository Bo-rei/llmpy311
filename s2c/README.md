# s2c

s2c 是一个开放世界意图识别系统：

```text
文本 → Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前系统和研究代码统一位于本目录。工作区职责、证据边界、实验结果和运行命令分别见：

- [PROJECT.md](docs/PROJECT.md)：项目布局、模块边界和证据来源。
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)：Gate-only、完整 Cascade 和公开结果索引。
- [RUNBOOK.md](docs/RUNBOOK.md)：环境检查、审计、测试和结果导出命令。
- [DATASETS.md](docs/DATASETS.md)：三方来源裁决、许可证与 dataset-level 准入状态。
- [DATA_PROTOCOL_V2.md](docs/DATA_PROTOCOL_V2.md)：独立数据、KIR registry、views 与 exporter 契约。
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)：重建官方 raw 版本与候选数据的边界。

源码、配置、测试和活动文档在 `s2c/`；protocol_v2 的本地语料、canonical 和固定 views 在
`data/`（不提交 Git），基础模型在 `../assets/`；原始实验结果在 `../artifacts/s2c/`；可公开的
轻量 CSV/JSON 快照在 `results/`。Gate-only 指标和完整
`Gate → Router → Expert` 指标必须分开解释。
