# s2c

s2c 是一个开放世界意图识别系统：

```text
文本 → Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前系统和研究代码统一位于本目录。工作区职责、证据边界、实验结果和运行命令分别见：

- [PROJECT.md](docs/PROJECT.md)：项目布局、模块边界和证据来源。
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)：Gate-only、完整 Cascade 和公开结果索引。
- [RUNBOOK.md](docs/RUNBOOK.md)：环境检查、审计、测试和结果导出命令。

源码、配置、测试和活动文档在 `s2c/`；本地数据和基础模型在 `../assets/`；原始实验结果在
`../artifacts/s2c/`；可公开的轻量 CSV/JSON 快照在 `results/`。Gate-only 指标和完整
`Gate → Router → Expert` 指标必须分开解释。
