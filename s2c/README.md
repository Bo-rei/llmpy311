# s2c

s2c 是一个开放世界意图识别系统：

```text
文本 → Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前系统和研究代码统一位于本目录。当前事实入口只有以下文档，分析结果与资产关系以
`configs/experiment_registry.yaml` 的 `analysis_bundles` 为准：

- [METHOD.md](docs/METHOD.md)：当前 Gate、公式和 split–merge 原型边界。
- [CURRENT_STATUS.md](docs/CURRENT_STATUS.md)：唯一研究状态、阻断和下一步。
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)：主表、消融、基线和证据边界。
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)：数据、hash、命令和 artifact 规则。

分析资产不再通过不断新增平行 Markdown、`V2/V3` 文件夹或一次性入口管理。结果目录、图目录、
报告、manifest、构建脚本和监督/协议层都必须登记在上述 registry；新资产使用 lower_snake_case，
历史文件名保留但通过 aliases/supersedes 归档。只读审计命令为：

```bash
python tools/maintenance/audit_asset_catalog.py
```

该命令会报告未登记的历史目录，但不会自动移动或删除它们。

旧协议、closeout 和方法报告只在 `docs/archive/` 中保存，不作为当前事实来源。

源码、配置、测试和活动文档在 `s2c/`；protocol_v2 的本地语料、canonical 和固定 views 在
`data/`（不提交 Git），基础模型在 `../assets/`；原始实验结果在 `../artifacts/s2c/`；可公开的
轻量 CSV/JSON 快照在 `results/`。Gate-only 指标和完整
`Gate → Router → Expert` 指标必须分开解释。
