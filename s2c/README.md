# s2c

s2c 是一个开放世界意图识别系统：

```text
文本 → Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

后续新实验默认使用与 `fulltex.tex` 对齐的历史 `multidataset/v19` 协议；已有
`protocol_v2_textoir_v1` 结果只作为冻结参考。协议定义和历史数据入口见
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

当前系统和研究代码统一位于本目录。不要从所有 Markdown 逐个开始读，最短阅读路径只有三步：

1. [CURRENT_STATUS.md](docs/CURRENT_STATUS.md)：当前进展、结论和下一步。
2. [统一对比与机制报告](docs/analysis/UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md)：具体结果和机制解释。
3. [关键图索引](docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md)：图、表和准确路径。

需要查协议或复现细节时，再看 [METHOD.md](docs/METHOD.md)、[EXPERIMENTS.md](docs/EXPERIMENTS.md)
和 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。分析结果与资产关系以
`configs/experiment_registry.yaml` 的 `analysis_bundles` 为准。

分析资产不再通过不断新增平行 Markdown、`V2/V3` 文件夹或一次性入口管理。结果目录、图目录、
报告、manifest、构建脚本和监督/协议层都必须登记在上述 registry；新资产使用 lower_snake_case，
历史文件名保留但通过 aliases/supersedes 归档。只读审计命令为：

```bash
python tools/maintenance/audit_asset_catalog.py
```

该命令会报告未登记的历史目录，但不会自动移动或删除它们。

## 分析资产怎么找

- `docs/analysis/`：当前只保留两份分析入口；先看
  `UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`，再看
  `VISUAL_ANALYSIS_INDEX_V1.md`。旧版性能总览已归档到
  `docs/archive/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`。
- `figures/`：当前 bundle 的图；历史或一次性图在 `figures/archive/analysis/`。
- `results/analysis/`：当前 bundle 的轻量 CSV/JSON/manifest；历史或一次性结果在
  `results/analysis/archive/analysis/`。
- `docs/archive/`：已完成、被替代或仅用于归因的报告，不作为新的事实入口。

新增分析前先登记 bundle；不要在 `docs/analysis/`、`figures/` 或
`results/analysis/` 直接创建一次性平行目录。没有闭合合同的结果保留在 archive，不能用目录存在代替最终指标。

旧协议、closeout 和方法报告只在 `docs/archive/` 中保存，不作为当前事实来源。

源码、配置、测试和活动文档在 `s2c/`；protocol_v2 的本地语料、canonical 和固定 views 在
`data/`（不提交 Git），基础模型在 `../assets/`；原始实验结果在 `../artifacts/s2c/`；可公开的
轻量 CSV/JSON 快照在 `results/`。Gate-only 指标和完整
`Gate → Router → Expert` 指标必须分开解释。
