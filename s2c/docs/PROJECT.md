# s2c 项目契约

## 项目边界

s2c 是开放世界意图识别系统，运行链路固定为：

```text
文本 → MiniLM Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前活动代码、配置、测试和文档都在 `s2c/`。本轮不新增方法、不重跑训练，
只维护可复现入口和轻量公开结果。

## 工作区职责

| 目录 | 职责 | 是否提交父仓库 |
| --- | --- | --- |
| `s2c/` | 活动源码、配置、测试、文档、轻量公开结果 | 是 |
| `../assets/` | 数据集和基础模型 | 否，保持忽略 |
| `../artifacts/` | 原始实验输出、checkpoint、embedding、逐样本结果 | 否，保持忽略 |
| `../archives/` | 本地历史材料和日志 | 否，保持忽略 |
| `../textoir/` | 独立上游仓库 | 否，保持独立 Git |

`s2c/results/` 只存由 `configs/public_results.yaml` 白名单导出的轻量 CSV/JSON，
不替代 `../artifacts/`，也不包含模型、embedding、checkpoint 或逐样本输出。

## 源码边界

- Router 的实现位于 `src/router/`，当前入口主要是 `src/router/router_model.py`。
- Expert 的实现位于 `src/models/expert.py`，级联推理由 `src/pipeline/` 组织。
- Gate 实现位于 `src/gate/` 和 `src/gate_minimal/`；后者保持严格基线语义。
- 训练/评价/导出入口位于 `tools/`；公开快照维护入口位于
  `tools/maintenance/export_public_results.py`。
- `configs/experiment_registry.yaml` 登记原始实验的入口、manifest 和汇总文件；
  `configs/public_results.yaml` 登记允许进入 GitHub 的精确文件。

## 当前活动入口

1. [README.md](../README.md)：项目入口。
2. [DATASETS.md](DATASETS.md)：数据来源裁决与准入状态。
3. [EXPERIMENTS.md](EXPERIMENTS.md)：结果层次和公开快照。
4. [RUNBOOK.md](RUNBOOK.md)：审计、测试和导出命令。

历史说明只在 `docs/archive/`，不能覆盖当前事实。读取完整实验数字时，先看
`../artifacts/s2c/outputs/experiments/` 下的 manifest，再看汇总 CSV；读取 GitHub
数字时只看 `results/MANIFEST.csv` 及其对应文件。
