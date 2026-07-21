# s2c 文档总入口

如果你第一次接触这个项目，不要把 `docs/` 下的 Markdown 全部当成当前说明。
项目经历过多轮实验，很多编号文件是历史方案或论文底稿。日常只需要读下面六个文件；
其余文档按需查阅。

## 先读这六个文件

| 顺序 | 文件 | 你会得到什么 |
| --- | --- | --- |
| 1 | [`README.md`](../../../README.md) | 项目是什么、代码/数据/结果分别在哪里 |
| 2 | [`项目阅读指南.md`](项目阅读指南.md) | 一条样本如何经过 Gate、Router、Expert，以及如何打开一个结果单元 |
| 3 | [`PROJECT.md`](../../PROJECT.md) | 当前 `main` 工作树的路径、协议、证据优先级和边界 |
| 4 | [`07-当前状态与里程碑.md`](07-当前状态与里程碑.md) | 已完成什么、结论能外推到哪里、仍缺什么 |
| 5 | [`EXPERIMENTS.md`](../../EXPERIMENTS.md) | Gate-only 与完整 Cascade 的结果目录、指标和读取顺序 |
| 6 | [`10-工具入口索引.md`](10-工具入口索引.md) | 要运行命令时从哪个入口开始 |

只想快速建立概念时，读完前 1–4 项即可；只有要查数字或运行实验时才打开第 5–6 项。

## 当前项目的一句话

```text
文本 → Gate（Known / OOS）→ Router（domain）→ Expert（intent）
```

当前研究重点是：冻结的 MiniLM 语义空间是否形成 Known intent 的局部多簇，以及这些局部支持区域能否拒绝 OOS，尤其是 near-OOS。`Known intent` 分类结果主要用于诊断 false reject 和下游级联影响，不是独立的主贡献。

## 当前事实（2026-07-20）

- Gate-only 的多簇、near-OOS、MiniLM 表示和随机分簇研究已有完整结果。
- KIR50 的完整 Cascade 已在 GPU 上完成：3 个数据集 × 3 个 seed × 4 个 Gate = 36/36。
- Banking77-OOS 和 StackOverflow 使用修复后的下游组件；单域数据集使用 constant Router，不训练无意义的一类 Router。
- MOGB 目前只有协议审计，没有可放入主表的公平复现数字。
- KIR25/KIR75 的四 Gate Cascade 尚未完成，因为对应 CE-Recon 表示 checkpoint 尚未准备。
- 外部 hard-negative 只覆盖 CLINC150 和 BANKING77-OOS；StackOverflow 没有兼容的公开文件。
- 本轮代码和文档修改留在当前 `main` 工作树，不提交或推送 `autoresearch`。

## 结果只按这一条路径读取

产物在仓库外的忽略目录：

```text
../artifacts/s2c/outputs/experiments/
```

论文数字和复现判断必须遵循：

```text
study_closeout/closeout_manifest.json
    → result_inventory.csv
    → 对应 experiment manifest
    → 汇总 CSV
    → predictions.json / scores.parquet（仅案例复核）
```

完整 Cascade 的当前入口是：

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/
├── component_plan.json
├── gates/gate_manifest.json
├── evaluations/matrix_manifest.json
├── cascade_summary.csv
└── cascade_error_decomposition.csv
```

先确认 `matrix_manifest.json` 的 `status=complete` 和 `completed_unit_count=36`，再读汇总表。不要从单个 `predictions.json` 推断整体结论，也不要把 Gate-only 的 OOS F1 当成完整 Cascade 的 Known macro-F1。

## 按目的选择文档

### 我想先看懂代码

读：`项目阅读指南.md` → `PROJECT.md` → `03-数据协议与版本.md` → `04-训练与评估流程.md`。

### 我想看当前研究结论

读：`07-当前状态与里程碑.md` → `EXPERIMENTS.md`，然后打开 `study_closeout/` 和对应汇总 CSV。

### 我想运行基础流程

读：`PROJECT.md` → `10-工具入口索引.md`。第一次只运行 `--help` 或 `--dry-run`。

### 我想追溯旧设计

再读下面的历史参考，不要用它们决定今天的运行命令：

```text
00-项目完整说明.md
01-项目总览.md
01-项目通俗详细解析.md
02-代码与目录结构.md
05-实验结果与性能基线.md
08-消融实验方案.md
09-消融方案的代码可行性分析.md
12-s2c意图识别方案技术底稿.md
13-本次改动记录.md
14-项目结构重整与职责边界.md
技术路线与代码映射.md
```

这些文件顶部已经标明了历史属性；发生冲突时以 manifest、当前源码和本入口为准。

## 不要做的事

- 不要从文件名中的 `v19/v20/v21/v22` 判断当前代码是否可运行；以当前 checkout 和 manifest 为准。
- 不要在 `s2c/` 内复制数据、模型或 `outputs/`；使用 `WorkspacePaths` 指向 `../assets` 和 `../artifacts`。
- 不要用 test 集选择 K、阈值或 checkpoint。
- 不要把外部 hard-negative 用于训练或调参。
- 不要为了“减少文件数量”删除历史文档；先通过本入口隔离用途。
- 不要切换或提交 `autoresearch`；当前交付入口是 `main` 工作树。

## 最短核验命令

在 `s2c/` 目录执行：

```bash
git branch --show-current
python -m src.cli --help
python -m tools.experiments.cluster_separability --help
python tools/eval/run_cascade_matrix.py --execute
pytest tests/unit -q
```

最后一条 Cascade 命令会复用已有完整单元；它不是重新训练。若结果目录缺失，先按 `PROJECT.md` 中的组件准备和 Gate 适配顺序执行。
