# s2c

s2c 是一个开放世界意图识别系统：输入先经过 `Gate` 判断是否为已知意图（Known/ID）或未知意图（OOS），已知样本再由 `Router` 路由到领域，最后由 `Expert` 完成域内意图分类。

```text
文本 → Gate(OOS) → Router(领域) → Expert(意图)
```

## 先看清楚：你现在打开的是哪一层？

本工作区同时保存了源码、研究结果和历史说明。它们不是同一个东西：

| 层 | 位置/分支 | 作用 | 当前可信边界 |
| --- | --- | --- | --- |
| 当前主源码 | `main` 分支的 `s2c/src`、`s2c/tools` | 可运行的基础 Pipeline、v19 主网格和 v20/v21 研究入口 | 以当前 checkout 的文件为准 |
| 最新研究收口源码 | `autoresearch/minilm-collision-20260718` 分支 | CE-Recon、边界形状、hard-negative、Cascade 收口等后续工具 | 只有切换到该分支后才能复现对应命令 |
| 实验结果 | `../artifacts/s2c/outputs/experiments/` | 已生成的 CSV、Parquet、JSON、模型和图表 | 先读 manifest；产物存在不等于当前分支有对应源码 |
| 数据与模型 | `../assets/` | 数据集、split、MiniLM、SmolLM 等大文件 | 不属于 Git 源码仓库 |
| 旧说明 | `docs/` 中未标为 active 的文件 | 历史背景、旧方案、论文草稿 | 只能解释历史，不能覆盖当前 manifest |

这也是“项目看起来改得很乱”的根因：实验产物目录不会随 Git 切分支自动回滚，因此当前 `main` 仍能看到研究分支生成的 `v22` 和 `study_closeout` 结果。

## 推荐阅读顺序

不要从所有 Markdown 文件开始。按下面顺序即可建立完整认识：

1. 本文件：确认仓库边界、分支和三层关系。
2. [文档总入口](docs/README.md)：确认当前应该读哪些文档，哪些只作历史参考。
3. [项目完整说明](docs/00-项目完整说明.md)：理解 Gate、Router、Expert、数据和结果解释。
4. [数据协议与版本](docs/03-数据协议与版本.md)：确认 split、KIR、seed 和数据路径。
5. [训练与评估流程](docs/04-训练与评估流程.md)：需要运行主 Pipeline 时再读。
6. [当前状态与里程碑](docs/07-当前状态与里程碑.md)：查看已经完成、尚未完成和下一步。
7. [工具入口索引](docs/10-工具入口索引.md)：只在要执行命令或追溯脚本时查。

其他编号文档保留为背景或历史材料，不必一次读完。

## 当前研究主线

论文修订没有重写 `Gate → Router → Expert`。当前科学问题是：

1. 冻结 `all-MiniLM-L6-v2` 是否形成可用于意图建模的语义空间；
2. 一个 Known intent 是否由多个局部语义簇组成；
3. 多簇边界是否改善 Known/OOS，尤其是 near-OOS 的可分性；
4. near-OOS 错误来自 MiniLM 表示碰撞，还是局部边界过覆盖。

Known intent classification 不是本轮独立主贡献；它主要用于分析 false reject、hard intent 和多簇收益是否能传到完整系统。

## 当前分支能运行什么

在 `s2c/` 目录执行。当前 `main` 的基础入口是：

```bash
python -m src.cli --help
python -m src.cli prepare --dataset clinc150 --kir 50 --seed 42 --dry-run
pytest tests/unit -q
```

`main` 还包含 v19 主网格、受控 baseline 和 v20/v21 研究入口：

```bash
python -m tools.experiments.cluster_separability --help
python -m tools.experiments.cluster_separability grid --help
python -m tools.experiments.cluster_separability v20-analysis --help
python -m tools.experiments.cluster_separability v21-semantic-probe --help
```

`v22`、`study-closeout`、代表性 Cascade 的最新实现位于研究分支，不要在 `main` 上假设这些命令存在。切换前先确认没有未提交修改：

```bash
git status --short
git switch autoresearch/minilm-collision-20260718
```

只想回到稳定基础源码时：

```bash
git switch main
```

## 数据、模型和产物在哪里

代码通过 `src.runtime.WorkspacePaths` 查找工作区，不使用写死的机器路径：

```text
../assets/datasets/s2c/source/                         原始数据
../assets/datasets/s2c/prepared/data/multidataset/v19/  当前准备好的 split
../assets/models/all-MiniLM-L6-v2/                     MiniLM
../assets/models/smollm135m/                           Router/Expert 主干
../artifacts/s2c/                                      checkpoint 与实验产物
```

不要在 `s2c/` 内新建 `data/`、`outputs/` 或复制模型；这会绕过运行时路径契约。

## 结果怎么读

研究结果的权威入口是：

```text
../artifacts/s2c/outputs/experiments/study_closeout/
├── closeout_manifest.json   # 快照使用的 commit、分支、总体状态
├── result_inventory.csv     # 每项结果的路径、hash、来源 commit
├── paper_source_map.csv     # 论文表/图对应的结果文件和列
└── known_gaps.json          # 明确列出还没有证明的部分
```

阅读任何结果时都按这个顺序：

1. 先看 `closeout_manifest.json`，确认它是在什么分支和 commit 生成的；
2. 再看 `result_inventory.csv`，确认文件是否存在、hash 是否登记；
3. 再打开汇总 CSV，而不是直接从单个预测文件猜结论；
4. 最后回到对应 manifest，确认数据 split、阈值、K 和模型来源。

特别注意：`Gate-only` 的 OOS F1 不能当作完整意图分类的 macro-F1；完整 Cascade 必须单独看 `Known macro-F1`、`overall accuracy` 和 Gate/Router/Expert 错误分解。

## 当前结论的简短版本

- v19 的 KIR × K × distance 主体实验和 v20/v21 的多簇、近 OOS、MiniLM 语义诊断已经形成较完整的 Gate-only 证据链。
- CE/SupCon 通常改善 `K=1` 的 OOS 或 near-OOS，但不保证 `K=2` 在所有数据集上有效。
- StackOverflow 的 `K>1` 退化是一个真实负结果，不能写成“多簇普遍有效”。
- near-OOS 的主要问题是 Known 语义支持区域内的表示碰撞；这解释了为什么单纯继续增大 K、半径或阈值的收益有限。
- 研究收口已经有三数据集、KIR50、seed42 的 12 个代表性 Cascade 单元，但它们是 smoke/代表性验证，不是三 seed、KIR25/75 或完整训练预算的稳定性证明。
- 外部 hard-negative 已有 CLINC150 和 BANKING77-OOS 结果；StackOverflow 没有兼容公开文件。
- MOGB 目前只有协议审计，没有纳入主结果的公平复现数字。

完整状态、数字和缺口请看 [docs/00-项目完整说明.md](docs/00-项目完整说明.md)；不要依据本 README 的短摘要替代 manifest。

## 维护原则

- 先修改 active 文档，再把历史材料放入 archive；不再创建新的轮次 Markdown 入口。
- 结果必须能追溯到 manifest、数据 hash 和源码 commit。
- 新实验不得覆盖历史 `paper_results`、detector 或 checkpoint。
- 若当前分支无法复现实验命令，文档必须明确写出“结果可查看、源码需切换分支”，不能把结果目录的存在当作可执行证明。
