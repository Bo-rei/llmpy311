# s2c

s2c 是一个开放世界意图识别系统：输入先经过 `Gate` 判断是否为已知意图（Known/ID）或未知意图（OOS），已知样本再由 `Router` 路由到领域，最后由 `Expert` 完成域内意图分类。

```text
文本 → Gate(OOS) → Router(领域) → Expert(意图)
```

## 先看清楚：你现在打开的是哪一层？

本工作区同时保存了源码、研究结果和历史说明。它们不是同一个东西：

| 层 | 位置/分支 | 作用 | 当前可信边界 |
| --- | --- | --- | --- |
| 当前主源码 | `main` checkout 的 `src/`、`tools/` | 可运行的基础 Pipeline、v19 主网格和 v20/v21 研究入口 | 以当前 checkout 的文件为准 |
| 后续研究收口源码 | 当前 `main` 工作树中的功能入口；历史产物仍可能记录 `autoresearch/minilm-collision-20260718` | CE-Recon、hard-negative、Cascade 修复与完整矩阵编排 | 本轮改动留在 `main` 工作树，不提交或推送 `autoresearch` |
| 实验结果 | `../artifacts/s2c/outputs/experiments/` | 已生成的 CSV、Parquet、JSON、模型和图表 | 先读 manifest；产物存在不等于当前分支有对应源码 |
| 数据与模型 | `../assets/` | 数据集、split、MiniLM、SmolLM 等大文件 | 不属于 Git 源码仓库 |
| 旧说明 | `docs/` 中未标为 active 的文件 | 历史背景、旧方案、论文草稿 | 只能解释历史，不能覆盖当前 manifest |

这也是“项目看起来改得很乱”的根因：实验产物目录不会随 Git 切分支自动回滚，因此当前 `main` 仍能看到研究分支生成的 `v22` 和 `study_closeout` 结果。

## 推荐阅读顺序

不要从所有 Markdown 文件开始。按下面顺序即可建立完整认识：

1. 本文件：确认仓库边界、分支和三层关系。
2. [文档总入口](docs/README.md)：确认当前应该读哪些文档，哪些只作历史参考。
3. [项目阅读指南](docs/项目阅读指南.md)：按源码→数据→结果走通一次完整流程。
4. [项目契约](docs/PROJECT.md)：确认路径、证据优先级和可复现边界。
5. [项目完整说明](docs/00-项目完整说明.md)：理解 Gate、Router、Expert、数据和结果解释。
6. [当前状态与里程碑](docs/07-当前状态与里程碑.md)：查看已经完成、尚未完成和下一步。
7. [实验目录与读法](docs/EXPERIMENTS.md)：读取 Gate-only 和完整 Cascade 结果。

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

`study-closeout` 等结果目录可能包含历史来源，但当前 Cascade 修复和完整矩阵入口已在 `main`
工作树中。不要切换到 `autoresearch` 或把它作为交付分支；直接按照 artifact 的 manifest 读取已有结果即可。
如果必须复核历史来源，只读取 artifact manifest 中的 `source_commit`、`git_branch` 和 hash；
不要为了查看结果切换分支。

本轮下游修复后的代表性完整 Cascade 已经回到 `main`，入口为：

```bash
python tools/eval/run_cascade_repair.py
```

它只检查/运行 KIR50、seed42 的 12 个代表单元，结果写入 `../artifacts/s2c/outputs/experiments/cascade_repair/gpu_kir50_seed42/`，不会覆盖历史 smoke。

完整三 seed Cascade 已完成，入口为：

```bash
python tools/train/run_cascade_components.py --execute
python tools/eval/prepare_cascade_gates.py --execute
python tools/eval/run_cascade_matrix.py --execute
```

结果写入 `../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/`，当前
`evaluations/matrix_manifest.json` 为 `status=complete`、`completed_unit_count=36`。

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
- 研究收口已经有三数据集、KIR50、三 seed、四 Gate 的 36 个完整 Cascade 单元；它们使用修复后的 Banking77/StackOverflow Expert。KIR25/75 仍未纳入这张四 Gate 表。
- 本轮 GPU 重训后，Banking77 Expert 测试准确率为 0.8490，StackOverflow Expert 为 0.8744；旧的约 0.122/0.246 只属于历史低预算 smoke，不应继续用于论文主表。
- 新 Cascade 汇总和样本级错误分解位于 `../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/`，先读 `cascade_summary.csv`、`cascade_error_decomposition.csv` 和 `evaluations/matrix_manifest.json`；seed42 修复目录仍保留用于单元级训练 provenance。
- `component_plan.json` 和各单元 `run.log` 记录 Router/Expert checkpoint 路径与 SHA256；`study_closeout/closeout_manifest.json` 登记本轮源码、测试和结果 hash。
- 外部 hard-negative 已有 CLINC150 和 BANKING77-OOS 结果；StackOverflow 没有兼容公开文件。
- MOGB 目前只有协议审计，没有纳入主结果的公平复现数字。

完整状态、数字和缺口请看 [docs/00-项目完整说明.md](docs/00-项目完整说明.md)；不要依据本 README 的短摘要替代 manifest。

## 维护原则

- 先修改 active 文档，再把历史材料放入 archive；不再创建新的轮次 Markdown 入口。
- 结果必须能追溯到 manifest、数据 hash 和源码 commit。
- 新实验不得覆盖历史 `paper_results`、detector 或 checkpoint。
- 若当前分支无法复现实验命令，文档必须明确写出“结果可查看、源码需切换分支”，不能把结果目录的存在当作可执行证明。
